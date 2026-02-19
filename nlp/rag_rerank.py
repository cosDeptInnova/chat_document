# rag_rerank.py
import os
import hashlib
import logging
from typing import List, Tuple, Optional
from collections import OrderedDict
import datetime
import numpy as np
from sentence_transformers import CrossEncoder
import torch
import threading
from torch.nn.functional import normalize

from device_manager import gpu_manager
from rag_utils import ensure_local_with_transformers, _safe_dir_name
from rag_metrics import MMR_TIME

logger = logging.getLogger("cosmos_nlp_v3.rerank")

_GLOBAL_RERANKER: Optional[CrossEncoder] = None
_RERANKER_DEVICE: Optional[torch.device] = None
_RERANKER_LOCK = threading.Lock()

_GLOBAL_COLBERT = None
_COLBERT_DEVICE: Optional[torch.device] = None
_COLBERT_LOCK = threading.Lock()


def get_reranker_impl(indexer) -> CrossEncoder:
    global _GLOBAL_RERANKER, _RERANKER_DEVICE

    if getattr(indexer, "_reranker", None) is not None:
        return indexer._reranker

    if _GLOBAL_RERANKER is None:
        with _RERANKER_LOCK:
            if _GLOBAL_RERANKER is None:
                rerank_dir = indexer._rerank_dir
                dev = gpu_manager.best_device(
                    min_free_mb=int(os.getenv("RERANK_MIN_FREE_MB", "2048"))
                )
                device_str = str(dev) if dev.type == "cuda" else "cpu"
                logger.info(f"(rerank) inicializando CrossEncoder en {device_str}")

                m = CrossEncoder(rerank_dir, device=device_str)

                # Importante: modo eval + sin gradientes (menos VRAM y más determinista)
                try:
                    if getattr(m, "model", None) is not None:
                        m.model.eval()
                        for p in m.model.parameters():
                            p.requires_grad_(False)
                except Exception:
                    pass

                _GLOBAL_RERANKER = m
                _RERANKER_DEVICE = dev

    indexer._reranker = _GLOBAL_RERANKER
    return indexer._reranker

def get_colbert_encoder_impl(indexer):
    from transformers import AutoTokenizer, AutoModel
    global _GLOBAL_COLBERT, _COLBERT_DEVICE

    if _GLOBAL_COLBERT is None:
        with _COLBERT_LOCK:
            if _GLOBAL_COLBERT is None:
                model_id = os.getenv("COLBERT_MODEL_ID", "bert-base-multilingual-cased")
                cache_root = os.path.expanduser(os.getenv("COSMOS_CACHE", "~/.cache/cosmos"))
                local = ensure_local_with_transformers(
                    model_id,
                    os.path.join(cache_root, _safe_dir_name(model_id)),
                    task="auto",
                )
                tok = AutoTokenizer.from_pretrained(local)
                mdl = AutoModel.from_pretrained(local)

                # eval + sin gradientes
                try:
                    mdl.eval()
                    for p in mdl.parameters():
                        p.requires_grad_(False)
                except Exception:
                    pass

                dev = gpu_manager.best_device(
                    min_free_mb=int(os.getenv("COLBERT_MIN_FREE_MB", "1024"))
                )
                if dev.type == "cuda":
                    logger.info(f"(colbert) moviendo modelo a {dev}")
                    mdl.to(dev)
                else:
                    logger.info("(colbert) usando CPU")

                _GLOBAL_COLBERT = (tok, mdl)
                _COLBERT_DEVICE = dev

    tok, mdl = _GLOBAL_COLBERT
    indexer._colbert_tok_mdl = (tok, mdl)
    return tok, mdl



def late_interaction_scores_impl(indexer, query: str, texts: List[str]) -> List[float]:
    global _COLBERT_DEVICE

    tok, mdl = getattr(indexer, "_colbert_tok_mdl", (None, None))
    if tok is None or mdl is None:
        tok, mdl = get_colbert_encoder_impl(indexer)

    scores: List[float] = []

    with gpu_manager.use_device_with_fallback(
        mdl,
        prefer_device=_COLBERT_DEVICE,
        min_free_mb=int(os.getenv("COLBERT_MIN_FREE_MB", "1024")),
        fallback_to_cpu=True,
        pool="colbert",
    ) as dev:
        if dev != _COLBERT_DEVICE:
            _COLBERT_DEVICE = dev

        def _enc(s: str) -> torch.Tensor:
            enc = tok(
                s,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            if dev.type == "cuda":
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

            # inference_mode reduce overhead y uso de memoria vs no_grad
            with torch.inference_mode():
                out = mdl(**enc).last_hidden_state.squeeze(0)

            # quitamos [CLS]/[SEP] si están
            if out.shape[0] > 2:
                out = out[1:-1]

            out = normalize(out, p=2, dim=1)
            return out

        Q = _enc(indexer._prep_query(query))
        for t in texts:
            D = _enc(indexer._prep_doc(t))
            sims = torch.matmul(Q, D.T)
            m, _ = sims.max(dim=1)
            scores.append(float(m.sum().item()))

            # liberar refs rápido (especialmente en CUDA bajo concurrencia)
            del D, sims, m

        del Q

    return scores


def rerank_results_impl(
    indexer,
    query: str,
    retrieved_chunks: List[Tuple[str, str, float]],
    model: Optional[CrossEncoder] = None,
    top_k: int = 5,
    mmr_lambda: float = 0.7
) -> List[Tuple[str, str, float, float]]:
    if not retrieved_chunks:
        return []

    global _RERANKER_DEVICE

    CE_CACHE_MAX = int(os.getenv("CE_CACHE_MAX", "5000"))
    LI_CACHE_MAX = int(os.getenv("LI_CACHE_MAX", "2000"))

    if not hasattr(indexer, "_ce_cache"):
        indexer._ce_cache = OrderedDict()
    if not hasattr(indexer, "_li_cache"):
        indexer._li_cache = OrderedDict()

    #  Locks por cache (OrderedDict NO es thread-safe)
    if not hasattr(indexer, "_ce_cache_lock"):
        indexer._ce_cache_lock = threading.Lock()
    if not hasattr(indexer, "_li_cache_lock"):
        indexer._li_cache_lock = threading.Lock()

    def _lru_get(cache, key, lock):
        with lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            return None

    def _lru_set(cache, key, value, max_size, lock):
        with lock:
            cache[key] = value
            cache.move_to_end(key)
            if len(cache) > max_size:
                cache.popitem(last=False)

    model = model or get_reranker_impl(indexer)

    texts = [t for (t, _, _) in retrieved_chunks]
    pairs = [(query, t) for (t, _, _) in retrieved_chunks]

    # -------------------------
    # 1) CrossEncoder + LRU (thread-safe)
    # -------------------------
    ce_scores: List[Optional[float]] = []
    to_infer: List[Tuple[str, str]] = []
    to_infer_idx: List[int] = []
    pair_keys: List[str] = []

    for (q, d) in pairs:
        pair_keys.append(hashlib.sha1((q + "||" + d).encode("utf-8")).hexdigest())

    for i, key in enumerate(pair_keys):
        cached = _lru_get(indexer._ce_cache, key, indexer._ce_cache_lock)
        if cached is None:
            to_infer.append(pairs[i])
            to_infer_idx.append(i)
            ce_scores.append(None)
        else:
            ce_scores.append(float(cached))

    if to_infer:
        ce_module = getattr(model, "model", None)

        if ce_module is not None:
            with gpu_manager.use_device_with_fallback(
                ce_module,
                prefer_device=_RERANKER_DEVICE or getattr(model, "target_device", None),
                min_free_mb=int(os.getenv("RERANK_MIN_FREE_MB", "2048")),
                fallback_to_cpu=True,
                pool="rerank",
            ) as dev:
                if hasattr(model, "target_device"):
                    model.target_device = dev
                _RERANKER_DEVICE = dev

                with torch.inference_mode():
                    inferred = model.predict(to_infer)
        else:
            with torch.inference_mode():
                inferred = model.predict(to_infer)

        for j, sc in enumerate(inferred):
            idx = to_infer_idx[j]
            ce_scores[idx] = float(sc)
            _lru_set(indexer._ce_cache, pair_keys[idx], float(sc), CE_CACHE_MAX, indexer._ce_cache_lock)

    # defensivo: si algo quedó None, lo fijamos a 0.0
    ce_scores = [0.0 if v is None else float(v) for v in ce_scores]
    ce_scores = np.array(ce_scores, dtype=np.float32)

    # -------------------------
    # 2) Late interaction (ColBERT) opcional + cache (thread-safe)
    # -------------------------
    li_weight = float(os.getenv("LI_WEIGHT", "0.35"))
    if li_weight > 0:
        li_scores: List[Optional[float]] = []
        to_li: List[str] = []
        to_li_idx: List[int] = []
        li_keys: List[str] = []

        for t in texts:
            li_keys.append(hashlib.sha1((query + "||LI||" + t).encode("utf-8")).hexdigest())

        for i, key in enumerate(li_keys):
            cached = _lru_get(indexer._li_cache, key, indexer._li_cache_lock)
            if cached is None:
                to_li.append(texts[i])
                to_li_idx.append(i)
                li_scores.append(None)
            else:
                li_scores.append(float(cached))

        if to_li:
            computed = np.array(
                late_interaction_scores_impl(indexer, query, to_li),
                dtype=np.float32,
            )
            for j, sc in enumerate(computed):
                idx = to_li_idx[j]
                li_scores[idx] = float(sc)
                _lru_set(indexer._li_cache, li_keys[idx], float(sc), LI_CACHE_MAX, indexer._li_cache_lock)

        li_scores = [0.0 if v is None else float(v) for v in li_scores]
        li_scores = np.array(li_scores, dtype=np.float32)
    else:
        li_scores = np.zeros(len(texts), dtype=np.float32)

    # -------------------------
    # 3) Normalización y combinación
    # -------------------------
    def _norm(x):
        x = np.array(x, dtype=np.float32)
        if x.size == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-8:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    ce_n = _norm(ce_scores)
    li_n = _norm(li_scores)
    final_scores = (1.0 - li_weight) * ce_n + li_weight * li_n

    order = np.argsort(-final_scores)
    retrieved_sorted = [retrieved_chunks[i] for i in order]
    final_sorted = final_scores[order]

    # -------------------------
    # 4) Embeddings densos para MMR ( sin device= para evitar .to() interno)
    # -------------------------
    docs_prep = [indexer._prep_doc(t) for (t, _, _) in retrieved_sorted]
    query_prep = [indexer._prep_query(query)]

    with gpu_manager.use_device_with_fallback(
        indexer.sbert_model,
        prefer_device=getattr(indexer, "embed_device", None),
        min_free_mb=int(os.getenv("HF_EMBED_MIN_FREE_MB", "2048")),
        fallback_to_cpu=True,
        pool="embed",
    ) as dev:
        # OJO: NO pasar device=... aquí (SentenceTransformer.encode hace .to() internamente)
        doc_embs = indexer.sbert_model.encode(
            docs_prep,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        query_emb = indexer.sbert_model.encode(
            query_prep,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        if dev != getattr(indexer, "embed_device", None):
            indexer.embed_device = dev

    # -------------------------
    # 5) MMR
    # -------------------------
    def _mmr_indices(doc_embs, query_emb, util, lambd, k):
        sim_q = (doc_embs @ query_emb) / (
            np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        selected = [int(np.argmax(util))]
        candidates = set(range(len(util))) - set(selected)
        for _ in range(min(k, len(util)) - 1):
            mmr_scores = {}
            for idx in candidates:
                red = max(
                    (doc_embs[idx] @ doc_embs[j]) /
                    (np.linalg.norm(doc_embs[idx]) * np.linalg.norm(doc_embs[j]) + 1e-8)
                    for j in selected
                )
                mmr_scores[idx] = (
                    lambd * (0.5 * util[idx] + 0.5 * sim_q[idx]) - (1 - lambd) * red
                )
            next_idx = max(mmr_scores, key=mmr_scores.get)
            selected.append(next_idx)
            candidates.remove(next_idx)
        return selected

    with MMR_TIME.time():
        chosen = _mmr_indices(doc_embs, query_emb, final_sorted, mmr_lambda, top_k)

    # -------------------------
    # 6) salida
    # -------------------------
    out: List[Tuple[str, str, float, float]] = []
    for i in chosen:
        t, n, dense_sc = retrieved_sorted[i]
        out.append((t, n, float(dense_sc), float(final_sorted[i])))
    return out


def evaluate_results_impl(results: List[Tuple], ground_truth: List[str]) -> float:
    if not results:
        return 0.0
    hits = sum(
        1
        for text, _, _, _ in results
        for gt in ground_truth
        if gt.lower() in text.lower()
    )
    return hits


def log_query_impl(query, results, latency, logger_):
    try:
        top_rerank_score = results[0][3] if results else 0
    except Exception:
        top_rerank_score = 0
    log = {
        "query": query,
        "result_count": len(results),
        "top_rerank_score": top_rerank_score,
        "latency_ms": latency,
        "timestamp": datetime.datetime.now().isoformat()
    }
    logger_.info(f"(SearchLog) {log}")

# rag_sparse.py
import os
import time
import logging
import os
import time
import logging
from typing import List, Optional, Dict, Any, Tuple
import torch
import threading
from qdrant_client.http import models as qmodels

from rag_utils import _safe_dir_name
from device_manager import gpu_manager


# Logger de módulo, alineado con el resto del proyecto
logger = logging.getLogger("cosmos_nlp_v3")

_SPLADE_GLOBAL = None
_SPLADE_DEVICE: Optional[torch.device] = None
_SPLADE_LOCK = threading.Lock()


if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ----------------- Helpers internos (compat) -----------------
class _NoopMetric:
    def labels(self, **kwargs):
        return self
    def inc(self, *args, **kwargs):
        return None
    def set(self, *args, **kwargs):
        return None
    def observe(self, *args, **kwargs):
        return None


def _make_sparse_query_vector(indices: List[int], values: List[float]) -> Any:
    """
    Construye el vector de consulta SPLADE con máxima compatibilidad:
      1) NamedSparseVector(name, SparseVector(...))     (preferente)
      2) dict {"name": ..., "vector": SparseVector(...)} (pydantic parsea)
    """
    sparse_name = os.getenv("QDRANT_SPARSE_NAME", "sparse")
    sv = qmodels.SparseVector(indices=indices, values=values)
    try:
        return qmodels.NamedSparseVector(name=sparse_name, vector=sv)
    except Exception:
        # Algunos clientes aceptan dict y lo parsean a modelo
        return {"name": sparse_name, "vector": sv}


def _filter_has_user_must(qfilter: Optional[qmodels.Filter]) -> bool:
    """
    True si el filtro ya contiene una condición must sobre key='user'.
    Best-effort (compat con distintas versiones de modelos).
    """
    if not isinstance(qfilter, qmodels.Filter):
        return False
    must = getattr(qfilter, "must", None) or []
    for c in must:
        try:
            if getattr(c, "key", None) == "user":
                return True
        except Exception:
            continue
    return False


def _merge_filters(base_must: List[qmodels.FieldCondition], qfilter: Optional[qmodels.Filter]) -> qmodels.Filter:
    """
    Merge robusto:
    - Combina must = base_must + qfilter.must
    - Preserva should y must_not si existen
    """
    if isinstance(qfilter, qmodels.Filter):
        must = list(base_must) + list(getattr(qfilter, "must", None) or [])
        should = list(getattr(qfilter, "should", None) or []) or None
        must_not = list(getattr(qfilter, "must_not", None) or []) or None
        return qmodels.Filter(must=must, should=should, must_not=must_not)
    return qmodels.Filter(must=list(base_must))

# ---------- Impl: update_vectors compatibility ----------
def _update_sparse_vectors_compat_client_impl(qdrant_client, collection_name: str, point_id: str, indices: List[int], values: List[float]) -> None:
    """
    Actualiza SOLO el vector 'sparse' de un punto en Qdrant con compat de versiones.
    """
    sparse_name = os.getenv("QDRANT_SPARSE_NAME", "sparse")
    sv = qmodels.SparseVector(indices=indices, values=values)

    last_exc: Optional[Exception] = None

    # Variante 1: 'vectors' (plural) + dict
    try:
        pv = qmodels.PointVectors(id=point_id, vectors={sparse_name: sv})
        qdrant_client.update_vectors(
            collection_name=collection_name,
            points=[pv],
            wait=True
        )
        return
    except Exception as e:
        last_exc = e

    # Variante 2: 'vector' (singular) + dict
    try:
        pv = qmodels.PointVectors(id=point_id, vector={sparse_name: sv})
        qdrant_client.update_vectors(
            collection_name=collection_name,
            points=[pv],
            wait=True
        )
        return
    except Exception as e:
        last_exc = e

    # Variante 3: 'vector' (singular) + NamedSparseVector
    try:
        nsv = qmodels.NamedSparseVector(name=sparse_name, vector=sv)
        pv = qmodels.PointVectors(id=point_id, vector=nsv)
        qdrant_client.update_vectors(
            collection_name=collection_name,
            points=[pv],
            wait=True
        )
        return
    except Exception as e:
        last_exc = e

    logger.warning(f"(splade-deferred) update_vectors compat falló; desactivo sparse. Detalle: {last_exc}")
    os.environ["QDRANT_ENABLE_SPARSE"] = "0"
    raise


def update_sparse_vectors_compat_impl(indexer, point_id: str, indices: List[int], values: List[float]) -> None:
    """
    Wrapper compat: mantiene API existente pero delega a implementación por cliente+colección,
    lo que permite usarlo también desde un worker global sin depender del indexer.
    """
    return _update_sparse_vectors_compat_client_impl(
        qdrant_client=indexer.qdrant_client,
        collection_name=indexer.collection_name,
        point_id=point_id,
        indices=indices,
        values=values,
    )


# ---------- Impl: worker deferred queue ----------
def ensure_sparse_worker_impl(indexer):
    """
    Arranca (una sola vez) un worker GLOBAL por proceso para SPLADE diferido.
    Evita 1 hilo por usuario (N usuarios => N threads => presión de VRAM).
    """
    import queue

    def _as_bool(v, default=True):
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in ("1", "true", "yes", "on")

    # flag por indexer, pero robusto (no usar "0" truthy)
    if getattr(indexer, "_sparse_deferred_enabled", None) is None:
        indexer._sparse_deferred_enabled = os.getenv("SPLADE_DEFERRED_ENABLED", "1")
    if not _as_bool(indexer._sparse_deferred_enabled, default=True):
        return

    g = globals()
    if "_SPLADE_WORKER_GUARD" not in g:
        g["_SPLADE_WORKER_GUARD"] = threading.Lock()

    with g["_SPLADE_WORKER_GUARD"]:
        if "_SPLADE_QUEUE_GLOBAL" not in g:
            g["_SPLADE_QUEUE_GLOBAL"] = queue.Queue(maxsize=int(os.getenv("SPLADE_QUEUE_MAX", "10000")))

        if "_SPLADE_WORKER_GLOBAL" not in g or not getattr(g["_SPLADE_WORKER_GLOBAL"], "is_alive", lambda: False)():
            g["_SPLADE_WORKER_GLOBAL"] = threading.Thread(
                target=sparse_worker_loop_impl,
                args=(None,),
                name="splade_worker_global",
                daemon=True
            )
            g["_SPLADE_WORKER_GLOBAL"].start()

    # compat: indexer apunta a la cola global (para qsize/metrics)
    indexer._sparse_queue = g["_SPLADE_QUEUE_GLOBAL"]

def enqueue_sparse_update_impl(indexer, point_id: str, text: str):
    """
    Encola un trabajo SPLADE (GLOBAL) si está habilitado modo diferido.
    No bloquea el path crítico de upsert.
    """
    def _as_bool(v, default=True):
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in ("1", "true", "yes", "on")

    try:
        ensure_sparse_worker_impl(indexer)
        if not _as_bool(getattr(indexer, "_sparse_deferred_enabled", "1"), default=True):
            return

        q = getattr(indexer, "_sparse_queue", None)
        if q is None:
            return

        if not point_id or not isinstance(text, str) or not text.strip():
            return

        prepared = indexer._prep_doc(text)

        task = {
            "qdrant_client": indexer.qdrant_client,
            "collection_name": indexer.collection_name,
            "point_id": point_id,
            "text": prepared,
        }

        try:
            q.put_nowait(task)
        except Exception:
            # cola llena: no bloqueamos el upsert
            logger.warning("(splade-deferred) Cola llena; descarto tarea para no bloquear upsert.")
            return

        # metrics (best-effort)
        try:
            qg, _, _ = indexer._get_sparse_metrics()
        except Exception:
            qg = _NoopMetric()
        try:
            qg.labels(collection=indexer.collection_name).set(q.qsize())
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"(splade-deferred) No se pudo encolar tarea: {e}")

def sparse_worker_loop_impl(_unused_indexer):
    """
    Worker GLOBAL que calcula SPLADE y actualiza el vector 'sparse' en Qdrant.
    """
    from queue import Empty

    g = globals()
    q = g.get("_SPLADE_QUEUE_GLOBAL")
    if q is None:
        # si por algún motivo se invoca sin init, creamos una cola mínima
        import queue
        q = queue.Queue(maxsize=int(os.getenv("SPLADE_QUEUE_MAX", "10000")))
        g["_SPLADE_QUEUE_GLOBAL"] = q

    if os.getenv("QDRANT_ENABLE_SPARSE", "1") != "1":
        logger.warning("(splade-deferred) Canal sparse desactivado; worker quedará en idle.")

    while True:
        task = None
        try:
            try:
                task = q.get(timeout=0.5)
            except Empty:
                continue

            if not isinstance(task, dict):
                q.task_done()
                continue

            point_id = task.get("point_id")
            text = task.get("text") or ""
            qdrant_client = task.get("qdrant_client")
            collection_name = task.get("collection_name")

            if not qdrant_client or not collection_name or not point_id or not isinstance(text, str) or not text.strip():
                q.task_done()
                continue

            t0 = time.monotonic()
            try:
                idxs, vals = _splade_encode_prepared_text_impl(text)
                if idxs and vals:
                    _update_sparse_vectors_compat_client_impl(
                        qdrant_client=qdrant_client,
                        collection_name=collection_name,
                        point_id=point_id,
                        indices=idxs,
                        values=vals,
                    )
            except RuntimeError as e:
                if "SPLADE_DISABLED" in str(e):
                    # canal sparse desactivado: no hacemos nada
                    pass
                else:
                    raise
            finally:
                q.task_done()

        except Exception as e:
            pid = None
            try:
                if isinstance(task, dict):
                    pid = task.get("point_id")
            except Exception:
                pid = None
            logger.warning(f"(splade-deferred) Error procesando tarea id={pid}: {e}")
            time.sleep(0.2)


# ---------- Impl: SPLADE encoders & PRF ----------
def get_splade_model_impl(indexer):
    """
    Carga (perezosa) un modelo SPLADE y lo comparte globalmente.
    """
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    global _SPLADE_GLOBAL, _SPLADE_DEVICE

    cache_root = os.path.expanduser(os.getenv("COSMOS_CACHE", "~/.cache/cosmos"))

    primary = os.getenv("SPLADE_MODEL_ID", "").strip()
    fallback_csv = os.getenv("SPLADE_MODEL_CANDIDATES", "").strip()
    candidates = [c.strip() for c in fallback_csv.split(",") if c.strip()]

    defaults = [
        "naver/splade-cocondenser-ensembledistil",
        "jpwahle/splade-cocondenser-ensembledistil-safetensors",
        "prithivida/Splade_PP_en_v1"
    ]
    tried = set()
    ordered = [primary] if primary else []
    for c in candidates:
        if c and c not in ordered:
            ordered.append(c)
    for d in defaults:
        if d not in ordered:
            ordered.append(d)

    last_exc = None

    if _SPLADE_GLOBAL is None:
        with _SPLADE_LOCK:
            if _SPLADE_GLOBAL is None:
                for model_id in ordered:
                    if not model_id or model_id in tried:
                        continue
                    tried.add(model_id)
                    cache_dir = os.path.join(cache_root, _safe_dir_name(model_id))
                    try:
                        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
                        mdl = AutoModelForMaskedLM.from_pretrained(
                            model_id,
                            cache_dir=cache_dir,
                            use_safetensors=True
                        )

                        # eval + sin gradientes
                        try:
                            mdl.eval()
                            for p in mdl.parameters():
                                p.requires_grad_(False)
                        except Exception:
                            pass

                        dev = gpu_manager.best_device(
                            min_free_mb=int(os.getenv("SPLADE_MIN_FREE_MB", "2048"))
                        )
                        if dev.type == "cuda":
                            logger.info(f"(splade) moviendo '{model_id}' a {dev}")
                            mdl.to(dev)
                        else:
                            logger.info(f"(splade) usando CPU para '{model_id}'")

                        _SPLADE_GLOBAL = (tok, mdl)
                        _SPLADE_DEVICE = dev
                        logger.info(f"(splade) cargado '{model_id}' con safetensors desde {cache_dir}")
                        break
                    except ValueError as e:
                        if "upgrade torch to at least v2.6" in str(e).lower():
                            logger.warning(
                                f"(splade) '{model_id}' no ofrece safetensors o tu torch < 2.6. Probando otro…"
                            )
                            last_exc = e
                            continue
                        last_exc = e
                    except Exception as e:
                        logger.warning(f"(splade) fallo con '{model_id}': {e}. Probando otro…")
                        last_exc = e

                if _SPLADE_GLOBAL is None:
                    logger.error(
                        "(splade) No pude cargar un modelo SPLADE con safetensors. Desactivo el canal sparse."
                    )
                    os.environ["QDRANT_ENABLE_SPARSE"] = "0"
                    raise RuntimeError("SPLADE_DISABLED_NO_SAFETENSORS") from last_exc

    tok, mdl = _SPLADE_GLOBAL

    # Si nos pasan indexer real, mantenemos compat guardando en el indexer
    if indexer is not None:
        try:
            indexer._splade_tok_mdl = (tok, mdl)
        except Exception:
            pass

    return tok, mdl

def _splade_encode_prepared_text_impl(text: str):
    """
    Igual que sparse_encode_text_impl pero asume texto ya preprocesado.
    Permite usar SPLADE desde un worker global sin depender del indexer.
    """
    import numpy as np

    if os.getenv("QDRANT_ENABLE_SPARSE", "1") != "1":
        raise RuntimeError("SPLADE_DISABLED")

    global _SPLADE_DEVICE

    # asegura modelo global
    tok, mdl = _SPLADE_GLOBAL if _SPLADE_GLOBAL is not None else get_splade_model_impl(None)

    with gpu_manager.use_device_with_fallback(
        mdl,
        prefer_device=_SPLADE_DEVICE,
        min_free_mb=int(os.getenv("SPLADE_MIN_FREE_MB", "2048")),
        fallback_to_cpu=True,
        pool="splade",
    ) as dev:
        if dev != _SPLADE_DEVICE:
            _SPLADE_DEVICE = dev

        enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
        if dev.type == "cuda":
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

        with torch.inference_mode():
            out = mdl(**enc).logits  # [B, T, V]

        vmax = out.max(dim=1).values.squeeze(0)  # [V]
        relu = torch.relu(vmax)
        vals = torch.log1p(relu).cpu().numpy()
        idx = np.where(vals > 0)[0].astype(np.int32)
        v = vals[idx].astype(np.float32)

    return idx.tolist(), v.tolist()

def sparse_encode_text_impl(indexer, text: str):
    """
    SPLADE: pesos = log(1 + ReLU(max_logits_por_vocab)).
    Devuelve (indices, values). Lanza RuntimeError si sparse está desactivado.
    """
    if os.getenv("QDRANT_ENABLE_SPARSE", "1") != "1":
        raise RuntimeError("SPLADE_DISABLED")

    # mantiene compat: guarda tok/mdl en indexer si existe
    try:
        if getattr(indexer, "_splade_tok_mdl", None) is None:
            get_splade_model_impl(indexer)
    except Exception:
        pass

    prepared = indexer._prep_doc(text)
    return _splade_encode_prepared_text_impl(prepared)


def sparse_encode_query_impl(
    indexer,
    query: str,
    prf_topk: int = 5,
    prf_docs: int = 100,
    qfilter: Optional[qmodels.Filter] = None,
):
    """
    SPLADE de la query + PRF (RM3) sobre top docs (opcional, best-effort).

    Cambios clave:
    - La query se prepara con _prep_query y se codifica DIRECTO (evita pasar por sparse_encode_text_impl,
      que usa _prep_doc y podía prefijar mal).
    - El PRF respeta qfilter (si viene) para no expandir con documentos fuera del scope.
    - En PRF pedimos with_payload=True (si no, payload suele venir vacío).
    """
    # --- canal sparse habilitado ---
    if os.getenv("QDRANT_ENABLE_SPARSE", "1") != "1":
        raise RuntimeError("SPLADE_DISABLED")

    # --- encode query (con prefijo correcto) ---
    try:
        prepared_q = indexer._prep_query(query or "")
        q_idx, q_val = _splade_encode_prepared_text_impl(prepared_q)
    except RuntimeError as e:
        if "SPLADE_DISABLED" in str(e):
            raise
        raise
    except Exception as e:
        # si SPLADE está roto, dejamos que el caller degrade a denso+léxico
        raise RuntimeError(f"SPLADE_DISABLED: {e}")

    # --- PRF (best-effort) ---
    try:
        prf_topk = max(0, int(prf_topk or 0))
        prf_docs = max(0, int(prf_docs or 0))
    except Exception:
        prf_topk, prf_docs = 5, 100

    if prf_topk <= 0 or prf_docs <= 0:
        return q_idx, q_val

    try:
        base_vec = _make_sparse_query_vector(q_idx, q_val)

        # filtro: asegurar user si no está ya en qfilter
        base_must: List[qmodels.FieldCondition] = []
        if not _filter_has_user_must(qfilter):
            base_must.append(
                qmodels.FieldCondition(
                    key="user",
                    match=qmodels.MatchValue(value=indexer.user_id)
                )
            )
        flt = _merge_filters(base_must, qfilter)

        hits = indexer.qdrant_client.search(
            collection_name=indexer.collection_name,
            query_vector=base_vec,
            limit=int(prf_docs),
            with_vectors=False,
            with_payload=True,   # ✅ necesario para leer texto
            query_filter=flt
        )

        acc: Dict[int, float] = {}
        m = min(int(prf_topk), len(hits))
        for h in hits[:m]:
            pld = getattr(h, "payload", None) or {}
            preview = (
                pld.get("text")
                or pld.get("chunk")
                or pld.get("chunk_preview")
                or ""
            )
            if not isinstance(preview, str) or not preview.strip():
                continue

            di, dv = sparse_encode_text_impl(indexer, preview)
            for i, w in zip(di, dv):
                try:
                    ii = int(i)
                    acc[ii] = acc.get(ii, 0.0) + float(w)
                except Exception:
                    continue

        if acc:
            alpha = float(os.getenv("PRF_RM3_ALPHA", "0.7"))
            alpha = min(1.0, max(0.0, alpha))

            from collections import defaultdict
            comb: Dict[int, float] = defaultdict(float)

            for i, w in zip(q_idx, q_val):
                comb[int(i)] += alpha * float(w)

            denom = max(1.0, float(m))
            for i, w in acc.items():
                comb[int(i)] += (1.0 - alpha) * float(w) / denom

            ci = list(comb.keys())
            cv = [float(comb[k]) for k in ci]
            return ci, cv

    except RuntimeError:
        # si algo de Qdrant/SPLADE falla, degradamos a query base
        pass
    except Exception as e:
        logger.warning(f"(sparse_prf) PRF falló, uso query base: {e}")

    return q_idx, q_val


# ---------- Impl: búsqueda sparse ----------
def sparse_search_raw_impl(
    indexer,
    query: str,
    limit: int = 100,
    qfilter: Optional[qmodels.Filter] = None
) -> List[Dict[str, object]]:
    """
    Búsqueda learned-sparse (SPLADE) si está habilitado.

    Cambios clave:
    - Respeta qfilter COMPLETO (must/should/must_not) vía _merge_filters.
    - No duplica el filtro de user si ya viene en qfilter.
    - Usa payload nuevo: 'text', 'file_name', 'doc_id' (fallback a legacy).
    - Devuelve estructura consistente con vector_search_raw_impl:
        uid, doc_id, text, doc_name, score, rank, meta
    """
    if os.getenv("QDRANT_ENABLE_SPARSE", "1") != "1":
        return []

    # saneo de límites
    try:
        lim = int(limit or 1)
    except Exception:
        lim = 100
    lim = max(1, min(lim, 500))

    q = (query or "").strip()
    if not q:
        return []

    # encode sparse query (+PRF) respetando qfilter
    try:
        idx, val = sparse_encode_query_impl(indexer, q, qfilter=qfilter)
    except RuntimeError as e:
        if "SPLADE_DISABLED" in str(e):
            logger.warning("(sparse) Canal sparse desactivado; continúo con denso+léxico.")
            return []
        raise

    # construir filtro final (asegura user si hace falta)
    base_must: List[qmodels.FieldCondition] = []
    if not _filter_has_user_must(qfilter):
        base_must.append(
            qmodels.FieldCondition(
                key="user",
                match=qmodels.MatchValue(value=indexer.user_id)
            )
        )
    flt = _merge_filters(base_must, qfilter)

    qv = _make_sparse_query_vector(idx, val)

    hits = indexer.qdrant_client.search(
        collection_name=indexer.collection_name,
        query_vector=qv,
        limit=lim,
        with_vectors=False,
        with_payload=True,
        query_filter=flt
    )

    results: List[Dict[str, object]] = []

    # campos meta (alineado con vector_search_raw_impl)
    meta_fields = (
        "page",
        "section_path",
        "section_ancestors",
        "tags",
        "backend",
        "file_name",
        "file_name_canon",
        "file_stem_canon",
        "file_ext",
        "source_path",
        "sheet",
        "row_idx",
        "headers",
        "row_kv",
        "table_id",
        "table_cache_path",
        "block_type",
        "chunk_index",
        "brand",
        "model",
        "class",
        "location",
        "asset_type",
        "serial",
        "inventory_id",
        "asset_id",
        "quantity",
        "price",
        "doc_id",
    )

    for rank, h in enumerate(hits or [], start=1):
        pld = getattr(h, "payload", None) or {}
        uid = str(getattr(h, "id", "") or "")

        txt = (
            pld.get("text")
            or pld.get("chunk")
            or pld.get("chunk_preview")
            or ""
        )
        if not isinstance(txt, str):
            txt = str(txt or "")

        dname = (
            pld.get("file_name")
            or pld.get("doc_name")
            or pld.get("file")
            or ""
        )
        if not isinstance(dname, str):
            dname = str(dname or "")

        # doc_id preferente (sha1 por fichero); fallback legacy chunk_id si existiera
        doc_id = pld.get("doc_id")
        if doc_id is None:
            legacy_chunk_id = pld.get("chunk_id")
            if isinstance(legacy_chunk_id, (int, str)):
                doc_id = legacy_chunk_id

        meta = {k: pld.get(k) for k in meta_fields if pld.get(k) is not None}
        if doc_id is not None:
            meta.setdefault("doc_id", doc_id)

        results.append({
            "uid": uid,
            "doc_id": doc_id,
            "text": txt.strip(),
            "doc_name": dname,
            "score": float(getattr(h, "score", 0.0) or 0.0),
            "rank": int(rank),
            "meta": meta,
        })

    return results


# rag_upsert.py
import logging
from typing import List, Dict
from qdrant_client.http import models as qmodels
import os
from contextlib import contextmanager
import torch

from rag_metrics import (
    DOCUMENTS_PROCESSED,
    EMBEDDING_DURATION,
    BATCH_UPSERT_DURATION,
    INDEXED_CHUNKS,
    QDRANT_COLLECTION_SIZE,
)

from device_manager import gpu_manager


@contextmanager
def _ingest_gpu_slot(indexer):
    """
    Cupo dinámico para INGEST usando el PoolLimiter del device_manager (pool="ingest").

    - Evita tormentas GPU en upserts sin competir igual que SEARCH.
    - No añade semáforos extra (ya hay cupo por GPU y pool).
    - No “pega” embed_device a CPU si hubo fallback puntual.
    """
    prefer = getattr(indexer, "embed_device", None)

    # Permite override específico para ingestas
    try:
        min_free = int(os.getenv("INGEST_MIN_FREE_MB", os.getenv("HF_EMBED_MIN_FREE_MB", "2048")))
    except Exception:
        min_free = 2048

    mdl = getattr(indexer, "sbert_model", None)
    if mdl is None:
        yield torch.device("cpu")
        return

    # Si no hay CUDA usable/permitida, no bloqueamos ni tocamos nada
    cuda_ok = False
    try:
        cuda_ok = bool(torch.cuda.is_available()) and bool(gpu_manager.list_devices())
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        yield torch.device("cpu")
        return

    with gpu_manager.use_device_with_fallback(
        mdl,
        prefer_device=prefer,
        min_free_mb=min_free,
        fallback_to_cpu=True,
        pool="ingest",
    ) as dev:
        # Solo actualizamos preferencia si realmente usamos CUDA
        try:
            if dev.type == "cuda" and dev != prefer:
                indexer.embed_device = dev
        except Exception:
            pass
        yield dev

logger = logging.getLogger("cosmos_nlp_v3.upsert")

def upsert_documents_to_db_impl(indexer, documents: List[str], doc_names: List[str]):
    """
    Upsert robusto y consistente con el payload nuevo y el pipeline de search:

    - Dense embeddings (E5) en streaming.
    - SPLADE inline o deferred (sin doble prefijo).
    - Payload enriquecido (base_payload) + legacy payload.
    - MUY IMPORTANTE: si indexer.client_tag existe, se inserta también en payload['tags']
      para que search_impl (que filtra por client_tag y tags a la vez) no se quede sin hits.
    - Batching adaptativo por bytes/puntos + retry binario si “payload too large”.
    """
    import time
    import uuid
    import datetime
    import numpy as np
    import os
    import hashlib
    from typing import Any, Tuple, Optional

    logger = logging.getLogger("cosmos_nlp_v3.upsert")

    if not documents:
        return 0

    # Normalizar doc_names
    if not isinstance(doc_names, list):
        doc_names = []
    if len(doc_names) != len(documents):
        # relleno defensivo: nunca truncar documents
        doc_names = list(doc_names) + [""] * max(0, len(documents) - len(doc_names))
        doc_names = doc_names[: len(documents)]

    # Asegurar colección + índices de payload
    try:
        indexer._ensure_collection()
    except Exception as e:
        logger.warning(f"(upsert) No se pudo asegurar la colección: {e}")

    try:
        indexer._ensure_payload_indexes()
    except Exception as e:
        logger.warning(f"(upsert) No se pudieron crear índices de payload (continuo): {e}")

    # Best-effort: índices útiles para filtros frecuentes
    try:
        from qdrant_client.models import PayloadSchemaType
        for field, schema in (("user", PayloadSchemaType.KEYWORD), ("client_tag", PayloadSchemaType.KEYWORD)):
            try:
                indexer.qdrant_client.create_payload_index(
                    collection_name=indexer.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass
    except Exception:
        pass

    total_docs = len(documents)
    logger.info(f"(upsert) Comenzando upsert de {total_docs} fragmentos…")

    # -------- Config ----------
    PREVIEW_CHARS     = int(os.getenv("PAYLOAD_CHUNK_PREVIEW_CHARS", "220"))
    CHUNK_MAX_CHARS   = int(os.getenv("QDRANT_PAYLOAD_CHUNK_MAXCHARS", "4096"))
    EMBED_MAX_CHARS   = int(os.getenv("QDRANT_EMBED_TEXT_MAXCHARS", str(CHUNK_MAX_CHARS)))
    BATCH_EMB         = int(os.getenv("HF_EMBED_BATCH", "32"))
    MAX_POINTS_BATCH  = int(os.getenv("QDRANT_UPSERT_CHUNK", "200"))
    MAX_REQ_MB        = float(os.getenv("QDRANT_MAX_REQUEST_MB", "24"))
    MAX_REQ_BYTES     = int(MAX_REQ_MB * 1024 * 1024)
    UPSERT_WAIT       = os.getenv("QDRANT_UPSERT_WAIT", "0") == "1"

    dense_name        = os.getenv("QDRANT_DENSE_NAME", "dense")
    sparse_enabled    = os.getenv("QDRANT_ENABLE_SPARSE", "1") == "1"
    sparse_name       = os.getenv("QDRANT_SPARSE_NAME", "sparse")

    # deferred activo cuando == "1"
    splade_deferred   = os.getenv("QDRANT_SPLADE_DEFERRED", "0") == "1"

    EMBED_CHUNK = int(os.getenv("QDRANT_UPSERT_EMBED_CHUNK", str(MAX_POINTS_BATCH)))
    EMBED_CHUNK = max(1, EMBED_CHUNK)
    EMBED_MAX_CHARS = max(256, EMBED_MAX_CHARS)

    # Registrar tag en Redis UNA vez
    if getattr(indexer, "client_tag", None):
        try:
            from redis import Redis as SyncRedis
            r = SyncRedis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=0,
                decode_responses=True,
            )
            r.sadd(f"tags:{indexer.collection_name}", indexer.client_tag)
        except Exception:
            pass

    def _is_size_error(msg: str) -> bool:
        t = (msg or "").lower()
        keys = [
            "larger than allowed", "too large", "request body is too large",
            "payload size", "exceeds limit", "exceeds maximum", "body too large"
        ]
        return any(k in t for k in keys)

    # buffers batch
    points_batch: List[qmodels.PointStruct] = []
    pending_sparse: List[Tuple[str, str]] = []  # (point_id, text_raw) para deferred
    approx_bytes = 0

    PER_FLOAT = 4
    META_OVERHEAD = 700  # algo conservador por payload enriquecido

    def flush_with_retry(batch_points: List[qmodels.PointStruct], sparse_to_enqueue: List[Tuple[str, str]]) -> None:
        nonlocal approx_bytes

        if not batch_points:
            return

        logger.info(f"(upsert) Upsertando lote de {len(batch_points)} puntos…")
        start_batch = time.time()

        def _try(points: List[qmodels.PointStruct]):
            try:
                indexer.qdrant_client.upsert(
                    collection_name=indexer.collection_name,
                    points=points,
                    wait=UPSERT_WAIT
                )
            except Exception as e:
                msg = str(e)
                if _is_size_error(msg) and len(points) > 1:
                    mid = len(points) // 2
                    _try(points[:mid])
                    _try(points[mid:])
                else:
                    raise

        _try(batch_points)

        # Encolar sparse SOLO tras éxito del upsert
        if splade_deferred and sparse_to_enqueue:
            for pid, txt in sparse_to_enqueue:
                try:
                    indexer._enqueue_sparse_update(pid, (txt or "")[:CHUNK_MAX_CHARS])
                except Exception as ee:
                    logger.warning(f"(upsert) No se pudo encolar SPLADE para {pid}: {ee}")

        BATCH_UPSERT_DURATION.labels(collection=indexer.collection_name).observe(time.time() - start_batch)
        approx_bytes = 0
        batch_points.clear()
        sparse_to_enqueue.clear()

    # SPLADE readiness
    splade_ready = False
    if sparse_enabled:
        if splade_deferred:
            try:
                indexer._ensure_sparse_worker()
                splade_ready = True
            except Exception as e:
                logger.warning(f"(upsert) No se pudo iniciar worker SPLADE diferido: {e}")
                splade_ready = False
        else:
            try:
                cur = getattr(indexer, "_splade_tok_mdl", None)
                if cur is None or cur[0] is None:
                    indexer._splade_tok_mdl = indexer._get_splade_model()
                splade_ready = True
            except RuntimeError as e:
                logger.warning(f"(upsert) SPLADE no disponible ({e}); continúo sin sparse.")
                splade_ready = False

    # -------- meta alignment robusto ----------
    # Caso ideal: load_and_fragment_files_impl ya extendió indexer._metas con metas nuevas
    metas = getattr(indexer, "_metas", None) or []
    docs_known = len(getattr(indexer, "documents", []) or [])
    metas_len = len(metas)

    # start candidato: metas al final para estos documents (incremental típico)
    meta_start = metas_len - total_docs if metas_len >= total_docs else 0

    # Heurística anti-error: si metas==docs_known y vamos a upsert solo una parte (incremental)
    # probablemente NO hay metas nuevas aún -> usar {} antes que “metas antiguas equivocadas”.
    no_new_metas_available = (metas_len == docs_known and total_docs != docs_known)

    def _meta_for(i: int, doc_name: str) -> Dict[str, Any]:
        if no_new_metas_available:
            return {}
        mi = meta_start + i
        if mi < 0 or mi >= metas_len:
            return {}

        m = metas[mi] if isinstance(metas[mi], dict) else {}
        if not m:
            return {}

        # Validación ligera: si el meta tiene file_name/source_path, que parezca del doc_name
        try:
            import os as _os
            dn = str(doc_name or "")
            dn_base = _os.path.basename(dn.split("::")[0]) if dn else ""
            mf = (m.get("file_name") or m.get("source_file_name") or "")
            ms = (m.get("source_path") or "")
            mf_base = _os.path.basename(ms) if ms else str(mf)

            if dn_base and mf_base and dn_base.lower() != str(mf_base).lower():
                # si no coincide, mejor no usarla (evita mezclar Excel meta de otro doc)
                return {}
        except Exception:
            pass

        return m

    # -------- Embeddings densos (streaming) ----------
    emb_t0 = time.time()

    for chunk_start in range(0, total_docs, EMBED_CHUNK):
        chunk_end = min(total_docs, chunk_start + EMBED_CHUNK)
        docs_chunk = documents[chunk_start:chunk_end]
        names_chunk = doc_names[chunk_start:chunk_end]

        # Acotar texto para embedding (evita secuencias extremas y latencias/picos innecesarios)
        docs_for_embed = [(t or "")[:EMBED_MAX_CHARS] for t in docs_chunk]

        # 1) Embedding del chunk bajo cupo INGEST (pool="ingest")
        try:
            docs_prefixed = [indexer._prep_doc(t) for t in docs_for_embed]
        except Exception:
            docs_prefixed = docs_for_embed

        vecs = None
        with _ingest_gpu_slot(indexer) as dev:
            device_arg = str(dev) if getattr(dev, "type", "cpu") == "cuda" else "cpu"
            try:
                vecs = indexer.sbert_model.encode(
                    docs_prefixed,
                    batch_size=BATCH_EMB,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    device=device_arg,
                )
            except TypeError:
                # compat con versiones donde encode no acepta device=
                vecs = indexer.sbert_model.encode(
                    docs_prefixed,
                    batch_size=BATCH_EMB,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as e:
                # fallback defensivo: CPU + batch menor
                logger.warning(f"(upsert) embedding falló en {device_arg}: {e}. Reintento CPU/batch menor…")
                try:
                    vecs = indexer.sbert_model.encode(
                        docs_prefixed,
                        batch_size=max(1, BATCH_EMB // 2),
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        device="cpu",
                    )
                except Exception:
                    vecs = indexer.sbert_model.encode(
                        docs_prefixed,
                        batch_size=max(1, BATCH_EMB // 2),
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )

        if not isinstance(vecs, np.ndarray):
            vecs = np.asarray(vecs)
        vecs = vecs.astype(np.float32)

        # 2) Construcción de puntos + batching
        for j, (txt, name) in enumerate(zip(docs_chunk, names_chunk)):
            global_idx = chunk_start + j
            dense_vec = vecs[j].tolist()

            preview = (txt or "")[:PREVIEW_CHARS]
            chunk_full = (txt or "")[:CHUNK_MAX_CHARS]

            meta = _meta_for(global_idx, name)

            base_payload = indexer._build_payload(text=chunk_full, doc_name=name, meta=meta)

            # ✅ Asegurar que client_tag también esté en tags (evita filtro AND sin hits)
            ct = getattr(indexer, "client_tag", None)
            if isinstance(ct, str) and ct.strip():
                tags = base_payload.get("tags")
                if not isinstance(tags, list):
                    tags = []
                # de-dup estable
                seen = {str(t).strip().lower() for t in tags if str(t).strip()}
                if ct.strip().lower() not in seen:
                    tags.append(ct.strip())
                base_payload["tags"] = tags

            legacy_payload = {
                "doc_name": name,
                "file": (name.split("::")[0] if isinstance(name, str) else ""),
                "chunk_preview": preview,
                "chunk": chunk_full,
                "user": indexer.user_id,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            if isinstance(ct, str) and ct.strip():
                legacy_payload["client_tag"] = ct.strip()

            payload = {**base_payload, **legacy_payload}

            point_id = str(uuid.uuid4())

            point_vector: Any = {dense_name: dense_vec}

            # SPLADE inline (si aplica) — ✅ NO doble _prep_doc
            if sparse_enabled and (not splade_deferred) and splade_ready:
                try:
                    idxs, vals = indexer._sparse_encode_text(chunk_full)
                    if idxs and vals:
                        point_vector[sparse_name] = qmodels.SparseVector(indices=idxs, values=vals)
                except Exception as e:
                    logger.warning(f"(upsert) SPLADE inline falló en idx {global_idx}: {e}")

            # SPLADE deferred: encolamos tras upsert OK
            if sparse_enabled and splade_deferred and splade_ready:
                pending_sparse.append((point_id, chunk_full))

            point = qmodels.PointStruct(id=point_id, vector=point_vector, payload=payload)

            approx_point = (
                len(dense_vec) * PER_FLOAT +
                META_OVERHEAD +
                len(preview.encode("utf-8", errors="ignore")) +
                len(chunk_full.encode("utf-8", errors="ignore"))
            )

            if (approx_bytes + approx_point > MAX_REQ_BYTES) or (len(points_batch) >= MAX_POINTS_BATCH):
                flush_with_retry(points_batch, pending_sparse)
                approx_bytes = 0

            points_batch.append(point)
            approx_bytes += approx_point

            DOCUMENTS_PROCESSED.labels(collection=indexer.collection_name).inc()
            INDEXED_CHUNKS.labels(collection=indexer.collection_name).inc()

    EMBEDDING_DURATION.labels(collection=indexer.collection_name).observe(time.time() - emb_t0)

    flush_with_retry(points_batch, pending_sparse)

    total = indexer.qdrant_client.count(collection_name=indexer.collection_name, exact=True).count
    QDRANT_COLLECTION_SIZE.labels(collection=indexer.collection_name).set(total)
    logger.info(f"(upsert) Upsert completado. Colección '{indexer.collection_name}' tiene ahora {total} puntos.")

    return total

def delete_document_by_name_impl(indexer, filename_or_path: str) -> int:
    """
    Borra de forma consistente TODO lo relacionado con un fichero:

    - Puntos en Qdrant (must=user) y (should por doc_id/file_name/source_path/canónicos).
    - Fragmentos en memoria (indexer.documents, indexer.doc_names, indexer._metas).
    - Pickles en disco (_documents.pkl, _docnames.pkl, _metas.pkl, _schema.json).
    - Invalida BM25.

    Devuelve el número de fragmentos eliminados en memoria.
    """
    import os
    import pickle
    import json
    import inspect
    import hashlib

    base_name = os.path.basename(filename_or_path or "")
    norm_target = os.path.normcase(os.path.normpath(filename_or_path or ""))

    keep_docs: list = []
    keep_names: list = []
    keep_metas: list = []

    # -------------------------------
    # 1) Limpiar arrays en memoria
    # -------------------------------
    for text, name, meta in zip(indexer.documents, indexer.doc_names, indexer._metas):
        m = meta or {}

        src_path = (m.get("source_path") or m.get("path") or "")
        src_norm = os.path.normcase(os.path.normpath(src_path)) if src_path else None
        src_base = os.path.basename(src_path) if src_path else None

        mf = (m.get("file_name") or m.get("source_file_name") or "")
        mf_base = os.path.basename(str(mf)) if mf else None

        dn = str(name or "")
        dn_base = os.path.basename(dn.split("::")[0]) if dn else None

        belongs = False
        if src_norm and norm_target and src_norm == norm_target:
            belongs = True
        elif base_name and src_base and src_base == base_name:
            belongs = True
        elif base_name and mf_base and mf_base == base_name:
            belongs = True
        elif base_name and dn_base and dn_base == base_name:
            belongs = True
        elif base_name and isinstance(name, str) and name.startswith(base_name + "::"):
            belongs = True

        if not belongs:
            keep_docs.append(text)
            keep_names.append(name)
            keep_metas.append(m)

    removed_count = len(indexer.documents) - len(keep_docs)

    indexer.documents = keep_docs
    indexer.doc_names = keep_names
    indexer._metas = keep_metas

    # -------------------------------
    # 2) Borrado en Qdrant (must=user + should=varios identificadores)
    # -------------------------------
    try:
        conditions = []

        # file_name exact
        if base_name:
            conditions.append(qmodels.FieldCondition(
                key="file_name",
                match=qmodels.MatchValue(value=base_name),
            ))

        # source_path exact
        if filename_or_path:
            conditions.append(qmodels.FieldCondition(
                key="source_path",
                match=qmodels.MatchValue(value=filename_or_path),
            ))

        # doc_id (sha1 por (source_path or file_name) según _build_payload)
        def _sha1(s: str) -> str:
            return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

        if base_name:
            conditions.append(qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchValue(value=_sha1(base_name)),
            ))
        if filename_or_path:
            conditions.append(qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchValue(value=_sha1(filename_or_path)),
            ))

        # canónicos (best-effort)
        try:
            canon = getattr(indexer, "_canon", None)
            if callable(canon) and base_name:
                conditions.append(qmodels.FieldCondition(
                    key="file_name_canon",
                    match=qmodels.MatchValue(value=canon(base_name)),
                ))
                stem = os.path.splitext(base_name)[0]
                conditions.append(qmodels.FieldCondition(
                    key="file_stem_canon",
                    match=qmodels.MatchValue(value=canon(stem)),
                ))
        except Exception:
            pass

        must = [
            qmodels.FieldCondition(
                key="user",
                match=qmodels.MatchValue(value=indexer.user_id)
            )
        ]

        qfilter = qmodels.Filter(must=must, should=conditions)

        delete_fn = indexer.qdrant_client.delete
        sig = inspect.signature(delete_fn)
        params = sig.parameters

        kwargs = {
            "collection_name": indexer.collection_name,
            "wait": True,
        }

        if "points_selector" in params:
            kwargs["points_selector"] = qfilter
        else:
            kwargs["filter"] = qfilter

        delete_fn(**kwargs)

    except Exception as e:
        logger.warning(f"(delete_document_by_name) Error al borrar en Qdrant '{filename_or_path}': {e}")

    # -------------------------------
    # 3) Persistir pickles
    # -------------------------------
    try:
        index_filepath = getattr(indexer, "index_filepath", None)
        if index_filepath:
            base_path = os.path.splitext(index_filepath)[0]
            docs_path = f"{base_path}_documents.pkl"
            names_path = f"{base_path}_docnames.pkl"
            metas_path = f"{base_path}_metas.pkl"
            schema_json = f"{base_path}_schema.json"

            with indexer._get_user_lock():
                indexer._atomic_write_file(docs_path, pickle.dumps(indexer.documents))
                indexer._atomic_write_file(names_path, pickle.dumps(indexer.doc_names))
                indexer._atomic_write_file(metas_path, pickle.dumps(indexer._metas))
                schema = {"version": 2, "documents_len": len(indexer.documents)}
                indexer._atomic_write_file(
                    schema_json,
                    json.dumps(schema, ensure_ascii=False, indent=2).encode("utf-8"),
                )
    except Exception as e:
        logger.warning(f"(delete_document_by_name) Error guardando pickles para '{filename_or_path}': {e}")

    # -------------------------------
    # 4) Invalida BM25
    # -------------------------------
    try:
        indexer._invalidate_bm25_if_needed()
    except Exception:
        pass

    logger.info(f"(delete_document_by_name) Eliminados {removed_count} fragmentos relacionados con '{filename_or_path}'")
    return removed_count

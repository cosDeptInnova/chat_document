# rag_search.py
import os
import time
from typing import List, Dict, Tuple, Optional, Any

from qdrant_client.http import models as qmodels

from rag_metrics import SEARCH_REQUESTS, SEARCH_LATENCY


def _safe_named_vector(vector):
    """
    Devuelve NamedVector con nombre QDRANT_DENSE_NAME si es posible,
    convirtiendo el embedding a list para evitar errores pydantic.
    """
    dense_name = os.getenv("QDRANT_DENSE_NAME", "dense")
    vec = vector.tolist() if hasattr(vector, "tolist") else (list(vector) if not isinstance(vector, list) else vector)
    try:
        return qmodels.NamedVector(name=dense_name, vector=vec)
    except Exception:
        return vec


def _merge_filters(base_must: List[qmodels.FieldCondition], qfilter: Optional[qmodels.Filter]) -> qmodels.Filter:
    """
    Fusiona base_must con qfilter (si existe), preservando should y must_not.
    """
    if isinstance(qfilter, qmodels.Filter):
        must = list(base_must) + list(qfilter.must or [])
        should = list(qfilter.should or []) if qfilter.should else None
        must_not = list(qfilter.must_not or []) if qfilter.must_not else None
        return qmodels.Filter(must=must, should=should, must_not=must_not)
    return qmodels.Filter(must=list(base_must))


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _as_list(x: Any, max_items: int = 32) -> List[Any]:
    """
    Normaliza un valor a lista (para MatchAny).
    Limita tamaño por seguridad.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        out = [v for v in x if v is not None]
    else:
        out = [x]
    out = out[: max_items]
    # limpia strings vacíos
    clean = []
    for v in out:
        if isinstance(v, str):
            s = v.strip()
            if s:
                clean.append(s)
        else:
            clean.append(v)
    return clean


def _try_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return int(x)
    except Exception:
        return None


def _try_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None


def _canon(indexer, s: str) -> str:
    """
    Canonización defensiva (usa indexer._canon si existe).
    """
    try:
        if hasattr(indexer, "_canon") and callable(indexer._canon):
            return indexer._canon(s)
    except Exception:
        pass
    return str(s or "").strip().lower()


def _build_conditions_for_scalar(key: str, value: Any) -> List[qmodels.FieldCondition]:
    """
    Devuelve FieldConditions para una clave con MatchValue o MatchAny.
    """
    vals = _as_list(value)
    if not vals:
        return []
    if len(vals) == 1:
        return [qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=vals[0]))]
    return [qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=vals))]


def _build_range_condition(key: str, value: Any) -> Optional[qmodels.FieldCondition]:
    """
    Soporta rangos tipo:
      - {"min": 10, "max": 20}
      - {"gte": 10, "lte": 20}
      - [10, 20]
    """
    gte = lte = gt = lt = None

    if isinstance(value, dict):
        gte = _try_float(value.get("gte", value.get("min")))
        lte = _try_float(value.get("lte", value.get("max")))
        gt  = _try_float(value.get("gt"))
        lt  = _try_float(value.get("lt"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        gte = _try_float(value[0])
        lte = _try_float(value[1])

    if any(v is not None for v in (gte, lte, gt, lt)):
        return qmodels.FieldCondition(
            key=key,
            range=qmodels.Range(gte=gte, lte=lte, gt=gt, lt=lt),
        )
    return None


def _extract_user_conditions(qfilter: Optional[qmodels.Filter]) -> List[qmodels.FieldCondition]:
    """
    Extrae condiciones must de un Filter (defensivo).
    """
    if not isinstance(qfilter, qmodels.Filter):
        return []
    must = qfilter.must or []
    out = []
    for c in must:
        try:
            if isinstance(c, qmodels.FieldCondition):
                out.append(c)
        except Exception:
            pass
    return out


def _qfilter_has_user(qfilter: Optional[qmodels.Filter], user_id: str) -> bool:
    """
    Detecta si qfilter ya contiene user==user_id en must.
    """
    if not isinstance(qfilter, qmodels.Filter):
        return False
    for c in (qfilter.must or []):
        try:
            if getattr(c, "key", None) != "user":
                continue
            m = getattr(c, "match", None)
            if m is None:
                continue
            v = getattr(m, "value", None)
            if v is not None and str(v) == str(user_id):
                return True
        except Exception:
            continue
    return False


def _build_qdrant_filter_from_filters(
    indexer,
    filters: Optional[Dict[str, Any]],
) -> Tuple[List[qmodels.FieldCondition], List[qmodels.FieldCondition], List[qmodels.FieldCondition]]:
    """
    Convierte filters dict (planner/tool) a (must, should, must_not) Qdrant.

    Regla:
    - filtros “normales” → MUST
    - filtro por documento (documento/doc_name/file_name) → SHOULD (OR entre campos)
      y se exige que exista SHOULD en el Filter final (Qdrant lo interpreta como OR requerido).

    Backward-compatible:
    - soporta puntos viejos sin *_canon usando file_name
    - soporta puntos nuevos usando file_name_canon / file_stem_canon
    """
    if not isinstance(filters, dict) or not filters:
        return [], [], []

    must: List[qmodels.FieldCondition] = []
    should: List[qmodels.FieldCondition] = []
    must_not: List[qmodels.FieldCondition] = []

    # aliases defensivos (por si planner manda en español)
    alias = {
        "ubicacion": "location",
        "ubicación": "location",
        "clase": "class",
        "tipo": "class",
        "documento": "documento",
        "doc_name": "documento",
        "file_name": "documento",
    }

    # --- 1) Documento ---
    doc_hint = None
    for k in ("documento", "doc_name", "file_name"):
        v = filters.get(k)
        if _is_nonempty_str(v):
            doc_hint = v.strip()
            break

    if doc_hint:
        resolved = None
        try:
            if hasattr(indexer, "resolve_document_name") and callable(indexer.resolve_document_name):
                resolved = indexer.resolve_document_name(doc_hint)
        except Exception:
            resolved = None

        target = (resolved or doc_hint).strip()

        # file_name exact (legacy)
        should.extend(_build_conditions_for_scalar("file_name", target))

        # canon + stem canon (nuevo)
        fn_canon = _canon(indexer, target)
        should.extend(_build_conditions_for_scalar("file_name_canon", fn_canon))

        import os
        stem = os.path.splitext(target)[0]
        stem_canon = _canon(indexer, stem)
        should.extend(_build_conditions_for_scalar("file_stem_canon", stem_canon))

        # si parece ruta, también por source_path exact
        if ("/" in doc_hint) or ("\\" in doc_hint):
            should.extend(_build_conditions_for_scalar("source_path", doc_hint))

    # --- 2) Otros filtros (MUST) ---
    # campos keyword típicos del payload
    keyword_keys = {
        "backend",
        "file_ext",
        "sheet",
        "table_id",
        "table_cache_path",
        "brand",
        "model",
        "class",
        "location",
        "asset_type",
        "serial",
        "inventory_id",
        "asset_id",
        "doc_id",
        "tags",
    }

    # campos int típicos
    int_keys = {"page", "row_idx", "row_id", "chunk_index"}

    # campos num/range típicos
    range_keys = {"quantity", "price"}

    for raw_k, v in filters.items():
        if v is None:
            continue

        k = alias.get(str(raw_k).strip().lower(), str(raw_k).strip())

        # ya procesado como documento
        if k == "documento":
            continue

        # tags: permitir lista
        if k == "tags":
            vals = _as_list(v)
            if vals:
                must.extend(_build_conditions_for_scalar("tags", vals))
            continue

        if k in keyword_keys:
            if isinstance(v, dict):
                # por si alguien manda {"value": "..."}
                vv = v.get("value")
                if vv is not None:
                    must.extend(_build_conditions_for_scalar(k, vv))
            else:
                must.extend(_build_conditions_for_scalar(k, v))
            continue

        if k in int_keys:
            if isinstance(v, (list, tuple, set)):
                vals = [_try_int(x) for x in v]
                vals = [x for x in vals if x is not None]
                if vals:
                    must.append(qmodels.FieldCondition(key=k, match=qmodels.MatchAny(any=vals[:32])))
            else:
                iv = _try_int(v)
                if iv is not None:
                    must.append(qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=iv)))
            continue

        if k in range_keys:
            rc = _build_range_condition(k, v)
            if rc is not None:
                must.append(rc)
            else:
                # fallback exact
                fv = _try_float(v)
                if fv is not None:
                    must.append(qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=fv)))
            continue

        # keys desconocidas: se ignoran (robustez)
        continue

    return must, should, must_not


def _qfilter_is_constrained(qfilter: Optional[qmodels.Filter]) -> bool:
    """
    Determina si el filtro tiene restricciones “reales” más allá de user.
    Si es constrained, BM25 sin filtro puede contaminar la fusión => lo desactivamos.
    """
    if not isinstance(qfilter, qmodels.Filter):
        return False
    for c in (qfilter.must or []):
        try:
            if getattr(c, "key", None) and getattr(c, "key") != "user":
                return True
        except Exception:
            pass
    if qfilter.should:
        return True
    if qfilter.must_not:
        return True
    return False

def vector_search_raw_impl(
    indexer,
    query: str,
    limit: int = 100,
    qfilter: Optional[qmodels.Filter] = None
) -> List[Dict[str, object]]:
    """
    Búsqueda densa en Qdrant con vector nombrado (default 'dense').
    Devuelve dicts: uid, doc_id, text, doc_name, score, rank, meta.
    """
    limit = max(1, min(int(limit or 1), 500))

    qvec = indexer._embed_texts([query or ""], is_query=True)[0]
    named_vec = _safe_named_vector(qvec)

    # base filter por user (solo si qfilter no lo trae ya)
    base_must: List[qmodels.FieldCondition] = []
    try:
        if not _qfilter_has_user(qfilter, str(indexer.user_id)):
            base_must.append(
                qmodels.FieldCondition(
                    key="user",
                    match=qmodels.MatchValue(value=indexer.user_id)
                )
            )
    except Exception:
        # si algo raro, mejor asegurar user
        base_must = [
            qmodels.FieldCondition(key="user", match=qmodels.MatchValue(value=indexer.user_id))
        ]

    flt = _merge_filters(base_must, qfilter)

    try:
        hits = indexer.qdrant_client.search(
            collection_name=indexer.collection_name,
            query_vector=named_vec,
            limit=limit,
            with_vectors=False,
            with_payload=True,
            query_filter=flt
        )
    except TypeError:
        # compat firmas antiguas
        hits = indexer.qdrant_client.search(
            indexer.collection_name,
            named_vec,
            limit,
            with_vectors=False,
            with_payload=True,
            query_filter=flt
        )
    except Exception:
        # no romper: devolver vacío
        return []

    results: List[Dict[str, object]] = []
    for rank, h in enumerate(hits or [], start=1):
        uid = str(getattr(h, "id", "")) if getattr(h, "id", None) is not None else ""
        pld = getattr(h, "payload", None) or {}

        txt = (
            pld.get("text")
            or pld.get("chunk")
            or pld.get("chunk_preview")
            or ""
        )
        dname = (
            pld.get("file_name")
            or pld.get("doc_name")
            or pld.get("file")
            or ""
        )

        doc_id = pld.get("doc_id")
        if doc_id is None:
            legacy_chunk_id = pld.get("chunk_id")
            if isinstance(legacy_chunk_id, (int, str)):
                doc_id = legacy_chunk_id

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
        )
        meta = {k: pld.get(k) for k in meta_fields if pld.get(k) is not None}
        if doc_id is not None:
            meta.setdefault("doc_id", doc_id)

        results.append({
            "uid": uid,
            "doc_id": doc_id,
            "text": txt or "",
            "doc_name": dname,
            "score": float(getattr(h, "score", 0.0) or 0.0),
            "rank": rank,
            "meta": meta,
        })

    return results


def hybrid_search_rrf_impl(
    indexer,
    query: str,
    top_k: int = 10,
    rrf_k: int = 60,
    alpha_vector: float = 1.0,
    alpha_sparse: float = 1.0,
    alpha_lexical: float = 1.0,
    qfilter: Optional[qmodels.Filter] = None
) -> List[Tuple[str, str, float, float]]:
    """
    Fusión RRF: denso + sparse + (BM25 si no hay filtros restrictivos) + rerank.
    Retorna [(text, doc_name, score_rrf, rerank_score)] top_k.
    """
    from collections import defaultdict

    top_k = max(1, min(int(top_k or 10), 50))
    vec_limit = max(50, top_k * 5)
    sp_limit  = max(50, top_k * 5)
    lex_limit = max(50, top_k * 5)

    # Asegura BM25 build si no está listo (solo si lo usaremos)
    use_lexical = True
    if alpha_lexical <= 0:
        use_lexical = False
    if _qfilter_is_constrained(qfilter):
        # ✅ consistente: BM25 no está filtrado por Qdrant -> puede contaminar.
        use_lexical = False

    if use_lexical:
        try:
            if not getattr(indexer._bm25_index, "ready", False):
                indexer._bm25_index.build(indexer.documents)
        except Exception:
            use_lexical = False

    vec_hits = indexer.vector_search_raw(query, limit=vec_limit, qfilter=qfilter)
    sp_hits  = indexer.sparse_search_raw(query, limit=sp_limit,  qfilter=qfilter)

    lex_hits = []
    if use_lexical:
        try:
            lex_hits = indexer.lexical_search(query, limit=lex_limit)
        except Exception:
            lex_hits = []

    rrf_scores = defaultdict(float)
    meta_map: Dict[object, Tuple[str, str]] = {}

    def _accumulate(hits, alpha, channel="vec"):
        if not hits or alpha <= 0:
            return
        for h in hits:
            key = h.get("uid") or (channel, h.get("doc_id"), h.get("rank"))
            rank = int(h.get("rank", 10**9))
            rrf_scores[key] += float(alpha) / (rrf_k + max(1, rank))
            if key not in meta_map:
                meta_map[key] = (h.get("text", "") or "", h.get("doc_name", "") or "")

    _accumulate(vec_hits, alpha_vector, channel="vec")
    _accumulate(sp_hits,  alpha_sparse, channel="sparse")

    if lex_hits and alpha_lexical > 0:
        lex_adapted = [
            {
                "uid": None,
                "doc_id": h.get("doc_id"),
                "text": h.get("text"),
                "doc_name": h.get("doc_name"),
                "rank": h.get("rank")
            }
            for h in (lex_hits or [])
        ]
        _accumulate(lex_adapted, alpha_lexical, channel="lex")

    if not rrf_scores:
        return []

    top_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(top_k) * 3)]

    retrieved_chunks: List[Tuple[str, str, float]] = []
    for key, sc in top_rrf:
        txt, dname = meta_map.get(key, ("", ""))
        # fallback por índice si doc_id es int (lex channel típico)
        if not txt and isinstance(key, tuple) and len(key) >= 3 and isinstance(key[1], int):
            did = key[1]
            if 0 <= did < len(indexer.documents):
                txt = indexer.documents[did]
                dname = indexer.doc_names[did] if did < len(indexer.doc_names) else dname
        if txt:
            retrieved_chunks.append((txt, dname, float(sc)))

    if not retrieved_chunks:
        return []

    reranked = indexer.rerank_results(query=query, retrieved_chunks=retrieved_chunks, top_k=top_k)
    if not reranked:
        return [(t, n, float(sc), 0.0) for (t, n, sc) in retrieved_chunks[:top_k]]

    rrf_lookup = {(t, n): s for (t, n, s) in retrieved_chunks}
    out: List[Tuple[str, str, float, float]] = []
    for (t, n, _orig, rerank_score) in reranked:
        out.append((t, n, float(rrf_lookup.get((t, n), 0.0)), float(rerank_score)))
    return out


def filter_by_score_threshold_impl(indexer, hits, ratio=0.3):
    if not hits:
        return []
    max_score = max(h.score for h in hits)
    return [h for h in hits if h.score >= ratio * max_score]

def search_impl(
    indexer,
    query: str,
    top_k: int = 5,
    matched_tags: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str, float, float]]:
    """
    SOTA (production):
      - Filtros duros: user + (opcional) tags/client_tag + filters (documento, location, brand, etc.).
      - Recuperación híbrida (denso + SPLADE + (BM25 si no hay filtros)) → RRF.
      - Rerank (CrossEncoder + Late Interaction) + MMR.
      - Agregación Excel si detecta intención de conteo, guarda en indexer._last_aggregation.
    """
    import re

    def _is_aggregation_intent(q: str) -> bool:
        return bool(re.search(r"\b(cu[aá]nt[oa]s?|n[úu]mero de|total de|cuenta|how many|count)\b", q, flags=re.I))

    from qdrant_client.http import models as qmodels

    # --- MUST base: user ---
    must: List[qmodels.FieldCondition] = [
        qmodels.FieldCondition(key="user", match=qmodels.MatchValue(value=indexer.user_id))
    ]
    should: List[qmodels.FieldCondition] = []
    must_not: List[qmodels.FieldCondition] = []

    # --- tags / client_tag ---
    if matched_tags:
        # client_tag (legacy) -> si existe en payload
        try:
            must.append(qmodels.FieldCondition(key="client_tag", match=qmodels.MatchAny(any=matched_tags)))
        except Exception:
            for t in matched_tags:
                must.append(qmodels.FieldCondition(key="client_tag", match=qmodels.MatchValue(value=t)))

        # tags enriquecidos (payload)
        try:
            must.append(qmodels.FieldCondition(key="tags", match=qmodels.MatchAny(any=matched_tags)))
        except Exception:
            for t in matched_tags:
                must.append(qmodels.FieldCondition(key="tags", match=qmodels.MatchValue(value=t)))

    # --- filters dict (planner/tool) ---
    try:
        f_must, f_should, f_must_not = _build_qdrant_filter_from_filters(indexer, filters)
        must.extend(f_must)
        should.extend(f_should)
        must_not.extend(f_must_not)
    except Exception:
        # no romper por filtros mal formados
        pass

    qfilter = qmodels.Filter(
        must=must,
        should=should or None,
        must_not=must_not or None,
    )

    # --- búsqueda principal ---
    SEARCH_REQUESTS.labels(status='start').inc()
    with SEARCH_LATENCY.time():
        start = time.time()
        reranked = indexer.hybrid_search_rrf(query=query, top_k=top_k, qfilter=qfilter)
        latency = int((time.time() - start) * 1000)
        indexer.log_query(query, reranked, latency)

    SEARCH_REQUESTS.labels(status='success' if reranked else 'empty').inc()

    # --- Agregación (best-effort) ---
    indexer._last_aggregation = None
    try:
        if reranked and _is_aggregation_intent(query):
            hits_for_agg = []
            for (text, doc_name, _rrf, _rer) in reranked:
                try:
                    idx = indexer.documents.index(text)
                except ValueError:
                    idx = -1
                if idx >= 0:
                    try:
                        meta = indexer._metas[idx] if idx < len(indexer._metas) else {}
                    except Exception:
                        meta = {}
                    hits_for_agg.append({"text": text, "doc_name": doc_name, "meta": meta})

            if hits_for_agg:
                indexer._last_aggregation = indexer.aggregate_excel(query, hits_for_agg, top_k_tables=3)
    except Exception:
        indexer._last_aggregation = None

    return reranked

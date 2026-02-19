# rag_similarity.py
import logging
import os
from typing import List, Dict, Optional, Any

from qdrant_client.http import models as qmodels

# Logger coherente con el resto del proyecto
logger = logging.getLogger("cosmos_nlp_v3")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def _named_query_vector(embedding):
    dense_name = os.getenv("QDRANT_DENSE_NAME", "dense")
    vec = embedding.tolist() if hasattr(embedding, "tolist") else (list(embedding) if not isinstance(embedding, list) else embedding)
    try:
        return qmodels.NamedVector(name=dense_name, vector=vec)
    except Exception:
        return vec

def _tokenize_basic(text: str) -> List[str]:
    import re
    return [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ0-9_+-]{3,}", text or "")]

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

def retrieve_similar_blocks_impl(
    indexer,
    doc_name: str,
    relevant_idx: int,
    documents: List[str],
    top_k: int = 4,
    global_search: bool = True
) -> List[dict]:
    """
    Recupera bloques similares usando Qdrant y metadata enriquecida.

    Robustez:
    - Siempre filtra por user.
    - Usa doc_name como pista para restringir ancla/búsqueda al documento (file_name).
    - Si global_search=True y faltan resultados, hace una segunda búsqueda sin restricción de doc.
    """
    import os

    if relevant_idx < 0 or relevant_idx >= len(documents):
        return []

    try:
        top_k = max(1, int(top_k or 1))
    except Exception:
        top_k = 4

    reference_text = documents[relevant_idx] or ""
    if not str(reference_text).strip():
        return []

    try:
        ref_emb = indexer._embed_texts([reference_text], is_query=False)[0]
    except Exception as e:
        logger.warning(f"(retrieve_similar_blocks) Error generando embedding: {e}")
        return []

    # normalizar doc_name -> file base (por si viene con ::)
    dn = str(doc_name or "")
    dn_base = os.path.basename(dn.split("::")[0]) if dn else ""
    dn_base = dn_base.strip()

    # vector nombrado
    qv = _named_query_vector(ref_emb)

    # filtro base (siempre user)
    base_must = [
        qmodels.FieldCondition(key="user", match=qmodels.MatchValue(value=indexer.user_id))
    ]

    # Anchor search (preferir en el mismo file_name si lo tenemos)
    anchor_filter = None
    if dn_base:
        anchor_filter = _merge_filters(
            base_must + [qmodels.FieldCondition(key="file_name", match=qmodels.MatchValue(value=dn_base))],
            None
        )
    else:
        anchor_filter = _merge_filters(base_must, None)

    try:
        hits = indexer.qdrant_client.search(
            collection_name=indexer.collection_name,
            query_vector=qv,
            limit=6,
            with_vectors=False,
            with_payload=True,
            query_filter=anchor_filter,
        )
        anchor = hits[0] if hits else None
    except Exception as e:
        logger.warning(f"(retrieve_similar_blocks) Búsqueda anchor falló: {e}")
        return []

    if not anchor or not getattr(anchor, "payload", None):
        return []

    p = anchor.payload or {}
    doc_id = p.get("doc_id")
    ancestors = p.get("section_ancestors") or []
    parent_section = ancestors[-2] if isinstance(ancestors, list) and len(ancestors) >= 2 else None

    # filtros para similares locales
    must_local = list(base_must)

    # preferimos doc_id si existe, si no, file_name por doc_name
    if doc_id:
        must_local.append(qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)))
    elif dn_base:
        must_local.append(qmodels.FieldCondition(key="file_name", match=qmodels.MatchValue(value=dn_base)))

    if parent_section:
        must_local.append(qmodels.FieldCondition(
            key="section_ancestors",
            match=qmodels.MatchValue(value=parent_section)
        ))

    local_filter = qmodels.Filter(must=must_local)

    def _meta_from_payload(pld: Dict[str, Any]) -> Dict[str, Any]:
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
        return {k: pld.get(k) for k in meta_fields if pld.get(k) is not None}

    similars: List[dict] = []
    seen_ids = set([getattr(anchor, "id", None)])

    # 1) Búsqueda local
    try:
        local_hits = indexer.qdrant_client.search(
            collection_name=indexer.collection_name,
            query_vector=qv,
            limit=top_k + 8,
            with_vectors=False,
            with_payload=True,
            query_filter=local_filter,
        )
    except Exception as e:
        logger.warning(f"(retrieve_similar_blocks) Búsqueda local falló: {e}")
        local_hits = []

    for h in local_hits or []:
        hid = getattr(h, "id", None)
        if hid in seen_ids:
            continue
        seen_ids.add(hid)

        lp = getattr(h, "payload", None) or {}
        lt = lp.get("text") or lp.get("chunk") or lp.get("chunk_preview") or ""
        if not isinstance(lt, str):
            lt = str(lt or "")
        lt = lt.strip()

        if not lt or lt == reference_text:
            continue

        similars.append({
            "text": lt,
            "score": float(getattr(h, "score", 0.0) or 0.0),
            "meta": _meta_from_payload(lp),
        })
        if len(similars) >= top_k:
            break

    # 2) Fallback global (si se pidió)
    if global_search and len(similars) < top_k:
        try:
            global_filter = qmodels.Filter(must=base_must)
            ghits = indexer.qdrant_client.search(
                collection_name=indexer.collection_name,
                query_vector=qv,
                limit=(top_k - len(similars)) + 16,
                with_vectors=False,
                with_payload=True,
                query_filter=global_filter,
            )
        except Exception:
            ghits = []

        for h in ghits or []:
            hid = getattr(h, "id", None)
            if hid in seen_ids:
                continue
            seen_ids.add(hid)

            lp = getattr(h, "payload", None) or {}
            lt = lp.get("text") or lp.get("chunk") or lp.get("chunk_preview") or ""
            if not isinstance(lt, str):
                lt = str(lt or "")
            lt = lt.strip()
            if not lt or lt == reference_text:
                continue

            similars.append({
                "text": lt,
                "score": float(getattr(h, "score", 0.0) or 0.0),
                "meta": _meta_from_payload(lp),
            })
            if len(similars) >= top_k:
                break

    return similars[:top_k]


def search_with_similar_blocks_impl(
    indexer,
    query: str,
    top_k_main: int = 5,
    top_k_similars: int = 4,
    matched_tags: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Búsqueda vectorial principal (E5 'query') con filtro por user y opcionalmente:
    - matched_tags: OR entre payload.tags y payload.client_tag
    - filters.documento/file_name/doc_name: restringe por file_name (si es resoluble)

    Devuelve objetos con:
      - text, doc_name, score
      - meta (enriquecida)
      - similar_blocks
    """
    import os
    from qdrant_client.http import models as qmodels

    if not query or not str(query).strip():
        return []

    try:
        top_k_main = max(1, int(top_k_main or 1))
    except Exception:
        top_k_main = 5
    try:
        top_k_similars = max(0, int(top_k_similars or 0))
    except Exception:
        top_k_similars = 4

    # Embed del query
    try:
        q_emb = indexer._embed_texts([query], is_query=True)[0]
    except Exception as e:
        logger.error(f"(search_with_similar_blocks) Error generando embedding de query: {e}")
        return []

    qv = _named_query_vector(q_emb)

    # ----------------------------
    # Filtros
    # ----------------------------
    must = [
        qmodels.FieldCondition(key="user", match=qmodels.MatchValue(value=indexer.user_id))
    ]
    should = []

    # matched_tags -> OR entre tags y client_tag
    if matched_tags:
        for t in matched_tags[:32]:
            if not isinstance(t, str) or not t.strip():
                continue
            tt = t.strip()
            should.append(qmodels.FieldCondition(key="tags", match=qmodels.MatchValue(value=tt)))
            should.append(qmodels.FieldCondition(key="client_tag", match=qmodels.MatchValue(value=tt)))

    # filters de documento (si vienen)
    doc_hint = None
    if isinstance(filters, dict):
        doc_hint = filters.get("documento") or filters.get("file_name") or filters.get("doc_name")

    resolved = None
    if isinstance(doc_hint, str) and doc_hint.strip():
        hint = doc_hint.strip()
        # si existe resolved en el indexer, úsalo (mejor que match literal)
        try:
            if hasattr(indexer, "resolve_document_name") and callable(indexer.resolve_document_name):
                resolved = indexer.resolve_document_name(hint)
        except Exception:
            resolved = None
        target = resolved or os.path.basename(hint.split("::")[0])
        if target:
            must.append(qmodels.FieldCondition(key="file_name", match=qmodels.MatchValue(value=target)))

    qfilter = qmodels.Filter(must=must, should=(should or None))

    # ----------------------------
    # Búsqueda principal
    # ----------------------------
    try:
        hits = indexer.qdrant_client.search(
            collection_name=indexer.collection_name,
            query_vector=qv,
            limit=max(20, top_k_main),
            with_vectors=False,
            with_payload=True,
            query_filter=qfilter,
        )
    except Exception as e:
        logger.error(f"(search_with_similar_blocks) Error en búsqueda Qdrant: {e}")
        return []

    # Rerank ligero por overlap léxico
    q_terms = set(_tokenize_basic(query))

    def _overlap_score(text: str) -> float:
        if not text:
            return 0.0
        t_terms = set(_tokenize_basic(text))
        if not t_terms:
            return 0.0
        inter = len(q_terms & t_terms)
        return inter / max(1, len(q_terms))

    rescored = []
    for h in hits or []:
        p = getattr(h, "payload", None) or {}
        t = p.get("text") or p.get("chunk") or p.get("chunk_preview") or ""
        if not isinstance(t, str):
            t = str(t or "")
        sim = float(getattr(h, "score", 0.0) or 0.0)
        lex = _overlap_score(t)
        score = 0.85 * sim + 0.15 * lex
        rescored.append((score, h))

    rescored.sort(key=lambda x: x[0], reverse=True)
    main_hits = [h for _, h in rescored[:top_k_main]]

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

    results: List[Dict[str, Any]] = []

    for h in main_hits:
        p = getattr(h, "payload", None) or {}
        main_text = p.get("text") or p.get("chunk") or p.get("chunk_preview") or ""
        if not isinstance(main_text, str):
            main_text = str(main_text or "")
        main_text = main_text.strip()

        main_doc = p.get("file_name") or p.get("doc_name") or p.get("source_path") or "desconocido"
        if not isinstance(main_doc, str):
            main_doc = str(main_doc or "")

        main_score = float(getattr(h, "score", 0.0) or 0.0)

        meta_main = {k: p.get(k) for k in meta_fields if p.get(k) is not None}

        similars: List[Dict[str, Any]] = []
        if top_k_similars > 0:
            try:
                doc_id = p.get("doc_id")
                ancestors = p.get("section_ancestors") or []
                parent_section = ancestors[-2] if isinstance(ancestors, list) and len(ancestors) >= 2 else None

                must_sim = [
                    qmodels.FieldCondition(key="user", match=qmodels.MatchValue(value=indexer.user_id))
                ]
                if doc_id:
                    must_sim.append(qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)))
                if parent_section:
                    must_sim.append(qmodels.FieldCondition(key="section_ancestors", match=qmodels.MatchValue(value=parent_section)))

                local_hits = indexer.qdrant_client.search(
                    collection_name=indexer.collection_name,
                    query_vector=qv,
                    limit=top_k_similars + 6,
                    with_vectors=False,
                    with_payload=True,
                    query_filter=qmodels.Filter(must=must_sim),
                )

                seen = {getattr(h, "id", None)}
                for lh in local_hits or []:
                    if getattr(lh, "id", None) in seen:
                        continue
                    seen.add(getattr(lh, "id", None))

                    lp = getattr(lh, "payload", None) or {}
                    lt = lp.get("text") or lp.get("chunk") or lp.get("chunk_preview") or ""
                    if not isinstance(lt, str):
                        lt = str(lt or "")
                    lt = lt.strip()
                    if not lt:
                        continue

                    meta_sim = {k: lp.get(k) for k in meta_fields if lp.get(k) is not None}
                    similars.append({
                        "text": lt,
                        "score": float(getattr(lh, "score", 0.0) or 0.0),
                        "meta": meta_sim,
                    })
                    if len(similars) >= top_k_similars:
                        break

                similars.sort(key=lambda x: x["score"], reverse=True)
                similars = similars[:top_k_similars]

            except Exception as e:
                logger.debug(f"(search_with_similar_blocks) Expansión similar falló: {e}")

        results.append({
            "text": main_text,
            "doc_name": main_doc,
            "score": main_score,
            "meta": meta_main,
            "similar_blocks": similars,
        })

    return results


# cosmos_mcp_app/tools.py

from __future__ import annotations

import os
import re
import asyncio
import logging
from typing import Optional, Dict, Any, Union, List

import httpx
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from app.core import settings
from models.inference import ChatCompletionRequest, ChatMessage
from services.llm_client import LLMClient
from services.tools_registry import cosmos_tool
from .server import mcp_server, MCPAppContext
from .utils_tools import build_nlp_auth

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuración base del microservicio NLP
# ---------------------------------------------------------------------

NLP_APP_BASE_URL: str = getattr(settings, "NLP_APP_BASE_URL", "http://127.0.0.1:5000")
NLP_REQUEST_TIMEOUT: float = float(getattr(settings, "NLP_REQUEST_TIMEOUT", 180.0))


# ---------------------------------------------------------------------
# Helpers comunes
# ---------------------------------------------------------------------

def _clean_llm_text(text: Optional[str]) -> str:
    """
    Limpia texto generado por el LLM:
    - recorta espacios
    - quita fences ``` y etiquetas ```json
    - elimina disclaimers típicos
    - colapsa saltos de línea redundantes
    """
    if not text:
        return ""

    t = str(text).strip()

    # Quitar fences ```lang ... ```
    if t.startswith("```") and t.endswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_\-]*\s*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)

    # Quitar 'Respuesta:' / 'Assistant:' al inicio
    t = re.sub(
        r"^(respuesta:|respuesta\s*|assistant:|asistente:)\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )

    # Eliminar disclaimers típicos
    disclaimer_patterns = [
        r"como modelo de lenguaje[^.\n]*\.",
        r"como ia(?: de)? lenguaje[^.\n]*\.",
    ]
    for pat in disclaimer_patterns:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)

    # Colapsar más de 2 saltos de línea
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def _maybe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _maybe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _norm_str(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    s = x.strip()
    return s if s else None

def _extract_document_hint_from_query(query: str) -> Optional[str]:
    """
    Extrae un posible nombre de archivo de la query (pdf/docx/xlsx/pptx/txt/csv).
    Devuelve el literal encontrado o None.
    """
    if not query:
        return None
    q = str(query).strip()

    m = re.search(
        r'([^\s"\'<>]+?\.(?:pdf|docx|doc|pptx|ppt|xlsx|xls|csv|txt))\b',
        q,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def _merge_filters(base: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge defensivo (extra pisa base). Limpia claves vacías.
    """
    out: Dict[str, Any] = {}
    if isinstance(base, dict):
        out.update(base)
    if isinstance(extra, dict):
        out.update(extra)

    clean: Dict[str, Any] = {}
    for k, v in out.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        clean[k] = v
    return clean

# ---------------------------------------------------------------------
# Tool: llama_chat_tool
# ---------------------------------------------------------------------

@cosmos_tool(
    name="llama_chat_tool",
    description=(
        "Llama al servidor llama.cpp con un prompt plano y devuelve la primera "
        "respuesta de texto ya limpiada."
    ),
    tags=["llm", "generic"],
    for_crews=["business_crew", "document_crew"],
)
@mcp_server.tool(
    name="llama_chat_tool",
    description="Llama a llama.cpp con prompt plano y devuelve texto limpio.",
)
async def llama_chat_tool(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    ctx: Optional[Context[ServerSession, MCPAppContext]] = None,
) -> str:
    if ctx is None:
        raise RuntimeError("Contexto MCP no disponible")

    llm_client: LLMClient = ctx.request_context.lifespan_context.llm_client

    request = ChatCompletionRequest(
        model=model or settings.LLAMA_MODEL,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    raw = await llm_client.chat_completion(request)
    choices = raw.get("choices", [])
    if not choices:
        return "No se recibió ninguna choice desde llama.cpp."

    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        return "Respuesta de llama.cpp sin 'message.content' válido."

    return _clean_llm_text(content)


# ---------------------------------------------------------------------
# Tool: rag_search_tool (negocio)
# ---------------------------------------------------------------------
@cosmos_tool(
    name="rag_search_tool",
    description=(
        "Envuelve al microservicio NLP (/search) para realizar búsquedas RAG "
        "sobre documentos corporativos. Soporta filtros (p.ej. documento/ubicación/empresa)."
    ),
    tags=["rag", "search", "business"],
    for_crews=["business_crew"],
)
@mcp_server.tool(
    name="rag_search_tool",
    description="Envuelve NLP /search para búsquedas RAG corporativas (con filtros).",
)
async def rag_search_tool(
    query: str,
    flow: str = "C",
    top_k: Optional[int] = None,

    # ✅ NUEVO: filtros directos (planner -> tool)
    filters: Optional[Dict[str, Any]] = None,

    # ✅ NUEVO: plan completo (compat) para extraer plan["filters"]
    plan: Optional[Dict[str, Any]] = None,

    # ✅ NUEVO: atajo explícito para documento
    documento: Optional[str] = None,

    access_token: Optional[str] = None,
    ctx: Optional[Context[ServerSession, MCPAppContext]] = None,
) -> Dict[str, Any]:
    flow_norm = (flow or "C").upper().strip() or "C"

    # 1) Normalizar filtros desde plan.filters + filters + documento + inferencia en query
    plan_filters = None
    if isinstance(plan, dict):
        pf = plan.get("filters")
        if isinstance(pf, dict):
            plan_filters = pf

    explicit_filters: Dict[str, Any] = {}
    if isinstance(documento, str) and documento.strip():
        explicit_filters["documento"] = documento.strip()

    inferred_doc = None
    if not explicit_filters.get("documento"):
        inferred_doc = _extract_document_hint_from_query(query)

    inferred_filters: Dict[str, Any] = {}
    if inferred_doc:
        inferred_filters["documento"] = inferred_doc

    final_filters = _merge_filters(plan_filters, filters)
    final_filters = _merge_filters(final_filters, explicit_filters)
    final_filters = _merge_filters(final_filters, inferred_filters)

    # 2) Payload al microservicio NLP
    payload: Dict[str, Any] = {"query": query}

    if flow_norm == "R":
        payload["top_k"] = top_k if top_k is not None else 3
    else:
        if top_k is not None:
            payload["top_k"] = top_k

    if final_filters:
        payload["filters"] = final_filters

    cookies, headers = await build_nlp_auth(access_token=access_token)

    async with httpx.AsyncClient(timeout=NLP_REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{NLP_APP_BASE_URL.rstrip('/')}/search",
            json=payload,
            cookies=cookies or None,
            headers=headers or None,
        )

    resp.raise_for_status()
    data = resp.json() or {}

    # 3) Respuesta uniforme
    return {
        "status": data.get("status") or "ok",
        "results": data.get("results") or [],
        "aggregation": data.get("aggregation"),
        "used_department": data.get("used_department"),
        "cleaned_query": data.get("cleaned_query") or query,
        "filters_applied": final_filters or None,
    }


# ---------------------------------------------------------------------
# Tool: chatdoc_ingest_tool (document_crew)
# ---------------------------------------------------------------------

@cosmos_tool(
    name="chatdoc_ingest_tool",
    description=(
        "Envuelve /chatdoc/ingest del servicio NLP para construir o reemplazar "
        "un índice efímero de chat para un documento."
    ),
    tags=["rag", "document", "chatdoc", "ingest"],
    for_crews=["document_crew"],
)
@mcp_server.tool(
    name="chatdoc_ingest_tool",
    description="Construye/reemplaza índice efímero para chat con un documento.",
)
async def chatdoc_ingest_tool(
    content_base64: Optional[str] = None,
    text: Optional[str] = None,
    document_id: Optional[str] = None,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    page_count: Optional[int] = None,
    access_token: Optional[str] = None,
    user_id: Optional[int] = None,  # compat
    ctx: Optional[Context[ServerSession, MCPAppContext]] = None,
) -> Dict[str, Any]:
    if not content_base64 and not text:
        return {"status": "error", "error": "Debe proporcionar 'text' o 'content_base64'."}

    payload: Dict[str, Any] = {"metadata": metadata or {}}

    if content_base64:
        payload["content_base64"] = content_base64
    if text:
        payload["text"] = text
    if document_id:
        payload["document_id"] = document_id
    if filename:
        payload["filename"] = filename
    if mime_type:
        payload["mime_type"] = mime_type
    if page_count is not None:
        payload["page_count"] = int(page_count)

    # compat heredada
    if user_id is not None:
        payload["user_id"] = int(user_id)

    cookies, headers = await build_nlp_auth(access_token=access_token)

    logger.info(
        "[chatdoc_ingest_tool] → NLP /chatdoc/ingest filename=%s doc_id=%s mime=%s has_b64=%s has_text=%s",
        filename, document_id, mime_type, bool(content_base64), bool(text),
    )

    async with httpx.AsyncClient(timeout=NLP_REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{NLP_APP_BASE_URL.rstrip('/')}/chatdoc/ingest",
            json=payload,
            cookies=cookies or None,
            headers=headers or None,
        )

    resp.raise_for_status()
    data = resp.json() or {}
    return data


# ---------------------------------------------------------------------
# Tool: chatdoc_query_tool (document_crew)
# ---------------------------------------------------------------------
@cosmos_tool(
    name="chatdoc_query_tool",
    description="Envuelve /chatdoc/query del servicio NLP para recuperar fragmentos relevantes.",
    tags=["rag", "document", "chatdoc", "query"],
    for_crews=["document_crew"],
)
@mcp_server.tool(
    name="chatdoc_query_tool",
    description="Recupera fragmentos relevantes usando el índice efímero ChatDoc.",
)
async def chatdoc_query_tool(
    document_id: str,
    query: str,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    window: Optional[int] = None,
    return_format: str = "text",
    access_token: Optional[str] = None,
    user_id: Optional[int] = None,
    ctx: Optional[Context[ServerSession, MCPAppContext]] = None,
) -> Union[str, Dict[str, Any]]:

    fmt = (return_format or "text").lower().strip()
    if fmt not in ("text", "json", "trace_json"):
        fmt = "text"

    payload: Dict[str, Any] = {
        "document_id": document_id,
        "query": query,
    }

    if top_k is not None:
        try:
            payload["top_k"] = int(top_k)
        except Exception:
            pass

    if min_score is not None:
        try:
            payload["min_score"] = float(min_score)
        except Exception:
            pass

    if window is not None:
        try:
            payload["window"] = int(window)
        except Exception:
            pass

    if user_id is not None:
        payload["user_id"] = int(user_id)

    cookies, headers = await build_nlp_auth(access_token=access_token)

    logger.info(
        "[chatdoc_query_tool] → NLP /chatdoc/query doc_id=%s top_k=%s min_score=%s window=%s fmt=%s",
        document_id, top_k, min_score, window, fmt
    )

    async with httpx.AsyncClient(timeout=NLP_REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{NLP_APP_BASE_URL.rstrip('/')}/chatdoc/query",
            json=payload,
            cookies=cookies or None,
            headers=headers or None,
        )

    resp.raise_for_status()
    data = resp.json() or {}
    results = data.get("results") or []

    # logging útil
    logger.info(
        "[chatdoc_query_tool] ← NLP status=%s results=%d",
        data.get("status"), len(results) if isinstance(results, list) else -1
    )

    # alias compat
    if isinstance(results, list) and results:
        data.setdefault("fragments", results)
        data.setdefault("chunks", results)

    # construir texto amigable
    text_block = ""
    if isinstance(results, list) and results:
        pieces = []
        for r in results:
            if isinstance(r, dict):
                t = (r.get("text") or "").strip()
                if t:
                    pieces.append(t)
        text_block = "\n\n---\n\n".join(pieces)

    if fmt == "text":
        return text_block or "Sin resultados para la consulta."

    if fmt == "trace_json":
        data.setdefault("trace", {})
        data["trace"].update({"return_format": fmt})

    return data


@cosmos_tool(
    name="chatdoc_dual_query_tool",
    description=(
        "Ejecuta dos recuperaciones contra /chatdoc/query (precision+coverage), "
        "fusiona resultados y devuelve trazabilidad por pasada."
    ),
    tags=["rag", "document", "chatdoc", "query", "dual", "production"],
    for_crews=["document_crew"],
)
@mcp_server.tool(
    name="chatdoc_dual_query_tool",
    description="Dual retrieval robusto para ChatDoc con dos pasadas paralelas.",
)
async def chatdoc_dual_query_tool(
    document_id: str,
    query: str,
    top_k: int = 8,
    min_score: float = 0.0,
    window: int = 1,
    return_format: str = "json",
    access_token: Optional[str] = None,
    user_id: Optional[int] = None,
    ctx: Optional[Context[ServerSession, MCPAppContext]] = None,
) -> Union[str, Dict[str, Any]]:
    """Dual retrieval con sesgo precision/coverage para latencia estable y mejor recall."""
    fmt = (return_format or "json").lower().strip()
    if fmt not in ("text", "json", "trace_json"):
        fmt = "json"

    base_top_k = max(1, min(int(top_k), 32))
    base_window = max(0, min(int(window), 5))
    base_min_score = float(min_score)

    precision_payload: Dict[str, Any] = {
        "document_id": document_id,
        "query": query,
        "top_k": base_top_k,
        "min_score": max(base_min_score, 0.12),
        "window": min(base_window, 2),
    }
    coverage_payload: Dict[str, Any] = {
        "document_id": document_id,
        "query": query,
        "top_k": min(32, base_top_k + 4),
        "min_score": max(0.0, base_min_score * 0.7),
        "window": min(5, base_window + 1),
    }

    if user_id is not None:
        precision_payload["user_id"] = int(user_id)
        coverage_payload["user_id"] = int(user_id)

    cookies, headers = await build_nlp_auth(access_token=access_token)
    endpoint = f"{NLP_APP_BASE_URL.rstrip('/')}/chatdoc/query"

    async with httpx.AsyncClient(timeout=NLP_REQUEST_TIMEOUT) as client:
        precision_task = client.post(endpoint, json=precision_payload, cookies=cookies or None, headers=headers or None)
        coverage_task = client.post(endpoint, json=coverage_payload, cookies=cookies or None, headers=headers or None)
        precision_resp, coverage_resp = await asyncio.gather(precision_task, coverage_task)

    precision_resp.raise_for_status()
    coverage_resp.raise_for_status()

    precision_data = precision_resp.json() or {}
    coverage_data = coverage_resp.json() or {}

    precision_results = precision_data.get("results") if isinstance(precision_data.get("results"), list) else []
    coverage_results = coverage_data.get("results") if isinstance(coverage_data.get("results"), list) else []

    merged: List[Dict[str, Any]] = []
    seen = set()
    for pass_name, items in (("precision", precision_results), ("coverage", coverage_results)):
        for item in items:
            if not isinstance(item, dict):
                continue
            txt = (item.get("text") or "").strip()
            if not txt:
                continue
            key = txt[:800]
            if key in seen:
                continue
            seen.add(key)
            enriched = dict(item)
            enriched["retrieval_pass"] = pass_name
            merged.append(enriched)

    merged.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    payload_out: Dict[str, Any] = {
        "status": "ok" if merged else "no_results",
        "document_id": document_id,
        "query": query,
        "results": merged[: min(48, len(merged))],
        "precision_count": len(precision_results),
        "coverage_count": len(coverage_results),
    }
    payload_out["chunks"] = payload_out["results"]
    payload_out["fragments"] = payload_out["results"]

    if fmt == "text":
        texts = [str(x.get("text") or "").strip() for x in payload_out["results"] if str(x.get("text") or "").strip()]
        return "\n\n---\n\n".join(texts) or "Sin resultados para la consulta."

    if fmt == "trace_json":
        payload_out["trace"] = {
            "precision_payload": precision_payload,
            "coverage_payload": coverage_payload,
            "return_format": fmt,
        }

    return payload_out


# ---------------------------------------------------------------------
# Tool: chatdoc_summary_tool (document_crew)
# ---------------------------------------------------------------------
@cosmos_tool(
    name="chatdoc_summary_tool",
    description="Envuelve /chatdoc/summary del servicio NLP para seleccionar fragmentos representativos.",
    tags=["rag", "document", "chatdoc", "summary"],
    for_crews=["document_crew"],
)
@mcp_server.tool(
    name="chatdoc_summary_tool",
    description="Selecciona fragmentos representativos para resumen ChatDoc.",
)
async def chatdoc_summary_tool(
    document_id: str,
    max_fragments: Optional[int] = None,
    strategy: Optional[str] = None,
    min_chars_per_chunk: Optional[int] = None,
    detail_level: Optional[str] = None,
    summary_profile: Optional[str] = None,
    trace: Optional[bool] = None,
    return_format: str = "text",
    access_token: Optional[str] = None,
    user_id: Optional[int] = None,
    ctx: Optional[Context[ServerSession, MCPAppContext]] = None,
) -> Union[str, Dict[str, Any]]:

    fmt = (return_format or "text").lower().strip()
    if fmt not in ("text", "json", "trace_json"):
        fmt = "text"

    payload: Dict[str, Any] = {"document_id": document_id}

    if max_fragments is not None:
        try:
            payload["max_fragments"] = int(max_fragments)
        except Exception:
            pass

    if min_chars_per_chunk is not None:
        try:
            payload["min_chars_per_chunk"] = int(min_chars_per_chunk)
        except Exception:
            pass

    if isinstance(strategy, str) and strategy.strip():
        payload["strategy"] = strategy.strip().lower()

    if isinstance(detail_level, str) and detail_level.strip():
        payload["detail_level"] = detail_level.strip().lower()

    if isinstance(summary_profile, str) and summary_profile.strip():
        payload["summary_profile"] = summary_profile.strip().lower()

    if trace is not None:
        payload["trace"] = bool(trace)

    if user_id is not None:
        payload["user_id"] = int(user_id)

    cookies, headers = await build_nlp_auth(access_token=access_token)

    logger.info(
        "[chatdoc_summary_tool] → NLP /chatdoc/summary doc_id=%s max_fragments=%s strategy=%s "
        "min_chars=%s detail_level=%s trace=%s fmt=%s",
        document_id, max_fragments, strategy, min_chars_per_chunk, detail_level,
        bool(trace) if trace is not None else None, fmt
    )

    async with httpx.AsyncClient(timeout=NLP_REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{NLP_APP_BASE_URL.rstrip('/')}/chatdoc/summary",
            json=payload,
            cookies=cookies or None,
            headers=headers or None,
        )

    resp.raise_for_status()
    data = resp.json() or {}

    fragments = data.get("fragments") or []
    if isinstance(fragments, list) and fragments:
        data.setdefault("results", fragments)
        data.setdefault("chunks", fragments)
    logger.info(f"\n\n\n[chatdoc_summary_tool]-DATA DEL RESUMEN DE CHAT_DOC: {data}")
    logger.info(
        "[chatdoc_summary_tool] ← NLP status=%s fragments=%d coverage=%s",
        data.get("status"),
        len(fragments) if isinstance(fragments, list) else -1,
        data.get("coverage"),
    )

    # texto preferente del endpoint
    pre = data.get("pre_summary_text")
    if isinstance(pre, str) and pre.strip():
        text_block = pre.strip()
    else:
        # fallback: concat textos de fragments
        pieces = []
        if isinstance(fragments, list):
            for f in fragments:
                if isinstance(f, dict):
                    t = (f.get("text") or "").strip()
                    if t:
                        pieces.append(t)
        text_block = "\n\n---\n\n".join(pieces)

    if fmt == "text":
        return text_block or "No se han podido seleccionar fragmentos para resumen."

    if fmt == "trace_json":
        data.setdefault("trace", {})
        data["trace"].update({"return_format": fmt})

    # meter también un campo text por si el consumidor lo usa
    data.setdefault("text", text_block)
    logger.info(f"\n\n\nDATA DEL RESUMEN DE CHAT_DOC: {data}")
    return data

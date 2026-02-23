"""
Microservicio `chat_document`: subir documentos (hasta 300 páginas) y
mantener una conversación tipo chat sobre su contenido.

- Protegido con el mismo sistema de autenticación/CSRF que modelo_negocio.
- Usa Redis para sesiones de documento e historial.
- Se apoya en un microservicio NLP externo para:
    * Ingestar/indexar documentos.
    * Recuperar fragmentos relevantes o muestras para resúmenes.
- Usa CrewAI con dos agentes:
    * Analista de documento.
    * Redactor de respuesta.
"""
from functools import partial
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Request,
    HTTPException,
    Depends,
    BackgroundTasks,
    Response
)
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, TypeVar
from mimetypes import guess_type
from datetime import datetime, timezone
import os
import time
import json
import uuid
import logging
import asyncio
import base64
import redis.asyncio as aioredis
import httpx
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
import anyio

from dotenv import load_dotenv
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from utils import REQUEST_COUNT, HTTP_REQUEST_DURATION  # noqa: E402
from mcp_client import MCPClient  # cliente MCP de este microservicio
from config.database import get_db
from config.models import (
    Conversation,
    Message,
    File as FileModel,
    UsageLog,
    AuditLog,
)
from sqlalchemy.orm import Session

from utils import (
    STORAGE_ROOT,
    ROLE_SUPERVISOR,
    ROLE_USER,
    generate_csrf_token,
    validate_csrf_double_submit,
    scan_with_clamav,
    allowed_file,
    clean_text,
    init_redis_clients,
    verify_token,
    extract_raw_bearer_token,
    get_session_from_redis,
    get_current_auth_chatdoc,
    _pick_doc_tool
)
from document_crew_orchestrator import DocumentCrewOrchestrator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat_document")

LOGS_ROOT = Path(
    os.getenv("COSMOS_CHATDOC_LOGS_DIR", str(BASE_DIR / "logs"))
).resolve()

# Configuración de URLs del microservicio NLP
NLP_CHATDOC_INGEST_URL = os.getenv(
    "NLP_CHATDOC_INGEST_URL",
    "http://cosmos.cosgs.int:5000/chatdoc/ingest",
).rstrip("/")

NLP_CHATDOC_QUERY_URL = os.getenv(
    "NLP_CHATDOC_QUERY_URL",
    "http://cosmos.cosgs.int:5000/chatdoc/query",
).rstrip("/")

NLP_CHATDOC_SUMMARY_URL = os.getenv(
    "NLP_CHATDOC_SUMMARY_URL",
    "http://cosmos.cosgs.int:5000/chatdoc/summary",
).rstrip("/")


# Límite defensivo de páginas
CHATDOC_MAX_PAGES = int(os.getenv("CHATDOC_MAX_PAGES", "300"))
# Límite defensivo de tamaño de archivo (MB)
CHATDOC_MAX_FILE_MB = int(os.getenv("CHATDOC_MAX_FILE_MB", "50"))
CHATDOC_MAX_FILE_BYTES = CHATDOC_MAX_FILE_MB * 1024 * 1024

# CSRF
CSRF_COOKIE_NAME = "csrftoken_chatdoc"
AUTH_ORIGIN = os.getenv("COSMOS_AUTH_ORIGIN", "http://cosmos.cosgs.int:7000")
FRONTEND_URL = os.getenv("COSMOS_FRONTEND_URL", "http://cosmos.cosgs.int").rstrip("/")

# Prometheus metrics específicas de este microservicio
DOC_UPLOAD_REQUESTS = Counter(
    "chatdoc_upload_requests_total",
    "Número total de peticiones de subida de documentos al servicio chat_document.",
)
DOC_UPLOAD_ERRORS = Counter(
    "chatdoc_upload_errors_total",
    "Número total de errores en subida de documentos al servicio chat_document.",
)
DOC_UPLOAD_DURATION = Histogram(
    "chatdoc_upload_duration_seconds",
    "Tiempo que tardan las peticiones de subida de documentos.",
)

DOC_CHAT_REQUESTS = Counter(
    "chatdoc_query_requests_total",
    "Número total de peticiones de chat con documento.",
)
DOC_CHAT_ERRORS = Counter(
    "chatdoc_query_errors_total",
    "Número total de errores en peticiones de chat con documento.",
)
DOC_CHAT_DURATION = Histogram(
    "chatdoc_query_duration_seconds",
    "Tiempo que tardan las peticiones de chat con documento.",
)


mcp_client = MCPClient()

# Redis global (se inicializa en startup)
redis_core_client: Optional[aioredis.Redis] = None
redis_conv_client: Optional[aioredis.Redis] = None

CHATDOC_SESSION_TTL = int(os.getenv("CHATDOC_SESSION_TTL", "3600"))
DOC_HISTORY_MAX_LEN = int(os.getenv("CHATDOC_HISTORY_MAX_LEN", "20"))

DOC_SESSION_KEY_PREFIX = "chatdoc:session"
DOC_HISTORY_KEY_PREFIX = "chatdoc:history"

CHATDOC_CREW_THREAD_LIMIT = max(1, int(os.getenv("CHATDOC_CREW_THREAD_LIMIT", "8") or "8"))
CHATDOC_PLANNER_THREAD_LIMIT = max(1, int(os.getenv("CHATDOC_PLANNER_THREAD_LIMIT", "8") or "8"))

CHATDOC_CREW_LIMITER = anyio.CapacityLimiter(CHATDOC_CREW_THREAD_LIMIT)
CHATDOC_PLANNER_LIMITER = anyio.CapacityLimiter(CHATDOC_PLANNER_THREAD_LIMIT)

T = TypeVar("T")


def _compute_ingest_timeout_s(content_size_bytes: int) -> float:
    """Calcula timeout adaptativo para ingesta según tamaño de archivo."""
    try:
        base_timeout = float(os.getenv("CHATDOC_NLP_INGEST_TIMEOUT_S", "600"))
    except Exception:
        base_timeout = 600.0

    try:
        max_timeout = float(os.getenv("CHATDOC_NLP_INGEST_TIMEOUT_MAX_S", "1800"))
    except Exception:
        max_timeout = 1800.0

    mb = max(1.0, float(content_size_bytes) / (1024.0 * 1024.0))
    adaptive_timeout = base_timeout + (mb * 15.0)
    return max(60.0, min(max_timeout, adaptive_timeout))

async def run_sync_kwargs(
    func: Callable[..., T],
    /,
    *args: Any,
    limiter: anyio.CapacityLimiter = CHATDOC_CREW_LIMITER,
    **kwargs: Any,
) -> T:
    """
    Ejecuta una función sync en un thread permitiendo kwargs.
    anyio.to_thread.run_sync NO acepta kwargs destinados a 'func', así que
    los bindemos con functools.partial.

    - limiter: limita concurrencia de threads (multiusuario, prod).
    """
    call = partial(func, *args, **kwargs)
    return await anyio.to_thread.run_sync(call, limiter=limiter)




app = FastAPI()
doc_crew_orchestrator = DocumentCrewOrchestrator()

# CORS (similar a modelo_negocio)
ALLOWED_ORIGINS_ENV = os.getenv("COSMOS_ALLOWED_ORIGINS", "")
if ALLOWED_ORIGINS_ENV:
    allowed_origins = [
        o.strip()
        for o in ALLOWED_ORIGINS_ENV.split(",")
        if o.strip()
    ]
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        AUTH_ORIGIN,
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class DocumentUploadResponse(BaseModel):
    doc_session_id: str
    document_id: str
    page_count: int
    file_name: str
    conversation_id: int


class DocumentQueryRequest(BaseModel):
    prompt: str
    doc_session_id: str
    conversation_id: Optional[int] = None
    mode: Optional[str] = None  # "qa" | "summary" | None


class DocumentSourceFragment(BaseModel):
    fragment_index: int
    page: Optional[int] = None
    score: Optional[float] = None


class DocumentQueryResponse(BaseModel):
    reply: str
    response: str
    conversation_id: int
    doc_session_id: str
    sources: List[DocumentSourceFragment] = Field(default_factory=list)



def validate_csrf(request: Request) -> None:
    """
    Wrapper del validador CSRF usando patrón double-submit cookie:
      - Cookie: csrftoken_chatdoc
      - Cabecera: X-CSRFToken
    """
    validate_csrf_double_submit(
        request,
        cookie_name=CSRF_COOKIE_NAME,
        header_name="X-CSRFToken",
        error_detail="CSRF token inválido o ausente en servicio chat_document.",
    )

@app.on_event("shutdown")
async def shutdown_redis():
    global redis_core_client, redis_conv_client
    try:
        if redis_core_client is not None:
            await redis_core_client.aclose()
    except Exception:
        pass
    try:
        if redis_conv_client is not None:
            await redis_conv_client.aclose()
    except Exception:
        pass

@app.get("/csrf-token", response_model=dict)
async def get_csrf_token_endpoint(request: Request, response: Response):
    """
    Devuelve un token CSRF específico para este microservicio y lo deja en cookie.

    - Reutiliza el token existente si ya hay cookie.
    - No construye Response manualmente (evitamos problemas de Content-Length).
    """
    token = request.cookies.get(CSRF_COOKIE_NAME)
    if not token:
        token = generate_csrf_token()

    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        httponly=False,   # el front React puede leerla
        secure=False,     # en prod: True + HTTPS
        samesite="Lax",
    )

    return {"csrf_token": token}

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path

    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    start = time.time()

    response = await call_next(request)

    duration = time.time() - start
    HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    return response


@app.on_event("startup")
async def initialize_redis():
    """
    Inicializa Redis para sesiones de documento e historial.
    """
    global redis_core_client, redis_conv_client

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    redis_core_client = aioredis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True,
    )
    redis_conv_client = aioredis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True,
        db=2,
    )

    init_redis_clients(redis_core_client, redis_conv_client)

    logger.info(
        "[startup-chatdoc] Redis conectado en %s:%s (db=0 y db=2) y registrado en utils.",
        redis_host,
        redis_port,
    )


async def store_doc_session(
    user_id: int,
    doc_session_id: str,
    payload: Dict[str, Any],
    ttl: Optional[int] = None,
) -> None:
    if redis_core_client is None:
        raise RuntimeError("Redis core client not initialized")
    key = f"{DOC_SESSION_KEY_PREFIX}:{user_id}:{doc_session_id}"
    await redis_core_client.set(
        key,
        json.dumps(payload, ensure_ascii=False),
        ex=ttl or CHATDOC_SESSION_TTL,
    )


async def get_doc_session(user_id: int, doc_session_id: str) -> Optional[Dict[str, Any]]:
    if redis_core_client is None:
        raise RuntimeError("Redis core client not initialized")
    key = f"{DOC_SESSION_KEY_PREFIX}:{user_id}:{doc_session_id}"
    raw = await redis_core_client.get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "No se pudo parsear doc_session en Redis (key=%s). Valor crudo truncado: %r",
            key,
            raw[:200],
        )
        return None


async def append_doc_history(
    user_id: int,
    doc_session_id: str,
    entry: Dict[str, Any],
    ttl: Optional[int] = None,
) -> None:
    if redis_conv_client is None:
        raise RuntimeError("Redis conv client not initialized")
    key = f"{DOC_HISTORY_KEY_PREFIX}:{user_id}:{doc_session_id}"
    raw = await redis_conv_client.get(key)
    if raw:
        try:
            history = json.loads(raw)
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []
    else:
        history = []
    history.append(entry)
    if len(history) > DOC_HISTORY_MAX_LEN:
        history = history[-DOC_HISTORY_MAX_LEN :]
    await redis_conv_client.set(
        key,
        json.dumps(history, ensure_ascii=False),
        ex=ttl or CHATDOC_SESSION_TTL,
    )


async def get_doc_history(user_id: int, doc_session_id: str) -> List[Dict[str, Any]]:
    if redis_conv_client is None:
        raise RuntimeError("Redis conv client not initialized")
    key = f"{DOC_HISTORY_KEY_PREFIX}:{user_id}:{doc_session_id}"
    raw = await redis_conv_client.get(key)
    if not raw:
        return []
    try:
        history = json.loads(raw)
        return history if isinstance(history, list) else []
    except json.JSONDecodeError:
        logger.warning(
            "No se pudo parsear doc_history en Redis (key=%s). Valor crudo truncado: %r",
            key,
            raw[:200],
        )
        return []


async def nlp_ingest_document(
    content: bytes,
    filename: str,
    mime_type: str,
    user_id: int,
    timeout_s: Optional[float] = None,
    access_token: Optional[str] = None,
    mcp_auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingesta de documento en el microservicio NLP para construir
    el índice efímero de chatdoc.

    Seguridad por capas:
    - Capa 1 (gateway MCP): Authorization HTTP con mcp_auth_token.
    - Capa 2 (tool -> NLP protegido): access_token dentro del payload para que
      la tool lo reenvíe al microservicio NLP.

    Feature flags:
    - CHATDOC_USE_MCP_INGEST=0/1 (default 0)
    - CHATDOC_NLP_INGEST_TIMEOUT_S (default 120)
    """

    # -------------------------------
    # Config
    # -------------------------------
    if timeout_s is not None and timeout_s > 0:
        timeout = timeout_s
    else:
        timeout = _compute_ingest_timeout_s(len(content or b""))

    use_mcp_ingest = os.getenv("CHATDOC_USE_MCP_INGEST", "0").strip() == "1"

    # -------------------------------
    # Normalización token usuario (para NLP HTTP directo)
    # -------------------------------
    headers: Dict[str, str] = {}
    normalized_user_token: Optional[str] = None

    if access_token:
        tok = access_token.strip()
        if tok.lower().startswith("bearer "):
            tok = tok[7:].strip()
        normalized_user_token = tok
        headers["Authorization"] = f"Bearer {tok}"

    # -------------------------------
    # Payload HTTP directo
    # -------------------------------
    payload_http: Dict[str, Any] = {
        "user_id": user_id,
        "filename": filename,
        "mime_type": mime_type,
        "content_base64": base64.b64encode(content).decode("utf-8"),
        "metadata": {
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": len(content or b""),
            "ingest_profile": "chat_document",
        },
    }

    logger.info(
        "[nlp_ingest_document] start user_id=%s filename=%s mime=%s bytes=%d use_mcp=%s timeout=%.1fs",
        user_id,
        filename,
        mime_type,
        len(content or b""),
        use_mcp_ingest,
        timeout,
    )

    # -------------------------------
    # Intento MCP (solo si habilitado)
    # -------------------------------
    if use_mcp_ingest:
        tool_name = _pick_doc_tool("ingest") if "_pick_doc_tool" in globals() else "chatdoc_ingest_tool"

        # ✅ Alineado con la firma real de la tool
        payload_mcp: Dict[str, Any] = {
            "filename": filename,
            "mime_type": mime_type,
            "content_base64": payload_http["content_base64"],
            "metadata": payload_http["metadata"],
            "access_token": normalized_user_token,
        }

        logger.info(
            "[chatdoc/api] Ingest MCP → tool=%s filename=%s mime=%s content_b64_len=%d",
            tool_name,
            filename,
            mime_type,
            len(payload_http["content_base64"] or ""),
        )

        try:
            data = await mcp_client.ainvoke_tool(
                tool_name,
                payload_mcp,
                auth_token=mcp_auth_token or access_token,  # capa gateway MCP
                timeout=timeout,
            )

            if isinstance(data, dict) and data.get("document_id"):
                logger.info(
                    "[nlp_ingest_document] Ingest MCP OK filename=%s doc_id=%s",
                    filename,
                    data.get("document_id"),
                )
                return data

            logger.warning(
                "[nlp_ingest_document] Ingest MCP respuesta incompleta. Fallback HTTP. type=%s keys=%s",
                type(data).__name__,
                list(data.keys()) if isinstance(data, dict) else None,
            )

        except Exception as exc:
            logger.warning(
                "[nlp_ingest_document] Fallo ingest vía MCP (%s). Fallback HTTP.",
                exc,
            )

    # -------------------------------
    # Ruta principal / fallback: HTTP directo a NLP
    # -------------------------------
    logger.info(
        "[nlp_ingest_document] Iniciando ingest HTTP → %s user_id=%s filename=%s timeout=%.1fs",
        NLP_CHATDOC_INGEST_URL,
        user_id,
        filename,
        timeout,
    )

    async def _post_ingest(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post(
            NLP_CHATDOC_INGEST_URL,
            headers=headers or None,
            json=payload_http,
        )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await _post_ingest(client)

            if resp.status_code == 504:
                retry_timeout = min(timeout * 1.5, timeout + 300.0)
                logger.warning(
                    "[nlp_ingest_document] NLP devolvió 504. Reintentando una vez con timeout=%.1fs filename=%s",
                    retry_timeout,
                    filename,
                )
                client.timeout = httpx.Timeout(retry_timeout)
                resp = await _post_ingest(client)
    except httpx.ReadTimeout:
        logger.error(
            "[nlp_ingest_document] Timeout HTTP NLP ingest url=%s filename=%s user_id=%s timeout=%.1fs",
            NLP_CHATDOC_INGEST_URL,
            filename,
            user_id,
            timeout,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "El servicio NLP tardó demasiado en responder al ingerir el archivo. "
                "El documento podría ser grande o el servicio estar saturado."
            ),
        )
    except httpx.RequestError as exc:
        logger.error(
            "[nlp_ingest_document] Error de red HTTP NLP ingest url=%s err=%s",
            NLP_CHATDOC_INGEST_URL,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail="No se pudo conectar con el servicio NLP de documentos.",
        )

    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        body_preview = resp.text[:500] if resp.text else ""
        logger.error(
            "[nlp_ingest_document] Error HTTP NLP ingest status=%s body=%r",
            resp.status_code,
            body_preview,
        )

        if resp.status_code in (401, 403):
            raise HTTPException(
                status_code=resp.status_code,
                detail="No autorizado por NLP. Posible sesión caducada.",
            )

        if 400 <= resp.status_code < 500:
            raise HTTPException(
                status_code=502,
                detail=(
                    "NLP rechazó la petición de ingest "
                    f"(status {resp.status_code}). Detalle: {body_preview}"
                ),
            )

        raise HTTPException(
            status_code=502,
            detail="NLP tuvo un error interno al ingerir el documento.",
        )

    try:
        data = resp.json()
    except Exception as e:
        logger.error(
            "[nlp_ingest_document] Respuesta no JSON NLP ingest filename=%s err=%s body=%r",
            filename,
            e,
            resp.text[:500] if resp.text else "",
        )
        raise HTTPException(
            status_code=502,
            detail="NLP devolvió una respuesta no válida al ingerir el documento.",
        )

    logger.info(
        "[nlp_ingest_document] Ingest HTTP OK filename=%s doc_id=%s keys=%s",
        filename,
        data.get("document_id") if isinstance(data, dict) else None,
        list(data.keys()) if isinstance(data, dict) else None,
    )
    return data



async def nlp_query_document(
    document_id: str,
    query: str,
    mode: str = "qa",
    top_k: int = 8,
    access_token: Optional[str] = None,
    detail_level: Optional[str] = None,
    summary_profile: Optional[str] = None,
    mcp_auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wrapper MCP de query/summary para ser usado por el ORQUESTADOR.

    Seguridad por capas:
      - auth del gateway MCP: Authorization HTTP usando mcp_auth_token (o access_token).
      - auth del NLP protegido: access_token dentro del payload.

    Devuelve estructura homogénea:
        {
          "status": str,
          "chunks": list[dict],
          "summary": str | None
        }
    """
    mode_norm = (mode or "qa").lower().strip()
    if mode_norm not in ("qa", "summary"):
        mode_norm = "qa"

    # Normalizamos Bearer para reenviar al NLP desde el tool
    tok: Optional[str] = None
    if access_token:
        t = access_token.strip()
        tok = t[7:].strip() if t.lower().startswith("bearer ") else t

    # Normalizamos detail_level
    if isinstance(detail_level, str):
        dl = detail_level.strip().lower()
        detail_level = dl if dl in ("low", "medium", "high") else None
    else:
        detail_level = None

    # Normalizamos summary_profile
    if isinstance(summary_profile, str):
        summary_profile = summary_profile.strip().lower()
    else:
        summary_profile = None

    # Timeouts defensivos
    try:
        timeout = float(os.getenv("CHATDOC_NLP_QUERY_TIMEOUT_S", "60"))
    except Exception:
        timeout = 60.0

    # -----------------------------
    # SUMMARY
    # -----------------------------
    if mode_norm == "summary":
        tool_name = _pick_doc_tool("summary")

        try:
            max_fragments = int(os.getenv("CHATDOC_SUMMARY_MAX_FRAGMENTS", "12"))
        except Exception:
            max_fragments = 12

        try:
            min_chars_per_chunk = int(os.getenv("CHATDOC_SUMMARY_MIN_CHARS", "300"))
        except Exception:
            min_chars_per_chunk = 300

        payload: Dict[str, Any] = {
            "document_id": document_id,
            "strategy": os.getenv("CHATDOC_SUMMARY_STRATEGY", "hybrid"),
            "max_fragments": max_fragments,
            "min_chars_per_chunk": min_chars_per_chunk,
            # token usuario para que el tool lo reenvíe al NLP protegido
            "access_token": tok,
        }
        if detail_level:
            payload["detail_level"] = detail_level
        if summary_profile:
            payload["summary_profile"] = summary_profile

        try:
            data = await mcp_client.ainvoke_tool(
                tool_name,
                payload,
                auth_token=mcp_auth_token or access_token,  # 👈 auth gateway MCP
                timeout=timeout,
            )
        except Exception as e:
            logger.exception("[chatdoc/api] Error MCP summary tool=%s: %s", tool_name, e)
            raise HTTPException(status_code=502, detail="Error llamando al tool MCP de resumen.")

        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="Respuesta inválida del tool MCP de resumen.")

        fragments = data.get("fragments") or []
        pre_summary_text = data.get("pre_summary_text") or ""

        return {
            "status": data.get("status") or ("ok" if fragments else "no_chunks"),
            "chunks": fragments if isinstance(fragments, list) else [],
            "summary": pre_summary_text or None,
        }

    # -----------------------------
    # QA
    # -----------------------------
    tool_name = _pick_doc_tool("query")

    # límites defensivos del API/orquestación
    try:
        max_top_k = int(os.getenv("CHATDOC_API_QUERY_MAX_TOPK", "16"))
    except Exception:
        max_top_k = 16
    top_k = max(1, min(int(top_k or 8), max_top_k))

    try:
        min_score = float(os.getenv("CHATDOC_QUERY_MIN_SCORE", "0.0"))
    except Exception:
        min_score = 0.0

    try:
        window = int(os.getenv("CHATDOC_SEARCH_WINDOW", "1"))
    except Exception:
        window = 1

    payload = {
        "document_id": document_id,
        "query": query,
        "top_k": top_k,
        "min_score": min_score,
        "window": window,
        # token usuario para que el tool lo reenvíe al NLP protegido
        "access_token": tok,
    }

    try:
        data = await mcp_client.ainvoke_tool(
            tool_name,
            payload,
            auth_token=mcp_auth_token or access_token,  # 👈 auth gateway MCP
            timeout=timeout,
        )
    except Exception as e:
        logger.exception("[chatdoc/api] Error MCP query tool=%s: %s", tool_name, e)
        raise HTTPException(status_code=502, detail="Error llamando al tool MCP de consulta.")

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Respuesta inválida del tool MCP de consulta.")

    results = data.get("results") or []
    status = data.get("status") or "ok"

    chunks: List[Dict[str, Any]] = []
    if isinstance(results, list):
        for r in results:
            if not isinstance(r, dict):
                continue
            text = (r.get("text") or "").strip()
            if not text:
                continue

            score = r.get("score")
            if score is None:
                score = (
                    r.get("similarity")
                    or r.get("similarity_score")
                    or r.get("rrf_score")
                    or 0.0
                )

            backend_meta = r.get("metadata") or r.get("meta") or {}
            if not isinstance(backend_meta, dict):
                backend_meta = {}

            page = backend_meta.get("page") or backend_meta.get("page_number")

            chunks.append(
                {
                    "text": text,
                    "page": page,
                    "score": float(score) if score is not None else None,
                    "meta": backend_meta,
                }
            )

    out_status = status if chunks or status != "ok" else "no_results"

    return {
        "status": out_status,
        "chunks": chunks,
        "summary": None,
    }


def log_chatdoc_interaction(
    user_id: int,
    doc_session_id: str,
    conversation_id: int,
    entry: Dict[str, Any],
) -> None:
    """
    Log JSONL por usuario y doc_session para auditoría específica
    del microservicio chat_document.
    """
    try:
        now_utc = datetime.now(timezone.utc)
        date_str = now_utc.strftime("%Y-%m-%d")
        day_dir = LOGS_ROOT / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        filename = f"user_{user_id}_chatdoc_{doc_session_id}_conv_{conversation_id}.log"
        log_path = day_dir / filename

        record = dict(entry)
        record.setdefault("timestamp", now_utc.isoformat())

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("No se pudo escribir log de chat_document (%s): %s", LOGS_ROOT, e)


@app.get("/metrics")
async def metrics():
    payload = generate_latest()
    return PlainTextResponse(payload, media_type=CONTENT_TYPE_LATEST)


_START_TIME = time.monotonic()

@app.get("/health", tags=["health"])
async def health():
    uptime_seconds = int(time.monotonic() - _START_TIME)
    return {
        "status": "ok",
        "service": os.getenv("SERVICE_NAME", "unknown-service"),
        "uptime_seconds": uptime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/document/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    auth: dict = Depends(get_current_auth_chatdoc),
    db: Session = Depends(get_db),
):
    """
    Sube un documento (hasta N MB, N páginas) y lo ingesta en el
    microservicio NLP (vía MCP opcional + fallback HTTP).
    Devuelve un doc_session_id que el front usará para chatear contra ese documento.

    Seguridad:
    - CSRF double-submit.
    - Auth usuario (get_current_auth_chatdoc).
    - Reenvío del JWT del usuario:
        * como auth_token hacia el gateway MCP (si se usa MCP),
        * y dentro del payload de la tool para el NLP protegido.
    """
    DOC_UPLOAD_REQUESTS.inc()
    start = time.time()

    # Protección CSRF
    validate_csrf(request)

    user_id = auth["user_id"]
    user_role = auth["role"]
    session_data = auth.get("session") or {}

    if user_role not in [ROLE_SUPERVISOR, ROLE_USER]:
        DOC_UPLOAD_ERRORS.inc()
        raise HTTPException(
            status_code=403,
            detail="Permisos insuficientes para cargar documentos.",
        )

    # JWT de SSO (cookie 'access_token') que vamos a reenviar
    access_token = request.cookies.get("access_token")

    user_directory = session_data.get("user_directory") or f"user_{user_id}"
    filename = file.filename or "document"
    logger.info("[chatdoc/upload] Procesando archivo %s para user_id=%s", filename, user_id)

    try:
        if not allowed_file(filename):
            DOC_UPLOAD_ERRORS.inc()
            raise HTTPException(
                status_code=400,
                detail="Formato de archivo no soportado para chat de documentos.",
            )

        content = await file.read()
        if len(content) > CHATDOC_MAX_FILE_BYTES:
            DOC_UPLOAD_ERRORS.inc()
            max_mb = CHATDOC_MAX_FILE_MB
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El archivo supera el tamaño máximo permitido "
                    f"({max_mb} MB) para chat de documentos."
                ),
            )

        # Antivirus
        av_result = await scan_with_clamav(content, filename=filename)
        logger.info("[chatdoc/upload] Resultado AV para %s: %s", filename, av_result)

        audit_av = AuditLog(
            user_id=user_id,
            entity_name="ChatDocumentScan",
            entity_id=0,
            action="CREATE",
            old_data=None,
            new_data={
                "filename": filename,
                "av_status": av_result.get("status"),
                "av_virus": av_result.get("virus_name"),
                "av_raw_result": av_result.get("raw_result"),
                "av_error": av_result.get("error"),
                "av_duration_s": av_result.get("duration_s"),
                "size_bytes": av_result.get("size_bytes"),
            },
            timestamp=datetime.now(timezone.utc),
        )
        db.add(audit_av)
        db.commit()

        if av_result.get("status") == "INFECTED":
            DOC_UPLOAD_ERRORS.inc()
            virus_name = av_result.get("virus_name") or "malware"
            raise HTTPException(
                status_code=400,
                detail=(
                    "El archivo ha sido bloqueado por el antivirus "
                    f"(detectado: {virus_name})."
                ),
            )

        if av_result.get("status") == "ERROR":
            DOC_UPLOAD_ERRORS.inc()
            raise HTTPException(
                status_code=400,
                detail=(
                    "No se ha podido analizar el archivo con el sistema antivirus. "
                    "Por seguridad, el archivo ha sido rechazado."
                ),
            )

        guessed_mime, _ = guess_type(filename)
        content_type = file.content_type or guessed_mime or "application/octet-stream"

        # ✅ Ingest protegido:
        #    - mcp_auth_token autentica el gateway MCP (si se usa MCP)
        #    - access_token viaja también en payload hacia NLP
        ingest_timeout = _compute_ingest_timeout_s(len(content))
        nlp_info = await nlp_ingest_document(
            content=content,
            filename=filename,
            mime_type=content_type,
            user_id=user_id,
            timeout_s=ingest_timeout,
            access_token=access_token,
            mcp_auth_token=access_token,
        )

        document_id = nlp_info.get("document_id")
        page_count = int(nlp_info.get("page_count") or 0)

        if not document_id:
            DOC_UPLOAD_ERRORS.inc()
            raise HTTPException(
                status_code=502,
                detail="La ingesta del documento no devolvió un document_id.",
            )

        if page_count > CHATDOC_MAX_PAGES:
            DOC_UPLOAD_ERRORS.inc()
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El documento tiene {page_count} páginas y supera el máximo "
                    f"permitido de {CHATDOC_MAX_PAGES} páginas."
                ),
            )

        # Guardar archivo en disco
        user_root = STORAGE_ROOT / user_directory
        save_dir = user_root / "chat_documents"
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        stored_path = save_dir / safe_name
        if stored_path.exists():
            stored_path = save_dir / f"{uuid.uuid4().hex}_{safe_name}"

        with stored_path.open("wb") as dest:
            dest.write(content)

        # Crear conversación asociada al documento
        conversation = Conversation(
            user_id=user_id,
            conversation_text=f"DOCUMENTO: {filename} ({page_count} páginas)",
            created_at=datetime.now(timezone.utc),
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        system_msg_text = (
            f"Documento '{filename}' procesado y comprendido para chatear. Comencemos!"
        )
        system_msg = Message(
            conversation_id=conversation.id,
            sender="SYSTEM",
            content=system_msg_text,
            created_at=datetime.now(timezone.utc),
        )
        db.add(system_msg)

        file_rec = FileModel(
            user_id=user_id,
            department_id=None,
            file_path=str(stored_path),
            file_name=filename,
            permission="READ",
            created_at=datetime.now(timezone.utc),
        )
        db.add(file_rec)

        audit_upload = AuditLog(
            user_id=user_id,
            entity_name="ChatDocumentUpload",
            entity_id=conversation.id,
            action="CREATE",
            old_data=None,
            new_data={
                "filename": filename,
                "document_id": document_id,
                "page_count": page_count,
                "nlp_keys": list(nlp_info.keys()) if isinstance(nlp_info, dict) else [],
            },
            timestamp=datetime.now(timezone.utc),
        )
        db.add(audit_upload)
        db.commit()

        # Crear sesión de documento en Redis
        doc_session_id = uuid.uuid4().hex
        doc_session = {
            "document_id": document_id,
            "conversation_id": conversation.id,
            "file_name": filename,
            "page_count": page_count,
            "created_at": datetime.utcnow().isoformat(),
        }
        await store_doc_session(user_id, doc_session_id, doc_session)

        return DocumentUploadResponse(
            doc_session_id=doc_session_id,
            document_id=document_id,
            page_count=page_count,
            file_name=filename,
            conversation_id=conversation.id,
        )

    except HTTPException as e:
        DOC_UPLOAD_ERRORS.inc()
        raise e
    except Exception as e:
        DOC_UPLOAD_ERRORS.inc()
        logger.exception("[chatdoc/upload] Error subiendo documento: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno procesando el documento: {e}",
        )
    finally:
        duration = time.time() - start
        DOC_UPLOAD_DURATION.observe(duration)



@app.post("/document/query", response_model=DocumentQueryResponse)
async def query_document(
    request: Request,
    background_tasks: BackgroundTasks,
    body: DocumentQueryRequest,
    auth: dict = Depends(get_current_auth_chatdoc),
    db: Session = Depends(get_db),
):
    DOC_CHAT_REQUESTS.inc()
    start = time.time()

    try:
        # -------------------------
        # 0) Seguridad / validaciones básicas
        # -------------------------
        validate_csrf(request)

        user_id = auth["user_id"]
        session_data = auth.get("session") or {}

        prompt = (body.prompt or "").strip()
        if not prompt:
            DOC_CHAT_ERRORS.inc()
            raise HTTPException(
                status_code=400,
                detail="No se ha proporcionado prompt para el chat con el documento.",
            )

        # JWT del usuario (cookie)
        access_token = request.cookies.get("access_token")

        doc_session_id = (body.doc_session_id or "").strip()
        if not doc_session_id:
            DOC_CHAT_ERRORS.inc()
            raise HTTPException(status_code=400, detail="doc_session_id es obligatorio.")

        doc_session = await get_doc_session(user_id, doc_session_id)
        if not doc_session:
            DOC_CHAT_ERRORS.inc()
            raise HTTPException(status_code=404, detail="Sesión de documento no encontrada o expirada.")

        document_id = (doc_session.get("document_id") or "").strip()
        if not document_id:
            DOC_CHAT_ERRORS.inc()
            raise HTTPException(status_code=500, detail="Sesión de documento corrupta: falta document_id.")

        # -------------------------
        # 1) Conversación BD (asegura Conversation)
        # -------------------------
        conversation_id = body.conversation_id or doc_session.get("conversation_id")
        conversation = None
        if conversation_id:
            conversation = (
                db.query(Conversation)
                .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
                .first()
            )

        if not conversation:
            conversation = Conversation(
                user_id=user_id,
                conversation_text="",
                created_at=datetime.now(timezone.utc),
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            # persistimos el conversation_id en la sesión del documento
            doc_session["conversation_id"] = conversation.id
            await store_doc_session(user_id, doc_session_id, doc_session)

        # Guardar mensaje USER
        conversation.conversation_text = (
            (conversation.conversation_text + "\n" if conversation.conversation_text else "")
            + f"USER: {prompt}"
        )
        user_msg = Message(
            conversation_id=conversation.id,
            sender="USER",
            content=prompt,
            created_at=datetime.now(timezone.utc),
        )
        db.add(user_msg)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        # Historial corto desde Redis
        history = await get_doc_history(user_id, doc_session_id)

        # -------------------------
        # 2) Detección base de modo
        # -------------------------
        lp = prompt.lower()
        summary_triggers = [
            "resumen", "resume", "sumario", "sinopsis", "síntesis", "sintesis",
            "de qué trata", "de que trata", "de qué va", "de que va",
            "de qué va el documento", "resúmeme", "haz un resumen", "explica brevemente",
        ]

        user_mode = (body.mode or "").lower().strip()
        if user_mode in ("qa", "summary"):
            mode = user_mode
            user_forced_mode = True
        else:
            mode = "summary" if any(t in lp for t in summary_triggers) else "qa"
            user_forced_mode = False

        doc_meta_for_planner: Dict[str, Any] = {
            "file_name": doc_session.get("file_name"),
            "page_count": doc_session.get("page_count"),
            "document_id": document_id,
        }

        # -------------------------
        # 3) Planner (THREAD SAFE: usar run_sync_kwargs)
        # -------------------------
        planner_ok = False
        doc_plan: Dict[str, Any] = {}
        try:
            doc_plan = await run_sync_kwargs(
                doc_crew_orchestrator.plan_doc_query,
                user_prompt=prompt,
                history=history,
                mode=mode,
                doc_meta=doc_meta_for_planner,
                access_token=access_token,
                limiter=CHATDOC_PLANNER_LIMITER,
            )
            if not isinstance(doc_plan, dict):
                doc_plan = {}
            planner_ok = True
        except Exception as e:
            logger.exception("[chatdoc/query] Error ejecutando planner de documento: %s", e)
            doc_plan = {}

        normalized_prompt = (doc_plan.get("normalized_question") or "").strip() or prompt

        planner_mode = (doc_plan.get("mode") or mode or "qa").lower()
        if planner_mode not in ("qa", "summary"):
            planner_mode = mode

        # Si el usuario NO forzó modo explícito, dejamos que planner + triggers decidan
        if not user_forced_mode:
            mode = "summary" if (any(t in lp for t in summary_triggers) or planner_mode == "summary") else "qa"

        # normalización defensiva de detail_level / summary_profile
        detail_level = doc_plan.get("detail_level")
        if isinstance(detail_level, str):
            dl = detail_level.strip().lower()
            detail_level = dl if dl in ("low", "medium", "high") else None
        else:
            detail_level = None

        summary_profile = doc_plan.get("summary_profile")
        summary_profile = summary_profile.strip().lower() if isinstance(summary_profile, str) else None

        query_type = doc_plan.get("query_type")

        if planner_ok:
            logger.info(
                "[chatdoc/query] planner_ok doc_id=%s mode=%s planner_mode=%s detail=%s profile=%s type=%s norm='%s'",
                document_id,
                mode,
                planner_mode,
                detail_level,
                summary_profile,
                query_type,
                normalized_prompt[:140].replace("\n", " "),
            )
        else:
            logger.warning(
                "[chatdoc/query] planner_fallback doc_id=%s mode=%s norm='%s'",
                document_id,
                mode,
                normalized_prompt[:140].replace("\n", " "),
            )

        # -------------------------
        # 4) Parámetros defensivos API → orquestador
        # -------------------------
        try:
            api_top_k = int(os.getenv("CHATDOC_API_QUERY_TOPK", "8"))
        except Exception:
            api_top_k = 8

        try:
            api_min_score = float(os.getenv("CHATDOC_QUERY_MIN_SCORE", "0.0"))
        except Exception:
            api_min_score = 0.0

        try:
            api_window = int(os.getenv("CHATDOC_SEARCH_WINDOW", "1"))
        except Exception:
            api_window = 1

        # límites razonables
        api_top_k = max(1, min(api_top_k, int(os.getenv("CHATDOC_API_QUERY_MAX_TOPK", "16")) or 16))
        api_window = max(0, min(api_window, int(os.getenv("CHATDOC_SEARCH_MAX_WINDOW", "5")) or 5))

        # -------------------------
        # 5) ORQUESTADOR (THREAD SAFE: usar run_sync_kwargs)
        # -------------------------
        try:
            response_text = await run_sync_kwargs(
                doc_crew_orchestrator.run_doc_chat,
                user_prompt=prompt,
                doc_context="",
                history=history,
                mode=mode,
                doc_meta={
                    "file_name": doc_session.get("file_name"),
                    "page_count": doc_session.get("page_count"),
                    "document_id": document_id,
                    "detail_level": detail_level,
                    "summary_profile": summary_profile,
                    "query_type": query_type,
                    "api_source": "chat_document",
                },
                precomputed_summary=None,
                normalized_prompt=normalized_prompt,
                plan=doc_plan,
                verify_answer=True,
                document_id=document_id,
                access_token=access_token,
                mcp_auth_token=None,
                top_k=api_top_k,
                min_score=api_min_score,
                window=api_window,
                limiter=CHATDOC_CREW_LIMITER,
            )
            response_text = clean_text(response_text or "")
        except Exception as e:
            DOC_CHAT_ERRORS.inc()
            logger.exception("[chatdoc/query] Error ejecutando DocumentCrewOrchestrator: %s", e)
            raise HTTPException(status_code=500, detail="Error generando la respuesta a partir del documento.")

        # -------------------------
        # 6) Persistencia async en background (no bloquea request)
        # -------------------------
        nlp_status = "mcp_orchestrator_autofetch"

        def persist_bot_and_usage(convo_id: int, content: str) -> None:
            db_local = next(get_db())
            try:
                bot_msg = Message(
                    conversation_id=convo_id,
                    sender="BOT",
                    content=content,
                    created_at=datetime.now(timezone.utc),
                )
                db_local.add(bot_msg)
                db_local.commit()
                db_local.refresh(bot_msg)

                usage = UsageLog(
                    message_id=bot_msg.id,
                    conversation_id=convo_id,
                    model_name=os.getenv("CHATDOC_MODEL_NAME", os.getenv("CREW_MODEL_NAME", "chatdoc-llm")),
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cost=0.0,
                    created_at=datetime.now(timezone.utc),
                )
                db_local.add(usage)

                audit = AuditLog(
                    user_id=user_id,
                    entity_name="ChatDocumentAnswer",
                    entity_id=convo_id,
                    action="CREATE",
                    old_data=None,
                    new_data={
                        "doc_session_id": doc_session_id,
                        "document_id": document_id,
                        "mode": mode,
                        "planner_mode": planner_mode,
                        "query_type": query_type,
                        "detail_level": detail_level,
                        "summary_profile": summary_profile,
                        "prompt": prompt,
                        "normalized_prompt": normalized_prompt,
                        "response": content,
                        "nlp_status": nlp_status,
                        "mcp_top_k": api_top_k,
                        "mcp_min_score": api_min_score,
                        "mcp_window": api_window,
                        "gateway_auth": "s2s_preferred",
                    },
                    timestamp=datetime.now(timezone.utc),
                )
                db_local.add(audit)
                db_local.commit()
            finally:
                db_local.close()

        background_tasks.add_task(persist_bot_and_usage, conversation.id, response_text)

        # conversación “cacheada” (texto largo) se actualiza sin bloquear background
        conversation.conversation_text = (
            (conversation.conversation_text + "\n" if conversation.conversation_text else "")
            + f"BOT: {response_text}"
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        # -------------------------
        # 7) Guardar historial en Redis + log JSONL
        # -------------------------
        now_utc = datetime.now(timezone.utc)
        history_entry = {
            "conversation_id": conversation.id,
            "user_id": user_id,
            "doc_session_id": doc_session_id,
            "prompt": prompt,
            "normalized_prompt": normalized_prompt,
            "response": response_text,
            "mode": mode,
            "planner_mode": planner_mode,
            "detail_level": detail_level,
            "summary_profile": summary_profile,
            "query_type": query_type,
            "timestamp": now_utc.isoformat(),
        }
        await append_doc_history(user_id, doc_session_id, history_entry)

        try:
            log_chatdoc_interaction(
                user_id=user_id,
                doc_session_id=doc_session_id,
                conversation_id=conversation.id,
                entry={**history_entry, "nlp_status": nlp_status, "nlp_chunks_count": 0},
            )
        except Exception:
            pass

        duration = time.time() - start
        DOC_CHAT_DURATION.observe(duration)

        return DocumentQueryResponse(
            reply=response_text,
            response=response_text,
            conversation_id=conversation.id,
            doc_session_id=doc_session_id,
            sources=[],
        )

    except HTTPException:
        raise
    except Exception as e:
        DOC_CHAT_ERRORS.inc()
        logger.exception("[chatdoc/query] Error inesperado: %s", e)
        raise HTTPException(status_code=500, detail="Error interno inesperado en chat_document.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)

# web_search/main.py
"""
Microservicio `web_search`: búsqueda web avanzada con CrewAI.

- Autenticado/CSRF igual que chat_document.
- Redis para sesiones e historial de búsquedas conversacionales.
- Consume tools MCP (ddg_*) a través del MCP Gateway (desde el orquestador).
- Orquesta una crew: planner → ejecutor de tools → analista → redactor → revisor (recursivo).
"""

from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    Depends,
    BackgroundTasks,
    Response,
    UploadFile,
    File,
)
from fastapi.responses import PlainTextResponse
# Si quieres devolver JSON con errores detallados, puedes usar JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime, timezone
import os
import time
import json
import uuid
import logging
import tempfile
import hashlib
import asyncio
import redis.asyncio as aioredis
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
from time import perf_counter

from config.database import get_db
from config.models import (
    Conversation,
    Message,
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
    init_redis_clients,
    get_current_auth_chatdoc,
    clean_text,
    extract_raw_bearer_token,
    allowed_file,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_doc,
    extract_text_from_xlsx,
    extract_text_from_csv,
    extract_text_from_pptx,
    extract_text_from_txt,
    add_ephemeral_file,
    get_ephemeral_files,
    set_user_context,
    scan_with_clamav,
)
from legal_crew_orchestrator import LegalSearchCrewOrchestrator
from metrics import (
    status_class,
    hash_user_id,
    user_metrics_enabled,
    WEBSEARCH_HTTP_REQUESTS_TOTAL,
    WEBSEARCH_HTTP_LATENCY_SECONDS,
    WEBSEARCH_HTTP_INFLIGHT,
    WEBSEARCH_USER_ENDPOINT_REQUESTS_TOTAL,
    WEBSEARCH_QUERY_REQUESTS_TOTAL,
    WEBSEARCH_QUERY_ERRORS_TOTAL,
    WEBSEARCH_QUERY_DURATION_SECONDS,
    WEBSEARCH_ANSWER_CHARS,
    WEBSEARCH_SOURCES_PER_ANSWER,
    WEBSEARCH_NORMALIZED_QUERIES_PER_ANSWER,
    WEBSEARCH_CREW_RUN_SECONDS,
    WEBSEARCH_AGENT_STEP_DURATION_SECONDS,
    WEBSEARCH_TOOL_CALLS_TOTAL,
    WEBSEARCH_TOOL_DURATION_SECONDS,
    WEBSEARCH_REDIS_OPS_TOTAL,
    WEBSEARCH_REDIS_LATENCY_SECONDS,
    WEBSEARCH_REDIS_CONNECTED,
    WEBSEARCH_DB_LATENCY_SECONDS,
    WEBSEARCH_DB_ERRORS_TOTAL,
    WEBSEARCH_BG_TASKS_TOTAL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_search")

BASE_DIR = Path(__file__).resolve().parent
LOGS_ROOT = Path(os.getenv("COSMOS_WEBSEARCH_LOGS_DIR", str(BASE_DIR / "logs"))).resolve()

# CSRF
CSRF_COOKIE_NAME = "csrftoken_websearch"
AUTH_ORIGIN = os.getenv("COSMOS_AUTH_ORIGIN", "http://localhost:7000")
FRONTEND_URL = os.getenv("COSMOS_FRONTEND_URL", "http://localhost").rstrip("/")

# Redis global (se inicializa en startup)
redis_core_client: Optional[aioredis.Redis] = None
redis_conv_client: Optional[aioredis.Redis] = None

WEBSEARCH_SESSION_TTL = int(os.getenv("WEBSEARCH_SESSION_TTL", "3600"))
WEBSEARCH_HISTORY_MAX_LEN = int(os.getenv("WEBSEARCH_HISTORY_MAX_LEN", "30"))

WS_SESSION_KEY_PREFIX = "websearch:session"
WS_HISTORY_KEY_PREFIX = "websearch:history"
WS_HITL_KEY_PREFIX = "websearch:hitl"
WS_FILE_KEY_PREFIX = "websearch:file"
WS_FILE_INDEX_KEY_PREFIX = "websearch:file_index"

app = FastAPI()

# CORS (similar a modelo_negocio / chat_document)
ALLOWED_ORIGINS_ENV = os.getenv("COSMOS_ALLOWED_ORIGINS", "")
if ALLOWED_ORIGINS_ENV:
    allowed_origins = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]
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


class WebSearchQueryRequest(BaseModel):
    prompt: str
    search_session_id: Optional[str] = None
    conversation_id: Optional[int] = None
    attached_file_ids: List[str] = Field(default_factory=list)
    human_in_the_loop: bool = False
    require_human_approval: bool = False
    # Opciones avanzadas
    top_k: Optional[int] = None      # URLs a profundizar por iteración
    max_iters: Optional[int] = None  # recursión (planner→ejecutor)


class WebSearchQueryResponse(BaseModel):
    reply: str
    response: str
    conversation_id: int
    search_session_id: str
    sources: List[Dict[str, Any]]
    normalized_queries: List[str] = Field(default_factory=list)
    plan_meta: Dict[str, Any] = Field(default_factory=dict)
    context_files: List[Dict[str, Any]] = Field(default_factory=list)
    hitl: Dict[str, Any] = Field(default_factory=dict)


class HumanReviewDecisionRequest(BaseModel):
    search_session_id: str
    conversation_id: int
    review_task_id: str
    decision: Literal["approve", "revise", "reject"]
    reviewer_notes: Optional[str] = ""
    revised_prompt: Optional[str] = ""


# -----------------------------
# Helpers de CSRF y Redis
# -----------------------------
def validate_csrf(request: Request) -> None:
    validate_csrf_double_submit(
        request,
        cookie_name=CSRF_COOKIE_NAME,
        header_name="X-CSRFToken",
        error_detail="CSRF token inválido o ausente en servicio web_search.",
    )

def _extract_text_for_legal_context(file_path: str, filename: str) -> str:
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    if ext == "docx":
        return extract_text_from_docx(file_path)
    if ext == "doc":
        return extract_text_from_doc(file_path)
    if ext == "xlsx":
        return extract_text_from_xlsx(file_path)
    if ext == "csv":
        return extract_text_from_csv(file_path)
    if ext == "pptx":
        return extract_text_from_pptx(file_path)
    if ext == "txt":
        return extract_text_from_txt(file_path)
    if ext in {"jpg", "jpeg", "png"}:
        return "[Imagen subida por usuario. OCR no habilitado en legal_search actualmente.]"
    return ""


def _legal_file_key(user_id: int, search_session_id: str, file_id: str) -> str:
    return f"{WS_FILE_KEY_PREFIX}:{user_id}:{search_session_id}:{file_id}"


def _legal_file_index_key(user_id: int, search_session_id: str) -> str:
    return f"{WS_FILE_INDEX_KEY_PREFIX}:{user_id}:{search_session_id}"


async def _store_legal_ephemeral_file(
    user_id: int,
    search_session_id: str,
    file_id: str,
    payload: Dict[str, Any],
    ttl: int,
) -> None:
    """Almacena cada fichero en clave dedicada para acceso O(1) por file_id.

    Mantiene también un índice por sesión para recuperar últimos ficheros cuando
    no se pasan `attached_file_ids` en la pregunta.
    """
    if redis_conv_client is None:
        return

    data = json.dumps(payload, ensure_ascii=False)
    file_key = _legal_file_key(user_id, search_session_id, file_id)
    idx_key = _legal_file_index_key(user_id, search_session_id)

    pipe = redis_conv_client.pipeline()
    pipe.set(file_key, data, ex=ttl)
    pipe.lpush(idx_key, file_id)
    pipe.ltrim(idx_key, 0, 199)  # límite defensivo por sesión
    pipe.expire(idx_key, ttl)
    await pipe.execute()


async def _load_legal_ephemeral_files(
    user_id: int,
    search_session_id: str,
    file_ids: List[str],
) -> List[Dict[str, Any]]:
    if redis_conv_client is None:
        return []

    wanted_ids = [str(fid).strip() for fid in (file_ids or []) if str(fid).strip()]

    if not wanted_ids:
        idx_key = _legal_file_index_key(user_id, search_session_id)
        wanted_ids = [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in (await redis_conv_client.lrange(idx_key, 0, 7))
        ]

    if not wanted_ids:
        return []

    keys = [_legal_file_key(user_id, search_session_id, fid) for fid in wanted_ids]
    rows = await redis_conv_client.mget(keys)

    out: List[Dict[str, Any]] = []
    for raw in rows:
        if not raw:
            continue
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            item = json.loads(raw)
            if isinstance(item, dict):
                out.append(item)
        except Exception:
            continue
    return out


async def _build_legal_user_context(user_id: int, search_session_id: str, conversation_id: Optional[int], file_ids: List[str]) -> Dict[str, Any]:
    selected = await _load_legal_ephemeral_files(
        user_id=user_id,
        search_session_id=search_session_id,
        file_ids=file_ids or [],
    )

    # Fallback retrocompatible: payloads antiguos en clave global ephemeral:{user_id}
    if not selected:
        raw_files = await get_ephemeral_files(user_id)
        wanted = set(file_ids or [])
        for item in raw_files:
            if not isinstance(item, dict):
                continue
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            item_sid = str(meta.get("search_session_id") or "").strip()
            item_cid = meta.get("conversation_id")
            item_id = str(meta.get("file_id") or "").strip()
            if item_sid and item_sid != search_session_id:
                continue
            if conversation_id and item_cid and int(item_cid) != int(conversation_id):
                continue
            if wanted and item_id not in wanted:
                continue
            selected.append(item)

    selected = selected[-8:]
    chunks: List[str] = []
    for idx, f in enumerate(selected, start=1):
        fn = f.get("filename") or f"archivo_{idx}"
        txt = (f.get("text") or "").strip()
        if not txt:
            continue
        chunks.append(f"[{idx}] Archivo: {fn}\n{txt[:2200]}")

    return {"context": "\n\n".join(chunks).strip(), "files": selected}




async def _store_hitl_task(user_id: int, payload: Dict[str, Any]) -> None:
    if redis_conv_client is None:
        raise RuntimeError("Redis conv client not initialized")
    key = f"{WS_HITL_KEY_PREFIX}:{user_id}:{payload['search_session_id']}:{payload['conversation_id']}:{payload['review_task_id']}"
    await redis_conv_client.set(key, json.dumps(payload, ensure_ascii=False), ex=WEBSEARCH_SESSION_TTL)


async def _read_hitl_task(user_id: int, search_session_id: str, conversation_id: int, review_task_id: str) -> Optional[Dict[str, Any]]:
    if redis_conv_client is None:
        raise RuntimeError("Redis conv client not initialized")
    key = f"{WS_HITL_KEY_PREFIX}:{user_id}:{search_session_id}:{conversation_id}:{review_task_id}"
    raw = await redis_conv_client.get(key)
    return json.loads(raw) if raw else None


def get_websearch_crew(request: Request) -> "LegalSearchCrewOrchestrator":
    """
    Devuelve el orquestador de la crew legal ligado al lifecycle de la app.

    Mantiene el nombre 'websearch_crew' para no tocar más wiring en main.py,
    aunque el orquestador sea LegalSearchCrewOrchestrator.
    """
    crew_obj = getattr(request.app.state, "websearch_crew", None)
    if crew_obj is None:
        # Fallback ultra-defensivo
        crew_obj = LegalSearchCrewOrchestrator()
        request.app.state.websearch_crew = crew_obj
    return crew_obj


@app.get("/csrf-token", response_model=dict)
async def get_csrf_token_endpoint(request: Request, response: Response):
    token = request.cookies.get(CSRF_COOKIE_NAME)
    if not token:
        token = generate_csrf_token()
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        httponly=False,   # el front puede leerla
        secure=False,     # en prod: True + HTTPS
        samesite="Lax",
    )
    return {"csrf_token": token}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method

    start = perf_counter()
    WEBSEARCH_HTTP_INFLIGHT.labels(endpoint=endpoint).inc()

    code = 500
    try:
        response = await call_next(request)
        code = getattr(response, "status_code", 200)
        return response
    except Exception:
        # Para excepciones no manejadas, contamos como 5xx
        code = 500
        raise
    finally:
        duration = perf_counter() - start
        WEBSEARCH_HTTP_LATENCY_SECONDS.labels(endpoint=endpoint, method=method).observe(duration)
        WEBSEARCH_HTTP_REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            method=method,
            status_class=status_class(code),
        ).inc()
        WEBSEARCH_HTTP_INFLIGHT.labels(endpoint=endpoint).dec()

@app.on_event("startup")
async def initialize_redis():
    global redis_core_client, redis_conv_client
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    t0 = perf_counter()
    try:
        redis_core_client = aioredis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        redis_conv_client = aioredis.Redis(host=redis_host, port=redis_port, decode_responses=True, db=3)

        # registro en utils (como ya haces)
        init_redis_clients(redis_core_client, redis_conv_client)

        # Best-effort ping para marcar "connected"
        try:
            await redis_core_client.ping()
            await redis_conv_client.ping()
            WEBSEARCH_REDIS_CONNECTED.set(1)
        except Exception:
            WEBSEARCH_REDIS_CONNECTED.set(0)

        logger.info(
            "[startup-websearch] Redis conectado en %s:%s (db=0 y db=3) y registrado en utils.",
            redis_host,
            redis_port,
        )
    finally:
        WEBSEARCH_REDIS_LATENCY_SECONDS.labels(op="startup_initialize_redis").observe(perf_counter() - t0)

    # Instancia del orquestador ligada al ciclo de vida de la app (AHORA: LegalSearchCrewOrchestrator)
    app.state.websearch_crew = LegalSearchCrewOrchestrator()
    logger.info("[startup-websearch] LegalSearchCrewOrchestrator inicializado y guardado en app.state (websearch_crew).")


@app.on_event("shutdown")
async def shutdown_redis():
    global redis_core_client, redis_conv_client
    t0 = perf_counter()
    try:
        if redis_core_client is not None:
            try:
                await redis_core_client.close()
            except Exception:
                pass
        if redis_conv_client is not None:
            try:
                await redis_conv_client.close()
            except Exception:
                pass
        WEBSEARCH_REDIS_CONNECTED.set(0)
        logger.info("[shutdown-websearch] Redis clients cerrados.")
    finally:
        WEBSEARCH_REDIS_LATENCY_SECONDS.labels(op="shutdown_redis").observe(perf_counter() - t0)


async def store_ws_session(
    user_id: int,
    search_session_id: str,
    payload: Dict[str, Any],
    ttl: Optional[int] = None,
) -> None:
    if redis_core_client is None:
        WEBSEARCH_REDIS_OPS_TOTAL.labels(op="store_ws_session", result="error").inc()
        raise RuntimeError("Redis core client not initialized")

    key = f"{WS_SESSION_KEY_PREFIX}:{user_id}:{search_session_id}"
    t0 = perf_counter()
    try:
        await redis_core_client.set(
            key,
            json.dumps(payload, ensure_ascii=False),
            ex=ttl or WEBSEARCH_SESSION_TTL,
        )
        WEBSEARCH_REDIS_OPS_TOTAL.labels(op="store_ws_session", result="ok").inc()
    except Exception:
        WEBSEARCH_REDIS_OPS_TOTAL.labels(op="store_ws_session", result="error").inc()
        raise
    finally:
        WEBSEARCH_REDIS_LATENCY_SECONDS.labels(op="store_ws_session").observe(perf_counter() - t0)



async def get_ws_session(user_id: int, search_session_id: str) -> Optional[Dict[str, Any]]:
    if redis_core_client is None:
        WEBSEARCH_REDIS_OPS_TOTAL.labels(op="get_ws_session", result="error").inc()
        raise RuntimeError("Redis core client not initialized")

    key = f"{WS_SESSION_KEY_PREFIX}:{user_id}:{search_session_id}"
    t0 = perf_counter()
    raw = None
    try:
        raw = await redis_core_client.get(key)
        if not raw:
            WEBSEARCH_REDIS_OPS_TOTAL.labels(op="get_ws_session", result="miss").inc()
            return None

        try:
            parsed = json.loads(raw)
            WEBSEARCH_REDIS_OPS_TOTAL.labels(op="get_ws_session", result="hit").inc()
            return parsed
        except json.JSONDecodeError:
            WEBSEARCH_REDIS_OPS_TOTAL.labels(op="get_ws_session", result="decode_error").inc()
            logger.warning(
                "No se pudo parsear ws_session en Redis (key=%s). Valor crudo truncado: %r",
                key,
                raw[:200],
            )
            return None
    except Exception:
        WEBSEARCH_REDIS_OPS_TOTAL.labels(op="get_ws_session", result="error").inc()
        raise
    finally:
        WEBSEARCH_REDIS_LATENCY_SECONDS.labels(op="get_ws_session").observe(perf_counter() - t0)



async def append_ws_history(
    user_id: int,
    search_session_id: str,
    entry: Dict[str, Any],
    ttl: Optional[int] = None,
) -> None:
    if redis_conv_client is None:
        WEBSEARCH_REDIS_OPS_TOTAL.labels(op="append_ws_history", result="error").inc()
        raise RuntimeError("Redis conv client not initialized")

    key = f"{WS_HISTORY_KEY_PREFIX}:{user_id}:{search_session_id}"
    t0 = perf_counter()
    try:
        raw = await redis_conv_client.get(key)
        if raw:
            try:
                history = json.loads(raw)
                if not isinstance(history, list):
                    history = []
            except json.JSONDecodeError:
                WEBSEARCH_REDIS_OPS_TOTAL.labels(op="append_ws_history", result="decode_error").inc()
                history = []
        else:
            history = []

        history.append(entry)
        if len(history) > WEBSEARCH_HISTORY_MAX_LEN:
            history = history[-WEBSEARCH_HISTORY_MAX_LEN:]

        await redis_conv_client.set(
            key,
            json.dumps(history, ensure_ascii=False),
            ex=ttl or WEBSEARCH_SESSION_TTL,
        )
        WEBSEARCH_REDIS_OPS_TOTAL.labels(op="append_ws_history", result="ok").inc()
    except Exception:
        WEBSEARCH_REDIS_OPS_TOTAL.labels(op="append_ws_history", result="error").inc()
        raise
    finally:
        WEBSEARCH_REDIS_LATENCY_SECONDS.labels(op="append_ws_history").observe(perf_counter() - t0)


async def get_ws_history(user_id: int, search_session_id: str) -> List[Dict[str, Any]]:
    if redis_conv_client is None:
        WEBSEARCH_REDIS_OPS_TOTAL.labels(op="get_ws_history", result="error").inc()
        raise RuntimeError("Redis conv client not initialized")

    key = f"{WS_HISTORY_KEY_PREFIX}:{user_id}:{search_session_id}"
    t0 = perf_counter()
    raw = None
    try:
        raw = await redis_conv_client.get(key)
        if not raw:
            WEBSEARCH_REDIS_OPS_TOTAL.labels(op="get_ws_history", result="miss").inc()
            return []

        try:
            history = json.loads(raw)
            WEBSEARCH_REDIS_OPS_TOTAL.labels(op="get_ws_history", result="hit").inc()
            return history if isinstance(history, list) else []
        except json.JSONDecodeError:
            WEBSEARCH_REDIS_OPS_TOTAL.labels(op="get_ws_history", result="decode_error").inc()
            logger.warning(
                "No se pudo parsear ws_history en Redis (key=%s). Valor crudo truncado: %r",
                key,
                raw[:200],
            )
            return []
    except Exception:
        WEBSEARCH_REDIS_OPS_TOTAL.labels(op="get_ws_history", result="error").inc()
        raise
    finally:
        WEBSEARCH_REDIS_LATENCY_SECONDS.labels(op="get_ws_history").observe(perf_counter() - t0)


# -----------------------------
# Endpoints
# -----------------------------
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

@app.post("/search/uploadfile", response_model=dict)
async def web_search_upload_file(
    request: Request,
    search_session_id: str,
    conversation_id: Optional[int] = None,
    files: List[UploadFile] = File(...),
    auth: dict = Depends(get_current_auth_chatdoc),
):
    validate_csrf(request)
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Usuario no autenticado.")

    max_file_mb = int(os.getenv("LEGALSEARCH_EPHEMERAL_MAX_FILE_MB", "20"))
    max_file_bytes = max_file_mb * 1024 * 1024
    ttl = int(os.getenv("LEGALSEARCH_EPHEMERAL_TTL", str(WEBSEARCH_SESSION_TTL)))

    responses: List[Dict[str, Any]] = []
    context_lines: List[str] = []

    for upload in files:
        filename = upload.filename or "uploaded_file"
        file_id = uuid.uuid4().hex
        file_result: Dict[str, Any] = {"filename": filename, "file_id": file_id, "status": "ok"}

        if not allowed_file(filename):
            file_result["status"] = "rejected"
            file_result["error"] = "Tipo de archivo no soportado"
            responses.append(file_result)
            continue

        content = await upload.read()
        if len(content) > max_file_bytes:
            file_result["status"] = "rejected"
            file_result["error"] = f"Archivo supera el límite de {max_file_mb}MB"
            responses.append(file_result)
            continue

        av_result = await scan_with_clamav(content, filename=filename)
        if av_result.get("status") == "INFECTED":
            file_result["status"] = "rejected"
            file_result["error"] = f"Malware detectado: {av_result.get('virus_name') or 'unknown'}"
            responses.append(file_result)
            continue

        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        tmp_path = None
        extracted_text = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            extracted_text = (_extract_text_for_legal_context(tmp_path, filename) or "").strip()
        except Exception as e:
            logger.warning("Error extrayendo texto %s: %s", filename, e)
            extracted_text = ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        if not extracted_text:
            extracted_text = f"[No se pudo extraer texto útil de {filename}]"

        digest = hashlib.sha256(content).hexdigest()
        meta = {
            "file_id": file_id,
            "search_session_id": search_session_id,
            "conversation_id": conversation_id,
            "sha256": digest,
            "size_bytes": len(content),
        }
        await add_ephemeral_file(user_id, filename, extracted_text, ttl=ttl, meta=meta)
        await _store_legal_ephemeral_file(
            user_id=user_id,
            search_session_id=search_session_id,
            file_id=file_id,
            payload={
                "filename": filename,
                "text": extracted_text,
                "uploaded_at": datetime.utcnow().isoformat(),
                "meta": meta,
            },
            ttl=ttl,
        )
        context_lines.append(f"Archivo {filename}:\n{extracted_text[:2400]}")
        file_result["chars_extracted"] = len(extracted_text)
        responses.append(file_result)

    if context_lines:
        await set_user_context(user_id, "\n\n".join(context_lines)[:12000], ttl=ttl)

    return {
        "search_session_id": search_session_id,
        "conversation_id": conversation_id,
        "files": responses,
        "accepted": len([x for x in responses if x.get("status") == "ok"]),
    }


@app.post("/search/hitl/decision", response_model=dict)
async def submit_human_review_decision(
    request: Request,
    body: HumanReviewDecisionRequest,
    auth: dict = Depends(get_current_auth_chatdoc),
):
    validate_csrf(request)
    user_id = auth.get("user_id")
    task = await _read_hitl_task(user_id, body.search_session_id, body.conversation_id, body.review_task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Tarea de revisión no encontrada.")

    task["decision"] = body.decision
    task["reviewer_notes"] = (body.reviewer_notes or "").strip()
    task["revised_prompt"] = (body.revised_prompt or "").strip()
    task["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    await _store_hitl_task(user_id, task)

    return {"status": "ok", "task": task}


@app.post("/search/query", response_model=WebSearchQueryResponse)
async def web_search_query(
    request: Request,
    background_tasks: BackgroundTasks,
    body: WebSearchQueryRequest,
    auth: dict = Depends(get_current_auth_chatdoc),
    db: Session = Depends(get_db),
    orchestrator: "LegalSearchCrewOrchestrator" = Depends(get_websearch_crew),
):
    endpoint_label = "/search/query"
    start_total = perf_counter()

    WEBSEARCH_QUERY_REQUESTS_TOTAL.labels(endpoint=endpoint_label).inc()

    user_id = auth.get("user_id")
    user_label = hash_user_id(user_id) if (user_id is not None) else "unknown"

    def _count_user(status_code: int) -> None:
        if user_metrics_enabled() and user_id is not None:
            WEBSEARCH_USER_ENDPOINT_REQUESTS_TOTAL.labels(
                endpoint=endpoint_label,
                user=user_label,
                status_class=status_class(status_code),
            ).inc()

    # --- CSRF ---
    try:
        validate_csrf(request)
    except HTTPException as e:
        WEBSEARCH_QUERY_ERRORS_TOTAL.labels(endpoint=endpoint_label, kind="csrf").inc()
        _count_user(e.status_code)
        raise

    prompt = (body.prompt or "").strip()
    if not prompt:
        WEBSEARCH_QUERY_ERRORS_TOTAL.labels(endpoint=endpoint_label, kind="bad_request").inc()
        _count_user(400)
        raise HTTPException(status_code=400, detail="No se ha proporcionado prompt para la búsqueda.")

    # --- Config ---
    try:
        default_top_k = int(os.getenv("WEBSEARCH_TOP_K", "4"))
    except ValueError:
        default_top_k = 4

    try:
        default_max_iters = int(os.getenv("WEBSEARCH_MAX_ITERS", "2"))
    except ValueError:
        default_max_iters = 2

    top_k = body.top_k or default_top_k
    max_iters = max(1, body.max_iters or default_max_iters)

    # Límite duro de rendimiento para legal_search: 1 iteración máxima.
    # Se aplica siempre (aunque el cliente/env pidan más) para mantener
    # latencia predecible en escenarios multiusuario concurrentes.
    LEGALSEARCH_MAX_ITERS_HARD_LIMIT = 1
    max_iters = min(max_iters, LEGALSEARCH_MAX_ITERS_HARD_LIMIT)

    # --- Session Redis ---
    search_session_id = body.search_session_id or uuid.uuid4().hex
    try:
        ws_session = await get_ws_session(user_id, search_session_id)
        if not ws_session:
            ws_session = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "queries": [],
            }
            await store_ws_session(user_id, search_session_id, ws_session)
    except Exception:
        WEBSEARCH_QUERY_ERRORS_TOTAL.labels(endpoint=endpoint_label, kind="redis").inc()
        _count_user(500)
        raise HTTPException(status_code=500, detail="Error de sesión Redis en web_search.")

    # --- Conversation DB ---
    conversation_id = body.conversation_id
    try:
        t_db = perf_counter()
        if conversation_id:
            conversation = (
                db.query(Conversation)
                .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
                .first()
            )
            WEBSEARCH_DB_LATENCY_SECONDS.labels(op="db_fetch_conversation").observe(perf_counter() - t_db)
        else:
            conversation = Conversation(
                user_id=user_id,
                conversation_text="",
                created_at=datetime.now(timezone.utc),
            )
            db.add(conversation)
            t_commit = perf_counter()
            db.commit()
            WEBSEARCH_DB_LATENCY_SECONDS.labels(op="db_commit_create_conversation").observe(perf_counter() - t_commit)
            db.refresh(conversation)
            conversation_id = conversation.id

        # persist USER message + conversation text
        if conversation.conversation_text:
            conversation.conversation_text += f"\nUSER: {prompt}"
        else:
            conversation.conversation_text = f"USER: {prompt}"

        user_msg = Message(
            conversation_id=conversation.id,
            sender="USER",
            content=prompt,
            created_at=datetime.now(timezone.utc),
        )
        db.add(user_msg)
        db.add(conversation)
        t_commit2 = perf_counter()
        db.commit()
        WEBSEARCH_DB_LATENCY_SECONDS.labels(op="db_commit_user_message").observe(perf_counter() - t_commit2)
        db.refresh(conversation)

    except Exception:
        WEBSEARCH_DB_ERRORS_TOTAL.labels(op="conversation_write").inc()
        WEBSEARCH_QUERY_ERRORS_TOTAL.labels(endpoint=endpoint_label, kind="db").inc()
        _count_user(500)
        raise HTTPException(status_code=500, detail="Error DB guardando conversación en web_search.")

    # --- Load history ---
    try:
        history = await get_ws_history(user_id, search_session_id)
    except Exception:
        WEBSEARCH_QUERY_ERRORS_TOTAL.labels(endpoint=endpoint_label, kind="redis").inc()
        _count_user(500)
        raise HTTPException(status_code=500, detail="Error leyendo historial Redis en web_search.")

    context_bundle = await _build_legal_user_context(
        user_id=user_id,
        search_session_id=search_session_id,
        conversation_id=conversation.id if conversation else conversation_id,
        file_ids=body.attached_file_ids or [],
    )
    user_context = context_bundle.get("context") or ""
    context_files = context_bundle.get("files") or []

    # Nota: no escalar iteraciones por complejidad; se mantiene límite duro = 1.

    # --- Orquestación CrewAI (LEGAL) ---
    crew_start = perf_counter()
    try:
        user_auth_header = request.headers.get("Authorization") or ""
        if not user_auth_header:
            raw = auth.get("access_token") or extract_raw_bearer_token(request)
            if raw:
                user_auth_header = f"Bearer {raw}"

        result = await asyncio.to_thread(
            orchestrator.run_legal_search,
            user_prompt=prompt,
            history=history,
            top_k=top_k,
            max_iters=max_iters,
            user_context=user_context,
            auth_token=user_auth_header,
        )
        
    except Exception as e:
        WEBSEARCH_QUERY_ERRORS_TOTAL.labels(endpoint=endpoint_label, kind="crew").inc()
        logger.exception("[web_search/query] Error ejecutando LegalSearchCrew: %s", e)
        _count_user(500)
        raise HTTPException(status_code=500, detail="Error generando la respuesta de búsqueda legal.")
    finally:
        WEBSEARCH_CREW_RUN_SECONDS.labels(top_k=str(top_k), max_iters=str(max_iters)).observe(perf_counter() - crew_start)

    # --- Métricas deep: pasos/agentes y tools (si el orquestador devuelve trazas) ---
    try:
        agent_steps = result.get("agent_steps") or result.get("steps") or []
        if isinstance(agent_steps, list):
            for s in agent_steps:
                if not isinstance(s, dict):
                    continue
                step = (s.get("step") or s.get("name") or "unknown").strip() or "unknown"
                dur = s.get("duration") or s.get("seconds")
                if dur is not None:
                    try:
                        WEBSEARCH_AGENT_STEP_DURATION_SECONDS.labels(step=step).observe(float(dur))
                    except Exception:
                        pass

        tool_events = result.get("tool_events") or result.get("tools") or result.get("tool_calls") or []
        if isinstance(tool_events, list):
            for ev in tool_events:
                if not isinstance(ev, dict):
                    continue
                tool = (ev.get("tool") or ev.get("name") or "unknown").strip() or "unknown"
                status = (ev.get("status") or "unknown").strip() or "unknown"
                WEBSEARCH_TOOL_CALLS_TOTAL.labels(tool=tool, status=status).inc()
                dur = ev.get("duration") or ev.get("seconds")
                if dur is not None:
                    try:
                        WEBSEARCH_TOOL_DURATION_SECONDS.labels(tool=tool).observe(float(dur))
                    except Exception:
                        pass
    except Exception:
        # No queremos romper la request por métricas.
        pass

    # --- Respuesta / normalización ---
    answer_text: str = clean_text(result.get("final_answer") or "")
    sources: List[Dict[str, Any]] = result.get("sources") or []
    normalized_queries: List[str] = result.get("normalized_queries") or []

    hitl_payload: Dict[str, Any] = {}
    if body.human_in_the_loop or body.require_human_approval:
        review_task_id = uuid.uuid4().hex
        task = {
            "review_task_id": review_task_id,
            "search_session_id": search_session_id,
            "conversation_id": conversation.id,
            "user_id": user_id,
            "status": "pending_review",
            "decision": None,
            "prompt": prompt,
            "draft_answer": answer_text,
            "sources": sources,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await _store_hitl_task(user_id, task)
        hitl_payload = {
            "enabled": True,
            "status": "pending_review",
            "review_task_id": review_task_id,
            "require_human_approval": bool(body.require_human_approval),
        }

    # Métricas de output
    WEBSEARCH_ANSWER_CHARS.observe(len(answer_text))
    WEBSEARCH_SOURCES_PER_ANSWER.observe(len(sources) if isinstance(sources, list) else 0)
    WEBSEARCH_NORMALIZED_QUERIES_PER_ANSWER.observe(len(normalized_queries) if isinstance(normalized_queries, list) else 0)

    # --- Background persistence con métricas ---
    def persist_bot_and_usage(convo_id: int, content: str, meta: Dict[str, Any]) -> None:
        t_task = perf_counter()
        status = "ok"
        db_local = next(get_db())
        try:
            t_db1 = perf_counter()
            bot_msg = Message(
                conversation_id=convo_id,
                sender="BOT",
                content=content,
                created_at=datetime.now(timezone.utc),
            )
            db_local.add(bot_msg)
            db_local.commit()
            db_local.refresh(bot_msg)
            WEBSEARCH_DB_LATENCY_SECONDS.labels(op="db_commit_bot_message").observe(perf_counter() - t_db1)

            t_db2 = perf_counter()
            usage = UsageLog(
                message_id=bot_msg.id,
                conversation_id=convo_id,
                # ✅ opcional: renombrar env var (sin romper, con fallback)
                model_name=os.getenv("LEGALSEARCH_MODEL_NAME", os.getenv("WEBSEARCH_MODEL_NAME", os.getenv("CREW_MODEL_NAME", "legalsearch-llm"))),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost=0.0,
                created_at=datetime.now(timezone.utc),
            )
            db_local.add(usage)

            audit = AuditLog(
                user_id=user_id,
                # ✅ opcional: entidad distinta para auditoría (sin romper si no la usas)
                entity_name=os.getenv("LEGALSEARCH_AUDIT_ENTITY", "LegalSearchAnswer"),
                entity_id=convo_id,
                action="CREATE",
                old_data=None,
                new_data={
                    "search_session_id": search_session_id,
                    "prompt": prompt,
                    "normalized_queries": meta.get("normalized_queries") or [],
                    "sources": meta.get("sources") or [],
                    "response": content,
                    # ✅ guardamos metadatos del plan si existen (sin romper)
                    "plan_meta": meta.get("plan_meta") or {},
                },
                timestamp=datetime.now(timezone.utc),
            )
            db_local.add(audit)
            db_local.commit()
            WEBSEARCH_DB_LATENCY_SECONDS.labels(op="db_commit_usage_audit").observe(perf_counter() - t_db2)

        except Exception:
            status = "error"
            WEBSEARCH_DB_ERRORS_TOTAL.labels(op="bg_persist").inc()
            logger.exception("[web_search/bg] Error persistiendo bot/usage/audit")
        finally:
            db_local.close()
            WEBSEARCH_BG_TASKS_TOTAL.labels(task="persist_bot_and_usage", status=status).inc()
            WEBSEARCH_DB_LATENCY_SECONDS.labels(op="bg_task_total").observe(perf_counter() - t_task)

    background_tasks.add_task(
        persist_bot_and_usage,
        conversation.id,
        answer_text,
        {
            "sources": sources,
            "normalized_queries": normalized_queries,
            "plan_meta": result.get("plan_meta") or {},
        },
    )

    # --- Persist BOT answer en conversation_text (en request-thread) ---
    try:
        t_db3 = perf_counter()
        if conversation.conversation_text:
            conversation.conversation_text += f"\nBOT: {answer_text}"
        else:
            conversation.conversation_text = f"BOT: {answer_text}"
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        WEBSEARCH_DB_LATENCY_SECONDS.labels(op="db_commit_conversation_append_bot").observe(perf_counter() - t_db3)
    except Exception:
        WEBSEARCH_DB_ERRORS_TOTAL.labels(op="conversation_append_bot").inc()
        WEBSEARCH_QUERY_ERRORS_TOTAL.labels(endpoint=endpoint_label, kind="db").inc()
        _count_user(500)
        raise HTTPException(status_code=500, detail="Error DB guardando respuesta en conversación.")

    # --- History append Redis ---
    history_entry = {
        "conversation_id": conversation.id,
        "user_id": user_id,
        "search_session_id": search_session_id,
        "prompt": prompt,
        "normalized_queries": normalized_queries,
        "response": answer_text,
        "sources": sources,
        # ✅ guardamos metadatos del plan legal si existen
        "plan_meta": result.get("plan_meta") or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        await append_ws_history(user_id, search_session_id, history_entry)
    except Exception:
        WEBSEARCH_QUERY_ERRORS_TOTAL.labels(endpoint=endpoint_label, kind="redis").inc()
        _count_user(500)
        raise HTTPException(status_code=500, detail="Error guardando historial Redis en web_search.")

    # --- File log best-effort ---
    try:
        log_websearch_interaction(
            user_id=user_id,
            search_session_id=search_session_id,
            conversation_id=conversation.id,
            entry=history_entry,
        )
    except Exception:
        pass

    # --- Final metrics ---
    total_duration = perf_counter() - start_total
    WEBSEARCH_QUERY_DURATION_SECONDS.labels(endpoint=endpoint_label).observe(total_duration)
    _count_user(200)

    return WebSearchQueryResponse(
        reply=answer_text,
        response=answer_text,
        conversation_id=conversation.id,
        search_session_id=search_session_id,
        sources=sources,
        normalized_queries=normalized_queries,
        plan_meta=result.get("plan_meta") or {},
        context_files=[
            {
                "filename": f.get("filename"),
                "file_id": (f.get("meta") or {}).get("file_id") if isinstance(f.get("meta"), dict) else None,
                "uploaded_at": f.get("uploaded_at"),
            }
            for f in context_files
        ],
        hitl=hitl_payload,
    )


def log_websearch_interaction(
    user_id: int,
    search_session_id: str,
    conversation_id: int,
    entry: Dict[str, Any],
) -> None:
    try:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        day_dir = LOGS_ROOT / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        filename = f"user_{user_id}_websearch_{search_session_id}_conv_{conversation_id}.log"
        log_path = day_dir / filename
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("No se pudo escribir log de web_search (%s): %s", LOGS_ROOT, e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("LEGALSEARCH_PORT", "8201")))

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends, BackgroundTasks, Response, Form
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
import os
import uvicorn
from pyngrok import ngrok
import tempfile
import logging
from typing import List, Optional, Dict, Any
import json
import uuid
from datetime import datetime
import asyncio
from sqlalchemy.orm import Session
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import redis.asyncio as aioredis
from types import SimpleNamespace
from mimetypes import guess_type
import httpx
from datetime import datetime, timezone
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from collections import OrderedDict
import re
from config.database import get_db
from config.models import (
    Conversation, Message, Attachment, AuditLog,
    File as FileModel, UsageLog,
    ConversationMeta, ConversationHistory, Session as DBSession
)
from utils import (
    # --- Constantes / configuración compartida ---
    STORAGE_ROOT,
    TALK_TO_DOCUMENT_URL,
    NLP_UPLOAD_FILE_URL,
    ROLE_SUPERVISOR,
    ROLE_USER,

    # --- Métricas Prometheus ---
    REQUEST_COUNT,
    HTTP_REQUEST_DURATION,
    UPLOADFILE_REQUESTS,
    UPLOADFILE_ERRORS,
    UPLOADFILE_DURATION,

    llm_query_counter,
    llm_query_requests_by_status_total,
    llm_query_latency,

    # --- Helpers varios ---
    allowed_file,
    clean_text,
    perform_rag_search,

    # --- NUEVOS HELPERS COMPARTIDOS ---
    generate_csrf_token,
    validate_csrf_double_submit,
    get_or_create_conversation,
    init_redis_clients,
    save_conversation_to_redis,
    add_ephemeral_file,
    get_ephemeral_files,
    get_conversation_to_redis,
    reset_conversation_in_redis,
    set_user_context,
    get_current_auth,
    scan_with_clamav,
    _resolve_notetaker_identity_like_me,
    _build_notetaker_url

)
from crew_orchestrator import CosmosCrewOrchestrator
from files_on_fly import process_ephemeral_file, extract_text_from_image_file
from cosmos_crew_src.agents import create_rag_router_agent
from cosmos_crew_src.prompts import build_rag_router_prompt
logging.basicConfig(level=logging.DEBUG)
 
BASE_DIR = Path(__file__).resolve().parent
LOGS_ROOT = Path(
    os.getenv("COSMOS_BIZ_LOGS_DIR", str(BASE_DIR / "logs"))
).resolve()

# URL interna para hablar de backend a backend (docker network)
AUTH_SSO_INTERNAL_BASE = os.getenv(
    "AUTH_SSO_INTERNAL_BASE",
    "http://auth_sso:7100",  # en docker-compose, nombre de servicio del micro de SSO
).rstrip("/")

# URL pública de auth_sso (la que verá el navegador); suele coincidir con lo que pasas como auth_origin
AUTH_SSO_PUBLIC_BASE = os.getenv(
    "AUTH_SSO_PUBLIC_BASE",
    AUTH_SSO_INTERNAL_BASE,
).rstrip("/")
_USER_LLM_LOCKS: "OrderedDict[str, asyncio.Lock]" = OrderedDict()
_USER_LLM_LOCKS_MAX = int(os.getenv("LLM_USER_LLM_LOCKS_MAX", "5000"))

_ATTACH_PAT = re.compile(
    r"\b(adjunt[oa]s?|anexo|en\s+el\s+adjunto|archivo\s+adjunto|documento\s+adjunto|"
    r"te\s+lo\s+acabo\s+de\s+subir|acabo\s+de\s+subir|he\s+subido|archivo\s+subido|"
    r"este\s+(pdf|doc|docx|excel|xlsx)|documento\s+subido)\b",
    re.IGNORECASE,
)

_INTERNAL_DOCS_PAT = re.compile(
    r"\b(documentaci[oó]n\s+interna|inventario\s+corporativo|pol[ií]tica\s+interna|"
    r"procedimiento\s+interno|busca\s+en\s+los\s+documentos|en\s+la\s+base\s+de\s+conocimiento)\b",
    re.IGNORECASE,
)

_FILE_NAME_PAT = re.compile(r"\b[\w\-]+\.(pdf|docx|xlsx|pptx|csv|txt)\b", re.IGNORECASE)


def _wants_ephemeral(prompt: str, attached_file_ids: List[str]) -> bool:
    return bool(attached_file_ids) or bool(_ATTACH_PAT.search(prompt or ""))


def _wants_internal_docs(prompt: str) -> bool:
    return bool(_INTERNAL_DOCS_PAT.search(prompt or ""))


def _mentions_named_file(prompt: str) -> bool:
    return bool(_FILE_NAME_PAT.search(prompt or ""))


class QueryRequest(BaseModel):
    message: Optional[str] = None
    prompt: Optional[str] = None
    files: Optional[List[str]] = None
    choice: Optional[str] = None
    conversation_id: Optional[int] = None
    department_directory: Optional[str] = None
 
class ImageRequest(BaseModel):
    image: str
 
class TranscriptionRequest(BaseModel):
    text: str

class ImagePromptRequest(BaseModel):
    prompt: str
    context: Optional[str] = ""
    image_data: str

class ContextRequest(BaseModel):
    context: str

class InternetRequest(BaseModel):
    query: str

class PromptRequest(BaseModel):
    prompt: str

class MessageOut(BaseModel):
    id: int
    sender: str
    content: str
    created_at: datetime
    is_liked: Optional[bool] = None

    class Config:
        orm_mode = True

class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: datetime
    is_favorite: bool = False

    class Config:
        orm_mode = True

class ConversationDetail(BaseModel):
    id: int
    created_at: datetime
    messages: List[MessageOut]

    class Config:
        orm_mode = True

class MeResponse(BaseModel):
    user_id: int
    username: str
    email: str
    role_id: int
    role_name: str
    departments: List[Dict[str, str]] = []

    class Config:
        orm_mode = False

class ResetContextResponse(BaseModel):
    ok: bool

class FeedBackUpdate(BaseModel):
    is_liked: Optional[bool] = None

class ConversationFavoriteUpdate(BaseModel):
    is_favorite: bool

llm_query_requests = llm_query_requests_by_status_total

CSRF_COOKIE_NAME = "csrftoken_app"
AUTH_ORIGIN = os.getenv("COSMOS_AUTH_ORIGIN", "http://localhost:7000")
AUTH_SSO_INTERNAL_BASE = "http://localhost:7100"
FRONTEND_URL = os.getenv("COSMOS_FRONTEND_URL", "http://localhost").rstrip("/")

logging.info("\n\n(fastapiAPP - main)  Configurando entorno de servidor y endpoints...")

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")
crew_orchestrator = CosmosCrewOrchestrator()
ALLOWED_ORIGINS_ENV = os.getenv("COSMOS_ALLOWED_ORIGINS", "")
if ALLOWED_ORIGINS_ENV:
    allowed_origins = [
        o.strip()
        for o in ALLOWED_ORIGINS_ENV.split(",")
        if o.strip()
    ]
else:
    # Valores por defecto seguros para desarrollo
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        AUTH_ORIGIN,  # p.ej. http://localhost:7000 (SSO / UI)
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,              # para que viajen cookies (SSO + CSRF)
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],                 # X-CSRFToken, Authorization, etc.
)


logging.info("\n\n(fastapiAPP - main)  Preparando renderizado y configuración de la GUI principal de COSMOS...")
templates = Jinja2Templates(directory="./templates")

logger = logging.getLogger("modelo_negocio")
logger.setLevel(logging.INFO)


def _get_user_llm_lock(user_id: str) -> asyncio.Lock:
    """
    Lock por user para evitar carreras intra-proceso (dos requests simultáneas del mismo usuario)
    que pisan Redis/historial/persistencia.
    """
    uid = str(user_id or "")
    lock = _USER_LLM_LOCKS.get(uid)
    if lock is None:
        lock = asyncio.Lock()
        _USER_LLM_LOCKS[uid] = lock
    else:
        _USER_LLM_LOCKS.move_to_end(uid)

    if len(_USER_LLM_LOCKS) > _USER_LLM_LOCKS_MAX:
        _USER_LLM_LOCKS.popitem(last=False)

    return lock

def validate_csrf(request: Request) -> None:
    """
    Wrapper fino alrededor del validador CSRF de utils
    usando el patrón double-submit cookie:

      - Cookie: csrftoken_app
      - Cabecera: X-CSRFToken
    """
    validate_csrf_double_submit(
        request,
        cookie_name=CSRF_COOKIE_NAME,
        header_name="X-CSRFToken",
        error_detail="CSRF token inválido o ausente en servicio principal.",
    )

def log_llm_interaction(
    user_id: int,
    session_id: Optional[int],
    entry: Dict[str, Any],
) -> None:
    """
    Registra una interacción LLM en un fichero de log por usuario+sesión,
    dentro de un subdirectorio por día (UTC).

    Formato:
      logs/YYYY-MM-DD/user_<user_id>_session_<session_id>.log

    Cada línea es un JSON independiente (JSONL), fácil de parsear
    para auditoría/forénsica.
    """
    try:
        # Directorio por día (UTC)
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        day_dir = LOGS_ROOT / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        session_part = str(session_id) if session_id is not None else "no-session"
        filename = f"user_{user_id}_session_{session_part}.log"
        log_path = day_dir / filename

        # Escribimos una línea JSON por interacción
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception as e:
        # Nunca rompemos el flujo de negocio por fallo de logging
        logging.warning("No se pudo escribir en log de sesión (%s): %s", LOGS_ROOT, e)


@app.get("/metrics")
def metrics():
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

# Asegúrate de que CSRF_COOKIE_NAME y generate_csrf_token ya existen
# CSRF_COOKIE_NAME = "csrftoken_app"  (por ejemplo)

@app.get("/csrf-token", response_model=dict)
async def get_csrf_token_endpoint(request: Request, response: Response):
    """
    Devuelve un token CSRF y lo deja también en una cookie.

    IMPORTANTE:
      - Si el cliente ya traía un csrftoken_app en las cookies, se reutiliza
        el mismo valor para evitar desincronizar cookie y cabecera cuando
        hay varias pestañas o múltiples llamadas a /csrf-token.
    """
    # 1) Intentar reutilizar el token ya existente en la cookie
    token = request.cookies.get(CSRF_COOKIE_NAME)

    # 2) Si no había cookie, generamos uno nuevo
    if not token:
        token = generate_csrf_token()

    # 3) Reescribimos la cookie con el mismo valor (refresco de atributos)
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        httponly=False,   # el front React puede leerlo si quiere
        secure=False,     # en prod: True + HTTPS
        samesite="Lax",   # o "Strict" según tu política
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

    if method == "POST" and endpoint == "/query/llm":
        # contador único definido arriba
        llm_query_counter.inc()

        status = "success" if response.status_code < 400 else "error"
        llm_query_requests_by_status_total.labels(status=status).inc()

        llm_query_latency.observe(duration)

    return response


@app.on_event("startup")
async def initialize_redis():
    """
    Inicializa la conexión a Redis cuando la aplicación arranca.
    Usa las mismas variables de entorno que el resto de servicios.

    Registra los clientes en utils.init_redis_clients para que
    los helpers compartidos puedan usarlos.
    """
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    core_client = aioredis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True,
    )
    conv_client = aioredis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True,
        db=2,
    )

    # Registramos los clientes en utils (única fuente de verdad)
    init_redis_clients(core_client, conv_client)

    logging.info(
        f"[startup] Redis conectado en {redis_host}:{redis_port} (db=0 y db=2) y registrado en utils."
    )


@app.post("/start_chat")
async def start_chat():
    """
    Recibe el POST del formulario y redirige al chat de documentos.
    """
    return RedirectResponse(url=TALK_TO_DOCUMENT_URL, status_code=302)


@app.post("/back_upload_file")
async def upload_file_post():
    """
    POST: redirige con 302 al endpoint de subida de ficheros en el servicio NLP.
    """
    return RedirectResponse(url=NLP_UPLOAD_FILE_URL, status_code=302)


@app.get("/chat_cosmos_principal", include_in_schema=False)
async def read_root(auth: dict = Depends(get_current_auth)):
    """
    Raíz del microservicio de modelo de negocio (8000).

    - Ya NO sirve HTML ni la SPA.
    - La SPA vive detrás de Nginx en COSMOS_FRONTEND_URL (por ejemplo http://localhost/).
    - Este endpoint se usa como "healthcheck autenticado":
        * comprueba JWT y sesión unívoca en Redis (get_current_auth)
        * devuelve un JSON sencillo de estado.

    Seguridad:
      - JWT validado por verify_token (en utils).
      - Sesión en Redis validada y ligada al mismo JWT (sesión unívoca).
    """
    user_id = auth.get("user_id")
    logging.info(
        "(modeloNegocio - GET /) Healthcheck autenticado OK para user_id=%s",
        user_id,
    )

    return JSONResponse(
        {
            "service": "cosmos-modelo-negocio",
            "status": "ok",
            "user_id": user_id,
            "frontend_url": FRONTEND_URL,
        }
    )


@app.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    auth: dict = Depends(get_current_auth),
    db: Session = Depends(get_db),
):
    """
    Devuelve TODAS las conversaciones del usuario autenticado,
    QUE NO estén eliminadas,
    para poblar la barra lateral (Historial).

    Seguridad:
      - Usa get_current_auth → JWT válido + sesión unívoca en Redis.
    """
    user_id = auth["user_id"]

    # Sacamos todas las conversaciones del usuario (sin LIMIT artificial)
    convos = (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .filter(Conversation.is_deleted == False)
        .order_by(Conversation.created_at.desc())
        .all()
    )

    logging.info(
        "list_conversations: user_id=%s → %d conversaciones devueltas",
        user_id,
        len(convos),
    )

    summaries: List[ConversationSummary] = []

    for c in convos:
        raw = c.conversation_text or ""

        # Tomamos la primera línea NO vacía como título
        title = ""
        if raw:
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    title = line
                    break

        if not title:
            title = f"Conversación {c.id}"

        if len(title) > 80:
            title = title[:77] + "..."

        summaries.append(
            ConversationSummary(
                id=c.id,
                title=title,
                created_at=c.created_at,
                is_favorite=c.is_favorite,
            )
        )

    return summaries

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    auth: dict = Depends(get_current_auth),
    db: Session = Depends(get_db)
):
    """
    Marca solo una conversación como si estuviera eliminada (hacemos un Soft Delete).
    Solo permite al usuario borrar si la conversación pertenece al usuario autenticado
    """

    user_id = auth["user_id"]

    # Buscamos la conversación para asegurarnos de que pertenece al usuario
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .filter(Conversation.user_id == user_id)
        .first()
    )

    if not conversation:
        # Si la conversación no existe o no es del usuario, devoolvemos un error 404
        logging.warning(
            "delete_conversation: Intento de borrar ID=%s fallido. User_id=%s. No encontrada o sin permisos.",
            conversation_id, user_id
        )
        raise HTTPException(status_code=404, detail="Conversación no encontrada o acceso denegado.")
    
    # Si existe o tiene permisos, aplicamos el Soft Delete
    conversation.is_deleted = True

    try:
        db.commit()
        logging.info(
            "delete_conversation: User_id=%s eliminó (soft) la conversación ID=%s",
            user_id, conversation_id
        )
    except Exception as e:
        db.rollback()
        logging.error("Error en DB al borrar la conversación: %s", e)
        raise HTTPException(status_code=500, detail="Error interno al eliminar la conversación.")

    return {"message": "Conversación eliminada correctamente", "id": conversation_id} 

@app.put("/messages/{message_id}/feedback")
async def rate_message(
    message_id: int,
    feedback: FeedBackUpdate,
    auth: dict = Depends(get_current_auth),
    db: Session = Depends(get_db)
):
    """
    Actualiza el estado de la variable is_liked, que mide si el
    mensaje le ha gustado o no al usuario de un mensaje específico
    """
    
    user_id = auth["user_id"]

    # Primero buscamos el mensaje para comprobar que pertenece al usuario actual
    # Hacemos también un JOIN con Conversation para comprobar el user_id
    message = (
        db.query(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .filter(Message.id == message_id)
        .filter(Conversation.user_id == user_id)
        .first()
    )

    if not message:
        logging.warning(f"Intento de feedback en mensaje {message_id} fallido. Usuario: {user_id}")
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")
    
    # Para que el usuario no pueda dar like a sus propios mensajes
    if message.sender == 'USER' or message.sender == 'SYSTEM':
        raise HTTPException(status_code=400, detail="No puedes valorar tus propios mensajes")
    
    # Por último, actualizamos el campo y lo guardamos
    message.is_liked = feedback.is_liked
    try:
        db.commit()
        
        logging.info(f"Feedback actualizado: Msg {message_id} -> {feedback.is_liked}")
        
        return {
            "message": "Feedback actualizado correctamente",
            "id": message.id,
            "is_liked": message.is_liked
        }
        
    except Exception as e:
        db.rollback()
        logging.error(f"Error guardando feedback: {e}")
        raise HTTPException(status_code=500, detail="Error interno al guardar el feedback")
    
@app.patch("/conversations/{conversation_id}/favorite")
async def toggle_favorite_conversation(
    conversation_id: int,
    update_data: ConversationFavoriteUpdate,
    auth: dict = Depends(get_current_auth),
    db: Session = Depends(get_db),
):
    user_id = auth["user_id"]

    # Buscamos la conversación del usuario
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .filter(Conversation.user_id == user_id)
        .first()
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")

    # Actualizamos el valor
    conversation.is_favorite = update_data.is_favorite
    
    db.commit()
    
    return {"message": "Estado de favorito actualizado", "is_favorite": conversation.is_favorite}


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    auth: dict = Depends(get_current_auth),
    db: Session = Depends(get_db),
):
    """
    Devuelve todos los mensajes de una conversación del usuario.

    Se valida:
      - JWT (via verify_token).
      - Sesión en Redis.
      - Coincidencia de token (sesión unívoca).
    """
    user_id = auth["user_id"]

    convo = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .first()
    )
    if not convo:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")

    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    return ConversationDetail(
        id=convo.id,
        created_at=convo.created_at,
        messages=[
            MessageOut(
                id=m.id,
                sender=m.sender,
                content=m.content,
                created_at=m.created_at,
                is_liked=m.is_liked,
            )
            for m in msgs
        ],
    )

@app.post("/uploadfile/", response_model=dict)
async def create_upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    conversation_id: Optional[int] = Form(None),
    auth: dict = Depends(get_current_auth),
    db: Session = Depends(get_db),
):
    """
    Sube uno o varios archivos y los procesa como 'archivos en vuelo'.

     IMPORTANTE (fix de mezcla / desalineación):
    - Los archivos en vuelo y el user_context se guardan en Redis SCOPED por conversation_id:
        ephemeral:{user_id}:{conversation_id}
        context:{user_id}:{conversation_id}
    - Si el front no envía conversation_id, se crea una conversación nueva (comportamiento legacy).
    """
    UPLOADFILE_REQUESTS.inc()
    start = time.time()

    validate_csrf(request)

    logging.info("(main - /uploadfile) Validando token y permisos para adjuntar archivos...")

    try:
        user_id = auth["user_id"]
        user_role = auth["role"]
        session_data = auth["session"]

        if session_data is None:
            raise HTTPException(status_code=403, detail="Sesión no encontrada o expirada.")

        user_directory = session_data.get("user_directory")  # p.ej. 'users/email@empresa.com'

        if user_role not in [ROLE_SUPERVISOR, ROLE_USER]:
            raise HTTPException(status_code=403, detail="Permisos insuficientes para cargar archivos.")

        #  Conversación: reutiliza si el front la pasa; si no, crea una nueva (sin romper)
        created_conversation = False
        conversation: Optional[Conversation] = None

        if conversation_id:
            conversation = (
                db.query(Conversation)
                .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
                .first()
            )
            if conversation is None:
                logging.warning(
                    "(main - /uploadfile) conversation_id=%s no encontrada para user_id=%s. Creando nueva.",
                    conversation_id,
                    user_id,
                )

        if conversation is None:
            conversation = Conversation(
                user_id=user_id,
                conversation_text="",
                created_at=datetime.now(timezone.utc),
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            created_conversation = True

        logging.info(
            "(main - /uploadfile) Usando conversation_id=%s (created=%s) para user_id=%s",
            conversation.id,
            created_conversation,
            user_id,
        )

        responses: List[Dict[str, Any]] = []
        processed_files: List[str] = []
        ephemeral_ids: List[str] = []

        # Directorio físico donde se guardan los ficheros del usuario
        user_root = STORAGE_ROOT / user_directory
        save_dir = user_root / "uploaded_files"
        save_dir.mkdir(parents=True, exist_ok=True)

        max_file_mb = int(os.getenv("EPHEMERAL_MAX_FILE_MB", "25"))
        max_file_bytes = max_file_mb * 1024 * 1024

        conversation_snippets: List[str] = []
        snippet_limit = int(os.getenv("EPHEMERAL_CONVERSATION_SNIPPET_CHARS", "2000"))

        for upload in files:
            filename = upload.filename or "uploaded_file"
            processed_files.append(filename)
            logging.info(f"(main - /uploadfile) Procesando {filename}...")

            file_result: Dict[str, Any] = {
                "filename": filename,
                "file_id": None,
                "message": None,
                "error": None,
                "av_status": None,
                "av_virus": None,
                "conversation_id": conversation.id,  # útil para el front/debug
            }

            if not allowed_file(filename):
                msg = "Formato no soportado"
                logging.warning("(main - /uploadfile) Archivo rechazado por extensión: %s", filename)
                file_result["error"] = msg
                file_result["av_status"] = "SKIPPED_UNSUPPORTED_TYPE"
                responses.append(file_result)

                audit = AuditLog(
                    user_id=user_id,
                    entity_name="EphemeralFileScan",
                    entity_id=conversation.id,
                    action="CREATE",
                    old_data=None,
                    new_data={
                        "filename": filename,
                        "reason": "unsupported_extension",
                        "av_status": "SKIPPED_UNSUPPORTED_TYPE",
                        "conversation_id": conversation.id,
                    },
                    timestamp=datetime.now(timezone.utc),
                )
                db.add(audit)
                db.commit()
                continue

            ext = filename.rsplit(".", 1)[-1].lower()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")

            try:
                content = await upload.read()

                if len(content) > max_file_bytes:
                    msg = (
                        f"El archivo supera el tamaño máximo permitido "
                        f"({max_file_mb} MB) para procesamiento en vuelo."
                    )
                    logging.warning(
                        "(main - /uploadfile) %s (archivo=%s, tamaño=%.2f MB)",
                        msg,
                        filename,
                        len(content) / (1024 * 1024),
                    )
                    file_result["error"] = msg
                    file_result["av_status"] = "SKIPPED_TOO_LARGE"

                    audit = AuditLog(
                        user_id=user_id,
                        entity_name="EphemeralFileScan",
                        entity_id=conversation.id,
                        action="CREATE",
                        old_data=None,
                        new_data={
                            "filename": filename,
                            "reason": "file_too_large",
                            "limit_mb": max_file_mb,
                            "size_mb": round(len(content) / (1024 * 1024), 2),
                            "av_status": "SKIPPED_TOO_LARGE",
                            "conversation_id": conversation.id,
                        },
                        timestamp=datetime.now(timezone.utc),
                    )
                    db.add(audit)
                    db.commit()

                    responses.append(file_result)
                    continue

                # 1) AV
                av_result = await scan_with_clamav(content, filename=filename)
                file_result["av_status"] = av_result.get("status")
                file_result["av_virus"] = av_result.get("virus_name")

                audit_av = AuditLog(
                    user_id=user_id,
                    entity_name="EphemeralFileScan",
                    entity_id=conversation.id,
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
                        "conversation_id": conversation.id,
                    },
                    timestamp=datetime.now(timezone.utc),
                )
                db.add(audit_av)
                db.commit()

                if av_result.get("status") == "INFECTED":
                    virus_name = av_result.get("virus_name") or "malware"
                    msg = f"El archivo ha sido bloqueado por el antivirus (detectado: {virus_name})."
                    logging.warning(
                        "(main - /uploadfile) Archivo infectado bloqueado: %s (%s)",
                        filename,
                        virus_name,
                    )
                    file_result["error"] = msg
                    responses.append(file_result)
                    continue

                if av_result.get("status") == "ERROR":
                    msg = (
                        "No se ha podido analizar el archivo con el sistema antivirus. "
                        "Por seguridad, el archivo ha sido rechazado. Inténtalo más tarde."
                    )
                    logging.error(
                        "(main - /uploadfile) Error AV, archivo rechazado: %s | %s",
                        filename,
                        av_result.get("error"),
                    )
                    file_result["error"] = msg
                    responses.append(file_result)
                    continue

                # 2) Guardar temporal y extraer texto
                tmp.write(content)
                tmp.flush()

                guessed_mime, _ = guess_type(filename)
                content_type = upload.content_type or guessed_mime or "application/octet-stream"

                if content_type.startswith("image/"):
                    logging.info(
                        "(main - /uploadfile) Detectada imagen, usando OCR para %s (mime=%s)",
                        filename,
                        content_type,
                    )
                    try:
                        text = extract_text_from_image_file(
                            file_path=tmp.name,
                            filename=filename,
                            mime_type=content_type,
                        )
                        extraction_result = SimpleNamespace(
                            text=text,
                            metadata={"source": "ephemeral_image_upload", "mime_type": content_type},
                        )
                    except Exception as e:
                        logging.error(
                            "(main - /uploadfile) Error OCR para imagen %s: %s",
                            filename,
                            e,
                            exc_info=True,
                        )
                        file_result["error"] = "No se pudo extraer texto de la imagen mediante OCR."
                        responses.append(file_result)
                        continue
                else:
                    try:
                        extraction_result = process_ephemeral_file(
                            file_path=tmp.name,
                            filename=filename,
                            mime_type=content_type,
                        )
                    except Exception as e:
                        logging.error("(main - /uploadfile) Error extrayendo texto de %s: %s", filename, e)
                        file_result["error"] = f"No se pudo extraer texto del archivo: {e}"
                        responses.append(file_result)
                        db.rollback()
                        continue

                text = (extraction_result.text or "").strip()
                if not text:
                    file_result["error"] = "No se pudo extraer texto del archivo."
                    responses.append(file_result)
                    continue

                # ID efímero para el archivo
                file_id = str(uuid.uuid4())
                file_result["file_id"] = file_id
                ephemeral_ids.append(file_id)

                #  Redis: scoped por conversación (FIX CLAVE)
                meta = extraction_result.metadata or {}
                meta = {
                    **meta,
                    "file_id": file_id,
                    "original_filename": filename,
                    "conversation_id": conversation.id,
                }

                # Contexto “último texto” (si lo usas en algún sitio): también scoped
                await set_user_context(user_id, text, ttl=3600, conversation_id=conversation.id)

                # Lista de archivos en vuelo: scoped
                await add_ephemeral_file(
                    user_id=user_id,
                    filename=filename,
                    text=text,
                    ttl=3600,
                    meta=meta,
                    conversation_id=conversation.id,
                )

                # Snippet para conversation_text (BD)
                snippet = text[:snippet_limit]
                if len(text) > snippet_limit:
                    snippet += "..."
                conversation_snippets.append(f"--- {filename} ---\n{snippet}")

                # Mensaje de usuario en BD (log de extracción)
                user_msg = Message(
                    conversation_id=conversation.id,
                    sender="USER",
                    content=f"Extracción de {filename}: {snippet}",
                    created_at=datetime.now(timezone.utc),
                )
                db.add(user_msg)
                db.flush()

                # Guardar archivo original en disco (nota: si el mismo filename se sube varias veces, sobrescribe)
                stored_path = save_dir / filename
                with open(stored_path, "wb") as dest:
                    dest.write(content)

                # Registro en tabla files
                file_rec = FileModel(
                    user_id=user_id,
                    department_id=None,
                    file_path=str(stored_path),
                    file_name=filename,
                    permission="READ",
                    created_at=datetime.now(timezone.utc),
                )
                db.add(file_rec)
                db.flush()

                # Attachment
                attachment = Attachment(
                    message_id=user_msg.id,
                    file_id=file_rec.id,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(attachment)
                db.commit()

                file_result["message"] = "Archivo procesado y registrado correctamente (en vuelo, no indexado)."
                responses.append(file_result)

            except Exception as e:
                logging.error(f"(main - /uploadfile) Error procesando {filename}: {e}", exc_info=True)
                responses.append({"filename": filename, "error": str(e), "conversation_id": conversation.id})
                db.rollback()
            finally:
                tmp.close()
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        #  Actualizar conversation_text SIN pisar si ya tenía texto (cuando reusas conversation_id)
        new_block = "\n\n".join(conversation_snippets).strip()
        if new_block:
            if conversation.conversation_text:
                conversation.conversation_text = (conversation.conversation_text + "\n\n" + new_block).strip()
            else:
                conversation.conversation_text = new_block

        db.add(conversation)
        db.commit()

        bot_response = (
            "Todos los archivos válidos han sido procesados; ya tienes el contexto en vuelo "
            "disponible en esta conversación.")
        bot_msg = Message(
            conversation_id=conversation.id,
            sender="BOT",
            content=bot_response,
            created_at=datetime.now(timezone.utc),
        )
        db.add(bot_msg)
        db.commit()

        result = {
            "message": bot_response,
            "details": responses,
            "conversation_id": conversation.id,
            "ephemeral_file_ids": ephemeral_ids,
        }

        audit = AuditLog(
            user_id=user_id,
            entity_name="Conversation",
            entity_id=conversation.id,
            action="UPDATE",
            old_data=None,
            new_data={
                "files": processed_files,
                "responses": responses,
                "ephemeral_max_file_mb": max_file_mb,
                "ephemeral_file_ids": ephemeral_ids,
                "conversation_id": conversation.id,
            },
            timestamp=datetime.now(timezone.utc),
        )
        db.add(audit)
        db.commit()

        return result

    except Exception:
        UPLOADFILE_ERRORS.inc()
        raise

    finally:
        duration = time.time() - start
        UPLOADFILE_DURATION.labels(method="POST", endpoint="/uploadfile/").observe(duration)



@app.post("/query/llm", response_model=dict)
async def handle_query_to_llm(
    request: Request,
    background_tasks: BackgroundTasks,
    query: QueryRequest,
    auth: dict = Depends(get_current_auth),
    db: Session = Depends(get_db),
):
    import uuid

    start_time = time.time()
    status_label = "success"
    request_id = uuid.uuid4().hex[:8]

    validate_csrf(request)

    session_data = auth["session"]
    user_id = auth["user_id"]
    access_token = auth["access_token"]

    try:
        logging.info(f"[LLM_ENTRY {request_id}] (main - /query/llm) Extrayendo el prompt de usuario...")

        raw_prompt = query.prompt or query.message
        prompt = (raw_prompt or "").strip()
        attached_file_ids = query.files or []

        requested_dept_dir = getattr(query, "department_directory", None)
        if requested_dept_dir:
            logging.info("[LLM_ENTRY %s] department_directory solicitado: %s", request_id, requested_dept_dir)

        if not prompt:
            status_label = "error"
            msg = "\n\n(main - /query/llm) No prompt provided"
            return {"error": msg, "reply": msg, "response": msg, "is_choice_needed": False}

        if not user_id:
            status_label = "error"
            raise HTTPException(status_code=401, detail="Token inválido")

        if not session_data:
            status_label = "error"
            raise HTTPException(status_code=403, detail="Sesión expirada o no encontrada")

        if not access_token:
            status_label = "error"
            raise HTTPException(status_code=403, detail="Token de sesión no disponible")

        flow = (query.choice or "C").upper()
        logging.info(f"[LLM_ENTRY {request_id}] Flow={flow}. Prompt len={len(prompt)}")

        # ---------------------------------------------------------
        # 0) Determinar conversation_id efectivo (robustez adjuntos)
        # ---------------------------------------------------------
        effective_conversation_id: Optional[int] = query.conversation_id

        # Si hay adjuntos pero el front no manda conversation_id,
        # intentamos recuperar el current_conversation_id desde Redis.
        wants_files_early = _wants_ephemeral(prompt, attached_file_ids)
        if effective_conversation_id is None and wants_files_early:
            try:
                # OJO: usa keyword para no confundir expires_in
                state = await get_conversation_to_redis(user_id=user_id, conversation_id=None)
                cand = state.get("current_conversation_id")
                if cand is not None:
                    effective_conversation_id = int(cand)
                    logging.info(
                        "[LLM_ENTRY %s] Inferido conversation_id=%s desde Redis (por adjuntos).",
                        request_id,
                        effective_conversation_id,
                    )
            except Exception as e:
                logging.debug("[LLM_ENTRY %s] No se pudo inferir conversation_id desde Redis: %s", request_id, e)

        # -------------------------------------------
        # 1) Obtener o crear conversación (BD)
        # -------------------------------------------
        conversation, created = get_or_create_conversation(
            db=db,
            user_id=user_id,
            conversation_id=effective_conversation_id,
        )

        logging.info(
            f"[LLM_ENTRY {request_id}] Conversación {'creada' if created else 'recuperada'} "
            f"(conversation_id={conversation.id}, user_id={user_id})"
        )

        # -------------------------------------------
        # 2) Historial de conversación desde Redis (SCOPED)
        # -------------------------------------------
        if created:
            # FIX CRÍTICO: usar keyword conversation_id (si no, se interpreta como expires_in)
            try:
                conversation_data = await reset_conversation_in_redis(
                    user_id=user_id,
                    conversation_id=conversation.id,
                )
            except Exception:
                conversation_data = {"conversation_history": []}
        else:
            # FIX CRÍTICO: conversation.id debe ir en conversation_id=
            conversation_data = await get_conversation_to_redis(
                user_id=user_id,
                conversation_id=conversation.id,
            ) or {"conversation_history": []}

        logging.info(
            f"[LLM_ENTRY {request_id}] Historial Redis (conv={conversation.id}): "
            f"{len((conversation_data or {}).get('conversation_history', []))} entradas"
        )

        # -------------------------------------------
        # 3) Persistencia del mensaje USER en BD
        # -------------------------------------------
        conversation.conversation_text = (
            (conversation.conversation_text + f"\nUSER: {prompt}").strip()
            if conversation.conversation_text
            else f"USER: {prompt}"
        )

        user_message = Message(
            conversation_id=conversation.id,
            sender="USER",
            content=prompt,
            created_at=datetime.now(timezone.utc),
        )
        db.add(user_message)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        # -------------------------------------------
        # 4) Planner (CrewAI)
        # -------------------------------------------
        user_ctx = {
            "user_id": user_id,
            "access_token": access_token,
            "flow": flow,
            "department_directory": requested_dept_dir,
            "conversation_id": conversation.id,
            "attached_file_ids": attached_file_ids,
        }

        logging.info(f"[LLM_ENTRY {request_id}] Llamando al planner...")
        plan = await asyncio.to_thread(
            crew_orchestrator.plan_query,
            prompt,
            conversation_data,
            user_ctx,
        )

        logging.info(f"[LLM_ENTRY {request_id}] Plan planner: {plan}")

        # -------------------------------------------
        # 4.1) Heurística determinista anti-fallos planner
        # -------------------------------------------
        wants_files = _wants_ephemeral(prompt, attached_file_ids)
        wants_internal = _wants_internal_docs(prompt)
        named_file = _mentions_named_file(prompt)

        if wants_files:
            plan["needs_files"] = True
            if not wants_internal:
                plan["needs_rag"] = False

        if named_file and not wants_files:
            plan["needs_rag"] = True
            plan["needs_files"] = False

        normalized_prompt = plan.get("normalized_question") or prompt
        rag_query = plan.get("rag_query") or normalized_prompt
        intent = plan.get("intent") or "otro"
        needs_rag = bool(plan.get("needs_rag", True))
        needs_files = bool(plan.get("needs_files", False))
        needs_web = bool(plan.get("needs_web", False))
        filters = plan.get("filters") or {}

        logging.info(
            f"[LLM_ENTRY {request_id}] Plan final: intent={intent}, needs_rag={needs_rag}, "
            f"needs_files={needs_files}, needs_web={needs_web}, filters={filters}"
        )

        # -------------------------------------------
        # 5) Archivos en vuelo (SCOPED + filtrado por IDs)
        # -------------------------------------------
        ephemeral_files: List[Dict[str, Any]] = []
        if needs_files:
            try:
                ephemeral_files = await get_ephemeral_files(
                    user_id=user_id,
                    conversation_id=conversation.id,
                    file_ids=attached_file_ids or None,
                )
                logging.info(
                    f"[LLM_ENTRY {request_id}] Ephemeral (conv={conversation.id}): {len(ephemeral_files)} archivos"
                )
            except Exception as e:
                logging.warning(f"[LLM_ENTRY {request_id}] No se pudieron recuperar efímeros: {e}")
                ephemeral_files = []

        # -------------------------------------------
        # 6) RAG (solo si procede)
        # -------------------------------------------
        rag_payload: Dict[str, Any] = {
            "status": None,
            "results": [],
            "aggregation": None,
            "used_department": None,
            "cleaned_query": rag_query,
        }

        scope_prefix = "contexto departamentos seleccionados:"
        rag_input_for_search = prompt if scope_prefix in prompt.lower() else rag_query

        if needs_rag:
            try:
                logging.info("[LLM_ENTRY %s] perform_rag_search query='%s'", request_id, rag_input_for_search)
                rag_payload = await perform_rag_search(rag_input_for_search, access_token, flow=flow)
            except Exception as e:
                logging.warning("[LLM_ENTRY %s] Error perform_rag_search: %s", request_id, e)

        rag_results: List[Dict[str, Any]] = rag_payload.get("results") or []
        rag_aggregation: Any = rag_payload.get("aggregation")
        rag_used_department: Optional[str] = rag_payload.get("used_department")
        rag_cleaned_query: str = rag_payload.get("cleaned_query") or rag_query

        if not needs_rag:
            rag_status = "skipped"
        else:
            raw_status = rag_payload.get("status")
            rag_status = raw_status or ("no_results" if not rag_results else "ok")

        plan["rag_status"] = rag_status
        plan["rag_used_department"] = rag_used_department
        plan["rag_cleaned_query"] = rag_cleaned_query
        plan["rag_aggregation"] = rag_aggregation
        plan["requested_department"] = requested_dept_dir
        plan["attached_file_ids"] = attached_file_ids
        plan["conversation_id"] = conversation.id

        # -------------------------------------------
        # 7) CrewAI run_chat
        # -------------------------------------------
        response_text = await asyncio.to_thread(
            crew_orchestrator.run_chat,
            normalized_prompt,
            rag_results,
            ephemeral_files,
            conversation_data,
            flow,
            plan,
        )
        response_text = clean_text(response_text or "")

        # -------------------------------------------
        # 8) Guardar historial en Redis (SCOPED)
        # -------------------------------------------
        conversation_entry = {
            "conversation_id": conversation.id,
            "user_id": user_id,
            "prompt": prompt,
            "normalized_prompt": normalized_prompt,
            "response": response_text,
            "flow": flow,
            "planner_plan": plan,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await save_conversation_to_redis(
            user_id=user_id,
            conversation_id=conversation.id,
            expires_in=480,
            conversation_entry=conversation_entry,
        )

        # -------------------------------------------
        # 9) Persistencia BOT en BD
        # -------------------------------------------
        bot_msg = Message(
            conversation_id=conversation.id,
            sender="BOT",
            content=response_text,
            created_at=datetime.now(timezone.utc),
        )
        db.add(bot_msg)
        db.commit()
        db.refresh(bot_msg)

        conversation.conversation_text = (
            (conversation.conversation_text + f"\nBOT: {response_text}").strip()
            if conversation.conversation_text
            else f"BOT: {response_text}"
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return {
            "reply": response_text,
            "response": response_text,
            "is_choice_needed": False,
            "conversation_id": conversation.id,
            "id": bot_msg.id,
            "message_id": bot_msg.id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"[LLM_ENTRY {request_id}] Error procesando consulta")
        status_label = "error"
        error_msg = f"Error procesando la consulta: {e}"
        return {"reply": error_msg, "response": error_msg, "is_choice_needed": False}
    finally:
        llm_query_counter.inc()
        llm_query_requests.labels(status=status_label).inc()
        elapsed = time.time() - start_time
        llm_query_latency.observe(elapsed)
        logging.info(f"[LLM_ENTRY {request_id}] Métricas. status={status_label}, latency={elapsed:.3f}s")


@app.get("/me", response_model=MeResponse)
async def get_me(
    request: Request,
    auth: Dict[str, Any] = Depends(get_current_auth),
):
    """
    Devuelve información básica del usuario autenticado, lista para
    mostrar en el panel de ajustes del front.

    Flujo:
      - Usa get_current_auth (JWT + sesión unívoca en Redis).
      - Intenta enriquecer la información preguntando a auth_sso (/auth/me).
      - Si auth_sso no responde o no está disponible, hace fallback a la
        información de la sesión en Redis.

    No expone el token ni datos sensibles.
    """
    session = auth["session"]

    sso_data: Optional[Dict[str, Any]] = None

    # ------------------------------
    # 1) Intentar consultar auth_sso
    # ------------------------------
    if AUTH_SSO_INTERNAL_BASE:
        try:
            # reenviamos la cookie de sesión al microservicio de SSO
            cookies: Dict[str, str] = {}
            access_cookie = request.cookies.get("access_token")
            if access_cookie:
                cookies["access_token"] = access_cookie

            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"{AUTH_SSO_INTERNAL_BASE}/auth/me",
                    cookies=cookies,
                )

            if resp.status_code == 200:
                sso_data = resp.json()
            else:
                logger.warning(
                    "get_me: auth_sso /auth/me devolvió código %s",
                    resp.status_code,
                )
        except Exception as e:
            logger.warning("get_me: error llamando a auth_sso /auth/me: %s", e)

    # ------------------------------
    # 2) Componer la respuesta final
    # ------------------------------
    sso = sso_data or {}

    # email: preferimos lo que diga SSO, si no tiramos de Redis
    email = sso.get("email") or session.get("email", "")

    # username: name de Entra → username interno; si no, el de Redis o el prefijo del email
    username = (
        sso.get("name")
        or session.get("username", "")
        or (email.split("@")[0] if email else "")
    )

    # role_id numérico: si auth_sso lo da, usamos ese; si no, del token interno (auth["role"])
    role_id = sso.get("role_id")
    if not isinstance(role_id, int):
        role_id = auth.get("role") if isinstance(auth.get("role"), int) else -1

    # role_name legible (ej. "Supervisor") lo seguimos sacando de la sesión Redis,
    # que es donde guardas el nombre del rol.
    role_name = session.get("role") or ""

    # departamentos: normalizamos a { "directory": "<ruta>" }
    raw_departments = (
        sso.get("departments")
        or session.get("departments", [])
        or []
    )
    departments_sanitized: List[Dict[str, str]] = []
    for dep in raw_departments:
        # En sesión de login guardas "department_directory"
        # En JWT guardas "directory"
        directory = dep.get("department_directory") or dep.get("directory")
        if directory:
            departments_sanitized.append({"directory": directory})

    return MeResponse(
        user_id=auth["user_id"],
        username=username,
        email=email,
        role_id=role_id,
        role_name=role_name,
        departments=departments_sanitized,
    )



@app.post("/logout")
async def logout(auth: dict = Depends(get_current_auth)):
    """
    Cierre de sesión desde el microservicio de modelo de negocio.

    - Requiere sesión válida (get_current_auth: JWT + Redis).
    - No toca directamente Redis ni la tabla sessions: delega el
      cierre "real" en el microservicio auth_sso (/auth/logout).
    - Devuelve un 303 hacia auth_sso para que se ejecute el flujo
      completo de logout (incluyendo el logout en Entra, si aplica).

    Desde el front:
      - Se hace fetch() a /logout.
      - El navegador sigue internamente los redirects (auth_sso → Entra → post-logout).
      - Después, el propio JS redirige al usuario a la pantalla de login.
    """
    if not AUTH_SSO_PUBLIC_BASE:
        logger.warning(
            "logout: AUTH_SSO_PUBLIC_BASE no configurado; no se puede delegar logout en auth_sso."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout SSO no disponible en este entorno.",
        )

    target = f"{AUTH_SSO_PUBLIC_BASE}/auth/logout"
    logger.info(
        "logout: redirigiendo a flujo de logout SSO en %s para user_id=%s.",
        target,
        auth["user_id"],
    )

    # No necesitamos manipular cookies aquí: auth_sso lo hará,
    # y el RP-initiated logout contra Entra se dispara desde auth_sso.
    return RedirectResponse(url=target, status_code=303)


@app.post("/context/reset", response_model=ResetContextResponse)
async def reset_context(
    request: Request,
    auth: dict = Depends(get_current_auth),
):
    """
    Resetea el contexto efímero de conversación en Redis para el usuario actual:
      - Historial ligero de conversación (get/save_conversation_to_redis)
      - (Opcional) Archivos en vuelo, si tienes helper específico

    No borra nada de la BD (Conversation, Message, etc.).
    """
    # Protección CSRF igual que en /query/llm
    validate_csrf(request)

    user_id = auth["user_id"]

    # Resetea el historial ligero en Redis
    await reset_conversation_in_redis(user_id)

    # Si en utils tienes algo tipo clear_ephemeral_files(user_id), úsalo aquí.
    # from utils import clear_ephemeral_files
    # await clear_ephemeral_files(user_id)

    logging.info(
        "[CONTEXT_RESET] Contexto efímero reseteado para user_id=%s", user_id
    )

    return ResetContextResponse(ok=True)


@app.post("/integrations/notetaker/sso-url", response_model=dict)
async def notetaker_sso_url(
    request: Request,
    response: Response,
    auth: Dict[str, Any] = Depends(get_current_auth),
    db: Session = Depends(get_db),
):
    """
    Devuelve una URL de SSO hacia Notetaker (/sso-callback) para el usuario autenticado.
    Alineado con /me para construir username/display_name.
    """
    try:
        validate_csrf(request)

        session_data = auth.get("session")
        if session_data is None:
            raise HTTPException(status_code=403, detail="Sesión no encontrada o expirada.")

        user_role = auth.get("role")
        if user_role not in [ROLE_SUPERVISOR, ROLE_USER]:
            raise HTTPException(status_code=403, detail="Permisos insuficientes para acceder a Notetaker.")

        raw_user_id = auth.get("user_id")
        if raw_user_id is None:
            raise HTTPException(status_code=500, detail="Auth inválida: falta user_id.")

        try:
            user_id_int = raw_user_id if isinstance(raw_user_id, int) else int(raw_user_id)
        except Exception:
            raise HTTPException(status_code=500, detail=f"user_id no numérico: {raw_user_id!r}")

        # Identidad alineada con /me
        email, display_name = await _resolve_notetaker_identity_like_me(request, auth)

        # Construcción URL Notetaker
        built = _build_notetaker_url(email=email, display_name=display_name, user_id=user_id_int)
        url = built["url"]

        # Auditoría (no bloqueante)
        try:
            audit = AuditLog(
                user_id=user_id_int,
                entity_name="IntegrationNotetaker",
                entity_id=user_id_int,
                action="CREATE",
                old_data=None,
                new_data={
                    "email": email,
                    "display_name": display_name or None,
                    "notetaker_base": built.get("notetaker_base"),
                    "includes_cosmos_token": built.get("includes_cosmos_token"),
                    "expires_in": built.get("expires_in"),
                },
                timestamp=datetime.now(timezone.utc),
            )
            db.add(audit)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.warning("notetaker_sso_url: fallo audit log (no bloqueante): %s", e)

        response.headers["Cache-Control"] = "no-store"

        return {
            "url": url,
            "includes_cosmos_token": built["includes_cosmos_token"],
            "expires_in": built["expires_in"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("notetaker_sso_url: 500 inesperado: %s", e)
        raise HTTPException(status_code=500, detail="Error interno generando SSO URL.")
    
if __name__ == "__main__":
    public_url = ngrok.connect(8000)
    logging.info(f" * ngrok URL: {public_url}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import secrets
from datetime import datetime, timedelta, timezone
import time
from pathlib import Path
from typing import Optional
import msal
import redis.asyncio as aioredis
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from jose import jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session as OrmSession

from config.database import get_db
from config.models import User, AuditLog, Session as DBSession

# =========================================================
#  LOGGING A FICHERO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
LOGS_ROOT = BASE_DIR / "logs"
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_ROOT / "auth_sso.log"

logger = logging.getLogger("auth_sso")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5,
    encoding="utf-8",
)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# =========================================================
#  CONFIG ENTORNO / JWT / TENANT
# =========================================================

TENANT_ID = os.getenv("TENANT_ID", "").strip()
CLIENT_ID = os.getenv("CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "").strip()

# Base pública del SSO (lo que ve el navegador / Entra) → SIEMPRE HTTPS
AUTH_PUBLIC_BASE = os.getenv(
    "AUTH_PUBLIC_BASE",
    "https://cosmos.cosgs.int"
).rstrip("/")

# Front donde vive el asistente / plataforma
FRONTEND_URL = os.getenv(
    "COSMOS_FRONTEND_URL",
    "https://cosmos.cosgs.int"
).rstrip("/")

# JWT interno (debe ser el mismo que usa login/modeloNegocio)
SECRET_KEY = os.getenv("COSMOS_SECRET_KEY", "secretkey123")
ALGORITHM = os.getenv("COSMOS_JWT_ALG", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "300")
)

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

ACCESS_TOKEN_COOKIE_NAME = "access_token"

# =========================================================
#  APP FASTAPI + MIDDLEWARES
# =========================================================

app = FastAPI(title="COSMOS Auth SSO (Entra)")

from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "dev-session-secret"),
    same_site="lax",         # en prod: "lax" o "strict"
    https_only=False,        # en prod: True + HTTPS
    max_age=60 * 60 * 8,     # 8 horas
)

# CORS
COSMOS_ENV = os.getenv("COSMOS_ENV", "dev").lower()
if COSMOS_ENV == "dev":
    ALLOWED_ORIGINS = ["*"]
else:
    raw_origins = os.getenv("COSMOS_AUTH_CORS_ORIGINS", "")
    ALLOWED_ORIGINS = [
        o.strip() for o in raw_origins.split(",") if o.strip()
    ] or [FRONTEND_URL]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)

# TrustedHost
raw_hosts = os.getenv("COSMOS_AUTH_ALLOWED_HOSTS", "*")
if raw_hosts != "*":
    ALLOWED_HOSTS = [h.strip() for h in raw_hosts.split(",") if h.strip()]
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=ALLOWED_HOSTS + ["localhost", "127.0.0.1", "cosmos.cosgs.int"],
    )

# Cabeceras de seguridad
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    response.headers.setdefault(
        "Referrer-Policy", "strict-origin-when-cross-origin"
    )
    response.headers.setdefault(
        "Permissions-Policy",
        "geolocation=(), microphone=(), camera=(), payment=(), interest-cohort=()",
    )
    if os.getenv("COSMOS_AUTH_ENABLE_HSTS", "false").lower() == "true":
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=63072000; includeSubDomains; preload",
        )
    return response

# =========================================================
#  REDIS + CRIPTO
# =========================================================

redis_client: Optional[aioredis.Redis] = None
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = aioredis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
    )
    logger.info("Redis conectado en %s:%s", REDIS_HOST, REDIS_PORT)


@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()
        logger.info("Redis cerrado correctamente.")


# =========================================================
#  CLIENTE OIDC (MSAL)
# =========================================================

class EntraOIDCClient:
    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        base_url: str,
        redirect_path: str = "/api/auth/callback",
        scopes: Optional[list[str]] = None,
        prompt: str = "select_account",
    ):
        if not all([tenant_id, client_id, client_secret, base_url]):
            raise RuntimeError(
                "Faltan TENANT_ID / CLIENT_ID / CLIENT_SECRET / AUTH_PUBLIC_BASE"
            )
        self.tenant_id = tenant_id.strip()
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.base_url = base_url.rstrip("/")
        self.redirect_path = redirect_path
        # 👇 ESTA es la URL que verá Entra y el navegador
        self.redirect_uri = f"{self.base_url}{self.redirect_path}"
        self.scopes = scopes or []
        self.prompt = prompt
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"

    def _msal_app(self) -> msal.ConfidentialClientApplication:
        return msal.ConfidentialClientApplication(
            self.client_id,
            authority=self.authority,
            client_credential=self.client_secret,
        )

    def build_auth_url(self, state: str) -> str:
        return self._msal_app().get_authorization_request_url(
            scopes=self.scopes,
            state=state,
            redirect_uri=self.redirect_uri,
            prompt=self.prompt,
        )

    def exchange_code(self, code: str) -> dict:
        return self._msal_app().acquire_token_by_authorization_code(
            code=code,
            scopes=self.scopes,
            redirect_uri=self.redirect_uri,
        )


entra_client = EntraOIDCClient(
    tenant_id=TENANT_ID,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    base_url=AUTH_PUBLIC_BASE,       # https://cosmos.cosgs.int
    redirect_path="/api/auth/callback",
    scopes=[],  # solo login; añade scopes de Graph si quieres
)

# =========================================================
#  HELPERS: JWT interno, Redis, sesiones
# =========================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    logger.info("Creando access_token interno para COSMOS (SSO Entra).")
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta if expires_delta else timedelta(minutes=15)
    )
    to_encode.update({"exp": expire})
    encoded = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded


async def save_session_to_redis(
    user: User,
    token: str,
    expires_in: int,
    session_id: Optional[int],
):
    if redis_client is None:
        logger.warning("redis_client no inicializado; no se guarda sesión.")
        return

    user_role = user.role.name
    user_departments = [
        {
            "department_directory": dep.department_directory,
            "faiss_index_path": dep.faiss_index_path,
            "vectorizer_path": dep.vectorizer_path,
        }
        for dep in user.departments
    ]

    session_data = {
        "token": token,
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user_role,
        "departments": user_departments,
        "user_directory": user.user_directory,
        "session_id": session_id,
    }

    await redis_client.set(
        f"session:{user.id}",
        json.dumps(session_data),
        ex=expires_in,
    )
    logger.info(
        "Sesión de usuario %s almacenada en Redis (session_id=%s).",
        user.id,
        session_id,
    )


async def get_session_from_redis(user_id: int):
    if redis_client is None:
        return None
    raw = await redis_client.get(f"session:{user_id}")
    return json.loads(raw) if raw else None


def create_persistent_session(
    user: User,
    token: str,
    expires_at: datetime,
    db: OrmSession,
) -> DBSession:
    logger.info(
        "Registrando sesión persistente en BD para user_id=%s.", user.id
    )

    hashed_token = pwd_context.hash(token)

    try:
        db.query(DBSession).filter(
            DBSession.user_id == user.id,
            DBSession.expires_at < datetime.utcnow(),
        ).delete(synchronize_session=False)
    except Exception as e:
        logger.warning(
            "No se pudieron limpiar sesiones caducadas para user_id=%s: %s",
            user.id,
            e,
        )

    session_row = DBSession(
        user_id=user.id,
        session_token=hashed_token,
        created_at=datetime.utcnow(),
        expires_at=expires_at,
    )
    db.add(session_row)
    db.commit()
    db.refresh(session_row)

    return session_row


def decode_access_token_from_cookie(request: Request) -> tuple[Optional[dict], Optional[str]]:
    raw_cookie = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    if not raw_cookie:
        logger.debug("decode_access_token_from_cookie: cookie %s ausente.", ACCESS_TOKEN_COOKIE_NAME)
        return None, None

    raw_token = raw_cookie
    if raw_token.startswith("Bearer "):
        raw_token = raw_token.split(" ", 1)[1].strip()

    try:
        payload = jwt.decode(raw_token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.debug(
            "decode_access_token_from_cookie: token decodificado correctamente para user_id=%s.",
            payload.get("user_id"),
        )
        return payload, raw_token
    except Exception as e:
        logger.warning("decode_access_token_from_cookie: no se pudo decodificar token de cookie: %s", e)
        return None, None


# =========================================================
#  ENDPOINTS
# =========================================================

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


@app.get("/auth/login")
async def auth_login(request: Request, next: str = "/"):
    state = secrets.token_urlsafe(16)
    request.session["oidc_state"] = state
    request.session["oidc_next"] = next if next else "/"

    try:
        auth_url = entra_client.build_auth_url(state)
        logger.info("Redirigiendo a Entra para login (state=%s).", state)
        return RedirectResponse(auth_url)
    except Exception as e:
        logger.exception("Error construyendo auth_url contra Entra:")
        return HTMLResponse(
            f"<h3>Configuración OIDC incompleta: {e}</h3>",
            status_code=500,
        )


@app.get("/auth/callback")
async def auth_callback(
    request: Request,
    state: str = "",
    code: str = "",
    db: OrmSession = Depends(get_db),
):
    stored_state = request.session.get("oidc_state")
    next_url = request.session.get("oidc_next", "/") or "/"

    if not state or state != stored_state:
        logger.warning("State inválido en callback OIDC (%s != %s).", state, stored_state)
        return HTMLResponse("<h3>State inválido</h3>", status_code=400)

    if not code:
        return HTMLResponse("<h3>Falta el parámetro 'code'</h3>", status_code=400)

    try:
        result = entra_client.exchange_code(code)
    except Exception as e:
        logger.exception("Error autenticando contra Entra:")
        return HTMLResponse(f"<h3>Error autenticando: {e}</h3>", status_code=500)

    if "id_token_claims" not in result:
        err = result.get("error_description") or str(result)
        logger.warning("Login Entra fallido: %s", err)
        return HTMLResponse(
            f"<h3>Login Entra fallido</h3><pre>{err}</pre>",
            status_code=401,
        )

    claims = result["id_token_claims"]
    oid = claims.get("oid")
    tid = claims.get("tid")
    upn = claims.get("preferred_username") or claims.get("email")
    name = claims.get("name") or upn

    logger.info(
        "Callback OIDC para usuario upn=%s (oid=%s, tid=%s).", upn, oid, tid
    )

    if not upn:
        return HTMLResponse(
            "<h3>No se pudo obtener el correo del usuario desde Entra.</h3>",
            status_code=400,
        )

    user = db.query(User).filter(User.email == upn).first()
    if user is None:
        logger.warning(
            "Usuario %s autenticado en tenant pero no registrado en COSMOS.",
            upn,
        )
        return HTMLResponse(
            "<h3>Usuario no autorizado en la plataforma COSMOS.</h3>",
            status_code=403,
        )

    departments_info = [
        {
            "directory": dep.department_directory,
            "index_filepath": dep.faiss_index_path,
            "vectorizer_path": dep.vectorizer_path,
        }
        for dep in user.departments
    ]

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    token_data = {
        "sub": user.email,
        "user_id": user.id,
        "role": user.role_id,
        "departments": departments_info,
        "oid": oid,
        "tid": tid,
        "name": name,
    }

    internal_jwt = create_access_token(
        data=token_data,
        expires_delta=access_token_expires,
    )
    session_expires_at = datetime.utcnow() + access_token_expires

    db_session = create_persistent_session(
        user=user,
        token=internal_jwt,
        expires_at=session_expires_at,
        db=db,
    )

    await save_session_to_redis(
        user=user,
        token=internal_jwt,
        expires_in=int(access_token_expires.total_seconds()),
        session_id=db_session.id,
    )

    audit_login = AuditLog(
        user_id=user.id,
        entity_name="UserLoginEntra",
        entity_id=user.id,
        action="UPDATE",
        old_data=None,
        new_data={
            "result": "success",
            "oid": oid,
            "tid": tid,
        },
        timestamp=datetime.utcnow(),
    )
    db.add(audit_login)

    audit_session = AuditLog(
        user_id=user.id,
        entity_name="UserSession",
        entity_id=db_session.id,
        action="CREATE",
        old_data=None,
        new_data={
            "created_at": db_session.created_at.isoformat(),
            "expires_at": db_session.expires_at.isoformat(),
        },
        timestamp=datetime.utcnow(),
    )
    db.add(audit_session)
    db.commit()

    request.session.pop("oidc_state", None)
    request.session.pop("oidc_next", None)

    redirect_to = f"{FRONTEND_URL}/{next_url.lstrip('/')}"
    logger.info("Login SSO OK, redirigiendo a %s.", redirect_to)

    response = RedirectResponse(url=redirect_to, status_code=303)
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        value=f"Bearer {internal_jwt}",
        httponly=True,
        secure=False,  # en prod: True + HTTPS
        samesite="Lax",
        max_age=int(access_token_expires.total_seconds()),
    )
    return response


@app.get("/auth/me")
async def auth_me(request: Request):
    payload, raw_token = decode_access_token_from_cookie(request)
    if not payload or not raw_token:
        logger.info("auth_me: sin token válido en cookie. Devolviendo 401.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado.")

    user_id = payload.get("user_id")
    if not user_id:
        logger.warning("auth_me: token sin user_id. Devolviendo 401.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido (sin user_id).",
        )

    session_data = await get_session_from_redis(user_id)
    if not session_data:
        logger.info(
            "auth_me: sesión no encontrada en Redis para user_id=%s. Probablemente expirada.",
            user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sesión expirada o no encontrada.",
        )

    session_token = session_data.get("token")
    if session_token != raw_token:
        logger.info(
            "auth_me: token de la cookie no coincide con sesión activa en Redis "
            "para user_id=%s. Posible sesión revocada o reemplazada.",
            user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sesión revocada o reemplazada. Inicia sesión de nuevo.",
        )

    logger.info(
        "auth_me: sesión válida para user_id=%s email=%s",
        user_id,
        payload.get("sub"),
    )

    return {
        "user_id": payload.get("user_id"),
        "email": payload.get("sub"),
        "role_id": payload.get("role"),
        "departments": payload.get("departments") or [],
        "name": payload.get("name"),
        "oid": payload.get("oid"),
        "tid": payload.get("tid"),
    }


@app.get("/auth/logout")
async def auth_logout(request: Request, db: OrmSession = Depends(get_db)):
    payload, raw_token = decode_access_token_from_cookie(request)
    user_id: Optional[int] = payload.get("user_id") if payload else None
    session_id: Optional[int] = None

    if user_id is not None:
        session_data = await get_session_from_redis(user_id)
        if session_data:
            session_id = session_data.get("session_id")

        if redis_client:
            try:
                await redis_client.delete(f"session:{user_id}")
            except Exception as e:
                logger.warning(
                    "Error eliminando clave de sesión en Redis para user_id=%s: %s",
                    user_id,
                    e,
                )

        try:
            now = datetime.utcnow()
            if session_id is not None:
                db_sess = db.query(DBSession).filter(
                    DBSession.id == session_id
                ).first()
                if db_sess:
                    db_sess.expires_at = now
                    db.add(db_sess)
            else:
                db.query(DBSession).filter(
                    DBSession.user_id == user_id,
                    DBSession.expires_at > now,
                ).update({"expires_at": now}, synchronize_session=False)

            audit = AuditLog(
                user_id=user_id,
                entity_name="UserSession",
                entity_id=session_id or 0,
                action="UPDATE",
                old_data=None,
                new_data={"action": "logout"},
                timestamp=now,
            )
            db.add(audit)
            db.commit()
        except Exception as e:
            logger.warning("Error actualizando sesión persistente en logout: %s", e)

    authority = entra_client.authority
    post_logout = f"{AUTH_PUBLIC_BASE}/api/auth/post-logout"
    ms_logout = (
        f"{authority}/oauth2/v2.0/logout?post_logout_redirect_uri={post_logout}"
    )

    response = RedirectResponse(url=ms_logout, status_code=303)
    response.delete_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        httponly=True,
        secure=False,  # en prod: True
        samesite="Lax",
    )
    return response


@app.get("/auth/post-logout")
async def post_logout():
    return RedirectResponse(url=f"{FRONTEND_URL}/")


if __name__ == "__main__":
    import uvicorn

    logger.info("Iniciando auth_sso en 0.0.0.0:7100 ...")
    uvicorn.run(app, host="0.0.0.0", port=7100)

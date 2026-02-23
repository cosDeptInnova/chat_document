"""FastAPI main application."""
import sys
import asyncio
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from app.api.routes import auth, meetings, vexa, integrations, analytics, security
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.database import engine, Base
from app.config import settings
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Fix para Windows: usar ProactorEventLoop para soportar subprocesses (Playwright)
# Solo configurar si no hay un policy ya establecido (evita conflictos durante reload)
if sys.platform == "win32":
    try:
        current_policy = asyncio.get_event_loop_policy()
        if not isinstance(current_policy, asyncio.WindowsProactorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        # Si hay algún problema, intentar establecer el policy de todas formas
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Configurar logging
# Crear directorio de logs si no existe
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Fichero base de log (rotación diaria a medianoche)
log_file = log_dir / "api.log"

# Formato de log
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Handler de rotación diaria
file_handler = TimedRotatingFileHandler(
    filename=log_file,
    when="midnight",       # Rota a medianoche (hora local)
    interval=1,
    backupCount=30,        # Conservar ~30 días de logs
    encoding="utf-8",
    utc=False,
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))

# Configurar logging básico (consola + fichero rotado)
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # Consola
        file_handler,             # Fichero con rotación diaria
    ],
)
logger = logging.getLogger(__name__)
logger.info("[API] Logging a fichero activo (rotación diaria): %s", log_file)

# Configurar logging de acceso HTTP
# Filtrar mensajes de heartbeat para no llenar la consola
access_logger = logging.getLogger("uvicorn.access")

class HeartbeatFilter(logging.Filter):
    """Filtro para no mostrar logs de heartbeat en la consola."""
    def filter(self, record):
        # Filtrar mensajes que contengan /api/auth/heartbeat
        message = record.getMessage()
        return "/api/auth/heartbeat" not in message

access_logger.addFilter(HeartbeatFilter())
access_logger.setLevel(logging.INFO)

# Crear aplicación FastAPI
app = FastAPI(
    title="Cosmos Notetaker API",
    description="API para transcripción y análisis de reuniones Teams",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Scheduler para monitoreo de Celery (independiente de Celery)
scheduler: Optional[AsyncIOScheduler] = None


@app.on_event("startup")
async def startup_event():
    """
    Evento de startup: crear tablas de la base de datos y inicializar scheduler
    para monitoreo de Celery.
    Se ejecuta después de que uvicorn complete su inicialización,
    evitando problemas durante el reload.
    """
    global scheduler
    
    # Crear tablas de la base de datos
    # En producción, usar Alembic migrations
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tablas de base de datos creadas/verificadas")
    except Exception as e:
        logger.warning(f"⚠️ Error creando tablas: {e}")
    
    # Inicializar scheduler para monitoreo de Celery
    # Este scheduler ejecuta verificaciones periódicas independientemente de Celery
    try:
        from app.services.monitoring_service import monitor_celery_health
        
        scheduler = AsyncIOScheduler()
        
        # Programar monitoreo de Celery cada 5 minutos
        scheduler.add_job(
            func=monitor_celery_health,
            trigger=IntervalTrigger(minutes=5),
            id="monitor_celery_health",
            name="Monitoreo de salud de Celery",
            replace_existing=True,
        )
        
        scheduler.start()
        logger.info("✅ Scheduler de monitoreo de Celery iniciado (verificacion cada 5 minutos)")
    except Exception as e:
        logger.error(f"❌ Error iniciando scheduler de monitoreo: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Evento de shutdown: detener scheduler de monitoreo.
    """
    global scheduler
    if scheduler:
        try:
            scheduler.shutdown()
            logger.info("✅ Scheduler de monitoreo de Celery detenido")
        except Exception as e:
            logger.warning(f"⚠️ Error deteniendo scheduler: {e}")

# Umbral en segundos para considerar un endpoint "lento" y enviar alerta
_SLOW_REQUEST_THRESHOLD_SEC = 30
_PATHS_TO_SKIP_FOR_SLOW_ALERT = ("/api/auth/heartbeat", "/health", "/")


# Middleware para logging de peticiones HTTP
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # No loguear peticiones de heartbeat para no llenar la consola
        is_heartbeat = request.url.path == "/api/auth/heartbeat"
        
        # Log de petición entrante (excepto heartbeat)
        if not is_heartbeat:
            logger.info(f"📥 {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")
        
        # Procesar petición
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Alerta si la petición fue lenta (sintomático de saturación)
            if (
                process_time > _SLOW_REQUEST_THRESHOLD_SEC
                and request.url.path not in _PATHS_TO_SKIP_FOR_SLOW_ALERT
            ):
                try:
                    from app.services.monitoring_service import alert_timeout
                    alert_timeout(request.url.path, _SLOW_REQUEST_THRESHOLD_SEC)
                except Exception:
                    pass
            
            # Log de respuesta (excepto heartbeat)
            if not is_heartbeat:
                status_code = response.status_code
                status_emoji = "✅" if 200 <= status_code < 300 else "⚠️" if 300 <= status_code < 400 else "❌"
                logger.info(
                    f"{status_emoji} {request.method} {request.url.path} - "
                    f"Status: {status_code} - Time: {process_time:.3f}s"
                )
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"❌ {request.method} {request.url.path} - "
                f"Error: {str(e)} - Time: {process_time:.3f}s",
                exc_info=True
            )
            raise

# Agregar middleware de logging ANTES de CORS
app.add_middleware(LoggingMiddleware)

# Agregar middleware de headers de seguridad
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware
# Permitir todos los orígenes para acceso desde red interna/VPN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite acceso desde cualquier IP (red interna/VPN)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 # Incluir routers
app.include_router(auth.router)
app.include_router(meetings.router)
app.include_router(vexa.router)
app.include_router(vexa.internal_router)  # Router interno para webhooks de bot-manager
app.include_router(integrations.router)
app.include_router(analytics.router)
app.include_router(security.router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Cosmos Notetaker API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Servir imagen estática del bot
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)

@app.get("/api/bot-avatar")
async def get_bot_avatar():
    """
    Endpoint para servir la imagen del bot directamente desde el backend.
    La imagen debe estar en backend/static/bot-avatar.png
    """
    avatar_path = static_dir / "bot-avatar.png"
    
    if not avatar_path.exists():
        logger.warning("⚠️ Imagen del bot no encontrada en %s", avatar_path)
        return {"error": "Avatar no encontrado"}, 404
    
    return FileResponse(
        avatar_path,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7000,
        reload=settings.debug
    )


"""Celery tasks for calendar synchronization."""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from celery import Task
from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.user import User
from app.services.calendar_sync_service import CalendarSyncService

logger = logging.getLogger(__name__)

# Umbral en dias: renovar suscripcion Outlook si expira en menos de este tiempo
OUTLOOK_RENEW_DAYS_THRESHOLD = 1.0


class DatabaseTask(Task):
    """Task base class that provides database session."""
    
    _db = None
    
    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Hook llamado cuando una tarea falla."""
        logger.error(
            f"[CELERY] Tarea {task_id} falló: {exc}",
            exc_info=einfo
        )
    
    def after_return(self, *args, **kwargs):
        """Close database session after task completes."""
        if self._db is not None:
            self._db.close()
            self._db = None


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    max_retries=2,
    default_retry_delay=60,
)
def run_calendar_sync_for_user(self, user_id: str):
    """
    Tarea Celery para sincronizar calendarios de un usuario.
    
    Esta tarea ejecuta el sync de calendarios (Outlook + Google) en un worker Celery,
    evitando bloquear el proceso API.
    
    Args:
        user_id: ID del usuario a sincronizar
    """
    logger.info(f"[CELERY] Iniciando sync de calendario para usuario {user_id}")
    
    db = self.db
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        logger.error(f"[CELERY] Usuario {user_id} no encontrado")
        return {"success": False, "error": "User not found"}
    
    try:
        # Renovacion de suscripciones (Outlook/Google) + sync
        sync_service = CalendarSyncService()
        
        async def _run_sync():
            return await sync_service.run_full_sync_with_renewal(user, db)
        
        results = asyncio.run(_run_sync())
        
        logger.info(
            f"[CELERY] Sync completado para usuario {user.email} (ID: {user_id}): "
            f"Google={{synced: {results.get('google', {}).get('synced', 0)}, "
            f"created: {results.get('google', {}).get('created', 0)}}}, "
            f"Outlook={{synced: {results.get('outlook', {}).get('synced', 0)}, "
            f"created: {results.get('outlook', {}).get('created', 0)}}}"
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "user_email": user.email,
            "results": results
        }
        
    except Exception as e:
        logger.error(
            f"[CELERY] Error en sync de calendario para usuario {user_id}: {e}",
            exc_info=True
        )
        
        # Reintentar si aún hay intentos disponibles
        if self.request.retries < self.max_retries:
            logger.info(
                f"[CELERY] Reintentando sync para usuario {user_id} "
                f"(intento {self.request.retries + 1}/{self.max_retries})"
            )
            raise self.retry(exc=e)
        
        return {
            "success": False,
            "user_id": user_id,
            "error": str(e)
        }


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    max_retries=1,
)
def renew_outlook_subscriptions_if_needed(self):
    """
    Tarea Beat: para cada usuario con Outlook conectado y suscripcion por expirar
    (o sin suscripcion), encola run_calendar_sync_for_user para renovar y sincronizar.
    Asi las suscripciones se renuevan antes de caducar sin que el usuario tenga que
    sincronizar manualmente.
    """
    db = self.db
    now = datetime.now(timezone.utc)
    threshold = now + timedelta(days=OUTLOOK_RENEW_DAYS_THRESHOLD)
    enqueued = 0
    try:
        users = db.query(User).all()
        for user in users:
            if not user.settings:
                continue
            outlook_data = user.settings.get("outlook_calendar")
            if not outlook_data:
                continue
            sub_id = outlook_data.get("subscription_id")
            sub_exp = outlook_data.get("subscription_expiration")
            should_sync = False
            if not sub_id:
                should_sync = True
            elif sub_exp:
                try:
                    s = str(sub_exp).replace("Z", "+00:00")
                    exp_dt = datetime.fromisoformat(s)
                    if exp_dt.tzinfo is None:
                        exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                    if exp_dt < threshold:
                        should_sync = True
                except Exception:
                    should_sync = True
            if should_sync:
                run_calendar_sync_for_user.delay(str(user.id))
                enqueued += 1
                logger.info(
                    "[CELERY] Sync encolado para renovar/crear suscripcion Outlook: usuario=%s",
                    user.email,
                )
        logger.info("[CELERY] renew_outlook_subscriptions_if_needed: %d usuarios encolados", enqueued)
        return {"enqueued": enqueued}
    finally:
        pass

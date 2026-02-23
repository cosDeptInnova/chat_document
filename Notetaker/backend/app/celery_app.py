"""Celery configuration for scheduled tasks."""
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from celery import Celery, signals
from app.config import settings

# Crear instancia de Celery
celery_app = Celery(
    "notetaker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.meeting_tasks", "app.tasks.summary_tasks", "app.tasks.calendar_tasks"]
)

# Configuración de Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutos máximo por tarea (solo para tareas generales)
    task_soft_time_limit=240,  # 4 minutos soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    # Definir colas
    task_queues={
        "celery": {"exchange": "celery", "routing_key": "celery"},
        "summary_queue": {"exchange": "summary", "routing_key": "summary"},
    },
    # Routing automático para tareas de resumen
    task_routes={
        "app.tasks.summary_tasks.process_meeting_summary": {"queue": "summary_queue"},
    },
    # Configurar tareas periodicas (Beat)
    beat_schedule={
        "check-stuck-meetings": {
            "task": "app.tasks.meeting_tasks.check_stuck_meetings",
            "schedule": 300.0,  # Cada 5 minutos
        },
        "check-and-finalize-completed-meetings": {
            "task": "app.tasks.meeting_tasks.check_and_finalize_completed_meetings",
            "schedule": 600.0,  # Cada 10 minutos
        },
        # Requiere tener Celery Beat en ejecucion (ej: celery -A app.celery_app beat)
        "check-system-health": {
            "task": "app.tasks.meeting_tasks.run_check_system_health",
            "schedule": 600.0,  # Cada 10 minutos
        },
        "check-and-redownload-incomplete-transcripts": {
            "task": "app.tasks.meeting_tasks.check_and_redownload_incomplete_transcripts",
            "schedule": 3600.0,  # Cada hora
        },
        "check-and-schedule-pending-meetings": {
            "task": "app.tasks.meeting_tasks.check_and_schedule_pending_meetings",
            "schedule": 300.0,  # Cada 5 minutos
        },
        "renew-outlook-subscriptions-if-needed": {
            "task": "app.tasks.calendar_tasks.renew_outlook_subscriptions_if_needed",
            "schedule": 43200.0,  # Cada 12 horas: renovar suscripciones Outlook antes de caducar
        },
    },
)

# Fase 6: Configurar logging de Celery a fichero (solo cuando el worker arranca)
@signals.worker_init.connect
def setup_celery_logging(**kwargs):
    """
    Configurar logging de Celery a fichero cuando el worker arranca.
    Se ejecuta solo en el proceso worker, no en el proceso API.
    """
    # Crear directorio de logs si no existe
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Fichero base de log (rotación diaria a medianoche)
    log_file = log_dir / "celery.log"

    # Formato de log (igual que el API)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

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
    file_handler.setFormatter(logging.Formatter(log_fmt))

    # Añadir handler a los loggers de Celery
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # También añadir a loggers específicos de Celery
    for logger_name in ("celery", "celery.worker", "celery.task", "celery.worker.job"):
        celery_logger = logging.getLogger(logger_name)
        celery_logger.addHandler(file_handler)
        celery_logger.setLevel(logging.INFO)

    # Log de confirmación
    logger = logging.getLogger(__name__)
    logger.info("[CELERY] Logging a fichero activo (rotación diaria): %s", log_file)


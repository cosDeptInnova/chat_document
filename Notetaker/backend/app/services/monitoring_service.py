"""Servicio de monitoreo y alertas del sistema."""
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from app.services.email_service import send_alert_email
from app.config import settings
from app.celery_app import celery_app

logger = logging.getLogger(__name__)

# Cache para evitar enviar múltiples alertas del mismo tipo en poco tiempo
_alert_cache: Dict[str, datetime] = {}
_ALERT_COOLDOWN_MINUTES = 30  # No enviar la misma alerta más de una vez cada 30 minutos
_CELERY_INSPECT_TIMEOUT = 5.0  # Tiempo de espera para inspeccionar workers


def _should_send_alert(alert_key: str) -> bool:
    """
    Verifica si se debe enviar una alerta basándose en el cooldown.
    
    Args:
        alert_key: Clave única para la alerta (ej: "timeout_/api/meetings/list")
        
    Returns:
        True si se debe enviar, False si está en cooldown
    """
    now = datetime.utcnow()
    last_sent = _alert_cache.get(alert_key)
    
    if last_sent is None:
        return True
    
    time_since_last = (now - last_sent).total_seconds() / 60  # minutos
    
    if time_since_last >= _ALERT_COOLDOWN_MINUTES:
        return True
    
    logger.debug(
        f"Alerta '{alert_key}' en cooldown (última vez hace {time_since_last:.1f} minutos)"
    )
    return False


def _record_alert_sent(alert_key: str):
    """Registra que se envió una alerta."""
    _alert_cache[alert_key] = datetime.utcnow()


# Nombres de colas Celery (deben coincidir con task_queues en celery_app.py)
_CELERY_QUEUE_NAMES = ("celery", "summary_queue")


def check_celery_queue_size() -> Optional[int]:
    """
    Verifica el tamaño real de las colas de Celery en Redis (tareas pendientes).
    Usa LLEN sobre las listas del broker; si Redis no está disponible, hace
    fallback a tareas reservadas por workers (menos preciso).

    Returns:
        Número de tareas pendientes en cola o None si no se puede verificar
    """
    try:
        import redis
        r = redis.from_url(settings.redis_url)
        try:
            total = 0
            for q in _CELERY_QUEUE_NAMES:
                total += r.llen(q)
            return total
        finally:
            r.close()
    except Exception as e_redis:
        logger.debug(f"No se pudo obtener longitud de cola desde Redis: {e_redis}")
        try:
            inspect = celery_app.control.inspect()
            if not inspect:
                return None
            reserved = inspect.reserved()
            if not reserved:
                return 0
            return sum(len(tasks) for tasks in reserved.values())
        except Exception as e:
            logger.error(f"Error verificando cola de Celery: {e}")
            return None


def check_celery_workers() -> Dict[str, Any]:
    """
    Verifica el estado de los workers de Celery.
    
    Returns:
        Diccionario con información de workers activos
    """
    try:
        # En Windows con pool 'solo', el worker no responde a inspect() si está ocupado.
        # Como esta verificación suele correr DESDE una tarea, el worker está ocupado.
        inspect = celery_app.control.inspect(timeout=_CELERY_INSPECT_TIMEOUT)
        
        stats = None
        try:
            stats = inspect.stats()
        except Exception as e_stats:
            logger.debug(f"Error obteniendo stats de Celery (posible timeout): {e_stats}")
            
        active = None
        try:
            active = inspect.active()
        except Exception as e_active:
            logger.debug(f"Error obteniendo tareas activas de Celery: {e_active}")
        
        if not stats:
            # Si no hay stats, podría ser un worker estancado O simplemente el timeout del pool 'solo'.
            # IMPORTANTE: Si estamos llamando a esto desde una tarea Celery, ¡sabemos que hay al menos un worker vivo!
            return {"active": None, "workers": [], "uncertain": True}
        
        workers_info = []
        for worker_name, worker_stats in stats.items():
            active_tasks = active.get(worker_name, []) if active else []
            workers_info.append({
                "name": worker_name,
                "status": "active",
                "active_tasks": len(active_tasks)
            })
        
        return {
            "active": len(workers_info) > 0,
            "workers": workers_info,
            "count": len(workers_info)
        }
    except Exception as e:
        logger.error(f"Error verificando workers de Celery: {e}")
        return {"active": False, "workers": [], "error": str(e)}


def alert_timeout(endpoint: str, timeout_seconds: float, request_count: int = 1):
    """
    Envía alerta cuando hay timeouts frecuentes en un endpoint.
    
    Args:
        endpoint: Ruta del endpoint (ej: "/api/meetings/list")
        timeout_seconds: Tiempo de timeout en segundos
        request_count: Número de requests con timeout
    """
    alert_key = f"timeout_{endpoint}"
    
    if not _should_send_alert(alert_key):
        return
    
    title = f"Timeouts frecuentes en {endpoint}"
    message = (
        f"Se detectaron {request_count} timeout(s) en el endpoint {endpoint}. "
        f"El timeout configurado es de {timeout_seconds} segundos."
    )
    details = {
        "Endpoint": endpoint,
        "Timeout configurado": f"{timeout_seconds}s",
        "Numero de timeouts": str(request_count),
        "Recomendacion": "Revisar logs y considerar aumentar timeout o optimizar el endpoint"
    }
    
    success = send_alert_email(
        alert_type="timeout",
        title=title,
        message=message,
        details=details
    )
    
    if success:
        _record_alert_sent(alert_key)


def alert_celery_queue(queue_size: int, threshold: int = 100):
    """
    Envía alerta cuando la cola de Celery tiene demasiadas tareas pendientes.
    
    Args:
        queue_size: Tamaño actual de la cola
        threshold: Umbral de alerta (default: 100)
    """
    if queue_size < threshold:
        return
    
    alert_key = "celery_queue_size"
    
    if not _should_send_alert(alert_key):
        return
    
    title = f"Cola de Celery con {queue_size} tareas pendientes"
    message = (
        f"La cola de Celery tiene {queue_size} tareas pendientes, "
        f"superando el umbral de {threshold} tareas."
    )
    details = {
        "Tareas en cola": str(queue_size),
        "Umbral configurado": str(threshold),
        "Recomendacion": "Considerar anadir mas workers de Celery o revisar tareas que estan fallando"
    }
    
    success = send_alert_email(
        alert_type="celery_queue",
        title=title,
        message=message,
        details=details
    )
    
    if success:
        _record_alert_sent(alert_key)


def alert_db_pool_exhausted(active_connections: int, max_connections: int):
    """
    Envía alerta cuando el pool de base de datos está agotado.
    
    Args:
        active_connections: Número de conexiones activas
        max_connections: Número máximo de conexiones
    """
    alert_key = "db_pool_exhausted"
    
    if not _should_send_alert(alert_key):
        return
    
    title = "Pool de base de datos agotado"
    message = (
        f"El pool de base de datos esta casi agotado: {active_connections}/{max_connections} conexiones activas."
    )
    details = {
        "Conexiones activas": str(active_connections),
        "Conexiones maximas": str(max_connections),
        "Porcentaje usado": f"{(active_connections/max_connections)*100:.1f}%",
        "Recomendacion": "Aumentar pool_size y max_overflow en la configuracion de SQLAlchemy"
    }
    
    success = send_alert_email(
        alert_type="db_pool",
        title=title,
        message=message,
        details=details
    )
    
    if success:
        _record_alert_sent(alert_key)


def alert_celery_worker_down(worker_name: Optional[str] = None):
    """
    Envía alerta cuando un worker de Celery está caído.
    
    Args:
        worker_name: Nombre del worker caído (opcional)
    """
    alert_key = f"worker_down_{worker_name or 'unknown'}"
    
    if not _should_send_alert(alert_key):
        return
    
    title = f"Worker de Celery caido: {worker_name or 'Desconocido'}"
    message = (
        f"Se detecto que un worker de Celery esta caido o no responde. "
        f"Worker: {worker_name or 'No identificado'}"
    )
    details = {
        "Worker": worker_name or "Desconocido",
        "Recomendacion": "Verificar logs del worker y reiniciarlo si es necesario"
    }
    
    success = send_alert_email(
        alert_type="worker_down",
        title=title,
        message=message,
        details=details
    )
    
    if success:
        _record_alert_sent(alert_key)


def check_celery_beat() -> Dict[str, Any]:
    """
    Verifica si Celery Beat está activo comprobando si el archivo de schedule
    se está actualizando recientemente.
    
    Funciona tanto si Celery Beat está como servicio de Windows como si está
    ejecutándose desde PowerShell/CMD.
    
    Returns:
        Diccionario con información sobre el estado de Beat
    """
    try:
        # Buscar el archivo celerybeat-schedule.dat en el directorio backend
        backend_dir = Path(__file__).resolve().parent.parent.parent
        schedule_file = backend_dir / "celerybeat-schedule.dat"
        
        # También buscar en el directorio actual por si acaso
        if not schedule_file.exists():
            schedule_file = backend_dir / "celerybeat-schedule"
        
        if not schedule_file.exists():
            logger.warning("[MONITORING] Archivo celerybeat-schedule.dat no encontrado")
            return {"active": False, "error": "Schedule file not found"}
        
        # Obtener la fecha de última modificación del archivo
        last_modified = datetime.fromtimestamp(schedule_file.stat().st_mtime)
        now = datetime.now()
        time_since_modification = (now - last_modified).total_seconds()
        
        # Si el archivo se modificó hace menos de 15 minutos, Beat está activo
        # (Beat actualiza el archivo cada vez que programa una tarea, normalmente cada minuto)
        BEAT_TIMEOUT_SECONDS = 15 * 60  # 15 minutos
        
        is_active = time_since_modification < BEAT_TIMEOUT_SECONDS
        
        return {
            "active": is_active,
            "last_modified": last_modified.isoformat(),
            "seconds_since_modification": time_since_modification,
            "schedule_file": str(schedule_file)
        }
    except Exception as e:
        logger.error(f"Error verificando Celery Beat: {e}")
        return {"active": False, "error": str(e)}


def alert_celery_beat_down():
    """
    Envía alerta cuando Celery Beat está caído.
    """
    alert_key = "beat_down"
    
    if not _should_send_alert(alert_key):
        return
    
    title = "Celery Beat caido"
    message = (
        "Se detecto que Celery Beat esta caido o no esta actualizando el archivo de schedule. "
        "Las tareas periodicas no se estan programando."
    )
    details = {
        "Componente": "Celery Beat",
        "Recomendacion": "Verificar logs de Celery Beat y reiniciarlo si es necesario. "
                        "Asegurarse de que el proceso este corriendo (servicio de Windows o PowerShell)"
    }
    
    success = send_alert_email(
        alert_type="beat_down",
        title=title,
        message=message,
        details=details
    )
    
    if success:
        _record_alert_sent(alert_key)


# Umbral de uso del pool de BD para alertar (porcentaje 0-1)
_DB_POOL_ALERT_THRESHOLD = 0.9


def check_and_alert_system_health():
    """
    Verifica el estado general del sistema y envía alertas si es necesario.
    Pensada para ser llamada periódicamente (ej: desde una tarea Celery Beat).
    """
    # Verificar cola de Celery (longitud real en Redis)
    queue_size = check_celery_queue_size()
    if queue_size is not None:
        alert_celery_queue(queue_size)

    # Verificar workers de Celery
    workers_info = check_celery_workers()
    # Solo alertar si estamos seguros de que no hay workers (active=False)
    # Si active es None, es que el resultado fue incierto (timeout del pool 'solo')
    if workers_info.get("active") is False and workers_info.get("count", 0) == 0:
        alert_celery_worker_down()
    elif workers_info.get("uncertain"):
        logger.info("[MONITORING] Verificación de workers incierta (posible timeout del pool 'solo'), omitiendo alerta.")

    # Verificar pool de base de datos
    try:
        from app.database import engine
        pool = engine.pool
        checkedout = pool.checkedout()
        # QueuePool: _pool_size + _max_overflow = capacidad total
        total_slots = getattr(pool, "_pool_size", 10) + getattr(pool, "_max_overflow", 20)
        if total_slots <= 0:
            total_slots = 30
        if checkedout >= total_slots * _DB_POOL_ALERT_THRESHOLD:
            alert_db_pool_exhausted(checkedout, int(total_slots))
    except Exception as e:
        logger.debug(f"No se pudo verificar pool de BD: {e}")


def monitor_celery_health():
    """
    Funcion principal de monitoreo de Celery que verifica tanto Worker como Beat.
    Esta funcion se ejecuta periodicamente desde el backend usando APScheduler,
    independientemente de si Celery esta funcionando o no.
    
    Funciona tanto si Celery esta como servicio de Windows como si esta
    ejecutandose desde PowerShell/CMD.
    """
    logger.info("[MONITORING] Iniciando verificacion de salud de Celery...")
    
    # Verificar Celery Worker
    workers_info = check_celery_workers()
    if workers_info.get("active") is False and workers_info.get("count", 0) == 0:
        logger.warning("[MONITORING] Celery Worker no esta activo")
        alert_celery_worker_down()
    elif workers_info.get("uncertain"):
        logger.info("[MONITORING] Verificacion de workers incierta (posible timeout del pool 'solo'), omitiendo alerta.")
    else:
        logger.info(f"[MONITORING] Celery Worker activo: {workers_info.get('count', 0)} worker(s)")
    
    # Verificar Celery Beat
    beat_info = check_celery_beat()
    if not beat_info.get("active", False):
        logger.warning("[MONITORING] Celery Beat no esta activo")
        alert_celery_beat_down()
    else:
        logger.info(f"[MONITORING] Celery Beat activo (ultima modificacion hace {beat_info.get('seconds_since_modification', 0):.0f} segundos)")
    
    # Verificar cola de Celery (longitud real en Redis)
    queue_size = check_celery_queue_size()
    if queue_size is not None:
        logger.info(f"[MONITORING] Cola de Celery: {queue_size} tareas pendientes")
        alert_celery_queue(queue_size)
    
    logger.info("[MONITORING] Verificacion de salud de Celery completada")

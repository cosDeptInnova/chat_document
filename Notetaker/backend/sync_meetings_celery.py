"""Script para sincronizar reuniones pendientes con Celery."""
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Asegurar que estamos en el directorio correcto
backend_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(backend_dir))

try:
    from app.database import SessionLocal
    from app.models.meeting import Meeting, MeetingStatus
    from app.tasks.meeting_tasks import join_bot_to_meeting
    from app.celery_app import celery_app
    import logging

    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    def verify_celery_task_exists(task_id: str) -> bool:
        """Verificar si una tarea de Celery existe y está activa."""
        if not task_id:
            return False
        try:
            # Intentar obtener el estado de la tarea
            result = celery_app.AsyncResult(task_id)
            # Si la tarea existe, result.state no será None
            # Si está PENDING, significa que está en la cola o programada
            # Si está en otro estado, también existe
            return result.state is not None
        except Exception as e:
            logger.debug(f"Error verificando tarea {task_id}: {e}")
            return False

    def sync_meetings_with_celery():
        """Sincronizar todas las reuniones PENDING con Celery."""
        db = SessionLocal()
        
        try:
            now = datetime.now(timezone.utc)
            now_naive = now.replace(tzinfo=None)
            
            # Buscar todas las reuniones PENDING que no están eliminadas
            pending_meetings = db.query(Meeting).filter(
                Meeting.status == MeetingStatus.PENDING,
                Meeting.deleted_at.is_(None),
            ).all()
            
            logger.info(f"Encontradas {len(pending_meetings)} reuniones en estado PENDING")
            
            synced_count = 0
            skipped_with_bot = 0
            skipped_past = 0
            skipped_valid_task = 0
            error_count = 0
            
            for meeting in pending_meetings:
                try:
                    # Verificar si ya tiene bot asignado
                    if meeting.recall_bot_id:
                        logger.info(
                            f"Reunion {meeting.id} ya tiene bot asignado (bot_id={meeting.recall_bot_id}), "
                            f"no necesita tarea programada"
                        )
                        skipped_with_bot += 1
                        continue
                    
                    # Verificar si hay otra reunión con la misma URL que ya tenga bot
                    shared_bot_meeting = db.query(Meeting).filter(
                        Meeting.meeting_url == meeting.meeting_url,
                        Meeting.recall_bot_id.isnot(None),
                        Meeting.id != meeting.id,
                        Meeting.deleted_at.is_(None),
                    ).first()
                    
                    if shared_bot_meeting:
                        logger.info(
                            f"Reunion {meeting.id} comparte URL con reunion {shared_bot_meeting.id} "
                            f"que ya tiene bot (bot_id={shared_bot_meeting.recall_bot_id}), "
                            f"reutilizando bot existente"
                        )
                        # Reutilizar el bot existente
                        meeting.recall_bot_id = shared_bot_meeting.recall_bot_id
                        meeting.recall_status = shared_bot_meeting.recall_status or "processing"
                        db.commit()
                        skipped_with_bot += 1
                        continue
                    
                    # Verificar que la fecha sea futura
                    scheduled_time = meeting.scheduled_start_time
                    if scheduled_time.tzinfo is None:
                        scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
                    
                    if scheduled_time <= now:
                        logger.warning(
                            f"Reunion {meeting.id} ya paso su hora de inicio "
                            f"({scheduled_time}), saltando"
                        )
                        skipped_past += 1
                        continue
                    
                    # Verificar si tiene celery_task_id válido
                    if meeting.celery_task_id:
                        if verify_celery_task_exists(meeting.celery_task_id):
                            logger.info(
                                f"Reunion {meeting.id} ya tiene tarea Celery valida "
                                f"(task_id={meeting.celery_task_id}), saltando"
                            )
                            skipped_valid_task += 1
                            continue
                        else:
                            logger.warning(
                                f"Reunion {meeting.id} tiene celery_task_id invalido "
                                f"({meeting.celery_task_id}), re-programando"
                            )
                    
                    # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
                    join_time = scheduled_time - timedelta(minutes=1)
                    if join_time.tzinfo is None:
                        join_time = join_time.replace(tzinfo=timezone.utc)
                    
                    logger.info(
                        f"Programando bot para reunion {meeting.id}: "
                        f"inicio={scheduled_time} (UTC), join_time={join_time} (UTC), "
                        f"now={now} (UTC)"
                    )
                    logger.info(
                        f"Diferencia de tiempo: {(join_time - now).total_seconds():.0f} segundos "
                        f"({(join_time - now).total_seconds() / 60:.1f} minutos)"
                    )
                    
                    # Solo programar si la fecha es futura
                    if join_time > now:
                        # Programar tarea usando apply_async con eta
                        try:
                            task = join_bot_to_meeting.apply_async(
                                args=[meeting.id],
                                eta=join_time
                            )
                            # Guardar task_id para poder cancelarlo después
                            meeting.celery_task_id = task.id
                            db.commit()
                            logger.info(
                                f"Tarea programada para unir bot a reunion {meeting.id} "
                                f"el {join_time} (UTC) (task_id={task.id})"
                            )
                            synced_count += 1
                        except Exception as e:
                            logger.error(
                                f"Error programando tarea Celery para reunion {meeting.id}: {e}",
                                exc_info=True
                            )
                            error_count += 1
                    else:
                        # Si ya pasó el tiempo, intentar unirse inmediatamente
                        time_diff = (join_time - now).total_seconds()
                        logger.warning(
                            f"La reunion {meeting.id} esta programada para el pasado o muy pronto. "
                            f"Diferencia: {time_diff:.0f} segundos. Intentando unirse inmediatamente..."
                        )
                        try:
                            join_bot_to_meeting.delay(meeting.id)
                            logger.info(f"Tarea enviada para ejecucion inmediata")
                            synced_count += 1
                        except Exception as e:
                            logger.error(
                                f"Error ejecutando tarea inmediatamente para reunion {meeting.id}: {e}",
                                exc_info=True
                            )
                            error_count += 1
                            
                except Exception as e:
                    logger.error(
                        f"Error procesando reunion {meeting.id}: {e}",
                        exc_info=True
                    )
                    error_count += 1
            
            logger.info("=" * 60)
            logger.info("Resumen de sincronizacion:")
            logger.info(f"  - Reuniones sincronizadas: {synced_count}")
            logger.info(f"  - Saltadas (ya tienen bot): {skipped_with_bot}")
            logger.info(f"  - Saltadas (ya pasaron): {skipped_past}")
            logger.info(f"  - Saltadas (tarea valida): {skipped_valid_task}")
            logger.info(f"  - Errores: {error_count}")
            logger.info(f"  - Total procesadas: {len(pending_meetings)}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error en sincronizacion: {e}", exc_info=True)
            sys.exit(1)
        finally:
            db.close()

    if __name__ == "__main__":
        print("Sincronizando reuniones pendientes con Celery...")
        sync_meetings_with_celery()
        print("Sincronizacion completada")

except Exception as e:
    print(f"Error inicializando script: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


"""Celery tasks for meeting management (VEXA bot and transcriptions)."""
from datetime import datetime, timedelta, timezone
from celery import Task
import uuid
from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.meeting import Meeting, MeetingStatus
from app.models.transcription import Transcription, TranscriptionSegment
from app.models.summary import Summary
from app.services.vexa_service import VexaService, VexaServiceError, parse_teams_meeting_url
from app.config import settings
import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _user_friendly_join_error(err_msg: str) -> str:
    """
    Convierte un mensaje de error tecnico al unir el bot (VEXA/Celery) en un texto
    claro para el usuario.
    """
    if not err_msg or not str(err_msg).strip():
        return "No se pudo unir el bot a la reunion. Intentalo de nuevo o contacta con soporte."
    msg = str(err_msg).strip().lower()
    if "conexion" in msg or "connection" in msg or "connect" in msg:
        return "Problemas de conexion con el servicio del bot (VEXA). Comprueba la red e intentalo de nuevo."
    if "timeout" in msg or "timed out" in msg:
        return "El servicio del bot tardo demasiado en responder. La reunion puede haber empezado ya; intentalo en otra reunion."
    if "401" in msg or "403" in msg or "no autorizado" in msg or "unauthorized" in msg:
        return "Error de configuracion del servicio del bot (acceso no autorizado). Contacta con el administrador."
    if "404" in msg or "not found" in msg:
        return "El servicio del bot no encontro la reunion. Comprueba que el enlace de la reunion sea correcto."
    if "500" in msg or "502" in msg or "503" in msg or "server" in msg:
        return "El servicio del bot no esta disponible temporalmente. Intentalo mas tarde."
    if "native_meeting_id" in msg or "url" in msg or "parse" in msg:
        return "El enlace de la reunion de Teams no es valido o no se pudo interpretar. Comprueba el enlace."
    if len(err_msg) > 400:
        return err_msg[:397] + "..."
    return err_msg


def schedule_summary_after_transcription(db: Session, meeting_id: str) -> bool:
    """
    Si la reunion tiene transcripcion con segmentos, crea Summary (pending) si no existe
    y encola process_meeting_summary para que Cosmos genere resumen e IA.
    No hace commit; el llamador debe haber hecho commit ya.
    Returns True si se encolo la tarea, False si no (ya habia summary completado o error).
    """
    from app.tasks.summary_tasks import process_meeting_summary

    summary = db.query(Summary).filter(Summary.meeting_id == meeting_id).first()
    if summary and summary.processing_status == "completed" and getattr(summary, "is_final", False):
        logger.info("Reunion %s ya tiene summary completado, no se reencola", meeting_id)
        return False
    if not summary:
        summary = Summary(
            id=str(uuid.uuid4()),
            meeting_id=meeting_id,
            processing_status="pending",
        )
        db.add(summary)
        db.flush()
        logger.info("Summary creado (pending) para reunion %s, encolando Cosmos", meeting_id)
    else:
        summary.processing_status = "pending"
        summary.error_message = None
        logger.info("Summary existente puesto a pending para reunion %s, encolando Cosmos", meeting_id)
    db.commit()
    try:
        process_meeting_summary.apply_async(args=[meeting_id], queue="summary_queue")
        logger.info("Tarea process_meeting_summary (Cosmos) encolada para reunion %s", meeting_id)
        return True
    except Exception as e:
        logger.warning("No se pudo encolar process_meeting_summary para reunion %s: %s", meeting_id, e)
        return False


class DatabaseTask(Task):
    """Task base class that provides database session."""

    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Hook llamado cuando una tarea falla. Para join_bot_to_meeting, guardar motivo en la reunion."""
        logger.error(
            "[CELERY] Tarea %s fallo: %s",
            task_id,
            exc,
            exc_info=einfo,
        )
        logger.error("[CELERY] Args: %s, Kwargs: %s", args, kwargs)
        if args and getattr(self, "name", "").endswith("join_bot_to_meeting"):
            meeting_id = args[0] if isinstance(args[0], str) else None
            if meeting_id:
                try:
                    session = SessionLocal()
                    meeting = session.query(Meeting).filter(Meeting.id == meeting_id).first()
                    if meeting:
                        err_msg = str(exc)
                        if not err_msg or len(err_msg) > 2000:
                            err_msg = (err_msg or "Error desconocido")[:2000]
                        meeting.status = MeetingStatus.FAILED
                        meeting.error_message = _user_friendly_join_error(err_msg)
                        session.commit()
                        logger.error(
                            "[CELERY] Reunion %s marcada como FAILED (error guardado): %s",
                            meeting_id,
                            (meeting.error_message or "")[:200],
                        )
                    session.close()
                except Exception as e:
                    logger.exception("[CELERY] No se pudo guardar error en reunion %s: %s", meeting_id, e)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Hook llamado cuando una tarea se reintenta."""
        logger.warning(
            f"[CELERY] Tarea {task_id} se reintenta (intento {self.request.retries + 1}/{self.max_retries}): {exc}"
        )

    def after_return(self, *args, **kwargs):
        """Close database session after task completes."""
        if self._db is not None:
            self._db.close()
            self._db = None


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def save_vexa_transcript_to_db(db: Session, meeting: Meeting, data: Dict[str, Any]) -> int:
    """
    Guarda la respuesta de VEXA (get_transcript) en Transcription y TranscriptionSegment,
    actualiza meeting.status a COMPLETED y meeting.actual_end_time.
    No hace commit; el llamador debe hacer db.commit().

    Args:
        db: sesion de BD
        meeting: reunion (debe tener recall_bot_id)
        data: dict devuelto por VexaService.get_transcript (con clave "segments")

    Returns:
        Numero de segmentos guardados (0 si no hay segmentos validos).
    """
    segments = data.get("segments") if isinstance(data, dict) else []
    if not segments:
        return 0

    def _abs_start(s):
        return (s.get("absolute_start_time") or "") or ""

    segments = sorted(segments, key=_abs_start)
    base_time: Optional[datetime] = None
    segs_for_db = []
    for s in segments:
        if not isinstance(s, dict):
            continue
        text = (s.get("text") or "").strip()
        if not text:
            continue
        start_str = s.get("absolute_start_time")
        end_str = s.get("absolute_end_time")
        if not start_str or not end_str:
            continue
        start_dt = _parse_iso(start_str)
        end_dt = _parse_iso(end_str)
        if start_dt is None or end_dt is None:
            continue
        if base_time is None:
            base_time = start_dt
        start_sec = (start_dt - base_time).total_seconds()
        end_sec = (end_dt - base_time).total_seconds()
        if start_sec < 0:
            start_sec = 0
        if end_sec <= start_sec:
            end_sec = start_sec + 0.1
        speaker = s.get("speaker") or "Unknown"
        segs_for_db.append({
            "speaker_id": speaker if speaker.startswith("Speaker") else f"Speaker:{speaker}",
            "speaker_name": speaker,
            "text": text,
            "start_time": start_sec,
            "end_time": end_sec,
            "duration": end_sec - start_sec,
        })

    if not segs_for_db:
        return 0

    meeting_id = str(meeting.id)
    existing = db.query(Transcription).filter(Transcription.meeting_id == meeting_id).first()
    if existing:
        # Verificar si hay segmentos existentes antes de reemplazar
        existing_segments_count = db.query(TranscriptionSegment).filter(
            TranscriptionSegment.transcription_id == existing.id
        ).count()
        
        new_segments_count = len(segs_for_db)
        
        # Solo reemplazar si hay más segmentos nuevos que los existentes
        if new_segments_count > existing_segments_count:
            logger.info(
                "[CELERY] Reunion %s: Reemplazando transcripcion (%d segmentos existentes -> %d segmentos nuevos)",
                meeting_id,
                existing_segments_count,
                new_segments_count,
            )
            db.query(TranscriptionSegment).filter(TranscriptionSegment.transcription_id == existing.id).delete()
            transcription = existing
            transcription.raw_transcript_json = data
            transcription.total_segments = len(segs_for_db)
            transcription.total_duration_seconds = max(s["end_time"] for s in segs_for_db) if segs_for_db else 0
            transcription.is_final = True
            transcription.updated_at = datetime.now(timezone.utc)
        else:
            logger.info(
                "[CELERY] Reunion %s: Manteniendo transcripcion existente (%d segmentos existentes >= %d segmentos nuevos)",
                meeting_id,
                existing_segments_count,
                new_segments_count,
            )
            # No reemplazar, mantener los existentes
            transcription = existing
            # Actualizar solo el timestamp de actualización
            transcription.updated_at = datetime.now(timezone.utc)
            # Retornar el número de segmentos existentes en lugar de los nuevos
            return existing_segments_count
    else:
        transcription = Transcription(
            meeting_id=meeting_id,
            raw_transcript_json=data,
            total_segments=len(segs_for_db),
            total_duration_seconds=max(s["end_time"] for s in segs_for_db) if segs_for_db else 0,
            is_final=True,
            language="es",
        )
        db.add(transcription)
        db.flush()

    for seg in segs_for_db:
        db.add(TranscriptionSegment(
            transcription_id=transcription.id,
            speaker_id=seg["speaker_id"],
            speaker_name=seg.get("speaker_name"),
            text=seg["text"],
            start_time=seg["start_time"],
            end_time=seg["end_time"],
            duration=seg["duration"],
        ))

    meeting.status = MeetingStatus.COMPLETED
    meeting.actual_end_time = meeting.actual_end_time or datetime.now(timezone.utc)
    return len(segs_for_db)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
)
def join_bot_to_meeting(self, meeting_id: str):
    """
    Tarea Celery para unir el bot VEXA a una reunion de Teams.

    Extrae native_meeting_id y passcode de meeting_url, llama a VEXA POST /bots
    y guarda el native_meeting_id en recall_bot_id para obtener transcripciones despues.
    """
    try:
        logger.info(
            "[CELERY] Iniciando join_bot_to_meeting (VEXA) para reunion %s",
            meeting_id,
        )
    except Exception as e:
        logger.error("[CELERY] Error en logging inicial: %s", e, exc_info=True)

    db = self.db
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()

    if not meeting:
        logger.error("Reunion %s no encontrada", meeting_id)
        return {"success": False, "error": "Meeting not found"}

    if meeting.status != MeetingStatus.PENDING:
        logger.warning(
            "Reunion %s no esta en PENDING (actual: %s)",
            meeting_id,
            meeting.status,
        )
        return {"success": False, "error": f"Meeting status is {meeting.status}, expected PENDING"}

    if meeting.recall_bot_id:
        logger.info(
            "Reunion %s ya tiene bot VEXA (native_id=%s), no se crea otro",
            meeting_id,
            meeting.recall_bot_id,
        )
        return {
            "success": True,
            "meeting_id": meeting_id,
            "bot_id": meeting.recall_bot_id,
            "message": "Bot ya existe",
        }

    existing_bot_meeting = db.query(Meeting).filter(
        Meeting.meeting_url == meeting.meeting_url,
        Meeting.recall_bot_id.isnot(None),
        Meeting.id != meeting_id,
        Meeting.deleted_at.is_(None),
        Meeting.status.notin_([MeetingStatus.COMPLETED, MeetingStatus.FAILED, MeetingStatus.CANCELLED]),
    ).first()

    if existing_bot_meeting:
        logger.info(
            "Reunion %s reutiliza bot de reunion %s (native_id=%s)",
            meeting_id,
            existing_bot_meeting.id,
            existing_bot_meeting.recall_bot_id,
        )
        meeting.recall_bot_id = existing_bot_meeting.recall_bot_id
        meeting.status = MeetingStatus.JOINING
        meeting.recall_status = existing_bot_meeting.recall_status or "active"
        db.commit()
        return {
            "success": True,
            "meeting_id": meeting_id,
            "bot_id": existing_bot_meeting.recall_bot_id,
            "message": "Bot reutilizado",
        }

    now = datetime.now(timezone.utc)
    scheduled_time = meeting.scheduled_start_time
    if scheduled_time.tzinfo is None:
        scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
    time_diff = (scheduled_time - now).total_seconds()

    if time_diff > 3600:
        logger.warning(
            "Reunion %s programada para mas de 1 hora, re-programando",
            meeting_id,
        )
        join_time = scheduled_time - timedelta(minutes=1)
        try:
            task = join_bot_to_meeting.apply_async(args=[meeting_id], eta=join_time)
            meeting.celery_task_id = task.id
            db.commit()
            return {
                "success": True,
                "meeting_id": meeting_id,
                "message": "Task rescheduled",
                "scheduled_for": join_time.isoformat(),
            }
        except Exception as e:
            logger.error("Error re-programando tarea: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}

    if time_diff < -3600:
        logger.warning("Reunion %s ya paso hace mas de 60 minutos (time_diff: %s)", meeting_id, time_diff)
        meeting.status = MeetingStatus.FAILED
        meeting.error_message = (
                "La reunion programada ya paso hace mas de 60 minutos. "
                "El bot solo puede unirse a reuniones que comenzaron recientemente o estan en curso."
            )
        db.commit()
        return {"success": False, "error": "Meeting start time has passed more than 60 minutes ago"}

    native_meeting_id: Optional[str] = None
    passcode: Optional[str] = None
    if meeting.extra_metadata and isinstance(meeting.extra_metadata, dict):
        native_meeting_id = (meeting.extra_metadata or {}).get("vexa_native_meeting_id") or (meeting.extra_metadata or {}).get("native_meeting_id")
        passcode = (meeting.extra_metadata or {}).get("vexa_passcode") or (meeting.extra_metadata or {}).get("passcode")
        if native_meeting_id:
            logger.info("[CELERY] Usando native_meeting_id extraido de extra_metadata: %s", native_meeting_id)

    if not native_meeting_id or not passcode:
        parsed_id, parsed_pw = parse_teams_meeting_url(meeting.meeting_url or "")
        if not native_meeting_id:
            native_meeting_id = parsed_id
        if not passcode:
            passcode = parsed_pw
            
    if not native_meeting_id:
        logger.error("No se pudo extraer native_meeting_id para reunion %s (URL: %s)", meeting_id, meeting.meeting_url)
        meeting.status = MeetingStatus.FAILED
        meeting.error_message = (
                "El enlace de la reunion de Teams no contiene un ID valido y no se encontro en la descripcion. "
                "Comprueba que el enlace sea de una reunion de Microsoft Teams."
            )
        db.commit()
        return {"success": False, "error": "Could not parse Teams meeting ID from URL"}

    # Validar que el native_meeting_id sea numerico (VEXA solo acepta IDs de 10-15 digitos)
    # Los IDs ahora se obtienen directamente del cuerpo del mensaje, por lo que siempre deben ser numericos
    if not native_meeting_id.strip().isdigit():
        logger.error(
            "native_meeting_id no es numerico para reunion %s: %s. "
            "VEXA requiere un ID numerico de 10-15 digitos.",
            meeting_id,
            native_meeting_id[:50]
        )
        meeting.status = MeetingStatus.FAILED
        meeting.error_message = (
            "El ID de la reunion no es numerico. VEXA requiere un ID numerico de 10-15 digitos. "
            "Verifica que el ID se este extrayendo correctamente del mensaje."
        )
        db.commit()
        return {
            "success": False,
            "error": "Meeting ID no es numerico. VEXA requiere ID numerico de 10-15 digitos"
        }

    bot_name: Optional[str] = None
    if meeting.extra_metadata and isinstance(meeting.extra_metadata, dict):
        bot_name = (meeting.extra_metadata or {}).get("bot_display_name")
    if not bot_name or not str(bot_name).strip():
        bot_name = getattr(settings, "default_bot_name", None) or "Notetaker"

    try:
        vexa = VexaService()
        vexa.start_teams_bot(
            native_meeting_id=native_meeting_id,
            passcode=passcode,
            bot_name=bot_name,
        )
    except VexaServiceError as e:
        logger.exception("VEXA start_teams_bot failed: %s", e)
        meeting.status = MeetingStatus.FAILED
        meeting.error_message = _user_friendly_join_error(str(e))
        db.commit()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        return {"success": False, "error": str(e)}

    meeting.recall_bot_id = native_meeting_id
    meeting.recall_status = "active"
    meeting.status = MeetingStatus.IN_PROGRESS
    meeting.actual_start_time = datetime.now(timezone.utc)
    if not meeting.extra_metadata:
        meeting.extra_metadata = {}
    meeting.extra_metadata["vexa_native_meeting_id"] = native_meeting_id
    if passcode:
        meeting.extra_metadata["vexa_passcode"] = passcode
    db.commit()

    # NOTA: Ya no programamos la transcripcion basada en scheduled_end_time aqui.
    # El webhook de Vexa (bot-exited) se encargara de programar la transcripcion
    # 5 minutos despues de que el bot abandone la reunion.
    # Esto es mas preciso porque sabemos exactamente cuando el bot sale.

    return {
        "success": True,
        "meeting_id": meeting_id,
        "bot_id": native_meeting_id,
        "message": "Bot VEXA unido a la reunion",
    }


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    task_time_limit=900,  # 15 minutos para reuniones muy largas (permite timeout HTTP de 180s + procesamiento)
    task_soft_time_limit=840,  # 14 minutos soft limit
)
def fetch_vexa_transcript_for_meeting(self, meeting_id: str, force_full: bool = True) -> Dict[str, Any]:
    """
    Obtiene la transcripcion de VEXA para una reunion y la guarda en BD.
    Por defecto usa timeout extendido (180s) para obtener mas segmentos en reuniones largas.

    meeting_id debe ser el UUID de la reunion en Notetaker (Meeting.id), no el
    native_meeting_id de VEXA. Se usa meeting.recall_bot_id para llamar a VEXA.
    Convierte segmentos VEXA a Transcription/TranscriptionSegment y marca reunion
    COMPLETED si hay segmentos.

    Args:
        meeting_id: UUID de la reunion en Notetaker
        force_full: Si es True, usa timeout extendido (180s) en lugar de 30s para obtener mas segmentos
    """
    logger.info(
        "[CELERY] fetch_vexa_transcript_for_meeting reunion %s (Notetaker meeting_id, force_full=%s)",
        meeting_id,
        force_full
    )
    db = self.db
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        # Fallback: si meeting_id parece ID de VEXA (numerico), buscar por recall_bot_id
        if meeting_id and meeting_id.strip().isdigit():
            meeting = db.query(Meeting).filter(Meeting.recall_bot_id == meeting_id.strip()).first()
            if meeting:
                logger.warning(
                    "fetch_vexa_transcript_for_meeting recibio id tipo VEXA (%s); se uso reunion Notetaker %s. "
                    "Programar siempre con Notetaker meeting_id (UUID).",
                    meeting_id,
                    meeting.id,
                )
        if not meeting:
            logger.warning(
                "Meeting not found para id=%s (debe ser UUID de Notetaker). "
                "Si se programa con native_meeting_id de VEXA, no se encontrara.",
                meeting_id,
            )
            return {"success": False, "error": "Meeting not found"}

    native_id = meeting.recall_bot_id
    if not native_id:
        logger.warning("Reunion %s no tiene recall_bot_id (native_meeting_id)", meeting_id)
        return {"success": False, "error": "No VEXA bot id for meeting"}

    try:
        vexa = VexaService()
        # Usar timeout extendido (180s) para obtener mas segmentos en reuniones largas
        # VEXA necesita mas tiempo para procesar/preparar respuestas con muchos segmentos
        data = vexa.get_transcript(
            native_meeting_id=native_id,
            platform="teams",
            use_extended_timeout=force_full  # Usar timeout extendido si force_full=True
        )
    except VexaServiceError as e:
        logger.warning("VEXA get_transcript failed: %s", e)
        return {"success": False, "error": str(e)}

    segments = data.get("segments") if isinstance(data, dict) else []
    if not segments:
        logger.info("Reunion %s: sin segmentos aun en VEXA", meeting_id)
        return {"success": True, "meeting_id": meeting_id, "segments_saved": 0}

    # Verificar si la transcripcion parece incompleta comparando con la duracion programada
    if meeting.scheduled_start_time and meeting.scheduled_end_time:
        expected_duration_minutes = (
            (meeting.scheduled_end_time - meeting.scheduled_start_time).total_seconds() / 60
        )
        if segments:
            last_segment = segments[-1]
            first_segment = segments[0]
            last_end_time_str = last_segment.get("absolute_end_time")
            first_start_time_str = first_segment.get("absolute_start_time")
            
            if last_end_time_str and first_start_time_str:
                try:
                    from datetime import timezone
                    last_end_dt = datetime.fromisoformat(last_end_time_str.replace("Z", "+00:00"))
                    first_start_dt = datetime.fromisoformat(first_start_time_str.replace("Z", "+00:00"))
                    actual_duration_minutes = (last_end_dt - first_start_dt).total_seconds() / 60
                    
                    # Calcular diferencia con el scheduled_end_time
                    scheduled_end_dt = meeting.scheduled_end_time
                    if scheduled_end_dt.tzinfo is None:
                        scheduled_end_dt = scheduled_end_dt.replace(tzinfo=timezone.utc)
                    
                    time_diff_minutes = (scheduled_end_dt - last_end_dt).total_seconds() / 60
                    
                    logger.info(
                        "[CELERY] Reunion %s: Duracion - Esperada: %.1f min, Transcripcion: %.1f min (%.1f%%), "
                        "Diferencia con fin programado: %.1f min",
                        meeting_id,
                        expected_duration_minutes,
                        actual_duration_minutes,
                        (actual_duration_minutes / expected_duration_minutes * 100) if expected_duration_minutes > 0 else 0,
                        time_diff_minutes
                    )
                    
                    # Si la duracion real es significativamente menor que la esperada (menos del 90%)
                    # O si el ultimo timestamp esta mas de 2 minutos antes del scheduled_end_time
                    # Programar retry automatico para obtener los segmentos faltantes
                    if actual_duration_minutes < expected_duration_minutes * 0.9 or time_diff_minutes > 2:
                        logger.warning(
                            "[CELERY] Reunion %s: TRANSCRIPCION POSIBLEMENTE INCOMPLETA. "
                            "Esperada: %.1f min, Obtenida: %.1f min (%.1f%%), "
                            "Ultimo segmento termina %.1f min antes del fin programado. "
                            "Programando retry automatico en 3 minutos para obtener segmentos faltantes.",
                            meeting_id,
                            expected_duration_minutes,
                            actual_duration_minutes,
                            (actual_duration_minutes / expected_duration_minutes * 100) if expected_duration_minutes > 0 else 0,
                            time_diff_minutes
                        )
                        # Programar retry automatico: esperar 3 minutos adicionales para que VEXA procese los ultimos segmentos
                        retry_time = datetime.now(timezone.utc) + timedelta(minutes=3)
                        try:
                            fetch_vexa_transcript_for_meeting.apply_async(
                                args=[meeting_id],
                                kwargs={"force_full": True},
                                eta=retry_time,
                            )
                            logger.info(
                                "[CELERY] Retry automatico programado para reunion %s en %s (para capturar segmentos faltantes)",
                                meeting_id,
                                retry_time.isoformat()
                            )
                        except Exception as e:
                            logger.error(
                                "[CELERY] Error programando retry automatico para reunion %s: %s",
                                meeting_id,
                                e,
                                exc_info=True
                            )
                except Exception as e:
                    logger.warning("Error calculando duracion de transcripcion: %s", e, exc_info=True)

    n = save_vexa_transcript_to_db(db, meeting, data)
    db.commit()
    logger.info(
        "[CELERY] Transcripcion VEXA guardada para reunion %s (%s segmentos, timeout_extendido=%s)",
        meeting_id,
        n,
        force_full
    )
    schedule_summary_after_transcription(db, meeting_id)
    return {"success": True, "meeting_id": meeting_id, "segments_saved": n, "extended_timeout_used": force_full}


@celery_app.task(
    bind=True,
    base=DatabaseTask,
)
def check_stuck_meetings(self):
    """Marca reuniones en JOINING que llevan mas de 15 minutos como FAILED."""
    logger.info("[CELERY] Verificando reuniones atascadas en JOINING...")
    db = self.db
    now = datetime.now(timezone.utc)
    timeout_minutes = 15
    timeout_threshold = now - timedelta(minutes=timeout_minutes)

    stuck_meetings = db.query(Meeting).filter(
        Meeting.status == MeetingStatus.JOINING,
        Meeting.scheduled_start_time < timeout_threshold,
        Meeting.deleted_at.is_(None),
    ).all()

    for meeting in stuck_meetings:
        meeting.status = MeetingStatus.FAILED
        meeting.error_message = (
            f"El bot no se unio a la reunion despues de {timeout_minutes} minutos. "
            "Posiblemente nadie invito al bot a la reunion o hubo un error."
        )
        meeting.recall_status = "timeout"
    if stuck_meetings:
        db.commit()
    logger.info("[CELERY] Reuniones atascadas actualizadas: %s", len(stuck_meetings))
    return {"checked": len(stuck_meetings), "updated": len(stuck_meetings)}


@celery_app.task(
    bind=True,
    base=DatabaseTask,
)
def check_and_finalize_completed_meetings(self):
    """
    Verifica reuniones en IN_PROGRESS que deberian haber terminado y las finaliza
    como fallback en caso de que el webhook de Vexa no se haya ejecutado.
    
    NOTA: Esta tarea es ahora un fallback de seguridad. Normalmente el webhook
    bot-exited de Vexa se encarga de programar la transcripcion cuando el bot sale.
    
    Busca reuniones que:
    - Estan en estado IN_PROGRESS
    - Tienen scheduled_end_time hace mas de 15 minutos (mas tiempo para dar margen al webhook)
    - Tienen recall_bot_id (native_meeting_id de VEXA)
    - NO tienen transcript_task_id (no se ha programado transcripcion aun)
    
    Para cada una, verifica el estado en VEXA y si esta completada, programa la transcripcion.
    """
    logger.info("[CELERY] Verificando reuniones que deberian estar completadas (fallback)...")
    db = self.db
    now = datetime.now(timezone.utc)
    grace_period_minutes = 15  # Aumentado a 15 minutos para dar mas margen al webhook
    threshold = now - timedelta(minutes=grace_period_minutes)
    
    # Buscar reuniones en IN_PROGRESS que deberian haber terminado y NO tienen tarea programada
    meetings_to_check = db.query(Meeting).filter(
        Meeting.status == MeetingStatus.IN_PROGRESS,
        Meeting.scheduled_end_time.isnot(None),
        Meeting.scheduled_end_time < threshold,
        Meeting.recall_bot_id.isnot(None),
        Meeting.transcript_task_id.is_(None),  # Solo las que NO tienen tarea programada
        Meeting.deleted_at.is_(None),
    ).all()
    
    logger.info(
        "[CELERY] Encontradas %s reuniones en IN_PROGRESS sin tarea de transcripcion programada "
        "(posible fallo del webhook)",
        len(meetings_to_check),
    )
    
    finalized_count = 0
    error_count = 0
    
    for meeting in meetings_to_check:
        native_id = meeting.recall_bot_id
        if not native_id:
            continue
        
        try:
            vexa = VexaService()
            # Verificar estado en VEXA
            vexa_status = vexa.get_meeting_status(
                native_meeting_id=native_id,
                platform="teams",
            )
            
            if not vexa_status:
                # No encontrada en VEXA, asumir que esta completada (fallback)
                logger.warning(
                    "Reunion %s (native_id=%s): No encontrada en VEXA y sin tarea programada. "
                    "Programando transcripcion como fallback.",
                    meeting.id,
                    native_id,
                )
                
                # Obtener tiempo de espera configurado (por defecto 5 minutos)
                wait_minutes = getattr(settings, "transcript_wait_minutes", 5)
                
                # Calcular cuándo se ejecutará la tarea (X minutos desde ahora)
                scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=wait_minutes)
                
                try:
                    task = fetch_vexa_transcript_for_meeting.apply_async(
                        args=[str(meeting.id)],
                        kwargs={"force_full": True},
                        eta=scheduled_time,
                    )
                    
                    meeting.transcript_task_id = task.id
                    meeting.transcript_scheduled_time = scheduled_time
                    meeting.status = MeetingStatus.COMPLETED
                    meeting.actual_end_time = meeting.actual_end_time or datetime.now(timezone.utc)
                    db.commit()
                    
                    logger.info(
                        "Reunion %s: Tarea de transcripcion programada como fallback para %s (espera=%d min)",
                        meeting.id,
                        scheduled_time.isoformat(),
                        wait_minutes,
                    )
                    finalized_count += 1
                except Exception as e:
                    logger.error(
                        "Reunion %s: Error programando tarea de transcripcion (fallback): %s",
                        meeting.id,
                        e,
                        exc_info=True,
                    )
                    error_count += 1
                continue
            
            # Verificar si el estado en VEXA es "completed" o "failed"
            vexa_meeting_status = vexa_status.get("status", "").lower()
            logger.info(
                "Reunion %s (native_id=%s): Estado en VEXA = %s (fallback check)",
                meeting.id,
                native_id,
                vexa_meeting_status,
            )
            
            if vexa_meeting_status in ("completed", "failed"):
                # La reunion esta completada en VEXA pero no se recibio el webhook
                logger.warning(
                    "Reunion %s: Completada en VEXA (status=%s) pero sin webhook recibido. "
                    "Programando transcripcion como fallback.",
                    meeting.id,
                    vexa_meeting_status,
                )
                
                # Obtener tiempo de espera configurado (por defecto 5 minutos)
                wait_minutes = getattr(settings, "transcript_wait_minutes", 5)
                
                # Calcular cuándo se ejecutará la tarea (X minutos desde ahora)
                scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=wait_minutes)
                
                try:
                    task = fetch_vexa_transcript_for_meeting.apply_async(
                        args=[str(meeting.id)],
                        kwargs={"force_full": True},
                        eta=scheduled_time,
                    )
                    
                    meeting.transcript_task_id = task.id
                    meeting.transcript_scheduled_time = scheduled_time
                    meeting.status = MeetingStatus.COMPLETED
                    meeting.actual_end_time = meeting.actual_end_time or datetime.now(timezone.utc)
                    db.commit()
                    
                    logger.info(
                        "Reunion %s: Tarea de transcripcion programada como fallback para %s (task_id=%s, espera=%d min)",
                        meeting.id,
                        scheduled_time.isoformat(),
                        task.id,
                        wait_minutes,
                    )
                    finalized_count += 1
                except Exception as e:
                    logger.error(
                        "Reunion %s: Error programando tarea de transcripcion (fallback): %s",
                        meeting.id,
                        e,
                        exc_info=True,
                    )
                    error_count += 1
            elif vexa_meeting_status in ("active", "in_progress", "stopping"):
                # Aun esta activa en VEXA, esperar mas tiempo
                logger.debug(
                    "Reunion %s: Aun activa en VEXA (status=%s), esperando...",
                    meeting.id,
                    vexa_meeting_status,
                )
            else:
                logger.info(
                    "Reunion %s: Estado desconocido en VEXA: %s",
                    meeting.id,
                    vexa_meeting_status,
                )
                
        except VexaServiceError as e:
            logger.warning(
                "Reunion %s: Error verificando estado en VEXA (fallback): %s",
                meeting.id,
                e,
            )
            error_count += 1
        except Exception as e:
            logger.exception(
                "Reunion %s: Error inesperado verificando estado (fallback): %s",
                meeting.id,
                e,
            )
            error_count += 1
    
    logger.info(
        "[CELERY] Verificacion fallback completada: %s finalizadas, %s errores",
        finalized_count,
        error_count,
    )
    return {
        "checked": len(meetings_to_check),
        "finalized": finalized_count,
        "errors": error_count,
    }


@celery_app.task
def run_check_system_health():
    """
    Tarea periodica (Beat) que ejecuta la verificacion de salud del sistema
    y envia alertas por email si aplica (cola Celery, workers caidos, pool BD).
    """
    try:
        from app.services.monitoring_service import check_and_alert_system_health
        check_and_alert_system_health()
        logger.debug("[CELERY] check_and_alert_system_health ejecutado")
    except Exception as e:
        logger.error(f"[CELERY] Error en run_check_system_health: {e}", exc_info=True)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
)
def check_and_redownload_incomplete_transcripts(self):
    """
    Tarea periodica que revisa reuniones pasadas completadas que pueden tener
    transcripciones incompletas y las vuelve a descargar con timeout extendido.

    Busca reuniones que:
    - Estan en estado COMPLETED
    - Tienen transcripcion guardada
    - Tienen scheduled_end_time hace mas de 20 minutos (para dar tiempo a VEXA)
    - La duracion de la transcripcion guardada es significativamente menor que la duracion programada
    """
    logger.info("[CELERY] Verificando reuniones con transcripciones posiblemente incompletas...")
    db = self.db
    now = datetime.now(timezone.utc)
    grace_period_minutes = 20  # Esperar 20 minutos despues del fin programado
    threshold = now - timedelta(minutes=grace_period_minutes)
    
    # Buscar reuniones completadas con transcripcion que terminaron hace mas de 20 minutos
    meetings_to_check = db.query(Meeting).filter(
        Meeting.status == MeetingStatus.COMPLETED,
        Meeting.scheduled_end_time.isnot(None),
        Meeting.scheduled_end_time < threshold,
        Meeting.recall_bot_id.isnot(None),
        Meeting.deleted_at.is_(None),
    ).all()
    
    logger.info(
        "[CELERY] Encontradas %s reuniones completadas para verificar",
        len(meetings_to_check)
    )
    
    redownloaded_count = 0
    skipped_count = 0
    error_count = 0
    
    for meeting in meetings_to_check:
        # Verificar si tiene transcripcion guardada
        transcription = db.query(Transcription).filter(
            Transcription.meeting_id == str(meeting.id)
        ).first()
        
        if not transcription or not transcription.total_segments:
            logger.debug(
                "[CELERY] Reunion %s: sin transcripcion guardada, omitiendo",
                meeting.id
            )
            skipped_count += 1
            continue
        
        # Calcular duracion esperada vs duracion real de la transcripcion
        if not meeting.scheduled_start_time or not meeting.scheduled_end_time:
            skipped_count += 1
            continue
        
        expected_duration_minutes = (
            (meeting.scheduled_end_time - meeting.scheduled_start_time).total_seconds() / 60
        )
        actual_duration_minutes = (
            transcription.total_duration_seconds / 60
            if transcription.total_duration_seconds
            else 0
        )
        
        # Si la transcripcion es significativamente mas corta (menos del 80% de lo esperado)
        # y la reunion duro mas de 40 minutos (para evitar falsos positivos en reuniones cortas)
        if expected_duration_minutes > 40 and actual_duration_minutes < expected_duration_minutes * 0.8:
            # Cooldown: no re-encolar si ya intentamos re-descargar hace menos de 24 horas.
            # Evita bucle infinito donde las mismas reuniones se re-descargan cada hora sin mejorar.
            redownload_cooldown_hours = 24
            transcription_updated = getattr(transcription, "updated_at", None)
            if transcription_updated:
                if transcription_updated.tzinfo is None:
                    transcription_updated = transcription_updated.replace(tzinfo=timezone.utc)
                hours_since_update = (now - transcription_updated).total_seconds() / 3600
                if hours_since_update < redownload_cooldown_hours:
                    logger.debug(
                        "[CELERY] Reunion %s: transcripcion corta pero ya intentada re-descarga hace %.1f h "
                        "(cooldown %d h). Omitiendo.",
                        meeting.id,
                        hours_since_update,
                        redownload_cooldown_hours,
                    )
                    skipped_count += 1
                    continue

            logger.info(
                "[CELERY] Reunion %s: transcripcion posiblemente incompleta. "
                "Esperada: %.1f min, Actual: %.1f min (%.1f%%). Re-descargando con timeout extendido...",
                meeting.id,
                expected_duration_minutes,
                actual_duration_minutes,
                (actual_duration_minutes / expected_duration_minutes * 100) if expected_duration_minutes > 0 else 0
            )

            try:
                # Encolar tarea para re-descargar con timeout extendido
                fetch_vexa_transcript_for_meeting.apply_async(
                    args=[str(meeting.id)],
                    kwargs={"force_full": True},
                )
                redownloaded_count += 1
                logger.info(
                    "[CELERY] Tarea de re-descarga encolada para reunion %s",
                    meeting.id
                )
            except Exception as e:
                logger.error(
                    "[CELERY] Error encolando re-descarga para reunion %s: %s",
                    meeting.id,
                    e,
                    exc_info=True
                )
                error_count += 1
        else:
            skipped_count += 1
            logger.debug(
                "[CELERY] Reunion %s: transcripcion parece completa (%.1f min de %.1f min esperados)",
                meeting.id,
                actual_duration_minutes,
                expected_duration_minutes
            )
    
    logger.info(
        "[CELERY] Verificacion de transcripciones incompletas completada: "
        "%s re-descargadas, %s omitidas, %s errores",
        redownloaded_count,
        skipped_count,
        error_count
    )
    
    return {
        "checked": len(meetings_to_check),
        "redownloaded": redownloaded_count,
        "skipped": skipped_count,
        "errors": error_count,
    }


@celery_app.task(
    bind=True,
    base=DatabaseTask,
)
def check_and_schedule_pending_meetings(self):
    """
    Tarea periodica que verifica reuniones PENDING sin celery_task_id asignado
    y programa la tarea join_bot_to_meeting para ellas.
    
    Esto cubre casos donde:
    - La reunion se creo antes de implementar la programacion automatica
    - Hubo un error al programar la tarea inicialmente
    - La reunion se reactivo pero no se reprogramo la tarea
    
    Busca reuniones que:
    - Estan en estado PENDING
    - No tienen celery_task_id asignado
    - No tienen recall_bot_id (no reutilizan bot compartido)
    - Tienen scheduled_start_time en el futuro
    - No estan eliminadas
    """
    logger.info("[CELERY] Verificando reuniones PENDING sin tarea Celery programada...")
    db = self.db
    now = datetime.now(timezone.utc)
    
    # Buscar reuniones PENDING sin celery_task_id
    meetings_to_schedule = db.query(Meeting).filter(
        Meeting.status == MeetingStatus.PENDING,
        Meeting.celery_task_id.is_(None),  # Sin tarea programada
        Meeting.recall_bot_id.is_(None),  # No reutiliza bot compartido
        Meeting.scheduled_start_time.isnot(None),
        Meeting.scheduled_start_time > now,  # Solo futuras
        Meeting.deleted_at.is_(None),
    ).all()
    
    logger.info(
        "[CELERY] Encontradas %d reuniones PENDING sin tarea Celery programada",
        len(meetings_to_schedule)
    )
    
    scheduled_count = 0
    skipped_count = 0
    error_count = 0
    
    for meeting in meetings_to_schedule:
        try:
            scheduled_start = meeting.scheduled_start_time  # type: ignore
            if scheduled_start.tzinfo is None:
                scheduled_start = scheduled_start.replace(tzinfo=timezone.utc)
            
            # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
            join_time = scheduled_start - timedelta(minutes=1)
            
            if join_time <= now:
                # Si ya pasó el tiempo o es muy pronto, ejecutar inmediatamente
                logger.info(
                    "[CELERY] Reunion %s: fecha programada muy proxima o pasada, ejecutando inmediatamente",
                    meeting.id
                )
                try:
                    join_bot_to_meeting.delay(str(meeting.id))
                    scheduled_count += 1
                    logger.info(
                        "[CELERY] Tarea join_bot_to_meeting enviada para ejecucion inmediata (reunion %s)",
                        meeting.id
                    )
                except Exception as e:
                    logger.error(
                        "[CELERY] Error enviando tarea inmediata para reunion %s: %s",
                        meeting.id,
                        e,
                        exc_info=True
                    )
                    error_count += 1
            else:
                # Programar para el futuro
                try:
                    task = join_bot_to_meeting.apply_async(
                        args=[str(meeting.id)],
                        eta=join_time
                    )
                    meeting.celery_task_id = task.id  # type: ignore
                    db.commit()
                    scheduled_count += 1
                    logger.info(
                        "[CELERY] Tarea Celery programada para reunion %s (task_id=%s, ejecutara el %s UTC)",
                        meeting.id,
                        task.id,
                        join_time.isoformat()
                    )
                except Exception as e:
                    logger.error(
                        "[CELERY] Error programando tarea Celery para reunion %s: %s",
                        meeting.id,
                        e,
                        exc_info=True
                    )
                    db.rollback()
                    error_count += 1
        except Exception as e:
            logger.error(
                "[CELERY] Error procesando reunion %s: %s",
                meeting.id,
                e,
                exc_info=True
            )
            error_count += 1
            skipped_count += 1
    
    logger.info(
        "[CELERY] Verificacion de reuniones PENDING completada: %d programadas, %d omitidas, %d errores",
        scheduled_count,
        skipped_count,
        error_count
    )
    
    return {
        "checked": len(meetings_to_schedule),
        "scheduled": scheduled_count,
        "skipped": skipped_count,
        "errors": error_count
    }

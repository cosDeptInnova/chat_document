"""Endpoints para gestión de reuniones."""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime, timedelta, timezone
from app.database import get_db
from app.models.meeting import Meeting, MeetingStatus
from app.models.user import User
from app.models.meeting_access import MeetingAccess
from app.models.transcription import Transcription, TranscriptionSegment
from app.config import settings
from app.services.calendar_sync_service import _cancel_all_celery_tasks_for_meeting  # type: ignore[attr-defined]
from app.services.vexa_service import VexaService, VexaServiceError, parse_teams_meeting_url
from app.tasks.meeting_tasks import save_vexa_transcript_to_db, schedule_summary_after_transcription
from app.tasks.summary_tasks import process_meeting_summary
import logging
import re
import uuid
import os
import base64
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/meetings", tags=["meetings"])


def get_current_organizer_name(meeting: Meeting, db: Session) -> Optional[str]:
    """
    Obtiene el nombre actual del organizador de una reunión.
    
    Si el organizador tiene un usuario en la BD con display_name actualizado,
    devuelve ese nombre. Si no, devuelve el organizer_name guardado.
    
    Args:
        meeting: Reunión de la que obtener el nombre del organizador
        db: Sesión de base de datos
        
    Returns:
        Nombre actual del organizador o None
    """
    organizer_email: Optional[str] = meeting.organizer_email  # type: ignore
    if not organizer_email:
        return meeting.organizer_name  # type: ignore
    
    # Buscar usuario por email del organizador
    organizer_user = db.query(User).filter(User.email == organizer_email).first()  # type: ignore
    
    if organizer_user:
        display_name: Optional[str] = organizer_user.display_name  # type: ignore
        if display_name:
            # Usar el display_name actual del usuario
            return display_name
    
    # Fallback: usar el organizer_name guardado
    return meeting.organizer_name  # type: ignore


def _meeting_duration_seconds(meeting: Meeting) -> Optional[float]:
    """
    Duracion total de la reunion en segundos (desde que el bot entro hasta que salio).
    Requiere actual_start_time y actual_end_time.
    """
    start = getattr(meeting, "actual_start_time", None)
    end = getattr(meeting, "actual_end_time", None)
    if start and end:
        delta = end - start
        return max(0.0, delta.total_seconds())
    return None


def _extract_basic_info_from_raw_transcript(raw_json: dict) -> tuple[int, Optional[float], List[str]]:
    """
    Extrae total_segments, total_duration_seconds y participantes del JSON raw.
    Soporta formato VEXA (segments) y formato legacy (utterances).
    Retorna (total_segments, total_duration_seconds, participants).
    """
    utterances: List[dict] = []
    if isinstance(raw_json, list):
        utterances = raw_json
    elif isinstance(raw_json, dict) and raw_json:
        if isinstance(raw_json.get("segments"), list):
            segs = [s for s in raw_json["segments"] if isinstance(s, dict)]
            total = len([s for s in segs if (s.get("text") or "").strip()])
            unique = list({str(s.get("speaker") or "Unknown").strip() for s in segs})
            max_end_sec: Optional[float] = None
            if segs:
                try:
                    first_start = (segs[0] or {}).get("absolute_start_time")
                    last_end = (segs[-1] or {}).get("absolute_end_time")
                    if first_start and last_end:
                        t0 = datetime.fromisoformat(str(first_start).replace("Z", "+00:00"))
                        t1 = datetime.fromisoformat(str(last_end).replace("Z", "+00:00"))
                        max_end_sec = (t1 - t0).total_seconds()
                except Exception:
                    pass
            return total, max_end_sec, sorted(unique) if unique else []
        if isinstance(raw_json.get("utterances"), list):
            utterances = raw_json["utterances"]
        elif isinstance(raw_json.get("transcript"), dict) and isinstance(
            (raw_json["transcript"] or {}).get("utterances"), list
        ):
            utterances = raw_json["transcript"]["utterances"]
        else:
            for k, v in raw_json.items():
                if isinstance(v, list) and v:
                    first = v[0]
                    if isinstance(first, dict) and (
                        first.get("text") or first.get("words") or first.get("speaker")
                        or first.get("start") is not None or first.get("start_time") is not None
                    ):
                        utterances = v
                        break
    if not utterances:
        return 0, None, []

    unique_speakers: List[str] = []
    seen: set = set()
    total = 0
    max_end: Optional[float] = None

    for u in utterances:
        if not isinstance(u, dict):
            continue
        text = u.get("text") or u.get("transcript") or ""
        if not text and isinstance(u.get("words"), list):
            text = " ".join(
                (w.get("word") or w.get("text") or "") for w in u["words"] if isinstance(w, dict)
            ).strip()
        if not text:
            continue

        part = u.get("participant")
        name_from_part = part.get("name") if isinstance(part, dict) else None
        speaker = (
            u.get("speaker_name") or u.get("speaker") or name_from_part
            or u.get("speaker_id") or "Speaker:unknown"
        )
        if isinstance(speaker, str) and speaker.strip() and speaker not in seen:
            seen.add(speaker)
            unique_speakers.append(speaker.strip())

        start = u.get("start") or u.get("start_time") or u.get("timestamp") or 0
        end = u.get("end") or u.get("end_time") or start
        if isinstance(u.get("words"), list) and u["words"]:
            fw = u["words"][0]
            lw = u["words"][-1]
            if isinstance(fw, dict) and isinstance(fw.get("start_timestamp"), dict):
                rel = (fw["start_timestamp"] or {}).get("relative")
                if rel is not None:
                    start = rel
            if isinstance(lw, dict) and isinstance(lw.get("end_timestamp"), dict):
                rel = (lw["end_timestamp"] or {}).get("relative")
                if rel is not None:
                    end = rel
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            start_f, end_f = 0.0, 0.0
        total += 1
        if max_end is None or end_f > max_end:
            max_end = end_f

    return total, max_end if max_end is not None else None, sorted(unique_speakers)


def can_user_delete_meeting(meeting: Meeting, db: Session) -> bool:
    """
    Determina si un usuario normal puede ver el botón de borrar una reunión.
    
    NOTA: El botón siempre debe aparecer para reuniones pasadas.
    La lógica de qué hacer al borrar (quitar acceso vs borrar reunión) está en delete_meeting.
    
    Reglas:
    - NO mostrar botón si está en JOINING o IN_PROGRESS (esperar a que termine)
    - SÍ mostrar botón en cualquier otro caso (incluyendo pasadas con contenido)
    
    Args:
        meeting: La reunión a verificar
        db: Sesión de base de datos (no usado, pero mantenido para compatibilidad)
        
    Returns:
        True si se puede mostrar el botón, False en caso contrario
    """
    # No permitir borrar si está en JOINING o IN_PROGRESS
    if meeting.status in [MeetingStatus.JOINING, MeetingStatus.IN_PROGRESS]:  # type: ignore
        return False
    
    # Para cualquier otro caso (incluyendo pasadas con contenido), mostrar el botón
    # La lógica de qué hacer al borrar está en delete_meeting
    return True


class CreateMeetingRequest(BaseModel):
    """Request para crear una reunión manualmente."""
    meeting_url: str = Field(..., description="URL completa de la reunión Teams")
    scheduled_start_time: datetime = Field(..., description="Fecha y hora programada de la reunión")
    scheduled_end_time: Optional[datetime] = Field(None, description="Fecha y hora de finalización (opcional)")
    title: Optional[str] = Field(None, description="Título de la reunión")
    organizer_email: Optional[str] = Field(None, description="Email del organizador (opcional)")
    organizer_name: Optional[str] = Field(None, description="Nombre del organizador (opcional)")
    bot_display_name: Optional[str] = Field(None, description="Nombre que usará el bot en la reunión")
    user_email: Optional[str] = Field(None, description="Email del usuario que crea la reunión (para identificación)")
    native_meeting_id: Optional[str] = Field(None, description="ID numerico de reunion Teams (formato largo corporativo)")
    passcode: Optional[str] = Field(None, description="Codigo de acceso/PIN de la reunion (formato largo corporativo)")


class TranscriptionBasicInfo(BaseModel):
    """Información básica de transcripción (disponible sin MeetingAccess)."""
    total_segments: int = 0
    total_duration_seconds: Optional[float] = None
    participants: List[str] = []
    has_transcription: bool = False


class MeetingResponse(BaseModel):
    """Response con información de la reunión."""
    id: str
    meeting_url: str
    title: Optional[str]
    scheduled_start_time: datetime
    scheduled_end_time: Optional[datetime]
    status: str
    created_at: datetime
    organizer_email: Optional[str]
    organizer_name: Optional[str]
    can_delete: bool = True  # Indica si el usuario puede borrar esta reunión
    transcription_basic_info: Optional[TranscriptionBasicInfo] = None  # Información básica disponible sin MeetingAccess
    necesita_resincronizar_calendario: Optional[bool] = None  # True si PENDING+Teams y faltan ID/passcode (solo en listado admin)
    total_meeting_duration_seconds: Optional[float] = None  # Duracion real reunion (bot entra -> bot sale)
    processing_status: Optional[str] = None  # Estado de procesamiento: "en_cola", "obteniendo_transcripcion", "generando_resumen", "completado", "en_progreso", "pendiente"
    estimated_completion_time: Optional[str] = None  # Timestamp ISO de cuándo se estima que estará lista
    time_remaining_seconds: Optional[int] = None  # Segundos restantes hasta la siguiente acción

    class Config:
        from_attributes = True


class MeetingJoinInfoResponse(BaseModel):
    """Diagnostico para comprobar si la reunion tiene ID y codigo extraidos (formato largo de Teams)."""
    meeting_id: str
    url_almacenada: str
    """URL guardada en la reunion (si tiene passcode, se muestra con ?p=*** por seguridad)."""
    native_meeting_id: Optional[str] = None
    """ID numerico que usara Vexa para unirse (extraido de la URL o de extra_metadata)."""
    passcode_presente: bool = False
    """True si hay codigo de acceso (en URL o extra_metadata)."""
    listo_para_vexa: bool = False
    """True si hay native_meeting_id valido (Vexa podra intentar unirse)."""
    origen: str = ""
    """'url' = extraido de la URL; 'extra_metadata' = del cuerpo del calendario; 'url_y_metadata' = combinado."""


def _meeting_join_readiness(meeting: Meeting) -> Tuple[Optional[str], bool, bool, str]:
    """
    Calcula si la reunion tiene ID y passcode para que Vexa pueda unirse.
    Returns: (native_meeting_id, passcode_presente, listo_para_vexa, origen).
    """
    url_almacenada = meeting.meeting_url or ""
    native_id: Optional[str] = None
    passcode_presente = False
    origen = ""
    if meeting.extra_metadata and isinstance(meeting.extra_metadata, dict):
        meta = meeting.extra_metadata
        native_id = meta.get("vexa_native_meeting_id") or meta.get("native_meeting_id")
        pw = meta.get("vexa_passcode") or meta.get("passcode")
        if pw:
            passcode_presente = True
        if native_id:
            origen = "extra_metadata"
    if not native_id or not passcode_presente:
        parsed_id, parsed_pw = parse_teams_meeting_url(url_almacenada)
        if not native_id:
            native_id = parsed_id
            if origen:
                origen = "url_y_metadata"
            elif native_id:
                origen = "url"
        if not passcode_presente and parsed_pw:
            passcode_presente = True
    listo = bool(native_id and (native_id.strip().isdigit() or len(native_id) >= 10))
    return (native_id, passcode_presente, listo, origen or ("url" if native_id else ""))


def _necesita_resincronizar_calendario(meeting: Meeting) -> bool:
    """True si reunion PENDING de Teams sin ID/passcode extraidos (el bot no se unira)."""
    if meeting.status != MeetingStatus.PENDING:
        return False
    url = (meeting.meeting_url or "").strip()
    if "teams.microsoft.com" not in url and "teams.live.com" not in url:
        return False
    _, passcode_presente, listo, _ = _meeting_join_readiness(meeting)
    return not (listo and passcode_presente)


def extract_thread_id_from_url(meeting_url: str) -> Optional[str]:
    """Extrae el thread_id de una URL de Teams."""
    # Patrones comunes de URLs de Teams
    patterns = [
        r'/l/meetup-join/([^/]+)',  # https://teams.microsoft.com/l/meetup-join/...
        r'/l/meeting/([^/?]+)',     # Otras variantes
        r'threadId=([^&]+)',        # threadId en query params
    ]
    
    for pattern in patterns:
        match = re.search(pattern, meeting_url)
        if match:
            return match.group(1)
    
    return None


@router.post("/create", response_model=MeetingResponse)
async def create_meeting_manual(
    request: CreateMeetingRequest,
    db: Session = Depends(get_db)
):
    """
    Crear una reunión manualmente sin necesidad de OAuth.
    
    El usuario solo necesita proporcionar la URL de la reunión Teams.
    El bot se unirá automáticamente a la hora programada usando la cuenta
    configurada en TEAMS_BOT_EMAIL.
    
    **NO requiere:**
    - OAuth
    - Permisos del usuario
    - Admin consent
    - Autorización de aplicación
    """
    try:
        # Validar que la URL sea de Teams
        if "teams.microsoft.com" not in request.meeting_url.lower() and "teams.live.com" not in request.meeting_url.lower():
            raise HTTPException(
                status_code=400,
                detail="La URL debe ser de Microsoft Teams"
            )
        
        # Validar que la fecha no sea en el pasado
        now = datetime.now(timezone.utc)
        scheduled_time = request.scheduled_start_time
        if scheduled_time.tzinfo is None:
            scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
        
        if scheduled_time < now:
            raise HTTPException(
                status_code=400,
                detail="No se puede crear una reunión con fecha en el pasado. La fecha de inicio debe ser futura."
            )
        
        # Extraer thread_id de la URL
        thread_id = extract_thread_id_from_url(request.meeting_url)
        
        # Buscar o crear usuario si se proporciona email
        user_id = None
        if request.user_email:
            user = db.query(User).filter(User.email == request.user_email).first()
            if not user:
                # Crear usuario básico sin OAuth (solo para identificación)
                user = User(
                    email=request.user_email,
                    display_name=request.user_email.split('@')[0],
                    microsoft_user_id=f"manual_{request.user_email}",
                    tenant_id="manual",
                    is_active=True
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                logger.info(f"Usuario manual creado: {request.user_email}")
            user_id = user.id  # type: ignore
        else:
            # Si no se proporciona email, usar un usuario "system" o crear uno por defecto
            # Por ahora, lanzamos error si no se proporciona
            raise HTTPException(
                status_code=400,
                detail="Se requiere user_email para identificar al usuario"
            )
        
        # Normalizar fecha/hora de inicio a UTC naive para comparar y guardar
        scheduled_start_naive = request.scheduled_start_time
        if scheduled_start_naive.tzinfo is not None:
            scheduled_start_naive = scheduled_start_naive.astimezone(timezone.utc).replace(tzinfo=None)
        
        # Comprobar si ya existe una reunión con la MISMA URL y MISMA FECHA/HORA (misma instancia)
        # Excluir reuniones eliminadas (soft delete)
        existing_meeting = (
            db.query(Meeting)
            .filter(
                Meeting.meeting_url == request.meeting_url,
                Meeting.scheduled_start_time == scheduled_start_naive,  # type: ignore
                Meeting.status.in_([
                    MeetingStatus.PENDING,
                    MeetingStatus.JOINING,
                    MeetingStatus.IN_PROGRESS,
                    MeetingStatus.COMPLETED,
                ]),
                Meeting.deleted_at.is_(None),  # No incluir reuniones eliminadas
            )
            .order_by(Meeting.created_at.desc())
            .first()
        )

        if existing_meeting:
            # Reutilizamos la reunión existente y solo gestionamos el acceso del usuario
            logger.info(
                "Reutilizando reunión existente %s para URL %s y usuario %s",
                existing_meeting.id,
                request.meeting_url,
                request.user_email,
            )
            # Si se envían ID/codigo (formato largo), actualizar extra_metadata para que el bot pueda unirse
            if request.native_meeting_id or request.passcode:
                meta = dict(existing_meeting.extra_metadata or {})
                if request.native_meeting_id and request.native_meeting_id.strip():
                    meta["vexa_native_meeting_id"] = request.native_meeting_id.strip()
                if request.passcode is not None and request.passcode.strip():
                    meta["vexa_passcode"] = request.passcode.strip()
                existing_meeting.extra_metadata = meta

            # Determinar permisos según licencia del usuario
            from app.utils.license_utils import get_user_license_level, get_meeting_access_permissions
            license_level = get_user_license_level(user)
            permissions = get_meeting_access_permissions(license_level)
            
            # Asegurar que el usuario tiene MeetingAccess
            access = db.query(MeetingAccess).filter(
                MeetingAccess.meeting_id == existing_meeting.id,
                MeetingAccess.user_id == user_id,
            ).first()

            if access:
                # Actualizar permisos según licencia actual
                access.can_view_transcript = permissions["can_view_transcript"]  # type: ignore
                access.can_view_audio = permissions["can_view_audio"]  # type: ignore
                access.can_view_video = permissions["can_view_video"]  # type: ignore
                logger.info(
                    "Actualizando permisos de acceso para usuario %s (licencia: %s) en meeting %s",
                    request.user_email,
                    license_level,
                    existing_meeting.id,
                )
            else:
                access = MeetingAccess(
                    meeting_id=existing_meeting.id,  # type: ignore
                    user_id=user_id,
                    can_view_transcript=permissions["can_view_transcript"],
                    can_view_audio=permissions["can_view_audio"],
                    can_view_video=permissions["can_view_video"],
                )
                db.add(access)
                logger.info(
                    "Creando acceso para usuario %s (licencia: %s) en meeting existente %s",
                    request.user_email,
                    license_level,
                    existing_meeting.id,
                )

            db.commit()

            # Si la reunión está en PENDING, verificar si necesita un bot programado
            if existing_meeting.status == MeetingStatus.PENDING:  # type: ignore
                # Función auxiliar para verificar si una tarea de Celery existe y es válida
                def verify_celery_task_exists(task_id: Optional[str]) -> bool:
                    if not task_id:
                        return False
                    try:
                        from app.celery_app import celery_app
                        result = celery_app.AsyncResult(task_id)
                        # Verificar que la tarea existe y no está en estado terminal (SUCCESS, FAILURE, REVOKED)
                        state = result.state
                        if state is None:
                            logger.warning(f"⚠️ Tarea Celery {task_id} no tiene estado (None), considerando inválida")
                            return False
                        # Si la tarea está en estado terminal, no es válida
                        if state in ['SUCCESS', 'FAILURE', 'REVOKED']:
                            logger.info(f"ℹ️ Tarea Celery {task_id} está en estado terminal ({state}), considerando inválida")
                            return False
                        # Si la tarea está en PENDING o RECEIVED, puede estar esperando ejecutarse
                        # Pero si está programada para el futuro, verificar que la fecha sea correcta
                        if state in ['PENDING', 'RECEIVED', 'STARTED']:
                            logger.info(f"ℹ️ Tarea Celery {task_id} está en estado {state}, considerando válida")
                            return True
                        # Para otros estados (como SCHEDULED), también considerarlos válidos
                        logger.info(f"ℹ️ Tarea Celery {task_id} está en estado {state}, considerando válida")
                        return True
                    except Exception as e:
                        logger.warning(f"⚠️ Error verificando tarea Celery {task_id}: {e}")
                        return False
                
                # Verificar si ya tiene bot asignado
                if existing_meeting.recall_bot_id:  # type: ignore
                    logger.info(
                        f"ℹ️ Reunión existente {existing_meeting.id} ya tiene bot asignado (bot_id={existing_meeting.recall_bot_id}), "  # type: ignore
                        f"no se programa nueva tarea"
                    )
                else:
                    # No tiene bot asignado, verificar si hay otro bot para la misma URL
                    # IMPORTANTE: Verificar si ya existe otra reunión con la misma URL que tenga un bot asignado
                    # Esto garantiza que solo se cree UN bot por URL física de Teams *y* que el bot siga activo
                    shared_bot_meeting = db.query(Meeting).filter(
                        Meeting.meeting_url == existing_meeting.meeting_url,
                        Meeting.recall_bot_id.isnot(None),
                        Meeting.id != existing_meeting.id,  # Excluir la reunión actual
                        Meeting.deleted_at.is_(None),  # Solo reuniones no eliminadas
                        # Solo reutilizar bots de reuniones que sigan activas
                        Meeting.status.in_([
                            MeetingStatus.PENDING,
                            MeetingStatus.JOINING,
                            MeetingStatus.IN_PROGRESS,
                        ]),
                    ).first()
                    
                    if shared_bot_meeting:
                        # Reutilizar el bot_id de la otra reunión
                        logger.info(
                            f"🔄 Reutilizando bot existente de otra reunión con la misma URL: "
                            f"reunión {existing_meeting.id} reutilizará bot_id={shared_bot_meeting.recall_bot_id} "
                            f"de reunión {shared_bot_meeting.id} (URL: {existing_meeting.meeting_url})"
                        )
                        existing_meeting.recall_bot_id = shared_bot_meeting.recall_bot_id  # type: ignore
                        existing_meeting.recall_status = shared_bot_meeting.recall_status or "processing"  # type: ignore
                        db.commit()
                        logger.info(
                            f"✅ Reunión {existing_meeting.id} actualizada para usar bot compartido (bot_id={shared_bot_meeting.recall_bot_id})"
                        )
                    else:
                        # No hay bot existente, verificar si necesita programar uno nuevo
                        # Verificar si ya tiene una tarea de Celery válida
                        has_valid_task = verify_celery_task_exists(existing_meeting.celery_task_id)  # type: ignore
                        
                        if not has_valid_task:
                            # No tiene tarea válida, programar una nueva
                            from app.tasks.meeting_tasks import join_bot_to_meeting
                            
                            # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
                            join_time = existing_meeting.scheduled_start_time - timedelta(minutes=1)  # type: ignore
                            # Asegurar que join_time sea timezone-aware (UTC)
                            if join_time.tzinfo is None:  # type: ignore
                                join_time = join_time.replace(tzinfo=timezone.utc)
                            now = datetime.now(timezone.utc)
                            
                            logger.info(
                                f"🕐 Programando bot para reunión existente {existing_meeting.id}: "
                                f"inicio={existing_meeting.scheduled_start_time}, join_time={join_time}, now={now}"
                            )
                            
                            # Solo programar si la fecha es futura
                            if join_time > now:  # type: ignore
                                # Programar tarea usando apply_async con eta (estimated time of arrival)
                                try:
                                    # Convertir join_time a datetime naive si es necesario (Celery espera datetime naive o aware)
                                    eta_datetime = join_time
                                    if eta_datetime.tzinfo is not None:
                                        # Celery puede manejar timezone-aware, pero asegurémonos de que sea UTC
                                        eta_datetime = eta_datetime.astimezone(timezone.utc)
                                    
                                    task = join_bot_to_meeting.apply_async(
                                        args=[existing_meeting.id],
                                        eta=eta_datetime,
                                        expires=eta_datetime + timedelta(hours=1)  # Expirar 1 hora después del join_time
                                    )
                                    # Guardar task_id para poder cancelarlo después
                                    existing_meeting.celery_task_id = task.id  # type: ignore
                                    db.commit()
                                    logger.info(
                                        f"📅 Tarea programada para unir bot a reunión existente {existing_meeting.id} "
                                        f"el {join_time} (UTC) (task_id={task.id}, eta={eta_datetime})"
                                    )
                                    logger.info(
                                        f"⏱️ Tiempo hasta ejecución: {(join_time - now).total_seconds():.0f} segundos "
                                        f"({(join_time - now).total_seconds() / 60:.1f} minutos)"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"❌ Error programando tarea Celery para reunión existente {existing_meeting.id}: {e}",
                                        exc_info=True
                                    )
                                    # Intentar ejecutar inmediatamente como fallback
                                    logger.warning(f"⚠️ Intentando ejecutar inmediatamente como fallback...")
                                    try:
                                        join_bot_to_meeting.delay(existing_meeting.id)
                                        logger.info(f"✅ Tarea enviada para ejecución inmediata (fallback)")
                                    except Exception as e2:
                                        logger.error(f"❌ Error ejecutando tarea inmediatamente: {e2}", exc_info=True)
                            else:
                                # Si ya pasó el tiempo, intentar unirse inmediatamente
                                time_diff = (join_time - now).total_seconds()
                                logger.warning(
                                    f"⚠️ La reunión existente {existing_meeting.id} está programada para el pasado o muy pronto. "
                                    f"Diferencia: {time_diff:.0f} segundos. Intentando unirse inmediatamente..."
                                )
                                try:
                                    join_bot_to_meeting.delay(existing_meeting.id)
                                    logger.info(f"✅ Tarea enviada para ejecución inmediata")
                                except Exception as e:
                                    logger.error(
                                        f"❌ Error ejecutando tarea inmediatamente para reunión existente {existing_meeting.id}: {e}",
                                        exc_info=True
                                    )
                        else:
                            logger.info(
                                f"ℹ️ Reunión existente {existing_meeting.id} ya tiene tarea Celery válida "
                                f"(task_id={existing_meeting.celery_task_id}), no se programa nueva tarea"  # type: ignore
                            )
            elif existing_meeting.status != MeetingStatus.PENDING:  # type: ignore
                logger.info(
                    f"ℹ️ Reunión existente {existing_meeting.id} no está en estado PENDING (status={existing_meeting.status}), "
                    f"no se programa nueva tarea"
                )

            return MeetingResponse(
                id=existing_meeting.id,  # type: ignore
                meeting_url=existing_meeting.meeting_url,  # type: ignore
                title=existing_meeting.title,  # type: ignore
                scheduled_start_time=existing_meeting.scheduled_start_time,  # type: ignore
                scheduled_end_time=existing_meeting.scheduled_end_time,  # type: ignore
                status=existing_meeting.status.value,  # type: ignore
                created_at=existing_meeting.created_at,  # type: ignore
                organizer_email=existing_meeting.organizer_email,  # type: ignore
                organizer_name=get_current_organizer_name(existing_meeting, db),  # type: ignore
                can_delete=can_user_delete_meeting(existing_meeting, db),
                total_meeting_duration_seconds=_meeting_duration_seconds(existing_meeting),
            )

        # No existe una reunión previa: crear una nueva
        # scheduled_start_naive ya está normalizado arriba a UTC naive
        scheduled_end_naive = request.scheduled_end_time
        if scheduled_end_naive and scheduled_end_naive.tzinfo is not None:
            scheduled_end_naive = scheduled_end_naive.astimezone(timezone.utc).replace(tzinfo=None)
        
        extra_meta: Dict[str, Any] = {
            "bot_display_name": request.bot_display_name or settings.default_bot_name,
            "created_manually": True,
            "requires_oauth": False,
        }
        if request.native_meeting_id and request.native_meeting_id.strip():
            extra_meta["vexa_native_meeting_id"] = request.native_meeting_id.strip()
        if request.passcode is not None and request.passcode.strip():
            extra_meta["vexa_passcode"] = request.passcode.strip()
        meeting = Meeting(
            user_id=user_id,
            meeting_url=request.meeting_url,
            thread_id=thread_id,
            title=request.title or "Reunión Teams",
            scheduled_start_time=scheduled_start_naive,
            scheduled_end_time=scheduled_end_naive,
            organizer_email=request.organizer_email,
            organizer_name=request.organizer_name,
            status=MeetingStatus.PENDING,
            extra_metadata=extra_meta,
        )

        db.add(meeting)
        db.commit()
        db.refresh(meeting)

        # Determinar permisos según licencia del usuario
        from app.utils.license_utils import get_user_license_level, get_meeting_access_permissions
        license_level = get_user_license_level(user)
        permissions = get_meeting_access_permissions(license_level)
        
        # Crear MeetingAccess para el usuario que crea la reunión según su licencia
        access = MeetingAccess(
            meeting_id=meeting.id,  # type: ignore
            user_id=user_id,
            can_view_transcript=permissions["can_view_transcript"],
            can_view_audio=permissions["can_view_audio"],
            can_view_video=permissions["can_view_video"],
        )
        db.add(access)
        db.commit()
        
        logger.info(
            "Permisos asignados para usuario %s (licencia: %s) en reunión %s: transcript=%s, audio=%s, video=%s",
            request.user_email,
            license_level,
            meeting.id,
            permissions["can_view_transcript"],
            permissions["can_view_audio"],
            permissions["can_view_video"],
        )

        logger.info(
            "Reunión creada manualmente: %s para %s con acceso completo",
            meeting.id,
            request.user_email,
        )

        # IMPORTANTE: Verificar si ya existe otra reunión con la misma URL que tenga un bot asignado
        # Esto garantiza que solo se cree UN bot por URL física de Teams
        shared_bot_meeting = db.query(Meeting).filter(
            Meeting.meeting_url == meeting.meeting_url,
            Meeting.recall_bot_id.isnot(None),
            Meeting.id != meeting.id,  # Excluir la reunión actual
            Meeting.deleted_at.is_(None),  # Solo reuniones no eliminadas
        ).first()
        
        if shared_bot_meeting:
            # Reutilizar el bot_id de la otra reunión
            logger.info(
                f"🔄 Reutilizando bot existente de otra reunión con la misma URL: "
                f"reunión {meeting.id} reutilizará bot_id={shared_bot_meeting.recall_bot_id} "
                f"de reunión {shared_bot_meeting.id} (URL: {meeting.meeting_url})"
            )
            meeting.recall_bot_id = shared_bot_meeting.recall_bot_id  # type: ignore
            meeting.recall_status = shared_bot_meeting.recall_status or "processing"  # type: ignore
            db.commit()
            logger.info(
                f"✅ Reunión {meeting.id} actualizada para usar bot compartido (bot_id={shared_bot_meeting.recall_bot_id})"
            )
        else:
            # No hay bot existente, programar uno nuevo
            # Programar tarea Celery para unir bot 1 minuto antes del inicio
            from app.tasks.meeting_tasks import join_bot_to_meeting
            
            # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
            join_time = meeting.scheduled_start_time - timedelta(minutes=1)
            # Asegurar que join_time sea timezone-aware (UTC)
            if join_time.tzinfo is None:  # type: ignore
                join_time = join_time.replace(tzinfo=timezone.utc)  # type: ignore
            now = datetime.now(timezone.utc)
            
            logger.info(
                f"🕐 Programando bot para reunión {meeting.id}: "
                f"inicio={meeting.scheduled_start_time} (UTC), join_time={join_time} (UTC), now={now} (UTC)"
            )
            logger.info(
                f"⏱️ Diferencia de tiempo: {(join_time - now).total_seconds():.0f} segundos "  # type: ignore
                f"({(join_time - now).total_seconds() / 60:.1f} minutos)"  # type: ignore
            )
            
            # Solo programar si la fecha es futura
            if join_time > now:  # type: ignore
                # Programar tarea usando apply_async con eta (estimated time of arrival)
                try:
                    task = join_bot_to_meeting.apply_async(
                        args=[meeting.id],
                        eta=join_time
                    )
                    # Guardar task_id para poder cancelarlo después
                    meeting.celery_task_id = task.id  # type: ignore
                    db.commit()
                    logger.info(
                        f"📅 Tarea programada para unir bot a reunión {meeting.id} "
                        f"el {join_time} (UTC) (task_id={task.id})"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Error programando tarea Celery para reunión {meeting.id}: {e}",
                        exc_info=True
                    )
                    # Intentar ejecutar inmediatamente como fallback
                    logger.warning(f"⚠️ Intentando ejecutar inmediatamente como fallback...")
                    try:
                        join_bot_to_meeting.delay(meeting.id)
                        logger.info(f"✅ Tarea enviada para ejecución inmediata (fallback)")
                    except Exception as e2:
                        logger.error(f"❌ Error ejecutando tarea inmediatamente: {e2}", exc_info=True)
            else:
                # Si ya pasó el tiempo, intentar unirse inmediatamente
                time_diff = (join_time - now).total_seconds()
                logger.warning(
                    f"⚠️ La reunión {meeting.id} está programada para el pasado o muy pronto. "
                    f"Diferencia: {time_diff:.0f} segundos. Intentando unirse inmediatamente..."
                )
                try:
                    join_bot_to_meeting.delay(meeting.id)
                    logger.info(f"✅ Tarea enviada para ejecución inmediata")
                except Exception as e:
                    logger.error(
                        f"❌ Error ejecutando tarea inmediatamente para reunión {meeting.id}: {e}",
                        exc_info=True
                    )

        return MeetingResponse(
            id=meeting.id,  # type: ignore
            meeting_url=meeting.meeting_url,  # type: ignore
            title=meeting.title,  # type: ignore
            scheduled_start_time=meeting.scheduled_start_time,  # type: ignore
            scheduled_end_time=meeting.scheduled_end_time,  # type: ignore
            status=meeting.status.value,  # type: ignore
            created_at=meeting.created_at,  # type: ignore
            organizer_email=meeting.organizer_email,  # type: ignore
            organizer_name=get_current_organizer_name(meeting, db),  # type: ignore
            can_delete=can_user_delete_meeting(meeting, db),
            total_meeting_duration_seconds=_meeting_duration_seconds(meeting),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creando reunión manual: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al crear la reunión: {str(e)}"
        )


@router.get("/list", response_model=list[MeetingResponse])
async def list_meetings(
    user_email: Optional[str] = None,
    status: Optional[str] = None,
    upcoming_only: Optional[bool] = None,
    as_admin: Optional[bool] = Query(False, description="Si es True y el usuario es admin, muestra todas las reuniones sin filtrar por acceso"),
    db: Session = Depends(get_db)
):
    """
    Listar reuniones, filtradas por usuario y/o estado.
    
    Si se proporciona user_email, solo se muestran reuniones a las que el usuario tiene acceso
    (vía MeetingAccess o user_id legacy), a menos que as_admin=True y el usuario sea admin.
    
    Si as_admin=True y el usuario es admin, se muestran TODAS las reuniones independientemente del acceso.
    Esto es para la vista de administración.
    
    Args:
        upcoming_only: Si es True, solo muestra reuniones con scheduled_start_time >= ahora
        status: Filtrar por estado específico (pending, completed, etc.)
        as_admin: Si es True y el usuario es admin, muestra todas las reuniones sin filtrar por acceso
    """
    from datetime import datetime
    
    # Determinar si es admin (función auxiliar para evitar problemas de scope)
    def check_is_admin(email: str) -> bool:
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email.endswith("@" + admin_domain.lower()):
            return True
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [e.strip().lower() for e in admin_emails.split(",") if e.strip()]
            if email.lower() in admin_list:
                return True
        return False
    
    # Normalizar as_admin: puede venir como boolean True o como string "true"/"false"
    # Con Query(False) como default, as_admin siempre será bool, pero por si acaso normalizamos
    as_admin_bool = False
    if as_admin is True or (isinstance(as_admin, str) and str(as_admin).lower() in ("true", "1")):
        as_admin_bool = True
    
    if user_email:
        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            return []
        is_admin = check_is_admin(user_email)
        
        # Filtrar reuniones eliminadas (soft delete) - siempre excluir las eliminadas
        # Solo mostrar todas las reuniones si as_admin=True y el usuario es admin (vista de administración)
        if as_admin_bool and is_admin:
            # Vista de administración: mostrar todas las reuniones sin filtrar por acceso
            query = db.query(Meeting).filter(Meeting.deleted_at.is_(None))
        else:
            # Vista personal del usuario (incluso si es admin): filtrar por MeetingAccess
            # También incluir reuniones donde user_id == user.id (legacy)
            access_meeting_ids = [
                ma.meeting_id for ma in db.query(MeetingAccess).filter(
                    MeetingAccess.user_id == user.id
                ).all()
            ]
            
            query = db.query(Meeting).filter(
                (Meeting.id.in_(access_meeting_ids)) | (Meeting.user_id == user.id)
            )
            # Filtrar reuniones eliminadas (soft delete) - solo mostrar si deleted_at es NULL
            query = query.filter(Meeting.deleted_at.is_(None))
    else:
        query = db.query(Meeting)
        # Si no hay user_email, también filtrar eliminadas
        query = query.filter(Meeting.deleted_at.is_(None))
    
    # Calcular now para comparaciones de fecha
    now_utc = datetime.now(timezone.utc)
    now_naive = now_utc.replace(tzinfo=None)
    
    # Para vista de administración (as_admin=True): si no se especifica status ni upcoming_only, mostrar TODAS las reuniones
    if as_admin_bool and is_admin and not status and upcoming_only is None:
        # Vista de administración sin filtros: mostrar todas las reuniones
        pass  # No aplicar ningún filtro adicional
    else:
        # Filtrar por fecha si upcoming_only=True
        if upcoming_only:
            # Para próximas: solo mostrar reuniones con scheduled_start_time >= ahora
            # IMPORTANTE: scheduled_start_time se guarda como naive datetime en UTC
            # Comparamos con now_naive que también es UTC (sin timezone)
            # Si la reunión está guardada en hora local por error, puede aparecer incorrectamente
            query = query.filter(Meeting.scheduled_start_time >= now_naive)
            logger.info(f"🔍 [FILTRO] upcoming_only=True: ahora UTC naive={now_naive}, filtrando reuniones >= {now_naive}")
        elif status == 'completed':
            # Para "pasadas": incluir reuniones completadas O reuniones que ya pasaron (aunque estén en pending/failed)
            query = query.filter(
                (Meeting.status == MeetingStatus.COMPLETED) |
                (
                    (Meeting.scheduled_start_time < now_naive) &
                    (Meeting.status.in_([MeetingStatus.PENDING, MeetingStatus.FAILED, MeetingStatus.JOINING, MeetingStatus.IN_PROGRESS]))
                )
            )
        elif status:
            # Filtrar por estado específico
            try:
                status_enum = MeetingStatus(status)
                query = query.filter(Meeting.status == status_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Estado inválido: {status}")
    
    # Ordenar: próximas por fecha ascendente, pasadas por fecha descendente
    try:
        if upcoming_only:
            meetings = query.order_by(Meeting.scheduled_start_time.asc()).all()
        else:
            meetings = query.order_by(Meeting.scheduled_start_time.desc()).all()
    except LookupError as e:
        # Manejar valores inconsistentes del enum (ej: 'cancelled' vs 'CANCELLED')
        error_msg = str(e)
        if "'cancelled'" in error_msg or "cancelled" in error_msg.lower():
            logger.warning(f"⚠️ Valores inconsistentes de enum detectados: {error_msg}")
            logger.warning("Corrigiendo valores en la base de datos...")
            try:
                from sqlalchemy import text
                # El enum en PostgreSQL acepta 'CANCELLED' (mayúsculas), pero hay registros con 'cancelled' (minúsculas)
                # Corregir: convertir 'cancelled' a 'CANCELLED'
                result = db.execute(text(
                    "UPDATE meetings SET status = 'CANCELLED'::meetingstatus "
                    "WHERE status::text = 'cancelled'"
                ))
                db.commit()
                logger.info(f"✅ {result.rowcount} valores corregidos (de 'cancelled' a 'CANCELLED'). Reintentando consulta...")
                # Reintentar la consulta
                if upcoming_only:
                    meetings = query.order_by(Meeting.scheduled_start_time.asc()).all()
                else:
                    meetings = query.order_by(Meeting.scheduled_start_time.desc()).all()
            except Exception as fix_error:
                logger.error(f"❌ Error al corregir valores inconsistentes: {fix_error}")
                db.rollback()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error al leer reuniones: valores inconsistentes en el enum de estado. "
                           f"Ejecuta el script SQL de corrección (fix_meeting_status_enum.sql). "
                           f"Error original: {error_msg}"
                )
        else:
            raise

    # No mostrar reuniones CANCELLED si existe otra reunion (misma hora, fecha, titulo, organizador) que no este cancelada
    def _meeting_duplicate_key(m: Meeting) -> tuple:
        return (
            (m.organizer_email or "").strip().lower(),
            m.scheduled_start_time,
            (m.title or "").strip().lower(),
        )

    active_keys = {
        _meeting_duplicate_key(m)
        for m in meetings
        if m.status != MeetingStatus.CANCELLED
    }
    meetings_filtered = [
        m for m in meetings
        if m.status != MeetingStatus.CANCELLED or _meeting_duplicate_key(m) not in active_keys
    ]

    def _resincronizar_flag(m: Meeting) -> Optional[bool]:
        if not (as_admin_bool and is_admin):
            return None
        return _necesita_resincronizar_calendario(m)

    return [
        MeetingResponse(
            id=m.id,  # type: ignore
            meeting_url=m.meeting_url,  # type: ignore
            title=m.title,  # type: ignore
            scheduled_start_time=m.scheduled_start_time,  # type: ignore
            scheduled_end_time=m.scheduled_end_time,  # type: ignore
            status=m.status.value,  # type: ignore
            created_at=m.created_at,  # type: ignore
            organizer_email=m.organizer_email,  # type: ignore
            organizer_name=get_current_organizer_name(m, db),  # type: ignore
            can_delete=can_user_delete_meeting(m, db),
            necesita_resincronizar_calendario=_resincronizar_flag(m),
            total_meeting_duration_seconds=_meeting_duration_seconds(m),
        )
        for m in meetings_filtered
    ]


@router.get("/{meeting_id}/join-info", response_model=MeetingJoinInfoResponse)
async def get_meeting_join_info(
    meeting_id: str,
    db: Session = Depends(get_db),
):
    """
    Diagnostico: comprobar si se ha extraido bien el ID de reunion y el codigo de acceso
    (sobre todo para reuniones con formato largo de Teams que vienen del calendario).
    No expone el passcode en claro; indica si esta presente y que ID usara Vexa.
    """
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunion no encontrada")

    url_almacenada = meeting.meeting_url or ""
    if "?p=" in url_almacenada or "&p=" in url_almacenada:
        url_para_mostrar = re.sub(r"([?&]p=)[^&]+", r"\1***", url_almacenada)
    else:
        url_para_mostrar = url_almacenada

    native_id, passcode_presente, listo, origen = _meeting_join_readiness(meeting)
    return MeetingJoinInfoResponse(
        meeting_id=meeting_id,
        url_almacenada=url_para_mostrar,
        native_meeting_id=native_id,
        passcode_presente=passcode_presente,
        listo_para_vexa=listo,
        origen=origen,
    )


@router.get("/{meeting_id}", response_model=MeetingResponse)
async def get_meeting(
    meeting_id: str,
    db: Session = Depends(get_db)
):
    """
    Obtener detalles de una reunión específica.
    
    Devuelve información básica de la reunión incluso sin MeetingAccess.
    Incluye información básica de transcripción (segmentos, duración, participantes) si está disponible.
    """
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunión no encontrada")
    
    # Obtener información básica de transcripción (sin necesidad de MeetingAccess)
    transcription = db.query(Transcription).filter(Transcription.meeting_id == meeting_id).first()
    has_segments_in_db = False
    has_raw_json = False
    if transcription:
        # Verificar si hay segmentos en la tabla dedicada
        seg_count = db.query(TranscriptionSegment).filter(
            TranscriptionSegment.transcription_id == transcription.id
        ).count()
        has_segments_in_db = seg_count > 0
        
        # Verificar si hay JSON crudo (común en reuniones antiguas de Recall)
        raw_json = getattr(transcription, "raw_transcript_json", None)
        has_raw_json = raw_json is not None and (len(raw_json) > 0 if isinstance(raw_json, (dict, list)) else False)

    # NO obtener transcripcion automaticamente cuando el usuario entra
    # La transcripcion se obtiene solo cuando se ejecuta la tarea programada
    # Solo verificamos el estado para informar al frontend
    
    recall_bot_id = getattr(meeting, "recall_bot_id", None)
    status_val = meeting.status.value if hasattr(meeting.status, "value") else str(meeting.status)
    has_any_data = has_segments_in_db or has_raw_json
    is_final_in_db = getattr(transcription, "is_final", False) if transcription else False
    
    logger.info(
        "[VEXA CHECK] Reunion %s: status=%s, recall_bot_id=%s, transcript_scheduled_time=%s, has_segments=%s",
        meeting_id,
        status_val,
        recall_bot_id,
        getattr(meeting, "transcript_scheduled_time", None),
        has_segments_in_db,
    )
    
    # Calcular estado de procesamiento y tiempo restante
    processing_status = None
    estimated_completion_time = None
    time_remaining_seconds = None
    
    now_utc = datetime.now(timezone.utc)
    transcript_scheduled_time = getattr(meeting, "transcript_scheduled_time", None)
    
    # Verificar estado del resumen
    from app.models.summary import Summary
    summary = db.query(Summary).filter(Summary.meeting_id == meeting_id).first()
    summary_status = summary.processing_status if summary else None
    
    if has_any_data and is_final_in_db:
        # Ya tiene transcripcion completa
        if summary_status == "completed":
            processing_status = "completado"
        elif summary_status in ("pending", "processing"):
            processing_status = "generando_resumen"
            estimated_completion_time = None  # Cosmos maneja esto internamente
        else:
            processing_status = "completado"  # Transcripcion lista, resumen no disponible o fallido
    elif transcript_scheduled_time:
        # Hay tarea programada para obtener transcripcion
        if transcript_scheduled_time > now_utc:
            # Aun no ha llegado el momento
            processing_status = "en_cola"
            estimated_completion_time = transcript_scheduled_time
            time_remaining_seconds = int((transcript_scheduled_time - now_utc).total_seconds())
        else:
            # Ya paso el tiempo programado pero aun no hay transcripcion
            processing_status = "obteniendo_transcripcion"
    elif meeting.status == MeetingStatus.IN_PROGRESS:
        # Reunion aun en curso
        processing_status = "en_progreso"
    elif meeting.status == MeetingStatus.COMPLETED and not has_any_data:
        # Completada pero sin transcripcion (puede estar esperando)
        processing_status = "obteniendo_transcripcion"
    else:
        processing_status = "completado" if has_any_data else "pendiente"
    
    # NO obtener transcripcion automaticamente - solo se obtiene cuando se ejecuta la tarea programada

    total_segments = 0
    total_duration_seconds = None
    participants: List[str] = []

    if transcription:
        # Prioridad 1: extraer de raw_transcript_json (datos reales; no se usan TranscriptionSegment)
        raw_json = getattr(transcription, "raw_transcript_json", None)
        if raw_json:
            total_segments, total_duration_seconds, participants = _extract_basic_info_from_raw_transcript(raw_json)

        # Prioridad 2: si no hay JSON o dio 0 segmentos, usar TranscriptionSegment
        if total_segments == 0 and not participants:
            seg_count = db.query(TranscriptionSegment).filter(
                TranscriptionSegment.transcription_id == transcription.id
            ).count()
            if seg_count > 0:
                total_segments = seg_count
                segments = db.query(TranscriptionSegment).filter(
                    TranscriptionSegment.transcription_id == transcription.id
                ).order_by(TranscriptionSegment.end_time).all()
                if segments:
                    last_segment = segments[-1]
                    total_duration_seconds = last_segment.end_time if last_segment.end_time else None
                unique_speakers: set = set()
                for segment in db.query(TranscriptionSegment).filter(
                    TranscriptionSegment.transcription_id == transcription.id
                ).all():
                    speaker_name = getattr(segment, "speaker_name", None)
                    speaker_id = getattr(segment, "speaker_id", None)
                    if speaker_name and str(speaker_name).strip():
                        unique_speakers.add(str(speaker_name).strip())
                    elif speaker_id and str(speaker_id).strip():
                        user = db.query(User).filter(User.email == speaker_id).first()
                        if user and user.display_name:
                            unique_speakers.add(user.display_name)
                        else:
                            unique_speakers.add(str(speaker_id))
                participants = sorted(list(unique_speakers))

    # Crear información básica de transcripción
    transcription_basic_info = TranscriptionBasicInfo(
        total_segments=total_segments,
        total_duration_seconds=total_duration_seconds,
        participants=participants,
        has_transcription=transcription is not None and transcription.is_final if transcription else False
    )
    
    return MeetingResponse(
        id=meeting.id,  # type: ignore
        meeting_url=meeting.meeting_url,  # type: ignore
        title=meeting.title,  # type: ignore
        scheduled_start_time=meeting.scheduled_start_time,  # type: ignore
        scheduled_end_time=meeting.scheduled_end_time,  # type: ignore
        status=meeting.status.value,  # type: ignore
        created_at=meeting.created_at,  # type: ignore
        organizer_email=meeting.organizer_email,  # type: ignore
        organizer_name=get_current_organizer_name(meeting, db),  # type: ignore
        can_delete=can_user_delete_meeting(meeting, db),
        transcription_basic_info=transcription_basic_info,
        total_meeting_duration_seconds=_meeting_duration_seconds(meeting),
        processing_status=processing_status,
        estimated_completion_time=estimated_completion_time.isoformat() if estimated_completion_time else None,
        time_remaining_seconds=time_remaining_seconds,
    )


def _get_transcript_from_db(meeting_id: str, db: Session):
    """
    Lee transcripcion y segmentos de la BD. Devuelve (meeting, transcription, segments)
    o (None, None, []) si no hay reunion o no hay transcripcion.
    """
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        return None, None, []
    transcription = db.query(Transcription).filter(Transcription.meeting_id == meeting_id).first()
    if not transcription:
        return meeting, None, []
    segments = (
        db.query(TranscriptionSegment)
        .filter(TranscriptionSegment.transcription_id == transcription.id)
        .order_by(TranscriptionSegment.start_time)
        .all()
    )
    return meeting, transcription, segments


@router.get("/{meeting_id}/transcription")
async def get_meeting_transcription(
    meeting_id: str,
    db: Session = Depends(get_db),
    user_email: Optional[str] = Query(None, description="Email del usuario para verificar acceso"),
):
    """
    Obtiene la transcripcion para el detalle de reunion (formato esperado por el frontend).

    Lee siempre de la BD local (Transcription + TranscriptionSegment). No llama a VEXA.
    El frontend llama a este endpoint (GET /transcription); si no existe devuelve null y no muestra segmentos.
    
    IMPORTANTE: Si existe raw_transcript_json, se devuelve SIEMPRE al frontend para que lo procese.
    Esto es compatible con el proyecto legacy de Recall.ai donde el frontend maneja el JSON.
    """
    meeting, transcription, segments = _get_transcript_from_db(meeting_id, db)
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunion no encontrada")
    
    if not transcription:
        return {
            "meeting_id": meeting_id,
            "has_transcription": False,
            "is_final": False,
            "total_segments": 0,
            "total_duration_seconds": None,
            "conversation": [],
        }
    
    # Si hay segmentos en la tabla TranscriptionSegment, usarlos (reuniones de Vexa)
    if segments:
        conversation = [
            {
                "speaker": getattr(s, "speaker_name", None) or getattr(s, "speaker_id", "") or "Unknown",
                "text": getattr(s, "text", "") or "",
                "start_time": float(getattr(s, "start_time", 0) or 0),
                "end_time": float(getattr(s, "end_time", 0) or 0),
                "duration": float(getattr(s, "duration", 0) or 0),
            }
            for s in segments
        ]
        return {
            "meeting_id": meeting_id,
            "has_transcription": True,
            "is_final": getattr(transcription, "is_final", False),
            "total_segments": len(conversation),
            "total_duration_seconds": getattr(transcription, "total_duration_seconds", None),
            "conversation": conversation,
            "raw_transcript_json": getattr(transcription, "raw_transcript_json", None),
        }
    
    # Si no hay segmentos pero hay raw_transcript_json, devolverlo directamente
    # El frontend lo procesará (compatible con proyecto legacy de Recall.ai)
    raw_json = getattr(transcription, "raw_transcript_json", None)
    if raw_json:
        logger.info(f"Reunion {meeting_id}: Devolviendo raw_transcript_json al frontend para procesamiento")
        return {
            "meeting_id": meeting_id,
            "has_transcription": True,
            "is_final": getattr(transcription, "is_final", False),
            "total_segments": 0,  # El frontend calculará esto
            "total_duration_seconds": getattr(transcription, "total_duration_seconds", None),
            "conversation": [],  # El frontend extraerá esto del raw_json
            "raw_transcript_json": raw_json,
        }
    
    # No hay ni segmentos ni JSON crudo
    return {
        "meeting_id": meeting_id,
        "has_transcription": False,
        "is_final": False,
        "total_segments": 0,
        "total_duration_seconds": None,
        "conversation": [],
    }


@router.get("/{meeting_id}/transcript")
async def get_meeting_transcript(
    meeting_id: str,
    db: Session = Depends(get_db)
):
    """
    Obtiene la transcripcion completa de una reunion (segmentos con texto, tiempo, speaker).

    Lee siempre de la BD local (Transcription + TranscriptionSegment). No llama a VEXA.
    Los datos se guardan en BD al ejecutar fetch_vexa_transcript_for_meeting (tarea de 5 min
    o endpoint fetch-transcript) o cuando GET /meetings/{id} detecta reunion COMPLETED sin segmentos.
    Si no hay transcripcion en BD, devuelve 404.
    """
    meeting, transcription, segments = _get_transcript_from_db(meeting_id, db)
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunion no encontrada")
    if not transcription:
        raise HTTPException(
            status_code=404,
            detail="Esta reunion no tiene transcripcion. Usa POST /api/meetings/{id}/fetch-transcript para pedirla a VEXA."
        )

    segments_data = [
        {
            "speaker_id": getattr(s, "speaker_id", None),
            "speaker_name": getattr(s, "speaker_name", None),
            "text": getattr(s, "text", "") or "",
            "start_time": getattr(s, "start_time", 0),
            "end_time": getattr(s, "end_time", 0),
            "duration": getattr(s, "duration", 0),
        }
        for s in segments
    ]
    return {
        "meeting_id": meeting_id,
        "total_segments": len(segments_data),
        "total_duration_seconds": getattr(transcription, "total_duration_seconds", None),
        "is_final": getattr(transcription, "is_final", False),
        "segments": segments_data,
    }


# ===== GESTIÓN DE ACCESO MULTI-USUARIO =====

class GrantAccessRequest(BaseModel):
    """Request para otorgar acceso a un usuario."""
    user_email: str = Field(..., description="Email del usuario al que se otorga acceso")
    can_view_transcript: bool = Field(True, description="Permiso para ver transcripción")
    can_view_audio: bool = Field(False, description="Permiso para ver/descargar audio")
    can_view_video: bool = Field(False, description="Permiso para ver/descargar video")


class AccessResponse(BaseModel):
    """Response con información de acceso."""
    id: str
    meeting_id: str
    user_id: str
    user_email: str
    can_view_transcript: bool
    can_view_audio: bool
    can_view_video: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


def check_meeting_access(
    meeting_id: str,
    user_id: str,
    permission_type: str,
    db: Session
) -> bool:
    """
    Verifica si un usuario tiene un permiso específico en una reunión.
    
    IMPORTANTE: Calcula los permisos basándose en la licencia ACTUAL del usuario,
    no en los permisos guardados en MeetingAccess.
    
    Args:
        meeting_id: ID de la reunión
        user_id: ID del usuario
        permission_type: "transcript", "audio", o "video"
        db: Sesión de base de datos
    
    Returns:
        True si el usuario tiene el permiso, False en caso contrario
    """
    # Obtener el usuario
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return False
    
    # Calcular permisos basados en la licencia ACTUAL del usuario
    from app.utils.license_utils import get_user_license_level, get_meeting_access_permissions
    license_level = get_user_license_level(user)
    current_permissions = get_meeting_access_permissions(license_level)
    
    if permission_type == "transcript":
        return current_permissions["can_view_transcript"]
    elif permission_type == "audio":
        return current_permissions["can_view_audio"]
    elif permission_type == "video":
        return current_permissions["can_view_video"]
    
    return False


@router.post("/{meeting_id}/access", response_model=AccessResponse)
async def grant_meeting_access(
    meeting_id: str,
    request: GrantAccessRequest,
    db: Session = Depends(get_db)
):
    """
    Otorgar acceso a un usuario para una reunión con permisos específicos.
    
    Si el usuario ya tiene acceso, se actualizan los permisos.
    """
    # Verificar que la reunión existe
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunión no encontrada")
    
    # Buscar o crear usuario
    user = db.query(User).filter(User.email == request.user_email).first()
    if not user:
        # Crear usuario básico si no existe
        user = User(
            email=request.user_email,
            display_name=request.user_email.split('@')[0],
            microsoft_user_id=f"manual_{request.user_email}",
            tenant_id="manual",
            is_active=True
        )
        db.add(user)
        db.flush()
    
    # Buscar acceso existente
    access = db.query(MeetingAccess).filter(
        MeetingAccess.meeting_id == meeting_id,
        MeetingAccess.user_id == user.id
    ).first()
    
    if access:
        # Actualizar permisos existentes
        access.can_view_transcript = request.can_view_transcript  # type: ignore
        access.can_view_audio = request.can_view_audio  # type: ignore
        access.can_view_video = request.can_view_video  # type: ignore
        logger.info(f"Permisos actualizados para usuario {request.user_email} en meeting {meeting_id}")
    else:
        # Crear nuevo acceso
        access = MeetingAccess(
            meeting_id=meeting_id,
            user_id=user.id,  # type: ignore
            can_view_transcript=request.can_view_transcript,
            can_view_audio=request.can_view_audio,
            can_view_video=request.can_view_video
        )
        db.add(access)
        logger.info(f"Acceso otorgado a usuario {request.user_email} para meeting {meeting_id}")
    
    db.commit()
    db.refresh(access)
    
    return AccessResponse(
        id=access.id,  # type: ignore
        meeting_id=access.meeting_id,  # type: ignore
        user_id=access.user_id,  # type: ignore
        user_email=user.email,  # type: ignore
        can_view_transcript=access.can_view_transcript,  # type: ignore
        can_view_audio=access.can_view_audio,  # type: ignore
        can_view_video=access.can_view_video,  # type: ignore
        created_at=access.created_at  # type: ignore
    )


@router.delete("/{meeting_id}/access/{user_id}")
async def revoke_meeting_access(
    meeting_id: str,
    user_id: str,
    db: Session = Depends(get_db)
):
    """Revocar acceso de un usuario a una reunión."""
    access = db.query(MeetingAccess).filter(
        MeetingAccess.meeting_id == meeting_id,
        MeetingAccess.user_id == user_id
    ).first()
    
    if not access:
        raise HTTPException(status_code=404, detail="Acceso no encontrado")
    
    db.delete(access)
    db.commit()
    
    logger.info(f"Acceso revocado para usuario {user_id} en meeting {meeting_id}")
    return {"success": True, "message": "Acceso revocado correctamente"}


@router.get("/{meeting_id}/access", response_model=list[AccessResponse])
async def list_meeting_access(
    meeting_id: str,
    db: Session = Depends(get_db)
):
    """
    Listar todos los usuarios con acceso a una reunión.
    
    IMPORTANTE: Devuelve todos los usuarios que tienen acceso a CUALQUIERA de las reuniones
    con la misma URL (misma reunión física de Teams), no solo los del meeting_id específico.
    Esto permite ver todos los usuarios que han creado/están asignados a la misma reunión.
    """
    # Verificar que la reunión existe
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunión no encontrada")
    
    # Buscar TODAS las reuniones con la misma URL (misma reunión física)
    # Esto incluye reuniones creadas por diferentes usuarios con la misma URL
    all_meetings_with_same_url = db.query(Meeting).filter(
        Meeting.meeting_url == meeting.meeting_url,
        Meeting.deleted_at.is_(None)  # Solo reuniones no eliminadas
    ).all()
    
    # Obtener todos los IDs de reuniones con la misma URL
    meeting_ids = [m.id for m in all_meetings_with_same_url]
    
    # Buscar todos los accesos a cualquiera de estas reuniones
    access_list = db.query(MeetingAccess).filter(
        MeetingAccess.meeting_id.in_(meeting_ids)
    ).all()
    
    # También incluir los organizadores de las reuniones (si no tienen MeetingAccess explícito)
    # Esto captura casos donde un usuario creó la reunión pero no tiene MeetingAccess registrado
    organizer_emails = set()
    for m in all_meetings_with_same_url:
        if m.organizer_email:  # type: ignore
            organizer_emails.add(m.organizer_email.lower())  # type: ignore
    
    result = []
    processed_user_ids = set()  # Para evitar duplicados
    
    # Procesar MeetingAccess existentes
    # IMPORTANTE: Calcular permisos basados en la licencia ACTUAL del usuario, no los guardados
    from app.utils.license_utils import get_user_license_level, get_meeting_access_permissions
    
    needs_commit = False
    for access in access_list:
        if access.user_id in processed_user_ids:  # type: ignore
            continue  # Evitar duplicados
        
        user = db.query(User).filter(User.id == access.user_id).first()  # type: ignore
        if user:
            processed_user_ids.add(access.user_id)  # type: ignore
            organizer_emails.discard(user.email.lower())  # type: ignore  # Remover de organizer_emails si ya está en access
            
            # Calcular permisos basados en la licencia ACTUAL del usuario
            license_level = get_user_license_level(user)
            current_permissions = get_meeting_access_permissions(license_level)
            
            # Actualizar permisos en la BD si han cambiado (sincronización)
            if ((access.can_view_transcript != current_permissions["can_view_transcript"]) or  # type: ignore
                (access.can_view_audio != current_permissions["can_view_audio"]) or  # type: ignore
                (access.can_view_video != current_permissions["can_view_video"])):  # type: ignore
                logger.info(
                    f"🔄 Sincronizando permisos para usuario {user.email} (licencia: {license_level}) "
                    f"en meeting {access.meeting_id}. "  # type: ignore
                    f"Antes: T={access.can_view_transcript}, A={access.can_view_audio}, V={access.can_view_video}. "  # type: ignore
                    f"Ahora: T={current_permissions['can_view_transcript']}, "
                    f"A={current_permissions['can_view_audio']}, V={current_permissions['can_view_video']}"
                )
                access.can_view_transcript = current_permissions["can_view_transcript"]  # type: ignore
                access.can_view_audio = current_permissions["can_view_audio"]  # type: ignore
                access.can_view_video = current_permissions["can_view_video"]  # type: ignore
                needs_commit = True
            
            result.append(AccessResponse(
                id=access.id,  # type: ignore
                meeting_id=access.meeting_id,  # type: ignore
                user_id=access.user_id,  # type: ignore
                user_email=user.email,  # type: ignore
                can_view_transcript=current_permissions["can_view_transcript"],
                can_view_audio=current_permissions["can_view_audio"],
                can_view_video=current_permissions["can_view_video"],
                created_at=access.created_at  # type: ignore
            ))
    
    # Hacer commit una sola vez si hubo cambios
    if needs_commit:
        db.commit()
    
    # Añadir organizadores que no tienen MeetingAccess explícito
    for email in organizer_emails:
        user = db.query(User).filter(User.email == email).first()
        if (user and user.id not in processed_user_ids):  # type: ignore
            # Calcular permisos basados en la licencia ACTUAL del organizador
            license_level = get_user_license_level(user)
            current_permissions = get_meeting_access_permissions(license_level)
            
            # Crear un AccessResponse "virtual" para el organizador
            # Usar el meeting_id original como referencia
            result.append(AccessResponse(
                id=f"organizer-{user.id}",  # type: ignore
                meeting_id=meeting_id,
                user_id=user.id,  # type: ignore
                user_email=user.email,  # type: ignore
                can_view_transcript=current_permissions["can_view_transcript"],
                can_view_audio=current_permissions["can_view_audio"],
                can_view_video=current_permissions["can_view_video"],
                created_at=meeting.created_at  # type: ignore  # Usar fecha de creación de la reunión
            ))
            processed_user_ids.add(user.id)  # type: ignore
    
    return result


@router.delete("/{meeting_id}")
async def delete_meeting(
    meeting_id: str,
    user_email: Optional[str] = None,
    delete_content: Optional[bool] = Query(False, description="Si es True, borra todo el contenido (archivos, transcripciones). Solo para admins."),
    as_admin: Optional[bool] = Query(False, description="Si es True, el admin borra como administrador (soft/hard delete). Si es False, borra solo su acceso como usuario normal."),
    db: Session = Depends(get_db),
):
    """
    Borrar una reunión o el acceso de un usuario a ella.
    
    Para ADMINS (cuando as_admin=True):
    - Puede borrar cualquier reunión sin restricción de estado
    - Si delete_content=True: hard delete (borra reunión, transcripción, segmentos, archivos físicos)
    - Si delete_content=False: soft delete (marca como deleted, mantiene todo para auditoría)
    - Si la reunión está COMPLETED y delete_content no se especifica, se debe preguntar al frontend
    
    Para ADMINS (cuando as_admin=False o no se envía):
    - Se comporta como usuario normal: solo borra su acceso
    - Si era el último usuario, cancela Celery y borra la reunión
    
    Para USUARIOS NORMALES:
    - Solo borra su acceso. Si era el último usuario, cancela Celery y borra la reunión.
    - No se puede borrar si la reunión está en JOINING o IN_PROGRESS (dejar que acabe)
    """
    # Buscar usuario
    if not user_email:
        raise HTTPException(status_code=400, detail="Se requiere user_email")
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Buscar reunión
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunión no encontrada")
    
    # Verificar si el usuario tiene acceso
    access = db.query(MeetingAccess).filter(
        MeetingAccess.meeting_id == meeting_id,
        MeetingAccess.user_id == user.id
    ).first()
    
    # Determinar si es admin
    admin_domain = getattr(settings, "admin_email_domain", None)
    admin_emails = getattr(settings, "admin_emails", None)
    is_admin = (
        (admin_domain and user.email.endswith(f"@{admin_domain}"))  # type: ignore
        or (admin_emails and user.email in admin_emails)  # type: ignore
    )
    
    # Si es admin Y as_admin=True: puede borrar cualquier reunión como administrador
    # Si es admin pero as_admin=False o no se envía: se comporta como usuario normal
    if is_admin and as_admin:  # type: ignore
        logger.info(f"🔐 Admin {user_email} intentando borrar reunión {meeting_id} (status={meeting.status.value}, delete_content={delete_content})")
        
        # IMPORTANTE: Verificar si hay otras reuniones con la MISMA URL (misma reunión física de Teams)
        # Las reuniones de Teams son únicas por URL, así que debemos verificar todas las reuniones con esa URL
        meeting_url = meeting.meeting_url
        other_meetings_same_url = db.query(Meeting).filter(
            Meeting.meeting_url == meeting_url,
            Meeting.id != meeting_id,
            Meeting.deleted_at.is_(None),  # No incluir reuniones eliminadas
        ).all()
        
        # Contar todos los accesos de todas las reuniones con la misma URL
        access_count_this_meeting = db.query(MeetingAccess).filter(MeetingAccess.meeting_id == meeting_id).count()
        if other_meetings_same_url:
            other_meeting_ids = [m.id for m in other_meetings_same_url]
            total_accesses_same_url = db.query(MeetingAccess).filter(
                MeetingAccess.meeting_id.in_([meeting_id] + other_meeting_ids)
            ).count()
            logger.info(
                f"🔍 [DELETE] Admin borrando reunión {meeting_id}. Hay {len(other_meetings_same_url)} otras reuniones con la misma URL. "
                f"Total accesos en todas las reuniones con esta URL: {total_accesses_same_url}"
            )
        else:
            total_accesses_same_url = access_count_this_meeting
            logger.info(
                f"🔍 [DELETE] Admin borrando reunión {meeting_id}. No hay otras reuniones con la misma URL. "
                f"Accesos en esta reunión: {access_count_this_meeting}"
            )
        
        # IMPORTANTE: Cuando un admin borra una reunión, SIEMPRE debemos cancelar las tareas de Celery
        # de esa reunión específica, independientemente de cuántos accesos haya, porque se está borrando.
        # Si hay otras reuniones con la misma URL, solo cancelamos las tareas de la reunión que se está borrando.
        if meeting.status == MeetingStatus.PENDING:  # type: ignore
            # Cancelar todas las tareas de Celery relacionadas con esta reunión
            cancelled_tasks = _cancel_all_celery_tasks_for_meeting(meeting_id)
            if cancelled_tasks > 0:
                logger.info(f"🚫 [DELETE] {cancelled_tasks} tareas Celery canceladas para reunión {meeting_id}")
            else:
                logger.info(f"ℹ️ [DELETE] No se encontraron tareas Celery para cancelar en reunión {meeting_id}")
            
            # También cancelar la tarea específica si tiene celery_task_id (por compatibilidad)
            if meeting.celery_task_id:  # type: ignore
                try:
                    from app.celery_app import celery_app
                    celery_app.control.revoke(meeting.celery_task_id, terminate=True)  # type: ignore
                    logger.info(f"🚫 [DELETE] Tarea Celery específica cancelada - task_id: {meeting.celery_task_id}, reunión: {meeting_id}")  # type: ignore
                    meeting.celery_task_id = None  # type: ignore
                except Exception as e:
                    logger.warning(f"⚠️ [DELETE] Error cancelando tarea Celery específica {meeting.celery_task_id}: {e}")  # type: ignore
        
        # Si no hay más usuarios en ninguna reunión con esta URL, también cancelar tareas de otras reuniones
        if total_accesses_same_url <= 1:
            meetings_to_cancel_celery = other_meetings_same_url
            for m in meetings_to_cancel_celery:
                if m.status == MeetingStatus.PENDING:  # type: ignore
                    cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(m.id))  # type: ignore
                    if cancelled_tasks > 0:
                        logger.info(f"🚫 [DELETE] {cancelled_tasks} tareas Celery canceladas para reunión relacionada {m.id}")  # type: ignore
                    
                    # También cancelar la tarea específica si tiene celery_task_id
                    if m.celery_task_id:  # type: ignore
                        try:
                            from app.celery_app import celery_app
                            celery_app.control.revoke(m.celery_task_id, terminate=True)  # type: ignore
                            logger.info(f"🚫 [DELETE] Tarea Celery específica cancelada - task_id: {m.celery_task_id}, reunión: {m.id}")  # type: ignore
                            m.celery_task_id = None  # type: ignore
                        except Exception as e:
                            logger.warning(f"⚠️ [DELETE] Error cancelando tarea Celery específica {m.celery_task_id}: {e}")  # type: ignore
        
        # Solo cancelar el bot VEXA si no hay otros usuarios con acceso a ninguna reunion con esa URL
        if total_accesses_same_url <= 1 and meeting.recall_bot_id:
            try:
                from app.services.vexa_service import VexaService, VexaServiceError
                vexa = VexaService()
                vexa.stop_teams_bot(native_meeting_id=meeting.recall_bot_id)
                logger.info(
                    "[DELETE] Bot VEXA cancelado - native_id=%s, reunion=%s",
                    meeting.recall_bot_id,
                    meeting_id,
                )
            except VexaServiceError as e:
                logger.warning("[DELETE] Error cancelando bot VEXA %s: %s", meeting.recall_bot_id, e)
            except Exception as e:
                logger.error("[DELETE] Error cancelando bot VEXA %s: %s", meeting.recall_bot_id, e, exc_info=True)
        elif total_accesses_same_url > 1:
            logger.info(
                "[DELETE] No se cancela el bot VEXA para reunion %s (hay %s usuarios con misma URL)",
                meeting_id,
                total_accesses_same_url,
            )
        
        if delete_content:
            # HARD DELETE: Borrar todo el contenido
            logger.info(f"🗑️ [DELETE] Iniciando HARD DELETE de reunión {meeting_id} por admin {user_email}")
            logger.info(f"📋 [DELETE] Información de la reunión a borrar:")
            logger.info(f"   - ID: {meeting.id}")
            logger.info(f"   - Título: {meeting.title}")
            logger.info(f"   - URL: {meeting.meeting_url}")
            logger.info(f"   - Estado: {meeting.status.value}")
            logger.info(f"   - Audio file path: {meeting.audio_file_path}")
            logger.info(f"   - Video file path: {meeting.video_file_path}")
            logger.info(f"   - Storage type: {meeting.storage_type}")
            logger.info(f"   - Recall bot ID: {meeting.recall_bot_id}")
            
            # Buscar transcripción
            transcription = db.query(Transcription).filter(Transcription.meeting_id == meeting_id).first()
            if transcription:
                logger.info(f"📝 [DELETE] Transcripción encontrada - ID: {transcription.id}")
                logger.info(f"   - Total segmentos: {transcription.total_segments}")
                logger.info(f"   - Duración: {transcription.total_duration_seconds} segundos")
                logger.info(f"   - Idioma: {transcription.language}")
                logger.info(f"   - Tiene raw_transcript_json: {transcription.raw_transcript_json is not None}")
                
                # Contar segmentos antes de borrar
                segment_count = db.query(TranscriptionSegment).filter(
                    TranscriptionSegment.transcription_id == transcription.id
                ).count()
                logger.info(f"   - Segmentos a borrar: {segment_count}")
                
                # Borrar segmentos (cascade debería hacerlo, pero lo hacemos explícito para logging)
                if segment_count > 0:
                    segments = db.query(TranscriptionSegment).filter(
                        TranscriptionSegment.transcription_id == transcription.id
                    ).all()
                    for segment in segments:
                        logger.info(f"   🗑️ [DELETE] Borrando segmento - ID: {segment.id}, Speaker: {segment.speaker_id}, Texto: {segment.text[:50]}...")
                    db.query(TranscriptionSegment).filter(
                        TranscriptionSegment.transcription_id == transcription.id
                    ).delete()
                    logger.info(f"✅ [DELETE] {segment_count} segmentos borrados de la BD")
                
                # Borrar transcripción
                db.delete(transcription)
                logger.info(f"✅ [DELETE] Transcripción {transcription.id} borrada de la BD")
            else:
                logger.info(f"ℹ️ [DELETE] No se encontró transcripción para esta reunión")
            
            # Borrar archivos físicos de audio
            if meeting.audio_file_path:  # type: ignore
                audio_path = meeting.audio_file_path  # type: ignore
                logger.info(f"🎵 [DELETE] Intentando borrar archivo de audio: {audio_path}")
                try:
                    if os.path.exists(audio_path):  # type: ignore
                        file_size = os.path.getsize(audio_path)  # type: ignore
                        os.remove(audio_path)  # type: ignore
                        logger.info(f"✅ [DELETE] Archivo de audio borrado - Ruta: {audio_path}, Tamaño: {file_size} bytes")
                    else:
                        logger.warning(f"⚠️ [DELETE] Archivo de audio no existe en ruta: {audio_path}")
                except Exception as e:
                    logger.error(f"❌ [DELETE] Error borrando archivo de audio {audio_path}: {e}")
            else:
                logger.info(f"ℹ️ [DELETE] No hay archivo de audio asociado a esta reunión")
            
            # Borrar archivos físicos de video
            if meeting.video_file_path:  # type: ignore
                video_path = meeting.video_file_path  # type: ignore
                logger.info(f"🎬 [DELETE] Intentando borrar archivo de video: {video_path}")
                try:
                    if os.path.exists(video_path):  # type: ignore
                        file_size = os.path.getsize(video_path)  # type: ignore
                        os.remove(video_path)  # type: ignore
                        logger.info(f"✅ [DELETE] Archivo de video borrado - Ruta: {video_path}, Tamaño: {file_size} bytes")
                    else:
                        logger.warning(f"⚠️ [DELETE] Archivo de video no existe en ruta: {video_path}")
                except Exception as e:
                    logger.error(f"❌ [DELETE] Error borrando archivo de video {video_path}: {e}")
            else:
                logger.info(f"ℹ️ [DELETE] No hay archivo de video asociado a esta reunión")
            
            # Borrar summaries antes de borrar la reunión (para evitar error de NOT NULL constraint)
            from app.models.summary import Summary
            summary = db.query(Summary).filter(Summary.meeting_id == meeting_id).first()
            if summary:
                logger.info(f"📄 [DELETE] Summary encontrado - ID: {summary.id}")
                logger.info(f"   - Status: {summary.processing_status}")
                logger.info(f"   - Tiene toon: {summary.toon is not None}")
                logger.info(f"   - Tiene ia_response_json: {summary.ia_response_json is not None}")
                db.delete(summary)
                logger.info(f"✅ [DELETE] Summary {summary.id} borrado de la BD")
            else:
                logger.info(f"ℹ️ [DELETE] No se encontró summary para esta reunión")
            
            # Contar accesos antes de borrar
            access_count = db.query(MeetingAccess).filter(MeetingAccess.meeting_id == meeting_id).count()
            logger.info(f"👥 [DELETE] Accesos a borrar: {access_count}")
            if access_count > 0:
                accesses = db.query(MeetingAccess).filter(MeetingAccess.meeting_id == meeting_id).all()
                for access in accesses:
                    logger.info(f"   🗑️ [DELETE] Borrando acceso - User ID: {access.user_id}, Meeting ID: {access.meeting_id}")
                db.query(MeetingAccess).filter(MeetingAccess.meeting_id == meeting_id).delete()
                logger.info(f"✅ [DELETE] {access_count} accesos borrados de la BD")
            
            # Borrar la reunión
            db.delete(meeting)
            db.commit()
            logger.info(f"✅ [DELETE] Reunión {meeting_id} borrada completamente de la BD (HARD DELETE)")
            logger.info(f"🎯 [DELETE] HARD DELETE completado para reunión {meeting_id} por admin {user_email}")
            
            return {"success": True, "message": "Reunión y todo su contenido borrados completamente"}
        else:
            # SOFT DELETE: Solo marcar como eliminada
            logger.info(f"🗑️ [DELETE] Iniciando SOFT DELETE de reunión {meeting_id} por admin {user_email}")
            logger.info(f"📋 [DELETE] Información de la reunión a ocultar:")
            logger.info(f"   - ID: {meeting.id}")
            logger.info(f"   - Título: {meeting.title}")
            logger.info(f"   - URL: {meeting.meeting_url}")
            logger.info(f"   - Estado: {meeting.status.value}")
            logger.info(f"   - Audio file path: {meeting.audio_file_path}")
            logger.info(f"   - Video file path: {meeting.video_file_path}")
            
            # Si la reunión está PENDING, cancelar tareas de Celery porque no se va a celebrar
            if meeting.status == MeetingStatus.PENDING:  # type: ignore
                cancelled_tasks = _cancel_all_celery_tasks_for_meeting(meeting_id)
                if cancelled_tasks > 0:
                    logger.info(f"🚫 [DELETE] {cancelled_tasks} tareas Celery canceladas para reunión {meeting_id} (SOFT DELETE)")
                else:
                    logger.info(f"ℹ️ [DELETE] No se encontraron tareas Celery para cancelar en reunión {meeting_id}")
                
                # También cancelar la tarea específica si tiene celery_task_id
                if meeting.celery_task_id:  # type: ignore
                    try:
                        from app.celery_app import celery_app
                        celery_app.control.revoke(meeting.celery_task_id, terminate=True)  # type: ignore
                        logger.info(f"🚫 [DELETE] Tarea Celery específica cancelada - task_id: {meeting.celery_task_id}, reunión: {meeting_id}")  # type: ignore
                        meeting.celery_task_id = None  # type: ignore
                    except Exception as e:
                        logger.warning(f"⚠️ [DELETE] Error cancelando tarea Celery específica {meeting.celery_task_id}: {e}")  # type: ignore
            
            # Verificar si hay transcripción
            transcription = db.query(Transcription).filter(Transcription.meeting_id == meeting_id).first()
            if transcription:
                segment_count = db.query(TranscriptionSegment).filter(
                    TranscriptionSegment.transcription_id == transcription.id
                ).count()
                logger.info(f"   - Transcripción ID: {transcription.id}, Segmentos: {segment_count}")
                logger.info(f"   ℹ️ [DELETE] Transcripción y segmentos se mantienen para auditoría")
            else:
                logger.info(f"   ℹ️ [DELETE] No hay transcripción asociada")
            
            # Contar accesos
            access_count = db.query(MeetingAccess).filter(MeetingAccess.meeting_id == meeting_id).count()
            logger.info(f"   - Accesos: {access_count}")
            logger.info(f"   ℹ️ [DELETE] Accesos se mantienen para auditoría")
            
            # Marcar como eliminada (soft delete)
            meeting.deleted_at = datetime.utcnow()  # type: ignore
            db.commit()
            logger.info(f"✅ [DELETE] Reunión {meeting_id} marcada como eliminada (deleted_at={meeting.deleted_at})")
            logger.info(f"🎯 [DELETE] SOFT DELETE completado para reunión {meeting_id} por admin {user_email}")
            logger.info(f"   ℹ️ [DELETE] Todos los archivos y datos se mantienen para auditoría")
            
            return {"success": True, "message": "Reunión ocultada. Todos los archivos y datos se mantienen para auditoría."}
    
    # Si es usuario normal: solo borrar su acceso
    # Verificar acceso tanto por MeetingAccess como por sistema legacy (user_id)
    has_access_via_meeting_access = access is not None
    has_access_via_legacy = (meeting.user_id == user.id) if meeting.user_id else False  # type: ignore
    
    logger.info(
        f"🔍 [DELETE] Verificando acceso para usuario {user_email} (user_id={user.id}) en reunión {meeting_id}: "
        f"MeetingAccess={has_access_via_meeting_access}, Legacy(user_id={meeting.user_id})={has_access_via_legacy}"  # type: ignore
    )
    
    if (not has_access_via_meeting_access and not has_access_via_legacy):  # type: ignore
        # Si no tiene acceso directo, verificar si aparece en su lista (puede tener acceso vía MeetingAccess de otra reunión con misma URL)
        # O puede ser que la reunión se importó desde calendario y no se creó MeetingAccess
        # En ese caso, crear el acceso automáticamente si la reunión aparece en su lista
        logger.warning(
            f"⚠️ [DELETE] Usuario {user_email} no tiene acceso directo a reunión {meeting_id}. "
            f"Verificando si aparece en su lista de reuniones..."
        )
        
        # Verificar si el usuario es el organizador de la reunión
        is_organizer = False
        if meeting.organizer_email and user.email:  # type: ignore
            is_organizer = meeting.organizer_email.lower() == user.email.lower()  # type: ignore
        
        # Verificar si la reunión aparece en la lista del usuario usando la MISMA lógica que list_meetings
        # Esto garantiza que si aparece en la lista, puede borrarla
        access_meeting_ids = [
            ma.meeting_id for ma in db.query(MeetingAccess).filter(
                MeetingAccess.user_id == user.id  # type: ignore
            ).all()
        ]
        
        # Verificar directamente si esta reunión específica está en la lista del usuario
        # usando EXACTAMENTE la misma query que list_meetings (líneas 558-562)
        # Primero construir la query igual que en list_meetings
        list_query = db.query(Meeting).filter(
            (Meeting.id.in_(access_meeting_ids)) | (Meeting.user_id == user.id)  # type: ignore
        )
        list_query = list_query.filter(Meeting.deleted_at.is_(None))
        
        # Verificar si esta reunión específica está en esa query
        query_check = list_query.filter(Meeting.id == meeting_id).first()
        
        appears_in_list = query_check is not None
        
        logger.info(
            f"🔍 [DELETE] Verificación adicional: is_organizer={is_organizer}, "
            f"en_access_list={meeting_id in access_meeting_ids}, "
            f"appears_in_list_query={appears_in_list}, "
            f"meeting.user_id={meeting.user_id}, user.id={user.id}"  # type: ignore
        )
        
        if appears_in_list or is_organizer:
            # La reunión aparece en su lista o es el organizador, crear el MeetingAccess
            reason = "es el organizador" if is_organizer else "aparece en lista del usuario"
            logger.info(f"📝 [DELETE] Reunión {meeting_id} {reason}, creando MeetingAccess...")
            from app.utils.license_utils import get_meeting_access_permissions, get_user_license_level
            license_level = get_user_license_level(user)
            permissions = get_meeting_access_permissions(license_level)
            access = MeetingAccess(
                meeting_id=meeting_id,
                user_id=user.id,  # type: ignore
                can_view_transcript=permissions["can_view_transcript"],
                can_view_audio=permissions["can_view_audio"],
                can_view_video=permissions["can_view_video"],
            )
            db.add(access)
            db.commit()
            db.refresh(access)
            logger.info(f"✅ [DELETE] MeetingAccess creado para usuario {user_email} en reunión {meeting_id}")
        else:
            # Realmente no tiene acceso
            logger.error(
                f"❌ [DELETE] Usuario {user_email} NO tiene acceso a reunión {meeting_id}. "
                f"MeetingAccess={has_access_via_meeting_access}, Legacy={has_access_via_legacy}, "
                f"AppearsInList={appears_in_list}, IsOrganizer={is_organizer}, "
                f"OrganizerEmail={meeting.organizer_email}, UserEmail={user.email}"  # type: ignore
            )
            raise HTTPException(status_code=403, detail="No tienes acceso a esta reunión")
    
    # Si no hay MeetingAccess pero tiene acceso legacy, crear el MeetingAccess para poder borrarlo
    if (not access and has_access_via_legacy):  # type: ignore
        # Crear MeetingAccess para poder borrarlo correctamente
        from app.utils.license_utils import get_meeting_access_permissions, get_user_license_level
        license_level = get_user_license_level(user)
        permissions = get_meeting_access_permissions(license_level)
        access = MeetingAccess(
            meeting_id=meeting_id,
            user_id=user.id,  # type: ignore
            can_view_transcript=permissions["can_view_transcript"],
            can_view_audio=permissions["can_view_audio"],
            can_view_video=permissions["can_view_video"],
        )
        db.add(access)
        db.commit()
        db.refresh(access)
        logger.info(f"📝 [DELETE] Creado MeetingAccess para reunión legacy {meeting_id} antes de borrar")
    
    # No permitir borrar si está en JOINING o IN_PROGRESS
    if meeting.status in [MeetingStatus.JOINING, MeetingStatus.IN_PROGRESS]:  # type: ignore
        raise HTTPException(
            status_code=400,
            detail=f"No se puede borrar una reunión que está en estado {meeting.status.value}. "
                   f"Espera a que termine."
        )
    
    # Guardar la URL antes de borrar (por si necesitamos usarla después)
    meeting_url = meeting.meeting_url
    
    # Borrar el acceso del usuario
    db.delete(access)
    db.commit()
    
    # Verificar si quedan más usuarios con acceso a ESTA reunión específica
    remaining_accesses_this_meeting = db.query(MeetingAccess).filter(
        MeetingAccess.meeting_id == meeting_id
    ).count()
    
    logger.info(
        f"🔍 [DELETE] Usuario {user_email} borró su acceso. "
        f"Accesos restantes en esta reunión específica ({meeting_id}): {remaining_accesses_this_meeting}"
    )
    
    # LÓGICA SIMPLIFICADA: Solo considerar esta reunión específica, no otras con la misma URL
    if remaining_accesses_this_meeting == 0:
        # Era el ÚNICO usuario en esta reunión
        logger.info(
            f"🗑️ [DELETE] No quedan usuarios en esta reunión específica ({meeting_id}). "
            f"Verificando si tiene contenido para auditoría..."
        )
        
        # VERIFICAR SI LA REUNIÓN TIENE CONTENIDO (transcripción, audio, video)
        # Si tiene contenido, NO borrar la reunión (mantener para auditoría)
        has_transcription = db.query(Transcription).filter(Transcription.meeting_id == meeting_id).first() is not None
        has_audio = meeting.audio_file_path is not None and meeting.audio_file_path != ""  # type: ignore
        has_video = meeting.video_file_path is not None and meeting.video_file_path != ""  # type: ignore
        has_content = has_transcription or has_audio or has_video
        
        # También verificar si la reunión está COMPLETED (ya se celebró)
        is_completed = meeting.status == MeetingStatus.COMPLETED  # type: ignore
        
        if has_content or is_completed:  # type: ignore
            # La reunión tiene contenido o está completada - NO borrar, mantener para auditoría
            logger.info(
                f"🔒 [DELETE] Reunión {meeting_id} tiene contenido o está completada - "
                f"mantener para auditoría (transcripción={has_transcription}, audio={has_audio}, "
                f"video={has_video}, completed={is_completed}). "
                f"Solo se borró el acceso del usuario."
            )
            
            return {
                "success": True,
                "message": "Acceso borrado. La reunión se mantiene en el sistema para auditoría porque tiene contenido grabado."
            }
        
        # La reunión NO tiene contenido y está en PENDING - se puede borrar
        logger.info(
            f"🗑️ [DELETE] Reunión {meeting_id} sin contenido y en estado PENDING. "
            f"Cancelando Celery y borrando la reunión."
        )
        
        # Cancelar todas las tareas de Celery de esta reunión específica
        if meeting.status == MeetingStatus.PENDING:  # type: ignore
            cancelled_tasks = _cancel_all_celery_tasks_for_meeting(meeting_id)
            if cancelled_tasks > 0:
                logger.info(f"🚫 [DELETE] {cancelled_tasks} tareas Celery canceladas para reunión {meeting_id} (último usuario)")
            else:
                logger.info(f"ℹ️ [DELETE] No se encontraron tareas Celery para cancelar en reunión {meeting_id}")
            
            # También cancelar la tarea específica si tiene celery_task_id
            if meeting.celery_task_id:  # type: ignore
                try:
                    from app.celery_app import celery_app
                    celery_app.control.revoke(meeting.celery_task_id, terminate=True)  # type: ignore
                    logger.info(f"🚫 [DELETE] Tarea Celery específica cancelada - task_id: {meeting.celery_task_id}, reunión: {meeting_id}")  # type: ignore
                    meeting.celery_task_id = None  # type: ignore
                except Exception as e:
                    logger.warning(f"⚠️ [DELETE] Error cancelando tarea Celery específica {meeting.celery_task_id}: {e}")  # type: ignore
        
        # Cancelar bot VEXA si existe
        if meeting.recall_bot_id:
            try:
                from app.services.vexa_service import VexaService, VexaServiceError
                vexa = VexaService()
                vexa.stop_teams_bot(native_meeting_id=meeting.recall_bot_id)
                logger.info(
                    "[DELETE] Bot VEXA cancelado (native_id=%s) - ultimo usuario en reunion %s",
                    meeting.recall_bot_id,
                    meeting_id,
                )
            except VexaServiceError as e:
                logger.warning("[DELETE] Error cancelando bot VEXA %s: %s", meeting.recall_bot_id, e)
            except Exception as e:
                logger.error("[DELETE] Error cancelando bot VEXA %s: %s", meeting.recall_bot_id, e, exc_info=True)
        
        # Borrar esta reunión (no se mostrará en pendientes)
        db.delete(meeting)
        db.commit()
        
        logger.info(
            f"✅ [DELETE] Reunión {meeting_id} borrada completamente. "
            f"Era el último usuario, sin contenido, Celery cancelado."
        )
        
        return {
            "success": True,
            "message": "Acceso borrado. Era el último usuario en esta reunión sin contenido, la reunión ha sido eliminada y las tareas de Celery canceladas."
        }
    else:
        # Quedan más usuarios en esta reunión: solo se borró el acceso (no se muestra en pendientes para este usuario)
        logger.info(
            f"ℹ️ [DELETE] Usuario {user_email} borró su acceso a reunión {meeting_id}. "
            f"Quedan {remaining_accesses_this_meeting} usuarios en esta reunión. "
            f"NO se cancela Celery (hay otros usuarios)."
        )
        
        return {
            "success": True,
            "message": f"Acceso borrado. Quedan {remaining_accesses_this_meeting} usuarios con acceso a esta reunión. La reunión no se mostrará en tus pendientes."
        }


@router.post("/{meeting_id}/recover")
async def recover_meeting(
    meeting_id: str,
    user_email: Optional[str] = None,
    restore_status: Optional[bool] = Query(True, description="Si es True y la reunión tiene transcripción, restaura el status a COMPLETED"),
    as_admin: Optional[bool] = Query(False, description="Solo admins pueden recuperar reuniones"),
    db: Session = Depends(get_db),
):
    """
    Recuperar una reunión que fue eliminada con soft delete.
    
    Requiere permisos de admin.
    
    - Quita el soft delete (deleted_at = NULL)
    - Opcionalmente restaura el status a COMPLETED si tiene transcripción y está en CANCELLED
    """
    # Buscar usuario
    if not user_email:
        raise HTTPException(status_code=400, detail="Se requiere user_email")
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Verificar si es admin
    admin_domain = getattr(settings, "admin_email_domain", None)
    admin_emails = getattr(settings, "admin_emails", None)
    is_admin = (
        (admin_domain and user.email.endswith(f"@{admin_domain}"))  # type: ignore
        or (admin_emails and user.email in admin_emails)  # type: ignore
    )
    
    if not is_admin or not as_admin:
        raise HTTPException(status_code=403, detail="Solo administradores pueden recuperar reuniones")
    
    # Buscar reunión (incluyendo las eliminadas)
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunión no encontrada")
    
    # Verificar si tiene soft delete
    if meeting.deleted_at is None:  # type: ignore
        logger.info(f"ℹ️ [RECOVER] Reunión {meeting_id} no tiene soft delete, solo verificando status...")
    else:
        # Quitar soft delete
        meeting.deleted_at = None  # type: ignore
        logger.info(f"✅ [RECOVER] Soft delete removido de reunión {meeting_id}")
    
    # Si restore_status es True y la reunión está CANCELLED, verificar si tiene transcripción
    if restore_status and meeting.status == MeetingStatus.CANCELLED:  # type: ignore
        transcription = db.query(Transcription).filter(
            Transcription.meeting_id == meeting_id,
            Transcription.is_final == True
        ).first()
        
        if transcription:
            meeting.status = MeetingStatus.COMPLETED  # type: ignore
            logger.info(f"✅ [RECOVER] Status restaurado a COMPLETED (reunión tiene transcripción final)")
        else:
            logger.info(f"ℹ️ [RECOVER] Reunión está CANCELLED pero no tiene transcripción final, manteniendo status")
    
    db.commit()
    
    return {
        "success": True,
        "message": "Reunión recuperada exitosamente",
        "meeting_id": meeting_id,
        "deleted_at": None,
        "status": meeting.status.value if meeting.status else None  # type: ignore
    }


@router.post("/{meeting_id}/fix-access")
async def fix_meeting_access(
    meeting_id: str,
    user_email: Optional[str] = None,
    grant_all_permissions: Optional[bool] = Query(True, description="Si es True, otorga todos los permisos (transcript, audio, video, delete)"),
    as_admin: Optional[bool] = Query(False, description="Solo admins pueden usar este endpoint"),
    db: Session = Depends(get_db),
):
    """
    Verificar y corregir el MeetingAccess de un usuario para una reunión.
    
    Si el MeetingAccess no existe, lo crea con los permisos especificados.
    Si existe pero los permisos están en False, los actualiza.
    
    Requiere permisos de admin.
    """
    # Buscar usuario
    if not user_email:
        raise HTTPException(status_code=400, detail="Se requiere user_email")
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Verificar si es admin
    admin_domain = getattr(settings, "admin_email_domain", None)
    admin_emails = getattr(settings, "admin_emails", None)
    is_admin = (
        (admin_domain and user.email.endswith(f"@{admin_domain}"))  # type: ignore
        or (admin_emails and user.email in admin_emails)  # type: ignore
    )
    
    if not is_admin or not as_admin:
        raise HTTPException(status_code=403, detail="Solo administradores pueden usar este endpoint")
    
    # Buscar reunión
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunión no encontrada")
    
    # Buscar MeetingAccess existente
    meeting_access = db.query(MeetingAccess).filter(
        MeetingAccess.meeting_id == meeting_id,
        MeetingAccess.user_id == user.id
    ).first()
    
    # Verificar qué usuario necesita acceso (por defecto el mismo que hace la petición)
    target_user_email = user_email
    target_user = user
    
    # Verificar si hay transcripción, audio y video disponibles
    transcription = db.query(Transcription).filter(
        Transcription.meeting_id == meeting_id,
        Transcription.is_final == True
    ).first()
    
    has_audio = meeting.audio_file_path is not None  # type: ignore
    has_video = meeting.video_file_path is not None  # type: ignore
    has_transcription = transcription is not None
    
    if meeting_access:
        # Actualizar permisos existentes
        if grant_all_permissions:
            meeting_access.can_view_transcript = True  # type: ignore
            meeting_access.can_view_audio = True  # type: ignore
            meeting_access.can_view_video = True  # type: ignore
            logger.info(f"✅ [FIX-ACCESS] Permisos actualizados para usuario {target_user_email} en reunión {meeting_id}")
        else:
            # Solo habilitar permisos para recursos que existen
            if has_transcription:
                meeting_access.can_view_transcript = True  # type: ignore
            if has_audio:
                meeting_access.can_view_audio = True  # type: ignore
            if has_video:
                meeting_access.can_view_video = True  # type: ignore
            logger.info(f"✅ [FIX-ACCESS] Permisos actualizados (selectivos) para usuario {target_user_email} en reunión {meeting_id}")
    else:
        # Crear nuevo MeetingAccess
        from app.utils.license_utils import get_user_license_level, get_meeting_access_permissions
        
        license_level = get_user_license_level(target_user)
        permissions = get_meeting_access_permissions(license_level)
        
        if grant_all_permissions:
            permissions = {
                "can_view_transcript": True,
                "can_view_audio": True,
                "can_view_video": True
            }
        else:
            # Solo habilitar permisos para recursos que existen
            permissions = {
                "can_view_transcript": has_transcription,
                "can_view_audio": has_audio,
                "can_view_video": has_video
            }
        
        meeting_access = MeetingAccess(
            meeting_id=meeting_id,
            user_id=target_user.id,  # type: ignore
            can_view_transcript=permissions["can_view_transcript"],
            can_view_audio=permissions["can_view_audio"],
            can_view_video=permissions["can_view_video"]
        )
        db.add(meeting_access)
        logger.info(f"✅ [FIX-ACCESS] MeetingAccess creado para usuario {target_user_email} en reunión {meeting_id}")
    
    db.commit()
    db.refresh(meeting_access)
    
    return {
        "success": True,
        "message": "Acceso corregido exitosamente",
        "meeting_id": meeting_id,
        "user_email": target_user_email,
        "meeting_access": {
            "can_view_transcript": meeting_access.can_view_transcript,  # type: ignore
            "can_view_audio": meeting_access.can_view_audio,  # type: ignore
            "can_view_video": meeting_access.can_view_video  # type: ignore
        },
        "resources_available": {
            "has_transcription": has_transcription,
            "has_audio": has_audio,
            "has_video": has_video
        }
    }


class SyncCeleryResponse(BaseModel):
    """Response de sincronización con Celery."""
    synced_count: int
    skipped_with_bot: int
    skipped_past: int
    skipped_valid_task: int
    error_count: int
    total_processed: int


@router.post("/sync-celery", response_model=SyncCeleryResponse)
async def sync_meetings_with_celery(
    user_email: Optional[str] = Query(None, description="Email del admin que ejecuta la sincronización"),
    db: Session = Depends(get_db),
):
    """
    Sincronizar todas las reuniones PENDING con Celery.
    
    Solo para admins. Re-programa las tareas de Celery para reuniones que:
    - Están en estado PENDING
    - No tienen bot asignado
    - No tienen celery_task_id válido
    - Tienen fecha futura
    
    Returns:
        Estadísticas de la sincronización
    """
    # Verificar que se proporcione user_email
    if not user_email:
        raise HTTPException(status_code=400, detail="Se requiere user_email")
    
    # Buscar usuario
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Verificar que sea admin
    admin_domain = getattr(settings, "admin_email_domain", None)
    admin_emails = getattr(settings, "admin_emails", None)
    is_admin = (
        (admin_domain and user.email.endswith(f"@{admin_domain}"))  # type: ignore
        or (admin_emails and user.email in admin_emails)  # type: ignore
    )
    
    if not is_admin:  # type: ignore
        raise HTTPException(status_code=403, detail="Solo los administradores pueden sincronizar reuniones con Celery")
    
    logger.info(f"Admin {user_email} iniciando sincronizacion de reuniones con Celery")
    
    now = datetime.now(timezone.utc)
    now_naive = now.replace(tzinfo=None)
    
    # Buscar todas las reuniones PENDING que no están eliminadas
    pending_meetings = db.query(Meeting).filter(
        Meeting.status == MeetingStatus.PENDING,
        Meeting.deleted_at.is_(None),
    ).all()
    
    logger.info(f"Encontradas {len(pending_meetings)} reuniones en estado PENDING")
    
    synced_count: int = 0
    skipped_with_bot: int = 0
    skipped_past: int = 0
    skipped_valid_task: int = 0
    error_count: int = 0
    
    # Función auxiliar para verificar si una tarea de Celery existe
    def verify_celery_task_exists(task_id: str) -> bool:
        if not task_id:
            return False
        try:
            from app.celery_app import celery_app
            result = celery_app.AsyncResult(task_id)
            return result.state is not None
        except Exception:
            return False
    
    for meeting in pending_meetings:
        try:
            # Verificar si ya tiene bot asignado
            if meeting.recall_bot_id:  # type: ignore
                logger.debug(
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
                meeting.recall_bot_id = shared_bot_meeting.recall_bot_id  # type: ignore
                meeting.recall_status = shared_bot_meeting.recall_status or "processing"  # type: ignore
                db.commit()
                skipped_with_bot += 1  # type: ignore
                continue
            
            # Verificar que la fecha sea futura
            scheduled_time = meeting.scheduled_start_time  # type: ignore
            if scheduled_time.tzinfo is None:  # type: ignore
                scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)  # type: ignore
            
            if scheduled_time <= now:  # type: ignore
                logger.debug(
                    f"Reunion {meeting.id} ya paso su hora de inicio "
                    f"({scheduled_time}), saltando"
                )
                skipped_past += 1  # type: ignore
                continue
            
            # Verificar si tiene celery_task_id válido
            if meeting.celery_task_id:  # type: ignore
                if verify_celery_task_exists(meeting.celery_task_id):  # type: ignore
                    logger.debug(
                        f"Reunion {meeting.id} ya tiene tarea Celery valida "
                        f"(task_id={meeting.celery_task_id}), saltando"
                    )
                    skipped_valid_task += 1  # type: ignore
                    continue
                else:
                    logger.info(
                        f"Reunion {meeting.id} tiene celery_task_id invalido "
                        f"({meeting.celery_task_id}), re-programando"
                    )
            
            # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
            from app.tasks.meeting_tasks import join_bot_to_meeting
            
            join_time = scheduled_time - timedelta(minutes=1)  # type: ignore
            if join_time.tzinfo is None:  # type: ignore
                join_time = join_time.replace(tzinfo=timezone.utc)  # type: ignore
            
            logger.info(
                f"Programando bot para reunion {meeting.id}: "
                f"inicio={scheduled_time} (UTC), join_time={join_time} (UTC), "
                f"now={now} (UTC)"
            )
            
            # Solo programar si la fecha es futura
            if join_time > now:  # type: ignore
                # Programar tarea usando apply_async con eta
                try:
                    task = join_bot_to_meeting.apply_async(
                        args=[meeting.id],
                        eta=join_time
                    )
                    # Guardar task_id para poder cancelarlo después
                    meeting.celery_task_id = task.id  # type: ignore
                    db.commit()
                    logger.info(
                        f"Tarea programada para unir bot a reunion {meeting.id} "
                        f"el {join_time} (UTC) (task_id={task.id})"
                    )
                    synced_count += 1  # type: ignore
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
                    error_count += 1  # type: ignore
                    
        except Exception as e:
            logger.error(
                f"Error procesando reunion {meeting.id}: {e}",
                exc_info=True
            )
            error_count += 1
    
    logger.info(
        f"Sincronizacion completada: {synced_count} sincronizadas, "
        f"{skipped_with_bot} saltadas (tienen bot), "
        f"{skipped_past} saltadas (ya pasaron), "
        f"{skipped_valid_task} saltadas (tarea valida), "
        f"{error_count} errores"
    )
    
    return SyncCeleryResponse(
        synced_count=synced_count,
        skipped_with_bot=skipped_with_bot,
        skipped_past=skipped_past,
        skipped_valid_task=skipped_valid_task,
        error_count=error_count,
        total_processed=len(pending_meetings)
    )


class SendSummaryEmailRequest(BaseModel):
    """Request para enviar resumen de reunión por email."""
    recipients: List[str] = Field(..., description="Lista de emails destinatarios")
    subject: str = Field(..., description="Asunto del email")
    cc: Optional[List[str]] = Field(None, description="Lista de emails en copia (opcional)")
    additional_recipients: Optional[List[str]] = Field(None, description="Emails adicionales que no son participantes")


def get_meeting_participants(meeting_id: str, db: Session) -> List[str]:
    """
    Obtiene la lista de participantes de una reunion.
    Prioriza los participantes del JSON de Cosmos (insights.turns), luego transcripcion, y complementa con MeetingAccess.
    
    Args:
        meeting_id: ID de la reunion
        db: Sesion de base de datos
        
    Returns:
        Lista de nombres de participantes (strings), ordenados alfabeticamente
    """
    from app.models.summary import Summary
    
    participants = set()
    
    # 1. PRIMERO: Intentar obtener participantes del JSON de Cosmos (fuente mas confiable)
    # Cosmos ya identifica correctamente todos los nombres en insights.turns
    summary = db.query(Summary).filter(Summary.meeting_id == meeting_id).first()
    
    if summary and summary.ia_response_json:  # type: ignore
        ia_data = summary.ia_response_json  # type: ignore
        insights = ia_data.get("insights", {})
        
        # insights.turns contiene {nombre_participante: numero_turnos}
        turns = insights.get("turns", {})
        if turns and isinstance(turns, dict):
            for participant_name in turns.keys():
                if participant_name and participant_name.strip():
                    participants.add(participant_name.strip())
    
    # 2. SEGUNDO: Si no hay participantes de Cosmos, intentar con transcripcion
    if not participants:
        transcription = db.query(Transcription).filter(
            Transcription.meeting_id == meeting_id
        ).first()
        
        if transcription:
            # Obtener TODOS los speakers unicos de la transcripcion
            all_segments = db.query(TranscriptionSegment).filter(
                TranscriptionSegment.transcription_id == transcription.id  # type: ignore
            ).all()
            
            # Usar un set para almacenar speakers unicos (nombre, id) como tuplas
            unique_speakers = set()
            
            for segment in all_segments:
                speaker_name = segment.speaker_name  # type: ignore
                speaker_id = segment.speaker_id  # type: ignore
                
                # Crear una tupla unica para cada speaker
                if speaker_name and speaker_name.strip():
                    unique_speakers.add(("name", speaker_name.strip()))
                elif speaker_id and speaker_id.strip():
                    unique_speakers.add(("id", speaker_id.strip()))
            
            # Procesar cada speaker unico
            for speaker_type, speaker_value in unique_speakers:
                if speaker_type == "name":
                    clean_name = speaker_value
                    # Intentar mapear a display_name de usuario (busqueda mas flexible)
                    user = db.query(User).filter(
                        User.display_name == clean_name  # type: ignore
                    ).first()
                    
                    if not user:
                        user = db.query(User).filter(
                            User.display_name.ilike(f"%{clean_name}%")  # type: ignore
                        ).first()
                    
                    if user and user.display_name:  # type: ignore
                        participants.add(user.display_name)  # type: ignore
                    else:
                        participants.add(clean_name)
                elif speaker_type == "id":
                    user = db.query(User).filter(User.email == speaker_value).first()  # type: ignore
                    if user and user.display_name:  # type: ignore
                        participants.add(user.display_name)  # type: ignore
    
    # Convertir a lista, eliminar vacios, y ordenar alfabeticamente
    participants_list = [p for p in participants if p and p.strip()]
    return sorted(participants_list)


@router.post("/{meeting_id}/send-summary-email")
async def send_summary_email(
    meeting_id: str,
    request: SendSummaryEmailRequest,
    db: Session = Depends(get_db)
):
    """
    Envía un email con el resumen de la reunión a los destinatarios especificados.
    
    Requiere que la reunión tenga datos (transcripción o summary).
    """
    from app.models.summary import Summary
    from app.services.email_service import send_meeting_summary_email
    from pathlib import Path
    import base64
    
    # Verificar que la reunión existe
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunión no encontrada")
    
    # Verificar que la reunión tiene datos
    transcription = db.query(Transcription).filter(
        Transcription.meeting_id == meeting_id
    ).first()
    
    summary = db.query(Summary).filter(
        Summary.meeting_id == meeting_id
    ).first()
    
    if not transcription and not summary:
        raise HTTPException(
            status_code=400,
            detail="Esta reunión no tiene datos disponibles (transcripción o resumen)"
        )
    
    # Obtener participantes
    participants = get_meeting_participants(meeting_id, db)
    
    # Obtener resumen de IA
    summary_text = None
    insights = None
    
    if summary and summary.processing_status == "completed" and summary.ia_response_json:
        ia_data = summary.ia_response_json
        insights = ia_data.get("insights", {})
        
        # Función auxiliar para detectar si un texto es un prompt (no un resumen real)
        def is_prompt(text: str) -> bool:
            """Detecta si el texto es un prompt en lugar de un resumen generado."""
            if not text:
                return True
            text_lower = text.lower()
            # Detectar frases típicas de prompts
            prompt_indicators = [
                "we need to generate",
                "must not include",
                "use content:",
                "provide concise",
                "generate summary",
                "must be in",
                "do not include"
            ]
            return any(indicator in text_lower for indicator in prompt_indicators)
        
        # 1. Intentar obtener el resumen de chunks[0].insights.summary (más confiable)
        chunks = ia_data.get("chunks", [])
        if chunks and len(chunks) > 0:
            chunk_insights = chunks[0].get("insights", {})
            chunk_summary = chunk_insights.get("summary")
            if chunk_summary and not is_prompt(chunk_summary):
                summary_text = chunk_summary
        
        # 2. Si no hay resumen en chunks, intentar insights.summary pero validar que no sea prompt
        if not summary_text:
            root_summary = insights.get("summary")
            if root_summary and not is_prompt(root_summary):
                summary_text = root_summary
        
        # 3. Fallback a toon o summary_text
        if not summary_text:
            summary_text = summary.toon or summary.summary_text  # type: ignore
    
    # Si no hay resumen de IA, intentar usar el toon o summary_text
    if not summary_text and summary:
        summary_text = summary.toon or summary.summary_text  # type: ignore
    
    # Calcular duración si es posible
    meeting_duration = None
    if meeting.scheduled_start_time and meeting.scheduled_end_time:  # type: ignore
        duration = meeting.scheduled_end_time - meeting.scheduled_start_time  # type: ignore
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        if hours > 0:
            meeting_duration = f"{hours}h {minutes}min"
        else:
            meeting_duration = f"{minutes}min"
    
    # Obtener nombre del organizador
    organizer_name = get_current_organizer_name(meeting, db) or meeting.organizer_name  # type: ignore
    
    # Cargar logos en base64 (incrustados en el email)
    logo_base64 = None
    cosmos_logo_base64 = None
    
    # Función auxiliar para cargar un logo
    def load_logo(filename: str) -> Optional[str]:
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "frontend" / "public" / filename,  # Desde backend/app/api/routes/
            Path(__file__).parent.parent.parent.parent.parent / "frontend" / "public" / filename,  # Alternativa
            Path.cwd() / "frontend" / "public" / filename,  # Desde directorio de trabajo
        ]
        
        for logo_path in possible_paths:
            if logo_path.exists():
                try:
                    with open(logo_path, 'rb') as f:
                        logo_data = f.read()
                        logo_b64 = base64.b64encode(logo_data).decode('utf-8')
                        logger.info(f"Logo {filename} cargado exitosamente desde: {logo_path}")
                        return logo_b64
                except Exception as e:
                    logger.warning(f"No se pudo cargar el logo {filename} desde {logo_path}: {e}")
        return None
    
    # Cargar ambos logos
    cosmos_logo_base64 = load_logo("cosmos-logo.png")
    logo_base64 = load_logo("notetaker-light.png")
    
    if not logo_base64 and not cosmos_logo_base64:
        logger.warning("No se pudieron encontrar los logos. El email se enviará sin logos incrustados.")
    
    # Enviar email
    try:
        success = send_meeting_summary_email(
            recipients=request.recipients,
            subject=request.subject,
            meeting_title=meeting.title,  # type: ignore
            meeting_date=meeting.scheduled_start_time,  # type: ignore
            meeting_duration=meeting_duration,
            participants=participants,
            summary_text=summary_text,
            insights=insights,
            organizer_name=organizer_name,
            meeting_id=meeting_id,
            cc=request.cc,
            logo_base64=logo_base64,
            cosmos_logo_base64=cosmos_logo_base64,
            frontend_url=settings.frontend_url
        )
        
        if success:
            return {
                "success": True,
                "message": f"Email enviado exitosamente a {len(request.recipients)} destinatario(s)"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Error al enviar el email. Verifica la configuración del servicio de email."
            )
    except Exception as e:
        logger.error(f"Error al enviar email de resumen: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al enviar el email: {str(e)}"
        )


@router.get("/{meeting_id}/summary")
async def get_meeting_summary(
    meeting_id: str,
    db: Session = Depends(get_db),
    user_email: Optional[str] = Query(None, description="Email del usuario para verificar acceso"),
):
    """
    Obtiene el estado y datos del análisis/resumen de IA para una reunión.

    El frontend usa este endpoint para mostrar el panel de análisis (Resumen, Insights, Análisis).
    Devuelve status: not_available | pending | processing | completed | failed,
    y cuando está completed incluye toon, insights (de Cosmos) y metadatos.
    Si no hay registro Summary, devuelve status 'not_available' (no 404) para que
    el front muestre mensaje coherente y pueda hacer polling cuando se programe uno.
    """
    from app.models.summary import Summary

    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunion no encontrada")

    summary = db.query(Summary).filter(Summary.meeting_id == meeting_id).first()
    if not summary:
        return {
            "status": "not_available",
            "message": "Esta reunion no tiene analisis disponible aun",
        }

    # Construir respuesta segun processing_status
    ia_data = (summary.ia_response_json or {}) if hasattr(summary, "ia_response_json") else {}
    insights = ia_data.get("insights") or {}

    payload = {
        "status": summary.processing_status or "pending",
        "processing_time_seconds": getattr(summary, "processing_time_seconds", None),
        "completed_at": summary.completed_at.isoformat() if summary.completed_at else None,
        "toon": getattr(summary, "toon", None) or ia_data.get("toon"),
        "insights": insights,
        "queue_estimation": None,
    }
    if summary.processing_status == "failed" and getattr(summary, "error_message", None):
        payload["message"] = summary.error_message
        payload["error"] = summary.error_message
    return payload


@router.get("/{meeting_id}/cosmos-json")
async def get_cosmos_json(
    meeting_id: str,
    db: Session = Depends(get_db)
):
    """
    Obtiene el JSON completo de la respuesta de Cosmos para una reunión.

    Este endpoint devuelve el JSON completo tal como lo recibió el sistema de Cosmos,
    almacenado en ia_response_json. Útil para debugging y verificar qué datos
    está devolviendo realmente Cosmos.

    Returns:
        JSON completo de la respuesta de Cosmos, o error si no está disponible
    """
    from app.models.summary import Summary

    # Verificar que la reunión existe
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunión no encontrada")
    
    # Obtener el summary
    summary = db.query(Summary).filter(Summary.meeting_id == meeting_id).first()
    if not summary:
        raise HTTPException(
            status_code=404,
            detail="Esta reunión no tiene resumen disponible"
        )
    
    # Verificar que tiene el JSON de Cosmos
    if not summary.ia_response_json:
        raise HTTPException(
            status_code=404,
            detail="Esta reunión no tiene JSON de Cosmos disponible. El procesamiento puede estar pendiente o haber fallado."
        )
    
    # Devolver el JSON completo
    return {
        "meeting_id": meeting_id,
        "processing_status": summary.processing_status,
        "cosmos_json": summary.ia_response_json,
        "toon": summary.toon,  # También incluir el toon extraído para referencia
        "completed_at": summary.completed_at.isoformat() if summary.completed_at else None,
    }


@router.post("/{meeting_id}/regenerate-summary")
async def regenerate_meeting_summary(
    meeting_id: str,
    user_email: Optional[str] = Query(None, description="Email del usuario que solicita la regeneración"),
    db: Session = Depends(get_db)
):
    """
    Fuerza la regeneración del resumen de IA para una reunión específica.
    
    Este endpoint permite volver a procesar el resumen de una reunión que ya tiene uno,
    útil cuando el resumen generado no es correcto o se quiere actualizar.
    
    Requiere que la reunión tenga transcripción disponible.
    SOLO ADMINISTRADORES pueden usar este endpoint.
    """
    from app.models.summary import Summary
    from app.tasks.summary_tasks import process_meeting_summary
    import uuid
    
    # Verificar permisos
    if not user_email:
        raise HTTPException(status_code=401, detail="Se requiere autenticación")
    
    # Función auxiliar para verificaciones
    def check_is_admin(email: str) -> bool:
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email.endswith("@" + admin_domain.lower()):
            return True
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [e.strip().lower() for e in admin_emails.split(",") if e.strip()]
            if email.lower() in admin_list:
                return True
        return False
    
    is_admin = check_is_admin(user_email)
    
    # Si NO es admin, verificar condiciones para permitir regeneración
    if not is_admin:
        # 1. Verificar existencia de la reunión y acceso del usuario
        meeting_access_check = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if not meeting_access_check:
             raise HTTPException(status_code=404, detail="Reunión no encontrada")
        
        # Verificar si usuario tiene acceso
        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            raise HTTPException(status_code=403, detail="Usuario no encontrado")

        has_access = False
        if meeting_access_check.user_id == user.id:
            has_access = True
        else:
            access = db.query(MeetingAccess).filter(
                MeetingAccess.meeting_id == meeting_id, 
                MeetingAccess.user_id == user.id
            ).first()
            if access:
                has_access = True
        
        if not has_access:
             raise HTTPException(status_code=403, detail="No tienes acceso a esta reunión")

        # 2. Verificar estado del resumen: Solo permitir si está fallido o no existe
        existing_summary = db.query(Summary).filter(Summary.meeting_id == meeting_id).first()
        
        can_regenerate = False
        if not existing_summary:
            can_regenerate = True
        elif existing_summary.processing_status == "failed":
            # Si falló, verificar si tiene datos parciales que valga la pena proteger
            has_data = bool(existing_summary.toon or (existing_summary.ia_response_json and existing_summary.ia_response_json.get("insights")))
            if not has_data:
                can_regenerate = True
        
        if not can_regenerate:
            raise HTTPException(
                status_code=403, 
                detail="Solo los administradores pueden regenerar un resumen que ya existe o se está procesando"
            )
    
    # Verificar que la reunión existe
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunión no encontrada")
    
    # Verificar que la reunión tiene transcripción
    transcription = db.query(Transcription).filter(
        Transcription.meeting_id == meeting_id,
        Transcription.raw_transcript_json.isnot(None),
        Transcription.is_final == True
    ).first()
    
    if not transcription:
        raise HTTPException(
            status_code=400,
            detail="Esta reunión no tiene transcripción disponible para generar el resumen"
        )
    
    # Buscar Summary existente o crear uno nuevo
    summary = db.query(Summary).filter(Summary.meeting_id == meeting_id).first()
    
    if not summary:
        # Crear nuevo Summary
        summary = Summary(
            id=str(uuid.uuid4()),
            meeting_id=meeting_id,
            processing_status="pending"
        )
        db.add(summary)
        logger.info(f"📝 Summary creado (pending) para meeting {meeting_id}")
    else:
        # Forzar reprocesamiento: cambiar estado a pending y limpiar errores
        summary.processing_status = "pending"
        summary.error_message = None
        logger.info(f"🔄 Summary forzado a reprocesar para meeting {meeting_id}")
    
    db.commit()
    
    # Encolar tarea de procesamiento
    try:
        process_meeting_summary.apply_async(args=[meeting_id], queue="summary_queue")
        logger.info(f"✅ Tarea de resumen encolada para meeting {meeting_id}")
        
        return {
            "success": True,
            "message": "El resumen se está regenerando. El proceso puede tardar unos minutos.",
            "meeting_id": meeting_id,
            "status": "pending"
        }
    except Exception as e:
        logger.error(f"❌ Error encolando tarea de resumen para meeting {meeting_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al iniciar el procesamiento del resumen: {str(e)}"
        )


@router.post("/join-bot-now")
async def join_bot_now(
    meeting_url: str = Query(..., description="URL de la reunión de Teams"),
    user_email: Optional[str] = Query(None, description="Email del usuario"),
    db: Session = Depends(get_db)
):
    """
    Unir el bot a una reunión inmediatamente usando la URL.
    
    Busca la reunión por URL, y si no existe la crea. Luego ejecuta la tarea
    de unir el bot inmediatamente.
    """
    try:
        # Buscar reunión existente por URL
        existing_meeting = db.query(Meeting).filter(
            Meeting.meeting_url == meeting_url,
            Meeting.deleted_at.is_(None),
        ).order_by(Meeting.created_at.desc()).first()
        
        if existing_meeting:
            meeting_id = existing_meeting.id  # type: ignore
            logger.info(f"🔍 Reunión existente encontrada: {meeting_id} para URL {meeting_url}")
        else:
            # Crear nueva reunión
            if not user_email:
                raise HTTPException(
                    status_code=400,
                    detail="Se requiere user_email para crear una nueva reunión"
                )
            
            # Buscar o crear usuario
            user = db.query(User).filter(User.email == user_email).first()
            if not user:
                user = User(
                    email=user_email,
                    display_name=user_email.split('@')[0],
                    microsoft_user_id=f"manual_{user_email}",
                    tenant_id="manual",
                    is_active=True
                )
                db.add(user)
                db.commit()
                db.refresh(user)
            
            user_id = user.id  # type: ignore
            thread_id = extract_thread_id_from_url(meeting_url)
            
            # Crear reunión con fecha/hora actual (para que se ejecute inmediatamente)
            now = datetime.now(timezone.utc)
            scheduled_start_naive = now.replace(tzinfo=None)
            
            meeting = Meeting(
                user_id=user_id,
                meeting_url=meeting_url,
                thread_id=thread_id,
                title="Reunión Teams",
                scheduled_start_time=scheduled_start_naive,
                status=MeetingStatus.PENDING,
                extra_metadata={
                    "created_manually": True,
                    "join_immediately": True,
                },
            )
            
            db.add(meeting)
            db.commit()
            db.refresh(meeting)
            meeting_id = meeting.id  # type: ignore
            
            # Crear MeetingAccess
            from app.utils.license_utils import get_user_license_level, get_meeting_access_permissions
            license_level = get_user_license_level(user)
            permissions = get_meeting_access_permissions(license_level)
            
            access = MeetingAccess(
                meeting_id=meeting_id,
                user_id=user_id,
                can_view_transcript=permissions["can_view_transcript"],
                can_view_audio=permissions["can_view_audio"],
                can_view_video=permissions["can_view_video"],
            )
            db.add(access)
            db.commit()
            
            logger.info(f"✅ Nueva reunión creada: {meeting_id} para URL {meeting_url}")
        
        # Ejecutar la tarea inmediatamente usando .delay() para que se ejecute en Celery
        # pero sin programación (inmediatamente)
        from app.tasks.meeting_tasks import join_bot_to_meeting
        
        logger.info(f"🚀 Ejecutando tarea de unión de bot para reunión {meeting_id} (inmediatamente)")
        
        # Ejecutar la tarea inmediatamente usando .delay() (sin eta)
        task = join_bot_to_meeting.delay(meeting_id)
        
        # Esperar un poco para que la tarea se ejecute (opcional, para obtener resultado)
        # En producción, podrías devolver el task_id y verificar el resultado después
        import time
        time.sleep(2)  # Esperar 2 segundos para que la tarea comience
        
        # Obtener el resultado de la tarea
        try:
            result = task.get(timeout=30)  # Esperar hasta 30 segundos
        except Exception as e:
            logger.warning(f"⚠️ No se pudo obtener el resultado inmediatamente: {e}")
            try:
                from app.services.monitoring_service import alert_timeout
                alert_timeout("/api/meetings/join-bot", 30.0)
            except Exception:
                pass
            result = {"success": "pending", "task_id": task.id, "message": "Tarea en ejecución"}
        
        logger.info(f"✅ Resultado de la tarea: {result}")
        
        return {
            "success": True,
            "message": "Bot unido a la reunión",
            "meeting_id": meeting_id,
            "meeting_url": meeting_url,
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error uniendo bot a reunión: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al unir el bot: {str(e)}"
        )


@router.post("/{meeting_id}/test-join-bot")
async def test_join_bot_to_meeting(
    meeting_id: str,
    user_email: Optional[str] = Query(None, description="Email del usuario que ejecuta la prueba"),
    db: Session = Depends(get_db)
):
    """
    Endpoint de prueba para ejecutar manualmente la tarea de unir el bot a una reunión.
    
    Útil para diagnosticar problemas con la programación de tareas de Celery.
    Solo para administradores.
    """
    # Verificar que el usuario es admin
    if not user_email:
        raise HTTPException(status_code=401, detail="Se requiere autenticación")
    
    # Función auxiliar para verificar si es admin
    def check_is_admin(email: str) -> bool:
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email.endswith("@" + admin_domain.lower()):
            return True
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [e.strip().lower() for e in admin_emails.split(",") if e.strip()]
            if email.lower() in admin_list:
                return True
        return False
    
    if not check_is_admin(user_email):
        raise HTTPException(
            status_code=403,
            detail="Solo los administradores pueden ejecutar esta prueba"
        )
    
    # Verificar que la reunión existe
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunión no encontrada")
    
    logger.info(f"🧪 [TEST] Admin {user_email} ejecutando prueba de unión de bot para reunión {meeting_id}")
    
    # Ejecutar la tarea inmediatamente
    try:
        from app.tasks.meeting_tasks import join_bot_to_meeting
        
        # Ejecutar la tarea inmediatamente usando .delay() (sin eta)
        task = join_bot_to_meeting.delay(meeting_id)
        
        # Esperar un poco para obtener el resultado
        import time
        time.sleep(2)
        
        try:
            result = task.get(timeout=30)
        except Exception as e:
            logger.warning(f"⚠️ No se pudo obtener el resultado inmediatamente: {e}")
            try:
                from app.services.monitoring_service import alert_timeout
                alert_timeout("/api/meetings/test-join-bot", 30.0)
            except Exception:
                pass
            result = {"success": "pending", "task_id": task.id, "message": "Tarea en ejecución"}
        
        logger.info(f"✅ [TEST] Resultado de la tarea: {result}")
        
        return {
            "success": True,
            "message": "Tarea ejecutada manualmente",
            "meeting_id": meeting_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"❌ [TEST] Error ejecutando tarea manualmente: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error al ejecutar la tarea: {str(e)}"
        )


@router.post("/{meeting_id}/fetch-transcript")
async def fetch_meeting_transcript_now(
    meeting_id: str,
    user_email: Optional[str] = Query(None, description="Email del usuario (admin para autorizar)"),
    db: Session = Depends(get_db)
):
    """
    Pide a VEXA la transcripcion de la reunion y la guarda en BD ahora.
    No espera a la hora programada (scheduled_end_time + 5 min).
    Si hay segmentos, la reunion pasa a COMPLETED.
    """
    def _is_admin(email: str) -> bool:
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email and email.endswith("@" + admin_domain.lower()):
            return True
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [e.strip().lower() for e in admin_emails.split(",") if e.strip()]
            if email and email.lower() in admin_list:
                return True
        return False

    if not user_email or not _is_admin(user_email):
        raise HTTPException(
            status_code=403,
            detail="Se requiere user_email de administrador (query: user_email=tu@email.com)"
        )

    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunion no encontrada")
    if not meeting.recall_bot_id:
        raise HTTPException(
            status_code=400,
            detail="Esta reunion no tiene bot VEXA (recall_bot_id). No se puede pedir transcripcion."
        )

    from app.tasks.meeting_tasks import fetch_vexa_transcript_for_meeting
    task = fetch_vexa_transcript_for_meeting.delay(meeting_id)
    logger.info("Tarea fetch_vexa_transcript_for_meeting encolada para reunion %s (task_id=%s)", meeting_id, task.id)
    return {
        "success": True,
        "message": "Tarea de transcripcion encolada. En unos segundos recarga la reunion; si VEXA tiene segmentos, pasara a Completada.",
        "meeting_id": meeting_id,
        "task_id": task.id,
    }


@router.post("/{meeting_id}/force-full-transcript")
def force_full_transcript_download(
    meeting_id: str,
    user_email: Optional[str] = Query(None, description="Email del administrador para autorizacion"),
    db: Session = Depends(get_db),
):
    """
    Fuerza la descarga completa de la transcripcion usando timeout extendido (180s).
    Este endpoint permite que VEXA tenga mas tiempo para procesar y devolver
    mas segmentos en reuniones largas, incluso si la reunion ya tiene transcripcion guardada.

    Requiere autenticacion de administrador via user_email query param.
    """
    def _is_admin(email: Optional[str]) -> bool:
        if not email:
            return False
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email and email.endswith("@" + admin_domain.lower()):
            return True
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [e.strip().lower() for e in admin_emails.split(",") if e.strip()]
            if email and email.lower() in admin_list:
                return True
        return False

    if not user_email or not _is_admin(user_email):
        raise HTTPException(
            status_code=403,
            detail="Se requiere user_email de administrador (query: user_email=tu@email.com)"
        )

    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunion no encontrada")
    if not meeting.recall_bot_id:
        raise HTTPException(
            status_code=400,
            detail="Esta reunion no tiene bot VEXA (recall_bot_id). No se puede pedir transcripcion."
        )

    from app.tasks.meeting_tasks import fetch_vexa_transcript_for_meeting
    # force_full=True para usar timeout extendido (180s en lugar de 30s)
    task = fetch_vexa_transcript_for_meeting.delay(meeting_id, force_full=True)
    logger.info(
        "[API] Tarea fetch_vexa_transcript_for_meeting (TIMEOUT EXTENDIDO) encolada para reunion %s (task_id=%s)",
        meeting_id,
        task.id
    )
    return {
        "success": True,
        "message": "Tarea de transcripcion con timeout extendido (180s) encolada. Esto permite que VEXA tenga mas tiempo para procesar y devolver mas segmentos.",
        "meeting_id": meeting_id,
        "task_id": task.id,
        "force_full": True,
    }



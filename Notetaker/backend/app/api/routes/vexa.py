"""Rutas para integracion con VEXA (bot en reuniones Teams y transcripciones)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.meeting import Meeting, MeetingStatus
from app.services.vexa_service import VexaService, VexaServiceError, parse_teams_meeting_url
from app.tasks.meeting_tasks import save_vexa_transcript_to_db, schedule_summary_after_transcription, fetch_vexa_transcript_for_meeting
from app.config import settings
from app.utils.vexa_failure_messages import vexa_failure_reason_to_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vexa", tags=["vexa"])

# Router interno para webhooks (sin prefijo /api/vexa)
internal_router = APIRouter(prefix="/api/internal/vexa", tags=["internal"])


class JoinBotRequest(BaseModel):
    """Peticion para crear un bot VEXA en una reunion existente."""
    meeting_id: str = Field(..., description="ID de la reunion en la base de datos")


class JoinBotResponse(BaseModel):
    success: bool
    bot_id: str
    status: str
    meeting_id: str


@router.post("/bot/join-meeting", response_model=JoinBotResponse)
def join_meeting_with_vexa(
    request: JoinBotRequest,
    db: Session = Depends(get_db),
) -> JoinBotResponse:
    """
    Crea un bot VEXA que se une a la reunion de Teams.
    Usa meeting_url para extraer native_meeting_id y passcode.
    """
    meeting = db.query(Meeting).filter(Meeting.id == request.meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunion no encontrada")

    meeting_url = meeting.meeting_url or ""
    native_meeting_id, passcode = parse_teams_meeting_url(meeting_url)
    if not native_meeting_id:
        # Diferenciar entre URL que no es de Teams y URL de Teams con formato no soportado
        if "teams.microsoft.com" not in meeting_url and "teams.live.com" not in meeting_url:
            detail = "La URL de la reunion no es de Microsoft Teams o no tiene un formato valido."
        else:
            detail = "No se pudo extraer un identificador de reunion valido desde meeting_url de Teams."
        raise HTTPException(
            status_code=400,
            detail=detail,
        )

    try:
        vexa = VexaService()
        vexa.start_teams_bot(native_meeting_id=native_meeting_id, passcode=passcode)
    except VexaServiceError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    meeting.recall_bot_id = native_meeting_id
    meeting.recall_status = "active"
    meeting.status = MeetingStatus.IN_PROGRESS
    if not meeting.extra_metadata:
        meeting.extra_metadata = {}
    meeting.extra_metadata["vexa_native_meeting_id"] = native_meeting_id
    if passcode:
        meeting.extra_metadata["vexa_passcode"] = passcode
    db.commit()

    logger.info("Bot VEXA asociado a reunion %s (native_id=%s)", meeting.id, native_meeting_id)
    return JoinBotResponse(
        success=True,
        bot_id=native_meeting_id,
        status="active",
        meeting_id=meeting.id,
    )


@router.get("/meetings/{meeting_id}/transcript")
def get_meeting_transcript_from_vexa(
    meeting_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Obtiene la transcripcion de la reunion desde VEXA.
    Usa meeting.recall_bot_id como native_meeting_id.
    """
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Reunion no encontrada")
    native_id = meeting.recall_bot_id
    if not native_id:
        raise HTTPException(
            status_code=400,
            detail="Reunion sin bot VEXA (recall_bot_id). Unir bot primero.",
        )
    try:
        vexa = VexaService()
        data = vexa.get_transcript(native_meeting_id=native_id, platform="teams")
    except VexaServiceError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return data


# ============================================================================
# Webhook interno: Bot-Manager -> Notetaker cuando el bot sale
# ============================================================================

class BotExitedWebhookPayload(BaseModel):
    """Payload del webhook cuando el bot sale de la reunion."""
    recall_bot_id: str = Field(..., description="Native meeting ID (recall_bot_id en Notetaker)")
    exit_code: int = Field(..., description="Codigo de salida del bot (0=exito, !=0=error)")
    status: str = Field(..., description="Estado final: 'completed' o 'failed'")
    reason: Optional[str] = Field(None, description="Razon de la salida")
    platform: str = Field(default="teams", description="Plataforma de la reunion")


def verify_webhook_auth(authorization: Optional[str] = Header(None)) -> bool:
    """
    Verifica autenticacion del webhook usando API key compartida.
    Espera header: Authorization: Bearer <API_KEY>
    """
    if not authorization:
        return False
    
    # Extraer el token del header "Bearer <token>"
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    
    token = parts[1].strip()
    
    # Comparar con la API key configurada (misma que usa VEXA_API_KEY)
    # Esta debe coincidir con ADMIN_TOKEN o NOTETAKER_WEBHOOK_API_KEY en bot-manager
    expected_key = getattr(settings, "vexa_api_key", None)
    
    if not expected_key:
        logger.warning("No hay API key configurada para webhooks internos")
        return False
    
    return token == expected_key


@internal_router.post("/bot-exited", include_in_schema=False)
def handle_bot_exited_webhook(
    payload: BotExitedWebhookPayload,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Webhook interno llamado por bot-manager cuando el bot sale de la reunion.
    
    Busca la reunion por recall_bot_id, actualiza el estado y obtiene la transcripcion
    si esta disponible.
    
    Autenticacion: Header Authorization: Bearer <API_KEY>
    """
    # Verificar autenticacion
    if not verify_webhook_auth(authorization):
        logger.warning("Intento de webhook sin autenticacion valida")
        raise HTTPException(status_code=401, detail="No autorizado")
    
    recall_bot_id = payload.recall_bot_id
    exit_code = payload.exit_code
    final_status = payload.status.lower()
    
    logger.info(
        "Webhook bot-exited recibido: recall_bot_id=%s, exit_code=%s, status=%s",
        recall_bot_id,
        exit_code,
        final_status,
    )
    
    # Buscar reunion por recall_bot_id
    meeting = db.query(Meeting).filter(Meeting.recall_bot_id == recall_bot_id).first()
    
    if not meeting:
        logger.warning(
            "Webhook bot-exited: No se encontro reunion con recall_bot_id=%s",
            recall_bot_id,
        )
        return {
            "status": "not_found",
            "message": f"Reunion con recall_bot_id={recall_bot_id} no encontrada",
        }
    
    logger.info(
        "Webhook bot-exited: Reunion encontrada: meeting_id=%s, estado_actual=%s",
        meeting.id,
        meeting.status.value if hasattr(meeting.status, "value") else str(meeting.status),
    )
    
    # Actualizar estado de la reunion
    if final_status == "completed":
        if meeting.status != MeetingStatus.COMPLETED:
            meeting.status = MeetingStatus.COMPLETED
            if not meeting.actual_end_time:
                meeting.actual_end_time = datetime.now(timezone.utc)
            logger.info("Reunion %s actualizada a COMPLETED", meeting.id)
    elif final_status == "failed":
        if meeting.status != MeetingStatus.FAILED:
            meeting.status = MeetingStatus.FAILED
            meeting.error_message = vexa_failure_reason_to_message(payload.reason)
            meeting.recall_status = (payload.reason or "failed").strip() or "failed"
            logger.info(
                "Reunion %s actualizada a FAILED (reason=%s): %s",
                meeting.id,
                payload.reason,
                (meeting.error_message or "")[:200],
            )
    
    db.commit()
    
    # Programar obtencion de transcripcion de VEXA si esta completada
    # Esperar 5 minutos desde ahora para dar tiempo a VEXA a procesar los ultimos segmentos
    if final_status == "completed":
        try:
            # Obtener tiempo de espera configurado (por defecto 5 minutos)
            wait_minutes = getattr(settings, "transcript_wait_minutes", 5)
            
            # Calcular cuándo se ejecutará la tarea (X minutos desde ahora)
            scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=wait_minutes)
            
            logger.info(
                "Webhook bot-exited: Programando tarea de transcripcion para reunion %s (recall_bot_id=%s) "
                "en %d minutos (a las %s)",
                meeting.id,
                recall_bot_id,
                wait_minutes,
                scheduled_time.isoformat(),
            )
            
            # Usar la tarea de Celery con force_full=True para timeout extendido (180s)
            # Programar para ejecutarse en X minutos desde ahora
            task = fetch_vexa_transcript_for_meeting.apply_async(
                args=[str(meeting.id)],
                kwargs={"force_full": True},
                eta=scheduled_time,
            )
            
            # Guardar información de la tarea programada
            meeting.transcript_task_id = task.id
            meeting.transcript_scheduled_time = scheduled_time
            db.commit()
            
            logger.info(
                "Tarea fetch_vexa_transcript_for_meeting (TIMEOUT EXTENDIDO) programada para reunion %s "
                "(task_id=%s, ejecutara en %d minutos)",
                meeting.id,
                task.id,
                wait_minutes,
            )
        except Exception as e:
            logger.error(
                "Error programando tarea de transcripcion para reunion %s: %s",
                meeting.id,
                e,
                exc_info=True,
            )
            # No fallar el webhook si no se puede encolar la tarea
            # Se intentara de nuevo cuando el usuario consulte la reunion o mediante tarea periodica
    
    return {
        "status": "processed",
        "meeting_id": meeting.id,
        "final_status": meeting.status.value if hasattr(meeting.status, "value") else str(meeting.status),
    }

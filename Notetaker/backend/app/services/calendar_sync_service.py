"""Servicio para sincronizar eventos de calendario con reuniones."""
from __future__ import annotations

import html
import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session  # type: ignore[reportMissingImports]

import uuid
from app.models.user import User
from app.models.meeting import Meeting, MeetingStatus
from app.services.google_calendar_service import GoogleCalendarService, InvalidTokenError
from app.services.outlook_calendar_service import OutlookCalendarService
from app.config import settings
from sqlalchemy.orm.attributes import flag_modified

logger = logging.getLogger(__name__)


def _cancel_all_celery_tasks_for_meeting(meeting_id: str) -> int:
    """
    Buscar y cancelar todas las tareas de Celery relacionadas con una reunión.
    
    Busca en tareas activas, reservadas y programadas (scheduled) para encontrar
    todas las tareas que tienen el meeting_id en sus argumentos.
    
    Args:
        meeting_id: ID de la reunión
        
    Returns:
        Número de tareas canceladas
    """
    try:
        from app.celery_app import celery_app
        
        cancelled_count = 0
        
        # Obtener inspector de Celery
        inspect = celery_app.control.inspect()
        
        if not inspect:
            logger.warning("⚠️ No se pudo obtener inspector de Celery (posiblemente no hay workers activos)")
            return 0
        
        # Buscar en tareas activas
        active_tasks = inspect.active()
        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    task_name = task.get("name", "")
                    task_args = task.get("args", [])
                    task_id = task.get("id")
                    
                    # Verificar si es la tarea join_bot_to_meeting y tiene el meeting_id en args
                    if "join_bot_to_meeting" in task_name and task_args:
                        if len(task_args) > 0 and str(task_args[0]) == meeting_id:
                            try:
                                celery_app.control.revoke(task_id, terminate=True)
                                cancelled_count += 1
                                logger.info(f"🚫 Tarea Celery activa cancelada: task_id={task_id}, meeting_id={meeting_id}")
                            except Exception as e:
                                logger.warning(f"⚠️ Error cancelando tarea activa {task_id}: {e}")
        
        # Buscar en tareas reservadas
        reserved_tasks = inspect.reserved()
        if reserved_tasks:
            for worker, tasks in reserved_tasks.items():
                for task in tasks:
                    task_name = task.get("name", "")
                    task_args = task.get("args", [])
                    task_id = task.get("id")
                    
                    if "join_bot_to_meeting" in task_name and task_args:
                        if len(task_args) > 0 and str(task_args[0]) == meeting_id:
                            try:
                                celery_app.control.revoke(task_id, terminate=True)
                                cancelled_count += 1
                                logger.info(f"🚫 Tarea Celery reservada cancelada: task_id={task_id}, meeting_id={meeting_id}")
                            except Exception as e:
                                logger.warning(f"⚠️ Error cancelando tarea reservada {task_id}: {e}")
        
        # Buscar en tareas programadas (scheduled)
        scheduled_tasks = inspect.scheduled()
        if scheduled_tasks:
            for worker, tasks in scheduled_tasks.items():
                for task in tasks:
                    task_name = task.get("name", "")
                    task_request = task.get("request", {})
                    task_args = task_request.get("args", [])
                    task_id = task_request.get("id")
                    
                    if "join_bot_to_meeting" in task_name and task_args:
                        if len(task_args) > 0 and str(task_args[0]) == meeting_id:
                            try:
                                celery_app.control.revoke(task_id, terminate=True)
                                cancelled_count += 1
                                logger.info(f"🚫 Tarea Celery programada cancelada: task_id={task_id}, meeting_id={meeting_id}")
                            except Exception as e:
                                logger.warning(f"⚠️ Error cancelando tarea programada {task_id}: {e}")
        
        return cancelled_count
        
    except Exception as e:
        logger.error(f"❌ Error buscando/cancelando tareas de Celery para reunión {meeting_id}: {e}", exc_info=True)
        return 0


def _cancel_meetings_not_in_any_calendar(
    user: User,
    all_calendar_meetings: Set[Tuple[str, datetime]],
    db: Session,
    cancelled_events: Optional[Set[Tuple[str, datetime]]] = None,
    all_events_seen: Optional[Set[Tuple[str, datetime]]] = None,
) -> int:
    """
    Marcar como canceladas las reuniones PENDING futuras que no están en ninguno
    de los calendarios conectados (Google, Outlook, etc.) como aceptadas.
    Solo se consideran reuniones con acceso (MeetingAccess o user_id legacy).
    Si se pasan cancelled_events y all_events_seen, solo se cancela la reunión
    cuando el evento fue cancelado por el organizador o ya no está en el calendario;
    si el usuario solo rechazó (evento en all_events_seen pero no en calendar_meetings),
    no se cancela aquí (se trata en _handle_user_declined_meetings).
    """
    from app.models.meeting_access import MeetingAccess

    now = datetime.now(timezone.utc)
    now_naive = now.replace(tzinfo=None)
    access_ids = [
        ma.meeting_id
        for ma in db.query(MeetingAccess).filter(MeetingAccess.user_id == user.id).all()
    ]
    legacy_ids = [m.id for m in db.query(Meeting).filter(Meeting.user_id == user.id).all()]
    meeting_ids = set(access_ids) | set(legacy_ids)
    if not meeting_ids:
        return 0
    all_pending_meetings = (
        db.query(Meeting)
        .filter(Meeting.id.in_(meeting_ids))
        .filter(Meeting.status == MeetingStatus.PENDING)
        .filter(Meeting.scheduled_start_time >= now_naive)
        .filter(Meeting.deleted_at.is_(None))
        .all()
    )
    cancelled_count = 0
    for meeting in all_pending_meetings:
        meeting_key = (meeting.meeting_url, meeting.scheduled_start_time)  # type: ignore
        if meeting_key not in all_calendar_meetings:
            if cancelled_events is not None and all_events_seen is not None:
                if meeting_key in all_events_seen and meeting_key not in cancelled_events:
                    continue
            meeting.status = MeetingStatus.CANCELLED  # type: ignore
            cancelled_count += 1
            logger.info(
                f"Reunión cancelada (no está en ningún calendario conectado): "
                f"{meeting.meeting_url} - {meeting.scheduled_start_time}"  # type: ignore
            )
            cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(meeting.id))  # type: ignore
            if meeting.celery_task_id:  # type: ignore
                try:
                    from app.celery_app import celery_app
                    celery_app.control.revoke(meeting.celery_task_id, terminate=True)  # type: ignore
                    logger.info(f"🚫 Tarea Celery específica cancelada para reunión {meeting.id} (task_id={meeting.celery_task_id})")  # type: ignore
                    meeting.celery_task_id = None  # type: ignore
                except Exception as e:
                    logger.warning(f"⚠️ Error cancelando tarea Celery específica {meeting.celery_task_id}: {e}")  # type: ignore
            if cancelled_tasks > 0:
                logger.info(f"✅ Total de tareas Celery canceladas para reunión {meeting.id}: {cancelled_tasks}")
            logger.info("Reunion %s cancelada (bot VEXA se gestiona al borrar reunion)", meeting.id)
    if cancelled_count > 0:
        logger.info(f"{cancelled_count} reuniones marcadas como canceladas (no están en ningún calendario conectado)")
    db.commit()
    return cancelled_count


def _handle_user_declined_meetings(
    user: User,
    declined_meeting_keys: Set[Tuple[str, datetime]],
    db: Session,
) -> int:
    """
    Para reuniones que el usuario tiene en su calendario pero rechazó (o no respondió):
    quitar su acceso (MeetingAccess) para que no aparezcan en próximas. Si queda
    ningún otro usuario con acceso, marcar reunión como cancelada y cancelar Celery.
    """
    from app.models.meeting_access import MeetingAccess

    if not declined_meeting_keys:
        return 0
    now = datetime.now(timezone.utc)
    now_naive = now.replace(tzinfo=None)
    access_ids = [
        ma.meeting_id
        for ma in db.query(MeetingAccess).filter(MeetingAccess.user_id == user.id).all()
    ]
    legacy_ids = [m.id for m in db.query(Meeting).filter(Meeting.user_id == user.id).all()]
    meeting_ids = set(access_ids) | set(legacy_ids)
    if not meeting_ids:
        return 0
    pending_meetings = (
        db.query(Meeting)
        .filter(Meeting.id.in_(meeting_ids))
        .filter(Meeting.status == MeetingStatus.PENDING)
        .filter(Meeting.scheduled_start_time >= now_naive)
        .filter(Meeting.deleted_at.is_(None))
        .all()
    )
    handled = 0
    for meeting in pending_meetings:
        meeting_key = (meeting.meeting_url, meeting.scheduled_start_time)  # type: ignore
        if meeting_key not in declined_meeting_keys:
            continue
        ma = (
            db.query(MeetingAccess)
            .filter(MeetingAccess.meeting_id == meeting.id, MeetingAccess.user_id == user.id)
            .first()
        )
        if ma:
            db.delete(ma)
            handled += 1
            logger.info(
                "Usuario %s rechazó/no respondió: acceso quitado para reunión %s (%s)",
                user.email,
                meeting.id,
                meeting.meeting_url[:80] if meeting.meeting_url else "",
            )
        if meeting.user_id == user.id:
            meeting.user_id = None
        remaining = db.query(MeetingAccess).filter(MeetingAccess.meeting_id == meeting.id).count()
        if remaining == 0 and meeting.user_id is None:
            meeting.status = MeetingStatus.CANCELLED
            meeting.deleted_at = now.replace(tzinfo=None)
            cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(meeting.id))
            if meeting.celery_task_id:
                try:
                    from app.celery_app import celery_app
                    celery_app.control.revoke(meeting.celery_task_id, terminate=True)
                    meeting.celery_task_id = None
                except Exception as e:
                    logger.warning("Error cancelando tarea Celery %s: %s", meeting.celery_task_id, e)
            if cancelled_tasks > 0:
                logger.info("Tareas Celery canceladas para reunión %s (último usuario rechazó): %d", meeting.id, cancelled_tasks)
    if handled > 0:
        db.commit()
    return handled


class CalendarSyncService:
    """Servicio para sincronizar eventos de calendarios externos con reuniones."""

    def __init__(self):
        """Inicializar el servicio de sincronización."""
        self.google_service = GoogleCalendarService()
        self.outlook_service = OutlookCalendarService()

    def _get_user_integration_settings(self, user: User, provider: str) -> Optional[Dict[str, Any]]:
        """
        Obtener configuración de integración del usuario.

        Args:
            user: Usuario
            provider: Proveedor ("google" o "outlook")

        Returns:
            Configuración de la integración o None si no está conectada
        """
        if not user.settings:  # type: ignore
            return None

        integration_key = f"{provider}_calendar"
        integration_data = user.settings.get(integration_key)

        if not integration_data:
            return None

        # Verificar que tenga los tokens necesarios
        if not integration_data.get("access_token"):
            return None

        return integration_data

    @staticmethod
    def _strip_html_for_text(content: str) -> str:
        """Quita etiquetas HTML y normaliza espacios para poder buscar texto (ej. Id. de reunion, Codigo de acceso)."""
        if not content:
            return ""
        # Quitar etiquetas HTML
        text = re.sub(r"<[^>]+>", " ", content)
        # Normalizar espacios
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_meeting_url_from_event(self, event: Dict[str, Any], provider: str) -> Optional[str]:
        """
        Extraer URL de reunión de un evento de calendario.
        Prioriza las URLs de Teams que contienen el ID numérico.
        Si la invitación tiene formato largo de Teams (sin ID en la URL) pero en el cuerpo
        aparecen "Id. de reunión" y "Código de acceso", construye la URL teams.live.com/meet/{id}?p={passcode}
        para que Vexa pueda unirse.

        Args:
            event: Evento del calendario
            provider: Proveedor ("google" o "outlook")

        Returns:
            URL de la reunión o None si no se encuentra
        """
        urls = []
        if provider == "google":
            # Google Calendar puede tener la URL en hangoutLink o conferenceData
            hangout_link = event.get("hangoutLink")
            if hangout_link:
                urls.append(hangout_link)

            conference_data = event.get("conferenceData", {})
            for entry_point in conference_data.get("entryPoints", []):
                if entry_point.get("entryPointType") == "video":
                    uri = entry_point.get("uri")
                    if uri:
                        urls.append(uri)

            description = event.get("description", "")
            if description:
                url_pattern = r"(https?://(?:teams\.microsoft\.com|zoom\.us|meet\.google\.com|.*\.zoom\.us)/[^\s<>\"']+)"
                urls.extend(re.findall(url_pattern, description, re.IGNORECASE))

        elif provider == "outlook":
            # Outlook Calendar puede tener la URL en onlineMeeting o body
            online_meeting = event.get("onlineMeeting")
            if online_meeting:
                join_url = online_meeting.get("joinUrl")
                if join_url:
                    urls.append(join_url)

            body = event.get("body", {})
            content = body.get("content", "")
            if content:
                url_pattern = r"(https?://(?:teams\.microsoft\.com|zoom\.us|meet\.google\.com|.*\.zoom\.us)/[^\s<>\"']+)"
                urls.extend(re.findall(url_pattern, content, re.IGNORECASE))

            location = event.get("location", {})
            if isinstance(location, dict):
                display_name = location.get("displayName", "")
                if display_name:
                    url_pattern = r"(https?://[^\s<>\"']+)"
                    urls.extend(re.findall(url_pattern, display_name))

        if not urls:
            return None

        # Priorizar la URL que contenga el ID numérico (10-15 dígitos) para Teams
        chosen: Optional[str] = None
        for url in urls:
            if "teams.microsoft.com" in url or "teams.live.com" in url:
                if re.search(r"\d{10,15}", url):
                    chosen = url
                    break
        if chosen is None:
            chosen = urls[0]

        # Si la URL elegida es de Teams pero NO tiene ID numérico (formato largo / meetup-join),
        # intentar construir la URL desde el cuerpo (Id. de reunión + Código de acceso)
        if chosen and ("teams.microsoft.com" in chosen or "teams.live.com" in chosen):
            if not re.search(r"\d{10,15}", chosen):
                body_text: str
                if provider == "google":
                    body_text = event.get("description", "") or ""
                else:
                    raw = (event.get("body", {}) or {}).get("content", "") or ""
                    body_text = self._strip_html_for_text(raw)
                body_text = html.unescape(body_text)
                teams_meta = self._extract_teams_metadata(body_text)
                native_id = teams_meta.get("vexa_native_meeting_id")
                if native_id:
                    built_url = f"https://teams.live.com/meet/{native_id}"
                    passcode = teams_meta.get("vexa_passcode")
                    if passcode:
                        built_url += f"?p={passcode}"
                    logger.info(
                        "URL Teams construida desde cuerpo (formato largo): id=%s, passcode=%s",
                        native_id,
                        "si" if passcode else "no",
                    )
                    return built_url

        return chosen

    def _extract_teams_metadata(self, content: str) -> Dict[str, Any]:
        """Extrae el Meeting ID y Passcode del texto del cuerpo del evento (invitación formato antiguo/largo de Teams)."""
        if not content:
            return {}

        content = html.unescape(content)
        metadata = {}

        # Buscar ID de la reunión (formatos: "340 925 748 447 3" o 123456789012)
        # Soportamos etiquetas en español (Id. de reunión como en el correo de Teams) e inglés
        id_patterns = [
            r"(?:Id\. de reunión|Meeting ID|ID de la reunión|ID de reunión):\s*([\d\s]{10,25})",
            r"ID:\s*([\d\s]{10,25})",
        ]
        found_id_raw = None
        for pattern in id_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                found_id_raw = match.group(1)
                clean_id = re.sub(r"\s+", "", found_id_raw)
                if 10 <= len(clean_id) <= 15:
                    metadata["vexa_native_meeting_id"] = clean_id
                    logger.info("Extracted Teams native_meeting_id: %s", clean_id)
                    break
        
        # Buscar Passcode / Código de acceso (ej: "Código de acceso: fS3JK7DX")
        # Unescape por si el cuerpo tiene &oacute; en lugar de ó; y soportar "Codigo" sin tilde
        pass_patterns = [
            r"(?:Passcode|Código de acceso|Codigo de acceso|Clave):\s*([A-Za-z0-9]+)",
            r"Código:\s*([A-Za-z0-9]+)",
            r"Codigo:\s*([A-Za-z0-9]+)",
        ]
        found_pw_raw = None
        for pattern in pass_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                found_pw_raw = match.group(1)
                metadata["vexa_passcode"] = found_pw_raw
                logger.info("Extracted Teams passcode: %s", found_pw_raw)
                break
        
        if found_id_raw and not metadata.get("vexa_native_meeting_id"):
            logger.warning("Found potential Teams ID raw '%s' but did not meet criteria (10-15 digits)", found_id_raw)
                
        return metadata

    def _extract_organizer_from_event(
        self, event: Dict[str, Any], provider: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extraer email y nombre del organizador del evento.
        Outlook: organizer.emailAddress.address, .name
        Google: organizer.email, organizer.displayName
        Returns:
            (organizer_email, organizer_name); (None, None) si no hay organizador.
        """
        if provider == "outlook":
            org = event.get("organizer") or {}
            addr = org.get("emailAddress") or {}
            email = (addr.get("address") or "").strip() or None
            name = (addr.get("name") or "").strip() or None
            if email:
                return (email, name or email)
            return (None, None)
        if provider == "google":
            org = event.get("organizer") or {}
            email = (org.get("email") or "").strip() or None
            name = (org.get("displayName") or "").strip() or None
            if email:
                return (email, name or email)
            return (None, None)
        return (None, None)

    def _ensure_user_meeting_access(
        self, meeting: Meeting, user: User, db: Session
    ) -> None:
        """Crear MeetingAccess para el usuario si no lo tiene (reuniones desde sync)."""
        from app.models.meeting_access import MeetingAccess
        from app.utils.license_utils import get_user_license_level, get_meeting_access_permissions

        existing = (
            db.query(MeetingAccess)
            .filter(MeetingAccess.meeting_id == meeting.id, MeetingAccess.user_id == user.id)
            .first()
        )
        if existing:
            return
        level = get_user_license_level(user)
        perms = get_meeting_access_permissions(level)
        access = MeetingAccess(
            meeting_id=meeting.id,
            user_id=user.id,
            can_view_transcript=perms["can_view_transcript"],
            can_view_audio=perms["can_view_audio"],
            can_view_video=perms["can_view_video"],
        )
        db.add(access)

    def _parse_event_datetime(self, dt_data: Any, provider: str) -> Optional[datetime]:
        """
        Parsear fecha/hora de un evento según el proveedor.
        Convierte todas las fechas a UTC antes de retornarlas.

        Args:
            dt_data: Datos de fecha/hora del evento
            provider: Proveedor ("google" o "outlook")

        Returns:
            Objeto datetime en UTC (timezone-aware) o None
        """
        try:
            if provider == "google":
                if isinstance(dt_data, dict):
                    # Puede ser date (solo fecha) o dateTime (fecha y hora)
                    if "dateTime" in dt_data:
                        # Google Calendar devuelve fechas con timezone (puede ser local o UTC)
                        dt_str = dt_data["dateTime"]
                        event_timezone = dt_data.get("timeZone", None)
                        
                        # Reemplazar Z por +00:00 para que fromisoformat lo entienda
                        if dt_str.endswith("Z"):
                            dt_str = dt_str.replace("Z", "+00:00")
                        
                        dt = datetime.fromisoformat(dt_str)
                        
                        # Si la fecha no tiene timezone pero el evento tiene timezone especificado,
                        # interpretar la fecha como si estuviera en ese timezone
                        if dt.tzinfo is None and event_timezone:
                            # Importar zoneinfo para manejar timezones por nombre
                            try:
                                from zoneinfo import ZoneInfo
                                # Interpretar la fecha como si estuviera en el timezone del evento
                                dt = dt.replace(tzinfo=ZoneInfo(event_timezone))
                                logger.info(f"🔍 Fecha sin timezone, aplicando timezone del evento '{event_timezone}': {dt.isoformat()}")
                            except ImportError:
                                # Fallback si zoneinfo no está disponible (Python < 3.9)
                                try:
                                    import pytz  # type: ignore[reportMissingModuleSource]
                                    tz = pytz.timezone(event_timezone)
                                    dt = tz.localize(dt)
                                    logger.info(f"🔍 Fecha sin timezone, aplicando timezone del evento '{event_timezone}' (pytz): {dt.isoformat()}")
                                except Exception as e:
                                    logger.warning(f"⚠️ No se pudo aplicar timezone {event_timezone}: {e}, asumiendo UTC")
                                    dt = dt.replace(tzinfo=timezone.utc)
                        elif dt.tzinfo is None:
                            # Si no tiene timezone y no hay timezone en el evento, asumir UTC
                            dt = dt.replace(tzinfo=timezone.utc)
                        
                        # Convertir a UTC
                        if dt.tzinfo and dt.tzinfo != timezone.utc:
                            dt_utc = dt.astimezone(timezone.utc)
                            logger.info(f"🕐 Fecha convertida a UTC: {dt_utc.isoformat()} (original con timezone: {dt.isoformat()}, timezone evento: {event_timezone})")
                            return dt_utc
                        else:
                            logger.info(f"🕐 Fecha ya en UTC o sin conversión: {dt.isoformat()} (timezone evento: {event_timezone})")
                            return dt
                    elif "date" in dt_data:
                        # Evento de todo el día (solo fecha, sin hora)
                        # Asumir inicio del día en UTC
                        date_str = dt_data["date"]
                        dt = datetime.fromisoformat(date_str)
                        dt = dt.replace(tzinfo=timezone.utc)
                        return dt
                elif isinstance(dt_data, str):
                    dt_str = dt_data
                    if dt_str.endswith("Z"):
                        dt_str = dt_str.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(dt_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return dt

            elif provider == "outlook":
                if isinstance(dt_data, dict):
                    date_time = dt_data.get("dateTime")
                    time_zone = dt_data.get("timeZone", "UTC")
                    if date_time:
                        # Parsear fecha/hora con timezone y convertir a UTC
                        dt_str = date_time
                        if dt_str.endswith("Z"):
                            dt_str = dt_str.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(dt_str)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                        return dt
                elif isinstance(dt_data, str):
                    dt_str = dt_data
                    if dt_str.endswith("Z"):
                        dt_str = dt_str.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(dt_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return dt

        except Exception as e:
            logger.warning(f"Error parseando fecha/hora: {e}")
            return None

        return None

    async def sync_user_calendars(self, user: User, db: Session) -> Dict[str, Any]:
        """
        Sincronizar eventos de calendarios del usuario con reuniones.

        Args:
            user: Usuario a sincronizar
            db: Sesión de base de datos

        Returns:
            Diccionario con resultados de la sincronización
        """
        results = {
            "google": {"synced": 0, "created": 0, "errors": []},
            "outlook": {"synced": 0, "created": 0, "errors": []},
        }

        # Sincronizar primero Outlook, luego Google (cuando hay sync Outlook -> Google externo, el origen es Outlook)
        outlook_settings = self._get_user_integration_settings(user, "outlook")
        if outlook_settings:
            try:
                outlook_result = await self._sync_outlook_calendar(user, outlook_settings, db)
                results["outlook"] = outlook_result
            except Exception as e:
                logger.error(f"Error sincronizando Outlook Calendar para usuario {user.email}: {e}")
                results["outlook"]["errors"].append(str(e))

        google_settings = self._get_user_integration_settings(user, "google")
        if google_settings:
            try:
                google_result = await self._sync_google_calendar(user, google_settings, db)
                results["google"] = google_result
            except Exception as e:
                logger.error(f"Error sincronizando Google Calendar para usuario {user.email}: {e}")
                results["google"]["errors"].append(str(e))

        # Cancelar reuniones que no están en NINGÚN calendario conectado (unión Google + Outlook)
        # Solo si al menos un sync completó (tiene calendar_meetings); si ambos fallaron, no cancelar
        all_calendar_meetings: Set[Tuple[str, datetime]] = set()
        all_events_seen: Set[Tuple[str, datetime]] = set()
        cancelled_events: Set[Tuple[str, datetime]] = set()
        gcal = results.get("google", {}).get("calendar_meetings")
        ocal = results.get("outlook", {}).get("calendar_meetings")
        if gcal is not None:
            all_calendar_meetings.update(gcal)
        if ocal is not None:
            all_calendar_meetings.update(ocal)
        gseen = results.get("google", {}).get("all_events_seen")
        oseen = results.get("outlook", {}).get("all_events_seen")
        if gseen is not None:
            all_events_seen.update(gseen)
        if oseen is not None:
            all_events_seen.update(oseen)
        gcanc = results.get("google", {}).get("cancelled_events")
        ocanc = results.get("outlook", {}).get("cancelled_events")
        if gcanc is not None:
            cancelled_events.update(gcanc)
        if ocanc is not None:
            cancelled_events.update(ocanc)
        if gcal is not None or ocal is not None:
            try:
                _cancel_meetings_not_in_any_calendar(
                    user, all_calendar_meetings, db,
                    cancelled_events=cancelled_events if cancelled_events or all_events_seen else None,
                    all_events_seen=all_events_seen if cancelled_events or all_events_seen else None,
                )
            except Exception as e:
                logger.error(f"Error cancelando reuniones no presentes en calendarios: {e}")
            try:
                declined_keys = (all_events_seen - cancelled_events - all_calendar_meetings) if all_events_seen else set()
                if declined_keys:
                    _handle_user_declined_meetings(user, declined_keys, db)
            except Exception as e:
                logger.error(f"Error aplicando rechazos de calendario: {e}")

        # Convertir sets a listas para que sean serializables por JSON (Celery)
        if "google" in results and "calendar_meetings" in results["google"]:
            results["google"]["calendar_meetings"] = list(results["google"]["calendar_meetings"])
        if "outlook" in results and "calendar_meetings" in results["outlook"]:
            results["outlook"]["calendar_meetings"] = list(results["outlook"]["calendar_meetings"])
        for key in ("all_events_seen", "cancelled_events"):
            if "google" in results and key in results["google"]:
                results["google"][key] = list(results["google"][key])
            if "outlook" in results and key in results["outlook"]:
                results["outlook"][key] = list(results["outlook"][key])

        return results

    async def run_full_sync_with_renewal(self, user: User, db: Session) -> Dict[str, Any]:
        """
        Renovar suscripciones/watch de calendario si estan por expirar y luego sincronizar.
        Usado por la sync manual (vista), por el boton de admin y por la tarea Beat de renovacion.
        """
        user_email = user.email or ""
        # Renovar watch de Google si esta por expirar (menos de 1 dia)
        if user.settings and user.settings.get("google_calendar") and getattr(settings, "backend_public_url", None):  # type: ignore
            google_data = user.settings.get("google_calendar") or {}  # type: ignore
            watch_expiration = google_data.get("watch_expiration")
            if watch_expiration:
                try:
                    expiration_dt = datetime.fromisoformat(str(watch_expiration).replace("Z", "+00:00"))
                    if expiration_dt.tzinfo is None:
                        expiration_dt = expiration_dt.replace(tzinfo=timezone.utc)
                    now = datetime.now(timezone.utc)
                    days_until_expiration = (expiration_dt - now).total_seconds() / 86400
                    if days_until_expiration < 1:
                        logger.info("Watch de push notifications expirando para %s, renovando...", user_email)
                        access_token = google_data.get("access_token")
                        refresh_token = google_data.get("refresh_token")
                        if access_token:
                            watch_channel_id = google_data.get("watch_channel_id")
                            watch_resource_id = google_data.get("watch_resource_id")
                            if watch_channel_id and watch_resource_id:
                                try:
                                    self.google_service.stop_watch(
                                        access_token=access_token,
                                        refresh_token=refresh_token,
                                        client_id=settings.google_client_id,  # type: ignore
                                        client_secret=settings.google_client_secret,  # type: ignore
                                        channel_id=watch_channel_id,
                                        resource_id=watch_resource_id,
                                    )
                                except Exception as stop_error:
                                    logger.warning("Error deteniendo watch anterior: %s", stop_error)
                            webhook_url = f"{settings.backend_public_url}/api/integrations/webhook/google-calendar"  # type: ignore
                            watch_info = self.google_service.watch_calendar(
                                access_token=access_token,
                                refresh_token=refresh_token,
                                client_id=settings.google_client_id,  # type: ignore
                                client_secret=settings.google_client_secret,  # type: ignore
                                calendar_id=google_data.get("calendar_id", "primary"),
                                webhook_url=webhook_url,
                                user_id=str(user.id),  # type: ignore
                            )
                            google_data["watch_channel_id"] = watch_info["channel_id"]
                            google_data["watch_resource_id"] = watch_info["resource_id"]
                            google_data["watch_expiration"] = watch_info["expiration"]
                            google_data["watch_webhook_url"] = webhook_url
                            flag_modified(user, "settings")
                            db.commit()
                            db.refresh(user)
                            logger.info("Watch de push notifications renovado para %s", user_email)
                except InvalidTokenError:
                    logger.warning("No se pudo renovar watch: token invalido para %s", user_email)
                except Exception as watch_error:
                    logger.warning("Error renovando watch de push notifications: %s", watch_error)

        # Renovar o crear suscripcion Outlook si esta por expirar o no existe
        if user.settings and user.settings.get("outlook_calendar") and getattr(settings, "backend_public_url", None):  # type: ignore
            from datetime import timezone as _tz
            outlook_data = user.settings.get("outlook_calendar") or {}  # type: ignore
            sub_id = outlook_data.get("subscription_id")
            sub_exp = outlook_data.get("subscription_expiration")
            if sub_id and sub_exp:
                try:
                    s = str(sub_exp).replace("Z", "+00:00")
                    exp_dt = datetime.fromisoformat(s)
                    if exp_dt.tzinfo is None:
                        exp_dt = exp_dt.replace(tzinfo=_tz.utc)
                    now = datetime.now(_tz.utc)
                    days_left = (exp_dt - now).total_seconds() / 86400
                    if days_left < 1:
                        logger.info("Suscripcion Outlook por expirar para %s, renovando...", user_email)
                        tok_exp = None
                        if outlook_data.get("token_expires_at"):
                            tok_exp = datetime.fromisoformat(str(outlook_data["token_expires_at"]))
                        renewed = await self.outlook_service.renew_subscription(
                            access_token=outlook_data.get("access_token"),
                            refresh_token=outlook_data.get("refresh_token"),
                            client_id=settings.outlook_client_id or getattr(settings, "graph_client_id", None),
                            client_secret=settings.outlook_client_secret or getattr(settings, "graph_client_secret", None),
                            subscription_id=sub_id,
                            token_expires_at=tok_exp,
                        )
                        outlook_data["subscription_expiration"] = renewed["expirationDateTime"]
                        flag_modified(user, "settings")
                        db.commit()
                        db.refresh(user)
                        logger.info("Suscripcion Outlook renovada para %s", user_email)
                except Exception as ren_error:
                    error_str = str(ren_error).lower()
                    if "404" in error_str or "not found" in error_str:
                        logger.warning("Suscripcion Outlook no existe en Graph (404) para %s, creando nueva...", user_email)
                        outlook_data.pop("subscription_id", None)
                        outlook_data.pop("subscription_expiration", None)
                        flag_modified(user, "settings")
                        db.commit()
                        db.refresh(user)
                    else:
                        logger.warning("Error renovando suscripcion Outlook: %s", ren_error)
            if not outlook_data.get("subscription_id"):
                try:
                    webhook_url = f"{settings.backend_public_url}/api/integrations/webhook/outlook-calendar"  # type: ignore
                    tok_exp = None
                    if outlook_data.get("token_expires_at"):
                        tok_exp = datetime.fromisoformat(str(outlook_data["token_expires_at"]))
                    sub_info = await self.outlook_service.create_subscription(
                        access_token=outlook_data.get("access_token"),
                        refresh_token=outlook_data.get("refresh_token"),
                        client_id=settings.outlook_client_id or getattr(settings, "graph_client_id", None),
                        client_secret=settings.outlook_client_secret or getattr(settings, "graph_client_secret", None),
                        notification_url=webhook_url,
                        token_expires_at=tok_exp,
                    )
                    outlook_data["subscription_id"] = sub_info["id"]
                    outlook_data["subscription_expiration"] = sub_info["expirationDateTime"]
                    outlook_data["subscription_webhook_url"] = webhook_url
                    flag_modified(user, "settings")
                    db.commit()
                    db.refresh(user)
                    logger.info("Suscripcion Outlook creada para %s - id=%s", user_email, sub_info["id"])
                except Exception as cr_err:
                    logger.error("Error creando suscripcion Outlook para %s: %s", user_email, cr_err, exc_info=True)

        return await self.sync_user_calendars(user, db)

    async def _sync_google_calendar(
        self, user: User, integration_settings: Dict[str, Any], db: Session
    ) -> Dict[str, Any]:
        """Sincronizar eventos de Google Calendar."""
        result = {"synced": 0, "created": 0, "errors": [], "needs_reauth": False}

        try:
            # Obtener tokens
            access_token = integration_settings.get("access_token")
            refresh_token = integration_settings.get("refresh_token")
            client_id = settings.google_client_id  # type: ignore
            client_secret = settings.google_client_secret  # type: ignore

            if not access_token or not client_id or not client_secret:
                result["errors"].append("Tokens o configuración faltante")
                return result

            # Obtener eventos de las próximas 30 días
            # Usar timezone-aware datetime (Google Calendar API requiere RFC3339 con timezone)
            time_min = datetime.now(timezone.utc)
            time_max = datetime.now(timezone.utc) + timedelta(days=30)

            # Obtener credenciales y refrescar si es necesario
            # Esto nos permite actualizar los tokens en la BD después de refrescarlos
            credentials = self.google_service.get_credentials_from_tokens(
                access_token, refresh_token, client_id, client_secret
            )
            
            # Refrescar si es necesario (esto puede lanzar InvalidTokenError)
            try:
                credentials = self.google_service._refresh_credentials_if_needed(credentials)
                
                # Si se refrescó el token, actualizar en la BD
                if credentials.token != access_token:
                    logger.info(f"Token refrescado para usuario {user.email}, actualizando en BD")
                    integration_settings["access_token"] = credentials.token
                    if credentials.expiry:
                        integration_settings["token_expires_at"] = credentials.expiry.isoformat()
                    # Guardar refresh_token si cambió (Google puede devolver uno nuevo)
                    if credentials.refresh_token and credentials.refresh_token != refresh_token:
                        integration_settings["refresh_token"] = credentials.refresh_token
                    if not user.settings:  # type: ignore
                        user.settings = {}  # type: ignore
                    user.settings["google_calendar"] = integration_settings  # type: ignore
                    flag_modified(user, "settings")
                    db.commit()
                    logger.info(f"Tokens actualizados en BD para usuario {user.email}")
            except InvalidTokenError as e:
                logger.error(f"Refresh token inválido para usuario {user.email}: {e}")
                result["errors"].append(str(e))
                result["needs_reauth"] = True
                # Marcar la integración como que necesita reautorización
                if not user.settings:  # type: ignore
                    user.settings = {}  # type: ignore
                if "google_calendar" in user.settings:  # type: ignore
                    user.settings["google_calendar"]["needs_reauth"] = True  # type: ignore
                    flag_modified(user, "settings")
                    db.commit()
                return result

            events = self.google_service.get_events(
                access_token=credentials.token,
                refresh_token=credentials.refresh_token,
                client_id=client_id,
                client_secret=client_secret,
                time_min=time_min,
                time_max=time_max,
            )

            result["synced"] = len(events)

            # Conjuntos para distinguir evento cancelado por organizador vs usuario rechazó
            calendar_meetings: Set[Tuple[str, datetime]] = set()
            all_events_seen: Set[Tuple[str, datetime]] = set()
            cancelled_events: Set[Tuple[str, datetime]] = set()

            # Procesar cada evento
            for event in events:
                try:
                    meeting_url = self._extract_meeting_url_from_event(event, "google")
                    if not meeting_url:
                        continue  # No es una reunión con URL

                    # Extraer información del evento
                    start_data = event.get("start", {})
                    end_data = event.get("end", {})
                    
                    # Log para depuración de timezone
                    if "dateTime" in start_data:
                        logger.debug(f"Fecha raw de Google Calendar: {start_data['dateTime']}, timezone: {start_data.get('timeZone', 'no especificado')}")
                    
                    start_time = self._parse_event_datetime(start_data, "google")
                    end_time = self._parse_event_datetime(end_data, "google")
                    
                    # Log fecha parseada
                    if start_time:
                        logger.debug(f"Fecha parseada a UTC: {start_time.isoformat()} (original: {start_data})")

                    if not start_time:
                        continue

                    # Convertir a naive datetime (sin timezone) para guardar en BD
                    start_time_naive = start_time.replace(tzinfo=None) if start_time.tzinfo else start_time
                    end_time_naive = end_time.replace(tzinfo=None) if end_time and end_time.tzinfo else end_time

                    meeting_key = (meeting_url, start_time_naive)
                    all_events_seen.add(meeting_key)

                    # Evento cancelado por el organizador: no procesar ni añadir a calendar_meetings
                    if event.get("status") == "cancelled":
                        cancelled_events.add(meeting_key)
                        continue

                    # Usuario rechazó o no respondió: no procesar ni añadir a calendar_meetings
                    response_status = self.google_service._get_user_response_status(event, user.email or "")
                    if response_status in ("declined", "needsAction"):
                        continue

                    calendar_meetings.add(meeting_key)

                    org_email, org_name = self._extract_organizer_from_event(event, "google")
                    organizer_email = org_email or user.email
                    organizer_name = org_name or (user.display_name or user.email)

                    # Detectar si es una instancia de reunión recurrente
                    # recurringEventId != null indica que es una instancia de una serie recurrente
                    recurring_event_id = event.get("recurringEventId")
                    is_recurring_instance = recurring_event_id is not None
                    
                    if is_recurring_instance:
                        logger.debug(
                            f"[Google Sync] Instancia recurrente detectada: '{event.get('summary', 'Sin titulo')}' "
                            f"fecha={start_time_naive}, recurringEventId={recurring_event_id[:30] if recurring_event_id else 'N/A'}..."
                        )

                    # Verificar si ya existe una reunión con esta URL + fecha/hora (incluyendo eliminadas)
                    # Lookup por (url, start) sin filtrar por organizador
                    existing_meeting = (
                        db.query(Meeting)
                        .filter(Meeting.meeting_url == meeting_url)
                        .filter(Meeting.scheduled_start_time == start_time_naive)  # type: ignore
                        .first()
                    )

                    # Si no se encontró por URL+tiempo exacto, buscar por URL + organizador
                    # Esto detecta cuando el organizador cambia la fecha/hora de una reunión existente
                    # IMPORTANTE: NO aplicar para instancias recurrentes (cada instancia es independiente)
                    if not existing_meeting and organizer_email and not is_recurring_instance:
                        # Primero verificar si ya existe una reunión con la fecha nueva (por si acaso)
                        # Esto evita conflictos cuando hay una reunión cancelada con fecha antigua y otra nueva con fecha nueva
                        meeting_with_new_date = (
                            db.query(Meeting)
                            .filter(Meeting.meeting_url == meeting_url)
                            .filter(Meeting.scheduled_start_time == start_time_naive)  # type: ignore
                            .filter(Meeting.deleted_at.is_(None))  # Solo reuniones no eliminadas
                            .first()
                        )
                        
                        if meeting_with_new_date:
                            # Ya existe una reunión con la fecha nueva, usar esa
                            logger.info(
                                f"✅ [Google Calendar Sync] Ya existe una reunión con la fecha nueva: "
                                f"URL={meeting_url[:80]}..., fecha={start_time_naive}, "
                                f"reunión_id={meeting_with_new_date.id}. Usando esta reunión."
                            )
                            existing_meeting = meeting_with_new_date
                        else:
                            # Buscar reuniones con la misma URL y organizador (PENDING o CANCELLED)
                            # Ordenar por fecha más reciente para tomar la última instancia
                            meeting_with_same_url_organizer = (
                                db.query(Meeting)
                                .filter(Meeting.meeting_url == meeting_url)
                                .filter(Meeting.organizer_email == organizer_email)  # type: ignore
                                .filter(Meeting.status.in_([MeetingStatus.PENDING, MeetingStatus.CANCELLED]))  # type: ignore
                                .filter(Meeting.deleted_at.is_(None))  # Solo reuniones no eliminadas
                                .order_by(Meeting.scheduled_start_time.desc())  # Tomar la más reciente
                                .first()
                            )
                            if meeting_with_same_url_organizer:
                                logger.info(
                                    f"🔄 [Google Calendar Sync] Detectada reunión reagendada (no recurrente): "
                                    f"URL={meeting_url[:80]}..., organizador={organizer_email}, "
                                    f"fecha antigua={meeting_with_same_url_organizer.scheduled_start_time}, "
                                    f"fecha nueva={start_time_naive}. Actualizando reunión existente."
                                )
                                existing_meeting = meeting_with_same_url_organizer
                    
                    # Si aún no se encontró, buscar reuniones a las que el usuario ya tiene acceso con la misma URL
                    # Esto ayuda a usuarios invitados a encontrar reuniones actualizadas por el organizador
                    # IMPORTANTE: NO aplicar para instancias recurrentes (cada instancia es independiente)
                    if not existing_meeting and not is_recurring_instance:
                        from app.models.meeting_access import MeetingAccess
                        # Buscar reuniones con la misma URL a las que el usuario tiene acceso
                        meeting_with_access = (
                            db.query(Meeting)
                            .join(MeetingAccess, Meeting.id == MeetingAccess.meeting_id)
                            .filter(MeetingAccess.user_id == user.id)
                            .filter(Meeting.meeting_url == meeting_url)
                            .filter(Meeting.status.in_([MeetingStatus.PENDING, MeetingStatus.CANCELLED]))  # type: ignore
                            .filter(Meeting.deleted_at.is_(None))  # Solo reuniones no eliminadas
                            .order_by(Meeting.scheduled_start_time.desc())  # Tomar la más reciente
                            .first()
                        )
                        if meeting_with_access:
                            logger.info(
                                f"🔄 [Google Calendar Sync] Detectada reunión con misma URL a la que el usuario tiene acceso pero diferente fecha/hora: "
                                f"URL={meeting_url[:80]}..., usuario={user.email}, "
                                f"fecha antigua={meeting_with_access.scheduled_start_time}, "
                                f"fecha nueva={start_time_naive}. Actualizando reunión existente."
                            )
                            existing_meeting = meeting_with_access

                    if existing_meeting:
                        # Si encontramos una reunión nueva con la fecha correcta y somos el organizador,
                        # limpiar reuniones canceladas duplicadas con fecha antigua (misma URL y organizador)
                        if (existing_meeting.scheduled_start_time == start_time_naive and  # type: ignore
                            existing_meeting.status == MeetingStatus.PENDING and  # type: ignore
                            organizer_email and
                            existing_meeting.organizer_email == organizer_email):  # type: ignore
                            # Buscar reuniones canceladas con la misma URL y organizador pero fecha diferente
                            duplicate_cancelled = (
                                db.query(Meeting)
                                .filter(Meeting.meeting_url == meeting_url)
                                .filter(Meeting.organizer_email == organizer_email)  # type: ignore
                                .filter(Meeting.status == MeetingStatus.CANCELLED)  # type: ignore
                                .filter(Meeting.scheduled_start_time != start_time_naive)  # type: ignore
                                .filter(Meeting.deleted_at.is_(None))  # Solo no eliminadas
                                .filter(Meeting.id != existing_meeting.id)  # Excluir la reunión actual
                                .all()
                            )
                            if duplicate_cancelled:
                                for dup in duplicate_cancelled:
                                    dup.deleted_at = datetime.utcnow()  # type: ignore
                                    logger.info(
                                        f"🗑️ [Google Calendar Sync] Eliminada reunión cancelada duplicada: "
                                        f"ID={dup.id}, URL={meeting_url[:80]}..., "
                                        f"fecha antigua={dup.scheduled_start_time}, organizador={organizer_email}"
                                    )
                        
                        # Actualizar reunión existente con información del calendario
                        event_title = event.get("summary", "Reunión desde Google Calendar")
                        updated = False
                        reactivated = False
                        if org_email is not None:
                            existing_meeting.organizer_email = org_email  # type: ignore
                            updated = True
                        if org_name is not None:
                            existing_meeting.organizer_name = org_name  # type: ignore
                            updated = True
                        self._ensure_user_meeting_access(existing_meeting, user, db)

                        # Si la reunión estaba eliminada (soft delete) y vuelve al calendario, reactivarla
                        if existing_meeting.deleted_at is not None:  # type: ignore
                            existing_meeting.deleted_at = None  # type: ignore
                            reactivated = True
                            updated = True
                            logger.info(f"Reunión reactivada (eliminada anteriormente): {meeting_url}")
                            
                            # Cancelar cualquier tarea de Celery antigua que pueda estar pendiente
                            if existing_meeting.celery_task_id:  # type: ignore
                                try:
                                    from app.celery_app import celery_app
                                    celery_app.control.revoke(existing_meeting.celery_task_id, terminate=True)  # type: ignore
                                    logger.info(f"🚫 Tarea Celery antigua cancelada al reactivar reunión: task_id={existing_meeting.celery_task_id}")  # type: ignore
                                    existing_meeting.celery_task_id = None  # type: ignore  # type: ignore
                                except Exception as e:
                                    logger.warning(f"⚠️ Error cancelando tarea Celery antigua al reactivar: {e}")
                            
                            # Buscar y cancelar cualquier otra tarea de Celery relacionada
                            cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(existing_meeting.id))  # type: ignore
                            if cancelled_tasks > 0:
                                logger.info(f"✅ {cancelled_tasks} tareas Celery adicionales canceladas al reactivar reunión")
                        
                        # Si la reunión estaba cancelada y vuelve al calendario, cambiarla a pending
                        if existing_meeting.status == MeetingStatus.CANCELLED:  # type: ignore
                            existing_meeting.status = MeetingStatus.PENDING  # type: ignore
                            updated = True
                            logger.info(f"Reunión cambió de CANCELLED a PENDING (volvió al calendario): {meeting_url}")
                            
                            # Cancelar cualquier tarea de Celery antigua que pueda estar pendiente
                            if existing_meeting.celery_task_id:  # type: ignore
                                try:
                                    from app.celery_app import celery_app
                                    celery_app.control.revoke(existing_meeting.celery_task_id, terminate=True)  # type: ignore
                                    logger.info(f"🚫 Tarea Celery antigua cancelada al reactivar reunión cancelada: task_id={existing_meeting.celery_task_id}")  # type: ignore
                                    existing_meeting.celery_task_id = None  # type: ignore  # type: ignore
                                except Exception as e:
                                    logger.warning(f"⚠️ Error cancelando tarea Celery antigua al reactivar: {e}")
                            
                            # Buscar y cancelar cualquier otra tarea de Celery relacionada
                            cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(existing_meeting.id))  # type: ignore
                            if cancelled_tasks > 0:
                                logger.info(f"✅ {cancelled_tasks} tareas Celery adicionales canceladas al reactivar reunión cancelada")
                            
                            # Programar tarea de Celery si no tiene bot asignado
                            if not existing_meeting.recall_bot_id:  # type: ignore
                                # IMPORTANTE: Verificar si ya existe otra reunion con la misma URL que tenga un bot asignado Y ACTIVO
                                # Excluimos reuniones COMPLETED, FAILED y CANCELLED porque sus bots ya no estan activos
                                shared_bot_meeting = db.query(Meeting).filter(
                                    Meeting.meeting_url == existing_meeting.meeting_url,
                                    Meeting.recall_bot_id.isnot(None),
                                    Meeting.id != existing_meeting.id,  # Excluir la reunion actual
                                    Meeting.deleted_at.is_(None),  # Solo reuniones no eliminadas
                                    Meeting.status.notin_([MeetingStatus.COMPLETED, MeetingStatus.FAILED, MeetingStatus.CANCELLED]),  # Solo bots activos
                                ).first()
                                
                                if shared_bot_meeting:
                                    # Reutilizar el bot_id de la otra reunion
                                    logger.info(
                                        f"🔄 [Google Calendar Sync] Reutilizando bot existente de otra reunion con la misma URL: "
                                        f"reunión {existing_meeting.id} reutilizará bot_id={shared_bot_meeting.recall_bot_id} "
                                        f"de reunión {shared_bot_meeting.id} (URL: {existing_meeting.meeting_url})"
                                    )
                                    
                                    # Cancelar tareas de Celery propias si las tenía (ya no las necesita)
                                    if existing_meeting.celery_task_id:  # type: ignore
                                        try:
                                            from app.celery_app import celery_app
                                            celery_app.control.revoke(existing_meeting.celery_task_id, terminate=True)  # type: ignore
                                            logger.info(f"🚫 Tarea Celery cancelada al reutilizar bot compartido: task_id={existing_meeting.celery_task_id}")  # type: ignore
                                        except Exception as e:
                                            logger.warning(f"⚠️ Error cancelando tarea Celery al reutilizar bot: {e}")
                                    
                                    # También buscar y cancelar cualquier otra tarea de Celery relacionada
                                    cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(existing_meeting.id))  # type: ignore
                                    if cancelled_tasks > 0:
                                        logger.info(f"✅ {cancelled_tasks} tareas Celery adicionales canceladas al reutilizar bot compartido")
                                    
                                    existing_meeting.recall_bot_id = shared_bot_meeting.recall_bot_id  # type: ignore
                                    existing_meeting.recall_status = shared_bot_meeting.recall_status or "processing"  # type: ignore
                                    existing_meeting.celery_task_id = None  # type: ignore  # type: ignore  # Limpiar celery_task_id ya que no necesita su propia tarea
                                    logger.info(
                                        f"✅ [Google Calendar Sync] Reunión {existing_meeting.id} actualizada para usar bot compartido (bot_id={shared_bot_meeting.recall_bot_id})"
                                    )
                                else:
                                    # No hay bot existente, programar uno nuevo
                                    try:
                                        from app.tasks.meeting_tasks import join_bot_to_meeting
                                        
                                        # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
                                        join_time = start_time_naive - timedelta(minutes=1)
                                        if join_time.tzinfo is None:
                                            join_time = join_time.replace(tzinfo=timezone.utc)
                                        now = datetime.now(timezone.utc)
                                        
                                        if join_time > now:
                                            task = join_bot_to_meeting.apply_async(
                                                args=[existing_meeting.id],
                                                eta=join_time
                                            )
                                            existing_meeting.celery_task_id = task.id  # type: ignore
                                            logger.info(f"📅 Tarea Celery programada para reunión reactivada {existing_meeting.id} (task_id={task.id})")
                                        else:
                                            # Si ya pasó el tiempo, ejecutar inmediatamente
                                            join_bot_to_meeting.delay(existing_meeting.id)
                                            logger.info(f"✅ Tarea Celery enviada para ejecución inmediata (reunión reactivada {existing_meeting.id})")
                                    except Exception as e:
                                        logger.error(f"❌ Error programando tarea Celery para reunión reactivada {existing_meeting.id}: {e}", exc_info=True)
                        
                        # Actualizar metadatos de Teams si están disponibles en la descripción
                        teams_meta = self._extract_teams_metadata(event.get("description", ""))
                        if teams_meta:
                            if not existing_meeting.extra_metadata:
                                existing_meeting.extra_metadata = {}
                            
                            # Asegurar que sea un dict
                            if not isinstance(existing_meeting.extra_metadata, dict):
                                existing_meeting.extra_metadata = dict(existing_meeting.extra_metadata)

                            changed_meta = False
                            for k, v in teams_meta.items():
                                if existing_meeting.extra_metadata.get(k) != v:
                                    existing_meeting.extra_metadata[k] = v
                                    changed_meta = True
                            
                            if changed_meta:
                                updated = True
                                # Forzar actualización del campo JSON en SQLAlchemy
                                flag_modified(existing_meeting, "extra_metadata")
                                logger.info(f"Metadatos de Teams actualizados para reunión existente Google {existing_meeting.id}")

                        # Actualizar título si cambió
                        if existing_meeting.title != event_title:
                            existing_meeting.title = event_title
                            updated = True
                        
                        # Fase 3: Actualizar fechas solo si realmente cambiaron (evitar trabajo redundante)
                        date_changed = False
                        # Verificar explícitamente si la fecha realmente cambió antes de actualizar
                        if existing_meeting.scheduled_start_time != start_time_naive:  # type: ignore
                            old_date = existing_meeting.scheduled_start_time  # type: ignore
                            date_changed = True
                            existing_meeting.scheduled_start_time = start_time_naive  # type: ignore
                            updated = True
                            logger.info(
                                f"🔄 [Google Calendar Sync] Fecha/hora cambiada para reunión {existing_meeting.id}: "
                                f"fecha antigua={old_date}, fecha nueva={start_time_naive}"
                            )
                            
                            # Si cambió la fecha/hora de inicio, cancelar tarea Celery antigua y reprogramar
                            if existing_meeting.celery_task_id:  # type: ignore
                                try:
                                    from app.celery_app import celery_app
                                    celery_app.control.revoke(existing_meeting.celery_task_id, terminate=True)  # type: ignore
                                    logger.info(f"🚫 Tarea Celery cancelada (fecha/hora cambiada): task_id={existing_meeting.celery_task_id}")  # type: ignore
                                    existing_meeting.celery_task_id = None  # type: ignore
                                except Exception as e:
                                    logger.warning(f"⚠️ Error cancelando tarea Celery antigua: {e}")
                            
                            # Cancelar todas las tareas relacionadas
                            cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(existing_meeting.id))  # type: ignore
                            if cancelled_tasks > 0:
                                logger.info(f"✅ {cancelled_tasks} tareas Celery canceladas por cambio de fecha/hora")
                            
                            # Reprogramar tarea de Celery con la nueva fecha/hora si no tiene bot compartido
                            if not existing_meeting.recall_bot_id:  # type: ignore
                                try:
                                    from app.tasks.meeting_tasks import join_bot_to_meeting
                                    
                                    # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
                                    join_time = start_time_naive - timedelta(minutes=1)
                                    if join_time.tzinfo is None:
                                        join_time = join_time.replace(tzinfo=timezone.utc)
                                    now = datetime.now(timezone.utc)
                                    
                                    if join_time > now:
                                        task = join_bot_to_meeting.apply_async(
                                            args=[existing_meeting.id],
                                            eta=join_time
                                        )
                                        existing_meeting.celery_task_id = task.id  # type: ignore
                                        logger.info(f"📅 Tarea Celery reprogramada para nueva fecha/hora {existing_meeting.id} (task_id={task.id})")
                                    else:
                                        # Si ya pasó el tiempo, ejecutar inmediatamente
                                        join_bot_to_meeting.delay(existing_meeting.id)
                                        logger.info(f"✅ Tarea Celery enviada para ejecución inmediata (fecha/hora cambiada, reunión {existing_meeting.id})")
                                except Exception as e:
                                    logger.error(f"❌ Error reprogramando tarea Celery para reunión {existing_meeting.id}: {e}", exc_info=True)
                        else:
                            # Fase 3: La fecha ya es la misma, pero verificar si falta programar la tarea Celery
                            logger.debug(
                                f"ℹ️ [Google Calendar Sync] Reunión {existing_meeting.id} ya tiene la fecha correcta "
                                f"({start_time_naive}), no se actualiza"
                            )
                            
                            # Verificar si falta programar la tarea Celery (puede pasar si la reunión se creó antes de implementar esta funcionalidad)
                            if not existing_meeting.celery_task_id and not existing_meeting.recall_bot_id:  # type: ignore
                                try:
                                    from app.tasks.meeting_tasks import join_bot_to_meeting
                                    
                                    # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
                                    join_time = start_time_naive - timedelta(minutes=1)
                                    if join_time.tzinfo is None:
                                        join_time = join_time.replace(tzinfo=timezone.utc)
                                    now = datetime.now(timezone.utc)
                                    
                                    if join_time > now:
                                        task = join_bot_to_meeting.apply_async(
                                            args=[existing_meeting.id],
                                            eta=join_time
                                        )
                                        existing_meeting.celery_task_id = task.id  # type: ignore
                                        updated = True
                                        logger.info(f"📅 Tarea Celery programada para reunión existente sin tarea {existing_meeting.id} (task_id={task.id})")
                                    else:
                                        # Si ya pasó el tiempo, ejecutar inmediatamente
                                        join_bot_to_meeting.delay(existing_meeting.id)
                                        logger.info(f"✅ Tarea Celery enviada para ejecución inmediata (reunión existente sin tarea {existing_meeting.id})")
                                except Exception as e:
                                    logger.error(f"❌ Error programando tarea Celery para reunión existente {existing_meeting.id}: {e}", exc_info=True)
                        
                        if end_time_naive and existing_meeting.scheduled_end_time != end_time_naive:  # type: ignore
                            existing_meeting.scheduled_end_time = end_time_naive  # type: ignore
                            updated = True
                        
                        if updated:
                            if reactivated:
                                logger.info(f"Reunión reactivada y actualizada desde Google Calendar: {meeting_url}")
                            elif date_changed:
                                logger.info(f"Reunión actualizada desde Google Calendar (fecha/hora cambiada): {meeting_url}")
                            else:
                                logger.info(f"Reunión actualizada desde Google Calendar: {meeting_url}")
                    else:
                        # Crear nueva reunión
                        # Convertir a naive datetime (sin timezone) para guardar en BD
                        start_time_naive = start_time.replace(tzinfo=None) if start_time.tzinfo else start_time
                        end_time_naive = end_time.replace(tzinfo=None) if end_time and end_time.tzinfo else end_time
                        
                        # Extraer metadatos adicionales del cuerpo (Teams ID/Passcode)
                        extra_metadata = {
                            "bot_display_name": getattr(settings, "default_bot_name", "VEXA Bot"),
                            "synced_from": "google"
                        }
                        teams_meta = self._extract_teams_metadata(event.get("description", ""))
                        if teams_meta:
                            extra_metadata.update(teams_meta)

                        new_meeting = Meeting(
                            id=str(uuid.uuid4()),
                            meeting_url=meeting_url,
                            title=event.get("summary", "Reunión desde Google Calendar"),
                            scheduled_start_time=start_time_naive,  # type: ignore
                            scheduled_end_time=end_time_naive,  # type: ignore
                            status=MeetingStatus.PENDING,  # type: ignore  # PENDING: detectada pero no iniciada
                            organizer_email=organizer_email,
                            organizer_name=organizer_name,
                            extra_metadata=extra_metadata,
                        )

                        # IMPORTANTE: Verificar si ya existe otra reunion con la misma URL que tenga un bot asignado Y ACTIVO
                        # Esto garantiza que solo se cree UN bot por URL fisica de Teams
                        # Excluimos reuniones COMPLETED, FAILED y CANCELLED porque sus bots ya no estan activos
                        shared_bot_meeting = db.query(Meeting).filter(
                            Meeting.meeting_url == meeting_url,
                            Meeting.recall_bot_id.isnot(None),
                            Meeting.id != new_meeting.id,  # Excluir la reunion actual (aunque aun no existe en BD)
                            Meeting.deleted_at.is_(None),  # Solo reuniones no eliminadas
                            Meeting.status.notin_([MeetingStatus.COMPLETED, MeetingStatus.FAILED, MeetingStatus.CANCELLED]),  # Solo bots activos
                        ).first()
                        
                        if shared_bot_meeting:
                            # Reutilizar el bot_id de la otra reunion
                            logger.info(
                                f"🔄 [Google Calendar Sync] Nueva reunion reutilizara bot existente: "
                                f"reunion {new_meeting.id} reutilizara bot_id={shared_bot_meeting.recall_bot_id} "
                                f"de reunion {shared_bot_meeting.id} (URL: {meeting_url})"
                            )
                            new_meeting.recall_bot_id = shared_bot_meeting.recall_bot_id  # type: ignore
                            new_meeting.recall_status = shared_bot_meeting.recall_status or "processing"  # type: ignore
                            # No asignar celery_task_id porque reutiliza el bot de otra reunion
                            new_meeting.celery_task_id = None  # type: ignore  # type: ignore
                        else:
                            # No hay bot existente, programar uno nuevo
                            try:
                                from app.tasks.meeting_tasks import join_bot_to_meeting
                                
                                # Calcular cuándo ejecutar la tarea (3 minutos antes del inicio)
                                join_time = start_time_naive - timedelta(minutes=3)
                                if join_time.tzinfo is None:
                                    join_time = join_time.replace(tzinfo=timezone.utc)
                                now = datetime.now(timezone.utc)
                                
                                if join_time > now:
                                    task = join_bot_to_meeting.apply_async(
                                        args=[new_meeting.id],
                                        eta=join_time
                                    )
                                    new_meeting.celery_task_id = task.id  # type: ignore
                                    logger.info(f"📅 Tarea Celery programada para nueva reunión {new_meeting.id} (task_id={task.id})")
                                else:
                                    # Si ya pasó el tiempo, ejecutar inmediatamente
                                    join_bot_to_meeting.delay(new_meeting.id)
                                    logger.info(f"✅ Tarea Celery enviada para ejecución inmediata (nueva reunión {new_meeting.id})")
                            except Exception as e:
                                logger.error(f"❌ Error programando tarea Celery para nueva reunión {new_meeting.id}: {e}", exc_info=True)

                        db.add(new_meeting)
                        self._ensure_user_meeting_access(new_meeting, user, db)
                        result["created"] += 1

                except Exception as e:
                    logger.warning(f"Error procesando evento de Google Calendar: {e}")
                    result["errors"].append(f"Error en evento {event.get('id', 'unknown')}: {str(e)}")

            result["calendar_meetings"] = calendar_meetings
            result["all_events_seen"] = all_events_seen
            result["cancelled_events"] = cancelled_events
            db.commit()


        except InvalidTokenError as e:
            logger.error(f"Token inválido durante sincronización para usuario {user.email}: {e}")
            result["errors"].append(str(e))
            result["needs_reauth"] = True
            # Marcar la integración como que necesita reautorización
            if not user.settings:  # type: ignore
                user.settings = {}  # type: ignore
            if "google_calendar" in user.settings:  # type: ignore
                user.settings["google_calendar"]["needs_reauth"] = True  # type: ignore
                flag_modified(user, "settings")
                db.commit()
            return result
        except Exception as e:
            db.rollback()
            logger.error(f"Error sincronizando Google Calendar: {e}")
            result["errors"].append(str(e))
            raise

        return result

    async def _sync_outlook_calendar(
        self, user: User, integration_settings: Dict[str, Any], db: Session
    ) -> Dict[str, Any]:
        """Sincronizar eventos de Outlook Calendar."""
        result = {"synced": 0, "created": 0, "errors": []}

        try:
            # Obtener tokens
            access_token = integration_settings.get("access_token")
            refresh_token = integration_settings.get("refresh_token")
            
            client_id = settings.outlook_client_id or settings.graph_client_id  # type: ignore
            client_secret = settings.outlook_client_secret or settings.graph_client_secret  # type: ignore
            
            token_expires_at = None
            if integration_settings.get("token_expires_at"):
                token_expires_at = datetime.fromisoformat(str(integration_settings["token_expires_at"]))

            if not access_token or not client_id or not client_secret:
                result["errors"].append("Tokens o configuración faltante")
                return result

            # Renovar token si es necesario (similar a Google Calendar)
            # Esto nos permite actualizar los tokens en la BD después de refrescarlos
            if refresh_token and token_expires_at:
                time_until_expiry = (token_expires_at - datetime.utcnow()).total_seconds()
                if time_until_expiry < 300:  # 5 minutos antes de expirar
                    try:
                        logger.info(f"Token de Outlook expirado o por expirar para usuario {user.email}, renovando...")
                        refreshed = self.outlook_service.refresh_token(
                            refresh_token=refresh_token,
                            client_id=client_id,
                            client_secret=client_secret,
                            is_bot=False,
                        )
                        
                        # Actualizar tokens en la BD
                        integration_settings["access_token"] = refreshed["access_token"]
                        if refreshed.get("token_expires_at"):
                            integration_settings["token_expires_at"] = refreshed["token_expires_at"].isoformat()
                        if refreshed.get("refresh_token"):
                            integration_settings["refresh_token"] = refreshed["refresh_token"]
                        
                        if not user.settings:  # type: ignore
                            user.settings = {}  # type: ignore
                        user.settings["outlook_calendar"] = integration_settings  # type: ignore
                        flag_modified(user, "settings")
                        db.commit()
                        logger.info(f"Tokens de Outlook actualizados en BD para usuario {user.email}")
                        
                        # Actualizar variables locales con los nuevos tokens
                        access_token = refreshed["access_token"]
                        refresh_token = refreshed.get("refresh_token", refresh_token)
                        token_expires_at = refreshed.get("token_expires_at")
                    except ValueError as e:
                        logger.error(f"Error renovando token de Outlook para usuario {user.email}: {e}")
                        result["errors"].append(f"Error renovando token: {str(e)}")
                        # Continuar con el token actual si falla la renovación

            # Obtener eventos de las próximas 30 días
            # Usar timezone-aware datetime para consistencia
            time_min = datetime.now(timezone.utc)
            time_max = datetime.now(timezone.utc) + timedelta(days=30)

            events = await self.outlook_service.get_events(
                access_token=access_token,
                refresh_token=refresh_token,
                client_id=client_id,
                client_secret=client_secret,
                time_min=time_min,
                time_max=time_max,
                token_expires_at=token_expires_at,
                is_bot=False,
            )

            result["synced"] = len(events)

            # Conjuntos para distinguir evento cancelado por organizador vs usuario rechazó
            calendar_meetings: Set[Tuple[str, datetime]] = set()
            all_events_seen: Set[Tuple[str, datetime]] = set()
            cancelled_events: Set[Tuple[str, datetime]] = set()
            skipped_no_url = 0
            skipped_no_start = 0

            # Procesar cada evento
            for event in events:
                try:
                    meeting_url = self._extract_meeting_url_from_event(event, "outlook")
                    if not meeting_url:
                        skipped_no_url += 1
                        continue  # No es una reunión con URL

                    # Extraer información del evento
                    start_data = event.get("start", {})
                    end_data = event.get("end", {})
                    start_time = self._parse_event_datetime(start_data, "outlook")
                    end_time = self._parse_event_datetime(end_data, "outlook")

                    if not start_time:
                        skipped_no_start += 1
                        continue

                    # Convertir a naive datetime (sin timezone) para guardar en BD
                    start_time_naive = start_time.replace(tzinfo=None) if start_time.tzinfo else start_time
                    end_time_naive = end_time.replace(tzinfo=None) if end_time and end_time.tzinfo else end_time

                    meeting_key = (meeting_url, start_time_naive)
                    all_events_seen.add(meeting_key)

                    # Evento cancelado por el organizador: no procesar ni añadir a calendar_meetings
                    if event.get("isCancelled") is True:
                        cancelled_events.add(meeting_key)
                        continue

                    # Usuario rechazó o no respondió: no procesar ni añadir a calendar_meetings
                    response_status = self.outlook_service._get_user_response_status(event, user.email or "")
                    if response_status in ("declined", "notResponded"):
                        continue

                    calendar_meetings.add(meeting_key)

                    org_email, org_name = self._extract_organizer_from_event(event, "outlook")
                    organizer_email = org_email or user.email
                    organizer_name = org_name or (user.display_name or user.email)

                    # Detectar si es una instancia de reunión recurrente
                    # seriesMasterId != null indica que es una instancia de una serie recurrente
                    series_master_id = event.get("seriesMasterId")
                    is_recurring_instance = series_master_id is not None
                    
                    if is_recurring_instance:
                        logger.debug(
                            f"[Outlook Sync] Instancia recurrente detectada: '{event.get('subject', 'Sin titulo')}' "
                            f"fecha={start_time_naive}, seriesMasterId={series_master_id[:30] if series_master_id else 'N/A'}..."
                        )

                    # Verificar si ya existe una reunión con esta URL + fecha/hora (incluyendo eliminadas)
                    # Lookup por (url, start) sin filtrar por organizador
                    existing_meeting = (
                        db.query(Meeting)
                        .filter(Meeting.meeting_url == meeting_url)
                        .filter(Meeting.scheduled_start_time == start_time_naive)  # type: ignore
                        .first()
                    )
                    
                    # Si no se encontró por URL+tiempo exacto, buscar por URL + organizador
                    # Esto detecta cuando el organizador cambia la fecha/hora de una reunión existente
                    # IMPORTANTE: NO aplicar esta lógica para instancias de reuniones recurrentes
                    # Las reuniones recurrentes comparten URL pero cada instancia es una reunión diferente
                    if not existing_meeting and organizer_email and not is_recurring_instance:
                        # Primero verificar si ya existe una reunión con la fecha nueva (por si acaso)
                        # Esto evita conflictos cuando hay una reunión cancelada con fecha antigua y otra nueva con fecha nueva
                        meeting_with_new_date = (
                            db.query(Meeting)
                            .filter(Meeting.meeting_url == meeting_url)
                            .filter(Meeting.scheduled_start_time == start_time_naive)  # type: ignore
                            .filter(Meeting.deleted_at.is_(None))  # Solo reuniones no eliminadas
                            .first()
                        )
                        
                        if meeting_with_new_date:
                            # Ya existe una reunión con la fecha nueva, usar esa
                            logger.info(
                                f"✅ [Outlook Calendar Sync] Ya existe una reunión con la fecha nueva: "
                                f"URL={meeting_url[:80]}..., fecha={start_time_naive}, "
                                f"reunión_id={meeting_with_new_date.id}. Usando esta reunión."
                            )
                            existing_meeting = meeting_with_new_date
                        else:
                            # Buscar reuniones con la misma URL y organizador (PENDING o CANCELLED)
                            # Ordenar por fecha más reciente para tomar la última instancia
                            meeting_with_same_url_organizer = (
                                db.query(Meeting)
                                .filter(Meeting.meeting_url == meeting_url)
                                .filter(Meeting.organizer_email == organizer_email)  # type: ignore
                                .filter(Meeting.status.in_([MeetingStatus.PENDING, MeetingStatus.CANCELLED]))  # type: ignore
                                .filter(Meeting.deleted_at.is_(None))  # Solo reuniones no eliminadas
                                .order_by(Meeting.scheduled_start_time.desc())  # Tomar la más reciente
                                .first()
                            )
                            if meeting_with_same_url_organizer:
                                logger.info(
                                    f"🔄 [Outlook Calendar Sync] Detectada reunión reagendada (no recurrente): "
                                    f"URL={meeting_url[:80]}..., organizador={organizer_email}, "
                                    f"fecha antigua={meeting_with_same_url_organizer.scheduled_start_time}, "
                                    f"fecha nueva={start_time_naive}. Actualizando reunión existente."
                                )
                                existing_meeting = meeting_with_same_url_organizer
                    
                    # Si aún no se encontró, buscar reuniones a las que el usuario ya tiene acceso con la misma URL
                    # Esto ayuda a usuarios invitados a encontrar reuniones actualizadas por el organizador
                    # IMPORTANTE: NO aplicar para instancias recurrentes (cada instancia es independiente)
                    if not existing_meeting and not is_recurring_instance:
                        from app.models.meeting_access import MeetingAccess
                        # Buscar reuniones con la misma URL a las que el usuario tiene acceso
                        meeting_with_access = (
                            db.query(Meeting)
                            .join(MeetingAccess, Meeting.id == MeetingAccess.meeting_id)
                            .filter(MeetingAccess.user_id == user.id)
                            .filter(Meeting.meeting_url == meeting_url)
                            .filter(Meeting.status.in_([MeetingStatus.PENDING, MeetingStatus.CANCELLED]))  # type: ignore
                            .filter(Meeting.deleted_at.is_(None))  # Solo reuniones no eliminadas
                            .order_by(Meeting.scheduled_start_time.desc())  # Tomar la más reciente
                            .first()
                        )
                        if meeting_with_access:
                            logger.info(
                                f"🔄 [Outlook Calendar Sync] Detectada reunión con misma URL a la que el usuario tiene acceso pero diferente fecha/hora: "
                                f"URL={meeting_url[:80]}..., usuario={user.email}, "
                                f"fecha antigua={meeting_with_access.scheduled_start_time}, "
                                f"fecha nueva={start_time_naive}. Actualizando reunión existente."
                            )
                            existing_meeting = meeting_with_access
                    
                    # Si aún no se encontró, buscar reuniones eliminadas con la misma URL
                    # (puede haber cambiado la hora en Outlook y estar eliminada)
                    # IMPORTANTE: Para instancias recurrentes, solo buscar por URL + fecha exacta
                    if not existing_meeting:
                        if is_recurring_instance:
                            # Para recurrentes: buscar eliminada con MISMA fecha exacta
                            deleted_with_same_url = (
                                db.query(Meeting)
                                .filter(Meeting.meeting_url == meeting_url)
                                .filter(Meeting.scheduled_start_time == start_time_naive)  # type: ignore
                                .filter(Meeting.deleted_at.isnot(None))  # Solo buscar entre eliminadas
                                .first()
                            )
                        else:
                            # Para no recurrentes: buscar cualquier eliminada con misma URL
                            deleted_with_same_url = (
                                db.query(Meeting)
                                .filter(Meeting.meeting_url == meeting_url)
                                .filter(Meeting.deleted_at.isnot(None))  # Solo buscar entre eliminadas
                                .order_by(Meeting.scheduled_start_time.desc())  # Tomar la más reciente
                                .first()
                            )
                        if deleted_with_same_url:
                            # Usar esta reunión eliminada para reactivarla con la nueva hora
                            existing_meeting = deleted_with_same_url

                    if existing_meeting:
                        # Si encontramos una reunión nueva con la fecha correcta y somos el organizador,
                        # limpiar reuniones canceladas duplicadas con fecha antigua (misma URL y organizador)
                        if (existing_meeting.scheduled_start_time == start_time_naive and  # type: ignore
                            existing_meeting.status == MeetingStatus.PENDING and  # type: ignore
                            organizer_email and
                            existing_meeting.organizer_email == organizer_email):  # type: ignore
                            # Buscar reuniones canceladas con la misma URL y organizador pero fecha diferente
                            duplicate_cancelled = (
                                db.query(Meeting)
                                .filter(Meeting.meeting_url == meeting_url)
                                .filter(Meeting.organizer_email == organizer_email)  # type: ignore
                                .filter(Meeting.status == MeetingStatus.CANCELLED)  # type: ignore
                                .filter(Meeting.scheduled_start_time != start_time_naive)  # type: ignore
                                .filter(Meeting.deleted_at.is_(None))  # Solo no eliminadas
                                .filter(Meeting.id != existing_meeting.id)  # Excluir la reunión actual
                                .all()
                            )
                            if duplicate_cancelled:
                                for dup in duplicate_cancelled:
                                    dup.deleted_at = datetime.utcnow()  # type: ignore
                                    logger.info(
                                        f"🗑️ [Outlook Calendar Sync] Eliminada reunión cancelada duplicada: "
                                        f"ID={dup.id}, URL={meeting_url[:80]}..., "
                                        f"fecha antigua={dup.scheduled_start_time}, organizador={organizer_email}"
                                    )
                        
                        # Actualizar reunión existente con información del calendario
                        event_title = event.get("subject", "Reunión desde Outlook Calendar")
                        updated = False
                        reactivated = False
                        if org_email is not None:
                            existing_meeting.organizer_email = org_email  # type: ignore
                            updated = True
                        if org_name is not None:
                            existing_meeting.organizer_name = org_name  # type: ignore
                            updated = True
                        self._ensure_user_meeting_access(existing_meeting, user, db)

                        # Si la reunión estaba eliminada (soft delete) y vuelve al calendario, reactivarla
                        if existing_meeting.deleted_at is not None:  # type: ignore
                            existing_meeting.deleted_at = None  # type: ignore
                            reactivated = True
                            updated = True
                            logger.info(f"Reunión reactivada (eliminada anteriormente): {meeting_url}")
                            
                            # Cancelar cualquier tarea de Celery antigua que pueda estar pendiente
                            if existing_meeting.celery_task_id:  # type: ignore
                                try:
                                    from app.celery_app import celery_app
                                    celery_app.control.revoke(existing_meeting.celery_task_id, terminate=True)  # type: ignore
                                    logger.info(f"🚫 Tarea Celery antigua cancelada al reactivar reunión: task_id={existing_meeting.celery_task_id}")  # type: ignore
                                    existing_meeting.celery_task_id = None  # type: ignore  # type: ignore
                                except Exception as e:
                                    logger.warning(f"⚠️ Error cancelando tarea Celery antigua al reactivar: {e}")
                            
                            # Buscar y cancelar cualquier otra tarea de Celery relacionada
                            cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(existing_meeting.id))  # type: ignore
                            if cancelled_tasks > 0:
                                logger.info(f"✅ {cancelled_tasks} tareas Celery adicionales canceladas al reactivar reunión")
                        
                        # Si la reunión estaba cancelada y vuelve al calendario, cambiarla a pending
                        if existing_meeting.status == MeetingStatus.CANCELLED:  # type: ignore
                            existing_meeting.status = MeetingStatus.PENDING  # type: ignore
                            updated = True
                            logger.info(f"Reunión cambió de CANCELLED a PENDING (volvió al calendario): {meeting_url}")
                            
                            # Cancelar cualquier tarea de Celery antigua que pueda estar pendiente
                            if existing_meeting.celery_task_id:  # type: ignore
                                try:
                                    from app.celery_app import celery_app
                                    celery_app.control.revoke(existing_meeting.celery_task_id, terminate=True)  # type: ignore
                                    logger.info(f"🚫 Tarea Celery antigua cancelada al reactivar reunión cancelada: task_id={existing_meeting.celery_task_id}")  # type: ignore
                                    existing_meeting.celery_task_id = None  # type: ignore  # type: ignore
                                except Exception as e:
                                    logger.warning(f"⚠️ Error cancelando tarea Celery antigua al reactivar: {e}")
                            
                            # Buscar y cancelar cualquier otra tarea de Celery relacionada
                            cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(existing_meeting.id))  # type: ignore
                            if cancelled_tasks > 0:
                                logger.info(f"✅ {cancelled_tasks} tareas Celery adicionales canceladas al reactivar reunión cancelada")
                            
                            # Programar tarea de Celery si no tiene bot asignado
                            if not existing_meeting.recall_bot_id:  # type: ignore
                                # IMPORTANTE: Verificar si ya existe otra reunion con la misma URL que tenga un bot asignado Y ACTIVO
                                # Excluimos reuniones COMPLETED, FAILED y CANCELLED porque sus bots ya no estan activos
                                shared_bot_meeting = db.query(Meeting).filter(
                                    Meeting.meeting_url == existing_meeting.meeting_url,
                                    Meeting.recall_bot_id.isnot(None),
                                    Meeting.id != existing_meeting.id,  # Excluir la reunion actual
                                    Meeting.deleted_at.is_(None),  # Solo reuniones no eliminadas
                                    Meeting.status.notin_([MeetingStatus.COMPLETED, MeetingStatus.FAILED, MeetingStatus.CANCELLED]),  # Solo bots activos
                                ).first()
                                
                                if shared_bot_meeting:
                                    # Reutilizar el bot_id de la otra reunion
                                    logger.info(
                                        f"🔄 [Outlook Calendar Sync] Reutilizando bot existente de otra reunion con la misma URL: "
                                        f"reunión {existing_meeting.id} reutilizará bot_id={shared_bot_meeting.recall_bot_id} "
                                        f"de reunión {shared_bot_meeting.id} (URL: {existing_meeting.meeting_url})"
                                    )
                                    
                                    # Cancelar tareas de Celery propias si las tenía (ya no las necesita)
                                    if existing_meeting.celery_task_id:  # type: ignore
                                        try:
                                            from app.celery_app import celery_app
                                            celery_app.control.revoke(existing_meeting.celery_task_id, terminate=True)  # type: ignore
                                            logger.info(f"🚫 Tarea Celery cancelada al reutilizar bot compartido: task_id={existing_meeting.celery_task_id}")  # type: ignore
                                        except Exception as e:
                                            logger.warning(f"⚠️ Error cancelando tarea Celery al reutilizar bot: {e}")
                                    
                                    # También buscar y cancelar cualquier otra tarea de Celery relacionada
                                    cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(existing_meeting.id))  # type: ignore
                                    if cancelled_tasks > 0:
                                        logger.info(f"✅ {cancelled_tasks} tareas Celery adicionales canceladas al reutilizar bot compartido")
                                    
                                    existing_meeting.recall_bot_id = shared_bot_meeting.recall_bot_id  # type: ignore
                                    existing_meeting.recall_status = shared_bot_meeting.recall_status or "processing"  # type: ignore
                                    existing_meeting.celery_task_id = None  # type: ignore  # type: ignore  # Limpiar celery_task_id ya que no necesita su propia tarea
                                    logger.info(
                                        f"✅ [Outlook Calendar Sync] Reunión {existing_meeting.id} actualizada para usar bot compartido (bot_id={shared_bot_meeting.recall_bot_id})"
                                    )
                                else:
                                    # No hay bot existente, programar uno nuevo
                                    try:
                                        from app.tasks.meeting_tasks import join_bot_to_meeting
                                        
                                        # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
                                        join_time = start_time_naive - timedelta(minutes=1)
                                        if join_time.tzinfo is None:
                                            join_time = join_time.replace(tzinfo=timezone.utc)
                                        now = datetime.now(timezone.utc)
                                        
                                        if join_time > now:
                                            task = join_bot_to_meeting.apply_async(
                                                args=[existing_meeting.id],
                                                eta=join_time
                                            )
                                            existing_meeting.celery_task_id = task.id  # type: ignore
                                            logger.info(f"📅 Tarea Celery programada para reunión reactivada {existing_meeting.id} (task_id={task.id})")
                                        else:
                                            # Si ya pasó el tiempo, ejecutar inmediatamente
                                            join_bot_to_meeting.delay(existing_meeting.id)
                                            logger.info(f"✅ Tarea Celery enviada para ejecución inmediata (reunión reactivada {existing_meeting.id})")
                                    except Exception as e:
                                        logger.error(f"❌ Error programando tarea Celery para reunión reactivada {existing_meeting.id}: {e}", exc_info=True)
                        
                        # Actualizar metadatos de Teams si están disponibles en el cuerpo
                        teams_meta = self._extract_teams_metadata(event.get("body", {}).get("content", ""))
                        if teams_meta:
                            if not existing_meeting.extra_metadata:
                                existing_meeting.extra_metadata = {}
                            
                            # Asegurar que sea un dict
                            if not isinstance(existing_meeting.extra_metadata, dict):
                                existing_meeting.extra_metadata = dict(existing_meeting.extra_metadata)

                            changed_meta = False
                            for k, v in teams_meta.items():
                                if existing_meeting.extra_metadata.get(k) != v:
                                    existing_meeting.extra_metadata[k] = v
                                    changed_meta = True
                            
                            if changed_meta:
                                updated = True
                                # Forzar actualización del campo JSON en SQLAlchemy
                                flag_modified(existing_meeting, "extra_metadata")
                                logger.info(f"Metadatos de Teams actualizados para reunión existente Outlook {existing_meeting.id}")

                        # Actualizar título si cambió
                        if existing_meeting.title != event_title:
                            existing_meeting.title = event_title
                            updated = True
                        
                        # Fase 3: Actualizar fechas solo si realmente cambiaron (evitar trabajo redundante)
                        date_changed = False
                        # Verificar explícitamente si la fecha realmente cambió antes de actualizar
                        if existing_meeting.scheduled_start_time != start_time_naive:  # type: ignore
                            old_date = existing_meeting.scheduled_start_time  # type: ignore
                            date_changed = True
                            existing_meeting.scheduled_start_time = start_time_naive  # type: ignore
                            updated = True
                            logger.info(
                                f"🔄 [Outlook Calendar Sync] Fecha/hora cambiada para reunión {existing_meeting.id}: "
                                f"fecha antigua={old_date}, fecha nueva={start_time_naive}"
                            )
                            
                            # Si cambió la fecha/hora de inicio, cancelar tarea Celery antigua y reprogramar
                            if existing_meeting.celery_task_id:  # type: ignore
                                try:
                                    from app.celery_app import celery_app
                                    celery_app.control.revoke(existing_meeting.celery_task_id, terminate=True)  # type: ignore
                                    logger.info(f"🚫 Tarea Celery cancelada (fecha/hora cambiada): task_id={existing_meeting.celery_task_id}")  # type: ignore
                                    existing_meeting.celery_task_id = None  # type: ignore
                                except Exception as e:
                                    logger.warning(f"⚠️ Error cancelando tarea Celery antigua: {e}")
                            
                            # Cancelar todas las tareas relacionadas
                            cancelled_tasks = _cancel_all_celery_tasks_for_meeting(str(existing_meeting.id))  # type: ignore
                            if cancelled_tasks > 0:
                                logger.info(f"✅ {cancelled_tasks} tareas Celery canceladas por cambio de fecha/hora")
                            
                            # Reprogramar tarea de Celery con la nueva fecha/hora si no tiene bot compartido
                            if not existing_meeting.recall_bot_id:  # type: ignore
                                try:
                                    from app.tasks.meeting_tasks import join_bot_to_meeting
                                    
                                    # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
                                    join_time = start_time_naive - timedelta(minutes=1)
                                    if join_time.tzinfo is None:
                                        join_time = join_time.replace(tzinfo=timezone.utc)
                                    now = datetime.now(timezone.utc)
                                    
                                    if join_time > now:
                                        task = join_bot_to_meeting.apply_async(
                                            args=[existing_meeting.id],
                                            eta=join_time
                                        )
                                        existing_meeting.celery_task_id = task.id  # type: ignore
                                        logger.info(f"📅 Tarea Celery reprogramada para nueva fecha/hora {existing_meeting.id} (task_id={task.id})")
                                    else:
                                        # Si ya pasó el tiempo, ejecutar inmediatamente
                                        join_bot_to_meeting.delay(existing_meeting.id)
                                        logger.info(f"✅ Tarea Celery enviada para ejecución inmediata (fecha/hora cambiada, reunión {existing_meeting.id})")
                                except Exception as e:
                                    logger.error(f"❌ Error reprogramando tarea Celery para reunión {existing_meeting.id}: {e}", exc_info=True)
                        else:
                            # Fase 3: La fecha ya es la misma, pero verificar si falta programar la tarea Celery
                            logger.debug(
                                f"ℹ️ [Outlook Calendar Sync] Reunión {existing_meeting.id} ya tiene la fecha correcta "
                                f"({start_time_naive}), no se actualiza"
                            )
                            
                            # Verificar si falta programar la tarea Celery (puede pasar si la reunión se creó antes de implementar esta funcionalidad)
                            if not existing_meeting.celery_task_id and not existing_meeting.recall_bot_id:  # type: ignore
                                try:
                                    from app.tasks.meeting_tasks import join_bot_to_meeting
                                    
                                    # Calcular cuándo ejecutar la tarea (1 minuto antes del inicio)
                                    join_time = start_time_naive - timedelta(minutes=1)
                                    if join_time.tzinfo is None:
                                        join_time = join_time.replace(tzinfo=timezone.utc)
                                    now = datetime.now(timezone.utc)
                                    
                                    if join_time > now:
                                        task = join_bot_to_meeting.apply_async(
                                            args=[existing_meeting.id],
                                            eta=join_time
                                        )
                                        existing_meeting.celery_task_id = task.id  # type: ignore
                                        updated = True
                                        logger.info(f"📅 Tarea Celery programada para reunión existente sin tarea {existing_meeting.id} (task_id={task.id})")
                                    else:
                                        # Si ya pasó el tiempo, ejecutar inmediatamente
                                        join_bot_to_meeting.delay(existing_meeting.id)
                                        logger.info(f"✅ Tarea Celery enviada para ejecución inmediata (reunión existente sin tarea {existing_meeting.id})")
                                except Exception as e:
                                    logger.error(f"❌ Error programando tarea Celery para reunión existente {existing_meeting.id}: {e}", exc_info=True)
                        
                        if end_time_naive and existing_meeting.scheduled_end_time != end_time_naive:  # type: ignore
                            existing_meeting.scheduled_end_time = end_time_naive  # type: ignore
                            updated = True
                        
                        if updated:
                            if reactivated:
                                logger.info(f"Reunión reactivada y actualizada desde Outlook Calendar: {meeting_url}")
                            elif date_changed:
                                logger.info(f"Reunión actualizada desde Outlook Calendar (fecha/hora cambiada): {meeting_url}")
                            else:
                                logger.info(f"Reunión actualizada desde Outlook Calendar: {meeting_url}")
                    else:
                        # Crear nueva reunión
                        # Convertir a naive datetime (sin timezone) para guardar en BD
                        start_time_naive = start_time.replace(tzinfo=None) if start_time.tzinfo else start_time
                        end_time_naive = end_time.replace(tzinfo=None) if end_time and end_time.tzinfo else end_time
                        
                        # Extraer metadatos adicionales del cuerpo (Teams ID/Passcode)
                        extra_metadata = {
                            "bot_display_name": getattr(settings, "default_bot_name", "VEXA Bot"),
                            "synced_from": "outlook"
                        }
                        teams_meta = self._extract_teams_metadata(event.get("body", {}).get("content", ""))
                        if teams_meta:
                            extra_metadata.update(teams_meta)

                        new_meeting = Meeting(
                            id=str(uuid.uuid4()),
                            meeting_url=meeting_url,
                            title=event.get("subject", "Reunión desde Outlook Calendar"),
                            scheduled_start_time=start_time_naive,  # type: ignore
                            scheduled_end_time=end_time_naive,  # type: ignore
                            status=MeetingStatus.PENDING,  # type: ignore  # PENDING: detectada pero no iniciada
                            organizer_email=organizer_email,
                            organizer_name=organizer_name,
                            extra_metadata=extra_metadata,
                        )

                        # IMPORTANTE: Verificar si ya existe otra reunion con la misma URL que tenga un bot asignado Y ACTIVO
                        # Esto garantiza que solo se cree UN bot por URL fisica de Teams
                        # Excluimos reuniones COMPLETED, FAILED y CANCELLED porque sus bots ya no estan activos
                        shared_bot_meeting = db.query(Meeting).filter(
                            Meeting.meeting_url == meeting_url,
                            Meeting.recall_bot_id.isnot(None),
                            Meeting.id != new_meeting.id,  # Excluir la reunion actual (aunque aun no existe en BD)
                            Meeting.deleted_at.is_(None),  # Solo reuniones no eliminadas
                            Meeting.status.notin_([MeetingStatus.COMPLETED, MeetingStatus.FAILED, MeetingStatus.CANCELLED]),  # Solo bots activos
                        ).first()
                        
                        if shared_bot_meeting:
                            # Reutilizar el bot_id de la otra reunion
                            logger.info(
                                f"🔄 [Outlook Calendar Sync] Nueva reunion reutilizara bot existente: "
                                f"reunion {new_meeting.id} reutilizara bot_id={shared_bot_meeting.recall_bot_id} "
                                f"de reunion {shared_bot_meeting.id} (URL: {meeting_url})"
                            )
                            new_meeting.recall_bot_id = shared_bot_meeting.recall_bot_id  # type: ignore
                            new_meeting.recall_status = shared_bot_meeting.recall_status or "processing"  # type: ignore
                            # No asignar celery_task_id porque reutiliza el bot de otra reunion
                            new_meeting.celery_task_id = None  # type: ignore  # type: ignore
                        else:
                            # No hay bot existente, programar uno nuevo
                            try:
                                from app.tasks.meeting_tasks import join_bot_to_meeting
                                
                                # Calcular cuándo ejecutar la tarea (3 minutos antes del inicio)
                                join_time = start_time_naive - timedelta(minutes=3)
                                if join_time.tzinfo is None:
                                    join_time = join_time.replace(tzinfo=timezone.utc)
                                now = datetime.now(timezone.utc)
                                
                                if join_time > now:
                                    task = join_bot_to_meeting.apply_async(
                                        args=[new_meeting.id],
                                        eta=join_time
                                    )
                                    new_meeting.celery_task_id = task.id  # type: ignore
                                    logger.info(f"📅 Tarea Celery programada para nueva reunión {new_meeting.id} (task_id={task.id})")
                                else:
                                    # Si ya pasó el tiempo, ejecutar inmediatamente
                                    join_bot_to_meeting.delay(new_meeting.id)
                                    logger.info(f"✅ Tarea Celery enviada para ejecución inmediata (nueva reunión {new_meeting.id})")
                            except Exception as e:
                                logger.error(f"❌ Error programando tarea Celery para nueva reunión {new_meeting.id}: {e}", exc_info=True)

                        db.add(new_meeting)
                        self._ensure_user_meeting_access(new_meeting, user, db)
                        result["created"] += 1

                except Exception as e:
                    logger.warning(f"Error procesando evento de Outlook Calendar: {e}")
                    result["errors"].append(f"Error en evento {event.get('id', 'unknown')}: {str(e)}")

            result["all_events_seen"] = all_events_seen
            result["cancelled_events"] = cancelled_events
            if skipped_no_url > 0 or skipped_no_start > 0:
                logger.info(
                    "Outlook sync: %d eventos omitidos sin URL de reunión, %d sin start válido; "
                    "%d reuniones procesadas",
                    skipped_no_url,
                    skipped_no_start,
                    len(calendar_meetings),
                )

            result["calendar_meetings"] = calendar_meetings
            db.commit()


        except Exception as e:
            db.rollback()
            logger.error(f"Error sincronizando Outlook Calendar: {e}")
            result["errors"].append(str(e))
            raise

        return result


"""
Cliente para la API de VEXA (bot en reuniones Teams y transcripciones).

Sustituye el uso de Playwright/Recall/API de Teams para:
- Meter el bot en una reunion de Teams (POST /bots)
- Parar el bot (DELETE /bots/teams/{native_id})
- Obtener transcripciones (GET /transcripts/teams/{native_id})

Configuracion: VEXA_API_BASE_URL, VEXA_API_KEY en .env.
Documentacion: docs/VEXA_API.md
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse, unquote

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

PLATFORM_TEAMS = "teams"


def parse_teams_meeting_url(meeting_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrae native_meeting_id y passcode de una URL de Teams.

    Formatos soportados:
    - https://teams.live.com/meet/9366473044740?p=waw4q9dPAvdIG3aknh
    - https://teams.microsoft.com/l/meetup-join/19%3ameeting_...@thread.v2/0?...
    - https://teams.microsoft.com/... (como fallback, busca secuencia de 10-15 digitos)

    Returns:
        (native_meeting_id, passcode) o (None, None) si no se puede extraer.
    """
    if not meeting_url or not isinstance(meeting_url, str):
        return (None, None)

    url = meeting_url.strip()
    native_id: Optional[str] = None
    passcode: Optional[str] = None

    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if "p" in query and query["p"]:
            passcode = query["p"][0].strip() or None

        # Caso 1: teams.live.com/meet/<numerico>
        if "teams.live.com" in parsed.netloc and "/meet/" in parsed.path:
            m_live = re.search(r"/meet/(\d{10,15})", parsed.path)
            if m_live:
                native_id = m_live.group(1)

        # Caso 2: teams.microsoft.com/l/meetup-join/19%3ameeting_...@thread.v2/0
        if not native_id and "teams.microsoft.com" in parsed.netloc and "/l/meetup-join/" in parsed.path:
            try:
                parts = [p for p in parsed.path.split("/") if p]
                # Esperado: ["l", "meetup-join", "19%3ameeting_...@thread.v2", "0"]
                if len(parts) >= 3 and parts[0] == "l" and parts[1] == "meetup-join":
                    encoded_identifier = parts[2]
                    decoded_identifier = unquote(encoded_identifier).strip()
                    if decoded_identifier:
                        native_id = decoded_identifier
            except Exception as e:
                logger.warning("parse_teams_meeting_url meetup-join parse failed: %s", e)

        # Caso 3 (fallback): buscar cualquier secuencia numerica larga en la URL
        if not native_id:
            m_any = re.search(r"(\d{10,15})", url)
            if m_any:
                native_id = m_any.group(1)
    except Exception as e:
        logger.warning("parse_teams_meeting_url failed: %s", e)

    return (native_id, passcode)


class VexaServiceError(Exception):
    """Error en llamada a VEXA API."""
    pass


class VexaService:
    """
    Cliente para la API de VEXA (User API - API Gateway).
    Usa X-API-Key para autenticacion.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or getattr(settings, "vexa_api_key", None)
        self.base_url = (base_url or getattr(settings, "vexa_api_base_url", "")).rstrip("/")

        if not self.api_key:
            raise ValueError(
                "VEXA_API_KEY no esta configurada. Anadela en tu .env como VEXA_API_KEY=..."
            )
        if not self.base_url:
            raise ValueError(
                "VEXA_API_BASE_URL no esta configurada. Ej: http://172.29.14.10:8056"
            )

    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def start_teams_bot(
        self,
        native_meeting_id: str,
        passcode: Optional[str] = None,
        bot_name: Optional[str] = None,
        language: Optional[str] = "es",
    ) -> Dict[str, Any]:
        """
        Crea un bot que se une a una reunion de Teams (POST /bots).

        Args:
            native_meeting_id: ID numerico de la reunion Teams (solo digitos).
            passcode: Codigo de acceso/PIN de la reunion si lo requiere.
            bot_name: Nombre con el que el bot aparece en la reunion (opcional).
            language: Codigo de idioma para la transcripcion (ej. 'es', 'en'). Por defecto 'es'.

        Returns:
            Diccionario con la respuesta de VEXA (ej. id del bot/reunion, status).

        Raises:
            VexaServiceError: Si la API devuelve error.
        """
        body: Dict[str, Any] = {
            "platform": PLATFORM_TEAMS,
            "native_meeting_id": native_meeting_id.strip(),
        }
        if passcode is not None and passcode.strip():
            body["passcode"] = passcode.strip()
        if bot_name is not None and bot_name.strip():
            body["bot_name"] = bot_name.strip()
        if language is not None and language.strip():
            body["language"] = language.strip()

        url = f"{self.base_url}/bots"
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    url,
                    headers=self._headers(),
                    json=body,
                )
        except httpx.RequestError as e:
            logger.exception("VEXA POST /bots request error: %s", e)
            raise VexaServiceError(f"Error de conexion con VEXA: {e}") from e

        if resp.status_code >= 400:
            logger.warning(
                "VEXA POST /bots failed: status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise VexaServiceError(
                f"VEXA API error {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json()
        except Exception as e:
            logger.warning("VEXA POST /bots response not JSON: %s", resp.text[:200])
            raise VexaServiceError(f"Respuesta no JSON de VEXA: {e}") from e

        logger.info(
            "VEXA bot started for teams meeting native_id=%s",
            native_meeting_id,
        )
        return data

    def stop_teams_bot(
        self,
        native_meeting_id: str,
        platform: str = PLATFORM_TEAMS,
    ) -> None:
        """
        Detiene el bot en una reunion (DELETE /bots/{platform}/{native_meeting_id}).

        Args:
            native_meeting_id: ID numerico de la reunion Teams.
            platform: Plataforma; por defecto "teams".

        Raises:
            VexaServiceError: Si la API devuelve error.
        """
        path = f"/bots/{platform}/{native_meeting_id.strip()}"
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.delete(url, headers=self._headers())
        except httpx.RequestError as e:
            logger.exception("VEXA DELETE /bots request error: %s", e)
            raise VexaServiceError(f"Error de conexion con VEXA: {e}") from e
        if resp.status_code >= 400:
            logger.warning("VEXA DELETE /bots failed: status=%s body=%s", resp.status_code, resp.text[:500])
            raise VexaServiceError(f"VEXA API error {resp.status_code}: {resp.text[:500]}")
        logger.info("VEXA bot stopped for teams meeting native_id=%s", native_meeting_id)

    def get_transcript(
        self,
        native_meeting_id: str,
        platform: str = PLATFORM_TEAMS,
        meeting_id: Optional[str] = None,
        use_extended_timeout: bool = False,
    ) -> Dict[str, Any]:
        """
        Obtiene la transcripcion de una reunion (GET /transcripts/{platform}/{native_id}).

        Args:
            native_meeting_id: ID numerico de la reunion (Teams).
            platform: Plataforma; por defecto "teams".
            meeting_id: ID interno de reunion en VEXA (opcional, para query ?meeting_id=...).
            use_extended_timeout: Si es True, usa timeout de 180s en lugar de 30s para reuniones largas.

        Returns:
            Diccionario con "segments" (lista de segmentos con text, speaker,
            absolute_start_time, absolute_end_time, etc.).

        Raises:
            VexaServiceError: Si la API devuelve error.
        """
        path = f"/transcripts/{platform}/{native_meeting_id.strip()}"
        if meeting_id:
            path += f"?meeting_id={meeting_id}"
        url = f"{self.base_url}{path}"

        # Usar timeout extendido para reuniones largas (permite que VEXA procese mas segmentos)
        timeout_value = 180.0 if use_extended_timeout else 30.0
        
        logger.info(
            "[VEXA] Obteniendo transcripcion para native_id=%s (timeout=%ss)",
            native_meeting_id,
            timeout_value
        )

        try:
            with httpx.Client(timeout=timeout_value) as client:
                resp = client.get(
                    url,
                    headers=self._headers(),
                )
        except httpx.RequestError as e:
            logger.exception("VEXA GET /transcripts request error: %s", e)
            raise VexaServiceError(f"Error de conexion con VEXA: {e}") from e

        if resp.status_code >= 400:
            logger.warning(
                "VEXA GET /transcripts failed: status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise VexaServiceError(
                f"VEXA API error {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json()
        except Exception as e:
            logger.warning(
                "VEXA GET /transcripts response not JSON: %s",
                resp.text[:200],
            )
            raise VexaServiceError(f"Respuesta no JSON de VEXA: {e}") from e

        segments = data.get("segments", [])
        if segments:
            logger.info(
                "[VEXA] Obtenidos %s segmentos. Timestamps: inicio=%s, fin=%s",
                len(segments),
                segments[0].get("absolute_start_time", "N/A") if segments else "N/A",
                segments[-1].get("absolute_end_time", "N/A") if segments else "N/A"
            )

        return data

    def get_transcript_segments(
        self,
        native_meeting_id: str,
        platform: str = PLATFORM_TEAMS,
        meeting_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Devuelve solo la lista de segmentos de la transcripcion.
        Misma API que get_transcript pero retorno tipado como lista.
        """
        data = self.get_transcript(
            native_meeting_id=native_meeting_id,
            platform=platform,
            meeting_id=meeting_id,
        )
        segments = data.get("segments") if isinstance(data, dict) else None
        if not isinstance(segments, list):
            return []
        return segments

    def list_meetings(
        self,
        platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Lista todas las reuniones del usuario desde VEXA (GET /meetings).
        
        Args:
            platform: Plataforma opcional para filtrar (ej. "teams", "google_meet").
            
        Returns:
            Lista de diccionarios con informacion de las reuniones.
            
        Raises:
            VexaServiceError: Si la API devuelve error.
        """
        url = f"{self.base_url}/meetings"
        params = {}
        if platform:
            params["platform"] = platform
        
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    url,
                    headers=self._headers(),
                    params=params if params else None,
                )
        except httpx.RequestError as e:
            logger.exception("VEXA GET /meetings (list) request error: %s", e)
            raise VexaServiceError(f"Error de conexion con VEXA: {e}") from e
        
        if resp.status_code >= 400:
            logger.warning(
                "VEXA GET /meetings (list) failed: status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise VexaServiceError(
                f"VEXA API error {resp.status_code}: {resp.text[:500]}"
            )
        
        try:
            data = resp.json()
            # La respuesta puede ser una lista o un objeto con clave "meetings"
            if isinstance(data, dict) and "meetings" in data:
                meetings = data["meetings"]
                if isinstance(meetings, list):
                    return meetings
            elif isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.warning(
                "VEXA GET /meetings (list) response not JSON: %s",
                resp.text[:200],
            )
            raise VexaServiceError(f"Respuesta no JSON de VEXA: {e}") from e

    def get_meeting_status(
        self,
        native_meeting_id: str,
        platform: str = PLATFORM_TEAMS,
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene el estado de una reunion desde VEXA (GET /meetings con filtros).
        
        Args:
            native_meeting_id: ID numerico de la reunion (Teams).
            platform: Plataforma; por defecto "teams".
            
        Returns:
            Diccionario con el estado de la reunion (status, end_time, etc.) o None si no se encuentra.
            
        Raises:
            VexaServiceError: Si la API devuelve error.
        """
        url = f"{self.base_url}/meetings"
        params = {
            "platform": platform,
            "native_meeting_id": native_meeting_id.strip(),
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    url,
                    headers=self._headers(),
                    params=params,
                )
        except httpx.RequestError as e:
            logger.exception("VEXA GET /meetings request error: %s", e)
            raise VexaServiceError(f"Error de conexion con VEXA: {e}") from e
        
        if resp.status_code == 404:
            # Reunion no encontrada
            return None
        
        if resp.status_code >= 400:
            logger.warning(
                "VEXA GET /meetings failed: status=%s body=%s",
                resp.status_code,
                resp.text[:500],
            )
            raise VexaServiceError(
                f"VEXA API error {resp.status_code}: {resp.text[:500]}"
            )
        
        try:
            data = resp.json()
            # La respuesta puede ser una lista o un objeto
            if isinstance(data, list) and data:
                return data[0]  # Devolver la primera reunion encontrada
            elif isinstance(data, dict):
                if "meetings" in data and isinstance(data["meetings"], list) and data["meetings"]:
                    return data["meetings"][0]
                return data
            return None
        except Exception as e:
            logger.warning(
                "VEXA GET /meetings response not JSON: %s",
                resp.text[:200],
            )
            raise VexaServiceError(f"Respuesta no JSON de VEXA: {e}") from e

    def get_latest_meeting_segments(
        self,
        platform: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene los segmentos de la reunion mas reciente desde VEXA.
        
        Args:
            platform: Plataforma opcional para filtrar (ej. "teams", "google_meet").
            
        Returns:
            Diccionario con:
            - "meeting": Informacion de la reunion mas reciente
            - "segments": Lista de segmentos de la transcripcion
            - "total_segments": Numero total de segmentos
            None si no hay reuniones o no hay segmentos.
            
        Raises:
            VexaServiceError: Si la API devuelve error.
        """
        try:
            # Listar todas las reuniones
            meetings = self.list_meetings(platform=platform)
            
            if not meetings:
                logger.info("No se encontraron reuniones en VEXA")
                return None
            
            # Encontrar la mas reciente (por end_time o created_at)
            latest_meeting = None
            latest_time = None
            
            def parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
                """Parsea una fecha ISO desde VEXA."""
                if not dt_str:
                    return None
                try:
                    from datetime import timezone
                    if dt_str.endswith('Z'):
                        dt_str = dt_str[:-1] + '+00:00'
                    return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                except Exception:
                    return None
            
            for meeting in meetings:
                meeting_time = None
                
                # Buscar end_time primero
                end_time_str = meeting.get("end_time") or meeting.get("actual_end_time")
                if end_time_str:
                    meeting_time = parse_dt(end_time_str)
                
                # Si no hay end_time, usar created_at
                if not meeting_time:
                    created_at_str = meeting.get("created_at") or meeting.get("start_time")
                    if created_at_str:
                        meeting_time = parse_dt(created_at_str)
                
                if meeting_time:
                    if latest_time is None or meeting_time > latest_time:
                        latest_time = meeting_time
                        latest_meeting = meeting
            
            if not latest_meeting:
                logger.warning("No se pudo determinar la reunion mas reciente")
                return None
            
            native_id = latest_meeting.get("platform_specific_id")
            platform_value = latest_meeting.get("platform", platform or "teams")
            
            if not native_id:
                logger.warning("La reunion mas reciente no tiene platform_specific_id")
                return None
            
            # Obtener transcripcion
            transcript_data = self.get_transcript(
                native_meeting_id=native_id,
                platform=platform_value,
            )
            
            segments = transcript_data.get("segments", [])
            
            if not segments:
                logger.info(f"La reunion mas reciente (native_id={native_id}) no tiene segmentos aun")
                return {
                    "meeting": latest_meeting,
                    "segments": [],
                    "total_segments": 0,
                }
            
            logger.info(
                "Obtenidos %s segmentos de la reunion mas reciente (native_id=%s)",
                len(segments),
                native_id,
            )
            
            return {
                "meeting": latest_meeting,
                "segments": segments,
                "total_segments": len(segments),
            }
            
        except VexaServiceError:
            raise
        except Exception as e:
            logger.exception("Error obteniendo ultima reunion y segmentos: %s", e)
            raise VexaServiceError(f"Error inesperado: {e}") from e

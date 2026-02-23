"""Servicio para interactuar con Google Calendar API."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
import httpx
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.exceptions import RefreshError

from app.config import settings

logger = logging.getLogger(__name__)


class InvalidTokenError(Exception):
    """Excepción cuando el refresh token es inválido y se necesita reautorización."""
    pass


class GoogleCalendarService:
    """Servicio para gestionar integraciones con Google Calendar."""

    # Scopes necesarios para Google Calendar
    SCOPES = [
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar.events",
    ]

    def __init__(self):
        """Inicializar el servicio de Google Calendar."""
        if not settings.google_client_id or not settings.google_client_secret:  # type: ignore
            logger.warning(
                "Google Calendar no configurado: faltan GOOGLE_CLIENT_ID o GOOGLE_CLIENT_SECRET"
            )

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """
        Generar URL de autorización OAuth para Google Calendar.

        Args:
            state: Estado opcional para mantener contexto en el flujo OAuth

        Returns:
            URL de autorización
        """
        if not settings.google_client_id or not settings.google_client_secret:  # type: ignore
            raise ValueError(
                "Google Calendar no está configurado. Configura GOOGLE_CLIENT_ID y GOOGLE_CLIENT_SECRET."
            )

        client_config = {
            "web": {
                "client_id": settings.google_client_id,  # type: ignore
                "client_secret": settings.google_client_secret,  # type: ignore
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.google_redirect_uri],  # type: ignore
            }
        }

        flow = Flow.from_client_config(
            client_config,
            scopes=self.SCOPES,
            redirect_uri=settings.google_redirect_uri,  # type: ignore
        )

        authorization_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="select_account",  # Forzar selector de cuentas: permite elegir o añadir otra cuenta
            state=state,
        )

        return authorization_url

    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """
        Intercambiar código de autorización por tokens de acceso.

        Args:
            code: Código de autorización recibido en el callback

        Returns:
            Diccionario con tokens y información de expiración
        """
        if not settings.google_client_id or not settings.google_client_secret:  # type: ignore
            raise ValueError(
                "Google Calendar no está configurado. Configura GOOGLE_CLIENT_ID y GOOGLE_CLIENT_SECRET."
            )

        client_config = {
            "web": {
                "client_id": settings.google_client_id,  # type: ignore
                "client_secret": settings.google_client_secret,  # type: ignore
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.google_redirect_uri],  # type: ignore
            }
        }

        flow = Flow.from_client_config(
            client_config,
            scopes=self.SCOPES,
            redirect_uri=settings.google_redirect_uri,  # type: ignore
        )

        flow.fetch_token(code=code)

        credentials = flow.credentials

        token_expires_at = None
        if credentials.expiry:
            # credentials.expiry es un datetime, calcular diferencia
            token_expires_at = credentials.expiry

        return {
            "access_token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_expires_at": token_expires_at,
            "token_uri": credentials.token_uri,  # type: ignore
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "scopes": credentials.scopes,
        }

    def refresh_token(self, refresh_token: str, client_id: str, client_secret: str) -> Dict[str, Any]:
        """
        Renovar token de acceso usando refresh token.

        Args:
            refresh_token: Refresh token almacenado
            client_id: Client ID de Google OAuth
            client_secret: Client Secret de Google OAuth

        Returns:
            Diccionario con nuevo access token y fecha de expiración
        """
        credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
        )

        try:
            credentials.refresh(Request())
        except RefreshError as e:
            error_str = str(e).lower()
            if "invalid_grant" in error_str or "bad request" in error_str:
                logger.error(f"Refresh token inválido o expirado: {e}")
                raise InvalidTokenError(
                    "El refresh token es inválido o ha expirado. Por favor, reconecta tu cuenta de Google Calendar."
                ) from e
            raise

        token_expires_at = None
        if credentials.expiry:
            token_expires_at = credentials.expiry

        return {
            "access_token": credentials.token,
            "token_expires_at": token_expires_at,
        }

    def _refresh_credentials_if_needed(self, credentials: Credentials) -> Credentials:
        """
        Refrescar credenciales si están expiradas.
        
        Args:
            credentials: Objeto Credentials de Google
            
        Returns:
            Credenciales refrescadas (o las mismas si no necesitaban refresh)
            
        Raises:
            InvalidTokenError: Si el refresh token es inválido
        """
        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except RefreshError as e:
                # Verificar si es un error de invalid_grant
                error_str = str(e).lower()
                if "invalid_grant" in error_str or "bad request" in error_str:
                    logger.error(f"Refresh token inválido o expirado: {e}")
                    raise InvalidTokenError(
                        "El refresh token es inválido o ha expirado. Por favor, reconecta tu cuenta de Google Calendar."
                    ) from e
                raise
        return credentials

    def get_credentials_from_tokens(
        self, access_token: str, refresh_token: Optional[str], client_id: str, client_secret: str
    ) -> Credentials:
        """
        Crear objeto Credentials desde tokens almacenados.

        Args:
            access_token: Token de acceso
            refresh_token: Refresh token (opcional)
            client_id: Client ID
            client_secret: Client Secret

        Returns:
            Objeto Credentials de Google
        """
        return Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=self.SCOPES,
        )

    def verify_token_validity(
        self, access_token: str, refresh_token: Optional[str], client_id: str, client_secret: str
    ) -> Dict[str, Any]:
        """
        Verificar si los tokens son válidos intentando renovarlos si es necesario.
        
        Args:
            access_token: Token de acceso actual
            refresh_token: Refresh token (opcional)
            client_id: Client ID
            client_secret: Client Secret
            
        Returns:
            Diccionario con:
            - valid: bool - Si los tokens son válidos
            - needs_reauth: bool - Si necesita reautorización
            - error: str - Mensaje de error si hay problema
        """
        try:
            credentials = self.get_credentials_from_tokens(access_token, refresh_token, client_id, client_secret)
            
            # Intentar refrescar si está expirado
            if credentials.expired:
                if not refresh_token:
                    return {
                        "valid": False,
                        "needs_reauth": True,
                        "error": "Token expirado y no hay refresh_token disponible"
                    }
                
                try:
                    credentials = self._refresh_credentials_if_needed(credentials)
                    return {
                        "valid": True,
                        "needs_reauth": False,
                        "error": None
                    }
                except InvalidTokenError as e:
                    return {
                        "valid": False,
                        "needs_reauth": True,
                        "error": str(e)
                    }
            
            # Si no está expirado, verificar que funcione haciendo una petición simple
            # Usar el método más ligero: obtener lista de calendarios (solo metadata)
            try:
                service = build("calendar", "v3", credentials=credentials)
                service.calendarList().list(maxResults=1).execute()
                return {
                    "valid": True,
                    "needs_reauth": False,
                    "error": None
                }
            except RefreshError as e:
                # Google intentó refrescar automáticamente y falló
                # Este es el error más común cuando el refresh_token está inválido
                error_str = str(e).lower()
                error_repr = repr(e).lower()
                
                is_invalid = (
                    "invalid_grant" in error_str or 
                    "bad request" in error_str or 
                    "expired" in error_str or 
                    "revoked" in error_str or
                    "invalid_grant" in error_repr or
                    "expired" in error_repr or
                    "revoked" in error_repr
                )
                
                if hasattr(e, 'args') and e.args:
                    for arg in e.args:
                        if isinstance(arg, (str, tuple)):
                            arg_str = str(arg).lower()
                            if "invalid_grant" in arg_str or "expired" in arg_str or "revoked" in arg_str:
                                is_invalid = True
                                break
                        elif isinstance(arg, dict):
                            error_val = str(arg.get('error', '')).lower()
                            if "invalid_grant" in error_val or "expired" in error_val or "revoked" in error_val:
                                is_invalid = True
                                break
                
                if is_invalid:
                    logger.error(f"Refresh token inválido o expirado durante verificación: {e}")
                    return {
                        "valid": False,
                        "needs_reauth": True,
                        "error": "El refresh token es inválido o ha expirado. Por favor, reconecta tu cuenta de Google Calendar."
                    }
                logger.error(f"Error refrescando token durante verificación: {e}")
                return {
                    "valid": False,
                    "needs_reauth": True,
                    "error": f"Error refrescando token: {str(e)}"
                }
            except HttpError as e:
                if e.resp.status == 401:  # Unauthorized
                    return {
                        "valid": False,
                        "needs_reauth": True,
                        "error": "Token inválido o revocado"
                    }
                return {
                    "valid": True,
                    "needs_reauth": False,
                    "error": None
                }
                
        except InvalidTokenError as e:
            return {
                "valid": False,
                "needs_reauth": True,
                "error": str(e)
            }
        except RefreshError as e:
            error_str = str(e).lower()
            if "invalid_grant" in error_str or "bad request" in error_str or "expired" in error_str or "revoked" in error_str:
                logger.error(f"Refresh token inválido o expirado durante verificación: {e}")
                return {
                    "valid": False,
                    "needs_reauth": True,
                    "error": "El refresh token es inválido o ha expirado. Por favor, reconecta tu cuenta de Google Calendar."
                }
            logger.error(f"Error refrescando token durante verificación: {e}")
            return {
                "valid": False,
                "needs_reauth": True,
                "error": f"Error refrescando token: {str(e)}"
            }
        except Exception as e:
            error_str = str(e).lower()
            error_repr = repr(e).lower()
            is_token_error = (
                "invalid_grant" in error_str or 
                "expired" in error_str or 
                "revoked" in error_str or
                "invalid_grant" in error_repr or
                "expired" in error_repr or
                "revoked" in error_repr
            )
            if hasattr(e, 'args') and e.args:
                for arg in e.args:
                    arg_str = str(arg).lower() if arg else ""
                    if "invalid_grant" in arg_str or "expired" in arg_str or "revoked" in arg_str:
                        is_token_error = True
                        break
            if is_token_error:
                logger.error(f"Token inválido detectado en excepción genérica: {e}")
                return {
                    "valid": False,
                    "needs_reauth": True,
                    "error": "El token es inválido o ha expirado. Por favor, reconecta tu cuenta de Google Calendar."
                }
            logger.error(f"Error verificando validez de token de Google: {e}")
            return {
                "valid": True,
                "needs_reauth": False,
                "error": None
            }

    def get_events(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        calendar_id: str = "primary",
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Obtener eventos de un calendario.

        Args:
            access_token: Token de acceso
            refresh_token: Refresh token para renovar si es necesario
            client_id: Client ID
            client_secret: Client Secret
            calendar_id: ID del calendario (default: "primary")
            time_min: Fecha/hora mínima para filtrar eventos
            time_max: Fecha/hora máxima para filtrar eventos
            max_results: Número máximo de resultados

        Returns:
            Lista de eventos
        """
        try:
            credentials = self.get_credentials_from_tokens(access_token, refresh_token, client_id, client_secret)

            # Renovar token si es necesario
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())

            service = build("calendar", "v3", credentials=credentials)

            # Preparar parámetros
            # Google Calendar API requiere formato RFC3339 con timezone
            # Si la fecha es naive, asumir UTC
            time_min_str = None
            if time_min:
                if time_min.tzinfo is None:
                    # Fecha naive, añadir UTC
                    from datetime import timezone
                    time_min = time_min.replace(tzinfo=timezone.utc)
                time_min_str = time_min.isoformat()
            
            time_max_str = None
            if time_max:
                if time_max.tzinfo is None:
                    # Fecha naive, añadir UTC
                    from datetime import timezone
                    time_max = time_max.replace(tzinfo=timezone.utc)
                time_max_str = time_max.isoformat()
            
            events_result = service.events().list(
                calendarId=calendar_id,
                timeMin=time_min_str,
                timeMax=time_max_str,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            ).execute()

            events = events_result.get("items", [])

            return events

        except InvalidTokenError:
            raise
        except HttpError as error:
            logger.error(f"Error obteniendo eventos de Google Calendar: {error}")
            raise
        except RefreshError as e:
            error_str = str(e).lower()
            if "invalid_grant" in error_str or "bad request" in error_str:
                logger.error(f"Refresh token inválido o expirado: {e}")
                raise InvalidTokenError(
                    "El refresh token es inválido o ha expirado. Por favor, reconecta tu cuenta de Google Calendar."
                ) from e
            raise
        except Exception as e:
            logger.error(f"Error inesperado en Google Calendar: {e}")
            raise

    @staticmethod
    def _get_user_response_status(event: Dict[str, Any], user_email: str) -> Optional[str]:
        """
        Obtener el estado de respuesta del usuario para un evento (invitado).
        Google Calendar: attendees[].responseStatus = accepted|tentative|declined|needsAction.
        Si el usuario es el organizador, se considera "accepted".

        Args:
            event: Evento de Google Calendar (tiene organizer, attendees).
            user_email: Email del usuario (comparacion case-insensitive).

        Returns:
            "accepted", "tentative", "declined", "needsAction" o None si no es invitado.
        """
        if not user_email:
            return None
        user_email_lower = user_email.strip().lower()
        org = event.get("organizer") or {}
        org_email = (org.get("email") or "").strip().lower()
        if org_email == user_email_lower:
            return "accepted"
        for att in event.get("attendees") or []:
            addr = (att.get("email") or "").strip().lower()
            if addr != user_email_lower:
                continue
            status = (att.get("responseStatus") or "needsAction").strip().lower()
            if status in ("accepted", "tentative", "declined", "needsaction"):
                return "needsAction" if status == "needsaction" else status
            return "needsAction"
        return None

    def create_event(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        calendar_id: str = "primary",
        event_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Crear un evento en el calendario.

        Args:
            access_token: Token de acceso
            refresh_token: Refresh token para renovar si es necesario
            client_id: Client ID
            client_secret: Client Secret
            calendar_id: ID del calendario (default: "primary")
            event_data: Datos del evento (summary, start, end, etc.)

        Returns:
            Evento creado
        """
        try:
            credentials = self.get_credentials_from_tokens(access_token, refresh_token, client_id, client_secret)

            # Renovar token si es necesario
            credentials = self._refresh_credentials_if_needed(credentials)

            service = build("calendar", "v3", credentials=credentials)

            event = service.events().insert(calendarId=calendar_id, body=event_data or {}).execute()

            return event

        except InvalidTokenError:
            raise
        except HttpError as error:
            logger.error(f"Error creando evento en Google Calendar: {error}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado creando evento: {e}")
            raise

    def get_calendars(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
    ) -> List[Dict[str, Any]]:
        """
        Obtener lista de calendarios del usuario.

        Args:
            access_token: Token de acceso
            refresh_token: Refresh token para renovar si es necesario
            client_id: Client ID
            client_secret: Client Secret

        Returns:
            Lista de calendarios
        """
        try:
            credentials = self.get_credentials_from_tokens(access_token, refresh_token, client_id, client_secret)

            # Renovar token si es necesario
            credentials = self._refresh_credentials_if_needed(credentials)

            service = build("calendar", "v3", credentials=credentials)

            calendar_list = service.calendarList().list().execute()
            calendars = calendar_list.get("items", [])

            return calendars

        except InvalidTokenError:
            raise
        except HttpError as error:
            logger.error(f"Error obteniendo calendarios de Google: {error}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado obteniendo calendarios: {e}")
            raise

    def watch_calendar(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        calendar_id: str = "primary",
        webhook_url: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Suscribirse a cambios en el calendario usando Google Calendar Push Notifications.

        Args:
            access_token: Token de acceso
            refresh_token: Refresh token para renovar si es necesario
            client_id: Client ID
            client_secret: Client Secret
            calendar_id: ID del calendario (default: "primary")
            webhook_url: URL pública HTTPS donde Google enviará notificaciones
            user_id: ID del usuario para identificar la suscripción

        Returns:
            Información de la suscripción (channel_id, resource_id, expiration)
        """
        try:
            credentials = self.get_credentials_from_tokens(access_token, refresh_token, client_id, client_secret)

            # Renovar token si es necesario
            credentials = self._refresh_credentials_if_needed(credentials)

            service = build("calendar", "v3", credentials=credentials)

            # Crear suscripción/watch
            # Las suscripciones expiran en ~7 días, se renuevan automáticamente
            # El channel_id debe cumplir el patrón [A-Za-z0-9\-_\+/=]+
            import secrets
            import base64
            
            # Generar un ID único y seguro que cumpla con el formato requerido
            # Usamos base64url encoding que solo contiene caracteres permitidos
            timestamp = int(datetime.utcnow().timestamp())
            random_bytes = secrets.token_bytes(16)
            random_b64 = base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')
            
            if user_id:
                # Limpiar user_id para que solo contenga caracteres permitidos
                clean_user_id = ''.join(c for c in str(user_id) if c.isalnum() or c in '-_+/=')
                channel_id = f"channel-{clean_user_id}-{timestamp}-{random_b64}"
            else:
                channel_id = f"channel-{timestamp}-{random_b64}"
            
            # Asegurarse de que el channel_id solo contiene caracteres permitidos
            channel_id = ''.join(c for c in channel_id if c.isalnum() or c in '-_+/=')
            
            watch_request = {
                "id": channel_id,
                "type": "web_hook",
                "address": webhook_url,
            }

            watch_response = service.events().watch(
                calendarId=calendar_id,
                body=watch_request,
            ).execute()

            return {
                "channel_id": watch_response.get("id"),
                "resource_id": watch_response.get("resourceId"),
                "expiration": watch_response.get("expiration"),
                "resource_uri": watch_response.get("resourceUri"),
            }

        except InvalidTokenError:
            raise
        except HttpError as error:
            logger.error(f"Error creando watch de Google Calendar: {error}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado creando watch: {e}")
            raise

    def stop_watch(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        channel_id: str,
        resource_id: str,
    ) -> None:
        """
        Detener una suscripción/watch de cambios en el calendario.

        Args:
            access_token: Token de acceso
            refresh_token: Refresh token para renovar si es necesario
            client_id: Client ID
            client_secret: Client Secret
            channel_id: ID del canal (de la respuesta de watch)
            resource_id: ID del recurso (de la respuesta de watch)
        """
        try:
            credentials = self.get_credentials_from_tokens(access_token, refresh_token, client_id, client_secret)

            # Renovar token si es necesario
            credentials = self._refresh_credentials_if_needed(credentials)

            service = build("calendar", "v3", credentials=credentials)

            # Detener el watch
            service.channels().stop(
                body={
                    "id": channel_id,
                    "resourceId": resource_id,
                }
            ).execute()

            logger.info(f"Watch detenido: channel_id={channel_id}, resource_id={resource_id}")

        except InvalidTokenError:
            raise
        except HttpError as error:
            logger.error(f"Error deteniendo watch de Google Calendar: {error}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado deteniendo watch: {e}")
            raise


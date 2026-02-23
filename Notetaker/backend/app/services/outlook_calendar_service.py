"""Servicio para interactuar con Microsoft Graph API (Outlook Calendar)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from urllib.parse import urlparse
import httpx
from msal import ConfidentialClientApplication

from app.config import settings

logger = logging.getLogger(__name__)


def _to_graph_datetime(dt: datetime) -> str:
    """Formatear datetime a ISO 8601 para Microsoft Graph."""
    return dt.isoformat()


class OutlookCalendarService:
    """Servicio para gestionar integraciones con Outlook Calendar usando Microsoft Graph API."""

    # Scopes para usuarios normales (solo lectura de calendarios)
    # Nota: offline_access se maneja automáticamente por MSAL cuando se solicita refresh_token
    USER_SCOPES = [
        "Calendars.Read",  # Solo lectura, no escritura
        "User.Read",       # Identificar al usuario
    ]

    # Scopes para el bot (acceso a grabaciones)
    # Nota: offline_access se maneja automáticamente por MSAL cuando se solicita refresh_token
    BOT_SCOPES = [
        "User.Read",                        # Identificar al usuario
        "OnlineMeetingRecording.Read.All",  # Para obtener grabaciones de Teams
    ]

    # Endpoints de Microsoft Graph API
    AUTHORITY_BASE = "https://login.microsoftonline.com"
    GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

    def __init__(self):
        """Inicializar el servicio de Outlook Calendar."""
        if not settings.outlook_client_id or not settings.outlook_client_secret:
            logger.warning(
                "Outlook Calendar no configurado: faltan OUTLOOK_CLIENT_ID o OUTLOOK_CLIENT_SECRET"
            )

    @staticmethod
    def _is_private_ip(ip: str) -> bool:
        """
        Detecta si una IP es privada (LAN/VPN).
        
        Args:
            ip: Dirección IP a verificar
            
        Returns:
            True si es IP privada, False en caso contrario
        """
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            
            first = int(parts[0])
            second = int(parts[1])
            
            # 10.0.0.0/8
            if first == 10:
                return True
            # 172.16.0.0/12
            if first == 172 and 16 <= second <= 31:
                return True
            # 192.168.0.0/16
            if first == 192 and second == 168:
                return True
            
            return False
        except (ValueError, IndexError):
            return False

    @staticmethod
    def _get_redirect_uri(explicit_redirect_uri: Optional[str] = None, is_bot: bool = False) -> str:
        """
        Determina el redirect_uri correcto según el entorno.
        
        Prioridad:
        1. Si hay un redirect_uri explícito configurado (OUTLOOK_REDIRECT_URI/OUTLOOK_BOT_REDIRECT_URI) → usarlo
        2. Si backend_public_url está configurado → construir desde ahí
        3. Si FRONTEND_URL contiene localhost → usar http://localhost:7000 (solo funciona si el usuario está en la misma máquina)
        4. Si FRONTEND_URL contiene IP privada (LAN/VPN) → REQUERIR backend_public_url (ngrok)
        5. Si FRONTEND_URL es ngrok/dominio público → usar la misma base pero con path del backend
        
        Args:
            explicit_redirect_uri: Redirect URI explícito (de settings o parámetro)
            is_bot: Si es True, usa configuración del bot
            
        Returns:
            Redirect URI normalizado
            
        Raises:
            ValueError: Si se detecta IP privada pero no hay backend_public_url configurado
        """
        # 1. Si hay un redirect_uri explícito, usarlo
        if explicit_redirect_uri:
            redirect_uri = explicit_redirect_uri.strip().rstrip('/')
            logger.debug(f"Usando redirect_uri explícito: {redirect_uri}")
            return redirect_uri
        
        # 2. Intentar desde configuración específica (bot o usuario)
        if is_bot:
            explicit_config = settings.outlook_bot_redirect_uri
        else:
            explicit_config = settings.outlook_redirect_uri
        
        if explicit_config:
            redirect_uri = explicit_config.strip().rstrip('/')
            logger.debug(f"Usando redirect_uri de configuración ({'bot' if is_bot else 'usuario'}): {redirect_uri}")
            return redirect_uri
        
        # 3. Construir desde backend_public_url si está configurado
        if settings.backend_public_url:
            base_url = settings.backend_public_url.strip().rstrip('/')
            redirect_uri = f"{base_url}/api/integrations/oauth/callback/outlook"
            logger.info(f"Construyendo redirect_uri desde backend_public_url: {redirect_uri}")
            return redirect_uri
        
        # 4. Detectar desde FRONTEND_URL
        frontend_url = settings.frontend_url.strip().rstrip('/') if settings.frontend_url else ""
        
        # Si FRONTEND_URL contiene localhost o 127.0.0.1, usar localhost para el backend
        # NOTA: Esto solo funciona si el usuario está en la misma máquina que el servidor
        if 'localhost' in frontend_url.lower() or '127.0.0.1' in frontend_url.lower():
            redirect_uri = "http://localhost:7000/api/integrations/oauth/callback/outlook"
            logger.info(f"Detectado localhost en FRONTEND_URL, usando redirect_uri local: {redirect_uri}")
            return redirect_uri
        
        # 5. Detectar IP privada (LAN/VPN) - REQUERIR ngrok
        if frontend_url:
            parsed = urlparse(frontend_url)
            if parsed.netloc:
                # Extraer IP o hostname del netloc (puede incluir puerto)
                host = parsed.netloc.split(':')[0]
                
                # Verificar si es IP privada
                if OutlookCalendarService._is_private_ip(host):
                    # Si es IP privada y NO hay backend_public_url, lanzar error
                    error_msg = (
                        f"Se detecto acceso via LAN/VPN (IP privada: {host}) pero no hay BACKEND_PUBLIC_URL configurado. "
                        f"Cuando accedes al frontend via VPN/LAN, Microsoft OAuth necesita una URL publica (ngrok) "
                        f"para redirigir correctamente al backend. Configura BACKEND_PUBLIC_URL en tu archivo .env "
                        f"con la URL de ngrok (ej: https://xxxx-xx-xx-xx-xx.ngrok-free.app)."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
        
        # 6. Si FRONTEND_URL es ngrok o dominio público, usar la misma base
        if frontend_url:
            parsed = urlparse(frontend_url)
            if parsed.scheme and parsed.netloc:
                redirect_uri = f"{parsed.scheme}://{parsed.netloc}/api/integrations/oauth/callback/outlook"
                logger.info(f"Construyendo redirect_uri desde FRONTEND_URL: {redirect_uri}")
                return redirect_uri
        
        # 7. Fallback a localhost por defecto
        redirect_uri = "http://localhost:7000/api/integrations/oauth/callback/outlook"
        logger.warning(f"No se pudo determinar redirect_uri automáticamente, usando fallback: {redirect_uri}")
        return redirect_uri

    def get_authorization_url(self, state: Optional[str] = None, is_bot: bool = False) -> str:
        """
        Generar URL de autorización OAuth para Outlook Calendar.

        Args:
            state: Estado opcional para mantener contexto en el flujo OAuth
            is_bot: Si es True, usa scopes del bot (grabaciones). Si es False, usa scopes de usuario (calendarios)

        Returns:
            URL de autorización
        """
        # Determinar qué client ID y secret usar según si es bot o usuario
        if is_bot:
            # Para el bot, preferir credenciales específicas del bot
            if settings.outlook_bot_client_id and settings.outlook_bot_client_secret:
                client_id = settings.outlook_bot_client_id
                client_secret = settings.outlook_bot_client_secret
                logger.info("Usando credenciales específicas del BOT (OUTLOOK_BOT_CLIENT_ID)")
            elif settings.outlook_client_id and settings.outlook_client_secret:
                client_id = settings.outlook_client_id
                client_secret = settings.outlook_client_secret
                logger.warning(
                    "⚠️ OUTLOOK_BOT_CLIENT_ID no configurado. Usando credenciales de usuario como fallback. "
                    "Esto hará que se use la aplicación de usuarios en lugar de la del bot."
                )
            else:
                client_id = None
                client_secret = None
            scopes = self.BOT_SCOPES
            redirect_uri = self._get_redirect_uri(is_bot=True)
            logger.info(
                f"Generando URL OAuth para BOT - "
                f"Client ID COMPLETO: {client_id}, "
                f"Scopes: {scopes}, Redirect URI: {redirect_uri}"
            )
        else:
            client_id = settings.outlook_client_id
            client_secret = settings.outlook_client_secret
            scopes = self.USER_SCOPES
            redirect_uri = self._get_redirect_uri(is_bot=False)
            logger.info(
                f"Generando URL OAuth para USUARIO - "
                f"Client ID COMPLETO: {client_id}, "
                f"Scopes: {scopes}, Redirect URI: {redirect_uri}"
            )

        if not client_id or not client_secret:
            raise ValueError(
                "Outlook Calendar no está configurado. Configura OUTLOOK_CLIENT_ID y OUTLOOK_CLIENT_SECRET "
                f"(y OUTLOOK_BOT_CLIENT_ID/OUTLOOK_BOT_CLIENT_SECRET si usas aplicaciones separadas)."
            )

        tenant_id = settings.outlook_tenant_id or settings.graph_tenant_id or "common"
        authority = f"{self.AUTHORITY_BASE}/{tenant_id}"

        app = ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=authority,
        )

        auth_url = app.get_authorization_request_url(
            scopes=scopes,
            redirect_uri=redirect_uri,
            state=state,
            prompt="login",  # Forzar reautenticación: obliga al usuario a ingresar credenciales nuevamente
        )
        
        logger.info(f"URL de autorización generada (primeros 200 caracteres): {auth_url[:200]}")

        return auth_url

    def exchange_code_for_token(self, code: str, is_bot: bool = False, redirect_uri: Optional[str] = None) -> Dict[str, Any]:
        """
        Intercambiar código de autorización por tokens de acceso.

        Args:
            code: Código de autorización recibido en el callback
            is_bot: Si es True, usa credenciales del bot. Si es False, usa credenciales de usuario
            redirect_uri: URI de redirección usado (debe coincidir con el usado en get_authorization_url)

        Returns:
            Diccionario con tokens y información de expiración
        """
        # Determinar qué client ID y secret usar según si es bot o usuario
        if is_bot:
            if settings.outlook_bot_client_id and settings.outlook_bot_client_secret:
                client_id = settings.outlook_bot_client_id
                client_secret = settings.outlook_bot_client_secret
                logger.info("exchange_code_for_token: Usando credenciales específicas del BOT")
            else:
                client_id = settings.outlook_client_id
                client_secret = settings.outlook_client_secret
                logger.warning("exchange_code_for_token: ⚠️ Usando credenciales de usuario como fallback para BOT")
            scopes = self.BOT_SCOPES
            redirect_uri = redirect_uri or self._get_redirect_uri(is_bot=True)
            logger.info(
                f"exchange_code_for_token para BOT - "
                f"Client ID: {client_id}, "
                f"Redirect URI: {redirect_uri}, "
                f"Scopes: {scopes}"
            )
        else:
            client_id = settings.outlook_client_id
            client_secret = settings.outlook_client_secret
            scopes = self.USER_SCOPES
            redirect_uri = redirect_uri or self._get_redirect_uri(is_bot=False)
            logger.info(
                f"exchange_code_for_token para USUARIO - "
                f"Client ID: {client_id}, "
                f"Redirect URI: {redirect_uri}, "
                f"Scopes: {scopes}"
            )

        if not client_id or not client_secret:
            raise ValueError(
                "Outlook Calendar no está configurado. Configura OUTLOOK_CLIENT_ID y OUTLOOK_CLIENT_SECRET "
                f"(y OUTLOOK_BOT_CLIENT_ID/OUTLOOK_BOT_CLIENT_SECRET si usas aplicaciones separadas)."
            )

        tenant_id = settings.outlook_tenant_id or settings.graph_tenant_id or "common"
        authority = f"{self.AUTHORITY_BASE}/{tenant_id}"

        app = ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=authority,
        )

        logger.info(f"Llamando acquire_token_by_authorization_code con redirect_uri={redirect_uri}")
        result = app.acquire_token_by_authorization_code(
            code=code,
            scopes=scopes,
            redirect_uri=redirect_uri,
        )

        if "error" in result:
            error_msg = result.get('error_description', result.get('error'))
            error_code = result.get('error_codes', [])
            logger.error(
                f"Error obteniendo token de Azure AD - "
                f"Error: {result.get('error')}, "
                f"Description: {error_msg}, "
                f"Error codes: {error_code}, "
                f"Client ID usado: {client_id}, "
                f"Redirect URI usado: {redirect_uri}"
            )
            raise ValueError(f"Error obteniendo token: {error_msg}")

        expires_in = result.get("expires_in", 3600)
        token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        return {
            "access_token": result["access_token"],
            "refresh_token": result.get("refresh_token"),
            "token_expires_at": token_expires_at,
            "id_token": result.get("id_token"),
        }

    def refresh_token(self, refresh_token: str, client_id: str, client_secret: str, is_bot: bool = False) -> Dict[str, Any]:
        """
        Renovar token de acceso usando refresh token.

        Args:
            refresh_token: Refresh token almacenado
            client_id: Client ID de Azure AD
            client_secret: Client Secret de Azure AD
            is_bot: Si es True, usa scopes del bot. Si es False, usa scopes de usuario

        Returns:
            Diccionario con nuevo access token y fecha de expiración
        """
        scopes = self.BOT_SCOPES if is_bot else self.USER_SCOPES
        
        tenant_id = settings.outlook_tenant_id or settings.graph_tenant_id or "common"
        authority = f"{self.AUTHORITY_BASE}/{tenant_id}"

        app = ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=authority,
        )

        result = app.acquire_token_by_refresh_token(
            refresh_token=refresh_token,
            scopes=scopes,
        )

        if "error" in result:
            raise ValueError(f"Error renovando token: {result.get('error_description', result.get('error'))}")

        expires_in = result.get("expires_in", 3600)
        token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        return {
            "access_token": result["access_token"],
            "token_expires_at": token_expires_at,
            "refresh_token": result.get("refresh_token", refresh_token),  # Puede devolver nuevo refresh token
        }

    def _get_access_token(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        token_expires_at: Optional[datetime] = None,
        is_bot: bool = False,
    ) -> str:
        """
        Obtener token de acceso válido, renovándolo si es necesario.

        Args:
            access_token: Token de acceso actual
            refresh_token: Refresh token para renovar si es necesario
            client_id: Client ID
            client_secret: Client Secret
            token_expires_at: Fecha de expiración del token
            is_bot: Si es True, usa scopes del bot. Si es False, usa scopes de usuario

        Returns:
            Token de acceso válido
        """
        # Renovar token si está expirado o está por expirar en menos de 5 minutos
        if refresh_token and token_expires_at:
            time_until_expiry = (token_expires_at - datetime.utcnow()).total_seconds()
            if time_until_expiry < 300:  # 5 minutos
                logger.info("Token de Outlook expirado o por expirar, renovando...")
                refreshed = self.refresh_token(refresh_token, client_id, client_secret, is_bot=is_bot)
                return refreshed["access_token"]

        return access_token

    async def verify_token_validity(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        token_expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Verificar si los tokens son válidos intentando renovarlos si es necesario.
        
        Args:
            access_token: Token de acceso actual
            refresh_token: Refresh token (opcional)
            client_id: Client ID
            client_secret: Client Secret
            token_expires_at: Fecha de expiración del token
            
        Returns:
            Diccionario con:
            - valid: bool - Si los tokens son válidos
            - needs_reauth: bool - Si necesita reautorización
            - error: str - Mensaje de error si hay problema
        """
        try:
            # Verificar si el token está expirado o por expirar
            if token_expires_at:
                time_until_expiry = (token_expires_at - datetime.utcnow()).total_seconds()
                if time_until_expiry < 300:  # Menos de 5 minutos
                    if not refresh_token:
                        return {
                            "valid": False,
                            "needs_reauth": True,
                            "error": "Token expirado y no hay refresh_token disponible"
                        }
                    
                    try:
                        refreshed = self.refresh_token(refresh_token, client_id, client_secret, is_bot=False)
                        # Token renovado exitosamente
                        return {
                            "valid": True,
                            "needs_reauth": False,
                            "error": None
                        }
                    except ValueError as e:
                        return {
                            "valid": False,
                            "needs_reauth": True,
                            "error": str(e)
                        }
            
            # Si no está expirado, verificar que funcione haciendo una petición simple
            # Usar el endpoint más ligero: obtener información del usuario
            try:
                url = f"{self.GRAPH_API_ENDPOINT}/me"
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                }
                
                async with httpx.AsyncClient(verify=settings.ssl_verify) as client:
                    response = await client.get(url, headers=headers, timeout=10.0)
                
                if response.status_code == 401:  # Unauthorized
                    return {
                        "valid": False,
                        "needs_reauth": True,
                        "error": "Token inválido o revocado"
                    }
                
                response.raise_for_status()
                return {
                    "valid": True,
                    "needs_reauth": False,
                    "error": None
                }
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    return {
                        "valid": False,
                        "needs_reauth": True,
                        "error": "Token inválido o revocado"
                    }
                # Otro error HTTP, pero el token podría ser válido
                return {
                    "valid": True,
                    "needs_reauth": False,
                    "error": None
                }
                
        except Exception as e:
            logger.error(f"Error verificando validez de token de Outlook: {e}")
            # En caso de error desconocido, asumir que es válido para no bloquear
            return {
                "valid": True,
                "needs_reauth": False,
                "error": None
            }

    async def get_events(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        calendar_id: str = None,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 100,
        token_expires_at: Optional[datetime] = None,
        is_bot: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Obtener eventos de un calendario.

        Si se proporcionan time_min y time_max, usa calendarView para expandir
        recurrentes y devolver cada instancia en el rango. Si no, usa /events
        con $filter (series sin expandir).

        Args:
            access_token: Token de acceso
            refresh_token: Refresh token para renovar si es necesario
            client_id: Client ID
            client_secret: Client Secret
            calendar_id: ID del calendario (None para calendario principal)
            time_min: Fecha/hora mínima para filtrar eventos
            time_max: Fecha/hora máxima para filtrar eventos
            max_results: Número máximo de resultados por página
            token_expires_at: Fecha de expiración del token
            is_bot: Si es True, usa scopes del bot. Si es False, usa scopes de usuario

        Returns:
            Lista de eventos (instancias expandidas si se usó calendarView)
        """
        try:
            valid_token = self._get_access_token(
                access_token, refresh_token, client_id, client_secret, token_expires_at, is_bot=is_bot
            )
            headers = {
                "Authorization": f"Bearer {valid_token}",
                "Content-Type": "application/json",
            }

            use_calendar_view = time_min is not None and time_max is not None

            if use_calendar_view:
                if calendar_id:
                    url = f"{self.GRAPH_API_ENDPOINT}/me/calendars/{calendar_id}/calendarView"
                else:
                    url = f"{self.GRAPH_API_ENDPOINT}/me/calendar/calendarView"
                params = {
                    "startDateTime": _to_graph_datetime(time_min),
                    "endDateTime": _to_graph_datetime(time_max),
                    "$top": min(max_results, 999),
                    "$orderby": "start/dateTime",
                }
            else:
                if calendar_id:
                    url = f"{self.GRAPH_API_ENDPOINT}/me/calendars/{calendar_id}/events"
                else:
                    url = f"{self.GRAPH_API_ENDPOINT}/me/calendar/events"
                params = {"$top": max_results, "$orderby": "start/dateTime"}
                if time_min:
                    params["$filter"] = f"start/dateTime ge '{_to_graph_datetime(time_min)}'"
                if time_max:
                    if "$filter" in params:
                        params["$filter"] += f" and start/dateTime le '{_to_graph_datetime(time_max)}'"
                    else:
                        params["$filter"] = f"start/dateTime le '{_to_graph_datetime(time_max)}'"

            all_events: List[Dict[str, Any]] = []
            next_url: Optional[str] = None

            async with httpx.AsyncClient(verify=settings.ssl_verify) as client:
                while True:
                    if next_url:
                        response = await client.get(next_url, headers=headers, timeout=30.0)
                    else:
                        response = await client.get(url, headers=headers, params=params, timeout=30.0)
                    response.raise_for_status()
                    data = response.json()
                    chunk = data.get("value", [])
                    all_events.extend(chunk)
                    next_url = data.get("@odata.nextLink")
                    if not next_url or (use_calendar_view and len(all_events) >= 5000):
                        break

            if use_calendar_view and all_events:
                logger.info(
                    "Outlook calendarView: %d instancias (recurrentes expandidas) en el rango",
                    len(all_events),
                )
            return all_events

        except httpx.HTTPStatusError as e:
            logger.error(f"Error HTTP obteniendo eventos de Outlook: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado obteniendo eventos de Outlook: {e}")
            raise

    @staticmethod
    def _get_user_response_status(event: Dict[str, Any], user_email: str) -> Optional[str]:
        """
        Obtener el estado de respuesta del usuario para un evento (invitado).
        Microsoft Graph: attendees[].status.response = accepted|tentative|declined|notResponded.
        Si el usuario es el organizador, se considera "accepted".

        Args:
            event: Evento de Microsoft Graph (tiene organizer, attendees).
            user_email: Email del usuario (comparacion case-insensitive).

        Returns:
            "accepted", "tentative", "declined", "notResponded" o None si no es invitado.
        """
        if not user_email:
            return None
        user_email_lower = user_email.strip().lower()
        org = event.get("organizer") or {}
        org_addr = (org.get("emailAddress") or {}).get("address") or ""
        if org_addr.strip().lower() == user_email_lower:
            return "accepted"
        for att in event.get("attendees") or []:
            addr = (att.get("emailAddress") or {}).get("address") or ""
            if addr.strip().lower() != user_email_lower:
                continue
            status = (att.get("status") or {}).get("response")
            if status in ("accepted", "tentative", "declined", "notResponded"):
                return status
            return "notResponded"
        return None

    async def create_event(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        calendar_id: str = None,
        event_data: Optional[Dict[str, Any]] = None,
        token_expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Crear un evento en el calendario.

        Args:
            access_token: Token de acceso
            refresh_token: Refresh token para renovar si es necesario
            client_id: Client ID
            client_secret: Client Secret
            calendar_id: ID del calendario (None para calendario principal)
            event_data: Datos del evento (subject, start, end, etc.)
            token_expires_at: Fecha de expiración del token

        Returns:
            Evento creado
        """
        try:
            # Asegurar que el token sea válido
            valid_token = self._get_access_token(
                access_token, refresh_token, client_id, client_secret, token_expires_at
            )

            # Construir URL del endpoint
            if calendar_id:
                url = f"{self.GRAPH_API_ENDPOINT}/me/calendars/{calendar_id}/events"
            else:
                url = f"{self.GRAPH_API_ENDPOINT}/me/calendar/events"

            headers = {
                "Authorization": f"Bearer {valid_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(verify=settings.ssl_verify) as client:
                response = await client.post(url, headers=headers, json=event_data or {}, timeout=30.0)

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Error HTTP creando evento en Outlook: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado creando evento en Outlook: {e}")
            raise

    async def get_calendars(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        token_expires_at: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Obtener lista de calendarios del usuario.

        Args:
            access_token: Token de acceso
            refresh_token: Refresh token para renovar si es necesario
            client_id: Client ID
            client_secret: Client Secret
            token_expires_at: Fecha de expiración del token

        Returns:
            Lista de calendarios
        """
        try:
            # Asegurar que el token sea válido
            valid_token = self._get_access_token(
                access_token, refresh_token, client_id, client_secret, token_expires_at
            )

            url = f"{self.GRAPH_API_ENDPOINT}/me/calendars"

            headers = {
                "Authorization": f"Bearer {valid_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(verify=settings.ssl_verify) as client:
                response = await client.get(url, headers=headers, timeout=30.0)

            response.raise_for_status()
            data = response.json()

            return data.get("value", [])

        except httpx.HTTPStatusError as e:
            logger.error(f"Error HTTP obteniendo calendarios de Outlook: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado obteniendo calendarios de Outlook: {e}")
            raise

    async def create_subscription(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        notification_url: str,
        token_expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Crear suscripcion a cambios en me/calendar/events (Microsoft Graph).
        Graph envia POST al notification_url cuando se crean/actualizan/eliminan eventos.
        La suscripcion expira en 2 dias; renovar antes con renew_subscription.

        Returns:
            {"id": "...", "expirationDateTime": "..."}
        """
        from datetime import timezone

        valid_token = self._get_access_token(
            access_token, refresh_token, client_id, client_secret, token_expires_at, is_bot=False
        )
        expiration = datetime.now(timezone.utc) + timedelta(days=2)
        expiration_str = expiration.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        body = {
            "changeType": "created,updated,deleted",
            "notificationUrl": notification_url,
            "resource": "me/calendar/events",
            "expirationDateTime": expiration_str,
        }

        url = f"{self.GRAPH_API_ENDPOINT}/subscriptions"
        headers = {
            "Authorization": f"Bearer {valid_token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(verify=settings.ssl_verify) as client:
            response = await client.post(url, headers=headers, json=body, timeout=15.0)
        response.raise_for_status()
        data = response.json()
        logger.info(
            "Suscripcion Outlook creada: id=%s, expiration=%s",
            data.get("id"),
            data.get("expirationDateTime"),
        )
        return {
            "id": data["id"],
            "expirationDateTime": data["expirationDateTime"],
        }

    async def delete_subscription(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        subscription_id: str,
        token_expires_at: Optional[datetime] = None,
    ) -> None:
        """Eliminar suscripcion de Graph (DELETE /subscriptions/{id})."""
        valid_token = self._get_access_token(
            access_token, refresh_token, client_id, client_secret, token_expires_at, is_bot=False
        )
        url = f"{self.GRAPH_API_ENDPOINT}/subscriptions/{subscription_id}"
        headers = {"Authorization": f"Bearer {valid_token}"}

        async with httpx.AsyncClient(verify=settings.ssl_verify) as client:
            response = await client.delete(url, headers=headers, timeout=10.0)
        if response.status_code == 404:
            logger.warning("Suscripcion Outlook ya no existe: id=%s", subscription_id)
            return
        response.raise_for_status()
        logger.info("Suscripcion Outlook eliminada: id=%s", subscription_id)

    async def renew_subscription(
        self,
        access_token: str,
        refresh_token: Optional[str],
        client_id: str,
        client_secret: str,
        subscription_id: str,
        token_expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Renovar suscripcion (PATCH) con nueva expirationDateTime (2 dias)."""
        from datetime import timezone

        valid_token = self._get_access_token(
            access_token, refresh_token, client_id, client_secret, token_expires_at, is_bot=False
        )
        expiration = datetime.now(timezone.utc) + timedelta(days=2)
        expiration_str = expiration.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        url = f"{self.GRAPH_API_ENDPOINT}/subscriptions/{subscription_id}"
        headers = {
            "Authorization": f"Bearer {valid_token}",
            "Content-Type": "application/json",
        }
        body = {"expirationDateTime": expiration_str}

        async with httpx.AsyncClient(verify=settings.ssl_verify) as client:
            response = await client.patch(url, headers=headers, json=body, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        logger.info(
            "Suscripcion Outlook renovada: id=%s, expiration=%s",
            subscription_id,
            data.get("expirationDateTime"),
        )
        return {"expirationDateTime": data["expirationDateTime"]}

"""OAuth service for Microsoft authentication (User Consent Only)."""
import redis
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from msal import ConfidentialClientApplication
import jwt
from app.config import settings

logger = logging.getLogger(__name__)


class OAuthService:
    """
    Maneja autenticación OAuth2 con Delegated Permissions.
    
    Permite que usuarios de cualquier tenant autoricen la aplicación
    para acceder a sus calendarios y unirse a reuniones.
    """
    
    def __init__(self):
        """Inicializar OAuthService."""
        # Configuración MSAL para multi-tenant
        tenant_id = getattr(settings, 'microsoft_tenant_id', 'common')
        if tenant_id and tenant_id != 'common':
            tenant_id = 'common'  # Forzar multi-tenant
        
        self.msal_app = ConfidentialClientApplication(
            settings.microsoft_client_id,
            authority=f"https://login.microsoftonline.com/{tenant_id}",
            client_credential=settings.microsoft_client_secret,
        )
        
        # Redis para almacenar tokens
        redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379/0')
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("✅ Redis conectado para almacenamiento de tokens")
        except Exception as e:
            logger.warning(f"⚠️ Redis no disponible ({e}), usando almacenamiento en memoria")
            self.redis_client = None
            self._memory_storage = {}
        
        # Scopes para delegated permissions (User Consent Only)
        # User.Read y Calendars.Read normalmente no requieren Admin Consent
        self.scopes = [
            "User.Read",              # Identificar al usuario
            "Calendars.Read",         # Leer calendarios del usuario
            "Calendars.ReadWrite",    # Leer y escribir calendarios (para sincronizar)
            # Offline_access es manejado automáticamente por MSAL
        ]
    
    def get_auth_url(self, redirect_uri: Optional[str] = None) -> str:
        """
        Genera URL de autorización OAuth2.
        
        Args:
            redirect_uri: URI de redirección (opcional)
            
        Returns:
            URL de autorización de Azure AD
        """
        if not redirect_uri:
            redirect_uri = settings.microsoft_redirect_uri
        
        auth_url = self.msal_app.get_authorization_request_url(
            scopes=self.scopes,
            redirect_uri=redirect_uri,
            prompt="select_account"  # Permitir seleccionar cuenta sin forzar consentimiento
        )
        
        logger.info(f"URL de autorización generada: {redirect_uri}")
        return auth_url
    
    async def exchange_code_for_token(self, code: str, redirect_uri: Optional[str] = None) -> Dict[str, Any]:
        """
        Intercambia authorization code por tokens.
        
        Args:
            code: Authorization code de OAuth2 callback
            redirect_uri: URI de redirección
            
        Returns:
            Dict con user_id, tenant_id, success, o error
        """
        try:
            if not redirect_uri:
                redirect_uri = settings.microsoft_redirect_uri
            
            token_response = self.msal_app.acquire_token_by_authorization_code(
                code=code,
                scopes=self.scopes,
                redirect_uri=redirect_uri
            )
            
            if "error" in token_response:
                error_msg = token_response.get("error_description", token_response.get("error"))
                logger.error(f"Error obteniendo token: {error_msg}")
                return {"error": error_msg}
            
            # Decodificar id_token para obtener información del usuario
            id_token_claims = self._decode_id_token(token_response.get("id_token", ""))
            
            user_id = id_token_claims.get("oid")  # Object ID del usuario
            tenant_id = id_token_claims.get("tid")  # Tenant ID
            user_email = id_token_claims.get("email") or id_token_claims.get("preferred_username", "")
            display_name = id_token_claims.get("name", "")
            
            if not user_id or not tenant_id:
                logger.error("No se pudo obtener user_id o tenant_id del token")
                return {"error": "Invalid token response"}
            
            # Preparar datos del token para almacenar
            expires_in = token_response.get("expires_in", 3600)
            token_data = {
                "access_token": token_response["access_token"],
                "refresh_token": token_response.get("refresh_token"),
                "tenant_id": tenant_id,
                "expires_at": (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
                "user_id": user_id,
                "user_email": user_email,
                "display_name": display_name
            }
            
            # Guardar tokens
            self._save_token(user_id, token_data)
            
            logger.info(f"✅ Token obtenido para usuario {user_id} del tenant {tenant_id}")
            
            return {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "user_email": user_email,
                "display_name": display_name,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Error en exchange_code_for_token: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def get_valid_token(self, user_id: str) -> str:
        """
        Obtiene un access token válido para el usuario.
        Si está expirado, lo refreshea automáticamente.
        
        Args:
            user_id: Object ID del usuario
            
        Returns:
            Access token válido
            
        Raises:
            ValueError: Si no se encuentra token o no se puede refreshear
        """
        token_data = self._get_token(user_id)
        
        if not token_data:
            raise ValueError(f"No se encontró token para el usuario {user_id}. Usuario debe autorizar primero.")
        
        expires_at = datetime.fromisoformat(token_data["expires_at"])
        
        # Si expira en menos de 5 minutos, refreshear
        if datetime.now() > expires_at - timedelta(minutes=5):
            logger.info(f"Token expirando para usuario {user_id}, refrescando...")
            token_data = await self._refresh_token(user_id, token_data.get("refresh_token"))
        
        return token_data["access_token"]
    
    async def _refresh_token(self, user_id: str, refresh_token: Optional[str]) -> Dict[str, Any]:
        """
        Refreshea un token expirado.
        
        Args:
            user_id: Object ID del usuario
            refresh_token: Refresh token del usuario
            
        Returns:
            Nuevos datos del token
        """
        if not refresh_token:
            raise ValueError("No hay refresh_token disponible. Usuario debe autorizar de nuevo.")
        
        try:
            token_response = self.msal_app.acquire_token_by_refresh_token(
                refresh_token=refresh_token,
                scopes=self.scopes
            )
            
            if "error" in token_response:
                error_msg = token_response.get("error_description", token_response.get("error"))
                logger.error(f"Error refrescando token: {error_msg}")
                raise ValueError(f"Token refresh failed: {error_msg}")
            
            # Decodificar id_token
            id_token_claims = self._decode_id_token(token_response.get("id_token", ""))
            
            expires_in = token_response.get("expires_in", 3600)
            token_data = {
                "access_token": token_response["access_token"],
                "refresh_token": token_response.get("refresh_token") or refresh_token,
                "tenant_id": id_token_claims.get("tid"),
                "expires_at": (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
                "user_id": user_id
            }
            
            # Actualizar token guardado
            self._save_token(user_id, token_data)
            
            logger.info(f"✅ Token refrescado para usuario {user_id}")
            return token_data
        
        except Exception as e:
            logger.error(f"Error refrescando token: {e}", exc_info=True)
            raise
    
    def _save_token(self, user_id: str, token_data: Dict[str, Any]):
        """Guarda token en Redis o memoria."""
        if self.redis_client:
            try:
                # Guardar con expiración de 90 días (refresh tokens duran ~90 días)
                self.redis_client.setex(
                    f"tokens:user:{user_id}",
                    90 * 24 * 3600,  # 90 días
                    json.dumps(token_data)
                )
            except Exception as e:
                logger.error(f"Error guardando token en Redis: {e}")
                self._memory_storage[f"tokens:user:{user_id}"] = token_data
        else:
            self._memory_storage[f"tokens:user:{user_id}"] = token_data
    
    def _get_token(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene token desde Redis o memoria."""
        if self.redis_client:
            try:
                token_json = self.redis_client.get(f"tokens:user:{user_id}")
                if token_json:
                    return json.loads(token_json)
            except Exception as e:
                logger.error(f"Error obteniendo token de Redis: {e}")
        
        return self._memory_storage.get(f"tokens:user:{user_id}")
    
    def _decode_id_token(self, id_token: str) -> Dict[str, Any]:
        """
        Decodifica ID token sin verificar firma (ya validado por MSAL).
        
        Args:
            id_token: ID token JWT
            
        Returns:
            Claims del token
        """
        try:
            decoded = jwt.decode(id_token, options={"verify_signature": False})
            return decoded
        except Exception as e:
            logger.warning(f"Error decodificando id_token: {e}")
            return {}
    
    def get_user_tenant(self, user_id: str) -> Optional[str]:
        """
        Obtiene el tenant_id de un usuario autorizado.
        
        Args:
            user_id: Object ID del usuario
            
        Returns:
            Tenant ID o None si no se encuentra
        """
        token_data = self._get_token(user_id)
        if token_data:
            return token_data.get("tenant_id")
        return None


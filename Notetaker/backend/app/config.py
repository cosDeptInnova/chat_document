"""Configuration management for the application."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
import os
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse, quote_plus


def load_env_file_utf8(env_file: str = ".env"):
    """Carga el archivo .env manualmente asegurando codificación UTF-8."""
    # Buscar .env en el directorio backend/
    env_path = Path(__file__).parent.parent / env_file
    if not env_path.exists():
        # Intentar también en el directorio actual de trabajo
        env_path = Path.cwd() / env_file
        if not env_path.exists():
            # Intentar también en el directorio actual (relativo)
            env_path = Path(env_file)
            if not env_path.exists():
                return
    
    content = None
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1', 'windows-1252']
    
    for encoding in encodings:
        try:
            with open(env_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            # Si se leyó exitosamente, intentar guardarlo en UTF-8 si no lo estaba
            if encoding != 'utf-8' and encoding != 'utf-8-sig':
                try:
                    # Guardar en UTF-8 para futuras lecturas
                    with open(env_path, 'w', encoding='utf-8', errors='replace') as f:
                        f.write(content)
                except Exception as e:
                    # Si no se puede escribir, continuar con el contenido leído
                    pass
            break
        except Exception as e:
            continue
    
    if content is None:
        raise ValueError(f"No se pudo leer {env_file} con ninguna codificación conocida")
    
    # Cargar variables al entorno
    loaded_count = 0
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Remover comentarios al final de la línea (solo si no está entre comillas)
            if '#' in value and not (value.startswith('"') or value.startswith("'")):
                # Dividir por # pero solo tomar la primera parte
                parts = value.split('#', 1)
                if len(parts) > 1:
                    # Verificar que el # no esté dentro de comillas
                    before_hash = parts[0]
                    if before_hash.count('"') % 2 == 0 and before_hash.count("'") % 2 == 0:
                        value = parts[0].strip()
            
            # Remover comillas si las tiene
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            
            # Asignar siempre desde .env para que API y Celery usen los mismos valores
            # (evita que un AI_SERVICE_URL=http en el entorno anule el https del .env)
            if key:
                # Si es DATABASE_URL, asegurar que la contraseña esté codificada correctamente
                if key == "DATABASE_URL" and "@" in value:
                    # Parsear y codificar correctamente la URL
                    try:
                        parsed = urlparse(value)
                        if parsed.password:
                            # Si la contraseña no está codificada, codificarla
                            encoded_password = quote_plus(parsed.password)
                            if parsed.password != encoded_password:
                                # Reconstruir la URL con la contraseña codificada
                                netloc = f"{parsed.username}:{encoded_password}@{parsed.hostname}"
                                if parsed.port:
                                    netloc += f":{parsed.port}"
                                value = urlunparse((
                                    parsed.scheme,
                                    netloc,
                                    parsed.path,
                                    parsed.params,
                                    parsed.query,
                                    parsed.fragment
                                ))
                    except Exception:
                        # Si falla el parsing, intentar codificar manualmente
                        # Buscar el patrón user:password@host
                        pattern = r'(postgresql://[^:]+):([^@]+)@'
                        match = re.search(pattern, value)
                        if match:
                            prefix = match.group(1)
                            password = match.group(2)
                            encoded_password = quote_plus(password)
                            value = value.replace(f":{password}@", f":{encoded_password}@")
                
                os.environ[key] = value
                loaded_count += 1


# Cargar .env ANTES de crear Settings
load_env_file_utf8()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OAuth Microsoft (OPCIONAL - Solo si quieres leer calendarios de usuarios automáticamente)
    # Si solo quieres unirse a reuniones con email/password, NO necesitas esto
    microsoft_client_id: Optional[str] = None
    microsoft_client_secret: Optional[str] = None
    microsoft_redirect_uri: str = "http://localhost:7000/api/auth/callback"
    microsoft_tenant_id: str = "common"  # "common" para multi-tenant
    
    # Deepgram
    deepgram_api_key: str
    
    # Database
    database_url: str
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Tu IA
    ai_service_url: str
    ai_service_api_key: Optional[str] = None
    
    # Storage
    storage_type: str = "local"  # local or s3
    audio_storage_path: str = "./storage/audio"
    video_storage_path: str = "./storage/video"
    
    # S3/MinIO (opcional)
    s3_endpoint: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_bucket_name: str = "cosmos-notetaker-audio"
    s3_use_ssl: bool = False
    
    # Application
    secret_key: str
    jwt_secret_key: Optional[str] = Field(
        None,
        description="Clave secreta para firmar JWT tokens. Si no se proporciona, se usa secret_key."
    )
    debug: bool = False
    frontend_url: str = "http://localhost:3000"
    
    # URL pública del backend para webhooks (necesaria para push notifications de Google Calendar)
    # En desarrollo: usa ngrok o similar
    # En producción: tu dominio HTTPS
    backend_public_url: Optional[str] = Field(
        None,
        description="URL pública HTTPS del backend para recibir webhooks (ej: https://tu-dominio.com o https://xxx.ngrok.io)"
    )
    
    # Bot name (configurable por usuario, default). En .env: DEFAULT_BOT_NAME=Tu Nombre Bot
    default_bot_name: str = "Notetaker"
    
    # Bot avatar/image (URL pública HTTPS de la imagen que se mostrará en la reunión)
    # Recomendado: 1280x720px (16:9) o 1080x1080px, formato PNG/JPG, < 5MB
    bot_avatar_image_url: Optional[str] = Field(
        None,
        description="URL pública HTTPS de la imagen/avatar del bot para mostrar en reuniones"
    )
    
    # Administración / roles sencillos
    admin_email_domain: Optional[str] = Field(
        None,
        description="Dominio de email que se considerará admin (ej: 'miempresa.com')"
    )
    admin_emails: Optional[str] = Field(
        None,
        description="Lista de emails admin separados por coma (prioridad sobre dominio)"
    )

    # VEXA (bot Teams y transcripciones en red local)
    vexa_api_base_url: Optional[str] = Field(
        None,
        description="URL base de la User API de VEXA (ej: http://172.29.14.10:8056)",
    )
    vexa_api_key: Optional[str] = Field(
        None,
        description="API key para X-API-Key (User API de VEXA). En self-hosted puede ser el token de usuario creado via Admin API.",
    )
    
    # Microsoft Graph API para envío de emails
    graph_tenant_id: Optional[str] = Field(
        None,
        description="Tenant ID de Azure AD para Microsoft Graph API"
    )
    graph_client_id: Optional[str] = Field(
        None,
        description="Client ID de la aplicación Azure AD registrada"
    )
    graph_client_secret: Optional[str] = Field(
        None,
        description="Client Secret de la aplicación Azure AD"
    )
    graph_user_email: Optional[str] = Field(
        None,
        description="Email de la cuenta que enviará los correos (ej: rpa@cosgs.com)"
    )
    
    # Password por defecto para admins nuevos
    admin_default_password: Optional[str] = Field(
        None,
        description="Contraseña por defecto para usuarios admin nuevos"
    )
    
    # Google Calendar OAuth
    google_client_id: Optional[str] = Field(
        None,
        description="Client ID de Google OAuth para Google Calendar"
    )
    google_client_secret: Optional[str] = Field(
        None,
        description="Client Secret de Google OAuth para Google Calendar"
    )
    google_redirect_uri: str = Field(
        "http://localhost:7000/api/integrations/oauth/callback/google",
        description="Redirect URI para OAuth de Google Calendar"
    )
    
    # Outlook Calendar OAuth (usar Graph API) - Para usuarios normales
    outlook_client_id: Optional[str] = Field(
        None,
        description="Client ID de Azure AD para Outlook Calendar (para usuarios normales - solo lectura de calendarios)"
    )
    outlook_client_secret: Optional[str] = Field(
        None,
        description="Client Secret de Azure AD para Outlook Calendar (para usuarios normales)"
    )
    outlook_redirect_uri: str = Field(
        "http://localhost:7000/api/integrations/oauth/callback/outlook",
        description="Redirect URI para OAuth de Outlook Calendar (usuarios normales)"
    )
    outlook_tenant_id: Optional[str] = Field(
        None,
        description="Tenant ID de Azure AD para Outlook Calendar (puede ser el mismo que graph_tenant_id)"
    )

    # Cosmos para diarización y transcripción
    cosmos_audio_endpoint: Optional[str] = Field(
        None,
        description="URL del endpoint de Cosmos para procesar audio (diarización y transcripción)"
    )
    cosmos_api_key: Optional[str] = Field(
        None,
        description="API key de Cosmos (si es necesario para autenticación)"
    )
    
    # Tiempo de espera antes de obtener transcripción de VEXA (en minutos)
    transcript_wait_minutes: int = Field(
        5,
        description="Minutos de espera después de que el bot abandone antes de obtener la transcripción de VEXA"
    )
    
    # Configuración de captura de audio
    audio_capture_format: str = Field(
        "mp4",
        description="Formato de audio para captura (mp3, wav, mp4, etc.)"
    )
    audio_capture_sample_rate: int = Field(
        16000,
        description="Sample rate para captura de audio (16000, 44100, etc.)"
    )
    
    # Configuración de desarrollo - Bypass SSL (SOLO PARA DESARROLLO)
    ssl_verify: bool = Field(
        True,
        description="Verificar certificados SSL. Establecer a False solo en desarrollo para bypass de certificados interceptados (ej: Fortinet)"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


settings = Settings()

# Log temporal para verificar que .env se carga correctamente (quitar tras diagnostico)
def _log_config_urls():
    import sys
    msg = (
        "[CONFIG DEBUG] ai_service_url=%s | vexa_api_base_url=%s | redis_url=%s"
        % (
            getattr(settings, "ai_service_url", "N/A"),
            getattr(settings, "vexa_api_base_url", "N/A"),
            getattr(settings, "redis_url", "N/A"),
        )
    )
    print(msg, file=sys.stderr)
    sys.stderr.flush()


_log_config_urls()

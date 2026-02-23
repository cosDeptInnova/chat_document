"""Endpoints para gestionar integraciones de calendario (Google Calendar y Outlook)."""
import asyncio
import json
import logging
import secrets
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import RedirectResponse, PlainTextResponse, Response
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from starlette.requests import Request, ClientDisconnect

from app.database import get_db, SessionLocal
from app.models.user import User
from app.services.google_calendar_service import GoogleCalendarService, InvalidTokenError
from app.services.outlook_calendar_service import OutlookCalendarService
from app.services.calendar_sync_service import CalendarSyncService
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/integrations", tags=["integrations"])

# Instancias de servicios
google_service = GoogleCalendarService()
outlook_service = OutlookCalendarService()
sync_service = CalendarSyncService()

# Debounce por user_id para webhook Outlook: evita N syncs cuando Graph envia varias notificaciones por un solo cambio
_OUTLOOK_WEBHOOK_DEBOUNCE: Dict[str, asyncio.Task] = {}
_OUTLOOK_DEBOUNCE_SECONDS = 5

# Cooldown tras sync completado: evita bucle de syncs cada pocos segundos
_OUTLOOK_LAST_SYNC_AT: Dict[str, datetime] = {}
COOLDOWN_SECONDS = 30  # No programar nuevo sync hasta que pasen 30 segundos desde el ultimo completado (reducido de 90s para mejorar latencia)

# Cola de webhooks descartados por cooldown: guarda webhooks que llegaron durante cooldown para procesarlos después
_OUTLOOK_PENDING_WEBHOOKS: Dict[str, List[datetime]] = {}  # user_id -> lista de timestamps de webhooks pendientes


def get_user_by_email(email: str, db: Session) -> User:
    """Obtener usuario por email."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return user


def get_frontend_url(request: Request) -> str:
    """
    Determina la URL del frontend a partir del request (headers).
    Asi, tras el callback de Azure/Google se redirige a la misma base desde la que
    se inicio el OAuth (produccion o desarrollo), no siempre a FRONTEND_URL del .env.

    Prioridad: X-Forwarded-Host, Host, Referer (no OAuth), Origin (no OAuth),
    URL del request, fallback settings.frontend_url.
    """
    from urllib.parse import urlparse

    oauth_domains = (
        "microsoftonline.com",
        "login.microsoft.com",
        "accounts.google.com",
        "oauth2.googleapis.com",
        "google.com",
    )

    def is_oauth_domain(url: str) -> bool:
        if not url:
            return False
        return any(d in url.lower() for d in oauth_domains)

    forwarded_host = request.headers.get("x-forwarded-host")
    forwarded_proto = request.headers.get("x-forwarded-proto", "https")
    host = request.headers.get("host")

    if forwarded_host:
        if "localhost" in forwarded_host.lower() or forwarded_host.startswith(("127.", "192.168.", "172.")):
            base = f"http://{forwarded_host}"
            logger.info("[get_frontend_url] X-Forwarded-Host (desarrollo): %s", base)
            return base
        if not is_oauth_domain(forwarded_host):
            base = f"{forwarded_proto}://{forwarded_host}"
            logger.info("[get_frontend_url] X-Forwarded-Host: %s", base)
            return base

    if host:
        if "localhost" in host.lower() or host.startswith(("127.", "192.168.", "172.")):
            logger.info("[get_frontend_url] Host (desarrollo): %s", settings.frontend_url)
            return settings.frontend_url
        if not is_oauth_domain(host):
            base = f"https://{host}" if forwarded_proto == "https" else f"http://{host}"
            logger.info("[get_frontend_url] Host: %s", base)
            return base

    referer = request.headers.get("referer")
    if referer and not is_oauth_domain(referer):
        try:
            parsed = urlparse(referer)
            if parsed.scheme and parsed.netloc and not is_oauth_domain(parsed.netloc):
                base = f"{parsed.scheme}://{parsed.netloc}"
                logger.info("[get_frontend_url] Referer: %s", base)
                return base
        except Exception:
            pass

    origin = request.headers.get("origin")
    if origin and not is_oauth_domain(origin):
        try:
            parsed = urlparse(origin)
            if parsed.scheme and parsed.netloc:
                base = f"{parsed.scheme}://{parsed.netloc}"
                logger.info("[get_frontend_url] Origin: %s", base)
                return base
        except Exception:
            pass

    try:
        parsed = urlparse(str(request.url))
        if parsed.hostname and not is_oauth_domain(parsed.hostname):
            if "localhost" in (parsed.hostname or "").lower():
                return settings.frontend_url
            base = f"{parsed.scheme}://{parsed.netloc}"
            logger.info("[get_frontend_url] Request URL: %s", base)
            return base
    except Exception:
        pass

    logger.info("[get_frontend_url] Fallback: %s", settings.frontend_url)
    return settings.frontend_url


class IntegrationStatusResponse(BaseModel):
    """Respuesta con estado de integraciones del usuario."""
    google_calendar: Dict[str, Any]
    outlook_calendar: Dict[str, Any]


@router.get("/status")
async def get_integration_status(
    user_email: str = Query(..., description="Email del usuario"),
    db: Session = Depends(get_db),
):
    """
    Obtener estado de todas las integraciones del usuario.
    Verifica realmente la validez de los tokens antes de mostrar "conectado".
    """
    user = get_user_by_email(user_email, db)

    # Inicializar respuesta
    response = {
        "google_calendar": {"connected": False, "connected_at": None},
        "outlook_calendar": {"connected": False, "connected_at": None},
    }

    # Verificar Google Calendar
    if user.settings:  # type: ignore
        google_data = user.settings.get("google_calendar")  # type: ignore
        if google_data and google_data.get("access_token"):  # type: ignore
            access_token = google_data.get("access_token")  # type: ignore
            refresh_token = google_data.get("refresh_token")  # type: ignore
            client_id = settings.google_client_id  # type: ignore
            client_secret = settings.google_client_secret  # type: ignore
            
            # Verificar realmente si el token es válido
            if access_token and client_id and client_secret:
                try:
                    token_status = google_service.verify_token_validity(
                        access_token=access_token,
                        refresh_token=refresh_token,
                        client_id=client_id,
                        client_secret=client_secret
                    )
                    
                    # Si el token es válido pero se renovó, actualizar en BD
                    if token_status["valid"]:
                        # Verificar si se renovó el token (comparando con el almacenado)
                        try:
                            credentials = google_service.get_credentials_from_tokens(
                                access_token, refresh_token, client_id, client_secret
                            )
                            credentials = google_service._refresh_credentials_if_needed(credentials)
                            
                            # Si el token cambió, actualizar en BD
                            if credentials.token != access_token:
                                logger.info(f"Token de Google renovado durante verificación para {user_email}")
                                google_data["access_token"] = credentials.token  # type: ignore
                                if credentials.expiry:
                                    google_data["token_expires_at"] = credentials.expiry.isoformat()  # type: ignore
                                if credentials.refresh_token and credentials.refresh_token != refresh_token:
                                    google_data["refresh_token"] = credentials.refresh_token  # type: ignore
                                google_data["needs_reauth"] = False  # type: ignore
                                flag_modified(user, "settings")
                                db.commit()
                        except InvalidTokenError:
                            # Si falla al refrescar, marcar como necesita reauth
                            token_status["valid"] = False
                            token_status["needs_reauth"] = True
                    
                    # Actualizar marca needs_reauth en BD si el token es inválido
                    if not token_status["valid"]:
                        google_data["needs_reauth"] = True  # type: ignore
                        flag_modified(user, "settings")
                        db.commit()
                    elif token_status["valid"] and google_data.get("needs_reauth"):  # type: ignore
                        # Si ahora es válido pero antes estaba marcado como needs_reauth, limpiar
                        google_data["needs_reauth"] = False  # type: ignore
                        flag_modified(user, "settings")
                        db.commit()
                    
                    response["google_calendar"] = {
                        "connected": token_status["valid"],
                        "connected_at": google_data.get("connected_at"),  # type: ignore
                        "calendar_id": google_data.get("calendar_id", "primary"),  # type: ignore
                        "push_notifications_active": bool(google_data.get("watch_channel_id")),  # type: ignore
                        "needs_reauth": token_status["needs_reauth"],
                        "error": token_status.get("error"),
                    }
                except Exception as e:
                    logger.error(f"Error verificando token de Google para {user_email}: {e}")
                    # Verificar si el error indica token inválido
                    error_str = str(e).lower()
                    is_token_invalid = (
                        "invalid_grant" in error_str or 
                        "expired" in error_str or 
                        "revoked" in error_str or
                        "invalid" in error_str
                    )
                    
                    # Actualizar marca needs_reauth en BD
                    if is_token_invalid:
                        google_data["needs_reauth"] = True  # type: ignore
                        flag_modified(user, "settings")
                        db.commit()
                    
                    # En caso de error, marcar como desconectado
                    response["google_calendar"] = {
                        "connected": False,
                        "connected_at": google_data.get("connected_at"),  # type: ignore
                        "calendar_id": google_data.get("calendar_id", "primary"),  # type: ignore
                        "push_notifications_active": bool(google_data.get("watch_channel_id")),  # type: ignore
                        "needs_reauth": True,
                        "error": f"Error verificando token: {str(e)}",
                    }
            else:
                # No hay tokens o configuración faltante
                response["google_calendar"] = {
                    "connected": False,
                    "connected_at": google_data.get("connected_at"),  # type: ignore
                    "needs_reauth": True,
                    "error": "Configuración incompleta",
                }

        # Verificar Outlook Calendar
        outlook_data = user.settings.get("outlook_calendar")  # type: ignore
        if outlook_data and outlook_data.get("access_token"):  # type: ignore
            access_token = outlook_data.get("access_token")  # type: ignore
            refresh_token = outlook_data.get("refresh_token")  # type: ignore
            client_id = settings.outlook_client_id or settings.graph_client_id  # type: ignore
            client_secret = settings.outlook_client_secret or settings.graph_client_secret  # type: ignore
            token_expires_at = None
            if outlook_data.get("token_expires_at"):  # type: ignore
                token_expires_at = datetime.fromisoformat(str(outlook_data["token_expires_at"]))  # type: ignore
            
            # Verificar realmente si el token es válido
            if access_token and client_id and client_secret:
                try:
                    token_status = await outlook_service.verify_token_validity(
                        access_token=access_token,
                        refresh_token=refresh_token,
                        client_id=client_id,
                        client_secret=client_secret,
                        token_expires_at=token_expires_at
                    )
                    
                    # Si el token es válido pero se renovó, actualizar en BD
                    if token_status["valid"] and refresh_token and token_expires_at:
                        time_until_expiry = (token_expires_at - datetime.utcnow()).total_seconds()
                        if time_until_expiry < 300:  # Se renovó
                            try:
                                refreshed = outlook_service.refresh_token(refresh_token, client_id, client_secret, is_bot=False)
                                logger.info(f"Token de Outlook renovado durante verificación para {user_email}")
                                outlook_data["access_token"] = refreshed["access_token"]  # type: ignore
                                if refreshed.get("token_expires_at"):
                                    outlook_data["token_expires_at"] = refreshed["token_expires_at"].isoformat()  # type: ignore
                                if refreshed.get("refresh_token"):
                                    outlook_data["refresh_token"] = refreshed["refresh_token"]  # type: ignore
                                flag_modified(user, "settings")
                                db.commit()
                            except Exception as refresh_error:
                                logger.warning(f"Error renovando token de Outlook durante verificación: {refresh_error}")
                    
                    response["outlook_calendar"] = {
                        "connected": token_status["valid"],
                        "connected_at": outlook_data.get("connected_at"),  # type: ignore
                        "calendar_id": outlook_data.get("calendar_id"),  # type: ignore
                        "push_notifications_active": bool(outlook_data.get("subscription_id")),  # type: ignore
                        "needs_reauth": token_status["needs_reauth"],
                        "error": token_status.get("error"),
                    }
                except Exception as e:
                    logger.error(f"Error verificando token de Outlook para {user_email}: {e}")
                    # En caso de error, marcar como desconectado
                    response["outlook_calendar"] = {
                        "connected": False,
                        "connected_at": outlook_data.get("connected_at"),  # type: ignore
                        "push_notifications_active": False,
                        "needs_reauth": True,
                        "error": f"Error verificando token: {str(e)}",
                    }
            else:
                # No hay tokens o configuración faltante
                response["outlook_calendar"] = {
                    "connected": False,
                    "connected_at": outlook_data.get("connected_at"),  # type: ignore
                    "needs_reauth": True,
                    "error": "Configuración incompleta",
                }

    return response


@router.get("/oauth/start/{provider}")
async def start_oauth_flow(
    provider: str,
    request: Request,
    user_email: str = Query(..., description="Email del usuario"),
    db: Session = Depends(get_db),
):
    """
    Iniciar flujo OAuth para conectar calendario.
    Guarda la URL del frontend desde la que se llamo para redirigir al mismo sitio tras el callback.
    """
    user = get_user_by_email(user_email, db)

    # URL del frontend desde el que se inicia (para volver aqui tras Azure/Google)
    frontend_base = get_frontend_url(request)
    logger.info("OAuth start: frontend_base=%s (desde request)", frontend_base)

    # Generar state para seguridad (incluir email del usuario)
    state = secrets.token_urlsafe(32)
    logger.info(
        "Iniciando OAuth para usuario %s, provider %s, state generado: %s",
        user_email,
        provider,
        state,
    )

    # Guardar state y frontend base temporalmente (en produccion usar Redis)
    if not user.settings:  # type: ignore
        user.settings = {}  # type: ignore
    user.settings["_oauth_state"] = state  # type: ignore
    user.settings["_oauth_provider"] = provider  # type: ignore
    user.settings["_oauth_user_email"] = user_email  # type: ignore
    user.settings["_oauth_frontend_base"] = frontend_base  # type: ignore
    # Marcar el campo JSON como modificado para que SQLAlchemy lo detecte
    flag_modified(user, "settings")
    db.commit()
    db.refresh(user)  # Refrescar para asegurar que se guardo
    logger.info(
        "State guardado para usuario %s: %s. Verificacion: %s",
        user_email,
        state,
        user.settings.get("_oauth_state"),  # type: ignore
    )

    try:
        if provider == "google":
            auth_url = google_service.get_authorization_url(state=state)
        elif provider == "outlook":
            auth_url = outlook_service.get_authorization_url(state=state)
        else:
            raise HTTPException(status_code=400, detail=f"Proveedor no soportado: {provider}")

        logger.info("Redirigiendo a OAuth URL para %s: %s...", provider, auth_url[:100])
        return RedirectResponse(url=auth_url)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error iniciando OAuth para {provider}: {e}")
        raise HTTPException(status_code=500, detail=f"Error iniciando OAuth: {str(e)}")


@router.get("/oauth/callback/{provider}")
async def oauth_callback(
    provider: str,
    request: Request,
    code: Optional[str] = Query(default=None, description="Codigo de autorizacion"),
    state: Optional[str] = Query(default=None, description="State para verificacion"),
    error: Optional[str] = Query(default=None, description="Error de OAuth"),
    error_description: Optional[str] = Query(default=None, description="Descripcion del error"),
    error_subcode: Optional[str] = Query(default=None, description="Subcodigo del error"),
    db: Session = Depends(get_db),
):
    """
    Callback OAuth después de autorización del usuario.

    Args:
        provider: "google" o "outlook"
        code: Código de autorización
        state: State para verificación
        error: Error si hubo algún problema
        error_description: Descripción del error
        error_subcode: Subcódigo del error (ej: cancel)
    """
    logger.info(f"OAuth callback recibido - Provider: {provider}, State: {state}, Error: {error}, Error subcode: {error_subcode}, Code: {'presente' if code else 'ausente'}")
    
    # Verificar errores primero (usuario canceló, acceso denegado, etc.)
    if error:
        error_msg = error
        if error_description:
            error_msg = f"{error}: {error_description}"
        logger.error(f"Error en OAuth callback de {provider}: {error_msg} (subcode: {error_subcode})")
        
        # Si el usuario cancelo, usar un mensaje mas amigable
        if error == "access_denied" or error_subcode == "cancel":
            error_msg = "authorization_cancelled"

        frontend_base = get_frontend_url(request)
        return RedirectResponse(
            url=f"{frontend_base}/settings?tab=integrations&error={error_msg}"
        )

    if not code:
        logger.error("Callback OAuth sin codigo - Provider: %s, State: %s", provider, state)
        frontend_base = get_frontend_url(request)
        return RedirectResponse(
            url=f"{frontend_base}/settings?tab=integrations&error=no_code"
        )

    if not state:
        logger.error("Callback OAuth sin state - Provider: %s", provider)
        frontend_base = get_frontend_url(request)
        return RedirectResponse(
            url=f"{frontend_base}/settings?tab=integrations&error=invalid_state"
        )

    # Buscar usuario por state guardado
    logger.info(f"Buscando usuario con state: {state}")
    users = db.query(User).all()
    logger.info(f"Total de usuarios en DB: {len(users)}")
    
    target_user = None
    is_bot = False
    for user in users:
        if user.settings:  # type: ignore
            stored_state = user.settings.get("_oauth_state")  # type: ignore
            logger.debug(f"Usuario {user.email} tiene state: {stored_state}")
            if stored_state == state:
                target_user = user
                is_bot = user.settings.get("_oauth_is_bot", False)  # type: ignore
                logger.info(f"Usuario encontrado: {user.email}, is_bot={is_bot}")
                break

    if not target_user:
        # Intentar encontrar usuario por provider almacenado también (fallback)
        logger.warning(f"No se encontró usuario con state {state}. Buscando por provider {provider}")
        for user in users:
            if user.settings:  # type: ignore
                stored_provider = user.settings.get("_oauth_provider")  # type: ignore
                stored_email = user.settings.get("_oauth_user_email")  # type: ignore
                if stored_provider == provider:
                    logger.info(f"Encontrado usuario por provider fallback: {user.email}")
                    target_user = user
                    is_bot = user.settings.get("_oauth_is_bot", False)  # type: ignore
                    # Actualizar state para futuras búsquedas
                    if user.settings.get("_oauth_state") != state:  # type: ignore
                        logger.warning(f"State no coincide, pero usando usuario encontrado por provider. State almacenado: {user.settings.get('_oauth_state')}, State recibido: {state}")  # type: ignore
                        user.settings["_oauth_state"] = state  # type: ignore
                        flag_modified(user, "settings")
                        db.commit()
                        db.refresh(user)
                    break

    if not target_user:
        logger.error(
            "No se encontro usuario con state %s ni por provider %s. Estados: %s",
            state,
            provider,
            [(u.email, (u.settings or {}).get("_oauth_state"), (u.settings or {}).get("_oauth_provider")) for u in users[:5]],
        )
        frontend_base = get_frontend_url(request)
        return RedirectResponse(
            url=f"{frontend_base}/settings?tab=integrations&error=invalid_state"
        )

    # Usar la URL del frontend guardada al iniciar OAuth (donde estaba el usuario)
    frontend_base = (target_user.settings or {}).get("_oauth_frontend_base") or get_frontend_url(request)  # type: ignore
    logger.info("Callback: redirigiendo a frontend_base=%s", frontend_base)

    try:
        # Intercambiar código por tokens
        if provider == "google":
            tokens = google_service.exchange_code_for_token(code)
            integration_key = "google_calendar"
        elif provider == "outlook":
            # Calendario Outlook: siempre flujo usuario (sin bot)
            redirect_uri = getattr(settings, "outlook_redirect_uri", None)
            logger.info("Intercambiando codigo Outlook (usuario) - redirect_uri: %s", redirect_uri)
            tokens = outlook_service.exchange_code_for_token(
                code, is_bot=False, redirect_uri=redirect_uri
            )
            logger.info("Tokens obtenidos exitosamente para %s", provider)
            integration_key = "outlook_calendar"
        else:
            return RedirectResponse(
                url=f"{frontend_base}/settings?tab=integrations&error=invalid_provider"
            )

        # Guardar tokens en settings del usuario
        if not target_user.settings:  # type: ignore
            target_user.settings = {}  # type: ignore

        target_user.settings[integration_key] = {  # type: ignore
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token"),
            "token_expires_at": (
                tokens["token_expires_at"].isoformat()
                if tokens.get("token_expires_at")
                else None
            ),
            "connected_at": datetime.utcnow().isoformat(),
            "calendar_id": "primary",  # Default, se puede cambiar después
        }

        # Limpiar datos temporales de OAuth
        target_user.settings.pop("_oauth_state", None)  # type: ignore
        target_user.settings.pop("_oauth_provider", None)  # type: ignore
        target_user.settings.pop("_oauth_user_email", None)  # type: ignore
        target_user.settings.pop("_oauth_is_bot", None)  # type: ignore
        target_user.settings.pop("_oauth_frontend_base", None)  # type: ignore

        # Actualizar lista de integraciones activas
        if "integrations_enabled" not in target_user.settings:  # type: ignore
            target_user.settings["integrations_enabled"] = []  # type: ignore
        if provider not in target_user.settings["integrations_enabled"]:  # type: ignore
            target_user.settings["integrations_enabled"].append(provider)  # type: ignore

        # Marcar el campo JSON como modificado para que SQLAlchemy lo detecte
        flag_modified(target_user, "settings")
        db.commit()
        db.refresh(target_user)  # Refrescar para asegurar que se guardó

        # Configurar push notifications si está configurado backend_public_url
        if provider == "google" and settings.backend_public_url:  # type: ignore
            try:
                webhook_url = f"{settings.backend_public_url}/api/integrations/webhook/google-calendar"  # type: ignore
                watch_info = google_service.watch_calendar(
                    access_token=tokens["access_token"],
                    refresh_token=tokens.get("refresh_token"),
                    client_id=settings.google_client_id,  # type: ignore
                    client_secret=settings.google_client_secret,  # type: ignore
                    calendar_id="primary",
                    webhook_url=webhook_url,
                    user_id=str(target_user.id),  # type: ignore
                )
                
                # Guardar información del watch para poder detenerlo después
                target_user.settings[integration_key]["watch_channel_id"] = watch_info["channel_id"]  # type: ignore
                target_user.settings[integration_key]["watch_resource_id"] = watch_info["resource_id"]  # type: ignore
                target_user.settings[integration_key]["watch_expiration"] = watch_info["expiration"]  # type: ignore
                target_user.settings[integration_key]["watch_webhook_url"] = webhook_url  # type: ignore
                
                flag_modified(target_user, "settings")
                db.commit()
                db.refresh(target_user)
                
                logger.info(f"Push notifications configuradas para usuario {target_user.email}: channel_id={watch_info['channel_id']}")
            except Exception as watch_error:
                logger.warning(f"Error configurando push notifications: {watch_error}")

        if provider == "outlook" and not is_bot and settings.backend_public_url:  # type: ignore
            try:
                webhook_url = f"{settings.backend_public_url}/api/integrations/webhook/outlook-calendar"  # type: ignore
                sub_info = await outlook_service.create_subscription(
                    access_token=tokens["access_token"],
                    refresh_token=tokens.get("refresh_token"),
                    client_id=settings.outlook_client_id or settings.graph_client_id,  # type: ignore
                    client_secret=settings.outlook_client_secret or settings.graph_client_secret,  # type: ignore
                    notification_url=webhook_url,
                    token_expires_at=tokens.get("token_expires_at"),
                )
                target_user.settings[integration_key]["subscription_id"] = sub_info["id"]  # type: ignore
                target_user.settings[integration_key]["subscription_expiration"] = sub_info["expirationDateTime"]  # type: ignore
                target_user.settings[integration_key]["subscription_webhook_url"] = webhook_url  # type: ignore
                flag_modified(target_user, "settings")
                db.commit()
                db.refresh(target_user)
                logger.info("Push Outlook configurado para usuario %s: subscription_id=%s", target_user.email, sub_info["id"])
            except Exception as sub_error:
                logger.warning("Error configurando push Outlook: %s", sub_error)

        # Sincronizar calendarios automáticamente después de conectar
        try:
            await sync_service.sync_user_calendars(target_user, db)
        except Exception as sync_error:
            logger.warning(f"Error en sincronización automática: {sync_error}")

        return RedirectResponse(
            url=f"{frontend_base}/settings?tab=integrations&connected={provider}"
        )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(
            "Error procesando callback OAuth de %s: %s\nTraceback:\n%s",
            provider,
            e,
            error_traceback,
        )
        db.rollback()
        return RedirectResponse(
            url=f"{frontend_base}/settings?tab=integrations&error=callback_failed"
        )


@router.post("/disconnect/{provider}")
async def disconnect_integration(
    provider: str,
    user_email: str = Query(..., description="Email del usuario"),
    db: Session = Depends(get_db),
):
    """
    Desconectar una integración de calendario.

    Args:
        provider: "google" o "outlook"
        user_email: Email del usuario
    """
    user = get_user_by_email(user_email, db)

    if not user.settings:  # type: ignore
        raise HTTPException(status_code=400, detail="No hay integraciones conectadas")

    integration_key = f"{provider}_calendar"

    if integration_key not in user.settings:  # type: ignore
        raise HTTPException(
            status_code=404, detail=f"Integración {provider} no está conectada"
        )

    # Detener watch/push notifications si existe
    if provider == "google" and integration_key in user.settings:  # type: ignore
        google_data = user.settings[integration_key]  # type: ignore
        watch_channel_id = google_data.get("watch_channel_id")  # type: ignore
        watch_resource_id = google_data.get("watch_resource_id")  # type: ignore
        
        if watch_channel_id and watch_resource_id:
            try:
                access_token = google_data.get("access_token")
                refresh_token = google_data.get("refresh_token")
                
                if access_token:
                    google_service.stop_watch(
                        access_token=access_token,
                        refresh_token=refresh_token,
                        client_id=settings.google_client_id,  # type: ignore
                        client_secret=settings.google_client_secret,  # type: ignore
                        channel_id=watch_channel_id,
                        resource_id=watch_resource_id,
                    )
                    logger.info(f"Watch detenido para usuario {user.email}")
            except Exception as watch_error:
                logger.warning(f"Error deteniendo watch: {watch_error}")

    if provider == "outlook" and integration_key in user.settings:  # type: ignore
        outlook_data = user.settings[integration_key]  # type: ignore
        sub_id = outlook_data.get("subscription_id")  # type: ignore
        if sub_id:
            try:
                token_expires_at = None
                if outlook_data.get("token_expires_at"):  # type: ignore
                    token_expires_at = datetime.fromisoformat(str(outlook_data["token_expires_at"]))  # type: ignore
                await outlook_service.delete_subscription(
                    access_token=outlook_data.get("access_token"),  # type: ignore
                    refresh_token=outlook_data.get("refresh_token"),  # type: ignore
                    client_id=settings.outlook_client_id or settings.graph_client_id,  # type: ignore
                    client_secret=settings.outlook_client_secret or settings.graph_client_secret,  # type: ignore
                    subscription_id=sub_id,
                    token_expires_at=token_expires_at,
                )
                logger.info("Suscripcion Outlook detenida para usuario %s", user.email)
            except Exception as sub_error:
                logger.warning("Error deteniendo suscripcion Outlook: %s", sub_error)

    # Eliminar integración
    user.settings.pop(integration_key, None)  # type: ignore

    # Actualizar lista de integraciones activas
    if "integrations_enabled" in user.settings:  # type: ignore
        if provider in user.settings["integrations_enabled"]:  # type: ignore
            user.settings["integrations_enabled"].remove(provider)  # type: ignore

    # Marcar el campo JSON como modificado
    flag_modified(user, "settings")
    db.commit()
    db.refresh(user)

    return {"message": f"Integración {provider} desconectada exitosamente"}


@router.get("/calendar/events")
async def get_calendar_events(
    user_email: str = Query(..., description="Email del usuario"),
    provider: Optional[str] = Query(None, description="Proveedor específico (google/outlook)"),
    calendar_id: Optional[str] = Query(None, description="ID del calendario"),
    days_ahead: int = Query(30, description="Días hacia adelante para obtener eventos"),
    db: Session = Depends(get_db),
):
    """
    Obtener eventos de calendarios sincronizados del usuario.
    """
    user = get_user_by_email(user_email, db)

    if not user.settings:  # type: ignore
        return {"events": []}

    events = []

    # Google Calendar
    if (not provider or provider == "google") and user.settings.get("google_calendar"):  # type: ignore
        try:
            google_data = user.settings["google_calendar"]  # type: ignore
            access_token = google_data.get("access_token")  # type: ignore
            refresh_token = google_data.get("refresh_token")  # type: ignore

            if access_token:
                from datetime import timedelta

                time_min = datetime.utcnow()
                time_max = datetime.utcnow() + timedelta(days=days_ahead)

                google_events = google_service.get_events(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    client_id=settings.google_client_id,  # type: ignore
                    client_secret=settings.google_client_secret,  # type: ignore
                    calendar_id=calendar_id or google_data.get("calendar_id", "primary"),  # type: ignore
                    time_min=time_min,
                    time_max=time_max,
                )

                for event in google_events:
                    events.append(
                        {
                            "provider": "google",
                            "id": event.get("id"),
                            "title": event.get("summary"),
                            "start": event.get("start"),
                            "end": event.get("end"),
                            "description": event.get("description"),
                            "location": event.get("location"),
                        }
                    )
        except Exception as e:
            logger.error(f"Error obteniendo eventos de Google Calendar: {e}")

    # Outlook Calendar
    if (not provider or provider == "outlook") and user.settings.get("outlook_calendar"):  # type: ignore
        try:
            outlook_data = user.settings["outlook_calendar"]  # type: ignore
            access_token = outlook_data.get("access_token")  # type: ignore
            refresh_token = outlook_data.get("refresh_token")  # type: ignore
            token_expires_at = None
            token_expires_at_str = outlook_data.get("token_expires_at")  # type: ignore
            if token_expires_at_str:
                token_expires_at = datetime.fromisoformat(str(token_expires_at_str))

            if access_token:
                from datetime import timedelta

                time_min = datetime.utcnow()
                time_max = datetime.utcnow() + timedelta(days=days_ahead)

                outlook_events = await outlook_service.get_events(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    client_id=settings.outlook_client_id or settings.graph_client_id,  # type: ignore
                    client_secret=settings.outlook_client_secret or settings.graph_client_secret,  # type: ignore
                    calendar_id=calendar_id or outlook_data.get("calendar_id"),  # type: ignore
                    time_min=time_min,
                    time_max=time_max,
                    token_expires_at=token_expires_at,
                )

                for event in outlook_events:
                    events.append(
                        {
                            "provider": "outlook",
                            "id": event.get("id"),
                            "title": event.get("subject"),
                            "start": event.get("start"),
                            "end": event.get("end"),
                            "description": event.get("body", {}).get("content"),
                            "location": event.get("location", {}).get("displayName"),
                        }
                    )
        except Exception as e:
            logger.error(f"Error obteniendo eventos de Outlook Calendar: {e}")

    return {"events": events}


@router.get("/calendar/calendars")
async def get_calendars(
    user_email: str = Query(..., description="Email del usuario"),
    provider: Optional[str] = Query(None, description="Proveedor específico (google/outlook)"),
    db: Session = Depends(get_db),
):
    """
    Obtener lista de calendarios disponibles del usuario.
    """
    user = get_user_by_email(user_email, db)

    if not user.settings:  # type: ignore
        return {"calendars": []}

    calendars = []

    # Google Calendar
    if (not provider or provider == "google") and user.settings.get("google_calendar"):  # type: ignore
        try:
            google_data = user.settings["google_calendar"]  # type: ignore
            access_token = google_data.get("access_token")  # type: ignore
            refresh_token = google_data.get("refresh_token")  # type: ignore

            if access_token:
                google_calendars = google_service.get_calendars(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    client_id=settings.google_client_id,  # type: ignore
                    client_secret=settings.google_client_secret,  # type: ignore
                )

                for cal in google_calendars:
                    calendars.append(
                        {
                            "provider": "google",
                            "id": cal.get("id"),
                            "name": cal.get("summary"),
                            "primary": cal.get("primary", False),
                        }
                    )
        except Exception as e:
            logger.error(f"Error obteniendo calendarios de Google: {e}")

    # Outlook Calendar
    if (not provider or provider == "outlook") and user.settings.get("outlook_calendar"):  # type: ignore
        try:
            outlook_data = user.settings["outlook_calendar"]  # type: ignore
            access_token = outlook_data.get("access_token")  # type: ignore
            refresh_token = outlook_data.get("refresh_token")  # type: ignore
            token_expires_at = None
            token_expires_at_str = outlook_data.get("token_expires_at")  # type: ignore
            if token_expires_at_str:
                token_expires_at = datetime.fromisoformat(str(token_expires_at_str))

            if access_token:
                outlook_calendars = await outlook_service.get_calendars(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    client_id=settings.outlook_client_id or settings.graph_client_id,  # type: ignore
                    client_secret=settings.outlook_client_secret or settings.graph_client_secret,  # type: ignore
                    token_expires_at=token_expires_at,
                )

                for cal in outlook_calendars:
                    calendars.append(
                        {
                            "provider": "outlook",
                            "id": cal.get("id"),
                            "name": cal.get("name"),
                            "primary": cal.get("canEdit", False),
                        }
                    )
        except Exception as e:
            logger.error(f"Error obteniendo calendarios de Outlook: {e}")

    return {"calendars": calendars}


@router.post("/calendar/sync")
async def sync_calendars(
    user_email: str = Query(..., description="Email del usuario"),
    db: Session = Depends(get_db),
):
    """
    Forzar sincronización manual de calendarios del usuario.
    """
    user = get_user_by_email(user_email, db)

    try:
        results = await sync_service.run_full_sync_with_renewal(user, db)
        
        # Verificar si se necesita reautorización
        needs_reauth = results.get("google", {}).get("needs_reauth", False)
        
        response = {
            "message": "Sincronización completada",
            "results": results,
        }
        
        if needs_reauth:
            response["warning"] = "Tu cuenta de Google Calendar necesita ser reconectada. Por favor, desconecta y vuelve a conectar tu cuenta."
        
        return response
    except Exception as e:
        logger.error(f"Error sincronizando calendarios: {e}")
        raise HTTPException(status_code=500, detail=f"Error en sincronización: {str(e)}")


@router.post("/calendar/enable-push-notifications")
async def enable_push_notifications(
    user_email: str = Query(..., description="Email del usuario"),
    provider: str = Query(..., description="Proveedor: google o outlook"),
    db: Session = Depends(get_db),
):
    """
    Activar push notifications (webhooks) para una integración ya conectada.
    
    Útil si conectaste el calendario antes de que se implementaran las push notifications.
    """
    user = get_user_by_email(user_email, db)

    if not settings.backend_public_url:  # type: ignore
        raise HTTPException(
            status_code=400,
            detail="BACKEND_PUBLIC_URL no está configurado. Configúralo en .env para activar push notifications."
        )

    if provider == "google":
        integration_key = "google_calendar"
        if not user.settings or not user.settings.get(integration_key):  # type: ignore
            raise HTTPException(
                status_code=404,
                detail="Google Calendar no está conectado. Conéctalo primero."
            )

        google_data = user.settings[integration_key]  # type: ignore
        
        # Verificar si ya tiene watch configurado
        if google_data.get("watch_channel_id"):  # type: ignore
            return {
                "message": "Push notifications ya están activas",
                "channel_id": google_data.get("watch_channel_id"),
            }

        try:
            access_token = google_data.get("access_token")
            refresh_token = google_data.get("refresh_token")
            
            if not access_token:
                raise HTTPException(status_code=400, detail="No hay tokens de acceso disponibles")

            webhook_url = f"{settings.backend_public_url}/api/integrations/webhook/google-calendar"  # type: ignore
            
            watch_info = google_service.watch_calendar(
                access_token=access_token,
                refresh_token=refresh_token,
                client_id=settings.google_client_id,  # type: ignore
                client_secret=settings.google_client_secret,  # type: ignore
                calendar_id=google_data.get("calendar_id", "primary"),  # type: ignore
                webhook_url=webhook_url,
                user_id=str(user.id),  # type: ignore
            )
            
            # Guardar información del watch
            google_data["watch_channel_id"] = watch_info["channel_id"]  # type: ignore
            google_data["watch_resource_id"] = watch_info["resource_id"]  # type: ignore
            google_data["watch_expiration"] = watch_info["expiration"]  # type: ignore
            google_data["watch_webhook_url"] = webhook_url  # type: ignore
            
            flag_modified(user, "settings")
            db.commit()
            db.refresh(user)
            
            logger.info(f"Push notifications activadas para usuario {user.email}: channel_id={watch_info['channel_id']}")
            
            return {
                "message": "Push notifications activadas exitosamente",
                "channel_id": watch_info["channel_id"],
                "expiration": watch_info["expiration"],
            }
        except Exception as e:
            logger.error(f"Error activando push notifications: {e}")
            raise HTTPException(status_code=500, detail=f"Error activando push notifications: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail="Solo Google Calendar soporta push notifications actualmente")


@router.post("/webhook/google-calendar")
async def google_calendar_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Endpoint para recibir notificaciones push de Google Calendar.
    
    Google envía notificaciones cuando hay cambios en calendarios suscritos.
    Este endpoint valida la notificación y sincroniza automáticamente.
    """
    # Google envía un token X-Goog-Channel-Token en los headers
    # También envía el channel-id y resource-id
    
    headers = request.headers
    channel_id = headers.get("X-Goog-Channel-Id")
    resource_id = headers.get("X-Goog-Resource-Id")
    channel_token = headers.get("X-Goog-Channel-Token")
    
    try:
        body = await request.body()
    except ClientDisconnect:
        body = b""
        logger.info("Webhook Google Calendar: cliente desconectado antes de enviar body, tratando como body vacio")
    except Exception as e:
        body = b""
        logger.warning(f"Webhook Google Calendar: error leyendo body: {e}, tratando como body vacio")
    
    body_str = body.decode('utf-8') if body else ""
    
    logger.info(
        f"Webhook de Google Calendar recibido - channel_id={channel_id}, "
        f"resource_id={resource_id}, token={channel_token[:20] if channel_token else None}, "
        f"body_length={len(body) if body else 0}"
    )
    
    # Buscar usuario por channel_id o resource_id
    # El channel_id lo guardamos cuando creamos el watch
    users = db.query(User).all()
    target_user = None
    
    for user in users:
        if user.settings:  # type: ignore
            google_data = user.settings.get("google_calendar")  # type: ignore
            if google_data:  # type: ignore
                stored_channel_id = google_data.get("watch_channel_id")  # type: ignore
                stored_resource_id = google_data.get("watch_resource_id")  # type: ignore
                
                if (stored_channel_id == channel_id) or (stored_resource_id == resource_id):
                    target_user = user
                    break
    
    if not target_user:
        # Si no encontramos usuario, puede ser una verificación inicial o un channel_id inválido
        if not body or len(body) == 0:
            logger.info("Notificación de verificación inicial de Google Calendar (sin usuario asociado)")
        else:
            logger.warning(f"No se encontró usuario con channel_id={channel_id} o resource_id={resource_id}")
        return {"status": "ok"}  # Responder OK para evitar reintentos
    
    # Si encontramos el usuario, sincronizar (ya sea verificación inicial o cambio real)
    # Google envía notificaciones incluso cuando el body está vacío, así que sincronizamos siempre
    try:
        if not body or len(body) == 0:
            logger.info(f"Notificación de verificación inicial para usuario {target_user.email} - sincronizando calendarios")
        else:
            logger.info(f"Notificación de cambio en calendario para usuario {target_user.email} - sincronizando")
        
        results = await sync_service.sync_user_calendars(target_user, db)
        logger.info(f"Sincronización completada para {target_user.email}: {results}")
        
        return {"status": "ok", "synced": True}
    except Exception as e:
        logger.error(f"Error sincronizando tras webhook de Google Calendar para usuario {target_user.email}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "ok"}  # Responder OK para evitar reintentos, pero loguear el error


@router.get("/webhook/google-calendar")
async def google_calendar_webhook_verification(
    request: Request,
):
    """
    Endpoint GET para verificación inicial del webhook de Google Calendar.
    
    Google puede hacer una petición GET para verificar que el endpoint existe.
    """
    logger.info("Verificación GET del webhook de Google Calendar")
    return {"status": "ok"}


@router.post("/webhook/outlook-calendar")
async def outlook_calendar_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Endpoint para notificaciones de Microsoft Graph (cambios en calendario Outlook).
    Validacion: Graph envia ?validationToken=...; devolver 200 con el token.
    Cambios: body JSON con subscriptionId; devolver 202 y sincronizar en background (debounce).
    """
    validation_token = request.query_params.get("validationToken")
    if validation_token:
        logger.info("Webhook Outlook: validacion (validationToken)")
        return PlainTextResponse(content=validation_token, status_code=200)

    try:
        body = await request.body()
    except ClientDisconnect:
        body = b""
        logger.info("Webhook Outlook: ClientDisconnect, body vacio")
    except Exception as e:
        body = b""
        logger.warning("Webhook Outlook: error leyendo body: %s", e)

    raw = body.decode("utf-8") if body else "{}"
    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}
    value = data.get("value", [])
    sub_ids = [n.get("subscriptionId") for n in value if n.get("subscriptionId")]

    # Fase 4: Si no hay subscriptionId, devolver 202 inmediatamente sin procesar
    if not sub_ids:
        logger.info(
            "[WEBHOOK_OUTLOOK] Sin subscriptionId - webhook_descartado=true, body_length=%d, timestamp=%s",
            len(body) if body else 0, datetime.utcnow().isoformat()
        )
        return Response(status_code=202)

    users = db.query(User).all()
    target_user = None
    for u in users:
        if not u.settings:
            continue
        oc = u.settings.get("outlook_calendar") or {}
        sid = oc.get("subscription_id")
        if sid and sid in sub_ids:
            target_user = u
            break

    if not target_user:
        logger.warning(
            "[WEBHOOK_OUTLOOK] Usuario no encontrado - subscription_ids=%s, webhook_descartado=true, timestamp=%s",
            sub_ids, datetime.utcnow().isoformat()
        )
        return Response(status_code=202)

    user_id = str(target_user.id)
    user_email = target_user.email

    # Fase 1: Verificar cooldown antes de programar sync
    now = datetime.utcnow()
    last_sync_at = _OUTLOOK_LAST_SYNC_AT.get(user_id)
    if last_sync_at:
        time_since_last_sync = (now - last_sync_at).total_seconds()
        if time_since_last_sync < COOLDOWN_SECONDS:
            # MEJORA: En lugar de descartar, guardar el webhook para procesarlo después del cooldown
            if user_id not in _OUTLOOK_PENDING_WEBHOOKS:
                _OUTLOOK_PENDING_WEBHOOKS[user_id] = []
            _OUTLOOK_PENDING_WEBHOOKS[user_id].append(now)
            
            # Programar procesamiento después del cooldown
            remaining_cooldown = COOLDOWN_SECONDS - time_since_last_sync
            async def _process_pending_webhook():
                try:
                    await asyncio.sleep(remaining_cooldown + 1)  # +1 para asegurar que pasó el cooldown
                    if user_id in _OUTLOOK_PENDING_WEBHOOKS and _OUTLOOK_PENDING_WEBHOOKS[user_id]:
                        # Limpiar webhooks pendientes
                        _OUTLOOK_PENDING_WEBHOOKS.pop(user_id, None)
                        # Encolar sync
                        from app.tasks.calendar_tasks import run_calendar_sync_for_user
                        task = run_calendar_sync_for_user.apply_async(args=[user_id])
                        logger.info(
                            "[WEBHOOK_OUTLOOK] Webhook pendiente procesado tras cooldown - usuario=%s, task_id=%s, subscription_ids=%s",
                            user_email, task.id, sub_ids
                        )
                        _OUTLOOK_LAST_SYNC_AT[user_id] = datetime.utcnow()
                except Exception as e:
                    logger.error("Error procesando webhook pendiente para %s: %s", user_email, e)
            
            asyncio.create_task(_process_pending_webhook())
            
            logger.info(
                "[WEBHOOK_OUTLOOK] Cooldown activo - usuario=%s, ultimo_sync_hace=%.1fs, cooldown=%ds, webhook_encolado_para_despues=true, subscription_ids=%s",
                user_email, time_since_last_sync, COOLDOWN_SECONDS, sub_ids
            )
            return Response(status_code=202)
    else:
        logger.info(
            "[WEBHOOK_OUTLOOK] Sin cooldown previo - usuario=%s, webhook_procesado=true, subscription_ids=%s",
            user_email, sub_ids
        )

    async def _debounced_sync() -> None:
        try:
            await asyncio.sleep(_OUTLOOK_DEBOUNCE_SECONDS)
        except asyncio.CancelledError:
            return
        _OUTLOOK_WEBHOOK_DEBOUNCE.pop(user_id, None)
        # Fase 2: Encolar tarea Celery en lugar de ejecutar sync en proceso API
        try:
            from app.tasks.calendar_tasks import run_calendar_sync_for_user
            task = run_calendar_sync_for_user.apply_async(args=[user_id])
            sync_timestamp = datetime.utcnow()
            logger.info(
                "[WEBHOOK_OUTLOOK] Tarea Celery encolada - usuario=%s, task_id=%s, debounce=%ds, timestamp=%s, subscription_ids=%s",
                user_email, task.id, _OUTLOOK_DEBOUNCE_SECONDS, sync_timestamp.isoformat(), sub_ids
            )
            # Fase 1: Actualizar timestamp del ultimo sync completado
            # Nota: El sync real se ejecutara en Celery, pero actualizamos el timestamp
            # cuando se encola la tarea. Si queremos ser mas precisos, podriamos actualizar
            # cuando la tarea Celery completa, pero eso requeriria un callback o polling.
            # Por ahora, actualizamos al encolar para evitar bucle inmediato.
            _OUTLOOK_LAST_SYNC_AT[user_id] = sync_timestamp
        except Exception as e:
            logger.error("Webhook Outlook: error encolando tarea Celery para %s: %s", user_email, e)

    prev = _OUTLOOK_WEBHOOK_DEBOUNCE.pop(user_id, None)
    if prev and not prev.done():
        prev.cancel()
        logger.info(
            "[WEBHOOK_OUTLOOK] Debounce activado - usuario=%s, debounce=%ds, webhook_reprogramado=true, subscription_ids=%s",
            user_email, _OUTLOOK_DEBOUNCE_SECONDS, sub_ids
        )
    t = asyncio.create_task(_debounced_sync())
    _OUTLOOK_WEBHOOK_DEBOUNCE[user_id] = t
    return Response(status_code=202)


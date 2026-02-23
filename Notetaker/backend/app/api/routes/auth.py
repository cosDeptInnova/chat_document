"""Endpoints de autenticación OAuth2 y login sencillo por email."""
from fastapi import APIRouter, Request, HTTPException, Query, Depends, Response
from fastapi import Request
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, EmailStr
from app.services.oauth_service import OAuthService
from app.database import get_db
from app.models.user import User
from app.config import settings
from app.utils.password_utils import hash_password, verify_password, generate_reset_token
from app.services.email_service import send_password_reset_email
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Instancia global del OAuthService
oauth_service = OAuthService()


class SimpleLoginRequest(BaseModel):
    """Login simplificado por email con contraseña."""

    email: EmailStr
    password: str
    display_name: str | None = None
    remember_me: bool = False  # Opción "Recordarme" - extiende expiración a 30 días


class ForgotPasswordRequest(BaseModel):
    """Request para solicitar reset de contraseña."""
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """Request para resetear contraseña con token."""
    token: str
    new_password: str


class ChangePasswordRequest(BaseModel):
    """Request para cambiar contraseña (requiere password actual)."""
    current_password: str
    new_password: str
    user_email: str


class SetInitialPasswordRequest(BaseModel):
    """Request para establecer password inicial (usuarios existentes sin password)."""
    email: EmailStr
    password: str


class AdminResetPasswordRequest(BaseModel):
    """Request para que admin resetee password de un usuario."""
    user_email: EmailStr
    new_password: str


class AdminCreateUserRequest(BaseModel):
    """Request para que admin cree un nuevo usuario."""
    email: EmailStr
    password: str  # Contraseña de 1 uso
    display_name: str | None = None


class SSOLoginRequest(BaseModel):
    """Request para login SSO desde Cosmos (get-or-create usuario)."""
    email: EmailStr
    display_name: str | None = None
    cosmos_token: str | None = None  # Opcional: token de Cosmos para validación futura


@router.get("/login", response_class=HTMLResponse)
async def login(redirect_uri: str = Query(None)):
    """
    Página de inicio de sesión OAuth2.
    Redirige al usuario a Azure AD para autorizar.
    """
    try:
        auth_url = oauth_service.get_auth_url(redirect_uri)
        
        logger.info(f"Generando URL de autorización: {auth_url[:100]}...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iniciar Sesión - Cosmos Notetaker</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        
        .container {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 500px;
            width: 100%;
            padding: 40px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 32px;
        }}
        
        .logo {{
            font-size: 32px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 16px;
        }}
        
        .message {{
            background: #f0f7ff;
            border-left: 4px solid #667eea;
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 24px;
            color: #333;
            font-size: 15px;
            line-height: 1.6;
        }}
        
        .auth-button {{
            display: block;
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            text-decoration: none;
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
        
        .auth-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }}
        
        .permissions {{
            background: #fff9e6;
            border-left: 4px solid #ffc107;
            padding: 16px 20px;
            border-radius: 8px;
            margin-top: 24px;
            font-size: 14px;
            color: #856404;
        }}
        
        .permissions ul {{
            margin-top: 8px;
            padding-left: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">📝 Cosmos Notetaker</div>
            <div class="subtitle">Transcripción y análisis de reuniones Teams</div>
        </div>
        
        <div class="message">
            Para usar este servicio, necesitamos tu autorización para acceder a tu calendario de Microsoft Teams 
            y poder unirnos automáticamente a tus reuniones.
        </div>
        
        <a href="{auth_url}" class="auth-button">
            🔐 Iniciar sesión con Microsoft
        </a>
        
        <div class="permissions">
            <strong>Permisos solicitados:</strong>
            <ul>
                <li>Leer tu perfil de usuario</li>
                <li>Leer y escribir en tu calendario</li>
                <li>Acceso a tus reuniones de Teams</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        logger.error(f"Error en login: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/callback")
async def callback(
    code: str = Query(None),
    error: str = Query(None),
    error_description: str = Query(None),
    db: Session = Depends(get_db)
):
    """
    Callback de OAuth2 después de la autorización.
    Intercambia el código por tokens y crea/actualiza el usuario.
    """
    if error:
        logger.error(f"Error en OAuth callback: {error} - {error_description}")
        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Error de Autorización</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            padding: 40px;
            max-width: 500px;
            text-align: center;
        }}
        .error {{
            color: #d32f2f;
            font-size: 18px;
            margin-bottom: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="error">❌ Error de Autorización</div>
        <p>{error_description or error}</p>
    </div>
</body>
</html>
"""
        return HTMLResponse(content=html_content, status_code=400)
    
    if not code:
        raise HTTPException(status_code=400, detail="No se recibió código de autorización")
    
    try:
        # Intercambiar código por tokens
        token_result = await oauth_service.exchange_code_for_token(code)
        
        if "error" in token_result:
            logger.error(f"Error obteniendo token: {token_result['error']}")
            raise HTTPException(status_code=400, detail=token_result["error"])
        
        user_id = token_result["user_id"]
        tenant_id = token_result["tenant_id"]
        user_email = token_result["user_email"]
        display_name = token_result.get("display_name", "")
        
        # Buscar o crear usuario en la base de datos
        user = db.query(User).filter(User.microsoft_user_id == user_id).first()
        
        if user:
            # Actualizar usuario existente
            user.email = user_email
            user.display_name = display_name
            user.tenant_id = tenant_id
            user.last_login_at = datetime.utcnow()
        else:
            # Crear nuevo usuario
            user = User(
                microsoft_user_id=user_id,
                tenant_id=tenant_id,
                email=user_email,
                display_name=display_name,
                bot_display_name=settings.default_bot_name,
                last_login_at=datetime.utcnow()
            )
            db.add(user)
        
        db.commit()
        db.refresh(user)
        
        logger.info(f"✅ Usuario autenticado: {user_email} ({user_id})")
        
        # Página de éxito
        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Autorización Exitosa</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            padding: 40px;
            max-width: 500px;
            text-align: center;
        }}
        .success {{
            color: #4caf50;
            font-size: 48px;
            margin-bottom: 16px;
        }}
        .message {{
            color: #333;
            font-size: 18px;
            margin-bottom: 24px;
        }}
        .close-button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="success">✅</div>
        <div class="message">¡Autorización exitosa!</div>
        <p>Tu cuenta ha sido vinculada correctamente. Ya puedes cerrar esta ventana.</p>
        <button class="close-button" onclick="window.close()">Cerrar</button>
    </div>
</body>
</html>
"""
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        logger.error(f"Error en OAuth callback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def auth_status(user_id: str = Query(..., description="Microsoft User ID (oid)"), db: Session = Depends(get_db)):
    """
    Verifica el estado de autenticación de un usuario.
    
    Returns:
        Dict con is_authenticated, user_id, email
    """
    try:
        user = db.query(User).filter(User.microsoft_user_id == user_id).first()
        
        if not user:
            return {
                "is_authenticated": False,
                "message": "Usuario no autenticado"
            }
        
        # Verificar si el token sigue válido (opcional)
        try:
            await oauth_service.get_valid_token(user.microsoft_user_id)
            token_valid = True
        except ValueError:
            token_valid = False
        
        return {
            "is_authenticated": token_valid,
            "user_id": user.id,
            "microsoft_user_id": user.microsoft_user_id,
            "email": user.email,
            "display_name": user.display_name,
            "tenant_id": user.tenant_id
        }
    
    except Exception as e:
        logger.error(f"Error verificando estado de auth: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logout")
async def logout(user_id: str = Query(..., description="Microsoft User ID (oid)"), db: Session = Depends(get_db)):
    """
    Cierra sesión y elimina tokens del usuario.
    """
    try:
        # Eliminar tokens de Redis/memoria
        if oauth_service.redis_client:
            try:
                oauth_service.redis_client.delete(f"tokens:user:{user_id}")
            except Exception as e:
                logger.warning(f"Error eliminando token de Redis: {e}")
        
        # Opcional: marcar usuario como inactivo en DB
        user = db.query(User).filter(User.microsoft_user_id == user_id).first()
        if user:
            user.is_active = False
            db.commit()
        
        return {"success": True, "message": "Sesión cerrada correctamente"}
    
    except Exception as e:
        logger.error(f"Error en logout: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simple-login")
async def simple_login(
    payload: SimpleLoginRequest,
    db: Session = Depends(get_db),
):
    """
    Login con email y contraseña para el frontend React.

    - El usuario DEBE existir en la base de datos.
    - El usuario DEBE tener contraseña configurada.
    - Si tiene must_change_password=True, se devuelve flag para forzar cambio.
    - Devuelve información del usuario y si es admin.
    """
    try:
        email = payload.email.strip().lower()

        # Buscar usuario - DEBE existir
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            # Usuario no existe - mensaje claro
            raise HTTPException(
                status_code=401,
                detail="El email no está registrado en el sistema. Contacta al administrador para crear tu cuenta."
            )
        
        # El usuario DEBE tener contraseña configurada
        if not user.hashed_password:
            raise HTTPException(
                status_code=401,
                detail="Tu cuenta no tiene contraseña configurada. Contacta al administrador para establecer una contraseña."
            )
        
        # Verificar contraseña
        if not verify_password(payload.password, user.hashed_password):
            logger.warning(f"❌ Intento de login fallido para {email}")
            raise HTTPException(
                status_code=401,
                detail="Usuario o contraseña incorrectos. Por favor, verifica tus credenciales e intenta nuevamente."
            )
        
        # Actualizar último login
        user.last_login_at = datetime.utcnow()
        db.commit()
        
        # Determinar si es admin
        is_admin = False
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email.endswith("@" + admin_domain.lower()):
            is_admin = True

        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [
                e.strip().lower() for e in admin_emails.split(",") if e.strip()
            ]
            if email in admin_list:
                is_admin = True

        # Determinar nivel de licencia
        from app.utils.license_utils import get_user_license_level
        license_level = get_user_license_level(user)
        
        # Generar JWT token con expiración
        from app.utils.jwt_utils import generate_token
        token = generate_token(
            user_id=user.id,
            email=user.email,
            remember_me=payload.remember_me
        )
        
        return {
            "user_id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "is_admin": is_admin,
            "is_premium": user.is_premium,
            "license": license_level,
            "must_change_password": user.must_change_password,  # Flag para cambio forzado
            "token": token,  # Token JWT con expiración
            "token_expires_in_days": 30 if payload.remember_me else 1,  # Información de expiración
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error en simple-login: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error en login sencillo")


@router.post("/sso-login")
async def sso_login(
    payload: SSOLoginRequest,
    db: Session = Depends(get_db),
):
    """
    Login SSO desde Cosmos: si el usuario existe se usa; si no, se crea con licencia básica.
    No requiere contraseña. Devuelve la misma forma que simple-login (token + user).
    """
    try:
        email = payload.email.strip().lower()
        display_name = (payload.display_name or "").strip() or email.split("@")[0]

        # Opcional: validar cosmos_token si COSMOS_SSO_SECRET está configurado (futuro)
        _ = getattr(settings, "cosmos_sso_secret", None) and payload.cosmos_token

        user = db.query(User).filter(User.email == email).first()

        if user:
            user.last_login_at = datetime.utcnow()
            if display_name:
                user.display_name = display_name
            db.commit()
            db.refresh(user)
        else:
            default_bot = getattr(settings, "default_bot_name", "Notetaker")
            new_user = User(
                email=email,
                display_name=display_name,
                microsoft_user_id=f"cosmos_sso_{email}",
                tenant_id="cosmos_sso",
                bot_display_name=default_bot,
                is_active=True,
                hashed_password=None,
                must_change_password=False,
                is_premium=False,
                last_login_at=datetime.utcnow(),
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            user = new_user
            logger.info("Usuario SSO creado con licencia basica: %s", email)

        is_admin = False
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email.endswith("@" + admin_domain.lower()):
            is_admin = True
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [e.strip().lower() for e in admin_emails.split(",") if e.strip()]
            if email in admin_list:
                is_admin = True

        from app.utils.license_utils import get_user_license_level
        from app.utils.jwt_utils import generate_token

        license_level = get_user_license_level(user)
        token = generate_token(user_id=user.id, email=user.email, remember_me=True)

        return {
            "user_id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "is_admin": is_admin,
            "is_premium": user.is_premium,
            "license": license_level,
            "must_change_password": user.must_change_password or False,
            "token": token,
            "token_expires_in_days": 30,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error en sso-login: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error en login SSO")


@router.get("/me")
async def get_current_user(
    user_email: str = Query(..., description="Email del usuario actual"),
    db: Session = Depends(get_db),
):
    """
    Obtiene la información actualizada del usuario actual.
    Útil para refrescar la información después de cambios (licencia, etc.).
    """
    try:
        email = user_email.strip().lower()
        
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # Recalcular si es admin (puede haber cambiado en ADMIN_EMAILS)
        is_admin = False
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email.endswith("@" + admin_domain.lower()):
            is_admin = True
        
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [
                e.strip().lower() for e in admin_emails.split(",") if e.strip()
            ]
            if email in admin_list:
                is_admin = True
        
        # Obtener nivel de licencia actualizado
        from app.utils.license_utils import get_user_license_level
        license_level = get_user_license_level(user)
        
        return {
            "user_id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "is_admin": is_admin,
            "is_premium": user.is_premium,
            "license": license_level,
            "must_change_password": user.must_change_password or False,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error obteniendo usuario actual: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error al obtener usuario actual")


def get_reset_password_url(request: Request) -> str:
    """
    Determina la URL del frontend para el reset de contraseña basándose en los headers de la petición.
    
    Prioridad:
    1. Si detecta IP VPN (172.x.x.x), usar http://172.29.14.14:5173/
    2. Si detecta producción (notetaker.cosgs.com), usar https://notetaker.cosgs.com
    3. Si settings.frontend_url contiene localhost, usar https://notetaker.cosgs.com
    4. Fallback: https://notetaker.cosgs.com
    """
    from urllib.parse import urlparse
    
    # 1. Detectar desde X-Forwarded-Host (Nginx)
    forwarded_host = request.headers.get("x-forwarded-host")
    if forwarded_host:
        # Si es IP VPN (172.x.x.x), usar la IP local
        if forwarded_host.startswith("172."):
            frontend_url = "http://172.29.14.14:5173"
            logger.info(f"[get_reset_password_url] Detectado VPN desde X-Forwarded-Host: {frontend_url}")
            return frontend_url
        # Si es notetaker.cosgs.com, usar HTTPS
        elif "notetaker.cosgs.com" in forwarded_host.lower():
            frontend_url = "https://notetaker.cosgs.com"
            logger.info(f"[get_reset_password_url] Detectado producción desde X-Forwarded-Host: {frontend_url}")
            return frontend_url
    
    # 2. Detectar desde Host header
    host = request.headers.get("host")
    if host:
        # Si es IP VPN (172.x.x.x), usar la IP local
        if host.startswith("172."):
            frontend_url = "http://172.29.14.14:5173"
            logger.info(f"[get_reset_password_url] Detectado VPN desde Host header: {frontend_url}")
            return frontend_url
        # Si es notetaker.cosgs.com, usar HTTPS
        elif "notetaker.cosgs.com" in host.lower():
            frontend_url = "https://notetaker.cosgs.com"
            logger.info(f"[get_reset_password_url] Detectado producción desde Host header: {frontend_url}")
            return frontend_url
    
    # 3. Detectar desde Referer
    referer = request.headers.get("referer")
    if referer:
        try:
            parsed = urlparse(referer)
            if parsed.hostname:
                # Si es IP VPN (172.x.x.x), usar la IP local
                if parsed.hostname.startswith("172."):
                    frontend_url = "http://172.29.14.14:5173"
                    logger.info(f"[get_reset_password_url] Detectado VPN desde Referer: {frontend_url}")
                    return frontend_url
                # Si es notetaker.cosgs.com, usar HTTPS
                elif "notetaker.cosgs.com" in parsed.hostname.lower():
                    frontend_url = "https://notetaker.cosgs.com"
                    logger.info(f"[get_reset_password_url] Detectado producción desde Referer: {frontend_url}")
                    return frontend_url
        except Exception as e:
            logger.debug(f"Error parseando Referer: {e}")
    
    # 4. Detectar desde Origin
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed = urlparse(origin)
            if parsed.hostname:
                # Si es IP VPN (172.x.x.x), usar la IP local
                if parsed.hostname.startswith("172."):
                    frontend_url = "http://172.29.14.14:5173"
                    logger.info(f"[get_reset_password_url] Detectado VPN desde Origin: {frontend_url}")
                    return frontend_url
                # Si es notetaker.cosgs.com, usar HTTPS
                elif "notetaker.cosgs.com" in parsed.hostname.lower():
                    frontend_url = "https://notetaker.cosgs.com"
                    logger.info(f"[get_reset_password_url] Detectado producción desde Origin: {frontend_url}")
                    return frontend_url
        except Exception as e:
            logger.debug(f"Error parseando Origin: {e}")
    
    # 5. Verificar settings.frontend_url - si contiene localhost, usar producción
    if settings.frontend_url and "localhost" in settings.frontend_url.lower():
        frontend_url = "https://notetaker.cosgs.com"
        logger.info(f"[get_reset_password_url] settings.frontend_url contiene localhost, usando producción: {frontend_url}")
        return frontend_url
    
    # 6. Fallback: usar producción por defecto
    frontend_url = "https://notetaker.cosgs.com"
    logger.info(f"[get_reset_password_url] Usando fallback (producción): {frontend_url}")
    return frontend_url


@router.post("/forgot-password")
async def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Solicita reset de contraseña. Envía email con link de recuperación.
    """
    try:
        email = payload.email.strip().lower()
        
        user = db.query(User).filter(User.email == email).first()
        
        # Por seguridad, siempre devolver éxito (no revelar si el email existe)
        if not user:
            logger.info(f"⚠️ Solicitud de reset de password para email inexistente: {email}")
            return {"message": "Si el email existe, se ha enviado un correo con instrucciones."}
        
        # Generar token de reset
        reset_token = generate_reset_token()
        reset_expires = datetime.utcnow() + timedelta(hours=1)  # Expira en 1 hora
        
        # Guardar token en usuario
        user.password_reset_token = reset_token
        user.password_reset_expires = reset_expires
        db.commit()
        
        # Construir URL de reset usando detección desde request
        frontend_url = get_reset_password_url(request).rstrip('/')
        reset_url = f"{frontend_url}/reset-password?token={reset_token}"
        
        # Enviar email
        email_sent = send_password_reset_email(email, reset_token, reset_url)
        
        if email_sent:
            logger.info(f"✅ Email de recuperación enviado a {email} con URL: {reset_url}")
        else:
            logger.error(f"❌ Error al enviar email de recuperación a {email}")
        
        return {"message": "Si el email existe, se ha enviado un correo con instrucciones."}
        
    except Exception as exc:
        logger.error("Error en forgot-password: %s", exc, exc_info=True)
        # Por seguridad, siempre devolver éxito
        return {"message": "Si el email existe, se ha enviado un correo con instrucciones."}


@router.post("/reset-password")
async def reset_password(
    payload: ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Resetea la contraseña usando el token de recuperación.
    """
    try:
        if len(payload.new_password) < 8:
            raise HTTPException(
                status_code=400,
                detail="La contraseña debe tener al menos 8 caracteres"
            )
        
        # Buscar usuario por token
        user = db.query(User).filter(
            User.password_reset_token == payload.token
        ).first()
        
        if not user:
            raise HTTPException(
                status_code=400,
                detail="Token inválido o expirado"
            )
        
        # Verificar que el token no haya expirado
        if user.password_reset_expires and user.password_reset_expires < datetime.utcnow():
            # Limpiar token expirado
            user.password_reset_token = None
            user.password_reset_expires = None
            db.commit()
            raise HTTPException(
                status_code=400,
                detail="Token expirado. Por favor, solicita un nuevo link de recuperación."
            )
        
        # Actualizar contraseña
        user.hashed_password = hash_password(payload.new_password)
        user.password_reset_token = None
        user.password_reset_expires = None
        db.commit()
        
        logger.info(f"✅ Contraseña reseteada exitosamente para {user.email}")
        
        return {"message": "Contraseña actualizada exitosamente"}
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error en reset-password: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error al resetear contraseña")


@router.post("/change-password")
async def change_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Cambia la contraseña del usuario autenticado (requiere password actual).
    """
    try:
        email = payload.user_email.strip().lower()
        
        if len(payload.new_password) < 8:
            raise HTTPException(
                status_code=400,
                detail="La contraseña debe tener al menos 8 caracteres"
            )
        
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        if not user.hashed_password:
            raise HTTPException(
                status_code=400,
                detail="Usuario sin contraseña configurada. Usa el endpoint set-initial-password."
            )
        
        # Verificar contraseña actual
        if not verify_password(payload.current_password, user.hashed_password):
            raise HTTPException(
                status_code=401,
                detail="Contraseña actual incorrecta"
            )
        
        # Actualizar contraseña
        user.hashed_password = hash_password(payload.new_password)
        # Si tenía must_change_password, limpiarlo (ya cambió la contraseña)
        if user.must_change_password:
            user.must_change_password = False
        db.commit()
        
        logger.info(f"✅ Contraseña cambiada exitosamente para {user.email}")
        
        return {"message": "Contraseña actualizada exitosamente"}
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error en change-password: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error al cambiar contraseña")


@router.post("/set-initial-password")
async def set_initial_password(
    payload: SetInitialPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Establece la contraseña inicial para usuarios existentes sin password.
    """
    try:
        email = payload.email.strip().lower()
        
        if len(payload.password) < 8:
            raise HTTPException(
                status_code=400,
                detail="La contraseña debe tener al menos 8 caracteres"
            )
        
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        if user.hashed_password:
            raise HTTPException(
                status_code=400,
                detail="El usuario ya tiene una contraseña configurada. Usa change-password."
            )
        
        # Establecer contraseña
        user.hashed_password = hash_password(payload.password)
        db.commit()
        
        logger.info(f"✅ Contraseña inicial configurada para {user.email}")
        
        return {"message": "Contraseña configurada exitosamente"}
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error en set-initial-password: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error al configurar contraseña inicial")


@router.post("/admin/reset-password")
async def admin_reset_password(
    payload: AdminResetPasswordRequest,
    admin_email: str = Query(..., description="Email del admin que realiza la operación"),
    db: Session = Depends(get_db),
):
    """
    Permite a un admin resetear la contraseña de cualquier usuario.
    Requiere autenticación admin.
    """
    try:
        # Verificar que el solicitante es admin
        admin = db.query(User).filter(User.email == admin_email.strip().lower()).first()
        if not admin:
            raise HTTPException(status_code=404, detail="Admin no encontrado")
        
        # Verificar que es admin
        is_admin = False
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and admin_email.strip().lower().endswith("@" + admin_domain.lower()):
            is_admin = True
        
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [
                e.strip().lower() for e in admin_emails.split(",") if e.strip()
            ]
            if admin_email.strip().lower() in admin_list:
                is_admin = True
        
        if not is_admin:
            raise HTTPException(status_code=403, detail="Solo administradores pueden resetear contraseñas")
        
        if len(payload.new_password) < 8:
            raise HTTPException(
                status_code=400,
                detail="La contraseña debe tener al menos 8 caracteres"
            )
        
        # Buscar usuario objetivo
        user_email = payload.user_email.strip().lower()
        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # Resetear contraseña y marcar como de 1 uso
        user.hashed_password = hash_password(payload.new_password)
        user.must_change_password = True  # Marcar como contraseña de 1 uso
        # Limpiar tokens de reset si existen
        user.password_reset_token = None
        user.password_reset_expires = None
        db.commit()
        
        logger.info(f"✅ Admin {admin_email} reseteó la contraseña de {user_email}")
        
        return {"message": f"Contraseña reseteada exitosamente para {user_email}"}
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error en admin/reset-password: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error al resetear contraseña")


@router.post("/admin/create-user")
async def admin_create_user(
    payload: AdminCreateUserRequest,
    admin_email: str = Query(..., description="Email del admin que realiza la operación"),
    db: Session = Depends(get_db),
):
    """
    Permite a un admin crear un nuevo usuario con contraseña de 1 uso.
    Requiere autenticación admin.
    """
    try:
        # Verificar que el solicitante es admin
        admin = db.query(User).filter(User.email == admin_email.strip().lower()).first()
        if not admin:
            raise HTTPException(status_code=404, detail="Admin no encontrado")
        
        # Verificar que es admin
        is_admin = False
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and admin_email.strip().lower().endswith("@" + admin_domain.lower()):
            is_admin = True
        
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [
                e.strip().lower() for e in admin_emails.split(",") if e.strip()
            ]
            if admin_email.strip().lower() in admin_list:
                is_admin = True
        
        if not is_admin:
            raise HTTPException(status_code=403, detail="Solo administradores pueden crear usuarios")
        
        email = payload.email.strip().lower()
        
        # Verificar que el usuario no exista
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="El usuario ya existe")
        
        if len(payload.password) < 8:
            raise HTTPException(
                status_code=400,
                detail="La contraseña debe tener al menos 8 caracteres"
            )
        
        # Crear nuevo usuario con contraseña de 1 uso
        display_name = payload.display_name or email.split("@")[0]
        
        new_user = User(
            email=email,
            display_name=display_name,
            microsoft_user_id=f"local_{email}",
            tenant_id="local",
            bot_display_name=settings.default_bot_name,
            is_active=True,
            hashed_password=hash_password(payload.password),
            must_change_password=True,  # Marcar como contraseña de 1 uso
            last_login_at=None,
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"✅ Admin {admin_email} creó usuario {email} con contraseña de 1 uso")
        
        return {
            "message": f"Usuario {email} creado exitosamente. Debe cambiar su contraseña en el primer login.",
            "user_id": new_user.id,
            "email": new_user.email,
            "display_name": new_user.display_name,
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error en admin/create-user: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error al crear usuario")


@router.get("/users/list")
async def list_users(
    response: Response,
    db: Session = Depends(get_db),
    user_email: str = Query(..., description="Email del usuario que solicita la lista (debe ser admin)"),
):
    """
    Lista todos los usuarios del sistema (solo para admins).
    
    Returns:
        Lista de usuarios con su información de admin y licencia.
    """
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    try:
        # Verificar que el usuario que solicita es admin
        email = user_email.strip().lower()
        is_admin = False
        
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email.endswith("@" + admin_domain.lower()):
            is_admin = True
        
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [
                e.strip().lower() for e in admin_emails.split(",") if e.strip()
            ]
            if email in admin_list:
                is_admin = True
        
        if not is_admin:
            raise HTTPException(status_code=403, detail="Solo los administradores pueden listar usuarios")
        
        # Obtener todos los usuarios
        users = db.query(User).all()
        
        # Importar función para determinar licencia
        from app.utils.license_utils import get_user_license_level
        
        # Construir lista de usuarios con información de admin y licencia
        users_list = []
        for user in users:
            user_email_lower = user.email.lower()
            
            # Determinar si es admin
            user_is_admin = False
            if admin_domain and user_email_lower.endswith("@" + admin_domain.lower()):
                user_is_admin = True
            
            if admin_emails:
                admin_list = [
                    e.strip().lower() for e in admin_emails.split(",") if e.strip()
                ]
                if user_email_lower in admin_list:
                    user_is_admin = True
            
            # Obtener nivel de licencia
            license_level = get_user_license_level(user)
            
            # Determinar si el usuario está online (último heartbeat en los últimos 30 segundos)
            is_online = False
            if user.last_heartbeat:  # type: ignore
                now = datetime.now(timezone.utc)
                # Asegurar que last_heartbeat sea timezone-aware
                last_heartbeat = user.last_heartbeat  # type: ignore
                if last_heartbeat.tzinfo is None:
                    last_heartbeat = last_heartbeat.replace(tzinfo=timezone.utc)
                time_diff = (now - last_heartbeat).total_seconds()
                is_online = time_diff <= 30  # Considerar online si el último heartbeat fue hace menos de 30 segundos
            
            # MEJORA: Verificar estado de webhooks de Outlook Calendar
            outlook_webhook_status = None
            outlook_webhook_expired = False
            if user.settings:  # type: ignore
                outlook_data = user.settings.get("outlook_calendar")  # type: ignore
                if outlook_data:
                    subscription_id = outlook_data.get("subscription_id")
                    subscription_expiration = outlook_data.get("subscription_expiration")
                    
                    if subscription_id:
                        # Hay suscripción activa, verificar si está expirada
                        if subscription_expiration:
                            try:
                                from datetime import timezone as tz
                                exp_str = str(subscription_expiration).replace("Z", "+00:00")
                                exp_dt = datetime.fromisoformat(exp_str)
                                if exp_dt.tzinfo is None:
                                    exp_dt = exp_dt.replace(tzinfo=tz.utc)
                                now_utc = datetime.now(tz.utc)
                                
                                if exp_dt < now_utc:
                                    outlook_webhook_expired = True
                                    outlook_webhook_status = "expirado"
                                else:
                                    days_until_expiry = (exp_dt - now_utc).total_seconds() / 86400
                                    if days_until_expiry < 1:
                                        outlook_webhook_status = f"expira_en_{int(days_until_expiry * 24)}h"
                                    else:
                                        outlook_webhook_status = f"expira_en_{int(days_until_expiry)}d"
                            except Exception as e:
                                logger.warning(f"Error verificando expiración de webhook Outlook para {user.email}: {e}")
                                # Si hay error parseando la fecha pero existe subscription_id, marcarlo como posiblemente expirado
                                outlook_webhook_status = "error_verificando"
                                outlook_webhook_expired = True  # Por seguridad, marcar como expirado si no se puede verificar
                        else:
                            # Si hay subscription_id pero no hay fecha de expiración, estado inconsistente
                            # Por seguridad, marcar como expirado para que se resincronice
                            outlook_webhook_status = "activo_sin_fecha"
                            outlook_webhook_expired = True
                    else:
                        # Outlook conectado pero sin subscription_id: no llegaran webhooks hasta
                        # que se resincronice (la sync crea la suscripcion si falta).
                        outlook_webhook_status = "no_configurado"
                        outlook_webhook_expired = True
            
            users_list.append({
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name or user.email.split("@")[0],
                "is_admin": user_is_admin,
                "is_premium": user.is_premium,
                "license": license_level,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
                "is_online": is_online,
                "last_heartbeat": user.last_heartbeat.isoformat() if user.last_heartbeat else None,  # type: ignore
                "outlook_webhook_status": outlook_webhook_status,
                "outlook_webhook_expired": outlook_webhook_expired,
            })
        
        return users_list
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error listando usuarios: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error al listar usuarios")


def _is_admin_email(email: str) -> bool:
    """Comprueba si el email pertenece a un administrador."""
    email = email.strip().lower()
    admin_domain = getattr(settings, "admin_email_domain", None)
    if admin_domain and email.endswith("@" + admin_domain.lower()):
        return True
    admin_emails = getattr(settings, "admin_emails", None)
    if admin_emails:
        admin_list = [e.strip().lower() for e in admin_emails.split(",") if e.strip()]
        if email in admin_list:
            return True
    return False


@router.post("/admin/sync-user-calendar")
async def admin_sync_user_calendar(
    admin_email: str = Query(..., description="Email del admin que solicita la accion"),
    target_user_email: str = Query(..., description="Email del usuario cuyo calendario se quiere sincronizar"),
    db: Session = Depends(get_db),
):
    """
    Sincronizar calendario de un usuario (solo admins).
    Ejecuta la misma logica que si el usuario pulsara Sincronizar en Ajustes (renovacion + sync).
    """
    if not _is_admin_email(admin_email):
        raise HTTPException(status_code=403, detail="Solo los administradores pueden sincronizar el calendario de otro usuario")
    target_email_normalized = target_user_email.strip().lower()
    target = db.query(User).filter(func.lower(User.email) == target_email_normalized).first()
    if not target:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    logger.info(
        "[ADMIN_SYNC_CALENDAR] Sincronizando calendario: target_user_email=%s, usuario=%s (id=%s)",
        target_user_email.strip(),
        target.email,
        target.id,
    )
    try:
        from app.services.calendar_sync_service import CalendarSyncService
        sync_service = CalendarSyncService()
        results = await sync_service.run_full_sync_with_renewal(target, db)
        return {"message": "Sincronizacion completada", "results": results}
    except Exception as e:
        logger.error("Error en admin sync calendario para %s: %s", target_user_email, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error sincronizando: {str(e)}")


@router.post("/heartbeat")
async def register_heartbeat(
    user_email: str = Query(..., description="Email del usuario que envía el heartbeat"),
    db: Session = Depends(get_db),
):
    """
    Registra un heartbeat del usuario para indicar que está online.
    
    Este endpoint debe ser llamado periódicamente (cada 15-20 segundos) por el frontend
    mientras el usuario está en la página. Si no se recibe un heartbeat en 30 segundos,
    el usuario se considera offline.
    """
    try:
        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # Actualizar last_heartbeat con la hora actual (UTC)
        now = datetime.now(timezone.utc)
        user.last_heartbeat = now  # type: ignore
        db.commit()
        
        # Log eliminado para no llenar la consola con mensajes de heartbeat
        
        return {"success": True, "timestamp": now.isoformat()}
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error registrando heartbeat para {user_email}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error al registrar heartbeat")


class UpdateUserRequest(BaseModel):
    """Request para actualizar un usuario."""
    display_name: str | None = None
    email: EmailStr | None = None
    is_premium: bool | None = None
    license: str | None = None  # 'basic', 'advanced', 'pro'
    # Nota: is_admin no se puede cambiar desde aquí, se determina por ADMIN_EMAILS en .env


class UpdateMyProfileRequest(BaseModel):
    """Request para actualizar el propio perfil del usuario."""
    display_name: str | None = None


@router.put("/users/me")
async def update_my_profile(
    payload: UpdateMyProfileRequest,
    user_email: str = Query(..., description="Email del usuario autenticado"),
    db: Session = Depends(get_db),
):
    """
    Permite a un usuario autenticado actualizar su propio perfil.
    Solo puede actualizar su display_name.
    """
    try:
        logger.info("📥 PUT /api/auth/users/me - Endpoint de perfil propio - Payload: %s", payload.model_dump())
        email = user_email.strip().lower()
        
        # Buscar el usuario
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # Actualizar display_name si se proporciona
        if payload.display_name is not None:
            user.display_name = payload.display_name
            db.commit()
            db.refresh(user)
            logger.info("✅ Usuario %s actualizó su display_name a: %s", email, payload.display_name)
        
        # Determinar si es admin
        is_admin = False
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email.endswith("@" + admin_domain.lower()):
            is_admin = True
        
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [
                e.strip().lower() for e in admin_emails.split(",") if e.strip()
            ]
            if email in admin_list:
                is_admin = True
        
        # Obtener nivel de licencia
        from app.utils.license_utils import get_user_license_level
        license_level = get_user_license_level(user)
        
        return {
            "user_id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "is_admin": is_admin,
            "is_premium": user.is_premium,
            "license": license_level,
            "must_change_password": user.must_change_password or False,
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error actualizando perfil propio: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error al actualizar perfil")


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    payload: UpdateUserRequest,
    db: Session = Depends(get_db),
    user_email: str = Query(..., description="Email del usuario que solicita la actualización (debe ser admin)"),
):
    """
    Actualiza un usuario (solo para admins).
    
    Permite actualizar:
    - display_name
    - email
    - is_premium (para cambiar licencia a advanced)
    - license (se convierte a is_premium si es 'advanced')
    """
    try:
        logger.info("📥 PUT /api/auth/users/%s - Payload recibido: %s", user_id, payload.model_dump())
        # Verificar que el usuario que solicita es admin
        email = user_email.strip().lower()
        is_admin = False
        
        admin_domain = getattr(settings, "admin_email_domain", None)
        if admin_domain and email.endswith("@" + admin_domain.lower()):
            is_admin = True
        
        admin_emails = getattr(settings, "admin_emails", None)
        if admin_emails:
            admin_list = [
                e.strip().lower() for e in admin_emails.split(",") if e.strip()
            ]
            if email in admin_list:
                is_admin = True
        
        if not is_admin:
            raise HTTPException(status_code=403, detail="Solo los administradores pueden actualizar usuarios")
        
        # Buscar el usuario a actualizar
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # Actualizar campos
        if payload.display_name is not None:
            user.display_name = payload.display_name
        
        if payload.email is not None:
            # Verificar que el email no esté en uso por otro usuario
            existing_user = db.query(User).filter(User.email == payload.email.lower(), User.id != user_id).first()
            if existing_user:
                raise HTTPException(status_code=400, detail="El email ya está en uso por otro usuario")
            user.email = payload.email.lower()
        
        # Actualizar licencia
        if payload.license is not None:
            logger.info("🔄 Actualizando licencia de usuario %s a: %s", user.email, payload.license)
            
            # Inicializar settings si no existe
            if user.settings is None:
                user.settings = {}
            elif not isinstance(user.settings, dict):
                # Si settings no es un dict, inicializarlo
                user.settings = {}
            
            if payload.license == "advanced":
                user.is_premium = True
                user.settings["license_level"] = "advanced"
                logger.info("✅ is_premium establecido a True y license_level='advanced' para usuario %s", user.email)
            elif payload.license == "basic":
                user.is_premium = False
                user.settings["license_level"] = "basic"
                logger.info("✅ is_premium establecido a False y license_level='basic' para usuario %s", user.email)
            elif payload.license == "pro":
                # Permitir asignar "pro" a usuarios no-admin guardándolo en settings
                user.settings["license_level"] = "pro"
                # Mantener is_premium como True para que tenga permisos avanzados
                user.is_premium = True
                logger.info("✅ license_level='pro' asignado a usuario %s (is_premium=True)", user.email)
            else:
                logger.warning("⚠️ Licencia desconocida: %s", payload.license)
            
            # Forzar que SQLAlchemy detecte el cambio en settings (JSON)
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(user, "settings")
            logger.info("📝 Settings actualizado: %s", user.settings)
            
            # Sincronizar permisos de MeetingAccess con la nueva licencia
            from app.utils.license_utils import get_user_license_level, get_meeting_access_permissions
            from app.models.meeting_access import MeetingAccess
            
            # Obtener la nueva licencia y permisos
            new_license_level = get_user_license_level(user)
            new_permissions = get_meeting_access_permissions(new_license_level)
            
            # Buscar todos los MeetingAccess del usuario
            user_accesses = db.query(MeetingAccess).filter(
                MeetingAccess.user_id == user.id
            ).all()
            
            if user_accesses:
                logger.info(
                    f"🔄 Sincronizando permisos de {len(user_accesses)} reuniones para usuario {user.email} "
                    f"(nueva licencia: {new_license_level})"
                )
                
                updated_count = 0
                for access in user_accesses:
                    # Actualizar permisos según la nueva licencia
                    if (access.can_view_transcript != new_permissions["can_view_transcript"] or
                        access.can_view_audio != new_permissions["can_view_audio"] or
                        access.can_view_video != new_permissions["can_view_video"]):
                        logger.info(
                            f"  📝 Actualizando permisos en meeting {access.meeting_id}: "
                            f"T={access.can_view_transcript}→{new_permissions['can_view_transcript']}, "
                            f"A={access.can_view_audio}→{new_permissions['can_view_audio']}, "
                            f"V={access.can_view_video}→{new_permissions['can_view_video']}"
                        )
                        access.can_view_transcript = new_permissions["can_view_transcript"]
                        access.can_view_audio = new_permissions["can_view_audio"]
                        access.can_view_video = new_permissions["can_view_video"]
                        updated_count += 1
                
                if updated_count > 0:
                    logger.info(f"✅ {updated_count} accesos a reuniones actualizados para usuario {user.email}")
                else:
                    logger.info(f"ℹ️ Todos los permisos ya estaban actualizados para usuario {user.email}")
        
        if payload.is_premium is not None:
            user.is_premium = payload.is_premium
        
        logger.info("💾 Guardando cambios de usuario %s: is_premium=%s, settings=%s", user.email, user.is_premium, user.settings)
        db.commit()
        db.refresh(user)
        logger.info("✅ Usuario guardado. is_premium=%s, settings=%s", user.is_premium, user.settings)
        
        # Determinar información actualizada
        from app.utils.license_utils import get_user_license_level
        
        user_email_lower = user.email.lower()
        user_is_admin = False
        if admin_domain and user_email_lower.endswith("@" + admin_domain.lower()):
            user_is_admin = True
        
        if admin_emails:
            admin_list = [
                e.strip().lower() for e in admin_emails.split(",") if e.strip()
            ]
            if user_email_lower in admin_list:
                user_is_admin = True
        
        license_level = get_user_license_level(user)
        
        logger.info("✅ Usuario actualizado: %s (id=%s)", user.email, user_id)
        
        return {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name or user.email.split("@")[0],
            "is_admin": user_is_admin,
            "is_premium": user.is_premium,
            "license": license_level,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error actualizando usuario: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Error al actualizar usuario")


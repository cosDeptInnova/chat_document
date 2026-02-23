"""Utilidades para determinar permisos según nivel de licencia."""
from app.models.user import User
from app.config import settings


def get_user_license_level(user: User) -> str:
    """
    Determina el nivel de licencia del usuario.
    
    Returns:
        'pro' si es admin o tiene licencia pro asignada, 'advanced' si es premium, 'basic' en caso contrario
    """
    # Primero verificar si tiene una licencia explícita guardada en settings
    if user.settings:
        # Si settings es un string (JSON serializado), parsearlo
        if isinstance(user.settings, str):
            import json
            try:
                user.settings = json.loads(user.settings)
            except (json.JSONDecodeError, TypeError):
                pass
        
        if isinstance(user.settings, dict):
            license_level = user.settings.get("license_level")
            if license_level in ["basic", "advanced", "pro"]:
                return license_level
    
    # Verificar si es admin (los admins siempre son "pro")
    admin_domain = getattr(settings, "admin_email_domain", None)
    if admin_domain and user.email.endswith("@" + admin_domain.lower()):
        return "pro"
    
    admin_emails = getattr(settings, "admin_emails", None)
    if admin_emails:
        admin_list = [
            e.strip().lower() for e in admin_emails.split(",") if e.strip()
        ]
        if user.email.lower() in admin_list:
            return "pro"
    
    # Verificar si es premium
    if user.is_premium:
        return "advanced"
    
    return "basic"


def get_meeting_access_permissions(license_level: str) -> dict:
    """
    Determina los permisos de acceso a una reunión según el nivel de licencia.
    
    Args:
        license_level: 'basic', 'advanced', o 'pro'
    
    Returns:
        Dict con can_view_transcript, can_view_audio, can_view_video
    """
    if license_level == "pro":
        return {
            "can_view_transcript": True,
            "can_view_audio": True,
            "can_view_video": True,
        }
    elif license_level == "advanced":
        return {
            "can_view_transcript": True,
            "can_view_audio": True,
            "can_view_video": False,
        }
    else:  # basic
        return {
            "can_view_transcript": True,
            "can_view_audio": False,
            "can_view_video": False,
        }


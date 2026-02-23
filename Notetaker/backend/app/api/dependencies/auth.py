"""Dependencies para autenticación y validación de tokens."""
from fastapi import HTTPException, Depends, Header
from typing import Optional
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.utils.jwt_utils import verify_token
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def is_user_admin(user: User) -> bool:
    """
    Determina si un usuario es administrador basándose en su email.
    
    Verifica si el email del usuario está en la lista de admin_emails
    o si pertenece al dominio admin_email_domain configurado.
    """
    if not user or not user.email:
        return False
    
    email_lower = user.email.strip().lower()
    
    # Verificar lista de emails admin
    admin_emails = getattr(settings, "admin_emails", None)
    if admin_emails:
        admin_emails_list = [
            e.strip().lower() for e in admin_emails.split(",") if e.strip()
        ]
        if email_lower in admin_emails_list:
            return True
    
    # Verificar dominio admin
    admin_domain = getattr(settings, "admin_email_domain", None)
    if admin_domain:
        if email_lower.endswith("@" + admin_domain.lower()):
            return True
    
    return False


async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency para obtener el usuario actual desde el token JWT.
    
    Valida el token en el header Authorization y devuelve el usuario correspondiente.
    Si el token no es válido o está expirado, lanza HTTPException 401.
    
    Uso:
        @router.get("/endpoint")
        async def my_endpoint(current_user: User = Depends(get_current_user)):
            ...
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Token de autenticación requerido",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extraer token del header "Bearer <token>"
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            raise ValueError("Esquema de autenticación inválido")
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Formato de token inválido. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verificar token
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Token inválido o expirado. Por favor, inicia sesión nuevamente.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Obtener usuario desde la base de datos
    user_id = payload.get("user_id")
    email = payload.get("email")
    
    if not user_id and not email:
        raise HTTPException(
            status_code=401,
            detail="Token inválido: falta información del usuario",
        )
    
    # Buscar usuario por ID o email
    user = None
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
    if not user and email:
        user = db.query(User).filter(User.email == email).first()
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Usuario no encontrado. Por favor, inicia sesión nuevamente.",
        )
    
    return user


async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Dependency opcional para obtener el usuario actual.
    
    Similar a get_current_user, pero no lanza excepción si no hay token.
    Devuelve None si no hay token o si es inválido.
    
    Uso:
        @router.get("/endpoint")
        async def my_endpoint(current_user: Optional[User] = Depends(get_current_user_optional)):
            if current_user:
                # Usuario autenticado
            else:
                # Usuario no autenticado
    """
    if not authorization:
        return None
    
    try:
        scheme, token = authorization.split(" ", 1)
        if scheme.lower() != "bearer":
            return None
    except ValueError:
        return None
    
    payload = verify_token(token)
    if not payload:
        return None
    
    user_id = payload.get("user_id")
    email = payload.get("email")
    
    if not user_id and not email:
        return None
    
    user = None
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
    if not user and email:
        user = db.query(User).filter(User.email == email).first()
    
    return user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency para verificar que el usuario actual es administrador.
    
    Uso:
        @router.get("/admin/endpoint")
        async def admin_endpoint(current_user: User = Depends(get_current_admin_user)):
            ...
    """
    if not is_user_admin(current_user):
        raise HTTPException(
            status_code=403,
            detail="Acceso denegado. Se requieren permisos de administrador.",
        )
    return current_user


"""Utilidades para generar y validar JWT tokens de autenticación."""
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Clave secreta para firmar tokens (usar variable de entorno o generar una)
JWT_SECRET_KEY = getattr(settings, "jwt_secret_key", None) or getattr(settings, "secret_key", "change-this-secret-key-in-production")
JWT_ALGORITHM = "HS256"

# Tiempos de expiración
TOKEN_EXPIRATION_DAYS = 1  # 1 día por defecto
REMEMBER_ME_EXPIRATION_DAYS = 30  # 30 días si "Recordarme" está activado


def generate_token(
    user_id: str,
    email: str,
    remember_me: bool = False
) -> str:
    """
    Genera un JWT token para un usuario.
    
    Args:
        user_id: ID del usuario
        email: Email del usuario
        remember_me: Si es True, el token expira en 30 días. Si es False, expira en 1 día.
    
    Returns:
        Token JWT como string
    """
    expiration_days = REMEMBER_ME_EXPIRATION_DAYS if remember_me else TOKEN_EXPIRATION_DAYS
    
    # Calcular tiempo de expiración
    now = datetime.now(timezone.utc)
    expiration = now + timedelta(days=expiration_days)
    
    # Payload del token
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "email": email,
        "exp": expiration,
        "iat": now,
        "remember_me": remember_me,
    }
    
    # Generar token
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    logger.info(
        f"Token generado para usuario {email} (user_id={user_id}), "
        f"expira en {expiration_days} días (remember_me={remember_me})"
    )
    
    return token


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verifica y decodifica un JWT token.
    
    Args:
        token: Token JWT a verificar
    
    Returns:
        Payload del token si es válido, None si es inválido o expirado
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expirado")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Token inválido: {e}")
        return None


def get_token_expiration(token: str) -> Optional[datetime]:
    """
    Obtiene la fecha de expiración de un token sin verificar su validez.
    Útil para verificar si un token está próximo a expirar.
    
    Args:
        token: Token JWT
    
    Returns:
        Fecha de expiración o None si no se puede decodificar
    """
    try:
        # Decodificar sin verificar (solo para obtener exp)
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM], options={"verify_signature": False})
        exp_timestamp = payload.get("exp")
        if exp_timestamp:
            return datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
        return None
    except Exception as e:
        logger.warning(f"Error obteniendo expiración del token: {e}")
        return None


def is_token_expiring_soon(token: str, hours_before: int = 24) -> bool:
    """
    Verifica si un token está próximo a expirar.
    
    Args:
        token: Token JWT
        hours_before: Horas antes de la expiración para considerar "próximo a expirar"
    
    Returns:
        True si el token expira en menos de hours_before horas
    """
    exp = get_token_expiration(token)
    if not exp:
        return True  # Si no se puede obtener, considerar como expirando
    
    now = datetime.now(timezone.utc)
    time_until_exp = (exp - now).total_seconds() / 3600  # Horas
    
    return time_until_exp < hours_before


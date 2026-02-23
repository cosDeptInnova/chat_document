"""Utilidades para manejo de contraseñas."""
import bcrypt
import secrets
from typing import Optional


def hash_password(password: str) -> str:
    """
    Hashea una contraseña usando bcrypt.
    
    Args:
        password: Contraseña en texto plano
        
    Returns:
        Hash bcrypt de la contraseña
    """
    # Generar salt y hashear
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifica si una contraseña coincide con su hash.
    
    Args:
        plain_password: Contraseña en texto plano
        hashed_password: Hash bcrypt almacenado
        
    Returns:
        True si la contraseña coincide, False en caso contrario
    """
    if not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False


def generate_reset_token() -> str:
    """
    Genera un token seguro para reset de contraseña.
    
    Returns:
        Token hexadecimal seguro de 32 bytes (64 caracteres)
    """
    return secrets.token_urlsafe(32)


#!/usr/bin/env python
"""
Script de consola para resetear la contraseña de un usuario admin.

Uso:
    python scripts/reset_admin_password.py <admin_email> <new_password>

Ejemplo:
    python scripts/reset_admin_password.py admin@example.com NuevaPassword123
"""
import sys
import os
from pathlib import Path

# Agregar el directorio backend al path para importar módulos
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Cargar variables de entorno antes de importar módulos
from app.config import load_env_file_utf8
load_env_file_utf8()

from app.database import SessionLocal
from app.models.user import User
from app.utils.password_utils import hash_password


def reset_admin_password(admin_email: str, new_password: str):
    """
    Resetea la contraseña de un usuario admin.
    
    Args:
        admin_email: Email del admin
        new_password: Nueva contraseña
    """
    if len(new_password) < 8:
        print("❌ Error: La contraseña debe tener al menos 8 caracteres")
        return False
    
    db = SessionLocal()
    try:
        email = admin_email.strip().lower()
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            print(f"❌ Error: Usuario {email} no encontrado")
            return False
        
        # Hashear y actualizar contraseña
        user.hashed_password = hash_password(new_password)
        # Limpiar tokens de reset si existen
        user.password_reset_token = None
        user.password_reset_expires = None
        
        db.commit()
        
        print(f"✅ Contraseña reseteada exitosamente para {email}")
        return True
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error al resetear contraseña: {e}")
        return False
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python scripts/reset_admin_password.py <admin_email> <new_password>")
        print("\nEjemplo:")
        print("  python scripts/reset_admin_password.py admin@example.com NuevaPassword123")
        sys.exit(1)
    
    admin_email = sys.argv[1]
    new_password = sys.argv[2]
    
    success = reset_admin_password(admin_email, new_password)
    sys.exit(0 if success else 1)


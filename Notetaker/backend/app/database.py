"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from urllib.parse import quote_plus, urlparse, urlunparse
import logging
from app.config import settings

logger = logging.getLogger(__name__)

# Asegurar que la URL de la base de datos esté correctamente codificada
def sanitize_database_url(url: str) -> str:
    """Sanitiza y codifica correctamente la URL de la base de datos."""
    try:
        # Intentar parsear la URL
        parsed = urlparse(url)
        
        # Si hay contraseña, codificarla
        if parsed.password:
            # Decodificar primero por si ya está codificada
            try:
                from urllib.parse import unquote_plus
                decoded_password = unquote_plus(parsed.password)
            except:
                decoded_password = parsed.password
            
            # Codificar la contraseña
            encoded_password = quote_plus(decoded_password)
            
            # Reconstruir la URL
            netloc = f"{parsed.username}:{encoded_password}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            
            sanitized_url = urlunparse((
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
            
            return sanitized_url
        else:
            return url
    except Exception as e:
        logger.warning(f"Error sanitizando URL de BD, usando URL original: {e}")
        return url

# Obtener y sanitizar la URL
database_url = sanitize_database_url(settings.database_url)

# Crear engine
engine = create_engine(
    database_url,
    pool_pre_ping=True,  # Verificar conexiones antes de usarlas
    pool_size=10,
    max_overflow=20,
    # Asegurar que la conexión use UTF-8
    connect_args={"options": "-c client_encoding=utf8"} if "postgresql" in database_url else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class para modelos
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency para FastAPI que retorna una sesión de DB.
    Usar en rutas como: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


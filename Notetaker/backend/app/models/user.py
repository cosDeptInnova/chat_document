"""User model."""
from sqlalchemy import Column, String, DateTime, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid


class User(Base):
    """Usuario que autoriza la aplicación."""
    
    __tablename__ = "users"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Microsoft/Azure AD info
    microsoft_user_id = Column(String, unique=True, nullable=False, index=True)  # oid del token
    tenant_id = Column(String, nullable=False, index=True)
    email = Column(String, nullable=False, index=True)
    display_name = Column(String)
    
    # Configuración del usuario
    bot_display_name = Column(String, default="Notetaker")  # Nombre con el que el bot se une a reuniones
    settings = Column(JSON, default={})  # Configuraciones adicionales
    
    # OAuth tokens (encriptados o referencias - NO guardar tokens en texto plano)
    # En producción, guardar en Redis o usar secrets manager
    access_token_encrypted = Column(String)  # Token encriptado (opcional, mejor usar Redis)
    refresh_token_encrypted = Column(String)  # Refresh token encriptado (opcional)
    token_expires_at = Column(DateTime)  # Fecha de expiración del access token
    
    # Estado
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)  # Para funcionalidades premium
    
    # Password fields
    hashed_password = Column(String, nullable=True)  # Hash bcrypt de la contraseña
    password_reset_token = Column(String, nullable=True, index=True)  # Token para recuperación de contraseña
    password_reset_expires = Column(DateTime, nullable=True)  # Fecha de expiración del token de reset
    must_change_password = Column(Boolean, default=False)  # Indica si el usuario debe cambiar su contraseña (de 1 uso)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login_at = Column(DateTime)
    last_heartbeat = Column(DateTime, nullable=True, index=True)  # Último heartbeat para tracking online/offline
    
    # Relación many-to-many con Meetings vía MeetingAccess
    meetings = relationship(
        "Meeting",
        secondary="meeting_access",
        back_populates="users",
        overlaps="access_list,meeting_access_list",
    )

    # Relación directa con registros de acceso (MeetingAccess)
    meeting_access_list = relationship(
        "MeetingAccess",
        back_populates="user",
        overlaps="meetings,access_list",
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, microsoft_user_id={self.microsoft_user_id})>"


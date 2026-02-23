"""MeetingAccess model for many-to-many relationship between Meetings and Users."""
from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid


class MeetingAccess(Base):
    """
    Relación many-to-many entre Meetings y Users con permisos granulares.
    
    Permite que múltiples usuarios accedan a la misma reunión con diferentes
    permisos según su licenciamiento.
    """
    
    __tablename__ = "meeting_access"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Permisos granulares según licenciamiento
    can_view_transcript = Column(Boolean, default=True, nullable=False)  # Acceso básico a transcripción
    can_view_audio = Column(Boolean, default=False, nullable=False)  # Acceso a audio (licencia premium)
    can_view_video = Column(Boolean, default=False, nullable=False)  # Acceso a video (licencia premium)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    # Nota: usamos overlaps para indicar a SQLAlchemy que estas relaciones
    # comparten las mismas columnas con las relaciones many-to-many
    # Meeting.users y User.meetings. Esto evita warnings de configuración
    # de mapeadores pero mantiene el comportamiento esperado.
    meeting = relationship(
        "Meeting",
        back_populates="access_list",
        # overlaps con:
        # - Meeting.users  (many-to-many)
        # - User.meetings  (many-to-many)
        # - User.meeting_access_list (relación directa)
        overlaps="users,meetings,meeting_access_list",
    )
    user = relationship(
        "User",
        back_populates="meeting_access_list",
        # overlaps con:
        # - User.meetings  (many-to-many)
        # - Meeting.users  (many-to-many)
        # - Meeting.access_list (relación directa)
        overlaps="meetings,users,access_list",
    )
    
    def __repr__(self):
        return f"<MeetingAccess(id={self.id}, meeting_id={self.meeting_id}, user_id={self.user_id}, transcript={self.can_view_transcript}, audio={self.can_view_audio}, video={self.can_view_video})>"


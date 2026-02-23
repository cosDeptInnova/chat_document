"""Meeting model."""
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid
import enum


class MeetingStatus(str, enum.Enum):
    """Estado de la reunión."""
    PENDING = "pending"  # Detectada pero no iniciada
    JOINING = "joining"  # En proceso de unirse (Playwright)
    IN_PROGRESS = "in_progress"  # Reunión en curso, capturando audio
    COMPLETED = "completed"  # Reunión finalizada, transcripción completa
    FAILED = "failed"  # Error al unirse o capturar
    CANCELLED = "cancelled"  # Reunión cancelada


class Meeting(Base):
    """Reunión de Teams detectada o creada."""
    
    __tablename__ = "meetings"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relación con usuario (nullable para soportar múltiples usuarios vía MeetingAccess)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)  # Cambiado a nullable
    user = relationship("User", foreign_keys=[user_id], backref="meetings_legacy")  # Mantener para compatibilidad
    
    # Relación many-to-many con usuarios vía MeetingAccess
    users = relationship(
        "User",
        secondary="meeting_access",
        back_populates="meetings",
        overlaps="access_list,meeting_access_list",
    )

    # Relación directa con registros de acceso (MeetingAccess)
    access_list = relationship(
        "MeetingAccess",
        back_populates="meeting",
        overlaps="users,meeting_access_list",
    )
    
    # Información de la reunión
    title = Column(String)  # Título de la reunión
    meeting_url = Column(Text, nullable=False)  # URL completa de Teams
    thread_id = Column(String, index=True)  # ID del thread de Teams (extraído de URL)
    organizer_email = Column(String)  # Email del organizador
    organizer_name = Column(String)  # Nombre del organizador
    
    # Fechas y horarios
    scheduled_start_time = Column(DateTime, nullable=False, index=True)  # Cuándo está programada
    scheduled_end_time = Column(DateTime)
    actual_start_time = Column(DateTime)  # Cuándo realmente empezó (bot se unió)
    actual_end_time = Column(DateTime)  # Cuándo terminó
    
    # Estado y metadata
    status = Column(SQLEnum(MeetingStatus), default=MeetingStatus.PENDING, nullable=False, index=True)
    extra_metadata = Column(JSON, default={})  # Información adicional (participantes, etc.) - "metadata" es reservado en SQLAlchemy
    
    # Archivos de audio/video (si se graban)
    audio_file_path = Column(String)  # Ruta al archivo de audio grabado
    video_file_path = Column(String)  # Ruta al archivo de video (si se graba)
    storage_type = Column(String, default="local")  # local, s3, etc.

    # Integración con Recall.ai
    recall_bot_id = Column(String, index=True)  # ID del bot en Recall.ai
    recall_status = Column(String)  # processing, active, stopped, failed, etc.
    recall_storage_plan = Column(String, default="none")  # none, retention_based, default
    recall_metadata = Column(JSON, default={})  # Respuesta/raw de Recall para debug/metadata
    
    # Celery task tracking
    celery_task_id = Column(String, index=True)  # ID de la tarea Celery programada para cancelarla si es necesario
    transcript_task_id = Column(String, index=True)  # ID de la tarea Celery programada para obtener transcripción
    transcript_scheduled_time = Column(DateTime, index=True)  # Timestamp de cuándo se programó la obtención de transcripción
    
    # Error tracking
    error_message = Column(Text)  # Si falló, el mensaje de error
    
    # Soft delete
    deleted_at = Column(DateTime, nullable=True, index=True)  # Fecha de eliminación (soft delete)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Meeting(id={self.id}, title={self.title}, status={self.status})>"


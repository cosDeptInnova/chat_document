"""Transcription models."""
from sqlalchemy import Column, String, DateTime, ForeignKey, Float, Text, Integer, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid


class Transcription(Base):
    """Transcripción completa de una reunión."""
    
    __tablename__ = "transcriptions"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relación con reunión
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False, unique=True, index=True)
    meeting = relationship("Meeting", backref="transcription", uselist=False)
    
    # Metadata de la transcripción
    language = Column(String, default="es")  # Idioma detectado
    confidence_score = Column(Float)  # Score general de confianza (0-1)
    
    # JSON raw original de Recall.ai (guardado antes de procesar)
    raw_transcript_json = Column(JSON, nullable=True)  # JSON completo tal como viene de Recall.ai
    
    # Estadísticas
    total_segments = Column(Integer, default=0)
    total_duration_seconds = Column(Float, default=0.0)
    
    # Estado
    is_final = Column(Boolean, default=False)  # Si la transcripción está completa
    is_processed = Column(Boolean, default=False)  # Si ya se generó el resumen
    
    # Timestamps
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relación con segmentos
    segments = relationship("TranscriptionSegment", back_populates="transcription", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Transcription(id={self.id}, meeting_id={self.meeting_id}, segments={self.total_segments})>"


class TranscriptionSegment(Base):
    """Segmento individual de la transcripción (palabra/frase por speaker)."""
    
    __tablename__ = "transcription_segments"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relación con transcripción
    transcription_id = Column(String, ForeignKey("transcriptions.id"), nullable=False, index=True)
    transcription = relationship("Transcription", back_populates="segments")
    
    # Speaker identification
    speaker_id = Column(String, nullable=False, index=True)  # Speaker:0, Speaker:1, etc. (de Deepgram)
    speaker_name = Column(String, index=True)  # Nombre real si se mapeó desde Teams subtitles
    
    # Contenido
    text = Column(Text, nullable=False)  # Texto transcrito
    language = Column(String, default="es")
    
    # Timing
    start_time = Column(Float, nullable=False, index=True)  # Segundos desde inicio de reunión
    end_time = Column(Float, nullable=False)
    duration = Column(Float)  # end_time - start_time
    
    # Metadata
    confidence = Column(Float)  # Confianza del segmento (0-1)
    words = Column(JSON)  # Array de palabras con timing individual (de Deepgram)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<TranscriptionSegment(id={self.id}, speaker={self.speaker_name or self.speaker_id}, text={self.text[:50]}...)>"


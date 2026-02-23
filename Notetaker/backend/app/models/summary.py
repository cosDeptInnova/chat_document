"""Summary model."""
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, JSON, Boolean, Float, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid


class Summary(Base):
    """Resumen y análisis de una reunión generado por LLM."""
    
    __tablename__ = "summaries"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relación con reunión
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False, unique=True, index=True)
    meeting = relationship("Meeting", backref="summary", uselist=False)
    
    # Relación con transcripción (opcional, para referencia)
    transcription_id = Column(String, ForeignKey("transcriptions.id"), nullable=True)
    transcription = relationship("Transcription")
    
    # Resumen
    summary_text = Column(Text)  # Resumen general de la reunión
    summary_json = Column(JSON)  # JSON estructurado para tu IA (formato personalizado)
    
    # Nuevos campos para respuesta de IA Cosmos
    toon = Column(Text)  # Brief en markdown (extraído para consulta rápida)
    ia_response_json = Column(JSONB)  # Respuesta COMPLETA de la IA (JSONB para PostgreSQL)
    
    # Análisis adicional
    key_points = Column(JSON)  # Array de puntos clave
    action_items = Column(JSON)  # Array de action items
    participants_summary = Column(JSON)  # Resumen de participantes y sus contribuciones
    sentiment_analysis = Column(JSON)  # Análisis de sentimiento (si aplica)
    
    # Metadata
    llm_model = Column(String)  # Modelo usado (ej: "gpt-4", "claude-3", "custom")
    llm_service = Column(String)  # Servicio usado (ej: "openai", "anthropic", "custom")
    processing_time_seconds = Column(Float)  # Tiempo que tardó en procesar
    
    # Control de procesamiento
    processing_status = Column(String, default="pending", index=True)  # pending/processing/completed/failed
    processing_started_at = Column(DateTime)  # Cuando empezó el procesamiento
    retry_count = Column(Integer, default=0)  # Número de reintentos realizados
    
    # Estado
    is_final = Column(Boolean, default=False)  # Si el resumen está completo
    error_message = Column(Text)  # Si hubo error al generar
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    completed_at = Column(DateTime)
    
    def __repr__(self):
        return f"<Summary(id={self.id}, meeting_id={self.meeting_id}, status={self.processing_status})>"


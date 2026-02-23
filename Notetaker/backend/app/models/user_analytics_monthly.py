"""User Analytics Monthly model - Historial de métricas mensuales."""
from sqlalchemy import Column, String, Integer, DateTime, Numeric, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base
import uuid


class UserAnalyticsMonthly(Base):
    """Métricas mensuales agregadas por usuario para analíticas históricas."""
    
    __tablename__ = "user_analytics_monthly"
    
    # Primary key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Relación con usuario
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    user = relationship("User", backref="analytics_monthly")
    
    # Período
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)  # 1-12
    
    # Métricas agregadas
    meetings_count = Column(Integer, default=0, nullable=False)
    total_hours = Column(Numeric(10, 2), default=0, nullable=False)
    total_talk_time_seconds = Column(Numeric(12, 2), default=0, nullable=False)
    
    # Participación
    average_participation_percent = Column(Numeric(5, 2), default=0, nullable=False)
    driver_count = Column(Integer, default=0, nullable=False)
    contributor_count = Column(Integer, default=0, nullable=False)
    average_responsivity = Column(Numeric(5, 2), default=0, nullable=False)
    
    # Calidad
    average_collaboration = Column(Numeric(5, 2), default=0, nullable=False)
    average_decisiveness = Column(Numeric(5, 2), default=0, nullable=False)
    average_conflict = Column(Numeric(5, 2), default=0, nullable=False)
    average_engagement = Column(Numeric(5, 2), default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    __table_args__ = (
        UniqueConstraint('user_id', 'year', 'month', name='unique_user_month'),
    )
    
    def __repr__(self):
        return f"<UserAnalyticsMonthly(user_id={self.user_id}, {self.year}-{self.month:02d}, meetings={self.meetings_count})>"

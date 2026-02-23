"""Rutas para Analytics - Métricas personales del usuario."""
import re
from calendar import monthrange
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel
from collections import defaultdict

from app.database import get_db
from app.models.user import User
from app.models.meeting import Meeting
from app.models.summary import Summary
from app.models.meeting_access import MeetingAccess
from app.models.user_analytics_monthly import UserAnalyticsMonthly

# Roles que se consideran "driver" (EN y ES). El resto se trata como contributor.
_DRIVER_ROLES = frozenset({
    "driver", "co-driver", "conductor", "coconductor", "co-conductor",
})
_CONTRIBUTOR_ROLES = frozenset({
    "contributor", "contribuidor", "assistant", "asistente",
})


def _is_driver_role(role: str) -> bool:
    """True si el rol (EN o ES) corresponde a Driver/Co-driver/Conductor."""
    if not role or not isinstance(role, str):
        return False
    r = role.strip().lower()
    return r in _DRIVER_ROLES


def _user_matches_speaker(user_email: str, display_name: Optional[str], speaker: str) -> bool:
    """True si el speaker corresponde al usuario (por email o display_name)."""
    if not speaker:
        return False
    speaker_lower = speaker.lower()
    email_part = user_email.split("@")[0].lower()
    if email_part and email_part in speaker_lower:
        return True
    if display_name and display_name.strip() and display_name.lower() in speaker_lower:
        return True
    return False


def _extract_role_from_toon(toon: str, speaker: str) -> Optional[str]:
    """Extrae la palabra del rol desde el toon (ej. 'Driver', 'Conductor')."""
    # Patron: **Nombre**: Rol —  o  Nombre: Rol ·
    pattern = rf"\*\*{re.escape(speaker)}\*\*\s*:\s*([^\s—·]+)"
    match = re.search(pattern, toon)
    if match:
        return match.group(1).strip()
    pattern_alt = rf"{re.escape(speaker)}\s*:\s*([^\s—·]+)"
    match = re.search(pattern_alt, toon)
    if match:
        return match.group(1).strip()
    return None


router = APIRouter()


class ParticipationStats(BaseModel):
    """Estadísticas de participación del usuario."""
    total_talk_time_seconds: float
    average_participation_percent: float
    driver_count: int
    contributor_count: int
    average_responsivity: float


class QualityMetrics(BaseModel):
    """Métricas de calidad de reuniones."""
    average_collaboration: float
    average_decisiveness: float
    average_conflict: float
    average_engagement: float


class MonthlyComparison(BaseModel):
    """Comparación con mes anterior."""
    meetings_change_percent: float
    hours_change_percent: float
    participation_change_percent: float


class TopCollaborator(BaseModel):
    """Persona con la que más se reúne."""
    name: str
    meeting_count: int
    average_collaboration: float


class PatternInsight(BaseModel):
    """Patrón detectado."""
    type: str  # 'best_hour', 'best_day', 'optimal_duration'
    value: str
    detail: str


class UserAnalyticsResponse(BaseModel):
    """Respuesta completa de analytics del usuario."""
    # Resumen mensual
    meetings_this_month: int
    meetings_last_month: int
    total_hours_this_month: float
    total_hours_last_month: float
    
    # Participación
    participation: Optional[ParticipationStats]
    
    # Calidad de reuniones
    quality: Optional[QualityMetrics]
    
    # Comparación mensual
    comparison: Optional[MonthlyComparison]
    
    # Top colaboradores
    top_collaborators: list[TopCollaborator]
    
    # Patrones
    patterns: list[PatternInsight]
    
    # Sugerencias personalizadas
    suggestions: list[str]


@router.get("/api/analytics/user", response_model=UserAnalyticsResponse)
async def get_user_analytics(
    user_id: str = Query(..., description="ID del usuario"),
    db: Session = Depends(get_db)
):
    """
    Obtiene métricas personales de analytics para un usuario.
    """
    # Verificar que el usuario existe
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    user_email = user.email
    
    # Fechas para cálculos
    now = datetime.utcnow()
    first_day_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    first_day_last_month = (first_day_this_month - timedelta(days=1)).replace(day=1)

    # Rango de comparación: mismo periodo del mes anterior (del día 1 hasta el mismo día que hoy)
    days_in_last_month = monthrange(first_day_last_month.year, first_day_last_month.month)[1]
    target_day_last_month = min(now.day, days_in_last_month)
    comparison_cutoff_last_month = first_day_last_month.replace(
        day=target_day_last_month,
        hour=now.hour,
        minute=now.minute,
        second=now.second,
        microsecond=now.microsecond,
    )
    
    # === REUNIONES DEL USUARIO ===
    # Obtener IDs de reuniones a las que tiene acceso vía MeetingAccess
    access_records = db.query(MeetingAccess).filter(
        MeetingAccess.user_id == user_id
    ).all()
    meeting_ids_with_access = [a.meeting_id for a in access_records]
    
    # Reuniones legacy donde el usuario es owner (user_id)
    own_meetings_by_user_id = db.query(Meeting).filter(
        and_(
            Meeting.user_id == user_id,
            Meeting.status == 'completed'
        )
    ).all()
    
    # Reuniones donde el usuario es organizador
    own_meetings_by_organizer = db.query(Meeting).filter(
        and_(
            Meeting.organizer_email == user_email,
            Meeting.status == 'completed'
        )
    ).all()
    
    own_meeting_ids = [m.id for m in own_meetings_by_user_id] + [m.id for m in own_meetings_by_organizer]
    all_meeting_ids = list(set(own_meeting_ids + meeting_ids_with_access))
    
    # Obtener todas las reuniones completadas del usuario
    meetings = db.query(Meeting).filter(
        and_(
            Meeting.id.in_(all_meeting_ids),
            Meeting.status == 'completed'
        )
    ).all()
    
    # === REUNIONES ESTE MES Y MES PASADO (mismo periodo) ===
    # Este mes: desde el día 1 hasta ahora
    meetings_this_month = [
        m
        for m in meetings
        if m.scheduled_start_time
        and first_day_this_month <= m.scheduled_start_time <= now
    ]
    # Mes anterior: mismo rango de días (del 1 hasta el mismo día que hoy)
    meetings_last_month = [
        m
        for m in meetings
        if m.scheduled_start_time
        and first_day_last_month <= m.scheduled_start_time <= comparison_cutoff_last_month
    ]
    
    # Calcular horas
    def calculate_hours(meeting_list):
        total_seconds = 0
        for m in meeting_list:
            # Buscar summary para obtener duración real
            summary = db.query(Summary).filter(Summary.meeting_id == m.id).first()
            if summary and summary.ia_response_json:
                insights = summary.ia_response_json.get('insights', {})
                talk_times = insights.get('talk_time_seconds', {})
                if talk_times:
                    total_seconds += sum(talk_times.values())
        return total_seconds / 3600  # Convertir a horas
    
    total_hours_this_month = calculate_hours(meetings_this_month)
    total_hours_last_month = calculate_hours(meetings_last_month)
    
    # === PARTICIPACIÓN ===
    participation_stats = None
    total_talk_time = 0
    participation_percents = []
    driver_count = 0
    contributor_count = 0
    responsivities = []
    
    for meeting in meetings:
        summary = db.query(Summary).filter(Summary.meeting_id == meeting.id).first()
        if not summary or not summary.ia_response_json:
            continue
        
        insights = summary.ia_response_json.get('insights', {})
        toon = summary.toon or ''
        participation = insights.get('participation_percent', {})
        talk_time = insights.get('talk_time_seconds', {})
        participant_roles = insights.get('participant_roles') or []

        # Prioridad 1: usar participant_roles si viene rellenado
        role_counted_from_participant_roles = False
        if isinstance(participant_roles, list) and len(participant_roles) > 0:
            for entry in participant_roles:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or ""
                if not _user_matches_speaker(user_email, user.display_name, name):
                    continue
                role_counted_from_participant_roles = True
                role_str = entry.get("role") or ""
                if _is_driver_role(role_str):
                    driver_count += 1
                else:
                    contributor_count += 1
                # Participacion y talk_time: buscar en participation/talk_time por nombre
                if name in participation:
                    participation_percents.append(participation[name])
                if name in talk_time:
                    total_talk_time += talk_time[name]
                # Responsividad: buscar en toon (Responsividad o resp XX/100)
                pattern = rf"{re.escape(name)}.*?(?:Responsividad|resp)\s*(\d+)/100"
                match = re.search(pattern, toon, re.IGNORECASE | re.DOTALL)
                if match:
                    responsivities.append(int(match.group(1)))
                break

        # Fallback: parsear toon (roles en ingles o castellano)
        if not role_counted_from_participant_roles:
            for speaker, percent in participation.items():
                if not _user_matches_speaker(user_email, user.display_name, speaker):
                    continue
                participation_percents.append(percent)
                if speaker in talk_time:
                    total_talk_time += talk_time[speaker]
                # Rol: extraer del toon (Driver, Conductor, Contributor, etc.)
                role_str = _extract_role_from_toon(toon, speaker)
                if role_str and _is_driver_role(role_str):
                    driver_count += 1
                else:
                    contributor_count += 1
            # Responsividad: Responsividad XX/100 o resp XX/100
            for speaker in participation.keys():
                if not _user_matches_speaker(user_email, user.display_name, speaker):
                    continue
                pattern = rf"{re.escape(speaker)}.*?(?:Responsividad|resp)\s*(\d+)/100"
                match = re.search(pattern, toon, re.IGNORECASE | re.DOTALL)
                if match:
                    responsivities.append(int(match.group(1)))
    
    if participation_percents:
        participation_stats = ParticipationStats(
            total_talk_time_seconds=total_talk_time,
            average_participation_percent=sum(participation_percents) / len(participation_percents),
            driver_count=driver_count,
            contributor_count=contributor_count,
            average_responsivity=sum(responsivities) / len(responsivities) if responsivities else 0
        )
    
    # === CALIDAD DE REUNIONES ===
    quality_metrics = None
    collaborations = []
    decisivenesses = []
    conflicts = []
    
    for meeting in meetings:
        summary = db.query(Summary).filter(Summary.meeting_id == meeting.id).first()
        if not summary or not summary.ia_response_json:
            continue
        
        insights = summary.ia_response_json.get('insights', {})
        
        if insights.get('collaboration', {}).get('score_0_100'):
            collaborations.append(insights['collaboration']['score_0_100'])
        if insights.get('decisiveness', {}).get('score_0_100'):
            decisivenesses.append(insights['decisiveness']['score_0_100'])
        if insights.get('conflict_level_0_100') is not None:
            conflicts.append(insights['conflict_level_0_100'])
    
    if collaborations or decisivenesses or conflicts:
        avg_collab = sum(collaborations) / len(collaborations) if collaborations else 0
        avg_decis = sum(decisivenesses) / len(decisivenesses) if decisivenesses else 0
        avg_conflict = sum(conflicts) / len(conflicts) if conflicts else 0
        avg_engagement = (avg_collab + avg_decis) / 2 if (collaborations and decisivenesses) else 0
        
        quality_metrics = QualityMetrics(
            average_collaboration=round(avg_collab, 1),
            average_decisiveness=round(avg_decis, 1),
            average_conflict=round(avg_conflict, 1),
            average_engagement=round(avg_engagement, 1)
        )
    
    # === COMPARACIÓN MENSUAL ===
    # Siempre devolver comparación, incluso si el mes anterior tuvo 0 reuniones/horas.
    prev_meetings = len(meetings_last_month)
    curr_meetings = len(meetings_this_month)
    if prev_meetings == 0:
        if curr_meetings == 0:
            meetings_change = 0.0
        else:
            # Sin reuniones en el periodo anterior y ahora sí hay: crecimiento positivo.
            meetings_change = 100.0
    else:
        meetings_change = ((curr_meetings - prev_meetings) / prev_meetings) * 100

    prev_hours = total_hours_last_month
    curr_hours = total_hours_this_month
    if prev_hours == 0:
        if curr_hours == 0:
            hours_change = 0.0
        else:
            hours_change = 100.0
    else:
        hours_change = ((curr_hours - prev_hours) / prev_hours) * 100
    
    # Para participación, comparar promedios (pendiente de implementar)
    participation_change = 0.0
    
    comparison = MonthlyComparison(
        meetings_change_percent=round(meetings_change, 1),
        hours_change_percent=round(hours_change, 1),
        participation_change_percent=round(participation_change, 1)
    )
    
    # === TOP COLABORADORES ===
    collaborator_counts = {}
    for meeting in meetings:
        summary = db.query(Summary).filter(Summary.meeting_id == meeting.id).first()
        if not summary or not summary.ia_response_json:
            continue
        
        insights = summary.ia_response_json.get('insights', {})
        participation = insights.get('participation_percent', {})
        
        for speaker in participation.keys():
            # Excluir al propio usuario
            if user_email.split('@')[0].lower() in speaker.lower():
                continue
            if user.display_name and user.display_name.lower() in speaker.lower():
                continue
            
            if speaker not in collaborator_counts:
                collaborator_counts[speaker] = {'count': 0, 'collaborations': []}
            collaborator_counts[speaker]['count'] += 1
            
            collab = insights.get('collaboration', {}).get('score_0_100')
            if collab:
                collaborator_counts[speaker]['collaborations'].append(collab)
    
    top_collaborators = []
    sorted_collaborators = sorted(collaborator_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
    for name, data in sorted_collaborators:
        avg_collab = sum(data['collaborations']) / len(data['collaborations']) if data['collaborations'] else 0
        top_collaborators.append(TopCollaborator(
            name=name,
            meeting_count=data['count'],
            average_collaboration=round(avg_collab, 1)
        ))
    
    # === PATRONES ===
    patterns = []
    
    # Mejor hora del día
    hour_engagement = {}
    for meeting in meetings:
        if not meeting.scheduled_start_time:
            continue
        hour = meeting.scheduled_start_time.hour
        summary = db.query(Summary).filter(Summary.meeting_id == meeting.id).first()
        if summary and summary.ia_response_json:
            insights = summary.ia_response_json.get('insights', {})
            collab = insights.get('collaboration', {}).get('score_0_100', 0)
            decis = insights.get('decisiveness', {}).get('score_0_100', 0)
            engagement = (collab + decis) / 2 if collab and decis else 0
            if hour not in hour_engagement:
                hour_engagement[hour] = []
            hour_engagement[hour].append(engagement)
    
    if hour_engagement:
        best_hour = max(hour_engagement.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)
        if best_hour[1]:
            patterns.append(PatternInsight(
                type='best_hour',
                value=f"{best_hour[0]:02d}:00",
                detail=f"Tus reuniones a las {best_hour[0]:02d}:00 tienen mejor engagement ({sum(best_hour[1])/len(best_hour[1]):.0f}%)"
            ))
    
    # Mejor día de la semana
    day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    day_counts = {}
    for meeting in meetings:
        if meeting.scheduled_start_time:
            day = meeting.scheduled_start_time.weekday()
            day_counts[day] = day_counts.get(day, 0) + 1
    
    if day_counts:
        most_common_day = max(day_counts.items(), key=lambda x: x[1])
        patterns.append(PatternInsight(
            type='busiest_day',
            value=day_names[most_common_day[0]],
            detail=f"Tu día con más reuniones es {day_names[most_common_day[0]]} ({most_common_day[1]} reuniones)"
        ))
    
    # === SUGERENCIAS ===
    suggestions = []
    
    if participation_stats:
        if participation_stats.average_participation_percent < 20:
            suggestions.append("Tu participación promedio es baja (<20%). Intenta intervenir más en las reuniones.")
        elif participation_stats.average_participation_percent > 60:
            suggestions.append("Dominas muchas reuniones (>60% del tiempo). Considera dar más espacio a otros participantes.")
        
        if participation_stats.average_responsivity < 50:
            suggestions.append("Tu responsividad es baja. Intenta responder más activamente a otros participantes.")
        
        if participation_stats.contributor_count > participation_stats.driver_count * 2:
            suggestions.append("Sueles ser Contributor más que Driver. Si quieres liderar más, toma la iniciativa en las reuniones.")
    
    if quality_metrics:
        if quality_metrics.average_decisiveness < 50:
            suggestions.append("Tus reuniones tienen baja decisividad. Intenta definir acciones claras al final de cada reunión.")
        if quality_metrics.average_conflict > 40:
            suggestions.append("Hay niveles de conflicto elevados en tus reuniones. Considera técnicas de facilitación.")
    
    if not suggestions:
        suggestions.append("¡Buen trabajo! Tus métricas de reuniones están en buen nivel.")
    
    return UserAnalyticsResponse(
        meetings_this_month=len(meetings_this_month),
        meetings_last_month=len(meetings_last_month),
        total_hours_this_month=round(total_hours_this_month, 1),
        total_hours_last_month=round(total_hours_last_month, 1),
        participation=participation_stats,
        quality=quality_metrics,
        comparison=comparison,
        top_collaborators=top_collaborators,
        patterns=patterns,
        suggestions=suggestions
    )


# ========== MODELOS PARA HISTÓRICO ==========

class MonthlyMetricsResponse(BaseModel):
    """Métricas de un mes específico."""
    year: int
    month: int
    meetings_count: int
    total_hours: float
    average_participation_percent: float
    average_collaboration: float
    average_decisiveness: float
    average_conflict: float
    average_engagement: float


class UserAnalyticsHistoryResponse(BaseModel):
    """Respuesta con histórico de métricas."""
    monthly_metrics: list[MonthlyMetricsResponse]
    total_months: int


# ========== FUNCIÓN HELPER PARA GUARDAR MÉTRICAS MENSUALES ==========

def save_monthly_analytics(
    db: Session,
    user_id: str,
    year: int,
    month: int,
    meetings_count: int,
    total_hours: float,
    participation_stats: Optional[ParticipationStats],
    quality_metrics: Optional[QualityMetrics]
):
    """
    Guarda o actualiza métricas mensuales en la tabla user_analytics_monthly.
    """
    existing = db.query(UserAnalyticsMonthly).filter(
        and_(
            UserAnalyticsMonthly.user_id == user_id,
            UserAnalyticsMonthly.year == year,
            UserAnalyticsMonthly.month == month
        )
    ).first()
    
    data = {
        'meetings_count': meetings_count,
        'total_hours': total_hours,
        'total_talk_time_seconds': float(participation_stats.total_talk_time_seconds) if participation_stats else 0,
        'average_participation_percent': float(participation_stats.average_participation_percent) if participation_stats else 0,
        'driver_count': participation_stats.driver_count if participation_stats else 0,
        'contributor_count': participation_stats.contributor_count if participation_stats else 0,
        'average_responsivity': float(participation_stats.average_responsivity) if participation_stats else 0,
        'average_collaboration': float(quality_metrics.average_collaboration) if quality_metrics else 0,
        'average_decisiveness': float(quality_metrics.average_decisiveness) if quality_metrics else 0,
        'average_conflict': float(quality_metrics.average_conflict) if quality_metrics else 0,
        'average_engagement': float(quality_metrics.average_engagement) if quality_metrics else 0,
    }
    
    if existing:
        # Actualizar registro existente
        for key, value in data.items():
            setattr(existing, key, value)
        existing.updated_at = datetime.utcnow()
    else:
        # Crear nuevo registro
        monthly = UserAnalyticsMonthly(
            user_id=user_id,
            year=year,
            month=month,
            **data
        )
        db.add(monthly)
    
    db.commit()


# ========== ENDPOINT PARA HISTÓRICO ==========

@router.get("/api/analytics/user/history", response_model=UserAnalyticsHistoryResponse)
async def get_user_analytics_history(
    user_id: str = Query(..., description="ID del usuario"),
    months: int = Query(12, description="Número de meses a retornar (default: 12)"),
    db: Session = Depends(get_db)
):
    """
    Obtiene histórico de métricas mensuales del usuario.
    Retorna los últimos N meses de datos agregados.
    Si no hay datos en user_analytics_monthly, los calcula sobre la marcha.
    """
    # Verificar que el usuario existe
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    user_email = user.email
    
    # Calcular fecha de inicio (hace N meses)
    now = datetime.utcnow()
    start_year = now.year
    start_month = now.month - months + 1
    
    # Ajustar si start_month es negativo
    while start_month <= 0:
        start_month += 12
        start_year -= 1
    
    # Consultar métricas mensuales de la tabla
    monthly_records = db.query(UserAnalyticsMonthly).filter(
        and_(
            UserAnalyticsMonthly.user_id == user_id,
            # Filtrar por año/mes
            or_(
                (UserAnalyticsMonthly.year == start_year) & (UserAnalyticsMonthly.month >= start_month),
                (UserAnalyticsMonthly.year > start_year)
            )
        )
    ).order_by(UserAnalyticsMonthly.year, UserAnalyticsMonthly.month).all()
    
    # Si hay datos en la tabla, usarlos
    if monthly_records:
        monthly_metrics = []
        for record in monthly_records:
            monthly_metrics.append(MonthlyMetricsResponse(
                year=record.year,
                month=record.month,
                meetings_count=record.meetings_count,
                total_hours=float(record.total_hours),
                average_participation_percent=float(record.average_participation_percent),
                average_collaboration=float(record.average_collaboration),
                average_decisiveness=float(record.average_decisiveness),
                average_conflict=float(record.average_conflict),
                average_engagement=float(record.average_engagement)
            ))
        
        return UserAnalyticsHistoryResponse(
            monthly_metrics=monthly_metrics,
            total_months=len(monthly_metrics)
        )
    
    # Si NO hay datos en la tabla, calcular sobre la marcha desde las reuniones
    # === OBTENER REUNIONES DEL USUARIO ===
    access_records = db.query(MeetingAccess).filter(
        MeetingAccess.user_id == user_id
    ).all()
    meeting_ids_with_access = [a.meeting_id for a in access_records]
    
    own_meetings_by_user_id = db.query(Meeting).filter(
        and_(
            Meeting.user_id == user_id,
            Meeting.status == 'completed'
        )
    ).all()
    
    own_meetings_by_organizer = db.query(Meeting).filter(
        and_(
            Meeting.organizer_email == user_email,
            Meeting.status == 'completed'
        )
    ).all()
    
    own_meeting_ids = [m.id for m in own_meetings_by_user_id] + [m.id for m in own_meetings_by_organizer]
    all_meeting_ids = list(set(own_meeting_ids + meeting_ids_with_access))
    
    meetings = db.query(Meeting).filter(
        and_(
            Meeting.id.in_(all_meeting_ids),
            Meeting.status == 'completed',
            Meeting.scheduled_start_time.isnot(None)
        )
    ).all()
    
    # Agrupar reuniones por año/mes
    meetings_by_month = defaultdict(list)
    
    for meeting in meetings:
        if meeting.scheduled_start_time:
            meeting_year = meeting.scheduled_start_time.year
            meeting_month = meeting.scheduled_start_time.month
            
            # Filtrar solo meses dentro del rango solicitado
            if (meeting_year > start_year) or (meeting_year == start_year and meeting_month >= start_month):
                meetings_by_month[(meeting_year, meeting_month)].append(meeting)
    
    # Calcular métricas por mes
    monthly_metrics = []
    
    for (year, month), month_meetings in sorted(meetings_by_month.items()):
        if len(month_meetings) == 0:
            continue
        
        # Calcular horas totales del mes
        total_seconds = 0
        for m in month_meetings:
            summary = db.query(Summary).filter(Summary.meeting_id == m.id).first()
            if summary and summary.ia_response_json:
                insights = summary.ia_response_json.get('insights', {})
                talk_times = insights.get('talk_time_seconds', {})
                if talk_times:
                    total_seconds += sum(talk_times.values())
        total_hours = total_seconds / 3600
        
        # Calcular participación promedio del mes
        participation_percents_month = []
        for m in month_meetings:
            summary = db.query(Summary).filter(Summary.meeting_id == m.id).first()
            if summary and summary.ia_response_json:
                insights = summary.ia_response_json.get('insights', {})
                participation = insights.get('participation_percent', {})
                toon = summary.toon or ''
                
                for speaker, percent in participation.items():
                    if user_email.split('@')[0].lower() in speaker.lower() or (user.display_name and user.display_name.lower() in speaker.lower()):
                        participation_percents_month.append(percent)
        
        avg_participation = sum(participation_percents_month) / len(participation_percents_month) if participation_percents_month else 0
        
        # Calcular calidad promedio del mes
        collaborations_month = []
        decisivenesses_month = []
        conflicts_month = []
        
        for m in month_meetings:
            summary = db.query(Summary).filter(Summary.meeting_id == m.id).first()
            if summary and summary.ia_response_json:
                insights = summary.ia_response_json.get('insights', {})
                
                if insights.get('collaboration', {}).get('score_0_100'):
                    collaborations_month.append(insights['collaboration']['score_0_100'])
                if insights.get('decisiveness', {}).get('score_0_100'):
                    decisivenesses_month.append(insights['decisiveness']['score_0_100'])
                if insights.get('conflict_level_0_100') is not None:
                    conflicts_month.append(insights['conflict_level_0_100'])
        
        avg_collab = sum(collaborations_month) / len(collaborations_month) if collaborations_month else 0
        avg_decis = sum(decisivenesses_month) / len(decisivenesses_month) if decisivenesses_month else 0
        avg_conflict = sum(conflicts_month) / len(conflicts_month) if conflicts_month else 0
        avg_engagement = (avg_collab + avg_decis) / 2 if (collaborations_month and decisivenesses_month) else 0
        
        monthly_metrics.append(MonthlyMetricsResponse(
            year=year,
            month=month,
            meetings_count=len(month_meetings),
            total_hours=round(total_hours, 1),
            average_participation_percent=round(avg_participation, 1),
            average_collaboration=round(avg_collab, 1),
            average_decisiveness=round(avg_decis, 1),
            average_conflict=round(avg_conflict, 1),
            average_engagement=round(avg_engagement, 1)
        ))
    
    return UserAnalyticsHistoryResponse(
        monthly_metrics=monthly_metrics,
        total_months=len(monthly_metrics)
    )

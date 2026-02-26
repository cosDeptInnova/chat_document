"""Celery tasks for meeting summary processing with IA."""
from datetime import datetime, timezone
import uuid
from celery import Task
from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.summary import Summary
from app.models.meeting import Meeting
from app.config import settings
import httpx
import logging
from typing import Any, Dict, List
from celery.exceptions import Retry

logger = logging.getLogger(__name__)


def _build_hybrid_rag_payload(
    meeting_id: str,
    ia_data: Dict[str, Any],
    meeting: Meeting | None,
) -> Dict[str, Any]:
    """Construye payload enriquecido para /ingest de Hybrid RAG con metadatos auditables."""
    payload = dict(ia_data or {})
    payload["meeting_id"] = meeting_id

    if meeting is not None:
        payload.setdefault("user_id", meeting.user_id)
        payload.setdefault("file_name", meeting.title or f"meeting-{meeting_id}")
        payload.setdefault("source_path", meeting.meeting_url)

        if meeting.actual_start_time:
            payload.setdefault("meeting_datetime", meeting.actual_start_time.isoformat())
        elif meeting.scheduled_start_time:
            payload.setdefault("meeting_datetime", meeting.scheduled_start_time.isoformat())

    payload.setdefault("ingest_source", "notetaker_backend.summary_tasks")
    payload.setdefault("ingest_version", "v1")
    return payload


def _send_to_hybrid_rag_ingest(
    meeting_id: str,
    ia_data: Dict[str, Any],
    meeting: Meeting | None,
) -> Dict[str, Any]:
    """Envía el resultado de /analyze_vexa al endpoint /ingest sin romper el flujo principal."""
    ingest_url = (settings.hybrid_rag_ingest_url or "").strip()
    if not ingest_url:
        return {"status": "disabled", "reason": "HYBRID_RAG_INGEST_URL no configurada"}

    request_id = str(uuid.uuid4())
    payload = _build_hybrid_rag_payload(meeting_id=meeting_id, ia_data=ia_data, meeting=meeting)

    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
        "X-Meeting-ID": meeting_id,
    }
    if settings.hybrid_rag_api_key:
        headers["Authorization"] = f"Bearer {settings.hybrid_rag_api_key}"
        headers["X-API-Key"] = settings.hybrid_rag_api_key

    timeout_seconds = int(settings.hybrid_rag_timeout_seconds)
    max_retries = int(settings.hybrid_rag_max_retries)
    backoff_schedule = [1, 2, 4, 8]

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "📚 [CELERY] Enviando a Hybrid RAG (attempt=%s/%s meeting_id=%s request_id=%s)",
                attempt,
                max_retries,
                meeting_id,
                request_id,
            )
            response = httpx.post(
                ingest_url,
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
                verify=settings.ssl_verify,
            )

            if response.status_code in (200, 201):
                return {
                    "status": "success",
                    "http_status": response.status_code,
                    "request_id": request_id,
                    "body": response.json() if response.content else {},
                }

            # 409 es idempotente/esperable cuando la reunión ya fue indexada.
            if response.status_code == 409:
                return {
                    "status": "already_indexed",
                    "http_status": 409,
                    "request_id": request_id,
                    "body": response.json() if response.content else {},
                }

            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except Exception as ex:  # noqa: BLE001
            last_error = str(ex)

        if attempt < max_retries:
            import time

            time.sleep(backoff_schedule[min(attempt - 1, len(backoff_schedule) - 1)])

    return {
        "status": "failed",
        "request_id": request_id,
        "error": last_error or "Error desconocido al invocar /ingest",
    }


def _parse_iso_to_seconds(s: str, base_seconds: float = 0.0) -> float:
    """Convierte ISO datetime a segundos relativos (o absolutos si base_seconds=0)."""
    if not s or not isinstance(s, str):
        return 0.0
    s = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() - base_seconds
    except (ValueError, TypeError):
        return 0.0


def transcript_to_cosmos_format(raw_list: List[Any]) -> List[Dict[str, Any]]:
    """
    Convierte la transcripcion (formato Notetaker/Vexa/Recall.ai) al formato que espera Cosmos.

    Entrada puede ser:
    - conversation: [ { speaker, text, start_time, end_time, duration } ]
    - Vexa segments: [ { text, speaker, absolute_start_time, absolute_end_time } ]
    - Recall.ai segments: [ { words: [{text, start_timestamp: {relative}, end_timestamp: {relative}}], participant: {name} } ]

    Salida: lista de utterances con text, speaker_name, start_time, end_time, participant.name
    para que Cosmos identifique bien a los participantes y los temas.
    """
    if not raw_list or not isinstance(raw_list, list):
        return []

    utterances = []
    base_ts = None  # Para Vexa: primer timestamp como base

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        
        # Formato Recall.ai: tiene "words" array y "participant"
        words = item.get("words")
        if words and isinstance(words, list) and len(words) > 0:
            # Extraer texto concatenando todas las palabras
            text_parts = []
            first_word = words[0]
            last_word = words[-1]
            
            for word in words:
                if isinstance(word, dict):
                    word_text = word.get("text", "").strip()
                    if word_text:
                        text_parts.append(word_text)
            
            text = " ".join(text_parts).strip()
            if not text:
                continue
            
            # Extraer timestamps relativos del primer y último word
            start_time = None
            end_time = None
            
            if isinstance(first_word, dict):
                start_ts = first_word.get("start_timestamp")
                if isinstance(start_ts, dict):
                    start_time = start_ts.get("relative")
            
            if isinstance(last_word, dict):
                end_ts = last_word.get("end_timestamp")
                if isinstance(end_ts, dict):
                    end_time = end_ts.get("relative")
            
            # Speaker del participant
            participant = item.get("participant") or {}
            speaker_name = participant.get("name") or "Unknown"
            
        else:
            # Formato estándar (Notetaker/Vexa)
            text = (item.get("text") or item.get("transcript") or "").strip()
            if not text:
                continue

            # Speaker: varios nombres posibles
            speaker = (
                item.get("speaker_name")
                or item.get("speaker")
                or (item.get("participant") or {}).get("name")
                or item.get("speaker_id")
                or "Unknown"
            )
            if isinstance(speaker, dict):
                speaker = speaker.get("name") or speaker.get("id") or "Unknown"
            speaker_name = (speaker or "Unknown").strip() or "Unknown"

            # Tiempos: relativos en segundos o absolutos ISO
            start_time = item.get("start_time")
            end_time = item.get("end_time")
            if start_time is None and end_time is None:
                start_str = item.get("absolute_start_time")
                end_str = item.get("absolute_end_time")
                if start_str and end_str:
                    if base_ts is None:
                        try:
                            s = start_str.strip().replace("Z", "+00:00")
                            base_ts = datetime.fromisoformat(s).timestamp()
                        except (ValueError, TypeError):
                            base_ts = 0.0
                    start_time = _parse_iso_to_seconds(start_str, base_ts or 0.0)
                    end_time = _parse_iso_to_seconds(end_str, base_ts or 0.0)
                else:
                    start_time = 0.0
                    end_time = 0.0
        
        # Validar y normalizar tiempos
        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = float(start_time) + 0.1
        try:
            start_time = float(start_time)
            end_time = float(end_time)
        except (TypeError, ValueError):
            start_time = 0.0
            end_time = 0.0
        if end_time <= start_time:
            end_time = start_time + 0.1

        utterances.append({
            "text": text,
            "speaker_name": speaker_name,
            "speaker": speaker_name,
            "start_time": start_time,
            "end_time": end_time,
            "participant": {"name": speaker_name},
        })
    return utterances


class DatabaseTask(Task):
    """Task base class that provides database session."""
    
    _db = None
    
    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db
    
    def after_return(self, *args, **kwargs):
        """Close database session after task completes."""
        if self._db is not None:
            self._db.close()
            self._db = None


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    max_retries=10,  # Aumentado a 10 para mayor resiliencia en generación inicial
    retry_backoff=60,  # Reintentar cada 60 segundos (exponencial: 60s, 120s, 240s...)
    retry_backoff_max=600,  # Máximo 10 minutos entre reintentos
    retry_jitter=True,
    acks_late=True,
    # Eliminado autoretry_for para manejar manualmente la lógica condicional
)
def process_meeting_summary(self, meeting_id: str):
    """
    Tarea Celery para procesar resumen e insights de una reunión usando IA Cosmos.
    
    Flujo:
    1. Obtiene la transcripción del endpoint interno
    2. Envía a la IA de Cosmos (URL configurada en AI_SERVICE_URL del .env)
    3. Guarda la respuesta en Summary
    
    Lógica de Reintentos:
    - Generación Inicial (sin datos previos): Reintenta agresivamente ante errores temporales (hasta 10 veces).
    - Regeneración (con datos previos): Aborta inmediatamente ("fail fast") ante cualquier error.
    
    Args:
        meeting_id: ID de la reunión a procesar
    
    Variables de entorno requeridas:
        AI_SERVICE_URL: URL del servicio de IA (ej: http://172.29.14.14/api/analyze_vexa)
        AI_SERVICE_API_KEY: (Opcional) API key para autenticación
    """
    start_time = datetime.now(timezone.utc)
    logger.info(f"🚀 [CELERY] Iniciando procesamiento de resumen para reunión {meeting_id}")
    logger.info(f"📋 [CELERY] Task ID: {self.request.id}, Retries: {self.request.retries}/{self.max_retries}")
    
    db = self.db

    # Función auxiliar para manejar errores y decidir si reintentar
    def handle_error_and_retry(summary_obj, error_desc, is_fatal=False):
        error_msg = f"Error procesando con IA: {str(error_desc)}"
        
        # Actualizar estado de error
        if summary_obj:
            summary_obj.processing_status = "failed"
            summary_obj.error_message = error_msg
            summary_obj.retry_count = self.request.retries + 1
            db.commit()

            # Verificar contexto: ¿Es regeneración?
            has_prev_data = bool(summary_obj.toon or (summary_obj.ia_response_json and summary_obj.ia_response_json.get("insights")))
            
            should_retry = False
            if is_fatal:
                # Error fatal: mostrar traceback completo
                logger.error(f"❌ [CELERY] {error_msg}", exc_info=True)
                logger.error(f"⛔ [CELERY] Error FATAL detectado. No se reintentará.")
            elif has_prev_data:
                # Regeneración: mostrar traceback completo porque es crítico
                logger.error(f"❌ [CELERY] {error_msg}", exc_info=True)
                logger.warning(f"⛔ [CELERY] Error en REGENERACIÓN (ya existían datos). Abortando para restaurar versión anterior.")
            elif self.request.retries < self.max_retries:
                # Error temporal que se va a reintentar: solo mensaje resumido, sin traceback
                logger.warning(f"⚠️ [CELERY] {error_msg} (Reintento {self.request.retries + 1}/{self.max_retries})")
                logger.info(f"🔄 [CELERY] Error temporal en GENERACIÓN INICIAL. Reintentando...")
                should_retry = True
            else:
                # Se agotaron los reintentos: mostrar traceback completo
                logger.error(f"❌ [CELERY] {error_msg}", exc_info=True)
                logger.error(f"❌ [CELERY] Se agotaron los reintentos ({self.max_retries}).")
            
            if should_retry:
                # Volver a poner en processing para que UI sepa que sigue vivo
                summary_obj.processing_status = "processing"
                db.commit()
                raise self.retry(exc=error_desc)
        else:
            # Si no hay summary_obj, mostrar error completo
            logger.error(f"❌ [CELERY] {error_msg}", exc_info=True)
        
        return {"success": False, "error": error_msg}
    
    try:
        # Buscar Summary
        summary = db.query(Summary).filter(Summary.meeting_id == meeting_id).first()
        if not summary:
            error_msg = f"Summary no encontrado para reunión {meeting_id}"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
        
        # Verificar si ya está completado - evitar reprocesar duplicados
        if summary.processing_status == "completed" and summary.is_final:
            logger.info(f"⏭️ [CELERY] Summary ya completado. Omitiendo (Task ID: {self.request.id})")
            return {"success": True, "message": "Summary ya completado", "skipped": True}
        
        # Detectar duplicados en processing (si no es retry)
        if summary.processing_status == "processing" and self.request.retries == 0:
            if summary.processing_started_at:
                time_since = (start_time - summary.processing_started_at).total_seconds()
                if time_since < 600:
                    logger.warning(f"⚠️ [CELERY] Summary ya en processing ({time_since:.0f}s). Posible duplicado.")
                    return {"success": True, "message": "Posible duplicado", "skipped": True}
        
        # Actualizar a processing
        summary.processing_status = "processing"
        summary.processing_started_at = start_time
        summary.retry_count = self.request.retries
        db.commit()
        
        # 1. Obtener transcripción
        # Usar backend_public_url si está configurado, sino usar 127.0.0.1 como fallback
        # (127.0.0.1 es más confiable que localhost en Windows)
        backend_url = settings.backend_public_url or "http://127.0.0.1:7000"
        # Asegurar que la URL no termine con /
        backend_url = backend_url.rstrip("/")
        
        transcript_url = f"{backend_url}/api/meetings/{meeting_id}/transcription"
        logger.info(f"📥 [CELERY] Obteniendo transcripción desde: {transcript_url}")
        
        try:
            transcript_response = httpx.get(transcript_url, timeout=30)
            transcript_response.raise_for_status()
            transcript_payload = transcript_response.json()
            
            if not transcript_payload.get("has_transcription"):
                # Sin transcripción no podemos hacer nada. Fail fast.
                return handle_error_and_retry(summary, Exception("No hay transcripción disponible"), is_fatal=True)

        except httpx.ConnectError as e:
            # Error específico de conexión (servidor no disponible o rechaza conexión)
            # El logging se maneja en handle_error_and_retry
            return handle_error_and_retry(summary, e, is_fatal=False)
        except httpx.RequestError as e:
            # Otros errores de red (timeout, etc.)
            # El logging se maneja en handle_error_and_retry
            return handle_error_and_retry(summary, e, is_fatal=False)
        except httpx.HTTPStatusError as e:
            is_fatal = e.response.status_code == 404
            return handle_error_and_retry(summary, e, is_fatal=is_fatal)

        # Preparar payload para Cosmos
        # El sistema debe manejar ambos formatos que pueden coexistir en la BD:
        # 1. Formato Vexa: conversation (segmentos ya procesados)
        # 2. Formato Recall.ai: raw_transcript_json (lista directa o dict con segments)
        
        raw_list = None
        format_detected = None
        
        # Prioridad 1: Formato Vexa (conversation ya procesada)
        conversation = transcript_payload.get("conversation")
        if conversation and isinstance(conversation, list) and len(conversation) > 0:
            raw_list = conversation
            format_detected = "Vexa"
        
        # Prioridad 2: Formato Recall.ai (raw_transcript_json)
        if not raw_list:
            raw_transcript_json = transcript_payload.get("raw_transcript_json")
            if isinstance(raw_transcript_json, list) and len(raw_transcript_json) > 0:
                raw_list = raw_transcript_json
                format_detected = "Recall.ai"
            elif isinstance(raw_transcript_json, dict):
                segments = raw_transcript_json.get("segments")
                if isinstance(segments, list) and len(segments) > 0:
                    raw_list = segments
                    format_detected = "Recall.ai/Vexa"
        
        # Validar que tenemos datos
        if not isinstance(raw_list, list) or len(raw_list) == 0:
            error_msg = "No se encontraron segmentos válidos en la transcripción"
            logger.error(f"❌ [CELERY] {error_msg}")
            logger.debug(f"   conversation: {type(conversation).__name__} con {len(conversation) if isinstance(conversation, list) else 'N/A'} elementos")
            logger.debug(f"   raw_transcript_json: {type(transcript_payload.get('raw_transcript_json')).__name__}")
            return handle_error_and_retry(summary, Exception(error_msg), is_fatal=True)
        
        logger.info(f"📝 [CELERY] Procesando {len(raw_list)} segmentos (formato: {format_detected})")
            
        # Convertir al formato que espera Cosmos
        cosmos_utterances = transcript_to_cosmos_format(raw_list)
        
        if not cosmos_utterances or len(cosmos_utterances) == 0:
            error_msg = f"No se generaron utterances útiles después de procesar {len(raw_list)} segmentos"
            logger.error(f"❌ [CELERY] {error_msg}")
            if raw_list and len(raw_list) > 0:
                logger.debug(f"   Primer segmento raw: {str(raw_list[0])[:500]}")
            return handle_error_and_retry(summary, Exception(error_msg), is_fatal=True)
        
        # Validar que los utterances tienen texto válido
        valid_utterances = [u for u in cosmos_utterances if u.get("text") and u.get("text").strip()]
        if len(valid_utterances) == 0:
            error_msg = f"Todos los {len(cosmos_utterances)} utterances generados tienen texto vacío"
            logger.error(f"❌ [CELERY] {error_msg}")
            if cosmos_utterances:
                logger.debug(f"   Muestra primer utterance: {str(cosmos_utterances[0])[:500]}")
            return handle_error_and_retry(summary, Exception(error_msg), is_fatal=True)
        
        # Usar solo los utterances válidos
        cosmos_utterances = valid_utterances
        
        logger.info(f"✅ [CELERY] Generados {len(cosmos_utterances)} utterances válidos para Cosmos")
        
        # Cosmos espera el campo "segments" en el payload
        payload_for_cosmos = {"segments": cosmos_utterances}
        
        # 2. Enviar a IA Cosmos
        ia_url = settings.ai_service_url
        if not ia_url:
            return handle_error_and_retry(summary, Exception("AI_SERVICE_URL no configurada"), is_fatal=True)
        
        logger.info(f"🤖 [CELERY] Enviando a IA Cosmos ({ia_url})")
        
        headers = {}
        if settings.ai_service_api_key:
            headers["Authorization"] = f"Bearer {settings.ai_service_api_key}"
            headers["X-API-Key"] = settings.ai_service_api_key
            
        try:
            ia_response = httpx.post(
                ia_url,
                json=payload_for_cosmos,
                headers=headers if headers else None,
                timeout=None, 
                verify=settings.ssl_verify,
            )
            
            if ia_response.status_code != 200:
                # El error 405 (Not Allowed) suele ser un problema de configuración de nginx/IA (ej. HTTPS vs HTTP)
                # que no se arregla con reintentos. Lo marcamos como fatal.
                is_fatal = ia_response.status_code in [401, 403, 405]
                
                error_msg_detail = ia_response.text[:200]
                if ia_response.status_code == 405:
                    error_desc = "IA respondió 405 (Método no permitido). Posible error de servidor/protocolo en Cosmos."
                else:
                    error_desc = f"IA respondió {ia_response.status_code}: {error_msg_detail}"
                
                return handle_error_and_retry(summary, Exception(error_desc), is_fatal=is_fatal)
            
            # Éxito
            ia_data = ia_response.json()
            logger.info(f"✅ [CELERY] Respuesta IA recibida")

            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
            hybrid_rag_result = _send_to_hybrid_rag_ingest(
                meeting_id=meeting_id,
                ia_data=ia_data,
                meeting=meeting,
            )
            logger.info(
                "📚 [CELERY] Resultado Hybrid RAG meeting_id=%s status=%s request_id=%s",
                meeting_id,
                hybrid_rag_result.get("status"),
                hybrid_rag_result.get("request_id"),
            )

            ia_data_with_audit = dict(ia_data)
            ia_data_with_audit["hybrid_rag_ingest"] = hybrid_rag_result

            summary.toon = ia_data.get("toon")
            summary.ia_response_json = ia_data_with_audit
            summary.processing_status = "completed"
            summary.is_final = True
            
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            summary.processing_time_seconds = processing_time
            summary.completed_at = end_time
            # Legacy
            if ia_data.get("toon"):
                summary.summary_text = ia_data.get("toon")
            if ia_data_with_audit.get("insights"):
                summary.summary_json = ia_data_with_audit.get("insights")
                
            db.commit()
            return {"success": True, "meeting_id": meeting_id, "time": processing_time}

        except (httpx.RequestError, httpx.TimeoutException) as e:
            return handle_error_and_retry(summary, e, is_fatal=False)
        except Retry:
            raise
        except Exception as e:
            return handle_error_and_retry(summary, e, is_fatal=False)

    except Retry:
        raise
    except Exception as e:
        # Error catastrófico (ej. DB caída)
        logger.error(f"❌ [CELERY] Error catastrófico: {e}", exc_info=True)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        return {"success": False, "error": str(e)}

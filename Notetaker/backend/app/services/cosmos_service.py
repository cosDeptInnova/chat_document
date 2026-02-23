"""Servicio para comunicación con endpoint de Cosmos para diarización y transcripción."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from pathlib import Path
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class CosmosService:
    """Servicio para enviar audio a Cosmos y procesar respuesta de diarización y transcripción."""

    def __init__(self):
        """Inicializar el servicio de Cosmos."""
        if not settings.cosmos_audio_endpoint:
            logger.warning(
                "Cosmos audio endpoint no configurado. Configura COSMOS_AUDIO_ENDPOINT en .env"
            )

    async def send_audio_for_processing(
        self,
        audio_file_path: Path,
        meeting_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Enviar archivo de audio a Cosmos para procesamiento.

        Args:
            audio_file_path: Ruta al archivo de audio
            meeting_id: ID de la reunión (opcional)
            metadata: Metadatos adicionales (opcional)

        Returns:
            Respuesta de Cosmos con diarización y transcripción
        """
        if not settings.cosmos_audio_endpoint:
            raise ValueError(
                "COSMOS_AUDIO_ENDPOINT no está configurado. Configúralo en el archivo .env"
            )

        try:
            # Verificar que el archivo existe
            if not audio_file_path.exists():
                raise FileNotFoundError(f"Archivo de audio no encontrado: {audio_file_path}")

            logger.info(f"Enviando audio a Cosmos: {audio_file_path}")

            # Preparar archivo para envío
            with open(audio_file_path, "rb") as audio_file:
                files = {
                    "audio_file": (
                        audio_file_path.name,
                        audio_file,
                        "audio/mpeg" if audio_file_path.suffix == ".mp3" else "audio/mp4",
                    )
                }

                # Preparar datos adicionales
                data = {}
                if meeting_id:
                    data["meeting_id"] = meeting_id
                if metadata:
                    data["metadata"] = metadata

                # Preparar headers
                headers = {}
                if settings.cosmos_api_key:
                    headers["Authorization"] = f"Bearer {settings.cosmos_api_key}"

                # Enviar request
                async with httpx.AsyncClient(timeout=600.0) as client:  # Timeout largo para procesamiento
                    response = await client.post(
                        settings.cosmos_audio_endpoint,
                        files=files,
                        data=data,
                        headers=headers,
                    )

                response.raise_for_status()
                result = response.json()

                logger.info(f"Respuesta recibida de Cosmos para reunión {meeting_id}")
                return result

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Error HTTP enviando audio a Cosmos: {e.response.status_code} - {e.response.text}"
            )
            raise
        except Exception as e:
            logger.error(f"Error inesperado enviando audio a Cosmos: {e}")
            raise

    def process_cosmos_response(
        self, cosmos_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesar respuesta de Cosmos y normalizar formato.

        Args:
            cosmos_response: Respuesta cruda de Cosmos

        Returns:
            Diccionario normalizado con:
            - transcription: Lista de segmentos de transcripción
            - diarization: Lista de segmentos de diarización
            - participants: Lista de participantes identificados
        """
        try:
            # Normalizar respuesta (puede venir en diferentes formatos)
            normalized = {
                "transcription": cosmos_response.get("transcription", []),
                "diarization": cosmos_response.get("diarization", []),
                "participants": cosmos_response.get("participants", []),
            }

            # Si la respuesta tiene un formato diferente, intentar mapear
            if "segments" in cosmos_response:
                normalized["transcription"] = cosmos_response["segments"]
                normalized["diarization"] = cosmos_response.get("speakers", [])

            logger.info(
                f"Respuesta procesada: {len(normalized['transcription'])} segmentos, "
                f"{len(normalized['diarization'])} speakers, "
                f"{len(normalized['participants'])} participantes"
            )

            return normalized

        except Exception as e:
            logger.error(f"Error procesando respuesta de Cosmos: {e}")
            raise

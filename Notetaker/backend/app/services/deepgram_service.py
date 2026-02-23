"""Deepgram service with diarization support for speaker identification."""
import asyncio
import json
from typing import List, Dict, Optional, Callable
import websockets
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from app.config import settings


class DeepgramService:
    """Service for Deepgram real-time transcription with speaker diarization."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Deepgram service.
        
        Args:
            api_key: Deepgram API key (defaults to config)
        """
        self.api_key = api_key or settings.deepgram_api_key
        
        # Debug: Verificar que la API key se cargó
        if not self.api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY no está configurada. "
                "Configúrala en el archivo .env"
            )
        
        # Verificar que no esté vacía o solo espacios
        if not self.api_key.strip():
            error_msg = (
                "❌ ERROR CRÍTICO: DEEPGRAM_API_KEY está vacía o solo contiene espacios.\n"
                "   Por favor, verifica el archivo .env y asegúrate de que contiene:\n"
                "   DEEPGRAM_API_KEY=tu_api_key_completa_aqui\n"
                "   La API key no debe tener espacios alrededor del valor."
            )
            print(f"[Deepgram] {error_msg}")
            raise ValueError(error_msg)
        
        # Validar longitud mínima (las API keys de Deepgram suelen tener 40 caracteres)
        if len(self.api_key.strip()) < 32:
            error_msg = (
                f"❌ ERROR CRÍTICO: DEEPGRAM_API_KEY parece estar incompleta.\n"
                f"   Longitud actual: {len(self.api_key.strip())} caracteres\n"
                f"   Longitud esperada: ~40 caracteres\n"
                f"   Por favor, verifica que la API key esté completa en el archivo .env"
            )
            print(f"[Deepgram] {error_msg}")
            raise ValueError(error_msg)
        
        # Debug log (sin mostrar la key completa por seguridad)
        api_key_preview = self.api_key[:8] + "..." if len(self.api_key) > 8 else "***"
        print(f"[Deepgram] ✅ API Key cargada correctamente: {api_key_preview} (longitud: {len(self.api_key)})")
        
        self.connection = None
        self.transcripts: List[Dict] = []
        self.on_transcript_callback: Optional[Callable] = None
        self.is_connected = False
        self.meeting_id = None
    
    async def start_streaming(
        self,
        meeting_id: str,
        language: str = "es",
        on_transcript: Optional[Callable] = None
    ):
        """
        Start streaming transcription with diarization.
        
        Args:
            meeting_id: Meeting identifier
            language: Language code (default: "es" for Spanish)
            on_transcript: Callback function for transcript events
                          Receives: {
                              'speaker': 0 or 1 or 2... (speaker ID from Deepgram),
                              'text': 'transcription text',
                              'start': 0.5,
                              'end': 2.3,
                              'confidence': 0.98,
                              'is_final': True
                          }
        """
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY no está configurada")
        
        self.meeting_id = meeting_id
        self.on_transcript_callback = on_transcript
        self.transcripts = []
        
        try:
            # Build Deepgram WebSocket URL with DIARIZATION enabled
            url = (
                f"wss://api.deepgram.com/v1/listen?"
                f"language={language}&"
                "interim_results=true&"
                "encoding=linear16&"
                "sample_rate=16000&"
                "channels=1&"
                "smart_format=true&"
                "diarize=true&"           # ← CLAVE: Diarización activada
                "utterances=true&"        # ← CLAVE: Agrupa por speaker
                "punctuate=true"
            )
            
            # Verificar que la API key esté presente antes de crear headers
            if not self.api_key or not self.api_key.strip():
                error_msg = (
                    "❌ ERROR CRÍTICO: DEEPGRAM_API_KEY está vacía o no está configurada.\n"
                    "   Por favor, verifica el archivo .env y asegúrate de que contiene:\n"
                    "   DEEPGRAM_API_KEY=tu_api_key_completa_aqui"
                )
                print(f"[Deepgram] {error_msg}")
                raise ValueError(error_msg)
            
            headers = {
                "Authorization": f"Token {self.api_key.strip()}",
                "User-Agent": "Cosmos-Notetaker/1.0"
            }
            
            # Debug: Log del header (sin mostrar la key completa)
            auth_preview = headers["Authorization"][:20] + "..." if len(headers["Authorization"]) > 20 else headers["Authorization"]
            print(f"[Deepgram] Headers de autenticación: {auth_preview}")
            
            print(f"[Deepgram] Iniciando conexión con diarización para meeting {meeting_id}...")
            
            try:
                self.connection = await websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5
                )
            except InvalidStatus as e:
                # Capturar específicamente el error HTTP 401 (autenticación fallida)
                # InvalidStatus en websockets tiene status_code como atributo directo
                status_code = getattr(e, 'status_code', None) or getattr(e, 'response_status', None)
                
                if status_code == 401:
                    error_msg = (
                        "❌ ERROR DE AUTENTICACIÓN: Deepgram rechazó la conexión con HTTP 401 (No autorizado).\n"
                        "   Esto significa que la API key es INVÁLIDA o está INCORRECTA.\n"
                        "   Por favor, verifica:\n"
                        "   1. Que DEEPGRAM_API_KEY en el archivo .env esté COMPLETA (no le falten dígitos)\n"
                        "   2. Que la API key sea VÁLIDA y no haya expirado\n"
                        "   3. Que no tenga espacios extra antes o después del valor\n"
                        f"   4. Longitud de la API key actual: {len(self.api_key.strip())} caracteres (esperada: ~40)\n"
                        "   5. Puedes obtener una nueva API key en: https://console.deepgram.com/"
                    )
                    print(f"[Deepgram] {error_msg}")
                    self.is_connected = False
                    raise RuntimeError("Deepgram authentication failed: Invalid API key") from e
                else:
                    # Otro error HTTP (403, 500, etc.)
                    error_msg = (
                        f"❌ ERROR HTTP {status_code or 'unknown'} al conectar con Deepgram: {str(e)}\n"
                        "   Revisa los logs de Deepgram o contacta con soporte."
                    )
                    print(f"[Deepgram] {error_msg}")
                    self.is_connected = False
                    raise RuntimeError(f"Deepgram connection failed: HTTP {status_code or 'unknown'}") from e
            
            self.is_connected = True
            print(f"[Deepgram] ✅ Conexión iniciada correctamente con diarización")
            
            # Start receiving transcripts in background
            asyncio.create_task(self._receive_transcripts())
            
        except RuntimeError:
            # Re-lanzar RuntimeError sin modificar (ya tiene el mensaje correcto)
            raise
        except Exception as e:
            error_msg = f"Error inesperado al iniciar Deepgram: {type(e).__name__}: {str(e)}"
            print(f"[Deepgram] ❌ {error_msg}")
            self.is_connected = False
            raise RuntimeError(error_msg) from e
    
    async def _receive_transcripts(self):
        """Receive transcript messages from Deepgram WebSocket."""
        try:
            async for message in self.connection:
                if isinstance(message, bytes):
                    continue
                
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    print(f"[Deepgram] JSON inválido recibido (ignorado)")
                    continue
                
                # Process Results messages
                if data.get("type") == "Results":
                    await self._process_results(data)
        
        except websockets.ConnectionClosed:
            print(f"[Deepgram] WebSocket cerrado para meeting {self.meeting_id}")
            self.is_connected = False
        except Exception as e:
            print(f"[Deepgram] Error recibiendo transcripciones: {e}")
            self.is_connected = False
    
    async def _process_results(self, data: dict):
        """Process Results message from Deepgram with diarization."""
        channel = data.get("channel", {})
        alternatives = channel.get("alternatives", [])
        
        if not alternatives:
            return
        
        # Check if we have utterances (grouped by speaker)
        utterances = alternatives[0].get("utterances", [])
        
        if utterances:
            # Process each utterance (each speaker segment)
            for utterance in utterances:
                speaker_id = utterance.get("speaker", 0)
                transcript = utterance.get("transcript", "").strip()
                
                if not transcript:
                    continue
                
                transcript_entry = {
                    'meeting_id': self.meeting_id,
                    'speaker': speaker_id,  # Speaker ID from Deepgram (0, 1, 2...)
                    'text': transcript,
                    'confidence': utterance.get("confidence"),
                    'is_final': data.get("is_final", False),
                    'start': utterance.get("start"),
                    'end': utterance.get("end"),
                }
                
                self.transcripts.append(transcript_entry)
                
                # Call callback if provided
                if self.on_transcript_callback:
                    try:
                        await self.on_transcript_callback(transcript_entry)
                    except Exception as e:
                        print(f"[Deepgram] Error en callback: {e}")
        else:
            # Fallback: single transcript without utterances
            transcript_text = alternatives[0].get("transcript", "").strip()
            if transcript_text:
                transcript_entry = {
                    'meeting_id': self.meeting_id,
                    'speaker': None,  # No speaker info
                    'text': transcript_text,
                    'confidence': alternatives[0].get("confidence"),
                    'is_final': data.get("is_final", False),
                    'start': data.get("start"),
                    'end': None,
                }
                
                self.transcripts.append(transcript_entry)
                
                if self.on_transcript_callback:
                    try:
                        await self.on_transcript_callback(transcript_entry)
                    except Exception as e:
                        print(f"[Deepgram] Error en callback: {e}")
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to Deepgram."""
        if self.connection and self.is_connected:
            try:
                await self.connection.send(audio_data)
            except Exception as e:
                print(f"[Deepgram] Error enviando audio: {e}")
                self.is_connected = False
                raise
    
    async def finish(self):
        """Finish transcription and close connection."""
        if self.connection:
            try:
                await self.connection.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            
            try:
                await self.connection.close()
            except Exception:
                pass
            
            self.connection = None
            self.is_connected = False
            print(f"[Deepgram] Conexión cerrada para meeting {self.meeting_id}")
    
    def get_transcripts(self) -> List[Dict]:
        """Get all collected transcripts with speaker information."""
        return self.transcripts


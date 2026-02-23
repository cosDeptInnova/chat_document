"""Gestor de instancias activas de bots de Teams."""
import asyncio
import logging
from typing import Dict, Optional
from app.services.teams_bot import TeamsBotService
from app.models.meeting import MeetingStatus

logger = logging.getLogger(__name__)


class BotManager:
    """
    Gestiona instancias activas de bots de Teams.
    
    Permite crear múltiples bots simultáneos (uno por reunión) y
    controlarlos centralmente.
    """
    
    def __init__(self):
        self.active_bots: Dict[str, TeamsBotService] = {}
        self._lock = asyncio.Lock()
    
    async def start_bot(
        self,
        meeting_id: str,
        meeting_url: str,
        bot_display_name: Optional[str] = None,
        on_transcript: Optional[callable] = None
    ) -> bool:
        """
        Iniciar un bot para una reunión.
        
        Args:
            meeting_id: ID de la reunión
            meeting_url: URL de la reunión Teams
            bot_display_name: Nombre del bot en la reunión
            on_transcript: Callback para transcripciones
        
        Returns:
            True si se inició correctamente, False si ya existe
        """
        async with self._lock:
            if meeting_id in self.active_bots:
                logger.warning(f"⚠️ Bot ya existe para reunión {meeting_id}")
                return False
            
            try:
                bot = TeamsBotService()
                
                # Iniciar bot en background
                task = asyncio.create_task(
                    bot.join_meeting(
                        meeting_url=meeting_url,
                        meeting_id=meeting_id,
                        bot_display_name=bot_display_name,
                        on_transcript=on_transcript
                    )
                )
                
                # Guardar referencia
                self.active_bots[meeting_id] = bot
                
                logger.info(f"✅ Bot iniciado para reunión {meeting_id}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error iniciando bot: {e}", exc_info=True)
                raise
    
    async def stop_bot(self, meeting_id: str) -> bool:
        """
        Detener un bot activo.
        
        Args:
            meeting_id: ID de la reunión
        
        Returns:
            True si se detuvo, False si no existía
        """
        async with self._lock:
            if meeting_id not in self.active_bots:
                logger.warning(f"⚠️ Bot no existe para reunión {meeting_id}")
                return False
            
            try:
                bot = self.active_bots[meeting_id]
                await bot.leave_meeting()
                del self.active_bots[meeting_id]
                
                logger.info(f"✅ Bot detenido para reunión {meeting_id}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error deteniendo bot: {e}", exc_info=True)
                return False
    
    def is_bot_active(self, meeting_id: str) -> bool:
        """Verificar si hay un bot activo para una reunión."""
        return meeting_id in self.active_bots
    
    def get_active_bots(self) -> list[str]:
        """Obtener lista de IDs de reuniones con bots activos."""
        return list(self.active_bots.keys())
    
    async def stop_all_bots(self):
        """Detener todos los bots activos."""
        async with self._lock:
            meeting_ids = list(self.active_bots.keys())
            for meeting_id in meeting_ids:
                try:
                    await self.stop_bot(meeting_id)
                except Exception as e:
                    logger.error(f"❌ Error deteniendo bot {meeting_id}: {e}")


# Instancia global del gestor
bot_manager = BotManager()


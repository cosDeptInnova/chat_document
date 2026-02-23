# cosmos_mcp/models/inference.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Rol del mensaje: system | user | assistant")
    content: str = Field(..., description="Contenido en texto plano del mensaje")


class ChatCompletionRequest(BaseModel):
    """
    Modelo interno del MCP para /api/v1/chat/completions.

    - Es compatible con el cliente OpenAI que usa CrewAI.
    - Acepta muchos parámetros modernos (para que FastAPI no falle con 422),
      pero SOLO traduce al servidor llama.cpp los campos que sabemos que ese
      servidor acepta bien.

    Actualmente reenviamos a llama.cpp SOLO:
      - model
      - messages
      - temperature

    (Es el mismo patrón que tu endpoint /query_graph, que sabemos que funciona.)
    """

    model: str = Field(..., description="Nombre del modelo que llama.cpp debe usar")
    messages: List[ChatMessage] = Field(
        ..., description="Historial de mensajes en formato OpenAI"
    )

    # Parámetros básicos que SÍ nos interesa manejar
    temperature: Optional[float] = Field(
        0.2,
        ge=0.0,
        le=2.0,
        description="Temperatura de muestreo; se reenvía a llama.cpp",
    )

    # ----- Parámetros modernos que podemos recibir pero NO reenviamos (de momento) -----
    # Los dejamos aquí para que el cliente OpenAI moderno no rompa con 422,
    # pero no tocaremos el payload que va al llama-server.

    max_tokens: Optional[int] = Field(
        None, ge=1, description="Límite de tokens de salida (NO se reenvía aún)"
    )
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Top-p sampling (NO se reenvía aún)"
    )
    n: Optional[int] = Field(
        None, description="Número de respuestas a generar (NO se reenvía aún)"
    )
    stream: Optional[bool] = Field(
        None, description="Streaming de tokens (NO se reenvía aún)"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Secuencias de parada (NO se reenvía aún)"
    )
    presence_penalty: Optional[float] = Field(
        None, description="Penalización de presencia (NO se reenvía aún)"
    )
    frequency_penalty: Optional[float] = Field(
        None, description="Penalización de frecuencia (NO se reenvía aún)"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, description="Logit bias (NO se reenvía aún)"
    )
    user: Optional[str] = Field(
        None, description="Identificador de usuario (NO se reenvía aún)"
    )

    # response_format, tools, tool_choice, seed, etc.
    response_format: Optional[Dict[str, Any]] = Field(
        None, description="Formato de respuesta OpenAI (NO se reenvía aún)"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tools OpenAI (NO se reenvían aún)"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None, description="Tool choice OpenAI (NO se reenvía aún)"
    )
    seed: Optional[int] = Field(
        None, description="Seed de reproducibilidad (NO se reenvía aún)"
    )

    class Config:
        # Cualquier cosa extra que mande el cliente OpenAI moderno, se ignora.
        extra = "ignore"

    def to_payload(self) -> Dict[str, Any]:
        """
        Convierte la petición del MCP al payload que espera llama.cpp.

        MUY IMPORTANTE:
        ---------------
        Para evitar errores 400 del servidor llama.cpp, aquí sólo mandamos
        lo que ya sabemos que funciona con tu build actual, que es:

            {
              "model": "...",
              "messages": [...],
              "temperature": ...
            }

        Si en el futuro ves (en logs) que llama.cpp acepta top_p, max_tokens,
        etc. sin problemas, se pueden ir añadiendo poco a poco.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in self.messages],
        }

        # Solo reenviamos temperature, porque sabemos que tu endpoint funcional lo usa
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)

        # OJO: NO reenviamos ni top_p ni max_tokens por ahora
        # if self.top_p is not None:
        #     payload["top_p"] = float(self.top_p)
        # if self.max_tokens is not None:
        #     payload["max_tokens"] = int(self.max_tokens)

        return payload

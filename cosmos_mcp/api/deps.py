# cosmos_mcp/api/deps.py

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status

from services.llm_client import LLMClient

logger = logging.getLogger(__name__)


async def get_llm_client(request: Request) -> LLMClient:
    """
    Devuelve la instancia de `LLMClient` almacenada en `app.state.llama_client`.

    - Se inicializa en el lifespan de FastAPI (main.py).
    - Si no está disponible, devolvemos 503 para indicar que el MCP aún no está listo.
    """
    client = getattr(request.app.state, "llama_client", None)
    if client is None:
        logger.error("LLMClient no disponible en app.state.llama_client")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Llama MCP no está listo: cliente LLM no inicializado.",
        )
    return client


# Alias de compatibilidad con el nombre antiguo
# (por si en algún sitio aún se importa get_llama_client)
get_llama_client = get_llm_client

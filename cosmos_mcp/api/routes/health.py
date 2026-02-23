# cosmos_mcp/api/routes/health.py

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_llama_client
from services.llm_client import LLMClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(llama_client: LLMClient = Depends(get_llama_client)):
    """
    Health check del MCP:
    - Verifica que el cliente LLM está vivo llamando a llama.cpp (health_check).
    """
    try:
        ok = await llama_client.health_check()
    except Exception as exc:
        logger.warning("Health check a llama.cpp falló: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error comprobando el LLM backend: {exc}",
        )

    if not ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Llama backend no responde correctamente.",
        )

    return {
        "status": "ok",
        "llm_backend": "up",
    }

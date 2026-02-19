# cosmos_mcp_app/server.py

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging

from mcp.server.fastmcp import FastMCP
from app.core import settings
from services.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class MCPAppContext:
    """Contexto compartido para el servidor MCP."""
    llm_client: LLMClient


@asynccontextmanager
async def mcp_lifespan(server: FastMCP) -> AsyncIterator[MCPAppContext]:
    """
    Lifespan del servidor MCP: inicializa y cierra un LLMClient propio del MCP.

    Esto es independiente del LLMClient que puedas estar usando en main.py
    para endpoints REST (health/inference).
    """
    llm_client = LLMClient(
        base_url=settings.LLAMA_SERVER_BASE_URL,
        api_key=settings.LLAMA_SERVER_API_KEY,
        timeout=settings.LLAMA_REQUEST_TIMEOUT,
    )
    await llm_client.startup()
    logger.info(
        "MCPAppContext inicializado · LLM base_url=%s",
        settings.LLAMA_SERVER_BASE_URL,
    )

    try:
        yield MCPAppContext(llm_client=llm_client)
    finally:
        logger.info("Cerrando MCPAppContext (LLMClient del MCP)")
        await llm_client.shutdown()


#Creamos el servidor MCP (sin registrar tools aquí directamente)
mcp_server = FastMCP(
    name="Cosmos Llama MCP",
    lifespan=mcp_lifespan,
    json_response=True,
    # Importante: así se monta limpio en /mcp sin el sufijo /mcp
    streamable_http_path="/",
)


def load_mcp_tools() -> None:
    """
    Carga explícitamente módulos que registran tools en el servidor MCP.

    + Instala middleware de auth S2S (igual que hoy) y user-context (JWT opcional).
    + Idempotente: seguro llamar varias veces.
    """
    import importlib
    import pkgutil

    # Evitar doble carga
    loaded = getattr(load_mcp_tools, "_loaded", set())
    if not isinstance(loaded, set):
        loaded = set()

    def _safe_import(modname: str):
        if modname in loaded:
            return
        try:
            importlib.import_module(modname)
            loaded.add(modname)
            logger.info("[MCP] Tools module cargado: %s", modname)
        except Exception:
            logger.exception("[MCP] Error importando tools module: %s", modname)

    # 0) Instalar middleware auth/context (idempotente)
    try:
        from cosmos_mcp_app.mcp_auth import install_mcp_auth_middleware
        if not getattr(load_mcp_tools, "_auth_installed", False):
            install_mcp_auth_middleware(mcp_server)
            load_mcp_tools._auth_installed = True
    except Exception:
        logger.exception("[MCP] No se pudo instalar middleware de auth/context")

    # 1) Módulos explícitos (críticos)
    _safe_import("cosmos_mcp_app.ddg_tools")

    # 2) Autodescubrimiento dentro del paquete
    try:
        import cosmos_mcp_app as pkg

        for m in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
            name = m.name
            base = name.rsplit(".", 1)[-1].lower()
            if base.endswith("_tools") or base == "tools":
                _safe_import(name)

    except Exception:
        logger.exception("[MCP] Autodescubrimiento de tools falló")

    load_mcp_tools._loaded = loaded


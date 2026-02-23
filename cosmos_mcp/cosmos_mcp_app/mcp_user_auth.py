# cosmos_mcp_app/mcp_user_auth.py

from __future__ import annotations

import logging
from typing import Optional, Set

from cosmos_mcp_app.user_jwt import (
    USER_AUTH_HEADER,
    MCP_REQUIRE_USER_JWT,
    decode_and_validate_user_jwt,
)

logger = logging.getLogger(__name__)

DEFAULT_EXEMPT_PATHS = {
    "/health", "/healthz", "/ready", "/readyz", "/live", "/livez"
}


def install_mcp_user_context_middleware(mcp_server, *, exempt_paths: Optional[Set[str]] = None) -> None:
    """
    Middleware opcional:
    - Lee JWT desde X-User-Authorization (o env MCP_USER_AUTH_HEADER).
    - Valida/decodifica y expone request.state.user / request.state.user_id.
    - Si no hay JWT: deja pasar (modo legacy).
    - Si MCP_REQUIRE_USER_JWT=1: exige JWT válido para acceder a tools.
    """
    app = getattr(mcp_server, "app", None)
    if app is None:
        logger.warning("[MCP_USER] FastMCP no expone .app; no se pudo instalar middleware.")
        return

    exempt = set(exempt_paths or set()) | DEFAULT_EXEMPT_PATHS

    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        class MCPUserContextMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                path = request.url.path or ""
                if path in exempt:
                    return await call_next(request)

                raw = request.headers.get(USER_AUTH_HEADER, "") or ""
                if not raw:
                    # Modo legacy: no hay usuario
                    request.state.user = None
                    request.state.user_id = None
                    if MCP_REQUIRE_USER_JWT:
                        return JSONResponse(
                            status_code=401,
                            content={"error": "Unauthorized", "detail": "User JWT requerido (X-User-Authorization)."},
                        )
                    return await call_next(request)

                claims, err = decode_and_validate_user_jwt(raw)
                if err:
                    request.state.user = None
                    request.state.user_id = None
                    if MCP_REQUIRE_USER_JWT:
                        return JSONResponse(
                            status_code=401,
                            content={"error": "Unauthorized", "detail": f"User JWT inválido: {err}"},
                        )
                    # No rompemos: seguimos como anonymous
                    return await call_next(request)

                # Exponer user context
                request.state.user = claims
                request.state.user_id = claims.get("_user_id")

                return await call_next(request)

        app.add_middleware(MCPUserContextMiddleware)
        logger.info("[MCP_USER] Middleware instalado. require_user_jwt=%s header=%s", MCP_REQUIRE_USER_JWT, USER_AUTH_HEADER)

    except Exception as e:
        logger.exception("[MCP_USER] Error instalando middleware: %s", e)

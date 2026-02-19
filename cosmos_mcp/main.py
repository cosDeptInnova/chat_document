# cosmos_mcp/main.py

from __future__ import annotations

import os
import logging
import hmac
import inspect
from contextlib import asynccontextmanager, AsyncExitStack
from typing import Optional, Dict, Any, List
import time
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from functools import wraps
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from datetime import datetime, timezone
from api.routes import health_router, inference_router
from app.core import settings
from app.core.logging_config import configure_logging
from cosmos_mcp_app.server import mcp_server, load_mcp_tools
from services.llm_client import LLMClient
from services.tools_registry import GLOBAL_TOOL_REGISTRY, ToolSpec
from observability.metrics import (
    WEBSEARCH_TOOL_CALLS_TOTAL,
    WEBSEARCH_TOOL_DURATION_SECONDS,
    WEBSEARCH_AGENT_STEP_DURATION_SECONDS,
)

configure_logging()
logger = logging.getLogger(__name__)

MCP_SERVER_TOKEN = (
    getattr(settings, "MCP_GATEWAY_TOKEN", os.getenv("MCP_GATEWAY_TOKEN", "")) or ""
).strip()


class BearerAuthASGI:
    """
    Wrapper ASGI mínimo para proteger /mcp (S2S) y, opcionalmente,
    capturar contexto de usuario via JWT para cuotas/rate-limit per-user.

    Compatibilidad:
    - Si no hay token configurado => modo abierto (como hoy).
    - Si no se envía user JWT => user_id=None (no rompe).
    - Si JWT user es inválido y MCP_USER_JWT_REQUIRE=1 => 401.
    """

    def __init__(self, app, token: str):
        self.app = app
        self.token = (token or "").strip()
        self._token_bytes = self.token.encode("utf-8") if self.token else b""

    def _get_header(self, scope, header_name: bytes) -> str:
        try:
            raw_headers = scope.get("headers") or []
            for k, v in raw_headers:
                if k.lower() == header_name:
                    return v.decode("utf-8", errors="ignore")
        except Exception:
            return ""
        return ""

    def _extract_auth(self, scope) -> str:
        return self._get_header(scope, b"authorization")

    def _extract_user_auth(self, scope) -> str:
        # header configurable en mcp_auth, default: X-User-Authorization
        try:
            from cosmos_mcp_app.mcp_auth import MCP_USER_AUTH_HEADER
            header_bytes = MCP_USER_AUTH_HEADER.encode("utf-8").lower()
        except Exception:
            header_bytes = b"x-user-authorization"
        return self._get_header(scope, header_bytes)

    def _is_authorized(self, auth_header: str) -> bool:
        if not self.token:
            return True

        if not auth_header:
            return False

        if not auth_header.lower().startswith("bearer "):
            return False

        candidate = auth_header[7:].strip().encode("utf-8", errors="ignore")
        return hmac.compare_digest(candidate, self._token_bytes)

    async def _inject_user_context(self, scope) -> Optional[Dict[str, Any]]:
        """
        Decodifica/valida JWT user (best-effort) y mete:
          scope["state"]["user_id"]
          scope["state"]["user_claims"]
          scope["state"]["user_jwt_error"]
        """
        try:
            from cosmos_mcp_app.mcp_auth import (
                MCP_ENABLE_USER_JWT,
                MCP_USER_JWT_REQUIRE,
                decode_and_validate_user_jwt,
                _extract_user_id_from_claims,
            )
        except Exception:
            # No rompemos si faltan imports/configs
            return None

        if not MCP_ENABLE_USER_JWT:
            return None

        user_hdr = self._extract_user_auth(scope)
        claims, err = await decode_and_validate_user_jwt(user_hdr)

        state = scope.setdefault("state", {})
        state.setdefault("user_id", None)
        state.setdefault("user_claims", None)
        state.setdefault("user_jwt_error", None)

        if claims:
            state["user_claims"] = claims
            state["user_id"] = _extract_user_id_from_claims(claims)
            state["user_jwt_error"] = None
            return None

        if err:
            state["user_claims"] = None
            state["user_id"] = None
            state["user_jwt_error"] = err

            if MCP_USER_JWT_REQUIRE:
                return {"detail": "User JWT inválido o ausente", "code": err}

        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        # Modo abierto si no hay token S2S configurado
        if not self.token:
            # Aun así, si hay user jwt lo inyectamos best-effort
            err_payload = await self._inject_user_context(scope)
            if err_payload:
                # Rechazo WS
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                body = (str(err_payload).encode("utf-8")) if not isinstance(err_payload, dict) else (
                    ('{"detail":"%s","code":"%s"}' % (err_payload.get("detail"), err_payload.get("code"))).encode("utf-8")
                )
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return

            return await self.app(scope, receive, send)

        # Validación S2S
        auth = self._extract_auth(scope)
        if not self._is_authorized(auth):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return

            body = b'{"detail":"Unauthorized MCP access"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        # Inyectar user context (opcional)
        err_payload = await self._inject_user_context(scope)
        if err_payload:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            body = ('{"detail":"%s","code":"%s"}' % (err_payload.get("detail"), err_payload.get("code"))).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        return await self.app(scope, receive, send)

# -------------------------------------------------------------------
# Helpers: arranque robusto de MCP dentro del lifespan principal
# -------------------------------------------------------------------
async def _enter_mcp_runtime(stack: AsyncExitStack) -> None:
    """
    Inicializa el runtime del MCP Streamable HTTP cuando el servidor MCP
    está montado como sub-app.

    Soluciona:
        RuntimeError: Task group is not initialized. Make sure to use run().

    Estrategia tolerante a versiones:
    1) Si existe mcp_server.run() -> usarlo.
    2) Si no, mirar mcp_server.session_manager.run().
    3) Soporta que run() devuelva:
       - un async context manager
       - o un awaitable simple.
    """

    run_obj = None

    # 1) Preferido: run() en el servidor
    run_fn = getattr(mcp_server, "run", None)
    if callable(run_fn):
        try:
            run_obj = run_fn()
        except Exception as e:
            logger.warning("No se pudo invocar mcp_server.run(): %s", e)

    # 2) Fallback: run() en session_manager
    if run_obj is None:
        mgr = getattr(mcp_server, "session_manager", None)
        run_fn = getattr(mgr, "run", None) if mgr is not None else None
        if callable(run_fn):
            try:
                run_obj = run_fn()
            except Exception as e:
                logger.warning("No se pudo invocar session_manager.run(): %s", e)

    if run_obj is None:
        # No todas las versiones requieren esto explícitamente.
        logger.info("[MCP] No se detectó run() explícito; continuando sin runtime manual.")
        return

    # 3) Entrar como context manager si aplica
    if hasattr(run_obj, "__aenter__") and hasattr(run_obj, "__aexit__"):
        await stack.enter_async_context(run_obj)
        logger.info("[MCP] Runtime inicializado via async context manager.")
        return

    # 4) Si es awaitable, await directo
    if inspect.isawaitable(run_obj):
        await run_obj
        logger.info("[MCP] Runtime inicializado via awaitable run().")
        return

    # 5) Caso raro: tipo no soportado
    logger.warning("[MCP] run() devolvió un objeto no soportado: %r", type(run_obj))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan unificado:

    - Inicializa LLMClient para endpoints REST /api/v1/*
    - Garantiza runtime del MCP streamable HTTP
    - Cierre ordenado
    """
    async with AsyncExitStack() as stack:
        # ---------------------------
        # LLM client para REST
        # ---------------------------
        llama_client = LLMClient(
            base_url=settings.LLAMA_SERVER_BASE_URL,
            api_key=settings.LLAMA_SERVER_API_KEY,
            timeout=settings.LLAMA_REQUEST_TIMEOUT,
        )
        await llama_client.startup()
        app.state.llama_client = llama_client
        stack.push_async_callback(llama_client.shutdown)

        # ---------------------------
        # ✅ Runtime MCP
        # ---------------------------
        await _enter_mcp_runtime(stack)

        logger.info(
            "Cosmos MCP FastAPI app iniciada · LLM base_url=%s",
            settings.LLAMA_SERVER_BASE_URL,
        )

        yield

        logger.info("Cerrando Cosmos MCP FastAPI app")

def register_cosmos_registry_tools() -> None:
    """
    Registra herramientas de infraestructura del ecosistema dentro del MCP real.

    - Añade un listado enriquecido con metadata Cosmos (tags, for_crews).
    - No rompe el estándar MCP: es una tool adicional y opcional.
    """

    @mcp_server.tool(
        name="cosmos_registry_list_tools",
        description="Lista tools Cosmos con filtros por crew y tags (metadata enriquecida).",
    )
    async def cosmos_registry_list_tools(
        crew: Optional[str] = None,
        tags_any: Optional[str] = None,
        tags_all: Optional[str] = None,
    ) -> Dict[str, Any]:
        any_tags = [t.strip() for t in (tags_any or "").split(",") if t.strip()]
        all_tags = [t.strip() for t in (tags_all or "").split(",") if t.strip()]

        tools: List[ToolSpec] = GLOBAL_TOOL_REGISTRY.filter(
            crew_name=(crew or None),
            tags_any=any_tags or None,
            tags_all=all_tags or None,
        )

        payload: List[Dict[str, Any]] = []
        for t in tools:
            to_pub = getattr(t, "to_public_dict", None)
            if callable(to_pub):
                payload.append(to_pub())
            else:
                payload.append(
                    {
                        "name": getattr(t, "name", ""),
                        "description": getattr(t, "description", "") or "",
                        "tags": list(getattr(t, "tags", []) or []),
                        "for_crews": list(getattr(t, "for_crews", []) or []),
                        "extra": dict(getattr(t, "extra", {}) or {}),
                    }
                )

        return {"tools": payload, "count": len(payload)}
    

def create_app() -> FastAPI:
    patch_mcp_tool_decorator_for_metrics()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        lifespan=lifespan,
        redirect_slashes=False,
    )

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )

    app.include_router(health_router, prefix=settings.API_V1_PREFIX)
    app.include_router(inference_router, prefix=settings.API_V1_PREFIX)

    # Cargar tools MCP reales (ya instrumentadas por el patch)
    load_mcp_tools()

    # Registrar tool de infraestructura del bus (también instrumentada)
    register_cosmos_registry_tools()

    # Prewarm para evitar "No data"
    prewarm_mcp_tool_metrics()

    # Montar MCP real como BUS único de tools
    mcp_asgi = McpMetricsASGI(mcp_server.streamable_http_app())
    app.mount("/mcp", BearerAuthASGI(mcp_asgi, token=MCP_SERVER_TOKEN))

    logger.info(
        "[MCP] Bus montado en /mcp · auth=%s",
        "ON" if MCP_SERVER_TOKEN else "OFF",
    )

    return app

class McpMetricsASGI:
    """
    Wrapper ASGI para medir duración total del tráfico MCP (/mcp),
    y publicar pasos agregados en websearch_agent_step_duration_seconds{step=...}.

    Esto NO depende del payload MCP; mide el total por request (incluye streaming).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        start = time.perf_counter()
        try:
            return await self.app(scope, receive, send)
        finally:
            dur = time.perf_counter() - start
            # Step agregado: duración total de requests MCP
            WEBSEARCH_AGENT_STEP_DURATION_SECONDS.labels(step="mcp_http_total").observe(dur)

def patch_mcp_tool_decorator_for_metrics() -> None:
    """
    Parchea mcp_server.tool para que TODAS las tools registradas a partir de aquí
    queden instrumentadas automáticamente.

    - Incrementa websearch_tool_calls_total{tool,result}
    - Observa websearch_tool_duration_seconds{tool,result}
    - Observa websearch_agent_step_duration_seconds{step="mcp_tool_total"}
    """
    if getattr(mcp_server, "_cosmos_metrics_patched", False):
        return

    original_tool = mcp_server.tool

    def instrumented_tool(*, name: str, description: str, **kwargs):
        registrar = original_tool(name=name, description=description, **kwargs)

        def decorator(fn):
            tool_name = name or getattr(fn, "__name__", "unknown_tool")
            is_async = inspect.iscoroutinefunction(fn)

            if is_async:
                @wraps(fn)
                async def wrapped(*args, **kw):
                    start = time.perf_counter()
                    result = "ok"
                    try:
                        return await fn(*args, **kw)
                    except Exception:
                        result = "error"
                        raise
                    finally:
                        dur = time.perf_counter() - start
                        WEBSEARCH_TOOL_CALLS_TOTAL.labels(tool=tool_name, result=result).inc()
                        WEBSEARCH_TOOL_DURATION_SECONDS.labels(tool=tool_name, result=result).observe(dur)
                        WEBSEARCH_AGENT_STEP_DURATION_SECONDS.labels(step="mcp_tool_total").observe(dur)

                return registrar(wrapped)

            @wraps(fn)
            def wrapped_sync(*args, **kw):
                start = time.perf_counter()
                result = "ok"
                try:
                    return fn(*args, **kw)
                except Exception:
                    result = "error"
                    raise
                finally:
                    dur = time.perf_counter() - start
                    WEBSEARCH_TOOL_CALLS_TOTAL.labels(tool=tool_name, result=result).inc()
                    WEBSEARCH_TOOL_DURATION_SECONDS.labels(tool=tool_name, result=result).observe(dur)
                    WEBSEARCH_AGENT_STEP_DURATION_SECONDS.labels(step="mcp_tool_total").observe(dur)

            return registrar(wrapped_sync)

        return decorator

    mcp_server.tool = instrumented_tool  # monkey patch controlado
    setattr(mcp_server, "_cosmos_metrics_patched", True)
    logger.info("[MCP][metrics] mcp_server.tool parcheado para instrumentación.")

def prewarm_mcp_tool_metrics() -> None:
    """
    Crea (sin incrementar) los label-children para que /metrics ya exporte series
    y Grafana no muestre 'No data'.

    Importante: prometheus_client solo expone una combinación de labels
    después de que se haya creado al menos una vez via .labels(...)
    """
    try:
        tools = GLOBAL_TOOL_REGISTRY.filter(crew_name=None, tags_any=None, tags_all=None)
    except Exception:
        logger.exception("[MCP][metrics] No se pudo listar tools para precalentar métricas.")
        return

    # Prewarm pasos fijos
    WEBSEARCH_AGENT_STEP_DURATION_SECONDS.labels(step="mcp_http_total")
    WEBSEARCH_AGENT_STEP_DURATION_SECONDS.labels(step="mcp_tool_total")

    # Prewarm por tool
    for t in tools:
        tool_name = getattr(t, "name", None) or "unknown_tool"
        # OK / ERROR para que existan las series en ambos resultados
        WEBSEARCH_TOOL_CALLS_TOTAL.labels(tool=tool_name, result="ok")
        WEBSEARCH_TOOL_CALLS_TOTAL.labels(tool=tool_name, result="error")
        WEBSEARCH_TOOL_DURATION_SECONDS.labels(tool=tool_name, result="ok")
        WEBSEARCH_TOOL_DURATION_SECONDS.labels(tool=tool_name, result="error")

    logger.info("[MCP][metrics] Prewarm completado (%d tools).", len(tools))


app = create_app()

_START_TIME = time.monotonic()

@app.get("/health", tags=["health"])
async def health():
    uptime_seconds = int(time.monotonic() - _START_TIME)
    return {
        "status": "ok",
        "service": os.getenv("SERVICE_NAME", "unknown-service"),
        "uptime_seconds": uptime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
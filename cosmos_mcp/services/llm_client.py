# cosmos_mcp/services/llm_client.py

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import uuid
import httpx
import time

from app.core import settings
from models.inference import ChatCompletionRequest

logger = logging.getLogger(__name__)

try:
    from observability.metrics import (
    LLM_REQUESTS_TOTAL,
    LLM_ERRORS_TOTAL,
    LLM_INFLIGHT,
    LLM_LATENCY_SECONDS,
    LLM_TOKENS_TOTAL,
    LLM_TOKENS_PER_REQUEST,
    LLM_USAGE_MISSING_TOTAL,
)
    _METRICS_ENABLED = True
except Exception:
    _METRICS_ENABLED = False

# Fallback explícito al puerto donde tienes corriendo llama-server
DEFAULT_LLAMA_BASE_URL = "http://127.0.0.1:8002/v1"


class LLMClient:
    """
    Cliente asincrónico para un servidor llama.cpp con API OpenAI-compatible.

    - Usa httpx.AsyncClient con base_url tipo "http://127.0.0.1:8002/v1".
    - Reutiliza la conexión entre peticiones.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        effective_base = (
            base_url
            or getattr(settings, "LLAMA_SERVER_BASE_URL", None)
            or DEFAULT_LLAMA_BASE_URL
        )
        self._base_url = effective_base.rstrip("/")
        self._api_key = api_key or getattr(settings, "LLAMA_SERVER_API_KEY", None)
        self._timeout = timeout or getattr(settings, "LLAMA_REQUEST_TIMEOUT", 60.0)
        self._client: Optional[httpx.AsyncClient] = None

    async def startup(self) -> None:
        """Inicializa el cliente HTTP (se llama en el lifespan de FastAPI)."""
        if self._client is None:
            logger.info("Inicializando LLMClient con base_url=%s", self._base_url)
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )

    async def shutdown(self) -> None:
        """Cierra el cliente HTTP."""
        if self._client is not None:
            logger.info("Cerrando LLMClient")
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    def _build_headers(self) -> Dict[str, str]:
        """
        Construye los headers a enviar a llama.cpp.

        Punto crítico: httpx exige que las cabeceras sean ASCII. Si la API key
        tiene acentos u otros caracteres no-ASCII, evitamos reventar con
        UnicodeEncodeError y simplemente NO enviamos Authorization.
        """
        headers: Dict[str, str] = {}

        if self._api_key:
            auth_value = f"Bearer {self._api_key}"
            try:
                # Validamos que sea ASCII-safe
                auth_value.encode("ascii")
            except UnicodeEncodeError:
                logger.error(
                    "LLAMA_SERVER_API_KEY contiene caracteres no ASCII; "
                    "se omite el header Authorization para evitar errores de codificación."
                )
            else:
                headers["Authorization"] = auth_value

        return headers

    # ------------------------------------------------------------------
    # Llamada principal al modelo
    # ------------------------------------------------------------------
    async def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """
        Llama al endpoint /chat/completions de llama.cpp.

        Devuelve el JSON completo tal cual responde llama.cpp.
        """
        if self._client is None:
            raise RuntimeError("LLMClient no inicializado. Llama a startup() primero.")

        # ID único por llamada para seguirla en logs
        call_id = uuid.uuid4().hex[:8]

        headers = self._build_headers()
        payload = request.to_payload()

        messages = payload.get("messages", []) or []
        model = payload.get("model")
        max_tokens = payload.get("max_tokens")
        temperature = payload.get("temperature")

        # Log de alto nivel de la llamada
        logger.info(
            "[LLM_CALL %s] → llama.cpp /chat/completions "
            "(model=%s, n_messages=%d, max_tokens=%s, temperature=%s)",
            call_id,
            model,
            len(messages),
            max_tokens,
            temperature,
        )

        # Detalle del payload solo en DEBUG para no reventar logs
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[LLM_CALL %s] payload (truncado)=%r",
                call_id,
                str(payload)[:2000],
            )
            for i, m in enumerate(messages):
                content = m.get("content", "") or ""
                logger.debug(
                    "[LLM_CALL %s] msg[%d] role=%s len=%d: %r",
                    call_id,
                    i,
                    m.get("role"),
                    len(content),
                    content[:500],
                )

        start = time.perf_counter()
        backend = "unknown"
        moved_inflight = False
        recorded = False

        if _METRICS_ENABLED:
            # Al inicio todavía no sabemos el backend real; Nginx lo devuelve en headers
            LLM_INFLIGHT.labels(backend="unknown").inc()

        try:
            response = await self._client.post(
                "chat/completions",
                json=payload,
                headers=headers or None,
            )

            # Backend real: lo añade tu Nginx (X-LLM-Backend)
            backend = (response.headers.get("X-LLM-Backend") or "unknown").strip() or "unknown"

            # Transferimos inflight de "unknown" -> backend real (una sola vez)
            if _METRICS_ENABLED and not moved_inflight:
                LLM_INFLIGHT.labels(backend="unknown").dec()
                LLM_INFLIGHT.labels(backend=backend).inc()
                moved_inflight = True

            logger.info(
                "[LLM_CALL %s] ← llama.cpp status=%s",
                call_id,
                response.status_code,
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[LLM_CALL %s] raw_response (truncado)=%s",
                    call_id,
                    response.text[:2000],
                )

            # Si hay error HTTP aquí, saltará a except HTTPStatusError
            response.raise_for_status()

            # Intentamos parsear JSON
            try:
                data = response.json()
            except ValueError:
                logger.error(
                    "[LLM_CALL %s] Respuesta de llama.cpp no es JSON. Status=%s, body=%s",
                    call_id,
                    response.status_code,
                    response.text,
                )
                if _METRICS_ENABLED:
                    dur = time.perf_counter() - start
                    LLM_ERRORS_TOTAL.labels(backend=backend, kind="json").inc()
                    LLM_REQUESTS_TOTAL.labels(backend=backend, status_class="5xx").inc()
                    LLM_LATENCY_SECONDS.labels(backend=backend).observe(dur)
                    recorded = True
                raise httpx.HTTPStatusError(
                    "Non-JSON response from llama.cpp",
                    request=response.request,
                    response=response,
                )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[LLM_CALL %s] parsed_json (truncado)=%s",
                    call_id,
                    str(data)[:2000],
                )

            #Métricas de éxito
            if _METRICS_ENABLED and not recorded:
                dur = time.perf_counter() - start
                status_class = f"{response.status_code // 100}xx"
                LLM_REQUESTS_TOTAL.labels(backend=backend, status_class=status_class).inc()
                LLM_LATENCY_SECONDS.labels(backend=backend).observe(dur)

                usage = data.get("usage") or {}
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
                tt = usage.get("total_tokens")

                if not usage:
                    LLM_USAGE_MISSING_TOTAL.labels(backend=backend).inc()

                if isinstance(pt, int):
                    LLM_TOKENS_TOTAL.labels(backend=backend, type="prompt").inc(pt)
                    LLM_TOKENS_PER_REQUEST.labels(backend=backend, type="prompt").observe(pt)

                if isinstance(ct, int):
                    LLM_TOKENS_TOTAL.labels(backend=backend, type="completion").inc(ct)
                    LLM_TOKENS_PER_REQUEST.labels(backend=backend, type="completion").observe(ct)

                if isinstance(tt, int):
                    LLM_TOKENS_TOTAL.labels(backend=backend, type="total").inc(tt)
                    LLM_TOKENS_PER_REQUEST.labels(backend=backend, type="total").observe(tt)


                recorded = True

            return data

        except httpx.HTTPStatusError as http_err:
            # Backend puede estar disponible en headers aunque haya error
            try:
                resp = http_err.response
                if resp is not None:
                    backend = (resp.headers.get("X-LLM-Backend") or backend or "unknown").strip() or "unknown"
            except Exception:
                backend = backend or "unknown"

            logger.error(
                "[LLM_CALL %s] Error HTTP llamando a llama.cpp: "
                "status=%s url=%s body=%s",
                call_id,
                http_err.response.status_code if http_err.response is not None else "N/A",
                str(http_err.request.url) if http_err.request is not None else "N/A",
                http_err.response.text if http_err.response is not None else "",
            )

            if _METRICS_ENABLED and not recorded:
                dur = time.perf_counter() - start
                status_code = http_err.response.status_code if http_err.response is not None else 500
                status_class = f"{status_code // 100}xx"
                LLM_ERRORS_TOTAL.labels(backend=backend, kind="http_status").inc()
                LLM_REQUESTS_TOTAL.labels(backend=backend, status_class=status_class).inc()
                LLM_LATENCY_SECONDS.labels(backend=backend).observe(dur)
                recorded = True

            raise

        except httpx.RequestError as req_err:
            logger.error(
                "[LLM_CALL %s] Error de red llamando a llama.cpp: %s",
                call_id,
                req_err,
            )

            if _METRICS_ENABLED and not recorded:
                dur = time.perf_counter() - start
                LLM_ERRORS_TOTAL.labels(backend=backend or "unknown", kind="network").inc()
                LLM_REQUESTS_TOTAL.labels(backend=backend or "unknown", status_class="5xx").inc()
                LLM_LATENCY_SECONDS.labels(backend=backend or "unknown").observe(dur)
                recorded = True

            raise

        except Exception:
            # Cualquier excepción inesperada: no altera el flujo, solo contabiliza si hay métricas
            logger.exception("[LLM_CALL %s] Error inesperado en chat_completion()", call_id)
            if _METRICS_ENABLED and not recorded:
                dur = time.perf_counter() - start
                LLM_ERRORS_TOTAL.labels(backend=backend or "unknown", kind="unknown").inc()
                LLM_REQUESTS_TOTAL.labels(backend=backend or "unknown", status_class="5xx").inc()
                LLM_LATENCY_SECONDS.labels(backend=backend or "unknown").observe(dur)
                recorded = True
            raise

        finally:
            if _METRICS_ENABLED:
                # Si hicimos transfer a backend real, decrementa ese; si no, decrementa unknown
                try:
                    if moved_inflight:
                        LLM_INFLIGHT.labels(backend=backend or "unknown").dec()
                    else:
                        LLM_INFLIGHT.labels(backend="unknown").dec()
                except Exception:
                    # Nunca romper por métricas
                    pass

    async def health_check(self) -> bool:
        """
        Comprueba que llama.cpp está vivo.

        Intenta una petición rápida (GET /health). Si falla, devuelve False.
        """
        if self._client is None:
            raise RuntimeError("LLMClient no inicializado. Llama a startup() primero.")

        try:
            resp = await self._client.get("/health")
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Health check a llama.cpp falló: %s", exc)
            return False

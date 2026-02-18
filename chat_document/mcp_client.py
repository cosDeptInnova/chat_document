# chat_document/mcp_client.py
"""
Cliente MCP para el microservicio chat_document.

Objetivos:
- Consumir tools vía MCP Streamable HTTP (bus real).
- Mantener interfaz estable ya asumida por el orquestador:
    * list_tools(crew=...)
    * alist_tools(crew=...)  (opcional)
    * invoke_tool(...)
    * ainvoke_tool(...)
- Propagar autenticación al gateway MCP vía Authorization Bearer.
- NO modificar el payload funcional de tools:
  si el orquestador incluye `access_token` en el payload,
  este cliente lo envía tal cual.

Configuración:
- CHATDOC_MCP_SERVER_URL | MCP_SERVER_URL | MCP_GATEWAY_BASE_URL
    default: http://localhost:8090/mcp
- CHATDOC_MCP_SERVICE_TOKEN | MCP_SERVICE_TOKEN | MCP_AUTH_TOKEN | MCP_GATEWAY_TOKEN
    token Bearer del bus (S2S)

Timeouts / retries:
- MCP_TIMEOUT_S (default 30)
- MCP_HTTP_TIMEOUT (default 15)
- MCP_MAX_RETRIES (default 2)
- MCP_BACKOFF_BASE (default 0.35)

Notas:
- Camino feliz: fastmcp.Client sobre Streamable HTTP.
- Fallback HTTP: POST /mcp/call_tool si existe
  y/o JSON-RPC sobre POST /mcp/.
"""

from __future__ import annotations

import os
import re
import time
import json
import asyncio
import logging
import threading
import importlib
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("mcp_client")


# ---------------------------------------------------------------------
# Errores
# ---------------------------------------------------------------------
class MCPClientError(RuntimeError):
    """Errores del cliente MCP (bus real)."""
    pass


# ---------------------------------------------------------------------
# Utilidades env/token/url
# ---------------------------------------------------------------------
def _env_first(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _normalize_bearer(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    t = str(token).strip()
    if not t:
        return None
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    return t or None


def _normalize_mcp_url(raw: str) -> str:
    """
    Normaliza URL hacia forma canónica:
      http(s)://host[:port]/mcp/

    Blindajes:
    - Si pasan base REST /api/v1 -> reconducir a /mcp
    - Si pasan /mcp/call_tool -> recortar a /mcp
    - Evitar dobles slashes
    """
    candidate = (raw or "").strip()
    if not candidate:
        return "http://localhost:8090/mcp/"

    # Si alguien pasa base REST por error
    try:
        if "/api/v1" in candidate:
            base = candidate.split("/api/v1", 1)[0].rstrip("/")
            candidate = base + "/mcp"
    except Exception:
        pass

    # Si alguien pasó /mcp/call_tool u otras rutas bajo /mcp
    candidate = re.sub(r"/mcp/(call_tool|list_tools|tools).*?$", "/mcp", candidate)

    # Quitar trailing spaces/slashes extra
    candidate = candidate.rstrip()

    # Garantizar sufijo /mcp/
    if candidate.endswith("/mcp"):
        candidate = candidate + "/"
    elif candidate.endswith("/mcp/"):
        pass
    else:
        # Si no contiene /mcp al final, intentamos añadirlo
        candidate = candidate.rstrip("/") + "/mcp/"

    # Normalizar múltiples /
    candidate = re.sub(r"/mcp/+$", "/mcp/", candidate)

    return candidate


# ---------------------------------------------------------------------
# Cliente MCP
# ---------------------------------------------------------------------
class MCPClient:
    """
    Cliente MCP-only con fallback HTTP.

    Interfaz estable:
    - list_tools(..., auth_token=..., timeout=...)
    - alist_tools(...)
    - invoke_tool(name, payload, auth_token=..., timeout=...)
    - ainvoke_tool(...)

    Seguridad:
    - auth_token autentica el gateway MCP (Bearer header).
    - access_token del usuario puede viajar en payload para que el tool
      lo reenvíe al microservicio NLP protegido.
    """

    # Cache de clases importadas (fastmcp)
    _FAST_CLIENT_CLS: Optional[type] = None
    _FAST_BEARER_CLS: Optional[type] = None
    _FAST_TRANSPORT_CLS: Optional[type] = None
    _FAST_RESOLVE_ERROR: Optional[Exception] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_s: Optional[float] = None,
        auth_token: Optional[str] = None,
        verify_ssl: bool = True,
        max_retries: Optional[int] = None,
        backoff_base: Optional[float] = None,
    ) -> None:
        # URL
        env_server = _env_first("CHATDOC_MCP_SERVER_URL", "MCP_SERVER_URL", "MCP_GATEWAY_BASE_URL")
        raw_candidate = (base_url or env_server or "http://localhost:8090/mcp").strip()
        self.base_url = _normalize_mcp_url(raw_candidate).rstrip("/")  # guardamos sin slash final para concatenar fácil

        # Timeouts/retries
        try:
            self.timeout_s = float(timeout_s or os.getenv("MCP_TIMEOUT_S", os.getenv("MCP_HTTP_TIMEOUT", "15")))
        except Exception:
            self.timeout_s = 15.0

        self.verify_ssl = verify_ssl
        self.max_retries = int(max_retries or os.getenv("MCP_MAX_RETRIES", "2"))
        self.backoff_base = float(backoff_base or os.getenv("MCP_BACKOFF_BASE", "0.35"))

        # Token por defecto (bus)
        self.default_auth_token = _normalize_bearer(
            auth_token
            or _env_first(
                "CHATDOC_MCP_SERVICE_TOKEN",
                "MCP_SERVICE_TOKEN",
                "MCP_AUTH_TOKEN",
                "MCP_GATEWAY_TOKEN",
            )
        )

        # lock para construir client fastmcp
        self._lock = asyncio.Lock()

        logger.info("[MCPClient] base_url=%s timeout=%.1fs", self.base_url, self.timeout_s)

    # ------------------------------------------------------------------
    # Backoff
    # ------------------------------------------------------------------
    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.backoff_base * (2 ** attempt)
        time.sleep(delay)

    async def _asleep_backoff(self, attempt: int) -> None:
        delay = self.backoff_base * (2 ** attempt)
        await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Header / token resolución
    # ------------------------------------------------------------------
    def _resolve_gateway_token(self, auth_token: Optional[str]) -> Optional[str]:
        return _normalize_bearer(auth_token) or self.default_auth_token

    def _build_headers(self, auth_token: Optional[str]) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        tok = self._resolve_gateway_token(auth_token)
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
        return headers

    # ------------------------------------------------------------------
    # fastmcp imports tolerantes
    # ------------------------------------------------------------------
    @classmethod
    def _try_import_attr(cls, module_name: str, attr: str):
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, attr, None)
        except Exception:
            return None

    @classmethod
    def _resolve_fastmcp(cls) -> Tuple[type, Optional[type], Optional[type]]:
        """
        Devuelve (Client, BearerAuth, StreamableHttpTransport).
        """
        if cls._FAST_CLIENT_CLS:
            return cls._FAST_CLIENT_CLS, cls._FAST_BEARER_CLS, cls._FAST_TRANSPORT_CLS

        if cls._FAST_RESOLVE_ERROR:
            raise cls._FAST_RESOLVE_ERROR

        try:
            Client = (
                cls._try_import_attr("fastmcp", "Client")
                or cls._try_import_attr("fastmcp.client", "Client")
            )
            if Client is None:
                raise ImportError(
                    "No se pudo importar fastmcp.Client. "
                    "Instala 'fastmcp' en el microservicio chat_document."
                )

            BearerAuth = cls._try_import_attr("fastmcp.client.auth", "BearerAuth")

            Transport = (
                cls._try_import_attr("fastmcp.client.transports", "StreamableHttpTransport")
                or cls._try_import_attr("fastmcp.client.transports", "StreamableHTTPTransport")
            )

            cls._FAST_CLIENT_CLS = Client
            cls._FAST_BEARER_CLS = BearerAuth
            cls._FAST_TRANSPORT_CLS = Transport

            return Client, BearerAuth, Transport

        except Exception as e:
            cls._FAST_RESOLVE_ERROR = e
            raise

    def _build_auth_obj(self, BearerAuth, token: Optional[str]):
        tok = _normalize_bearer(token)
        if not tok or BearerAuth is None:
            return None
        try:
            return BearerAuth(tok)
        except Exception:
            return None

    def _build_fast_client(self, *, auth_token: Optional[str] = None):
        """
        Construye fastmcp.Client de forma tolerante.

        Prioridad:
        1) Client(url, auth=BearerAuth(token))
        2) Client(url, auth=token)
        3) Client(transport=StreamableHttpTransport(url, headers=..., timeout=...))
        4) Client(url) sin auth
        """
        Client, BearerAuth, Transport = self._resolve_fastmcp()

        tok = self._resolve_gateway_token(auth_token)

        auth_obj = self._build_auth_obj(BearerAuth, tok)

        # 1) URL + auth obj
        if auth_obj is not None:
            try:
                return Client(self.base_url + "/", auth=auth_obj)
            except TypeError:
                pass
            except Exception:
                pass

        # 2) URL + token string
        if tok:
            try:
                return Client(self.base_url + "/", auth=tok)
            except TypeError:
                pass
            except Exception:
                pass

        # 3) Transport explícito
        if Transport is not None:
            headers = self._build_headers(auth_token=auth_token)
            transport = None

            try:
                transport = Transport(url=self.base_url + "/", headers=headers or None, timeout=self.timeout_s)
            except TypeError:
                try:
                    transport = Transport(url=self.base_url + "/", headers=headers or None)
                except TypeError:
                    try:
                        transport = Transport(self.base_url + "/", headers=headers or None)
                    except Exception:
                        transport = None
                except Exception:
                    transport = None
            except Exception:
                transport = None

            if transport is not None:
                try:
                    return Client(transport=transport)
                except TypeError:
                    return Client(transport)

        # 4) Sin auth
        return Client(self.base_url + "/")

    # ------------------------------------------------------------------
    # Unwrap de resultados
    # ------------------------------------------------------------------
    def _unwrap(self, obj: Any) -> Any:
        """
        Normaliza resultados de fastmcp/Streamable HTTP y JSON-RPC.

        Soporta:
        - objetos ToolResult con .data / .result / .content / .value
        - dict wrapper: {"data": ...}, {"result": ...}
        - envelopes JSON-RPC: {"jsonrpc":..., "id":..., "result": ...}
        - list_tools style: {"tools":[...]}
        """
        if obj is None:
            return None

        if isinstance(obj, (list, str, int, float, bool)):
            return obj

        # --- Dicts ---
        if isinstance(obj, dict):

            # 1) JSON-RPC envelope explícito
            #    (tools/call y tools/list suelen venir así)
            if "result" in obj and ("jsonrpc" in obj or "id" in obj):
                return self._unwrap(obj.get("result"))

            # 2) data/result wrappers simples
            if "data" in obj and len(obj.keys()) == 1:
                return self._unwrap(obj.get("data"))
            if "result" in obj and len(obj.keys()) == 1:
                return self._unwrap(obj.get("result"))

            # 3) tools list wrapper
            if "tools" in obj and isinstance(obj.get("tools"), list):
                # tolerante: aunque haya meta adicional
                if len(obj.keys()) <= 3:
                    return obj.get("tools")

            return obj

        # --- Objetos SDK ---
        for attr in ("data", "result", "content", "value"):
            try:
                v = getattr(obj, attr, None)
                if v is not None:
                    return self._unwrap(v)
            except Exception:
                pass

        return obj


    # ------------------------------------------------------------------
    # Filtrado local de tools
    # ------------------------------------------------------------------
    def _filter_tools_client_side(
        self,
        tools: List[Dict[str, Any]],
        *,
        crew: Optional[str] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not tools:
            return []

        crew = (crew or "").strip() or None
        any_tags = [str(t).strip() for t in (tags_any or []) if str(t).strip()]
        all_tags = [str(t).strip() for t in (tags_all or []) if str(t).strip()]

        out: List[Dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if not name:
                continue

            if crew:
                fc = t.get("for_crews") or []
                if isinstance(fc, list) and fc:
                    if "*" not in fc and crew not in fc:
                        continue

            if any_tags:
                tt = t.get("tags") or []
                if isinstance(tt, list):
                    if not any(tag in tt for tag in any_tags):
                        continue

            if all_tags:
                tt = t.get("tags") or []
                if isinstance(tt, list):
                    if not all(tag in tt for tag in all_tags):
                        continue

            out.append(t)

        return out

    # ------------------------------------------------------------------
    # HTTP fallback: JSON-RPC + /call_tool si existe
    # ------------------------------------------------------------------
    async def _alist_tools_http_fallback(
        self,
        *,
        crew: Optional[str],
        tags_any: Optional[List[str]],
        tags_all: Optional[List[str]],
        auth_token: Optional[str],
        timeout: Optional[float],
    ) -> List[Dict[str, Any]]:
        headers = self._build_headers(auth_token)
        base = self.base_url + "/"

        # 0) Construir params para registry enriquecido
        payload: Dict[str, Any] = {}
        if crew:
            payload["crew"] = crew
        if tags_any:
            payload["tags_any"] = ",".join([t for t in tags_any if t])
        if tags_all:
            payload["tags_all"] = ",".join([t for t in tags_all if t])

        async with httpx.AsyncClient(timeout=timeout or self.timeout_s, verify=self.verify_ssl) as client:
            # 1) Intento clásico /call_tool con registry enriquecido
            try:
                r = await client.post(
                    base + "call_tool",
                    json={"name": "cosmos_registry_list_tools", "arguments": payload},
                    headers=headers or None,
                )
                if r.status_code < 400:
                    data = self._unwrap(r.json())
                    if isinstance(data, dict) and isinstance(data.get("tools"), list):
                        tools = data.get("tools") or []
                        return self._filter_tools_client_side(tools, crew=crew, tags_any=tags_any, tags_all=tags_all)
                    if isinstance(data, list):
                        return self._filter_tools_client_side(data, crew=crew, tags_any=tags_any, tags_all=tags_all)
            except Exception:
                pass

            # 2) JSON-RPC directo contra root /mcp/
            # tools/list
            try:
                rpc = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": payload or {},
                }
                r = await client.post(base, json=rpc, headers=headers or None)
                if r.status_code < 400:
                    data = self._unwrap(r.json())
                    # Algunas implementaciones devuelven {"result":{"tools":[...]}}
                    if isinstance(data, dict):
                        tools = None
                        if isinstance(data.get("tools"), list):
                            tools = data.get("tools")
                        else:
                            res = data.get("result")
                            if isinstance(res, dict) and isinstance(res.get("tools"), list):
                                tools = res.get("tools")
                        if tools:
                            return self._filter_tools_client_side(tools, crew=crew, tags_any=tags_any, tags_all=tags_all)
                    if isinstance(data, list):
                        return self._filter_tools_client_side(data, crew=crew, tags_any=tags_any, tags_all=tags_all)
            except Exception:
                pass

        return []

    async def _ainvoke_tool_http_fallback(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        auth_token: Optional[str],
        timeout: Optional[float],
    ) -> Any:
        """
        Fallback HTTP:
        1) POST /mcp/call_tool (si existe)
        2) JSON-RPC tools/call contra /mcp/

        Logging:
        - Qué ruta se usó
        - Status code
        - Keys de respuesta raw/unwrapped
        """
        headers = self._build_headers(auth_token)
        base = self.base_url + "/"
        safe_args = args or {}

        client_timeout = timeout or self.timeout_s

        async with httpx.AsyncClient(timeout=client_timeout, verify=self.verify_ssl) as client:
            # 1) Intento clásico /call_tool
            try:
                r = await client.post(
                    base + "call_tool",
                    json={"name": name, "arguments": safe_args},
                    headers=headers or None,
                )

                if r.status_code < 400:
                    try:
                        raw = r.json()
                    except Exception as je:
                        logger.warning(
                            "[MCPClient][fallback:/call_tool] JSON decode error name=%s status=%s err=%s",
                            name, r.status_code, str(je)
                        )
                        raw = None

                    unwrapped = self._unwrap(raw)

                    logger.info(
                        "[MCPClient][fallback:/call_tool] OK name=%s status=%s raw_keys=%s unwrapped_type=%s unwrapped_keys=%s",
                        name,
                        r.status_code,
                        list(raw.keys()) if isinstance(raw, dict) else None,
                        type(unwrapped).__name__,
                        list(unwrapped.keys()) if isinstance(unwrapped, dict) else None,
                    )
                    return unwrapped

                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[MCPClient][fallback:/call_tool] non-2xx name=%s status=%s",
                            name, r.status_code
                        )

            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[MCPClient][fallback:/call_tool] exception name=%s err=%s",
                        name, str(e)
                    )

            # 2) JSON-RPC tools/call contra root
            try:
                rpc = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": safe_args},
                }

                r = await client.post(base, json=rpc, headers=headers or None)
                r.raise_for_status()

                try:
                    raw = r.json()
                except Exception as je:
                    logger.warning(
                        "[MCPClient][fallback:jsonrpc] JSON decode error name=%s status=%s err=%s",
                        name, r.status_code, str(je)
                    )
                    raw = None

                unwrapped = self._unwrap(raw)

                logger.info(
                    "[MCPClient][fallback:jsonrpc] OK name=%s status=%s raw_keys=%s unwrapped_type=%s unwrapped_keys=%s",
                    name,
                    r.status_code,
                    list(raw.keys()) if isinstance(raw, dict) else None,
                    type(unwrapped).__name__,
                    list(unwrapped.keys()) if isinstance(unwrapped, dict) else None,
                )

                return unwrapped

            except Exception as e:
                logger.error(
                    "[MCPClient][fallback] invoke_tool FAILED name=%s err=%s",
                    name, str(e)
                )
                raise MCPClientError(f"Fallback HTTP invoke_tool falló para '{name}': {e}") from e


    # ------------------------------------------------------------------
    # MCP vía fastmcp
    # ------------------------------------------------------------------
    async def _alist_tools_mcp(
        self,
        *,
        crew: Optional[str],
        tags_any: Optional[List[str]],
        tags_all: Optional[List[str]],
        auth_token: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        1) Intenta cosmos_registry_list_tools si existe.
        2) Fallback a client.list_tools().
        3) Si fastmcp falla, HTTP fallback.
        """
        try:
            async with self._lock:
                client = self._build_fast_client(auth_token=auth_token)

            async with client:
                # 1) registry enriquecido
                try:
                    payload: Dict[str, Any] = {}
                    if crew:
                        payload["crew"] = crew
                    if tags_any:
                        payload["tags_any"] = ",".join([t for t in tags_any if t])
                    if tags_all:
                        payload["tags_all"] = ",".join([t for t in tags_all if t])

                    data = await client.call_tool("cosmos_registry_list_tools", payload)
                    data = self._unwrap(data)

                    if isinstance(data, dict) and isinstance(data.get("tools"), list):
                        tools = data.get("tools") or []
                        return self._filter_tools_client_side(tools, crew=crew, tags_any=tags_any, tags_all=tags_all)
                    if isinstance(data, list):
                        return self._filter_tools_client_side(data, crew=crew, tags_any=tags_any, tags_all=tags_all)
                except Exception:
                    pass

                # 2) listado estándar
                raw = await client.list_tools()
                raw = self._unwrap(raw)

                tools_out: List[Dict[str, Any]] = []

                if isinstance(raw, dict) and isinstance(raw.get("tools"), list):
                    raw = raw.get("tools") or []

                if isinstance(raw, list):
                    for t in raw:
                        if isinstance(t, dict):
                            tools_out.append(t)
                            continue

                        # objetos tool del SDK
                        name = getattr(t, "name", None)
                        desc = getattr(t, "description", None)
                        args = getattr(t, "arguments", None) or getattr(t, "inputSchema", None)
                        fc = getattr(t, "for_crews", None)
                        tags = getattr(t, "tags", None)

                        if name:
                            item = {"name": name, "description": desc or "", "arguments": args}
                            if fc:
                                item["for_crews"] = fc
                            if tags:
                                item["tags"] = tags
                            tools_out.append(item)

                return self._filter_tools_client_side(tools_out, crew=crew, tags_any=tags_any, tags_all=tags_all)

        except Exception as e:
            logger.warning("fastmcp.list_tools falló, usando fallback HTTP: %s", e)

        return await self._alist_tools_http_fallback(
            crew=crew,
            tags_any=tags_any,
            tags_all=tags_all,
            auth_token=auth_token,
            timeout=None,
        )

    async def _ainvoke_tool_mcp(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        auth_token: Optional[str],
        timeout: Optional[float],
    ) -> Any:
        """
        Intenta invocar tool vía fastmcp.
        Si falla, cae a HTTP fallback.

        Logging:
        - Tipo de respuesta raw vs unwrapped.
        - Keys principales si hay dict.
        """
        safe_args = args or {}

        # --- vía fastmcp ---
        try:
            async with self._lock:
                client = self._build_fast_client(auth_token=auth_token)

            async with client:
                res = await client.call_tool(name, safe_args)
                unwrapped = self._unwrap(res)

                try:
                    raw_type = type(res).__name__
                except Exception:
                    raw_type = "unknown"

                uw_type = type(unwrapped).__name__
                uw_keys = list(unwrapped.keys()) if isinstance(unwrapped, dict) else None

                logger.info(
                    "[MCPClient] call_tool OK name=%s raw_type=%s unwrapped_type=%s unwrapped_keys=%s",
                    name, raw_type, uw_type, uw_keys
                )

                # Debug extra: solo si hace falta
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[MCPClient] call_tool DEBUG name=%s args_keys=%s",
                        name, list(safe_args.keys())
                    )

                return unwrapped

        except Exception as e:
            logger.warning(
                "fastmcp.call_tool falló, usando fallback HTTP: name=%s err=%s",
                name, str(e)
            )

        # --- fallback HTTP ---
        return await self._ainvoke_tool_http_fallback(
            name,
            safe_args,
            auth_token=auth_token,
            timeout=timeout,
        )


    # ------------------------------------------------------------------
    # API ASYNC pública
    # ------------------------------------------------------------------
    async def alist_tools(
        self,
        *,
        crew: Optional[str] = None,
        crew_name: Optional[str] = None,
        namespace: Optional[str] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        auth_token: Optional[str] = None,
        timeout: Optional[float] = None,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        crew_value = (crew or crew_name or namespace or "").strip() or None

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._alist_tools_mcp(
                    crew=crew_value,
                    tags_any=tags_any,
                    tags_all=tags_all,
                    auth_token=auth_token,
                )
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    await self._asleep_backoff(attempt)
                else:
                    break

        raise MCPClientError(f"Fallo list_tools MCP: {last_exc}")

    async def ainvoke_tool(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        auth_token: Optional[str] = None,
        timeout: Optional[float] = None,
        **_ignored: Any,
    ) -> Any:
        if not name or not isinstance(name, str):
            raise MCPClientError("ainvoke_tool requiere un nombre de tool válido.")

        args = dict(payload or {})

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._ainvoke_tool_mcp(
                    name,
                    args,
                    auth_token=auth_token,
                    timeout=timeout,
                )
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    await self._asleep_backoff(attempt)
                else:
                    break

        raise MCPClientError(f"Fallo invoke_tool MCP '{name}': {last_exc}")

    # Aliases async de compatibilidad
    async def call_tool_async(self, name: str, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        return await self.ainvoke_tool(name, payload, **kwargs)

    # ------------------------------------------------------------------
    # API SYNC pública
    # ------------------------------------------------------------------
    def list_tools(
        self,
        *,
        crew: Optional[str] = None,
        crew_name: Optional[str] = None,
        namespace: Optional[str] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        auth_token: Optional[str] = None,
        timeout: Optional[float] = None,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        return self._run_sync(
            self.alist_tools(
                crew=crew,
                crew_name=crew_name,
                namespace=namespace,
                tags_any=tags_any,
                tags_all=tags_all,
                auth_token=auth_token,
                timeout=timeout,
            )
        )

    def invoke_tool(
        self,
        name: str,
        payload: Dict[str, Any],
        *,
        auth_token: Optional[str] = None,
        timeout: Optional[float] = None,
        **_ignored: Any,
    ) -> Any:
        return self._run_sync(self.ainvoke_tool(name, payload, auth_token=auth_token, timeout=timeout))

    # Alias sync de compatibilidad
    def call_tool(self, name: str, payload: Dict[str, Any], **kwargs: Any) -> Any:
        return self.invoke_tool(name, payload, **kwargs)

    # ------------------------------------------------------------------
    # Runner sync robusto (como web_search)
    # ------------------------------------------------------------------
    def _run_sync(self, coro):
        """
        Ejecuta coroutine desde contexto sync.

        - Sin loop activo en este hilo: asyncio.run
        - Con loop activo: ejecuta en hilo auxiliar
        """
        try:
            asyncio.get_running_loop()
            result_container = {"value": None, "error": None}

            def _worker():
                try:
                    result_container["value"] = asyncio.run(coro)
                except Exception as e:
                    result_container["error"] = e

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            t.join(timeout=self.timeout_s * 3)

            if result_container["error"]:
                raise result_container["error"]
            return result_container["value"]

        except RuntimeError:
            return asyncio.run(coro)

    # ------------------------------------------------------------------
    # Cierre (no mantenemos sesión persistente)
    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        return None

    def close(self) -> None:
        return None

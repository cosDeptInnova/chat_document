# web_search/mcp_client.py

from __future__ import annotations

import os
import time
import asyncio
import logging
import threading
import importlib
from typing import Any, Dict, List, Optional, Tuple
import re

from utils import _normalize_mcp_url

logger = logging.getLogger(__name__)


class MCPClientError(RuntimeError):
    """Errores del cliente MCP (bus real)."""
    pass


def _env_first(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _normalize_bearer(token: Optional[str]) -> str:
    """
    Devuelve el token crudo sin prefijo Bearer.
    Evita 'Bearer Bearer xxx' y errores con BearerAuth.
    """
    if not token:
        return ""
    t = str(token).strip()
    if not t:
        return ""
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    return t


def _build_auth_headers(service_token_raw: str, *, user_token_raw: Optional[str] = None) -> Dict[str, str]:
    """
    Headers para MCP Streamable HTTP con doble capa:

    - Authorization: Bearer <service_token> (S2S / gateway)
    - X-User-Authorization: Bearer <user_jwt> (usuario, opcional)

    Backward compatible: si no hay user_token_raw, no añade cabecera.
    """
    headers: Dict[str, str] = {"Accept": "application/json"}

    svc = _normalize_bearer(service_token_raw)
    if svc:
        headers["Authorization"] = f"Bearer {svc}"

    usr = _normalize_bearer(user_token_raw)
    if usr:
        headers["X-User-Authorization"] = f"Bearer {usr}"

    return headers

class MCPClient:
    """
    Cliente MCP-only para un ecosistema Cosmos con UN bus central.

    Objetivo:
    - Consumir tools exclusivamente vía MCP Streamable HTTP.
    - Sin gateway REST.
    - Compatible con FastMCP Client usando Streamable HTTP.

    Variables:
    - MCP_SERVER_URL / MCP_GATEWAY_BASE_URL
    - MCP_GATEWAY_TOKEN / MCP_SERVICE_TOKEN / MCP_AUTH_TOKEN
      (se normaliza para aceptar con o sin prefijo 'Bearer ').
    """

    # Cache de clases importadas
    _FAST_CLIENT_CLS: Optional[type] = None
    _FAST_BEARER_CLS: Optional[type] = None
    _FAST_TRANSPORT_CLS: Optional[type] = None
    _FAST_RESOLVE_ERROR: Optional[Exception] = None

    def __init__(
        self,
        *,
        mcp_server_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_base: Optional[float] = None,
    ) -> None:
        env_server = os.getenv("MCP_SERVER_URL", "").strip()
        env_gateway_base = os.getenv("MCP_GATEWAY_BASE_URL", "").strip()
        default_server = "http://localhost:8090/mcp"

        raw_candidate = (mcp_server_url or env_server or env_gateway_base or default_server).strip()

        # ✅ Blindaje fuerte: si te pasan una base REST por error, reconducimos a /mcp
        try:
            if "/api/v1" in raw_candidate:
                base = raw_candidate.split("/api/v1", 1)[0].rstrip("/")
                raw_candidate = base + "/mcp"
        except Exception:
            pass

        candidate = _normalize_mcp_url(raw_candidate).strip()

        # Si alguien puso ".../mcp/call_tool" o similares:
        candidate = re.sub(r"/mcp/(call_tool|list_tools|tools).*?$", "/mcp", candidate)

        if candidate.endswith("/mcp"):
            candidate = candidate + "/"

        # Si alguien puso ".../mcp//"
        candidate = re.sub(r"/mcp/+$", "/mcp/", candidate)

        self.mcp_server_url = candidate

        # ---------------------------------------------------------
        # ✅ Token por defecto (bus) consistente con chat_document
        # ---------------------------------------------------------
        token_candidate = (
            token
            if token is not None
            else _env_first(
                # específicos de web_search si los usas
                "WEBSEARCH_MCP_SERVICE_TOKEN",
                "WEBSEARCH_MCP_GATEWAY_TOKEN",
                # ✅ incluir el alias de chat_document para despliegues compartidos
                "CHATDOC_MCP_SERVICE_TOKEN",
                # ✅ estándar
                "MCP_SERVICE_TOKEN",
                "MCP_AUTH_TOKEN",
                "MCP_GATEWAY_TOKEN",
            )
        )

        self.default_auth_token = _normalize_bearer(token_candidate)

        self.timeout = float(timeout or os.getenv("MCP_HTTP_TIMEOUT", "15"))
        self.max_retries = int(max_retries or os.getenv("MCP_MAX_RETRIES", "2"))
        self.backoff_base = float(backoff_base or os.getenv("MCP_BACKOFF_BASE", "0.35"))

        self._lock = asyncio.Lock()

        logger.info(
            "[web_search/MCPClient] url=%s timeout=%.1fs default_token_present=%s",
            self.mcp_server_url,
            self.timeout,
            bool(self.default_auth_token),
        )

    def _resolve_gateway_token(self, auth_token: Optional[str]) -> str:
        """
        Devuelve el token efectivo para autenticar el bus:
        - auth_token explícito si viene
        - si no, el default del servicio
        """
        tok = _normalize_bearer(auth_token)
        if tok:
            return tok
        return self.default_auth_token or ""


    def _build_headers(
        self,
        auth_token: Optional[str],
        *,
        user_auth_token: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Headers efectivos para MCP:
        - auth_token: token S2S (si None, se usa default_auth_token)
        - user_auth_token: JWT/token de usuario (va en X-User-Authorization)
        """
        svc = self._resolve_gateway_token(auth_token)
        return _build_auth_headers(svc, user_token_raw=user_auth_token)

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
    # Resolución de FastMCP Client + Auth + Transport
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
                    "Asegura que 'fastmcp' está instalado en web_search."
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

    # ------------------------------------------------------------------
    # Construcción de cliente
    # ------------------------------------------------------------------
    def _build_auth_obj(self, BearerAuth, token_raw: str):
        if not token_raw or BearerAuth is None:
            return None
        try:
            return BearerAuth(token_raw)  # token crudo sin "Bearer "
        except Exception:
            return None

    def _build_client(self, *, auth_token: Optional[str] = None, user_auth_token: Optional[str] = None):
        """
        Construye una instancia de fastmcp.Client de forma tolerante.

        Reglas:
        - Si user_auth_token está presente, intentamos PRIORIZAR Transport con headers
        (para poder enviar X-User-Authorization).
        - Si no hay user_auth_token, mantenemos el comportamiento previo (auth obj / string / transport / sin auth).
        """
        Client, BearerAuth, Transport = self._resolve_fastmcp()

        svc_tok = self._resolve_gateway_token(auth_token)
        usr_tok = _normalize_bearer(user_auth_token)

        # 0) Si hay user token, preferimos Transport con headers (si está disponible)
        if usr_tok and Transport is not None:
            headers = self._build_headers(auth_token, user_auth_token=usr_tok)

            transport = None
            try:
                transport = Transport(url=self.mcp_server_url, headers=headers or None, timeout=self.timeout)
            except TypeError:
                try:
                    transport = Transport(url=self.mcp_server_url, headers=headers or None)
                except TypeError:
                    try:
                        transport = Transport(self.mcp_server_url, headers=headers or None)
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

            # Si no se pudo, seguimos con fallback clásico (pero NO podremos enviar user header)
            logger.warning(
                "[MCPClient] No se pudo construir Transport con headers; continuando sin X-User-Authorization."
            )

        # 1) URL + auth obj (solo S2S)
        auth_obj = self._build_auth_obj(BearerAuth, svc_tok)
        if auth_obj is not None:
            try:
                return Client(self.mcp_server_url, auth=auth_obj)
            except TypeError:
                pass
            except Exception:
                pass

        # 2) URL + token string CRUDO (solo S2S)
        if svc_tok:
            try:
                return Client(self.mcp_server_url, auth=svc_tok)
            except TypeError:
                pass
            except Exception:
                pass

        # 3) Transport explícito con headers (solo S2S)
        if Transport is not None:
            headers = self._build_headers(auth_token, user_auth_token=None)

            transport = None
            try:
                transport = Transport(url=self.mcp_server_url, headers=headers or None, timeout=self.timeout)
            except TypeError:
                try:
                    transport = Transport(url=self.mcp_server_url, headers=headers or None)
                except TypeError:
                    try:
                        transport = Transport(self.mcp_server_url, headers=headers or None)
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
        try:
            return Client(self.mcp_server_url)
        except Exception as e:
            raise MCPClientError(f"No se pudo construir fastmcp.Client: {e}") from e

    # ------------------------------------------------------------------
    # Unwrap de resultados
    # ------------------------------------------------------------------
    def _unwrap(self, obj: Any) -> Any:
        if obj is None:
            return None

        if isinstance(obj, (list, str, int, float, bool)):
            return obj

        if isinstance(obj, dict):
            if "data" in obj and len(obj.keys()) == 1:
                return self._unwrap(obj.get("data"))
            if "result" in obj and len(obj.keys()) == 1:
                return self._unwrap(obj.get("result"))
            if "tools" in obj and isinstance(obj.get("tools"), list) and len(obj.keys()) <= 2:
                return obj.get("tools")
            return obj

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
    ) -> List[Dict[str, Any]]:
        if not tools:
            return []

        crew = (crew or "").strip() or None
        any_tags = [str(t).strip() for t in (tags_any or []) if str(t).strip()]

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

            out.append(t)

        return out

    async def alist_tools(
        self,
        crew: Optional[str] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        *,
        auth_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._alist_tools_mcp(
                    crew=crew,
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
        args: Optional[Dict[str, Any]] = None,
        *,
        auth_token: Optional[str] = None,          # S2S (Authorization)
        user_auth_token: Optional[str] = None,     # USER (X-User-Authorization)
    ) -> Any:
        args = dict(args or {})
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._ainvoke_tool_mcp(
                    name,
                    args,
                    auth_token=auth_token,
                    user_auth_token=user_auth_token,
                )
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    await self._asleep_backoff(attempt)
                else:
                    break
        raise MCPClientError(f"Fallo invoke_tool MCP '{name}': {last_exc}")


    async def _ainvoke_tool_mcp(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        auth_token: Optional[str] = None,
        user_auth_token: Optional[str] = None,
    ) -> Any:
            try:
                async with self._lock:
                    client = self._build_client(auth_token=auth_token, user_auth_token=user_auth_token)

                async with client:
                    res = await client.call_tool(name, args or {})
                    return self._unwrap(res)

            except Exception as e:
                logger.warning("fastmcp.call_tool falló, usando fallback HTTP: %s", e)

            return await self._ainvoke_tool_http_fallback(
                name,
                args or {},
                auth_token=auth_token,
                user_auth_token=user_auth_token,
            )

    async def _ainvoke_tool_http_fallback(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        auth_token: Optional[str] = None,
        user_auth_token: Optional[str] = None,
    ) -> Any:
        import httpx

        headers = self._build_headers(auth_token, user_auth_token=user_auth_token)
        base = self.mcp_server_url.rstrip("/") + "/"
        req = {"name": name, "arguments": args or {}}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(base + "call_tool", json=req, headers=headers or None)
            r.raise_for_status()
            data = r.json()
            return self._unwrap(data)


    async def _alist_tools_http_fallback(
        self,
        crew: Optional[str] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        *,
        auth_token: Optional[str] = None,
        user_auth_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        import httpx
        from urllib.parse import urlparse

        headers = self._build_headers(auth_token, user_auth_token=user_auth_token)
        base = self.mcp_server_url.rstrip("/") + "/"

        def _normalize_tools_payload(raw: Any) -> List[Dict[str, Any]]:
            raw = self._unwrap(raw)

            if isinstance(raw, dict) and isinstance(raw.get("tools"), list):
                raw = raw.get("tools") or []

            if isinstance(raw, list):
                tools_out = []
                for t in raw:
                    if isinstance(t, dict) and t.get("name"):
                        tools_out.append(t)
                return self._filter_tools_client_side(tools_out, crew=crew, tags_any=tags_any)

            return []

        payload: Dict[str, Any] = {}
        if crew:
            payload["crew"] = crew
        if tags_any:
            payload["tags_any"] = ",".join([t for t in tags_any if t])
        if tags_all:
            payload["tags_all"] = ",".join([t for t in tags_all if t])

        req = {"name": "cosmos_registry_list_tools", "arguments": payload}

        # 1) registry enriquecido
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(base + "call_tool", json=req, headers=headers or None)
                if r.status_code < 400:
                    data = r.json()
                    tools_norm = _normalize_tools_payload(data)
                    if tools_norm:
                        return tools_norm
        except Exception:
            pass

        # 2) listado estándar
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    base + "call_tool",
                    json={"name": "list_tools", "arguments": payload},
                    headers=headers or None,
                )
                if r.status_code < 400:
                    data = r.json()
                    tools_norm = _normalize_tools_payload(data)
                    if tools_norm:
                        return tools_norm
        except Exception:
            pass

        # 3) Último recurso: REST /api/v1/tools
        try:
            p = urlparse(self.mcp_server_url)
            rest_base = f"{p.scheme}://{p.netloc}/api/v1/tools"

            params = {}
            if crew:
                params["crew"] = crew
            if tags_any:
                params["tags"] = ",".join([t for t in tags_any if t])

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(rest_base, params=params or None, headers=headers or None)
                if r.status_code < 400:
                    data = r.json()
                    tools_list = []
                    if isinstance(data, dict) and isinstance(data.get("tools"), list):
                        tools_list = data.get("tools") or []
                    if isinstance(tools_list, list):
                        return self._filter_tools_client_side(tools_list, crew=crew, tags_any=tags_any)
        except Exception as e:
            logger.warning("Fallback REST list_tools falló: %s", e)

        return []

    # ------------------------------------------------------------------
    # API ASYNC pública
    # ------------------------------------------------------------------
    async def _alist_tools_mcp(
        self,
        crew: Optional[str] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        *,
        auth_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            async with self._lock:
                client = self._build_client(auth_token=auth_token)

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
                        return self._filter_tools_client_side(tools, crew=crew, tags_any=tags_any)
                    if isinstance(data, list):
                        return self._filter_tools_client_side(data, crew=crew, tags_any=tags_any)
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

                return self._filter_tools_client_side(tools_out, crew=crew, tags_any=tags_any)

        except Exception as e:
            logger.warning("fastmcp.list_tools falló, usando fallback HTTP: %s", e)

        return await self._alist_tools_http_fallback(
            crew=crew,
            tags_any=tags_any,
            tags_all=tags_all,
            auth_token=auth_token,
        )


    # ------------------------------------------------------------------
    # API SYNC
    # ------------------------------------------------------------------
    def list_tools(
        self,
        crew: Optional[str] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        *,
        auth_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._run_sync(
            self.alist_tools(crew=crew, tags_any=tags_any, tags_all=tags_all, auth_token=auth_token)
        )


    def invoke_tool(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        auth_token: Optional[str] = None,
    ) -> Any:
        return self._run_sync(self.ainvoke_tool(name, args=args, auth_token=auth_token))


    # ------------------------------------------------------------------
    # Runner sync robusto
    # ------------------------------------------------------------------
    def _run_sync(self, coro):
        try:
            asyncio.get_running_loop()
            result_container = {"value": None, "error": None}
            done_evt = threading.Event()

            def _worker():
                try:
                    result_container["value"] = asyncio.run(coro)
                except Exception as e:
                    result_container["error"] = e
                finally:
                    done_evt.set()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

            # timeout defensivo
            done_evt.wait(timeout=self.timeout * 3)

            if result_container["error"]:
                raise result_container["error"]

            if not done_evt.is_set():
                raise MCPClientError("Timeout ejecutando llamada async desde contexto sync.")

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

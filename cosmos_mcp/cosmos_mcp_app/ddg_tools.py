# cosmos_mcp_app/ddg_tools.py
# -*- coding: utf-8 -*-
"""
MCP Tools - DuckDuckGo HTML Search + Content Fetch

Objetivo
--------
Estas tools están diseñadas para que una crew de agentes pueda:
1) Realizar búsquedas web resilientes usando DuckDuckGo HTML.
2) Extraer y limpiar el contenido principal de los resultados.
3) (Opcional) Agregar enlaces internos de forma ligera.
4) Proveer salida en formato 'text', 'json' o 'trace_json'.

Notas de integración
--------------------
- Mantiene los nombres públicos: ddg_search_tool, ddg_fetch_content_tool,
  ddg_cache_stats_tool para no romper registries existentes.
- Implementación autocontenida y compatible con tu arquitectura:
  cosmos_mcp_app.server define el FastMCP real expuesto en /mcp.

Dependencias recomendadas
-------------------------
- httpx
- beautifulsoup4

Dependencias opcionales para mejor ranking:
- spacy + modelo es_core_news_md
- sentence-transformers

"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from urllib.parse import parse_qs, unquote, urlparse
import httpx
from bs4 import BeautifulSoup

from services.tools_registry import cosmos_tool



try:
    from .server import mcp_server, MCPAppContext
except Exception:  # pragma: no cover
    # Fallback si alguien ejecuta el archivo suelto
    mcp_server = type("_Dummy", (), {"tool": lambda *a, **k: (lambda f: f)})()
    MCPAppContext = Any  # type: ignore

# Tipos MCP (solo para typing / ctx.info)
try:
    from mcp.server.session import ServerSession  # type: ignore
    from mcp.server.fastmcp import Context  # type: ignore
except Exception:  # pragma: no cover
    class ServerSession:  # noqa: D401
        """Stub"""
    class Context:  # noqa: D401
        """Stub"""
        async def info(self, *_a, **_k):
            return None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
DEFAULT_REGION = os.getenv("DDG_DEFAULT_REGION", "es-es")
DDG_STRIP_DIACRITICS = os.getenv("DDG_STRIP_DIACRITICS", "1") not in ("0", "false", "False")

DDG_SEARCH_RPM = int(os.getenv("DDG_SEARCH_RPM", "90"))
DDG_FETCH_RPM = int(os.getenv("DDG_FETCH_RPM", "120"))
DDG_FETCH_CONCURRENCY = int(os.getenv("DDG_FETCH_CONCURRENCY", "5"))

DDG_DEFAULT_FETCH_DEPTH = int(os.getenv("DDG_DEFAULT_FETCH_DEPTH", "1"))
DDG_DEFAULT_MAX_CHARS = int(os.getenv("DDG_DEFAULT_MAX_CHARS", "8000"))

SEARCH_CACHE_TTL = int(os.getenv("DDG_SEARCH_CACHE_TTL", "300"))
FETCH_CACHE_TTL = int(os.getenv("DDG_FETCH_CACHE_TTL", "900"))

MAX_INTERNAL_LINKS = int(os.getenv("DDG_MAX_INTERNAL_LINKS", "2"))
MAX_RESULTS_HARD = int(os.getenv("DDG_MAX_RESULTS_HARD", "25"))

# Ranking semántico opcional (permite desactivarlo en producción ligera)
DDG_ENABLE_SEMANTIC_RANKING = os.getenv("DDG_ENABLE_SEMANTIC_RANKING", "1") not in ("0", "false", "False")

# Delay humano para batch
try:
    BATCH_DELAY_MIN = float(os.getenv("DDG_BATCH_DELAY_MIN", "0.18"))
    BATCH_DELAY_MAX = float(os.getenv("DDG_BATCH_DELAY_MAX", "0.45"))
except ValueError:
    BATCH_DELAY_MIN, BATCH_DELAY_MAX = 0.18, 0.45
BATCH_DELAY_MIN = max(0.05, BATCH_DELAY_MIN)
BATCH_DELAY_MAX = max(BATCH_DELAY_MIN, BATCH_DELAY_MAX)


# -----------------------------------------------------------------------------
# HTTP client y headers
# -----------------------------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
]
ACCEPT_HEADER = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
LANGUAGE_HEADER = "es-ES,es;q=0.9,en;q=0.6"

# Cliente lazy para evitar problemas de lifecycle en reloads
_SESSION: Optional[httpx.AsyncClient] = None
_SEM_FETCH = asyncio.Semaphore(DDG_FETCH_CONCURRENCY)
_MCP_RL_REDIS = None
_MCP_RL_REDIS_URL = (os.getenv("MCP_REDIS_URL", "") or "").strip()


def _get_user_id_from_ctx(ctx) -> Optional[str]:
    """
    Extrae user_id inyectado en el contexto MCP.

    Compatibilidad multi-versión FastMCP:
    - request.state.user_id (si hay Request Starlette)
    - session.request.state.user_id
    - scope["state"]["user_id"] (ASGI wrapper)
    """
    if not ctx:
        return None

    # 1) ctx.request.state.user_id
    for attr in ("request", "_request"):
        req = getattr(ctx, attr, None)
        if req is not None:
            try:
                st = getattr(req, "state", None)
                uid = getattr(st, "user_id", None) if st is not None else None
                if uid is not None:
                    return str(uid)
            except Exception:
                pass

    # 2) ctx.session.request.state.user_id
    sess = getattr(ctx, "session", None) or getattr(ctx, "_session", None)
    if sess is not None:
        req = getattr(sess, "request", None) or getattr(sess, "_request", None)
        if req is not None:
            try:
                st = getattr(req, "state", None)
                uid = getattr(st, "user_id", None) if st is not None else None
                if uid is not None:
                    return str(uid)
            except Exception:
                pass

        # 3) sess.scope["state"]["user_id"]
        scope = getattr(sess, "scope", None)
        if isinstance(scope, dict):
            st = scope.get("state") or {}
            if isinstance(st, dict):
                uid = st.get("user_id")
                if uid is not None:
                    return str(uid)

    # 4) ctx.scope["state"]["user_id"]
    scope = getattr(ctx, "scope", None) or getattr(ctx, "_scope", None)
    if isinstance(scope, dict):
        st = scope.get("state") or {}
        if isinstance(st, dict):
            uid = st.get("user_id")
            if uid is not None:
                return str(uid)

    return None


async def _get_rate_limit_redis():
    """
    Devuelve un cliente redis.asyncio reutilizable si MCP_REDIS_URL está configurado.
    Best-effort: si falla, devuelve None (no rompe legacy).
    """
    global _MCP_RL_REDIS
    if not _MCP_RL_REDIS_URL:
        return None

    # Si ya existe, lo reutilizamos
    if _MCP_RL_REDIS is not None:
        try:
            # ping best-effort; si falla, recreamos
            await _MCP_RL_REDIS.ping()
            return _MCP_RL_REDIS
        except Exception:
            _MCP_RL_REDIS = None

    try:
        import redis.asyncio as aioredis  # type: ignore
        r = aioredis.from_url(_MCP_RL_REDIS_URL, decode_responses=True)
        try:
            await r.ping()
        except Exception:
            return None
        _MCP_RL_REDIS = r
        return _MCP_RL_REDIS
    except Exception:
        _MCP_RL_REDIS = None
        return None



async def _rate_limit_user_best_effort(
    user_id: Optional[str],
    *,
    action: str,
    limit: int,
    window_sec: int,
) -> bool:
    """
    Rate limit por usuario usando Redis (si está configurado).
    - Si no hay user_id => no limita (legacy / anónimo).
    - Si no hay Redis o falla => permite (no rompe).
    - Implementación atómica con Lua: INCR + TTL consistente.
    """
    if not user_id:
        return True

    try:
        limit = int(limit)
        window_sec = int(window_sec)
        if limit <= 0 or window_sec <= 0:
            return True
    except Exception:
        return True

    r = await _get_rate_limit_redis()
    if r is None:
        return True

    # Clave por usuario + acción
    key = f"mcp:rl:{action}:{user_id}"

    lua = """
    local key = KEYS[1]
    local window = tonumber(ARGV[1])
    local limit = tonumber(ARGV[2])

    local v = redis.call("INCR", key)

    if v == 1 then
        redis.call("EXPIRE", key, window)
    else
        local ttl = redis.call("TTL", key)
        if ttl < 0 then
            redis.call("EXPIRE", key, window)
        end
    end

    if v > limit then
        return 0
    end
    return 1
    """

    try:
        ok = await r.eval(lua, 1, key, str(window_sec), str(limit))
        return bool(int(ok))
    except Exception:
        return True


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": ACCEPT_HEADER,
        "Accept-Language": LANGUAGE_HEADER,
    }

def _rl_params(action: str) -> tuple[int, int]:
    """
    Devuelve (limit, window) por acción leyendo env vars.
    Defaults conservadores:
      - ddg_search: 30 / 60s
      - ddg_fetch:  60 / 60s
    """
    action = (action or "").strip().lower()
    if action == "ddg_fetch":
        lim_env = "DDG_USER_FETCH_LIMIT"
        win_env = "DDG_USER_FETCH_WINDOW"
        default_lim, default_win = 60, 60
    else:
        lim_env = "DDG_USER_SEARCH_LIMIT"
        win_env = "DDG_USER_SEARCH_WINDOW"
        default_lim, default_win = 30, 60

    try:
        limit = int(os.getenv(lim_env, str(default_lim)))
    except Exception:
        limit = default_lim

    try:
        window = int(os.getenv(win_env, str(default_win)))
    except Exception:
        window = default_win

    return max(1, limit), max(1, window)

def _get_session() -> httpx.AsyncClient:
    global _SESSION
    if _SESSION is None or _SESSION.is_closed:
        _SESSION = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    return _SESSION


# -----------------------------------------------------------------------------
# Cachés simples con TTL
# -----------------------------------------------------------------------------
class _TTLCache:
    def __init__(self, ttl_seconds: int):
        self.ttl = max(1, int(ttl_seconds))
        self._store: Dict[str, Any] = {}
        self._exp: Dict[str, float] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any:
        now = time.time()
        exp = self._exp.get(key)
        if exp is not None and exp > now and key in self._store:
            self.hits += 1
            return self._store[key]
        if key in self._store:
            self._store.pop(key, None)
            self._exp.pop(key, None)
        self.misses += 1
        return None

    def set(self, key: str, value: Any):
        self._store[key] = value
        self._exp[key] = time.time() + self.ttl

    def stats(self) -> Dict[str, Any]:
        return {
            "ttl": self.ttl,
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
        }


SEARCH_CACHE = _TTLCache(SEARCH_CACHE_TTL)
FETCH_CACHE = _TTLCache(FETCH_CACHE_TTL)


# -----------------------------------------------------------------------------
# Modelos opcionales de ranking (lazy import)
# -----------------------------------------------------------------------------
_nlp = None
_sbert = None


def _lazy_load_rankers():
    """
    Carga opcional de spacy + SBERT.
    Se desactiva si DDG_ENABLE_SEMANTIC_RANKING=0
    """
    global _nlp, _sbert

    if not DDG_ENABLE_SEMANTIC_RANKING:
        _nlp = False
        _sbert = False
        return

    # Ya intentado
    if _nlp is not None or _sbert is not None:
        return

    try:
        import spacy  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        _nlp = spacy.load("es_core_news_md")
        _sbert = SentenceTransformer("intfloat/multilingual-e5-large")
        logger.info("[DDG] Rankers cargados: spacy + SBERT")
    except Exception as e:
        _nlp = False
        _sbert = False
        logger.warning("[DDG] Ranking semántico no disponible: %s", e)


# -----------------------------------------------------------------------------
# Tipos de datos internos
# -----------------------------------------------------------------------------
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    content: str = ""
    position: int = 0
    relevance: float = 0.0


# -----------------------------------------------------------------------------
# Utilidades de normalización
# -----------------------------------------------------------------------------
def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def _norm_query(q: str) -> str:
    q = str(q or "").strip()
    q = re.sub(r"\s+", " ", q)
    if DDG_STRIP_DIACRITICS:
        q = _strip_diacritics(q)
    return q


def _cache_key_search(query: str, max_results: int, region: str) -> str:
    return f"s::{region}::{max_results}::{query}"


def _cache_key_fetch(url: str, max_chars: int, depth: int) -> str:
    return f"f::{depth}::{max_chars}::{url}"


# -----------------------------------------------------------------------------
# HTTP resiliente
# -----------------------------------------------------------------------------
async def _safe_get(
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 15.0,
) -> httpx.Response:
    backoff = 1.0
    last_exc: Optional[Exception] = None
    session = _get_session()

    for attempt in range(4):
        try:
            resp = await session.get(url, params=params, headers=headers or _headers(), timeout=timeout)

            # DDG HTML a veces devuelve 202; también puede caer en 429/503.
            if resp.status_code in (202, 429, 503):
                logger.warning(
                    "[DDG] Rate-limit/status %s en %s (attempt %s). Backoff %.1fs",
                    resp.status_code, url, attempt + 1, backoff
                )
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

            resp.raise_for_status()
            return resp

        except Exception as e:
            last_exc = e
            await asyncio.sleep(backoff)
            backoff *= 2

    if last_exc:
        raise last_exc
    raise RuntimeError("safe_get failed")


# -----------------------------------------------------------------------------
# Parseo de URLs de DDG
# -----------------------------------------------------------------------------
def _extract_real_url(href: str) -> str:
    """
    DuckDuckGo HTML puede devolver URLs ofuscadas con /l/?uddg=...
    """
    try:
        if not href:
            return href
        parsed = urlparse(href)
        if parsed.path.startswith("/l/") or "uddg=" in href:
            qs = parse_qs(parsed.query)
            uddg = qs.get("uddg", [""])[0]
            if uddg:
                return unquote(uddg)
        return href
    except Exception:
        return href


# -----------------------------------------------------------------------------
# Búsqueda DuckDuckGo HTML
# -----------------------------------------------------------------------------
async def _duckduckgo_html_search(
    query: str,
    *,
    max_results: int = 10,
    region: str = "",
) -> List[SearchResult]:
    query = _norm_query(query)
    if not query:
        return []

    region = (region or DEFAULT_REGION).strip() or DEFAULT_REGION
    max_results = max(1, min(int(max_results), MAX_RESULTS_HARD))

    cache_key = _cache_key_search(query, max_results, region)
    cached = SEARCH_CACHE.get(cache_key)
    if cached is not None:
        # Ojo: devolvemos la lista cacheada tal cual.
        return cached

    search_url = "https://duckduckgo.com/html/"
    params = {"q": query}
    if region:
        params["kl"] = region

    resp = await _safe_get(search_url, params=params, headers=_headers(), timeout=10.0)

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[SearchResult] = []

    for block in soup.select("div.result"):
        a = block.find("a", class_="result__a")
        if not a or not a.get("href"):
            continue

        raw_href = a.get("href")
        url = _extract_real_url(raw_href)
        title = a.get_text(" ", strip=True) or ""

        sn_tag = block.find("a", class_="result__snippet") or block.find("div", class_="result__snippet")
        snippet = sn_tag.get_text(" ", strip=True) if sn_tag else ""

        if url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))

        if len(results) >= max_results:
            break

    for i, r in enumerate(results, start=1):
        r.position = i

    SEARCH_CACHE.set(cache_key, results)
    return results


# -----------------------------------------------------------------------------
# Limpieza HTML
# -----------------------------------------------------------------------------
def _clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html_text)
        return re.sub(r"\s+", " ", text).strip()


def _extract_http_links(html_text: str) -> List[str]:
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        links: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a.get("href") or ""
            if href.startswith("http"):
                links.append(href)

        # dedup manteniendo orden
        seen = set()
        out: List[str] = []
        for l in links:
            if l in seen:
                continue
            seen.add(l)
            out.append(l)

        return out
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Fetch + agregación ligera de internos
# -----------------------------------------------------------------------------
async def _fetch_html(url: str) -> str:
    async with _SEM_FETCH:
        try:
            resp = await _safe_get(url, headers=_headers(), timeout=15.0)
            return resp.text or ""
        except Exception as e:
            logger.debug("[DDG] Fetch failed for %s: %s", url, e)
            return ""


async def _get_clean_content(
    url: str,
    *,
    depth: int = 1,
    max_chars: int = DDG_DEFAULT_MAX_CHARS,
) -> str:
    url = (url or "").strip()
    if not url:
        return ""

    depth = max(0, int(depth))
    max_chars = max(500, int(max_chars))

    cache_key = _cache_key_fetch(url, max_chars, depth)
    cached = FETCH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    html = await _fetch_html(url)
    if not html:
        FETCH_CACHE.set(cache_key, "")
        return ""

    main_text = _clean_html(html)
    aggregated: List[str] = [main_text]

    if depth > 0:
        links = _extract_http_links(html)
        for link in links[:MAX_INTERNAL_LINKS]:
            aggregated.append(await _get_clean_content(link, depth=depth - 1, max_chars=max_chars))

    content = "\n".join([t for t in aggregated if t])
    if len(content) > max_chars:
        content = content[:max_chars]

    FETCH_CACHE.set(cache_key, content)
    return content


# -----------------------------------------------------------------------------
# Ranking
# -----------------------------------------------------------------------------
def _simple_overlap_score(query: str, text: str) -> float:
    q_tokens = set(re.findall(r"\w+", query.lower()))
    t_tokens = set(re.findall(r"\w+", (text or "").lower()))
    if not q_tokens or not t_tokens:
        return 0.0
    inter = q_tokens.intersection(t_tokens)
    return len(inter) / max(1, len(q_tokens))


async def _rank_results(query: str, results: List[SearchResult]) -> List[SearchResult]:
    """
    Ordena resultados por relevancia.
    Usa SBERT si está disponible; fallback por solapamiento.
    """
    if not results:
        return results

    _lazy_load_rankers()

    if _nlp is False or _sbert is False:
        for r in results:
            blob = f"{r.title}. {r.snippet} {r.content}"
            r.relevance = _simple_overlap_score(query, blob)
        return sorted(results, key=lambda x: x.relevance, reverse=True)

    try:
        # type: ignore
        doc = _nlp(query)
        qproc = " ".join(t.lemma_ for t in doc if not t.is_stop)

        # type: ignore
        q_emb = _sbert.encode(qproc)

        import numpy as np

        for r in results:
            text = f"{r.title}. {r.snippet} {r.content}"
            # type: ignore
            c_emb = _sbert.encode(text)
            dot = float(np.dot(q_emb, c_emb))
            n1 = float(np.linalg.norm(q_emb))
            n2 = float(np.linalg.norm(c_emb))
            r.relevance = dot / (n1 * n2) if n1 and n2 else 0.0

        return sorted(results, key=lambda x: x.relevance, reverse=True)

    except Exception as e:
        logger.debug("[DDG] Ranking semántico falló. Fallback. Error: %s", e)
        for r in results:
            blob = f"{r.title}. {r.snippet} {r.content}"
            r.relevance = _simple_overlap_score(query, blob)
        return sorted(results, key=lambda x: x.relevance, reverse=True)


# -----------------------------------------------------------------------------
# Formateadores
# -----------------------------------------------------------------------------
def _format_results_for_llm(results: List[SearchResult], query: str) -> str:
    if not results:
        return f"Sin resultados para: {query}"

    lines = [f"Resultados DDG para: {query}"]
    for r in results:
        title = r.title or "(Sin título)"
        snippet = (r.snippet or "").strip()
        url = r.url
        rel = f" | rel={r.relevance:.3f}" if r.relevance else ""
        lines.append(f"{r.position}. {title}{rel}\n   {url}")
        if snippet:
            lines.append(f"   {snippet}")
        if r.content:
            lines.append(f"   [contenido] {r.content[:300].strip()}...")
    return "\n".join(lines)


def _format_results_json(results: List[SearchResult], query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "count": len(results),
        "results": [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "content": r.content,
                "position": r.position,
                "relevance": r.relevance,
            }
            for r in results
        ],
    }


# -----------------------------------------------------------------------------
# Tool 1: ddg_search_tool
# -----------------------------------------------------------------------------
@cosmos_tool(
    name="ddg_search_tool",
    description=(
        "Realiza una búsqueda web resiliente usando DuckDuckGo HTML. "
        "Incluye normalización de query, caché TTL, backoff en rate limit, "
        "y extracción de contenido de los resultados. "
        "return_format: 'text' | 'json' | 'trace_json'. "
        "Opcional: 'queries' para modo batch."
    ),
    tags=["web", "search", "duckduckgo", "ddg-html"],
    for_crews=["web_search_crew", "*"],
)
@mcp_server.tool(
    name="ddg_search_tool",
    description="Realiza una búsqueda web con opción de batch y salida estructurada.",
)
async def ddg_search_tool(
    query: str = "",
    max_results: int = 10,
    return_format: str = "text",
    region: str = "",
    queries: Optional[List[str]] = None,
    trace: Optional[bool] = None,
    ctx: Optional[Context[ServerSession, MCPAppContext]] = None,
) -> Union[str, Dict[str, Any]]:
    """
    return_format:
      - text: bloque amigable para LLM
      - json: resultados estructurados
      - trace_json: json + trazas ligeras de caché/tiempos

    Multiusuario (sin romper legacy):
      - Si el middleware MCPUserContextMiddleware está instalado, se extrae user_id desde ctx.
      - Opcionalmente aplica rate-limit por usuario si MCP_REDIS_URL está configurado.
    """
    fmt = (return_format or "text").lower().strip()
    if fmt not in ("text", "json", "trace_json"):
        fmt = "text"

    max_results = max(1, min(int(max_results or 10), MAX_RESULTS_HARD))
    region = (region or DEFAULT_REGION).strip() or DEFAULT_REGION

    use_trace = bool(trace) if trace is not None else (fmt == "trace_json")

    # ---- User context ----
    user_id = _get_user_id_from_ctx(ctx)

    # ---- Rate limit params ----
    user_limit, user_window = _rl_params("ddg_search")

    # Nota: no hacemos "global check" si es batch para no contar doble.
    if not (queries and isinstance(queries, list) and any(str(q or "").strip() for q in queries)):
        allowed = await _rate_limit_user_best_effort(
            user_id, action="ddg_search", limit=user_limit, window_sec=user_window
        )
        if not allowed:
            if fmt == "text":
                return "Rate limit excedido para usuario. Intenta de nuevo en unos segundos."
            return {
                "error": "rate_limited",
                "detail": "User rate limit exceeded",
                "meta": {"user_id": user_id, "limit": user_limit, "window_sec": user_window},
            }


    effective_queries: List[str] = []
    if queries and isinstance(queries, list):
        effective_queries = [_norm_query(str(q or "")) for q in queries if str(q or "").strip()]

    async def _run_single(q: str) -> Dict[str, Any]:
        t0 = time.time()

        # ---- Search ----
        norm_q = _norm_query(q)
        s_key = _cache_key_search(norm_q, max_results, region)
        cached_list = SEARCH_CACHE.get(s_key)

        if cached_list is not None:
            base_results: List[SearchResult] = cached_list
        else:
            base_results = await _duckduckgo_html_search(norm_q, max_results=max_results, region=region)

        logger.debug("[DDG] base_results=%d query=%r user_id=%s", len(base_results or []), (q or "")[:120], user_id)

        # ---- Fetch content por resultado ----
        fetch_depth = DDG_DEFAULT_FETCH_DEPTH
        max_chars = DDG_DEFAULT_MAX_CHARS

        async def _attach_content(r: SearchResult):
            if not r.url:
                return r
            f_key = _cache_key_fetch(r.url, max_chars, fetch_depth)
            f_cached = FETCH_CACHE.get(f_key)
            if f_cached is not None:
                r.content = f_cached or ""
            else:
                r.content = await _get_clean_content(r.url, depth=fetch_depth, max_chars=max_chars)
            return r

        if base_results:
            base_results = list(await asyncio.gather(*[_attach_content(r) for r in base_results]))

        # ---- Ranking ----
        ranked = await _rank_results(q, base_results)

        # Reindex tras ranking
        for i, r in enumerate(ranked, start=1):
            r.position = i

        elapsed = time.time() - t0
        payload = _format_results_json(ranked, q)

        if use_trace:
            payload["trace"] = {
                "elapsed_sec": round(elapsed, 3),
                "search_cache": SEARCH_CACHE.stats(),
                "fetch_cache": FETCH_CACHE.stats(),
                "region": region,
                "max_results": max_results,
                "defaults": {"fetch_depth": fetch_depth, "max_chars": max_chars},
                "ranking_semantic_enabled": bool(DDG_ENABLE_SEMANTIC_RANKING),
                # user context (útil para debugging / cuotas)
                "user_id": user_id,
            }

        return payload

    # -----------------------
    # Batch
    # -----------------------
    if effective_queries:
        per_query_payloads: List[Dict[str, Any]] = []
        all_results: List[SearchResult] = []

        for idx, q in enumerate(effective_queries):
            if ctx:
                await ctx.info(f"[DDG_TOOL][batch] Searching: {q}" + (f" user={user_id}" if user_id else ""))

            # (Opcional) contar cada query del batch como 1 operación de rate-limit:
            allowed_q = await _rate_limit_user_best_effort(
                user_id,
                action="ddg_search",
                limit=user_limit,
                window_sec=user_window,
            )
            if not allowed_q:
                if fmt == "text":
                    return "Rate limit excedido para usuario durante batch. Intenta de nuevo en unos segundos."
                return {
                    "error": "rate_limited",
                    "detail": "User rate limit exceeded (batch)",
                    "meta": {"user_id": user_id, "limit": user_limit, "window_sec": user_window},
                }

            pq = await _run_single(q)
            per_query_payloads.append(pq)

            for item in pq.get("results", []) or []:
                all_results.append(
                    SearchResult(
                        title=item.get("title", "") or "",
                        url=item.get("url", "") or "",
                        snippet=item.get("snippet", "") or "",
                        content=item.get("content", "") or "",
                        position=int(item.get("position", 0) or 0),
                        relevance=float(item.get("relevance", 0.0) or 0.0),
                    )
                )

            if idx < len(effective_queries) - 1:
                await asyncio.sleep(random.uniform(BATCH_DELAY_MIN, BATCH_DELAY_MAX))

        # Dedup por URL
        seen = set()
        deduped: List[SearchResult] = []
        for r in all_results:
            if not r.url or r.url in seen:
                continue
            seen.add(r.url)
            deduped.append(r)

        for i, r in enumerate(deduped, start=1):
            r.position = i

        payload: Dict[str, Any] = {
            "mode": "batch",
            "queries": effective_queries,
            "count": len(deduped),
            "per_query": per_query_payloads,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "content": r.content,
                    "position": r.position,
                    "relevance": r.relevance,
                }
                for r in deduped
            ],
            "meta": {
                "region": region,
                "max_results": max_results,
                "strip_diacritics": DDG_STRIP_DIACRITICS,
                "user_id": user_id,
            },
        }

        if fmt == "text":
            blocks = []
            for pq in per_query_payloads:
                q = pq.get("query", "")
                rs = [
                    SearchResult(
                        title=i.get("title", "") or "",
                        url=i.get("url", "") or "",
                        snippet=i.get("snippet", "") or "",
                        content=i.get("content", "") or "",
                        position=int(i.get("position", 0) or 0),
                        relevance=float(i.get("relevance", 0.0) or 0.0),
                    )
                    for i in (pq.get("results") or [])
                ]
                blocks.append(_format_results_for_llm(rs, q))
            return "\n\n".join(blocks)

        return payload

    # -----------------------
    # Single
    # -----------------------
    query = _norm_query(query)
    if not query:
        if fmt == "text":
            return "No query provided."
        return {"query": "", "count": 0, "results": [], "meta": {"region": region, "user_id": user_id}}

    if ctx:
        await ctx.info(f"[DDG_TOOL] Searching: {query}" + (f" user={user_id}" if user_id else ""))

    payload = await _run_single(query)
    payload.setdefault("meta", {})
    payload["meta"].update(
        {
            "region": region,
            "max_results": max_results,
            "strip_diacritics": DDG_STRIP_DIACRITICS,
            "user_id": user_id,
        }
    )

    if fmt == "text":
        rs = [
            SearchResult(
                title=i.get("title", "") or "",
                url=i.get("url", "") or "",
                snippet=i.get("snippet", "") or "",
                content=i.get("content", "") or "",
                position=int(i.get("position", 0) or 0),
                relevance=float(i.get("relevance", 0.0) or 0.0),
            )
            for i in (payload.get("results") or [])
        ]
        return _format_results_for_llm(rs, query)

    return payload


# -----------------------------------------------------------------------------
# Tool 2: ddg_fetch_content_tool
# -----------------------------------------------------------------------------
@cosmos_tool(
    name="ddg_fetch_content_tool",
    description=(
        "Descarga y limpia el contenido principal de una URL. "
        "Incluye rate limit/backoff, caché TTL y agregación ligera de enlaces internos. "
        "return_format: 'text' | 'json' | 'trace_json'. "
        "Opcional: depth para agregación ligera de enlaces internos."
    ),
    tags=["web", "fetch", "duckduckgo"],
    for_crews=["web_search_crew", "*"],
)
@mcp_server.tool(
    name="ddg_fetch_content_tool",
    description="Descarga y limpia contenido textual de una URL, con caché TTL.",
)
async def ddg_fetch_content_tool(
    url: str,
    max_chars: int = 8000,
    return_format: str = "text",
    trace: Optional[bool] = None,
    depth: Optional[int] = None,
    ctx: Optional[Context[ServerSession, MCPAppContext]] = None,
) -> Union[str, Dict[str, Any]]:
    fmt = (return_format or "text").lower().strip()
    if fmt not in ("text", "json", "trace_json"):
        fmt = "text"

    use_trace = bool(trace) if trace is not None else (fmt == "trace_json")

    url = (url or "").strip()
    max_chars = max(500, int(max_chars or DDG_DEFAULT_MAX_CHARS))
    eff_depth = DDG_DEFAULT_FETCH_DEPTH if depth is None else max(0, int(depth or 0))

    if not url:
        if fmt == "text":
            return ""
        return {"url": "", "count_chars": 0, "content": ""}

    # ---- User context ----
    user_id = _get_user_id_from_ctx(ctx)

    # ---- Rate limit (best-effort) ----
    user_limit, user_window = _rl_params("ddg_fetch")
    allowed = await _rate_limit_user_best_effort(
        user_id,
        action="ddg_fetch",
        limit=user_limit,
        window_sec=user_window,
    )
    if not allowed:
        if fmt == "text":
            return "Rate limit excedido para usuario (fetch). Intenta de nuevo en unos segundos."
        return {
            "error": "rate_limited",
            "detail": "User rate limit exceeded (fetch)",
            "meta": {"user_id": user_id, "limit": user_limit, "window_sec": user_window},
        }

    if ctx:
        await ctx.info(f"[DDG_FETCH] Fetching: {url}" + (f" user={user_id}" if user_id else ""))

    t0 = time.time()
    cache_key = _cache_key_fetch(url, max_chars, eff_depth)
    cached = FETCH_CACHE.get(cache_key)
    cache_hit = cached is not None

    if cache_hit:
        text = cached or ""
    else:
        text = await _get_clean_content(url, depth=eff_depth, max_chars=max_chars)

    elapsed = time.time() - t0

    if fmt == "text":
        return text

    payload: Dict[str, Any] = {
        "url": url,
        "count_chars": len(text or ""),
        "content": text,
        "meta": {
            "max_chars": max_chars,
            "depth": eff_depth,
            "engine": "clean_html_recursive_light",
            "user_id": user_id,
        },
    }

    if use_trace:
        payload["trace"] = {
            "cache_hit": cache_hit,
            "elapsed_sec": round(elapsed, 3),
            "fetch_cache": FETCH_CACHE.stats(),
            "user_id": user_id,
        }

    return payload


# -----------------------------------------------------------------------------
# Tool 3: ddg_cache_stats_tool
# -----------------------------------------------------------------------------
@cosmos_tool(
    name="ddg_cache_stats_tool",
    description="Devuelve estadísticas internas de caché de búsqueda y fetch.",
    tags=["web", "debug", "cache"],
    for_crews=["web_search_crew", "*"],
)
@mcp_server.tool(
    name="ddg_cache_stats_tool",
    description="Devuelve estadísticas internas de caché.",
)
async def ddg_cache_stats_tool(
    return_format: str = "json",
) -> Dict[str, Any]:
    _ = (return_format or "json").lower().strip()

    # Señales de configuración (sin exponer secretos)
    rl_enabled = bool(_MCP_RL_REDIS_URL)

    return {
        "search_cache": SEARCH_CACHE.stats(),
        "fetch_cache": FETCH_CACHE.stats(),
        "meta": {
            "html_first": True,
            "search_rpm": DDG_SEARCH_RPM,
            "fetch_rpm": DDG_FETCH_RPM,
            "fetch_concurrency": DDG_FETCH_CONCURRENCY,
            "strip_diacritics": DDG_STRIP_DIACRITICS,
            "default_fetch_depth": DDG_DEFAULT_FETCH_DEPTH,
            "default_max_chars": DDG_DEFAULT_MAX_CHARS,
            "max_internal_links": MAX_INTERNAL_LINKS,
            "semantic_ranking_enabled": bool(DDG_ENABLE_SEMANTIC_RANKING),
            "rate_limit": {
                "enabled": rl_enabled,
                "search": {"limit": _rl_params("ddg_search")[0], "window_sec": _rl_params("ddg_search")[1]},
                "fetch": {"limit": _rl_params("ddg_fetch")[0], "window_sec": _rl_params("ddg_fetch")[1]},
            },
        },
    }

# -----------------------------------------------------------------------------
# Limpieza del cliente HTTP al cerrar tu app (opcional)
# -----------------------------------------------------------------------------
async def _aclose_http_client():
    global _SESSION
    try:
        if _SESSION and not _SESSION.is_closed:
            await _SESSION.aclose()
    except Exception:
        pass
    finally:
        _SESSION = None

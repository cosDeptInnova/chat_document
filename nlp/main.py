import os
import json
import pickle
import uuid
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import aiofiles
import redis.asyncio as redis
import requests
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends, Form, Query, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge
from starlette.responses import Response
from difflib import SequenceMatcher
import re
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from config.database import get_db
from config.models import User, File as FileModel, AuditLog, Department
from pathlib import Path
import secrets
from fastapi import status
import base64
import time
import hashlib
import threading
from collections import OrderedDict

from cosmos_nlp_v3 import DocumentIndexer
from chatdoc_indexer import DocumentChatIndex
from rag_metrics import (
    RAG_HTTP_REQUESTS_TOTAL,
    RAG_HTTP_LATENCY_SECONDS,
    RAG_HTTP_INFLIGHT,
    RAG_USER_ENDPOINT_REQUESTS_TOTAL,
    CHATDOC_STATUS_TOTAL,
    CHATDOC_ERRORS_TOTAL,
    CHATDOC_CACHE_TOTAL,
    CHATDOC_INGEST_BUILD_SECONDS,
    CHATDOC_INGEST_REDIS_SECONDS,
    CHATDOC_INGEST_TOTAL_SECONDS,
    CHATDOC_INGEST_BYTES,
    CHATDOC_INGEST_CHUNKS,
    CHATDOC_INGEST_TEXT_CHARS,
    CHATDOC_INGEST_PAGE_COUNT,
    CHATDOC_QUERY_RESULTS_COUNT,
    CHATDOC_SUMMARY_FRAGMENTS,
)


logging.basicConfig(level=logging.INFO)
logging.error(traceback.format_exc())

app = FastAPI()
logger = logging.getLogger("chatdoc.ingest")

# -------------------------
# Roles y Debug Forzado
# -------------------------
ROLE_SUPERVISOR = 1
ROLE_USER = 2

OVERRIDE_USER_ID = 9999
OVERRIDE_ROLE_SESSION = "Supervisor"   # en sesión Redis se usa string
OVERRIDE_ROLE_TOKEN = ROLE_SUPERVISOR  # en token "simulado" se usa int
OVERRIDE_USER_DIRECTORY = "debug_user"  # relativo a PATH_BASE
OVERRIDE_DEPARTMENTS: List[Dict[str, Any]] = []  # puedes sembrar aquí si quieres
OVERRIDE_CLIENT_TAG = ""

REQUEST_COUNT = Counter("http_requests_total", "Total de solicitudes HTTP", ["method", "endpoint"])
RESPONSE_TIME = Histogram("http_response_time_seconds", "Tiempo de respuesta HTTP", ["method", "endpoint"])

# -------------------------
# RAG backends (config)
# -------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

LLAMA_EMBED_URL = os.getenv("LLAMA_EMBED_URL", "http://localhost:8000/v1/embeddings")
LLAMA_EMBED_MODEL = os.getenv("LLAMA_EMBED_MODEL", "local")
LLAMA_EMBED_TIMEOUT = int(os.getenv("LLAMA_EMBED_TIMEOUT", "60"))

os.environ.setdefault("QDRANT_HOST", QDRANT_HOST)
os.environ.setdefault("QDRANT_PORT", str(QDRANT_PORT))
os.environ.setdefault("LLAMA_EMBED_URL", LLAMA_EMBED_URL)
os.environ.setdefault("LLAMA_EMBED_MODEL", LLAMA_EMBED_MODEL)

# -------------------------
# Storage coherente con login/addUser
# -------------------------
STORAGE_ROOT = Path(os.getenv("COSMOS_STORAGE_ROOT", ".")).resolve()
PATH_BASE = Path(os.getenv("PATH_BASE", str(STORAGE_ROOT))).resolve()
PATH_BASE_DEPARTMENTS = Path(os.getenv("PATH_BASE_DEPARTMENTS", str(STORAGE_ROOT))).resolve()

SECRET_KEY = os.getenv("COSMOS_SECRET_KEY", "secretkey123")
ALGORITHM = os.getenv("COSMOS_JWT_ALG", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 300
redis_client = None


CSRF_COOKIE_NAME = "csrftoken_nlp"

# --------- Configuración Chat-Documento (índices efímeros) ---------
CHATDOC_REDIS_TTL = int(os.getenv("CHATDOC_REDIS_TTL", "3600"))  # 1h por defecto
CHATDOC_SUMMARY_MAX_CHARS = int(os.getenv("CHATDOC_SUMMARY_MAX_CHARS", "12000"))
CHATDOC_DEFAULT_CHUNK_CHARS = int(os.getenv("CHATDOC_CHUNK_CHARS", "1200"))
CHATDOC_DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHATDOC_CHUNK_OVERLAP", "200"))
CHATDOC_DEFAULT_LANGUAGE = os.getenv("CHATDOC_LANGUAGE", "spanish")


_CHATDOC_CACHE_MAX = int(os.getenv("CHATDOC_CACHE_MAX", "128"))
_CHATDOC_CACHE_LOCK = threading.RLock()

# full_key -> (expires_at, index_obj)
_CHATDOC_CACHE: "OrderedDict[str, tuple[float, DocumentChatIndex]]" = OrderedDict()
# base_key(user:doc) -> full_key(user:doc:sha)
_CHATDOC_ALIAS: dict[str, str] = {}

_INDEXER_CACHE: Dict[Tuple[str, str], DocumentIndexer] = {}
_INDEXER_CACHE_LOCK = threading.RLock()

# Concurrencia chatDoc (CPU/IO)
CHATDOC_INGEST_CONCURRENCY = int(os.getenv("CHATDOC_INGEST_CONCURRENCY", "2"))
# Limita extracción/fragmentación (CPU/IO) simultánea
CHATDOC_INGEST_SEMAPHORE = asyncio.Semaphore(max(1, CHATDOC_INGEST_CONCURRENCY))



app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="templates")
valid_extensions = [".doc", ".docx", ".xlsx", ".pptx", ".txt", ".jpg", ".png", ".pdf"]

def _invalidate_indexer_cache(index_filepath: str, user_id: str) -> None:
    """
    Invalida el indexer en cache para (index_filepath, user_id).
    Thread-safe.
    Best-effort close del cliente (si existe).
    """
    key = (str(index_filepath), str(user_id))
    idx = None
    with _INDEXER_CACHE_LOCK:
        idx = _INDEXER_CACHE.pop(key, None)

    if idx is not None:
        try:
            logging.info(f"[RAG] Invalidando cache de DocumentIndexer para {key}")
        except Exception:
            pass

        try:
            qc = getattr(idx, "qdrant_client", None)
            if qc is not None and hasattr(qc, "close") and callable(qc.close):
                qc.close()
        except Exception:
            pass

def generate_csrf_token() -> str:
    """
    Genera un token CSRF aleatorio y seguro.
    """
    return secrets.token_urlsafe(32)


def validate_csrf(request: Request) -> None:
    """
    Protege endpoints de escritura frente a CSRF.

    Estrategia: double-submit cookie
    - Cookie: csrftoken_nlp
    - Cabecera: X-CSRFToken
    Deben coincidir.
    """
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get("X-CSRFToken")

    if not cookie_token or not header_token or cookie_token != header_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token inválido o ausente en servicio RAG.",
        )


def make_indexer(*, files, index_filepath, user_id, client_tag="", use_cache: bool = True):
    """
    Crea (o reutiliza) un DocumentIndexer.

    FIX CRÍTICO:
    - No reasigna _INDEXER_CACHE (evita UnboundLocalError).
    - Implementa LRU usando dict (orden de inserción) con pop+reinsert.
    - Cache solo para files=None (lectura).
    - use_cache=False permite bypass (operaciones de escritura/borrado).

    Variables:
    - RAG_INDEXER_CACHE_MAX (default 32). Si 0 => cache deshabilitada.
    """
    if redis_client is None:
        raise RuntimeError("Redis no inicializado todavía (startup no ejecutado).")

    cache_key = (str(index_filepath), str(user_id))

    try:
        cache_max = int(os.getenv("RAG_INDEXER_CACHE_MAX", "32"))
    except Exception:
        cache_max = 32
    cache_max = max(0, cache_max)

    # Reutilizamos indexer solo cuando files=None (modo lectura) y cache habilitada
    if files is None and use_cache and cache_max > 0:
        with _INDEXER_CACHE_LOCK:
            cached = _INDEXER_CACHE.get(cache_key)
            if cached is not None:
                # LRU touch: mover al final (sin reasignar el dict)
                try:
                    _INDEXER_CACHE.pop(cache_key, None)
                    _INDEXER_CACHE[cache_key] = cached
                except Exception:
                    pass

        if cached is not None:
            logging.info(f"[RAG] make_indexer: usando indexer en cache para {cache_key}")
            return cached

    kwargs_common = dict(
        files=files,
        index_filepath=str(index_filepath),
        user_id=user_id,
        client_tag=(client_tag or ""),
        redis_client=redis_client,
    )

    # Estos kwargs extra pueden no existir en DocumentIndexer (tu código hace fallback)
    kwargs_rag = dict(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        embedding_url=LLAMA_EMBED_URL,
        embedding_model=LLAMA_EMBED_MODEL,
        embedding_timeout=LLAMA_EMBED_TIMEOUT,
    )

    # Mantener exactamente tu intención:
    # - files=None: cargar pickles y, si Qdrant está vacío, re-upsert (bootstrap)
    # - files!=None: indexado explícito incremental / desde cero
    force_auto_index_on_load = True

    logging.info(
        f"[RAG] Indexer -> qdrant={QDRANT_HOST}:{QDRANT_PORT} | "
        f"embed={LLAMA_EMBED_URL} | model={LLAMA_EMBED_MODEL} | "
        f"files={'None' if files is None else len(files)}"
    )

    try:
        idx = DocumentIndexer(
            **kwargs_common,
            **kwargs_rag,
            auto_index_on_load=force_auto_index_on_load,
        )
    except TypeError:
        logging.warning(
            "[RAG] DocumentIndexer no acepta kwargs extendidos; usando configuración interna/env."
        )
        idx = DocumentIndexer(
            **kwargs_common,
            auto_index_on_load=force_auto_index_on_load,
        )

    # Guardar en cache sólo en modo lectura si está habilitada
    if files is None and use_cache and cache_max > 0:
        with _INDEXER_CACHE_LOCK:
            # Insert/refresh + LRU touch
            try:
                _INDEXER_CACHE.pop(cache_key, None)
            except Exception:
                pass
            _INDEXER_CACHE[cache_key] = idx

            # Evicción LRU si excede tamaño
            while len(_INDEXER_CACHE) > cache_max:
                try:
                    oldest_key = next(iter(_INDEXER_CACHE))
                    old_idx = _INDEXER_CACHE.pop(oldest_key, None)
                except Exception:
                    break

                try:
                    logging.info(f"[RAG] Evict LRU indexer cache: {oldest_key}")
                except Exception:
                    pass

                # best-effort close (si existe)
                try:
                    qc = getattr(old_idx, "qdrant_client", None) if old_idx else None
                    if qc is not None and hasattr(qc, "close") and callable(qc.close):
                        qc.close()
                except Exception:
                    pass

    return idx

def _search_sync(
    indexer,
    query: str,
    top_k: int,
    matched_tags,
    filters: Optional[Dict[str, Any]] = None,
):
    """
    Wrapper síncrono de búsqueda RAG.
    Se ejecuta fuera del event loop.

    Compatibilidad:
    - Si DocumentIndexer.search soporta 'filters', se pasa.
    - Si no lo soporta (TypeError), fallback a llamada antigua.
    """
    try:
        return indexer.search(
            query=query,
            top_k=top_k,
            matched_tags=matched_tags,
            filters=filters,
        )
    except TypeError:
        # Backwards-compat: DocumentIndexer.search sin parámetro filters
        return indexer.search(
            query=query,
            top_k=top_k,
            matched_tags=matched_tags,
        )

def _search_with_similar_blocks_sync(
    indexer,
    query: str,
    top_k_main: int,
    matched_tags,
    filters: Optional[Dict[str, Any]] = None,
    top_k_similars: int = 4,
):
    """
    Wrapper síncrono que devuelve LISTA DE DICTS (más útil para el orquestador).
    """
    return indexer.search_with_similar_blocks(
        query=query,
        top_k_main=top_k_main,
        top_k_similars=top_k_similars,
        matched_tags=matched_tags,
        filters=filters,
    )

####################################### endpoints para chat_doc ###############################################################################
def _parse_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default

def _safe_json_from_response(resp) -> Optional[dict]:
    try:
        ct = (resp.headers.get("content-type") or "").lower()
        if "application/json" not in ct:
            return None
        body = getattr(resp, "body", None)
        if not body:
            return None
        if isinstance(body, (bytes, bytearray)):
            return json.loads(body.decode("utf-8", errors="ignore"))
        return None
    except Exception:
        return None

def _chatdoc_cache_base(user_id: Any, document_id: str) -> str:
    return f"{user_id}:{document_id}"


def _chatdoc_cache_key(user_id: Any, document_id: str, file_sha256: Optional[str]) -> str:
    sha = (file_sha256 or "").strip() or "nosha"
    return f"{user_id}:{document_id}:{sha}"


def _chatdoc_cache_get(user_id: Any, document_id: str) -> Optional["DocumentChatIndex"]:
    now = time.time()
    base = _chatdoc_cache_base(user_id, document_id)

    with _CHATDOC_CACHE_LOCK:
        full = _CHATDOC_ALIAS.get(base)
        if not full:
            return None

        item = _CHATDOC_CACHE.get(full)
        if not item:
            _CHATDOC_ALIAS.pop(base, None)
            return None

        exp, idx = item
        if exp <= now:
            _CHATDOC_CACHE.pop(full, None)
            _CHATDOC_ALIAS.pop(base, None)
            return None

        _CHATDOC_CACHE.move_to_end(full, last=True)
        return idx


def _chatdoc_cache_put(user_id: Any, document_id: str, index_obj: "DocumentChatIndex", ttl_s: int) -> None:
    try:
        sha = (getattr(index_obj, "metadata", {}) or {}).get("file_sha256")
    except Exception:
        sha = None

    full = _chatdoc_cache_key(user_id, document_id, sha)
    base = _chatdoc_cache_base(user_id, document_id)
    exp = time.time() + max(1, int(ttl_s))

    with _CHATDOC_CACHE_LOCK:
        _CHATDOC_CACHE[full] = (exp, index_obj)
        _CHATDOC_CACHE.move_to_end(full, last=True)
        _CHATDOC_ALIAS[base] = full

        while len(_CHATDOC_CACHE) > max(1, _CHATDOC_CACHE_MAX):
            old_full, _ = _CHATDOC_CACHE.popitem(last=False)
            for k, v in list(_CHATDOC_ALIAS.items()):
                if v == old_full:
                    _CHATDOC_ALIAS.pop(k, None)


def _chatdoc_cache_touch(user_id: Any, document_id: str, ttl_s: int) -> None:
    now = time.time()
    base = _chatdoc_cache_base(user_id, document_id)

    with _CHATDOC_CACHE_LOCK:
        full = _CHATDOC_ALIAS.get(base)
        if not full:
            return
        item = _CHATDOC_CACHE.get(full)
        if not item:
            _CHATDOC_ALIAS.pop(base, None)
            return

        exp, idx = item
        if exp <= now:
            _CHATDOC_CACHE.pop(full, None)
            _CHATDOC_ALIAS.pop(base, None)
            return

        _CHATDOC_CACHE[full] = (now + max(1, int(ttl_s)), idx)
        _CHATDOC_CACHE.move_to_end(full, last=True)

def _chatdoc_error_kind(status_code: int) -> str:
    if status_code == 400:
        return "bad_request"
    if status_code == 403:
        return "auth"
    if status_code == 413:
        return "too_large"
    if status_code == 422:
        return "unprocessable"
    if status_code >= 500:
        return "internal"
    return f"http_{status_code}"


def _record_chatdoc_metrics(path: str, status_code: int, dur_s: float, payload: Optional[dict]) -> None:
    # endpoint label estable
    endpoint = path

    # status semántico (si viene en JSON)
    status = None
    if isinstance(payload, dict):
        status = payload.get("status")
    if not status:
        status = "error" if status_code >= 400 else "ok"

    try:
        CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint, status=status).inc()
    except Exception:
        pass

    # Errores
    if status != "ok" or status_code >= 400:
        try:
            kind = _chatdoc_error_kind(int(status_code or 500))
            # Si el JSON no se pudo parsear en un caso que debería -> kind json_parse
            if status_code >= 400 and payload is None:
                kind = "json_parse"
            CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint, kind=kind).inc()
        except Exception:
            pass

    # Cache hit/miss (query + summary leen Redis; ingest escribe)
    if endpoint in ("/chatdoc/query", "/chatdoc/summary"):
        try:
            hit = "miss" if status == "no_index" else "hit"
            CHATDOC_CACHE_TOTAL.labels(endpoint=endpoint, result=hit).inc()
        except Exception:
            pass

    # Métricas específicas por endpoint (extraídas del JSON de respuesta)
    if endpoint == "/chatdoc/ingest" and isinstance(payload, dict):
        mode = "unknown"
        try:
            # preferimos trace.mode si viene
            trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
            mode = (trace.get("mode") or ("bytes" if payload.get("mime_type") else "unknown")) or "unknown"
            if mode not in ("bytes", "text", "unknown"):
                mode = "unknown"
        except Exception:
            mode = "unknown"

        try:
            timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}
            build_s = timings.get("build_s")
            redis_s = timings.get("redis_s")
            total_s = timings.get("total_s")

            if isinstance(build_s, (int, float)):
                CHATDOC_INGEST_BUILD_SECONDS.labels(mode=mode).observe(float(build_s))
            if isinstance(redis_s, (int, float)):
                CHATDOC_INGEST_REDIS_SECONDS.observe(float(redis_s))
            if isinstance(total_s, (int, float)):
                CHATDOC_INGEST_TOTAL_SECONDS.observe(float(total_s))
            else:
                # fallback al tiempo real del middleware
                CHATDOC_INGEST_TOTAL_SECONDS.observe(float(dur_s))
        except Exception:
            pass

        try:
            chunk_count = payload.get("chunk_count")
            char_length = payload.get("char_length")
            page_count = payload.get("page_count")

            if isinstance(chunk_count, int):
                CHATDOC_INGEST_CHUNKS.labels(mode=mode).observe(chunk_count)
            if isinstance(char_length, int):
                CHATDOC_INGEST_TEXT_CHARS.labels(mode=mode).observe(char_length)
            if isinstance(page_count, int) and page_count > 0:
                CHATDOC_INGEST_PAGE_COUNT.observe(page_count)
        except Exception:
            pass

        try:
            trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
            bytes_received = trace.get("bytes_received")
            if isinstance(bytes_received, int) and bytes_received > 0:
                CHATDOC_INGEST_BYTES.labels(mode=mode).observe(bytes_received)
        except Exception:
            pass

    if endpoint == "/chatdoc/query" and isinstance(payload, dict):
        try:
            results = payload.get("results")
            if isinstance(results, list):
                CHATDOC_QUERY_RESULTS_COUNT.observe(len(results))
        except Exception:
            pass

    if endpoint == "/chatdoc/summary" and isinstance(payload, dict):
        try:
            strategy = (payload.get("strategy") or "none")
            detail_level = payload.get("detail_level") or "none"
            if not isinstance(strategy, str):
                strategy = "none"
            if not isinstance(detail_level, str):
                detail_level = "none"
            strategy = strategy.strip().lower()
            detail_level = detail_level.strip().lower()

            frags = payload.get("fragments")
            if isinstance(frags, list):
                CHATDOC_SUMMARY_FRAGMENTS.labels(strategy=strategy, detail_level=detail_level).observe(len(frags))
        except Exception:
            pass

def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off"):
            return False
    return default


def _safe_dict(d: Any) -> Dict[str, Any]:
    return d if isinstance(d, dict) else {}


def _sha256_bytes(data: bytes) -> str:
    try:
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return ""


def _normalize_filename(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "document"
    # defensivo contra nombres absurdos
    return n[:200]


def _infer_mime_from_filename(filename: str) -> Optional[str]:
    # Fallback muy ligero si no viene mime_type
    fn = (filename or "").lower()
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if fn.endswith(".pptx"):
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if fn.endswith(".txt"):
        return "text/plain"
    return None


def _chatdoc_key(user_id: Any, document_id: str) -> str:
    return f"chatdoc:{user_id}:{document_id}"

def _compute_coverage(index: Any, fragments: List[Dict[str, Any]]) -> Optional[float]:
    try:
        full_len = len(getattr(index, "full_text", "") or "")
        if full_len <= 0:
            return None

        spans = []
        for frag in fragments:
            try:
                s = int(frag.get("char_start", 0))
                e = int(frag.get("char_end", 0))
            except Exception:
                continue
            if e <= s:
                continue
            s = max(0, min(s, full_len))
            e = max(0, min(e, full_len))
            spans.append((s, e))

        if not spans:
            return None

        spans.sort(key=lambda x: x[0])
        covered = 0
        cur_s, cur_e = spans[0]

        for s, e in spans[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                covered += (cur_e - cur_s)
                cur_s, cur_e = s, e

        covered += (cur_e - cur_s)
        return covered / float(full_len)
    except Exception:
        return None


def _infer_defaults_by_doc_size(index: Any) -> Dict[str, int]:
    """
    Defaults adaptativos para documentos grandes.
    No rompe compatibilidad porque sólo se aplica cuando
    el cliente NO especifica max_fragments/min_chars explícitos.
    """
    try:
        chunks = len(getattr(index, "_chunks", []) or [])
        full_len = len(getattr(index, "full_text", "") or "")
    except Exception:
        chunks, full_len = 0, 0

    # Base
    base_min_chars = _parse_int(os.getenv("CHATDOC_SUMMARY_MIN_CHARS_BASE", "300"), 300)
    base_medium = _parse_int(os.getenv("CHATDOC_SUMMARY_FRAGMENTS_MEDIUM", "12"), 12)

    # Escalado suave
    if chunks >= 120 or full_len >= 250_000:
        # Documentos muy grandes
        return {
            "max_fragments": min(48, max(base_medium * 3, 24)),
            "min_chars_per_chunk": max(150, int(base_min_chars * 0.8)),
        }
    if chunks >= 60 or full_len >= 120_000:
        return {
            "max_fragments": min(36, max(base_medium * 2, 18)),
            "min_chars_per_chunk": max(180, int(base_min_chars * 0.9)),
        }

    return {
        "max_fragments": base_medium,
        "min_chars_per_chunk": base_min_chars,
    }


def verify_token(request: Request):
    """
    Extrae el JWT de la cookie 'access_token' o del header Authorization.
    Devuelve el payload decodificado (user_id, role, etc.).
    """
    access_token = request.cookies.get("access_token")
    token_str = None

    if access_token:
        token_str = access_token.replace("Bearer ", "").strip()
    else:
        auth = request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token_str = auth[7:].strip()

    if not token_str:
        raise HTTPException(status_code=403, detail="No autorizado: token ausente.")

    try:
        payload = jwt.decode(token_str, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=403, detail="Token sin user_id.")
        return payload
    except JWTError:
        raise HTTPException(status_code=403, detail="Token no válido o expirado.")

def _safe_inc(fn):
    try:
        fn()
    except Exception:
        pass

def _safe_observe(fn):
    try:
        fn()
    except Exception:
        pass

def _chatdoc_emit_error(endpoint: str, kind: str, t0: float, status_sem: str = "error"):
    # status semántico + tipo de error + (si aplica) tiempos
    _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint, status=status_sem).inc())
    _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint, kind=kind).inc())
    # ingest total (solo para ingest)
    if endpoint == "/chatdoc/ingest":
        _safe_observe(lambda: CHATDOC_INGEST_TOTAL_SECONDS.observe(time.time() - t0))

def _chatdoc_emit_ingest_ok(
    *,
    t0: float,
    mode: str,
    build_s: float,
    redis_s: float,
    raw_bytes: Optional[bytes],
    text: str,
    chunk_count: int,
    char_length: int,
    page_count: int,
    hinted_page_count: Optional[int],
):
    endpoint = "/chatdoc/ingest"
    _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint, status="ok").inc())

    total_s = time.time() - t0
    _safe_observe(lambda: CHATDOC_INGEST_TOTAL_SECONDS.observe(total_s))
    _safe_observe(lambda: CHATDOC_INGEST_REDIS_SECONDS.observe(redis_s))
    _safe_observe(lambda: CHATDOC_INGEST_BUILD_SECONDS.labels(mode=mode).observe(build_s))

    # bytes (con label mode)
    if raw_bytes is not None:
        size_b = len(raw_bytes)
        _safe_observe(lambda: CHATDOC_INGEST_BYTES.labels(mode="bytes").observe(size_b))
    elif text:
        size_b = len(text.encode("utf-8"))
        _safe_observe(lambda: CHATDOC_INGEST_BYTES.labels(mode="text").observe(size_b))
    else:
        _safe_observe(lambda: CHATDOC_INGEST_BYTES.labels(mode=mode).observe(0))

    # chunks + chars (con label mode)
    _safe_observe(lambda: CHATDOC_INGEST_CHUNKS.labels(mode=mode).observe(chunk_count))
    _safe_observe(lambda: CHATDOC_INGEST_TEXT_CHARS.labels(mode=mode).observe(char_length))

    # page_count (sin label)
    if page_count and page_count > 0:
        _safe_observe(lambda: CHATDOC_INGEST_PAGE_COUNT.observe(page_count))
    elif hinted_page_count and hinted_page_count > 0:
        _safe_observe(lambda: CHATDOC_INGEST_PAGE_COUNT.observe(hinted_page_count))



@app.on_event("startup")
async def initialize_redis():
    """
    Conexión real a Redis. El login-service es quien crea 'session:<user_id>'.
    """
    global redis_client, redis_index_client

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
    redis_index_client = redis.Redis(host=redis_host, port=redis_port, db=1, decode_responses=True)

    logging.info(f"[startup] Redis conectado en {redis_host}:{redis_port}")
    logging.info(f"[startup] Qdrant: {QDRANT_HOST}:{QDRANT_PORT} | llama-server: {LLAMA_EMBED_URL} ({LLAMA_EMBED_MODEL})")


# =========================================================
# Helpers sesión
# =========================================================
async def get_session_from_redis(user_id: int):
    raw = await redis_client.get(f"session:{user_id}")
    if raw:
        return json.loads(raw)
    return None


#Middleware métricas
_USER_LABEL_MODE = os.getenv("RAG_METRICS_USER_LABEL", "hash").strip().lower()
_PER_USER_ENABLED = os.getenv("RAG_METRICS_PER_USER", "1").strip() not in ("0", "false", "no")

def _user_label(raw_user: object) -> str:
    if raw_user is None:
        return "unknown"
    s = str(raw_user).strip()
    if not s:
        return "unknown"
    if not _PER_USER_ENABLED:
        return "off"
    if _USER_LABEL_MODE == "raw":
        # ⚠️ cuidado cardinalidad / PII
        return s[:64]
    if _USER_LABEL_MODE == "off":
        return "off"
    # default: hash estable corto
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

@app.middleware("http")
async def rag_prometheus_middleware(request: Request, call_next):
    # Evita auto-scrape/ruido
    if request.url.path == "/metrics":
        return await call_next(request)

    endpoint = request.url.path  # rutas fijas (si usas router, mejor request.scope['route'].path)
    method = request.method
    start = time.perf_counter()

    try:
        RAG_HTTP_INFLIGHT.labels(endpoint=endpoint).inc()
    except Exception:
        pass

    status_code = 500
    response = None

    try:
        response = await call_next(request)
        status_code = getattr(response, "status_code", 200) or 200
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        dur = time.perf_counter() - start
        status_class = f"{int(status_code) // 100}xx"

        # --- HTTP global por endpoint ---
        try:
            RAG_HTTP_REQUESTS_TOTAL.labels(
                endpoint=endpoint, method=method, status_class=status_class
            ).inc()
            RAG_HTTP_LATENCY_SECONDS.labels(
                endpoint=endpoint, method=method
            ).observe(dur)
        except Exception:
            pass

        # --- Per-user (el endpoint debe setear request.state.user_id) ---
        try:
            user_id = getattr(request.state, "user_id", None)
            if user_id is not None and _PER_USER_ENABLED:
                RAG_USER_ENDPOINT_REQUESTS_TOTAL.labels(
                    endpoint=endpoint,
                    user=_user_label(user_id),
                    status_class=status_class,
                ).inc()
        except Exception:
            pass

        # --- ChatDoc semantic metrics (best-effort) ---
        try:
            if endpoint.startswith("/chatdoc/"):
                payload = _safe_json_from_response(response) if response is not None else None
                _record_chatdoc_metrics(
                    path=endpoint,
                    status_code=int(status_code or 500),
                    dur_s=float(dur),
                    payload=payload,
                )
        except Exception:
            pass

        # --- Inflight ---
        try:
            RAG_HTTP_INFLIGHT.labels(endpoint=endpoint).dec()
        except Exception:
            pass
        
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

@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/csrf-token")
async def get_nlp_csrf_token(request: Request) -> JSONResponse:
    """
    Devuelve un token CSRF para el micro NLP (servicio RAG) y lo fija
    en la cookie `CSRF_COOKIE_NAME` (p.ej. csrftoken_nlp).

    Se usa el mismo esquema double-submit cookie que valida `validate_csrf`:
    - Cookie: CSRF_COOKIE_NAME
    - Cabecera: X-CSRFToken
    """
    token = generate_csrf_token()

    response = JSONResponse({"csrf_token": token})
    response.set_cookie(
        key=CSRF_COOKIE_NAME,   # mismo nombre que usa validate_csrf
        value=token,
        httponly=False,         # JS debe poder leerla para enviar X-CSRFToken
        secure=False,           # en prod: True (HTTPS)
        samesite="Strict",      # o "Lax" según tu escenario
    )
    return response



# =========================================================
# Health RAG: llama-server + Qdrant
# =========================================================
@app.get("/rag_health")
async def rag_health():
    """
    Verifica:
    - llama-server (endpoint embeddings)
    - Qdrant (lista colecciones)
    """
    # 1) llama-server
    try:
        r = requests.post(
            LLAMA_EMBED_URL,
            json={"model": LLAMA_EMBED_MODEL, "input": "probar embedding"},
            timeout=LLAMA_EMBED_TIMEOUT
        )
        r.raise_for_status()
        emb_len = len(r.json()["data"][0]["embedding"])
        emb_ok = True
    except Exception as e:
        emb_ok = False
        emb_err = str(e)

    # 2) Qdrant
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
        cols = qc.get_collections()
        qdrant_ok = True
        col_names = [c.name for c in cols.collections]
    except Exception as e:
        qdrant_ok = False
        col_names = []
        qdrant_err = str(e)

    out = {
        "llama_server": {"url": LLAMA_EMBED_URL, "model": LLAMA_EMBED_MODEL, "ok": emb_ok},
        "qdrant": {"host": QDRANT_HOST, "port": QDRANT_PORT, "ok": qdrant_ok, "collections": col_names},
    }
    if not emb_ok:
        out["llama_server"]["error"] = emb_err
    if not qdrant_ok:
        out["qdrant"]["error"] = qdrant_err

    status = 200 if (emb_ok and qdrant_ok) else 500
    return JSONResponse(status_code=status, content=out)

@app.get("/upload_context")
async def get_upload_context(token: dict = Depends(verify_token)) -> JSONResponse:
    """
    Devuelve el contexto necesario para el panel de 'Cargar e Indexar':
      - role: rol del usuario (Supervisor, User, etc.)
      - departments: lista de departamentos a los que tiene acceso (solo si Supervisor)
      - user_directory: directorio privado del usuario

    NO modifica nada ni requiere CSRF (solo lectura).
    """
    user_id = token.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token sin user_id")

    session_data = await get_session_from_redis(user_id)
    if session_data is None:
        raise HTTPException(status_code=403, detail="Sesión no encontrada o expirada.")

    user_role = session_data.get("role")
    user_departments = session_data.get("departments", [])
    user_directory = session_data.get("user_directory")

    # Por coherencia con la plantilla antigua:
    departments = user_departments if user_role == "Supervisor" else []

    return JSONResponse(
        {
            "role": user_role,
            "departments": departments,      # misma estructura que en la sesión
            "user_directory": user_directory,
        }
    )


@app.get("/upload_file", response_class=HTMLResponse)
async def show_upload_form(request: Request, token: dict = Depends(verify_token)):
    user_id = token.get("user_id")
    request.state.user_id = user_id
    session_data = await get_session_from_redis(user_id)
    if session_data is None:
        raise HTTPException(status_code=403, detail="Sesión no encontrada o expirada.")

    user_role = session_data.get("role")
    user_departments = session_data.get("departments", [])
    departments = user_departments if user_role == "Supervisor" else []

    # Generar token CSRF para esta vista
    csrf_token = generate_csrf_token()

    context = {
        "request": request,
        "role": user_role,
        "departments": departments,
        "user_directory": session_data.get("user_directory"),
        "csrf_token": csrf_token,
    }

    response = templates.TemplateResponse("upload_file.html", context)

    # Cookie CSRF legible desde JS (no httponly)
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        httponly=False,      # Para que JS pueda leerla y mandarla en X-CSRFToken
        secure=False,        #En producción: True (HTTPS)
        samesite="Strict",   # O "Lax" según tu escenario
    )

    return response

@app.post("/upload_file")
async def upload_file(
    request: Request,
    files: List[UploadFile] = File(...),
    department: Optional[str] = Form(None),
    token: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    # --- Protección CSRF ---
    validate_csrf(request)

    user_id = token.get("user_id")
    session_data = await get_session_from_redis(user_id)
    if session_data is None:
        raise HTTPException(status_code=403, detail="Sesión no encontrada o expirada.")

    user_directory = session_data.get("user_directory")
    user_role = token.get("role")
    user_departments = session_data.get("departments", [])

    if user_role not in [ROLE_SUPERVISOR, ROLE_USER]:
        raise HTTPException(status_code=403, detail="Permisos insuficientes para cargar archivos.")

    # Directorio físico de destino
    if department:
        department_data = next(
            (dep for dep in user_departments if dep.get("department_directory") == department),
            None
        )
        if not department_data:
            raise HTTPException(status_code=403, detail=f"No tiene acceso al departamento {department}.")

        # department_directory: "departments/enagas"
        target_directory = PATH_BASE_DEPARTMENTS / department_data["department_directory"] / "uploaded_files"
        department_obj = db.query(Department).filter(
            Department.department_directory == department_data["department_directory"]
        ).first()
    else:
        target_directory = PATH_BASE / user_directory / "uploaded_files"
        department_obj = None

    target_directory.mkdir(parents=True, exist_ok=True)

    uploaded_files = []

    for f in files:
        file_path = target_directory / f.filename
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await f.read()
            await out_file.write(content)

        uploaded_files.append(str(file_path))

        # -------- Persistencia real en BD --------
        file_record = FileModel(
            user_id=user_id,
            department_id=department_obj.id if department_obj else None,
            file_path=str(file_path),
            file_name=f"[CONOCIMIENTO] {f.filename}",
            permission='READ',  # Enum('READ', 'WRITE', 'NONE')
            created_at=datetime.utcnow()
        )

        db.add(file_record)

        audit = AuditLog(
            user_id=user_id,
            entity_name="File",
            entity_id=file_record.id or 0,
            action="CREATE",
            old_data=None,
            new_data={
                "path": str(file_path),
                "label": "CONOCIMIENTO",
                "department": department or "private"
            },
            timestamp=datetime.utcnow()
        )
        db.add(audit)

    db.commit()
    return JSONResponse(status_code=200, content={"message": "Archivos cargados correctamente.", "files": uploaded_files})

@app.get("/list_files")
async def list_files(
    department: Optional[str] = Query(None),
    token: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    user_id = token["user_id"]
    session = await get_session_from_redis(user_id)
    if not session:
        raise HTTPException(403, "Sesión no encontrada o expirada.")

    user_departments = session.get("departments", [])
    user_directory = session.get("user_directory")

    if department:
        department_data = next(
            (dep for dep in user_departments if dep.get("department_directory") == department),
            None
        )
        if not department_data:
            raise HTTPException(403, f"No tiene acceso al departamento {department}.")
        target = PATH_BASE_DEPARTMENTS / department_data["department_directory"] / "uploaded_files"
    else:
        target = PATH_BASE / user_directory / "uploaded_files"

    files = [p.name for p in target.iterdir() if p.is_file()] if target.exists() else []

    db.add(AuditLog(
        user_id=user_id,
        entity_name="FileList",
        entity_id=0,
        action="UPDATE",
        old_data=None,
        new_data={"department": department or "private", "files": files},
        timestamp=datetime.utcnow()
    ))
    db.commit()

    return {"files": files}


@app.delete("/delete_files")
async def delete_files(
    request: Request,
    payload: dict = Body(...),
    token: dict = Depends(verify_token),
    db: Session = Depends(get_db),
):
    validate_csrf(request)

    filenames: List[str] = payload.get("filenames", [])
    department: Optional[str] = payload.get("department", None)

    user_id = token.get("user_id")
    request.state.user_id = user_id

    session_data = await get_session_from_redis(user_id)
    if session_data is None:
        raise HTTPException(status_code=403, detail="Sesión no encontrada o expirada.")

    user_directory = session_data.get("user_directory")
    user_departments = session_data.get("departments", [])

    if department:
        department_data = next(
            (dep for dep in user_departments if dep.get("department_directory") == department),
            None
        )
        if not department_data:
            raise HTTPException(status_code=403, detail=f"No tiene acceso al departamento {department}.")

        target_directory = PATH_BASE_DEPARTMENTS / department_data["department_directory"] / "uploaded_files"

        rec = db.query(Department).filter(
            Department.department_directory == department_data["department_directory"]
        ).first()
        if not rec or not rec.faiss_index_path:
            raise HTTPException(status_code=404, detail="El índice vectorial del departamento no está inicializado.")

        index_base = rec.faiss_index_path
        redis_key = f"indexer:{department_data['department_directory']}"
        indexer_user_id = Path(department_data["department_directory"]).name
    else:
        target_directory = PATH_BASE / user_directory / "uploaded_files"
        index_base = PATH_BASE / user_directory / "qdrant_indices" / "user_index"
        redis_key = f"indexer:{user_directory}"
        indexer_user_id = user_id

    dn_file = f"{index_base}_docnames.pkl"
    docs_file = f"{index_base}_documents.pkl"
    if not os.path.exists(dn_file) or not os.path.exists(docs_file):
        raise HTTPException(status_code=404, detail="Los archivos de índice no existen.")

    #  Importante: invalidar cache antes de tocar datos (evita usar un indexer cacheado)
    _invalidate_indexer_cache(str(index_base), str(indexer_user_id))

    # Instanciar indexer en thread (bypass cache: vamos a mutar)
    indexer = await asyncio.to_thread(
        make_indexer,
        files=None,
        index_filepath=str(index_base),
        user_id=indexer_user_id,
        client_tag="",
        use_cache=False,
    )

    raw_idx = await redis_client.get(redis_key)
    if raw_idx:
        redis_data = json.loads(raw_idx)
        if department:
            processed_files = set(redis_data.get("files", []))
        else:
            processed_files = set(redis_data.get("user", {}).get("files", []))
    else:
        processed_files = set()

    deleted: List[str] = []
    not_found: List[str] = []
    errors: List[dict] = []

    for fn in filenames:
        file_path = target_directory / fn
        if not file_path.exists():
            not_found.append(fn)
            continue

        try:
            file_path.unlink(missing_ok=False)

            # delete_document_by_name en thread
            removed_fragments = await asyncio.to_thread(indexer.delete_document_by_name, str(file_path))
            logging.info(f"[delete_files] Eliminados {removed_fragments} fragmentos para '{file_path}'")

            if department:
                norm_file = os.path.normpath(str(file_path))
            else:
                norm_file = os.path.normcase(os.path.normpath(str(file_path)))
            processed_files.discard(norm_file)

            deleted.append(fn)

            file_row = db.query(FileModel).filter(FileModel.file_path == str(file_path)).first()
            if file_row:
                db.delete(file_row)

            db.add(AuditLog(
                user_id=user_id,
                entity_name="File",
                entity_id=0,
                action="DELETE",
                old_data={"path": str(file_path)},
                new_data=None,
                timestamp=datetime.utcnow()
            ))
            db.add(AuditLog(
                user_id=user_id,
                entity_name="Index",
                entity_id=0,
                action="UPDATE",
                old_data={"removed_file": fn},
                new_data={"remaining_docs": len(indexer.documents)},
                timestamp=datetime.utcnow()
            ))

        except Exception as e:
            logging.error(f"[delete_files] Error eliminando {fn}: {e}")
            errors.append({"file": fn, "error": str(e)})

    if department:
        updated_data = {"index_filepath": str(index_base), "files": list(processed_files)}
    else:
        updated_data = {"user": {"index_filepath": str(index_base), "files": list(processed_files)}}

    await redis_client.set(redis_key, json.dumps(updated_data))

    db.add(AuditLog(
        user_id=user_id,
        entity_name="RedisIndex",
        entity_id=0,
        action="UPDATE",
        old_data=None,
        new_data=updated_data,
        timestamp=datetime.utcnow()
    ))
    db.commit()

    #  Importante: invalidar cache tras borrar (evita lecturas desfasadas)
    _invalidate_indexer_cache(str(index_base), str(indexer_user_id))

    result: dict = {"deleted": deleted}
    if not_found:
        result["not_found"] = not_found
    if errors:
        result["errors"] = errors

    return JSONResponse(status_code=200, content=result)


def _candidate_user_scan_dirs(user_directory: str) -> List[Path]:
    """
    Devuelve las carpetas candidatas donde buscar archivos del usuario.
    Incluye:
      - STORAGE_ROOT/<user_directory>/uploaded_files
      - STORAGE_ROOT/uploaded_files
      - <repo_root>/<user_directory>/uploaded_files
      - <repo_root>/uploaded_files
      - <repo_root>/Archivos_pruebas
    """
    repo_root = Path(__file__).resolve().parent
    base = PATH_BASE  # = STORAGE_ROOT

    candidates = [
        base / user_directory / "uploaded_files",
        base / "uploaded_files",
        repo_root / user_directory / "uploaded_files",
        repo_root / "uploaded_files",
        #repo_root / "Archivos_pruebas",
    ]

    seen = set()
    cleaned: List[Path] = []
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            continue
        key = os.path.normcase(str(rp))
        if key not in seen and rp.exists() and rp.is_dir():
            seen.add(key)
            cleaned.append(rp)
    return cleaned


# ----------------- Utilidades extracción de departamento -----------------
def extract_department_from_query(prompt, departments, similarity_threshold=0.8):
    prompt_lower = prompt.lower()
    department_mapping, department_names = {}, []
    for dept in departments:
        name = dept["department_directory"].replace("\\", "/").split("/")[-1].lower()
        norm_name = name.rstrip("s")
        department_names.append((name, norm_name))
        department_mapping[name] = dept

    tokens = re.findall(r"\b\w+\b", prompt_lower)

    for name, norm_name in department_names:
        pattern = rf"\b{name}\b"
        pattern_plural = rf"\b{norm_name}s?\b"
        if re.search(pattern, prompt_lower) or re.search(pattern_plural, prompt_lower):
            cleaned = re.sub(pattern_plural, "", prompt_lower).strip()
            return department_mapping[name], cleaned

    for token in tokens:
        for name, norm_name in department_names:
            score = SequenceMatcher(None, token, norm_name).ratio()
            if score >= similarity_threshold:
                cleaned_tokens = [t for t in tokens if t != token]
                return department_mapping[name], " ".join(cleaned_tokens)

    return None, prompt

# ----------------- RAG endpoints -----------------
@app.get('/instantiate_document_indexers')
async def instantiate_document_indexers(
    request: Request,
    token: dict = Depends(verify_token)):
    try:
        user_id = token.get("user_id")
        request.state.user_id = user_id

        session_data = await get_session_from_redis(user_id)
        if not session_data:
            return JSONResponse(status_code=404, content={"message": "No se encontró la sesión en Redis."})

        user_directory = session_data.get("user_directory")
        user_departments = session_data.get("departments", [])
        document_indexers = {}

        # Usuario
        if user_directory:
            user_indexer_json = await redis_client.get(f"indexer:{user_directory}")
            if user_indexer_json:
                user_indexer = json.loads(user_indexer_json).get("user")
                if user_indexer:
                    index_filepath = user_indexer.get("index_filepath")
                    if index_filepath:
                        base = os.path.splitext(index_filepath)[0]
                        dn = f"{base}_docnames.pkl"
                        dc = f"{base}_documents.pkl"
                        if os.path.exists(dn) and os.path.exists(dc):
                            document_indexers["user"] = await asyncio.to_thread(
                                make_indexer,
                                files=None,
                                index_filepath=index_filepath,
                                user_id=user_id,
                                client_tag=""
                            )

        # Departamentos
        for department in user_departments:
            department_directory = department.get("department_directory")
            if not department_directory:
                continue
            index_filepath = department.get("index_filepath") or department.get("faiss_index_path")
            if not index_filepath:
                continue
            base = os.path.splitext(index_filepath)[0]
            dn = f"{base}_docnames.pkl"
            dc = f"{base}_documents.pkl"
            if not (os.path.exists(dn) and os.path.exists(dc)):
                continue

            document_indexers[department_directory] = await asyncio.to_thread(
                make_indexer,
                files=None,
                index_filepath=index_filepath,
                user_id=user_id,
                client_tag=""
            )

        return JSONResponse(status_code=200, content={"message": "DocumentIndexers instanciados con éxito."})

    except Exception as e:
        logging.info(f"Error en instantiate_document_indexers: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Error inesperado al instanciar los DocumentIndexers."})


@app.get('/process_user_files')
async def process_user_files(
    request: Request,
    token: dict = Depends(verify_token),
    client_tag: str = Query(None, description="Etiqueta de cliente para enriquecer los fragmentos"),
    scan_dir: Optional[str] = Query(None, description="Directorio explícito a escanear (override opcional)")
):

    validate_csrf(request)

    t0 = datetime.utcnow()
    try:
        user_id = token.get("user_id")
        request.state.user_id = user_id

        session_key = f"session:{user_id}"
        raw = await redis_client.get(session_key)
        try:
            session_data = json.loads(raw) if raw else {}
        except Exception:
            session_data = {}

        # client_tag persistente en sesión
        if client_tag:
            session_data['client_tag'] = client_tag.strip()
            await redis_client.set(session_key, json.dumps(session_data))
        else:
            client_tag = session_data.get('client_tag')

        user_directory = session_data.get("user_directory")
        if not user_directory:
            return JSONResponse(status_code=200, content={
                "status": "error",
                "error": "No se ha encontrado el directorio privado en sesión.",
                "duration_s": (datetime.utcnow() - t0).total_seconds()
            })

        # 1) Construir lista de carpetas a escanear
        if scan_dir:
            p = Path(scan_dir).resolve()
            if not p.exists() or not p.is_dir():
                return JSONResponse(status_code=200, content={
                    "status": "error",
                    "error": "El 'scan_dir' indicado no existe o no es un directorio.",
                    "scan_dir": str(p),
                    "duration_s": (datetime.utcnow() - t0).total_seconds()
                })
            scan_dirs: List[Path] = [p]
        else:
            scan_dirs = _candidate_user_scan_dirs(user_directory)

        if not scan_dirs:
            return JSONResponse(status_code=200, content={
                "status": "ok",
                "message": "No se encontraron carpetas candidatas para escanear.",
                "user_directory": user_directory,
                "scanned_dirs": [],
                "found_count": 0,
                "duration_s": (datetime.utcnow() - t0).total_seconds()
            })

        # 2) Localizar ficheros válidos (recursivo en todas las carpetas)
        valid_exts = {e.lower() for e in valid_extensions}
        found_files: List[str] = []
        for root in scan_dirs:
            for r, _, files in os.walk(root):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        found_files.append(str(Path(r) / f))
        found_files = sorted(list(dict.fromkeys(found_files)))

        if not found_files:
            return JSONResponse(status_code=200, content={
                "status": "ok",
                "message": "No hay archivos válidos para indexar en las rutas escaneadas.",
                "user_directory": user_directory,
                "scanned_dirs": [str(d) for d in scan_dirs],
                "valid_extensions": sorted(list(valid_exts)),
                "found_count": 0,
                "duration_s": (datetime.utcnow() - t0).total_seconds()
            })

        # 3) Ruta del índice SIEMPRE en el espacio del usuario bajo STORAGE_ROOT/PATH_BASE
        idx_dir = PATH_BASE / user_directory / "qdrant_indices"
        idx_dir.mkdir(exist_ok=True, parents=True)
        index_filepath = idx_dir / "user_index"

        # 3.1) Detectar si el índice (pickles) existe
        dn_file = f"{index_filepath}_docnames.pkl"
        dc_file = f"{index_filepath}_documents.pkl"
        pickles_exist = os.path.exists(dn_file) and os.path.exists(dc_file)

        # 4) Leer estado previo (lista de ya procesados) desde Redis
        redis_key = f"indexer:{user_directory}"
        raw_idx = await redis_client.get(redis_key)
        stored = {}
        if raw_idx:
            try:
                stored = json.loads(raw_idx) or {}
            except Exception:
                stored = {}
        processed = set((stored.get("user", {}) or {}).get("files", []) or [])

        # 5) Normalizar paths (Windows-safe) y calcular incrementales
        def norm(p: str) -> str:
            return os.path.normcase(os.path.normpath(str(p or "")))
        norm_processed = {norm(p) for p in processed}
        norm_files     = {norm(p) for p in found_files}
        new_files_norm = sorted(list(norm_files - norm_processed))
        orig_by_norm = {norm(fp): fp for fp in found_files}
        new_files_abs = [orig_by_norm[n] for n in new_files_norm]

        # --- warmup ligero: sin query a Qdrant ---
        def _warmup_models(idx: DocumentIndexer) -> None:
            if os.getenv("RAG_WARMUP_MODELS", "1").strip().lower() in ("0", "false", "no", "off"):
                return

            # 1) E5 (SentenceTransformer)
            try:
                if hasattr(idx, "_embed_texts") and callable(idx._embed_texts):
                    _ = idx._embed_texts(["warmup"], is_query=True, batch_size=1)
                elif getattr(idx, "sbert_model", None) is not None:
                    _ = idx.sbert_model.encode(
                        ["query: warmup"],
                        batch_size=1,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
            except Exception:
                pass

            # 2) SPLADE (worker o load model)
            if os.getenv("QDRANT_ENABLE_SPARSE", "1") == "1":
                deferred = os.getenv("QDRANT_SPLADE_DEFERRED", "0") == "1"
                if deferred:
                    try:
                        if hasattr(idx, "_ensure_sparse_worker") and callable(idx._ensure_sparse_worker):
                            idx._ensure_sparse_worker()
                    except Exception:
                        pass
                else:
                    try:
                        if hasattr(idx, "_get_splade_model") and callable(idx._get_splade_model):
                            idx._get_splade_model()
                    except Exception:
                        pass

            # 3) (Opcional) reranker/colbert si quieres precargar explícitamente
            if os.getenv("RAG_WARMUP_RERANK", "0") == "1":
                try:
                    if hasattr(idx, "_get_reranker") and callable(idx._get_reranker):
                        _ = idx._get_reranker()
                except Exception:
                    pass
            if os.getenv("RAG_WARMUP_COLBERT", "0") == "1":
                try:
                    if hasattr(idx, "_get_colbert_encoder") and callable(idx._get_colbert_encoder):
                        _ = idx._get_colbert_encoder()
                except Exception:
                    pass

        # 6) Indexado (incremental / carga / primer indexado si no hay pickles)
        def _do_index(files_to_process: Optional[List[str]]):
            """
            Construye/actualiza el índice en Qdrant (dense + lo que defina DocumentIndexer).
            Mantiene tu lógica (incremental/initial/attach).
            Warmup: ahora es ligero y NO hace query a Qdrant.
            """
            idx = make_indexer(
                files=files_to_process,
                index_filepath=str(index_filepath),
                user_id=user_id,
                client_tag=client_tag or ""
            )

            # Verificación mínima de que el modelo de embeddings está operativo
            if not hasattr(idx, "sbert_model") or idx.sbert_model is None:
                raise RuntimeError("DocumentIndexer se creó sin sbert_model (E5). Revisa HF_EMBED_MODEL/LOCAL_EMBED_DIR.")
            try:
                dim = int(idx.sbert_model.get_sentence_embedding_dimension())
                if dim <= 0:
                    raise RuntimeError("Dimensión de sbert_model inválida.")
            except Exception as e:
                raise RuntimeError(f"sbert_model no usable: {e}")

            #  Warmup solo cuando es attach (files_to_process is None),
            #    porque en ingest ya se embebe y/o arranca SPLADE en rag_upsert.
            if files_to_process is None:
                try:
                    _warmup_models(idx)
                except Exception:
                    pass

        action = "no_new_files"
        loop = asyncio.get_running_loop()

        if new_files_abs:
            # Indexado incremental
            await loop.run_in_executor(None, _do_index, new_files_abs)
            norm_processed |= set(new_files_norm)
            action = "incremental"
            _invalidate_indexer_cache(str(index_filepath), str(user_id))
        else:
            if not pickles_exist and found_files:
                # Primer indexado
                await loop.run_in_executor(None, _do_index, found_files)
                norm_processed = set(norm_files)
                action = "initial_index"
                _invalidate_indexer_cache(str(index_filepath), str(user_id))
            else:
                # Solo carga/attach al índice existente
                await loop.run_in_executor(None, _do_index, None)
                # Aquí no invalidamos cache porque no hay cambios en documentos

        # 7) Persistir estado en Redis
        await redis_client.set(
            redis_key,
            json.dumps({
                "user": {
                    "index_filepath": str(index_filepath),
                    "files": list(norm_processed)
                }
            })
        )

        return JSONResponse(status_code=200, content={
            "status": "ok",
            "message": "Procesamiento completado.",
            "action": action,
            "user_directory": user_directory,
            "index_path": str(index_filepath),
            "scanned_dirs": [str(d) for d in scan_dirs],
            "new_files_count": len(new_files_abs) if action != "initial_index" else len(found_files),
            "new_files_samples": (new_files_abs if action != "initial_index" else found_files)[:10],
            "total_tracked_files": len(norm_processed),
            "duration_s": (datetime.utcnow() - t0).total_seconds()
        })

    except Exception as e:
        msg = str(e)
        hint = None
        if "timeout" in msg.lower():
            hint = (
                "Aumenta QDRANT_HTTP_TIMEOUT (p.ej. 120–300) y reduce HF_EMBED_BATCH (8–16). "
                "Verifica la disponibilidad de Qdrant y ancho de banda de disco."
            )
        if "sbert_model" in msg.lower() or "embed" in msg.lower():
            hint = (
                "Asegúrate de que el modelo E5 esté accesible: "
                "configura HF_EMBED_MODEL (por defecto intfloat/multilingual-e5-large) "
                "o LOCAL_EMBED_DIR (carpeta con config/tokenizer/weights)."
            )

        partial = {}
        try:
            user_id = token.get("user_id")
            sess = await get_session_from_redis(user_id)
            user_dir = sess.get("user_directory") if sess else None
            if user_dir:
                raw_idx = await redis_client.get(f"indexer:{user_dir}")
                if raw_idx:
                    stored = json.loads(raw_idx).get("user", {})
                    partial = {
                        "index_path": stored.get("index_filepath"),
                        "total_tracked_files": len(stored.get("files", []))
                    }
        except Exception:
            pass

        logging.error(f"Error en process_user_files: {msg}")
        return JSONResponse(status_code=200, content={
            "status": "error",
            "message": "Error inesperado al procesar archivos.",
            "detail": msg,
            "hint": hint,
            "partial": partial,
            "duration_s": (datetime.utcnow() - t0).total_seconds()
        })

@app.get('/process_department_files')
async def process_department_files(
    request: Request,
    token: dict = Depends(verify_token),
    db: Session = Depends(get_db)
):
    validate_csrf(request)

    try:
        user_id = token.get("user_id")
        request.state.user_id = user_id

        session_data = await get_session_from_redis(user_id)
        if not session_data:
            raise HTTPException(404, "Sesión no encontrada.")
        if session_data.get("role") != "Supervisor":
            raise HTTPException(403, "Solo supervisores pueden procesar.")

        departments = session_data.get("departments", [])
        indexers: Dict[str, DocumentIndexer] = {}
        had_files = False
        loop = asyncio.get_running_loop()

        def _warmup_models(idx: DocumentIndexer) -> None:
            if os.getenv("RAG_WARMUP_MODELS", "1").strip().lower() in ("0", "false", "no", "off"):
                return

            # E5
            try:
                if hasattr(idx, "_embed_texts") and callable(idx._embed_texts):
                    _ = idx._embed_texts(["warmup"], is_query=True, batch_size=1)
                elif getattr(idx, "sbert_model", None) is not None:
                    _ = idx.sbert_model.encode(
                        ["query: warmup"],
                        batch_size=1,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
            except Exception:
                pass

            # SPLADE
            if os.getenv("QDRANT_ENABLE_SPARSE", "1") == "1":
                deferred = os.getenv("QDRANT_SPLADE_DEFERRED", "0") == "1"
                if deferred:
                    try:
                        if hasattr(idx, "_ensure_sparse_worker") and callable(idx._ensure_sparse_worker):
                            idx._ensure_sparse_worker()
                    except Exception:
                        pass
                else:
                    try:
                        if hasattr(idx, "_get_splade_model") and callable(idx._get_splade_model):
                            idx._get_splade_model()
                    except Exception:
                        pass

            if os.getenv("RAG_WARMUP_RERANK", "0") == "1":
                try:
                    if hasattr(idx, "_get_reranker") and callable(idx._get_reranker):
                        _ = idx._get_reranker()
                except Exception:
                    pass
            if os.getenv("RAG_WARMUP_COLBERT", "0") == "1":
                try:
                    if hasattr(idx, "_get_colbert_encoder") and callable(idx._get_colbert_encoder):
                        _ = idx._get_colbert_encoder()
                except Exception:
                    pass

        failed_departments: List[Dict[str, str]] = []

        for dept in departments:
            dept_dir_rel = dept.get("department_directory")
            if not dept_dir_rel:
                continue

            dept_dir = PATH_BASE_DEPARTMENTS / dept_dir_rel / "uploaded_files"
            dept_dir.mkdir(parents=True, exist_ok=True)

            files_in_dir = sorted(list(dict.fromkeys([
                os.path.join(r, f)
                for r, _, files in os.walk(dept_dir)
                for f in files
                if os.path.splitext(f)[1].lower() in valid_extensions
            ])))
            if not files_in_dir:
                continue
            had_files = True

            dept_name = Path(dept_dir_rel).name

            idx_dir = PATH_BASE_DEPARTMENTS / dept_dir_rel / "qdrant_indices"
            idx_dir.mkdir(parents=True, exist_ok=True)
            default_path = idx_dir / f"{dept_name}_index"

            redis_key = f"indexer:{dept_dir_rel}"
            try:
                raw_idx = await redis_client.get(redis_key)
                if raw_idx:
                    try:
                        stored = json.loads(raw_idx) or {}
                    except Exception:
                        stored = {}
                    processed = set(stored.get("files", []) or [])
                    index_filepath = Path(stored.get("index_filepath") or default_path)
                else:
                    processed = set()
                    index_filepath = default_path

                norm_proc = {os.path.normcase(os.path.normpath(p)) for p in processed}
                norm_files = {os.path.normcase(os.path.normpath(p)) for p in files_in_dir}
                new_files = list(norm_files - norm_proc)

                if new_files:
                    new_files_set = set(new_files)
                    idx = await asyncio.to_thread(
                        make_indexer,
                        files=[p for p in files_in_dir if os.path.normcase(os.path.normpath(p)) in new_files_set],
                        index_filepath=str(index_filepath),
                        user_id=dept_name,
                        client_tag="",
                        use_cache=False,
                    )
                    norm_proc |= set(new_files)
                    _invalidate_indexer_cache(str(index_filepath), str(dept_name))
                else:
                    idx = await asyncio.to_thread(
                        make_indexer,
                        files=None,
                        index_filepath=str(index_filepath),
                        user_id=dept_name,
                        client_tag="",
                        use_cache=True,
                    )

                    try:
                        await loop.run_in_executor(None, _warmup_models, idx)
                    except Exception as e:
                        logging.warning(
                            "[process_department_files] warmup modelos falló para '%s': %s",
                            dept_name, e,
                        )

                indexers[dept_name] = idx

                rec = db.query(Department).filter(
                    Department.department_directory == dept_dir_rel
                ).first()
                if not rec:
                    rec = Department(name=dept_name, department_directory=dept_dir_rel)
                rec.faiss_index_path = str(index_filepath)
                db.add(rec)
                db.commit()

                await redis_client.set(
                    redis_key,
                    json.dumps({"index_filepath": str(index_filepath), "files": list(norm_proc)})
                )
            except Exception as dept_exc:
                logging.exception("[process_department_files] Error en '%s': %s", dept_dir_rel, dept_exc)
                failed_departments.append({"department": str(dept_dir_rel), "error": str(dept_exc)})
                try:
                    db.rollback()
                except Exception:
                    pass
                continue

        if not had_files:
            return Response(status_code=204)

        summary = {
            name: {"index_filepath": idx.index_filepath, "collection_name": idx.collection_name}
            for name, idx in indexers.items()
        }
        await redis_client.set(f"indexer:{user_id}:departments", json.dumps(summary))

        return JSONResponse(
            status_code=200,
            content={
                "message": "Procesamiento departamental completado.",
                "departments": summary,
                "failed_departments": failed_departments,
            }
        )

    except Exception as e:
        logging.error(f"Error en process_department_files: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": "Error inesperado al procesar departamentos."}
        )

@app.post('/search')
async def search_documents(
    request: Request,
    token: dict = Depends(verify_token)
):
    """
    Búsqueda SOTA + enriquecimiento para Excel:
    - Devuelve bloques principales, similares y meta (incluye sheet/row_kv… si aplica).
    - Si la intención es de agregación ('cuántos', 'número de', 'count'...), añade
      el resultado de `indexer._last_aggregation` en la respuesta.

    Convención de estados:
    - status="ok"           → búsqueda ejecutada con índice y con resultados.
    - status="no_results"   → índice existe pero no hay hits.
    - status="no_index"     → el usuario/departamento no tiene índice aún.
    """
    data = await request.json()
    query_text = (data.get("query") or "").strip()
    top_k_req = data.get("top_k")

    # ✅ NUEVO: filtros desde MCP/planner (p.ej. {"documento":"algo.pdf", "empresa":"AENA", ...})
    filters = data.get("filters") or {}
    if not isinstance(filters, dict):
        filters = {}

    requested_dept_dir = (
        data.get("department_directory")
        or data.get("department")
    )

    def _strip_scope_prefix(text: str) -> str:
        if not text:
            return text
        lower = text.lower()
        prefix = "contexto departamentos seleccionados:"
        if not lower.startswith(prefix):
            return text
        dot_idx = text.find(".")
        if dot_idx == -1:
            return text
        return text[dot_idx + 1:].lstrip()

    def _match_dept_in_text(text: str, departments_list: list):
        if not text or not departments_list:
            return None, None

        lower = text.lower()
        for d in departments_list:
            dep_dir = str(d.get("department_directory") or "")
            if not dep_dir:
                continue
            if dep_dir.lower() in lower:
                return d, dep_dir
        return None, None

    if not query_text:
        logging.warning("[/search] 400: query vacío. payload=%s", data)
        return JSONResponse(
            status_code=400,
            content={"error": "No se proporcionó un query."}
        )

    user_id = token.get("user_id")
    request.state.user_id = user_id

    session_data = await get_session_from_redis(user_id)
    if not session_data:
        logging.warning("[/search] 404: sesión no encontrada en Redis. user_id=%s", user_id)
        return JSONResponse(
            status_code=404,
            content={"error": "Sesión no encontrada en Redis."}
        )

    user_directory = session_data.get("user_directory")
    departments = session_data.get("departments", [])

    cleaned_query = query_text
    department = None
    used_department_dir: Optional[str] = None

    if not user_directory:
        logging.warning(
            "[/search] user_directory no configurado en sesión. session_data=%s",
            session_data
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "no_index",
                "message": "Directorio de usuario no configurado o sin índice asociado.",
                "results": [],
                "aggregation": None,
                "used_department": None,
                "cleaned_query": cleaned_query,
                "filters_applied": filters or None,
            }
        )

    idx_user_raw = await redis_client.get(f"indexer:{user_directory}")
    idx_user = json.loads(idx_user_raw) if idx_user_raw else {}

    dept_idx_map: Dict[str, Dict[str, Any]] = {}
    for d in departments:
        dep_dir = d["department_directory"]
        raw = await redis_client.get(f"indexer:{dep_dir}")
        dept_idx_map[dep_dir] = json.loads(raw) if raw else {}

    if requested_dept_dir:
        department = next(
            (d for d in departments if d.get("department_directory") == requested_dept_dir),
            None
        )
        if department is None:
            logging.warning(
                "[/search] Usuario sin acceso al departamento solicitado. user_id=%s dept_dir=%s",
                user_id,
                requested_dept_dir,
            )
            return JSONResponse(
                status_code=403,
                content={"error": "No tiene acceso al departamento solicitado."}
            )
        used_department_dir = requested_dept_dir
    else:
        dept_from_text, dept_dir = _match_dept_in_text(query_text, departments)
        if dept_from_text is not None:
            department = dept_from_text
            used_department_dir = dept_dir
        else:
            department, cleaned_query = extract_department_from_query(query_text, departments)
            if department:
                used_department_dir = department["department_directory"]

    cleaned_query = _strip_scope_prefix(cleaned_query)

    try:
        if department:
            info = dept_idx_map.get(used_department_dir) or {}
            index_path = info.get("index_filepath") or department.get("faiss_index_path")
            if not index_path:
                logging.warning(
                    "[/search] Índice departamental no configurado. dept=%s info_redis=%s",
                    used_department_dir, info
                )
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "no_index",
                        "message": (
                            f"Índice departamental no configurado para '{used_department_dir}'. "
                            "Ejecuta /process_department_files primero."
                        ),
                        "results": [],
                        "aggregation": None,
                        "used_department": used_department_dir,
                        "cleaned_query": cleaned_query,
                        "filters_applied": filters or None,
                    }
                )

            index_path = str(index_path)
            base = os.path.splitext(index_path)[0]
            dn = f"{base}_docnames.pkl"
            dc = f"{base}_documents.pkl"
            if not (os.path.exists(dn) and os.path.exists(dc)):
                logging.warning(
                    "[/search] Índice departamental sin pickles. index_path=%s dn=%s dc=%s",
                    index_path, dn, dc
                )
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "no_index",
                        "message": (
                            f"El índice departamental para '{used_department_dir}' todavía no contiene documentos. "
                            "Sube e indexa archivos antes de buscar."
                        ),
                        "results": [],
                        "aggregation": None,
                        "used_department": used_department_dir,
                        "cleaned_query": cleaned_query,
                        "filters_applied": filters or None,
                    }
                )

            dept_name = os.path.basename(used_department_dir)
            document_indexer = await asyncio.to_thread(
                make_indexer,
                files=None,
                index_filepath=index_path,
                user_id=dept_name,
                client_tag=""
            )
        else:
            info = idx_user.get("user", {})
            index_path = info.get("index_filepath")
            if not index_path:
                logging.info(
                    "[/search] Índice de usuario no encontrado. user_directory=%s info=%s",
                    user_directory, info
                )
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "no_index",
                        "message": (
                            "Índice de usuario no encontrado. "
                            "Ejecuta primero /process_user_files para indexar tus documentos."
                        ),
                        "results": [],
                        "aggregation": None,
                        "used_department": None,
                        "cleaned_query": cleaned_query,
                        "filters_applied": filters or None,
                    }
                )

            index_path = str(index_path)
            base = os.path.splitext(index_path)[0]
            dn = f"{base}_docnames.pkl"
            dc = f"{base}_documents.pkl"
            if not (os.path.exists(dn) and os.path.exists(dc)):
                logging.warning(
                    "[/search] Índice de usuario sin pickles. index_path=%s dn=%s dc=%s",
                    index_path, dn, dc
                )
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "no_index",
                        "message": (
                            "El índice de usuario existe pero todavía no contiene documentos. "
                            "Sube e indexa archivos antes de buscar."
                        ),
                        "results": [],
                        "aggregation": None,
                        "used_department": None,
                        "cleaned_query": cleaned_query,
                        "filters_applied": filters or None,
                    }
                )

            document_indexer = await asyncio.to_thread(
                make_indexer,
                files=None,
                index_filepath=index_path,
                user_id=user_id,
                client_tag=""
            )

    except Exception as e:
        logging.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Error iniciando indexador: {e}"}
        )

    matched_tags = await document_indexer.get_matched_tags(cleaned_query)

    try:
        try:
            top_k_main = int(top_k_req) if top_k_req is not None else int(os.getenv("SEARCH_TOPK", "5"))
        except Exception:
            top_k_main = int(os.getenv("SEARCH_TOPK", "5"))

        # ✅ PASAMOS filters a la búsqueda (sin romper si el indexer no lo soporta)
        try:
            sota = await asyncio.to_thread(
                _search_sync,
                document_indexer,
                cleaned_query,
                top_k_main,
                matched_tags,
                filters,
            )
        except Exception as e:
            logging.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": f"Error en la búsqueda: {e}"}
            )

        if not sota:
            logging.info(
                "[/search] Sin resultados. user_id=%s query='%s' used_department=%s",
                user_id, cleaned_query, used_department_dir
            )
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_results",
                    "message": "No se encontraron resultados para la consulta.",
                    "results": [],
                    "aggregation": None,
                    "used_department": used_department_dir,
                    "cleaned_query": cleaned_query,
                    "filters_applied": filters or None,
                }
            )

        def _find_idx(idxr, text) -> int:
            try:
                return idxr.documents.index(text)
            except ValueError:
                return -1

        out = []
        for (text, doc_name, rrf_score, rerank_score) in sota:
            similars = []
            meta_main: Dict[str, Any] = {}
            ref_idx = _find_idx(document_indexer, text)
            if ref_idx >= 0:
                try:
                    meta_main = (
                        document_indexer._metas[ref_idx]
                        if ref_idx < len(document_indexer._metas)
                        else {}
                    )
                except Exception:
                    meta_main = {}

                try:
                    sims = await asyncio.to_thread(
                        document_indexer.retrieve_similar_blocks,
                        doc_name=doc_name,
                        relevant_idx=ref_idx,
                        documents=document_indexer.documents,
                        top_k=3,
                        global_search=True
                    )
                    for s in sims or []:
                        st = s.get("text") or s.get("texto") or ""
                        if st:
                            similars.append({
                                "text": st,
                                "score": float(s.get("score", 0.0)),
                                "meta": s.get("meta", {})
                            })
                except Exception:
                    pass

            out.append({
                "doc_name": doc_name,
                "text": text,
                "rrf_score": float(rrf_score),
                "rerank_score": float(rerank_score),
                "meta": meta_main,
                "similar_blocks": similars,
            })

        aggregation = getattr(document_indexer, "_last_aggregation", None)

        content = {
            "status": "ok",
            "results": out,
            "aggregation": aggregation,
            "used_department": used_department_dir,
            "cleaned_query": cleaned_query,
            "filters_applied": filters or None,  # ✅ nuevo, no rompe consumidores
        }
        logging.info(f"RESPUESTA /SEARCH RAG: {content}")
        return JSONResponse(status_code=200, content=content)

    except Exception as e:
        logging.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Error en la búsqueda: {e}"}
        )


@app.post("/chatdoc/ingest")
async def chatdoc_ingest(
    request: Request,
    token: dict = Depends(verify_token),
):
    """
    Construye (o reemplaza) un índice efímero para hablar con UN documento.
    """
    endpoint_lbl = "/chatdoc/ingest"
    t0 = time.time()

    # -------------------------
    # 0) Semáforos globales (lazy-init, no rompe si ya existen)
    # -------------------------
    # Concurrencia "pesada": build + embeddings warm + posibles llamadas GPU
    heavy_sem = globals().get("CHATDOC_HEAVY_SEMAPHORE")
    if heavy_sem is None:
        try:
            cap = int(os.getenv("CHATDOC_HEAVY_CONCURRENCY", "2"))
        except Exception:
            cap = 2
        cap = max(1, cap)
        heavy_sem = asyncio.Semaphore(cap)
        globals()["CHATDOC_HEAVY_SEMAPHORE"] = heavy_sem

    # Mantén tu semáforo de ingesta si existe (compatibilidad)
    ingest_sem = globals().get("CHATDOC_INGEST_SEMAPHORE")
    if ingest_sem is None:
        # fallback razonable si no estuviera definido en tu módulo
        try:
            cap = int(os.getenv("CHATDOC_INGEST_CONCURRENCY", str(CHATDOC_INGEST_CONCURRENCY)))
        except Exception:
            cap = 1
        cap = max(1, cap)
        ingest_sem = asyncio.Semaphore(cap)
        globals()["CHATDOC_INGEST_SEMAPHORE"] = ingest_sem

    # Timeout defensivo para build (base + adaptativo por tamaño/páginas)
    try:
        build_timeout_base_s = float(os.getenv("CHATDOC_INGEST_BUILD_TIMEOUT_S", "300"))
    except Exception:
        build_timeout_base_s = 300.0
    try:
        build_timeout_max_s = float(os.getenv("CHATDOC_INGEST_BUILD_TIMEOUT_MAX_S", "1800"))
    except Exception:
        build_timeout_max_s = 1800.0

    # -------------------------
    # 1) Parse JSON
    # -------------------------
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Body JSON no es objeto.")
    except Exception as e:
        logger.warning("[/chatdoc/ingest] JSON inválido: %s", e)
        _chatdoc_emit_error(endpoint_lbl, kind="json_parse", t0=t0, status_sem="error")
        return JSONResponse(
            status_code=400,
            content={"error": "Body JSON inválido en /chatdoc/ingest.", "detail": str(e)},
        )

    trace = _parse_bool(data.get("trace"), False)
    safe_keys = list(data.keys())
    logger.info("[/chatdoc/ingest] payload keys=%s", safe_keys)

    # -------------------------
    # 2) Campos principales
    # -------------------------
    text = (data.get("text") or "").strip()
    content_b64 = data.get("content_base64")
    document_id = (data.get("document_id") or "").strip()
    raw_meta = _safe_dict(data.get("metadata"))

    filename = _normalize_filename(
        data.get("filename")
        or raw_meta.get("filename")
        or raw_meta.get("source_file_name")
        or "document"
    )

    mime_type = (data.get("mime_type") or raw_meta.get("mime_type") or None)
    if not mime_type:
        mime_type = _infer_mime_from_filename(filename)

    hinted_page_count = None
    try:
        hinted_page_count = int(data.get("page_count") or raw_meta.get("page_count") or 0)
        if hinted_page_count <= 0:
            hinted_page_count = None
    except Exception:
        hinted_page_count = None

    # -------------------------
    # 3) Token usuario
    # -------------------------
    user_id = token.get("user_id")
    if user_id is None:
        logger.warning(
            "[/chatdoc/ingest] Token sin user_id. token_keys=%s",
            list(token.keys())
        )
        _chatdoc_emit_error(endpoint_lbl, kind="auth", t0=t0, status_sem="error")
        return JSONResponse(status_code=403, content={"error": "Token sin user_id en /chatdoc/ingest."})
    request.state.user_id = user_id

    if not document_id:
        document_id = str(uuid.uuid4())

    mode = "bytes" if content_b64 else ("text" if text else "none")

    if mode == "none":
        _chatdoc_emit_error(endpoint_lbl, kind="bad_request", t0=t0, status_sem="error")
        return JSONResponse(
            status_code=400,
            content={"error": "Debe proporcionar 'text' o 'content_base64' en /chatdoc/ingest."},
        )

    logger.info(
        "[/chatdoc/ingest] start user_id=%s doc_id=%s filename=%s mime=%s mode=%s hinted_pages=%s",
        user_id, document_id, filename, mime_type, mode, hinted_page_count,
    )

    # -------------------------
    # 4) Enriquecer metadata con sesión
    # -------------------------
    t_session = time.time()
    try:
        session_data = await get_session_from_redis(user_id)
    except Exception as e:
        session_data = None
        logger.debug("[/chatdoc/ingest] No se pudo leer sesión Redis user_id=%s: %s", user_id, e)

    metadata = dict(raw_meta or {})
    metadata.setdefault("source_file_name", filename)
    if mime_type:
        metadata.setdefault("mime_type", mime_type)
    if hinted_page_count:
        metadata.setdefault("page_count_hint", hinted_page_count)

    if session_data and isinstance(session_data, dict):
        user_directory = session_data.get("user_directory")
        if user_directory:
            metadata.setdefault("user_directory", user_directory)

    logger.debug(
        "[/chatdoc/ingest] session_enrich done in %.3fs meta_keys=%s",
        time.time() - t_session,
        list(metadata.keys())[:50],
    )

    # -------------------------
    # 5) Guardas defensivas tamaño
    # -------------------------
    max_bytes = _parse_int(
        os.getenv("CHATDOC_INGEST_MAX_BYTES", str(50 * 1024 * 1024)),
        50 * 1024 * 1024
    )

    raw_bytes: Optional[bytes] = None
    file_hash: str = ""

    if content_b64:
        # Pre-check aproximado para evitar decodificar monstruos
        try:
            b64_len = len(str(content_b64))
            approx_bytes = (b64_len * 3) // 4  # aproximación
            if approx_bytes > int(max_bytes * 1.10):
                logger.warning(
                    "[/chatdoc/ingest] FILE_TOO_LARGE(approx) doc_id=%s approx=%d max=%d",
                    document_id, approx_bytes, max_bytes
                )
                _chatdoc_emit_error(endpoint_lbl, kind="too_large", t0=t0, status_sem="error")
                return JSONResponse(
                    status_code=413,
                    content={
                        "status": "error",
                        "error": "Documento demasiado grande para ingesta efímera (precheck).",
                        "document_id": document_id,
                        "approx_bytes": approx_bytes,
                        "max_bytes": max_bytes,
                        "file_name": filename,
                        "mime_type": mime_type,
                    },
                )
        except Exception:
            pass

        try:
            # validate=True rechaza caracteres inválidos
            raw_bytes = base64.b64decode(content_b64, validate=True)
        except TypeError:
            # compat: algunos inputs pueden no ser str/bytes
            try:
                raw_bytes = base64.b64decode(str(content_b64).encode("utf-8"), validate=True)
            except Exception as e:
                logger.warning("[/chatdoc/ingest] content_base64 inválido doc_id=%s err=%s", document_id, e)
                _chatdoc_emit_error(endpoint_lbl, kind="bad_request", t0=t0, status_sem="error")
                return JSONResponse(
                    status_code=400,
                    content={"error": "El campo 'content_base64' no es válido.", "detail": str(e)},
                )
        except Exception as e:
            logger.warning("[/chatdoc/ingest] content_base64 inválido doc_id=%s err=%s", document_id, e)
            _chatdoc_emit_error(endpoint_lbl, kind="bad_request", t0=t0, status_sem="error")
            return JSONResponse(
                status_code=400,
                content={"error": "El campo 'content_base64' no es válido.", "detail": str(e)},
            )

        if not raw_bytes:
            logger.warning("[/chatdoc/ingest] content_base64 decodificado vacío doc_id=%s", document_id)
            _chatdoc_emit_error(endpoint_lbl, kind="bad_request", t0=t0, status_sem="error")
            return JSONResponse(status_code=400, content={"error": "content_base64 decodificado está vacío."})

        if len(raw_bytes) > max_bytes:
            logger.warning(
                "[/chatdoc/ingest] FILE_TOO_LARGE doc_id=%s bytes=%d max=%d",
                document_id, len(raw_bytes), max_bytes
            )
            _chatdoc_emit_error(endpoint_lbl, kind="too_large", t0=t0, status_sem="error")
            return JSONResponse(
                status_code=413,
                content={
                    "status": "error",
                    "error": "Documento demasiado grande para ingesta efímera.",
                    "document_id": document_id,
                    "bytes": len(raw_bytes),
                    "max_bytes": max_bytes,
                    "file_name": filename,
                    "mime_type": mime_type,
                },
            )

        file_hash = _sha256_bytes(raw_bytes)
        if file_hash:
            metadata.setdefault("file_sha256", file_hash)

        logger.info("[/chatdoc/ingest] decoded bytes=%d doc_id=%s", len(raw_bytes), document_id)

    raw_mb = float(len(raw_bytes or b"")) / (1024.0 * 1024.0)
    hinted_pages_for_timeout = int(hinted_page_count or metadata.get("page_count") or metadata.get("page_count_hint") or 0)
    adaptive_timeout_s = build_timeout_base_s + (raw_mb * 8.0) + (hinted_pages_for_timeout * 2.5)
    build_timeout_s = max(90.0, min(build_timeout_max_s, adaptive_timeout_s))
    logger.info(
        "[/chatdoc/ingest] timeout_policy base=%.1fs adaptive=%.1fs final=%.1fs size_mb=%.2f hinted_pages=%s",
        build_timeout_base_s,
        adaptive_timeout_s,
        build_timeout_s,
        raw_mb,
        hinted_pages_for_timeout or None,
    )

    # -------------------------
    # 6) Construcción índice (FUERA del event loop + limitado)
    # -------------------------
    t_build = time.time()

    def _build_index_sync() -> DocumentChatIndex:
        warm = _parse_bool(os.getenv("CHATDOC_WARM_EMBEDDINGS", "1"), True)

        if raw_bytes:
            idx = DocumentChatIndex.from_bytes(
                document_id=document_id,
                raw_bytes=raw_bytes,
                filename=filename,
                mime_type=mime_type,
                metadata=metadata,
                language=CHATDOC_DEFAULT_LANGUAGE,
            )
        elif text:
            idx = DocumentChatIndex(
                document_id=document_id,
                full_text=text,
                metadata=metadata,
                chunk_chars=CHATDOC_DEFAULT_CHUNK_CHARS,
                chunk_overlap=CHATDOC_DEFAULT_CHUNK_OVERLAP,
                language=CHATDOC_DEFAULT_LANGUAGE,
            )
            try:
                idx.page_count = int(data.get("page_count") or metadata.get("page_count") or 0)
            except Exception:
                idx.page_count = 0
            idx.metadata.setdefault("extraction_engine", "plain_text")
        else:
            raise ValueError("Debe proporcionar 'text' o 'content_base64' en /chatdoc/ingest.")

        if warm:
            try:
                idx._ensure_embeddings()  # best-effort
            except Exception as e:
                logger.warning("[/chatdoc/ingest] warm embeddings failed doc_id=%s err=%s", document_id, e)

        return idx

    try:
        # ingest_sem mantiene tu “rate limit” de ingesta
        # heavy_sem coordina con query/summary para que no haya picos cruzados
        async with ingest_sem:
            async with heavy_sem:
                index = await asyncio.wait_for(
                    asyncio.to_thread(_build_index_sync),
                    timeout=build_timeout_s,
                )
    except asyncio.TimeoutError:
        logger.error("[/chatdoc/ingest] TIMEOUT construyendo índice doc_id=%s timeout=%.1fs", document_id, build_timeout_s)
        _chatdoc_emit_error(endpoint_lbl, kind="timeout", t0=t0, status_sem="error")
        return JSONResponse(
            status_code=504,
            content={
                "error": "Timeout construyendo el índice del documento.",
                "document_id": document_id,
                "timeout_s": build_timeout_s,
                "retryable": True,
            },
        )
    except Exception as e:
        logger.error("[/chatdoc/ingest] Error construyendo DocumentChatIndex doc_id=%s: %s", document_id, e)
        logger.error(traceback.format_exc())
        _chatdoc_emit_error(endpoint_lbl, kind="internal", t0=t0, status_sem="error")
        return JSONResponse(status_code=500, content={"error": f"No se pudo construir el índice del documento: {e}"})

    build_s = time.time() - t_build

    # -------------------------
    # 7) Diagnóstico post-build
    # -------------------------
    chunk_count = len(getattr(index, "_chunks", []) or [])
    char_length = len(getattr(index, "full_text", "") or "")
    page_count = int(getattr(index, "page_count", 0) or 0)

    index.metadata.setdefault("page_count", page_count or 0)
    index.metadata.setdefault("chunk_count", chunk_count)
    index.metadata.setdefault("full_text_chars", char_length)
    index.metadata.setdefault("ingested_at_unix", int(time.time()))
    index.metadata.setdefault("language", CHATDOC_DEFAULT_LANGUAGE)
    index.metadata.setdefault("extraction_engine", "rag_loader")

    logger.info(
        "[/chatdoc/ingest] built index doc_id=%s chunks=%d chars=%d page_count=%d filename=%s mime=%s build_s=%.3f",
        document_id, chunk_count, char_length, page_count, filename, mime_type, build_s,
    )

    if chunk_count == 0 or char_length == 0:
        logger.warning(
            "[/chatdoc/ingest] EMPTY_INDEX doc_id=%s user_id=%s bytes=%s hinted_pages=%s meta_keys=%s",
            document_id, user_id,
            len(raw_bytes) if raw_bytes else None,
            hinted_page_count,
            list(index.metadata.keys())[:50],
        )
        _chatdoc_emit_error(endpoint_lbl, kind="unprocessable", t0=t0, status_sem="error")
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "error": (
                    "No se pudo extraer texto del documento para indexación. "
                    "Posibles causas: PDF escaneado sin OCR, motor de extracción no disponible "
                    "o dependencias faltantes en el contenedor."
                ),
                "document_id": document_id,
                "chunk_count": chunk_count,
                "char_length": char_length,
                "page_count": page_count,
                "file_name": filename,
                "mime_type": mime_type,
                "meta_keys": list(index.metadata.keys()),
            },
        )

    # -------------------------
    # 8) Guardado Redis
    # -------------------------
    if redis_client is None:
        logger.error("[/chatdoc/ingest] Redis no inicializado.")
        _chatdoc_emit_error(endpoint_lbl, kind="internal", t0=t0, status_sem="error")
        return JSONResponse(status_code=500, content={"error": "Redis no está inicializado en /chatdoc/ingest."})

    key = _chatdoc_key(user_id, document_id)
    payload = index.to_dict()

    t_redis = time.time()
    try:
        await redis_client.set(key, json.dumps(payload, ensure_ascii=False), ex=CHATDOC_REDIS_TTL)
        try:
            _chatdoc_cache_put(user_id, document_id, index, CHATDOC_REDIS_TTL)
        except Exception:
            pass

    except Exception as e:
        logger.error("[/chatdoc/ingest] Error guardando índice en Redis key=%s: %s", key, e)
        logger.error(traceback.format_exc())
        _chatdoc_emit_error(endpoint_lbl, kind="internal", t0=t0, status_sem="error")
        return JSONResponse(status_code=500, content={"error": "No se pudo guardar el índice del documento en Redis."})

    redis_s = time.time() - t_redis

    logger.info(
        "[/chatdoc/ingest] stored OK doc_id=%s key=%s ttl=%s redis_s=%.3f total_s=%.3f",
        document_id, key, CHATDOC_REDIS_TTL, redis_s, time.time() - t0,
    )

    _chatdoc_emit_ingest_ok(
        t0=t0,
        mode=mode,
        build_s=build_s,
        redis_s=redis_s,
        raw_bytes=raw_bytes,
        text=text,
        chunk_count=chunk_count,
        char_length=char_length,
        page_count=page_count,
        hinted_page_count=hinted_page_count,
    )

    response: Dict[str, Any] = {
        "status": "ok",
        "document_id": document_id,
        "chunk_count": chunk_count,
        "char_length": char_length,
        "metadata": index.metadata,
        "ttl_seconds": CHATDOC_REDIS_TTL,
        "page_count": page_count,
        "file_name": filename,
        "mime_type": mime_type,
        "timings": {
            "build_s": round(build_s, 4),
            "redis_s": round(redis_s, 4),
            "total_s": round(time.time() - t0, 4),
        },
    }

    if trace:
        response["trace"] = {
            "mode": mode,
            "max_bytes": max_bytes,
            "bytes_received": len(raw_bytes) if raw_bytes else None,
            "file_sha256": file_hash or index.metadata.get("file_sha256"),
            "index_stats": {
                "chunks": chunk_count,
                "full_text_chars": char_length,
                "page_count": page_count,
                "extraction_engine": index.metadata.get("extraction_engine"),
            },
            "redis_key": key,
        }

    return JSONResponse(status_code=200, content=response)

@app.post("/chatdoc/query")
async def chatdoc_query(
    request: Request,
    token: dict = Depends(verify_token),
):
    endpoint_lbl = "/chatdoc/query"

    # -------------------------
    # 0) Semáforos globales (lazy-init)
    # -------------------------
    heavy_sem = globals().get("CHATDOC_HEAVY_SEMAPHORE")
    if heavy_sem is None:
        try:
            cap = int(os.getenv("CHATDOC_HEAVY_CONCURRENCY", "2"))
        except Exception:
            cap = 2
        cap = max(1, cap)
        heavy_sem = asyncio.Semaphore(cap)
        globals()["CHATDOC_HEAVY_SEMAPHORE"] = heavy_sem

    cpu_sem = globals().get("CHATDOC_CPU_SEMAPHORE")
    if cpu_sem is None:
        try:
            cap = int(os.getenv("CHATDOC_CPU_CONCURRENCY", "8"))
        except Exception:
            cap = 8
        cap = max(1, cap)
        cpu_sem = asyncio.Semaphore(cap)
        globals()["CHATDOC_CPU_SEMAPHORE"] = cpu_sem

    try:
        query_timeout_s = float(os.getenv("CHATDOC_QUERY_TIMEOUT_S", "60"))
    except Exception:
        query_timeout_s = 60.0

    try:
        rebuild_timeout_s = float(os.getenv("CHATDOC_INDEX_REBUILD_TIMEOUT_S", "30"))
    except Exception:
        rebuild_timeout_s = 30.0

    # -------------------------
    # 1) Parse JSON
    # -------------------------
    try:
        data = await request.json()
    except Exception:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="json_parse").inc())
        return JSONResponse(status_code=400, content={"error": "Body JSON inválido en /chatdoc/query."})

    return_format = (data.get("return_format") or "json").strip().lower()
    if return_format not in ("json", "text"):
        return_format = "json"

    document_id = (data.get("document_id") or "").strip()
    query_text = (data.get("query") or "").strip()

    if not document_id:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="bad_request").inc())
        return JSONResponse(status_code=400, content={"error": "Falta 'document_id' en /chatdoc/query."})

    if not query_text:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="bad_request").inc())
        return JSONResponse(status_code=400, content={"error": "Falta 'query' en /chatdoc/query."})

    # Recorte defensivo de query (evita embeddings gigantes)
    try:
        max_q_chars = int(os.getenv("CHATDOC_QUERY_MAX_CHARS", "2000"))
    except Exception:
        max_q_chars = 2000
    if max_q_chars > 0 and len(query_text) > max_q_chars:
        logger.info("[/chatdoc/query] query recortada %d -> %d chars", len(query_text), max_q_chars)
        query_text = query_text[:max_q_chars]

    user_id = token.get("user_id")
    if user_id is None:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="auth").inc())
        return JSONResponse(status_code=403, content={"error": "Token sin user_id en /chatdoc/query."})
    request.state.user_id = user_id

    # top_k con límites defensivos
    try:
        max_top_k = int(os.getenv("CHATDOC_QUERY_MAX_TOPK", "32"))
    except Exception:
        max_top_k = 32

    try:
        top_k_raw = data.get("top_k")
        if top_k_raw is None:
            top_k = int(os.getenv("CHATDOC_QUERY_DEFAULT_TOPK", "5"))
        else:
            top_k = int(top_k_raw)
    except Exception:
        top_k = 5
    top_k = max(1, min(top_k, max_top_k))

    # min_score
    try:
        min_score = float(data.get("min_score", 0.0))
    except Exception:
        min_score = 0.0

    # ventana de contexto
    try:
        default_window = int(os.getenv("CHATDOC_SEARCH_WINDOW", "1"))
    except Exception:
        default_window = 1
    try:
        max_window = int(os.getenv("CHATDOC_SEARCH_MAX_WINDOW", "5"))
    except Exception:
        max_window = 5
    try:
        window_raw = data.get("window")
        if window_raw is None:
            window = default_window
        else:
            window = int(window_raw)
    except Exception:
        window = default_window
    window = max(0, min(window, max_window))

    if redis_client is None:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="internal").inc())
        return JSONResponse(status_code=500, content={"error": "Redis no está inicializado en /chatdoc/query."})

    key = _chatdoc_key(user_id, document_id)

    # 1) Cache in-process
    index = None
    try:
        index = _chatdoc_cache_get(user_id, document_id)
    except Exception:
        index = None

    if index is not None:
        _safe_inc(lambda: CHATDOC_CACHE_TOTAL.labels(endpoint=endpoint_lbl, result="hit").inc())
        try:
            await redis_client.expire(key, CHATDOC_REDIS_TTL)
        except Exception:
            pass
        try:
            _chatdoc_cache_touch(user_id, document_id, CHATDOC_REDIS_TTL)
        except Exception:
            pass
    else:
        # 2) Redis -> reconstrucción (en thread + cpu_sem)
        raw = await redis_client.get(key)
        if not raw:
            _safe_inc(lambda: CHATDOC_CACHE_TOTAL.labels(endpoint=endpoint_lbl, result="miss").inc())
            _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="no_index").inc())
            _safe_observe(lambda: CHATDOC_QUERY_RESULTS_COUNT.observe(0))
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_index",
                    "message": "No hay índice efímero para este documento. Llama primero a /chatdoc/ingest.",
                    "document_id": document_id,
                    "query": query_text,
                    "top_k": top_k,
                    "min_score": min_score,
                    "window": window,
                    "results": [],
                },
            )

        _safe_inc(lambda: CHATDOC_CACHE_TOTAL.labels(endpoint=endpoint_lbl, result="hit").inc())

        def _rebuild_sync() -> "DocumentChatIndex":
            idx_data = json.loads(raw)
            return DocumentChatIndex.from_dict(idx_data)

        try:
            async with cpu_sem:
                index = await asyncio.wait_for(
                    asyncio.to_thread(_rebuild_sync),
                    timeout=rebuild_timeout_s,
                )
        except asyncio.TimeoutError:
            _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
            _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="timeout").inc())
            return JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "error": "Timeout reconstruyendo índice desde Redis.",
                    "document_id": document_id,
                    "query": query_text,
                    "top_k": top_k,
                    "min_score": min_score,
                    "window": window,
                },
            )
        except Exception as e:
            logging.error("[/chatdoc/query] Error reconstruyendo índice desde Redis: %s", e)
            logging.error(traceback.format_exc())
            _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
            _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="internal").inc())
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": f"No se pudo reconstruir el índice del documento: {e}",
                    "document_id": document_id,
                    "query": query_text,
                    "top_k": top_k,
                    "min_score": min_score,
                    "window": window,
                },
            )

        try:
            await redis_client.expire(key, CHATDOC_REDIS_TTL)
        except Exception:
            pass

        try:
            _chatdoc_cache_put(user_id, document_id, index, CHATDOC_REDIS_TTL)
        except Exception:
            pass

    # 3) Búsqueda (en thread + heavy_sem)
    try:
        async with heavy_sem:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    index.search,
                    query=query_text,
                    top_k=top_k,
                    min_score=min_score,
                    window=window,
                ),
                timeout=query_timeout_s,
            )
    except asyncio.TimeoutError:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="timeout").inc())
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "error": "Timeout ejecutando búsqueda en el documento.",
                "document_id": document_id,
                "query": query_text,
                "top_k": top_k,
                "min_score": min_score,
                "window": window,
            },
        )
    except ValueError as e:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="bad_request").inc())
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": str(e),
                "document_id": document_id,
                "query": query_text,
                "top_k": top_k,
                "min_score": min_score,
                "window": window,
            },
        )
    except Exception as e:
        logging.error("[/chatdoc/query] Error ejecutando búsqueda en DocumentChatIndex: %s", e)
        logging.error(traceback.format_exc())
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="internal").inc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"Error interno en la búsqueda del documento: {e}",
                "document_id": document_id,
                "query": query_text,
                "top_k": top_k,
                "min_score": min_score,
                "window": window,
            },
        )

    results_count = len(results) if isinstance(results, list) else 0
    _safe_observe(lambda: CHATDOC_QUERY_RESULTS_COUNT.observe(results_count))

    status_label = "ok" if results_count > 0 else "no_results"
    _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status=status_label).inc())

    payload = {
        "status": status_label,
        "document_id": document_id,
        "query": query_text,
        "top_k": top_k,
        "min_score": min_score,
        "window": window,
        "results": results,
    }

    # Soporte opcional: return_format="text" (sin romper json default)
    if return_format == "text":
        try:
            pieces = []
            for r in (results or []):
                if isinstance(r, dict):
                    t = (r.get("text") or r.get("chunk_text") or r.get("content") or "").strip()
                    if t:
                        pieces.append(t)
            payload["text"] = "\n\n---\n\n".join(pieces)
        except Exception:
            payload["text"] = ""

    return JSONResponse(status_code=200, content=payload)


@app.post("/chatdoc/summary")
async def chatdoc_summary(
    request: Request,
    token: dict = Depends(verify_token),
):
    endpoint_lbl = "/chatdoc/summary"

    # -------------------------
    # 0) Semáforos (lazy-init)
    # -------------------------
    cpu_sem = globals().get("CHATDOC_CPU_SEMAPHORE")
    if cpu_sem is None:
        try:
            cap = int(os.getenv("CHATDOC_CPU_CONCURRENCY", "8"))
        except Exception:
            cap = 8
        cap = max(1, cap)
        cpu_sem = asyncio.Semaphore(cap)
        globals()["CHATDOC_CPU_SEMAPHORE"] = cpu_sem

    try:
        summary_timeout_s = float(os.getenv("CHATDOC_SUMMARY_TIMEOUT_S", "60"))
    except Exception:
        summary_timeout_s = 60.0

    try:
        rebuild_timeout_s = float(os.getenv("CHATDOC_INDEX_REBUILD_TIMEOUT_S", "30"))
    except Exception:
        rebuild_timeout_s = 30.0

    # -------------------------
    # 1) Parse JSON
    # -------------------------
    try:
        data = await request.json()
    except Exception:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="json_parse").inc())
        return JSONResponse(status_code=400, content={"error": "Body JSON inválido en /chatdoc/summary."})

    return_format = (data.get("return_format") or "json").strip().lower()
    if return_format not in ("json", "text"):
        return_format = "json"

    document_id = (data.get("document_id") or "").strip()
    trace = _parse_bool(data.get("trace"), False)

    if not document_id:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="bad_request").inc())
        return JSONResponse(status_code=400, content={"error": "Falta 'document_id' en /chatdoc/summary."})

    user_id = token.get("user_id")
    if user_id is None:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="auth").inc())
        return JSONResponse(status_code=403, content={"error": "Token sin user_id en /chatdoc/summary."})
    request.state.user_id = user_id

    detail_level_raw = data.get("detail_level")
    if isinstance(detail_level_raw, str):
        dl = detail_level_raw.strip().lower()
        detail_level = dl if dl in ("low", "medium", "high") else None
    else:
        detail_level = None

    summary_profile = data.get("summary_profile")
    if isinstance(summary_profile, str):
        summary_profile = summary_profile.strip().lower()
    else:
        summary_profile = None

    strategy = (data.get("strategy") or "hybrid").lower().strip()
    if strategy not in ("uniform", "tfidf", "hybrid"):
        strategy = "hybrid"

    dl_label = detail_level or "none"

    default_medium = _parse_int(os.getenv("CHATDOC_SUMMARY_FRAGMENTS_MEDIUM", "12"), 12)
    default_low = _parse_int(
        os.getenv("CHATDOC_SUMMARY_FRAGMENTS_LOW", str(max(4, default_medium // 2))),
        max(4, default_medium // 2),
    )
    default_high = _parse_int(
        os.getenv("CHATDOC_SUMMARY_FRAGMENTS_HIGH", str(max(default_medium * 2, default_medium + 4))),
        max(default_medium * 2, default_medium + 4),
    )
    max_cap = _parse_int(os.getenv("CHATDOC_SUMMARY_FRAGMENTS_MAX", "48"), 48)

    explicit_max_fragments = data.get("max_fragments")
    explicit_min_chars = data.get("min_chars_per_chunk")

    if redis_client is None:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="internal").inc())
        return JSONResponse(status_code=500, content={"error": "Redis no está inicializado en /chatdoc/summary."})

    key = _chatdoc_key(user_id, document_id)

    # 1) Cache in-process
    index = None
    try:
        index = _chatdoc_cache_get(user_id, document_id)
    except Exception:
        index = None

    if index is not None:
        _safe_inc(lambda: CHATDOC_CACHE_TOTAL.labels(endpoint=endpoint_lbl, result="hit").inc())
        try:
            await redis_client.expire(key, CHATDOC_REDIS_TTL)
        except Exception:
            pass
        try:
            _chatdoc_cache_touch(user_id, document_id, CHATDOC_REDIS_TTL)
        except Exception:
            pass
    else:
        # 2) Redis -> reconstrucción (thread + cpu_sem)
        raw = await redis_client.get(key)

        if not raw:
            _safe_inc(lambda: CHATDOC_CACHE_TOTAL.labels(endpoint=endpoint_lbl, result="miss").inc())
            _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="no_index").inc())
            _safe_observe(lambda: CHATDOC_SUMMARY_FRAGMENTS.labels(strategy=strategy, detail_level=dl_label).observe(0))

            payload = {
                "status": "no_index",
                "message": "No hay índice efímero para este documento. Llama primero a /chatdoc/ingest.",
                "document_id": document_id,
                "fragments": [],
                "pre_summary_text": "",
                "coverage": None,
                "detail_level": detail_level,
                "strategy": strategy,
                "max_fragments": default_medium,
                "min_chars_per_chunk": _parse_int(os.getenv("CHATDOC_SUMMARY_MIN_CHARS_BASE", "300"), 300),
                "summary_profile": summary_profile,
            }
            if trace:
                payload["trace"] = {"cache_hit": False, "reason": "no_redis_key"}
            return JSONResponse(status_code=200, content=payload)

        _safe_inc(lambda: CHATDOC_CACHE_TOTAL.labels(endpoint=endpoint_lbl, result="hit").inc())

        def _rebuild_sync() -> "DocumentChatIndex":
            idx_data = json.loads(raw)
            return DocumentChatIndex.from_dict(idx_data)

        try:
            async with cpu_sem:
                index = await asyncio.wait_for(
                    asyncio.to_thread(_rebuild_sync),
                    timeout=rebuild_timeout_s,
                )
        except asyncio.TimeoutError:
            _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
            _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="timeout").inc())
            return JSONResponse(
                status_code=504,
                content={
                    "status": "error",
                    "error": "Timeout reconstruyendo índice desde Redis.",
                    "document_id": document_id,
                    "fragments": [],
                    "pre_summary_text": "",
                    "coverage": None,
                    "detail_level": detail_level,
                    "strategy": strategy,
                    "max_fragments": default_medium,
                    "min_chars_per_chunk": _parse_int(os.getenv("CHATDOC_SUMMARY_MIN_CHARS_BASE", "300"), 300),
                    "summary_profile": summary_profile,
                },
            )
        except Exception as e:
            logger.error("[/chatdoc/summary] Error reconstruyendo índice desde Redis: %s", e)
            logger.error(traceback.format_exc())
            _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
            _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="internal").inc())
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": f"No se pudo reconstruir el índice del documento: {e}",
                    "document_id": document_id,
                    "fragments": [],
                    "pre_summary_text": "",
                    "coverage": None,
                    "detail_level": detail_level,
                    "strategy": strategy,
                    "max_fragments": default_medium,
                    "min_chars_per_chunk": _parse_int(os.getenv("CHATDOC_SUMMARY_MIN_CHARS_BASE", "300"), 300),
                    "summary_profile": summary_profile,
                },
            )

        try:
            await redis_client.expire(key, CHATDOC_REDIS_TTL)
        except Exception:
            pass

        try:
            _chatdoc_cache_put(user_id, document_id, index, CHATDOC_REDIS_TTL)
        except Exception:
            pass

    # 3) Selección de fragmentos (thread + cpu_sem)
    adaptive_defaults = _infer_defaults_by_doc_size(index)

    if explicit_max_fragments is not None:
        max_fragments = _parse_int(explicit_max_fragments, default_medium)
    else:
        if detail_level == "low":
            max_fragments = default_low
        elif detail_level == "high":
            max_fragments = default_high
        elif detail_level == "medium":
            max_fragments = default_medium
        else:
            max_fragments = adaptive_defaults["max_fragments"]

    if max_fragments <= 0:
        max_fragments = default_medium
    max_fragments = min(max_fragments, max_cap)

    if explicit_min_chars is not None:
        min_chars_per_chunk = _parse_int(explicit_min_chars, 300)
    else:
        base_min_chars = _parse_int(os.getenv("CHATDOC_SUMMARY_MIN_CHARS_BASE", "300"), 300)
        if detail_level == "low":
            min_chars_per_chunk = int(base_min_chars * 1.5)
        elif detail_level == "high":
            min_chars_per_chunk = max(100, int(base_min_chars * 0.7))
        elif detail_level == "medium":
            min_chars_per_chunk = base_min_chars
        else:
            min_chars_per_chunk = adaptive_defaults["min_chars_per_chunk"]

    try:
        async with cpu_sem:
            fragments = await asyncio.wait_for(
                asyncio.to_thread(
                    index.select_summary_chunks,
                    max_fragments=max_fragments,
                    strategy=strategy,
                    min_chars_per_chunk=min_chars_per_chunk,
                ),
                timeout=summary_timeout_s,
            )
    except asyncio.TimeoutError:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="timeout").inc())
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "error": "Timeout generando fragmentos de resumen.",
                "document_id": document_id,
                "fragments": [],
                "pre_summary_text": "",
                "coverage": None,
                "detail_level": detail_level,
                "strategy": strategy,
                "max_fragments": max_fragments,
                "min_chars_per_chunk": min_chars_per_chunk,
                "summary_profile": summary_profile,
            },
        )
    except Exception as e:
        logger.error("[/chatdoc/summary] Error seleccionando fragmentos de resumen: %s", e)
        logger.error(traceback.format_exc())
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="error").inc())
        _safe_inc(lambda: CHATDOC_ERRORS_TOTAL.labels(endpoint=endpoint_lbl, kind="internal").inc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": f"Error interno generando fragmentos de resumen: {e}",
                "document_id": document_id,
                "fragments": [],
                "pre_summary_text": "",
                "coverage": None,
                "detail_level": detail_level,
                "strategy": strategy,
                "max_fragments": max_fragments,
                "min_chars_per_chunk": min_chars_per_chunk,
                "summary_profile": summary_profile,
            },
        )

    if not fragments:
        _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="no_chunks").inc())
        _safe_observe(lambda: CHATDOC_SUMMARY_FRAGMENTS.labels(strategy=strategy, detail_level=dl_label).observe(0))

        payload = {
            "status": "no_chunks",
            "message": "No se han podido seleccionar fragmentos representativos para el resumen.",
            "document_id": document_id,
            "fragments": [],
            "pre_summary_text": "",
            "coverage": None,
            "detail_level": detail_level,
            "strategy": strategy,
            "max_fragments": max_fragments,
            "min_chars_per_chunk": min_chars_per_chunk,
            "summary_profile": summary_profile,
        }
        if trace:
            payload["trace"] = {
                "cache_hit": True,
                "reason": "no_fragments_after_selection",
                "doc_stats": {
                    "full_text_chars": len(getattr(index, "full_text", "") or ""),
                    "chunks": len(getattr(index, "_chunks", []) or []),
                    "page_count": int(getattr(index, "page_count", 0) or 0),
                },
            }
        return JSONResponse(status_code=200, content=payload)

    frag_count = len(fragments) if isinstance(fragments, list) else 0
    _safe_inc(lambda: CHATDOC_STATUS_TOTAL.labels(endpoint=endpoint_lbl, status="ok").inc())
    _safe_observe(lambda: CHATDOC_SUMMARY_FRAGMENTS.labels(strategy=strategy, detail_level=dl_label).observe(frag_count))

    pieces = [frag.get("text", "") for frag in fragments if isinstance(frag, dict) and frag.get("text")]
    pre_summary_text = "\n\n---\n\n".join(pieces)
    if len(pre_summary_text) > CHATDOC_SUMMARY_MAX_CHARS:
        pre_summary_text = (
            pre_summary_text[:CHATDOC_SUMMARY_MAX_CHARS].rstrip()
            + "\n\n...(texto truncado para resumen)..."
        )

    coverage = _compute_coverage(index, fragments)

    response_payload: Dict[str, Any] = {
        "status": "ok",
        "document_id": document_id,
        "fragments": fragments,
        "pre_summary_text": pre_summary_text,
        "coverage": coverage,
        "detail_level": detail_level,
        "strategy": strategy,
        "max_fragments": max_fragments,
        "min_chars_per_chunk": min_chars_per_chunk,
        "summary_profile": summary_profile,
    }

    # Soporte opcional: return_format="text"
    if return_format == "text":
        response_payload["text"] = pre_summary_text

    if trace:
        response_payload["trace"] = {
            "cache_hit": True,
            "adaptive_defaults_used": explicit_max_fragments is None or explicit_min_chars is None,
            "adaptive_defaults": adaptive_defaults,
            "doc_stats": {
                "full_text_chars": len(getattr(index, "full_text", "") or ""),
                "chunks": len(getattr(index, "_chunks", []) or []),
                "page_count": int(getattr(index, "page_count", 0) or 0),
            },
        }

    return JSONResponse(status_code=200, content=response_payload)


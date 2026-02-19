# web_search/metrics.py
from __future__ import annotations

import os
import hashlib
from prometheus_client import Counter, Histogram, Gauge


# -----------------------------
# Utilidades
# -----------------------------
def status_class(code: int) -> str:
    try:
        return f"{int(code) // 100}xx"
    except Exception:
        return "unknown"


def hash_user_id(user_id: int) -> str:
    """
    Hash corto para label 'user'. Evita PII directa.
    OJO: cardinalidad sigue siendo por usuario; permite desactivar si quieres.
    """
    salt = os.getenv("METRICS_USER_HASH_SALT", "websearch_default_salt")
    raw = f"{salt}:{user_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def user_metrics_enabled() -> bool:
    return os.getenv("WEBSEARCH_USER_METRICS_ENABLED", "true").lower() in ("1", "true", "yes", "y")


# -----------------------------
# HTTP (estilo RAG)
# -----------------------------
WEBSEARCH_HTTP_REQUESTS_TOTAL = Counter(
    "websearch_http_requests_total",
    "Requests HTTP recibidas por el servicio web_search",
    ["endpoint", "method", "status_class"],
)

WEBSEARCH_HTTP_LATENCY_SECONDS = Histogram(
    "websearch_http_latency_seconds",
    "Latencia HTTP por endpoint (segundos)",
    ["endpoint", "method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 5, 10, 30, 60),
)

WEBSEARCH_HTTP_INFLIGHT = Gauge(
    "websearch_http_inflight",
    "Requests concurrentes en curso por endpoint",
    ["endpoint"],
)

# Requests por usuario (hasheado) - opcional
WEBSEARCH_USER_ENDPOINT_REQUESTS_TOTAL = Counter(
    "websearch_user_endpoint_requests_total",
    "Requests por usuario y endpoint (user hasheado)",
    ["endpoint", "user", "status_class"],
)


# -----------------------------
# Endpoint /search/query (semánticas)
# -----------------------------
WEBSEARCH_QUERY_REQUESTS_TOTAL = Counter(
    "websearch_query_requests_total",
    "Número total de peticiones de búsqueda (lógicas) al endpoint /search/query",
    ["endpoint"],
)

WEBSEARCH_QUERY_ERRORS_TOTAL = Counter(
    "websearch_query_errors_total",
    "Errores semánticos del endpoint /search/query",
    ["endpoint", "kind"],  # bad_request|csrf|auth|redis|db|crew|internal|json_parse
)

WEBSEARCH_QUERY_DURATION_SECONDS = Histogram(
    "websearch_query_duration_seconds",
    "Duración end-to-end de /search/query (segundos)",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13, 21, 34, 55),
)

WEBSEARCH_ANSWER_CHARS = Histogram(
    "websearch_answer_chars",
    "Longitud (chars) de la respuesta final",
    buckets=(0, 200, 500, 1000, 2000, 4000, 8000, 12000, 20000),
)

WEBSEARCH_SOURCES_PER_ANSWER = Histogram(
    "websearch_sources_per_answer",
    "Número de fuentes devueltas por respuesta",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30),
)

WEBSEARCH_NORMALIZED_QUERIES_PER_ANSWER = Histogram(
    "websearch_normalized_queries_per_answer",
    "Número de queries normalizadas usadas para producir una respuesta",
    buckets=(0, 1, 2, 3, 4, 5, 8, 10, 15),
)


# -----------------------------
# CrewAI / MCP tools (profundo)
# -----------------------------
WEBSEARCH_CREW_RUN_SECONDS = Histogram(
    "websearch_crew_run_seconds",
    "Duración del run de la crew (segundos)",
    ["top_k", "max_iters"],
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 3, 5, 8, 13, 21, 34, 55),
)

WEBSEARCH_AGENT_STEP_DURATION_SECONDS = Histogram(
    "websearch_agent_step_duration_seconds",
    "Duración por paso/agente (planner/executor/analyst/writer/reviewer) en segundos",
    ["step"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 5, 10, 30),
)

WEBSEARCH_TOOL_CALLS_TOTAL = Counter(
    "websearch_tool_calls_total",
    "Número de invocaciones a tools MCP (por nombre y status)",
    ["tool", "status"],  # ok|error|timeout|rate_limited|unknown
)

WEBSEARCH_TOOL_DURATION_SECONDS = Histogram(
    "websearch_tool_duration_seconds",
    "Duración de invocaciones a tools MCP (segundos)",
    ["tool"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 5, 10, 30),
)


# -----------------------------
# Redis
# -----------------------------
WEBSEARCH_REDIS_OPS_TOTAL = Counter(
    "websearch_redis_ops_total",
    "Operaciones Redis (hit/miss/error/decode_error)",
    ["op", "result"],
)

WEBSEARCH_REDIS_LATENCY_SECONDS = Histogram(
    "websearch_redis_latency_seconds",
    "Latencia operaciones Redis (segundos)",
    ["op"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1),
)

WEBSEARCH_REDIS_CONNECTED = Gauge(
    "websearch_redis_connected",
    "1 si Redis está inicializado/conectado (mejor-effort)",
)


# -----------------------------
# DB (SQLAlchemy)
# -----------------------------
WEBSEARCH_DB_LATENCY_SECONDS = Histogram(
    "websearch_db_latency_seconds",
    "Latencia DB por operación (segundos)",
    ["op"],
    buckets=(0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)

WEBSEARCH_DB_ERRORS_TOTAL = Counter(
    "websearch_db_errors_total",
    "Errores DB por operación",
    ["op"],
)


# -----------------------------
# Background tasks
# -----------------------------
WEBSEARCH_BG_TASKS_TOTAL = Counter(
    "websearch_bg_tasks_total",
    "Ejecuciones de background tasks (ok/error)",
    ["task", "status"],
)

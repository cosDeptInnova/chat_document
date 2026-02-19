from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge

# Peticiones a LLM
LLM_REQUESTS_TOTAL = Counter(
    "cosmos_llm_requests_total",
    "Total de peticiones al backend LLM",
    ["backend", "status_class"],
)

LLM_ERRORS_TOTAL = Counter(
    "cosmos_llm_errors_total",
    "Errores al llamar al backend LLM",
    ["backend", "kind"],
)

LLM_INFLIGHT = Gauge(
    "cosmos_llm_inflight",
    "Peticiones concurrentes al backend LLM",
    ["backend"],
)

LLM_LATENCY_SECONDS = Histogram(
    "cosmos_llm_latency_seconds",
    "Latencia de llamada al backend LLM (segundos)",
    ["backend"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80),
)

LLM_TOKENS_TOTAL = Counter(
    "cosmos_llm_tokens_total",
    "Tokens reportados por el backend LLM",
    ["backend", "type"],  # prompt|completion|total
)

LLM_TOKENS_PER_REQUEST = Histogram(
    "cosmos_llm_tokens_per_request",
    "Tokens por petición al backend LLM (distribución)",
    ["backend", "type"],  # prompt|completion|total
    buckets=(1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
)

LLM_USAGE_MISSING_TOTAL = Counter(
    "cosmos_llm_usage_missing_total",
    "Respuestas del backend LLM sin campo usage (no se pudo medir tokens)",
    ["backend"],
)



WEBSEARCH_TOOL_CALLS_TOTAL = Counter(
    "websearch_tool_calls_total",
    "Total de invocaciones a tools servidas por el MCP (Cosmos bus)",
    ["tool", "result"],  # result: ok|error
)

WEBSEARCH_TOOL_DURATION_SECONDS = Histogram(
    "websearch_tool_duration_seconds",
    "Duración de ejecución de tools MCP (segundos)",
    ["tool", "result"],  # result: ok|error
    # buckets pensados para tools rápidas y algunas lentas
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40),
)

WEBSEARCH_AGENT_STEP_DURATION_SECONDS = Histogram(
    "websearch_agent_step_duration_seconds",
    "Duración p/step de ejecución (pipeline interno: HTTP MCP / Tool dispatch, etc.)",
    ["step"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40),
)
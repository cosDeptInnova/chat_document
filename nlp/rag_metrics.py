from prometheus_client import Counter, Histogram, Gauge

# RAG (interno / qdrant)
INDEXED_CHUNKS = Counter('rag_indexed_chunks_total', 'Chunks indexados', ['collection'])
SEARCH_REQUESTS = Counter('rag_search_requests_total', 'Búsquedas recibidas', ['status'])
SEARCH_LATENCY = Histogram(
    'rag_search_latency_seconds',
    'Latencia búsquedas',
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
)
MMR_TIME = Histogram(
    'rag_mmr_compute_seconds',
    'Tiempo MMR',
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05]
)
QDRANT_COLLECTION_SIZE = Gauge('rag_qdrant_collection_size', 'Tamaño colección Qdrant', ['collection'])
DOCUMENTS_PROCESSED = Counter('upsert_documents_processed_total', 'Total de documentos procesados en upsert', ['collection'])
EMBEDDING_DURATION = Histogram('upsert_embedding_duration_seconds', 'Duración embedding por documento', ['collection'])
POINT_CREATION_DURATION = Histogram('upsert_point_creation_duration_seconds', 'Duración construcción Point', ['collection'])
BATCH_UPSERT_DURATION = Histogram('upsert_batch_duration_seconds', 'Duración upsert por lote', ['collection'])

# SPLADE diferido
SPARSE_QUEUE_SIZE = Gauge('rag_sparse_queue_size', 'Tamaño cola SPLADE diferida')
SPARSE_EMBED_DURATION = Histogram(
    'sparse_embed_duration_seconds',
    'Duración embedding SPLADE',
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)
SPARSE_TASKS_PROCESSED = Counter(
    'rag_sparse_tasks_processed_total',
    'Tareas SPLADE diferidas procesadas'
)

# --- HTTP observability (global por endpoint) ---
RAG_HTTP_REQUESTS_TOTAL = Counter(
    "rag_http_requests_total",
    "Requests HTTP recibidas por el servicio RAG",
    ["endpoint", "method", "status_class"],
)

RAG_HTTP_LATENCY_SECONDS = Histogram(
    "rag_http_latency_seconds",
    "Latencia HTTP por endpoint",
    ["endpoint", "method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 5, 10, 30, 60),
)

RAG_HTTP_INFLIGHT = Gauge(
    "rag_http_inflight",
    "Requests concurrentes en curso por endpoint",
    ["endpoint"],
)

RAG_USER_ENDPOINT_REQUESTS_TOTAL = Counter(
    "rag_user_endpoint_requests_total",
    "Requests por usuario y endpoint (user normalmente hasheado)",
    ["endpoint", "user", "status_class"],
)

# ChatDoc (documentos grandes) - NLP (5000)
CHATDOC_STATUS_TOTAL = Counter(
    "chatdoc_status_total",
    "Respuestas ChatDoc por endpoint y estado semántico",
    ["endpoint", "status"],  # ok|no_index|no_results|no_chunks|error
)

CHATDOC_ERRORS_TOTAL = Counter(
    "chatdoc_errors_total",
    "Errores ChatDoc por endpoint y tipo",
    ["endpoint", "kind"],  # bad_request|auth|too_large|unprocessable|internal|json_parse
)

CHATDOC_CACHE_TOTAL = Counter(
    "chatdoc_cache_total",
    "Cache hit/miss (Redis) para endpoints ChatDoc",
    ["endpoint", "result"],  # hit|miss
)

CHATDOC_INGEST_BUILD_SECONDS = Histogram(
    "chatdoc_ingest_build_seconds",
    "Tiempo de construcción del índice efímero (DocumentChatIndex)",
    ["mode"],  # bytes|text|none
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60),
)

CHATDOC_INGEST_REDIS_SECONDS = Histogram(
    "chatdoc_ingest_redis_seconds",
    "Tiempo de escritura en Redis del índice efímero",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)

CHATDOC_INGEST_TOTAL_SECONDS = Histogram(
    "chatdoc_ingest_total_seconds",
    "Tiempo total del endpoint /chatdoc/ingest (segundos)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60),
)

CHATDOC_INGEST_BYTES = Histogram(
    "chatdoc_ingest_bytes",
    "Tamaño del documento ingerido (bytes)",
    ["mode"],  # bytes|text|none
    buckets=(100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000),
)

CHATDOC_INGEST_CHUNKS = Histogram(
    "chatdoc_ingest_chunks",
    "Número de chunks generados en la ingesta",
    ["mode"],
    buckets=(1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000),
)

CHATDOC_INGEST_TEXT_CHARS = Histogram(
    "chatdoc_ingest_text_chars",
    "Longitud total del texto extraído (caracteres)",
    ["mode"],
    buckets=(1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000),
)

CHATDOC_INGEST_PAGE_COUNT = Histogram(
    "chatdoc_ingest_page_count",
    "Páginas detectadas/estimadas del documento ingerido",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
)

CHATDOC_QUERY_RESULTS_COUNT = Histogram(
    "chatdoc_query_results_count",
    "Número de resultados devueltos por /chatdoc/query",
    buckets=(0, 1, 2, 5, 10, 20, 50),
)

CHATDOC_SUMMARY_FRAGMENTS = Histogram(
    "chatdoc_summary_fragments",
    "Número de fragmentos seleccionados para /chatdoc/summary",
    ["strategy", "detail_level"],
    buckets=(0, 2, 4, 8, 12, 16, 24, 32, 48, 64),
)

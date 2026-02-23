# cosmos_nlp_v3.py
import os
import spacy
import logging
import torch
from typing import List, Optional, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from redis.asyncio import Redis
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import threading

from device_manager import gpu_manager
#-----------------------------------------------A PARTIR DE AQUÍ, LOS IMPORTS SON DEL PIPELINE RAG PROPIO------------------------------------
from rag_metrics import (
    SPARSE_QUEUE_SIZE,
    SPARSE_EMBED_DURATION,
    SPARSE_TASKS_PROCESSED,
)
from rag_utils import (
    ensure_local_with_transformers,
    _dir_has_hf_model,
    _atomic_write_file,
    _canon,
    _canon_key,
    _canon_val,
    _parse_number_es,
)
from rag_qdrant import create_collection_if_not_exists, ensure_collection_compatible
from rag_lexical import BM25Index
from rag_upsert import upsert_documents_to_db_impl, delete_document_by_name_impl
from rag_rerank import (
    get_reranker_impl,
    get_colbert_encoder_impl,
    late_interaction_scores_impl,
    rerank_results_impl,
    evaluate_results_impl,
    log_query_impl,
)
from rag_loader import (
    clean_text_impl,
    load_txt_impl,
    load_docx_impl,
    load_pdf_impl,
    load_pptx_impl,
    load_and_fragment_files_impl,
)
from rag_storage import (
    load_vector_db_impl,
    save_documents_impl,
    save_doc_names_impl,
    save_metas_impl,
)
from rag_tags import get_collection_tags_impl, get_matched_tags_impl
from rag_textsplit import dynamic_fragment_size_impl, fragment_text_semantically_impl
from rag_sparse import (
    update_sparse_vectors_compat_impl,
    ensure_sparse_worker_impl,
    enqueue_sparse_update_impl,
    sparse_worker_loop_impl,
    get_splade_model_impl,
    sparse_encode_text_impl,
    sparse_encode_query_impl,
    sparse_search_raw_impl,
)
from rag_similarity import (
    retrieve_similar_blocks_impl,
)
from rag_search import (
    vector_search_raw_impl,
    hybrid_search_rrf_impl,
    filter_by_score_threshold_impl,
    search_impl,
)


logger = logging.getLogger("cosmos_nlp_v3")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Usa el modelo mediano para buen sentence split
nlp = spacy.load("es_core_news_md")

LOCAL_EMBED_DIR  = os.getenv("LOCAL_EMBED_DIR", r"C:\Users\ADM-mcabo\Documents\cosmos_cache\st-e5")
LOCAL_RERANK_DIR = os.getenv("LOCAL_RERANK_DIR", r"C:\Users\ADM-mcabo\Documents\cosmos_cache\reranker-bge-v2-m3")
USE_E5_PREFIX = True
E5_QUERY_PREFIX = "query: "
E5_DOC_PREFIX   = "passage: "

# Singletons globales para evitar múltiples copias del mismo modelo en VRAM - HIPERIMPOTANTE porque a un usuario le ha saltado error por este motivo (cuello de botella por multiples instancias de modelos para RAG)
_GLOBAL_EMBED_MODEL = None
_GLOBAL_EMBED_DEVICE = None
_EMBED_INIT_LOCK = threading.Lock()

# --- Semáforo global para embeddings (RAG) ---
try:
    _RAG_EMBED_CONCURRENCY = int(os.getenv("RAG_EMBED_CONCURRENCY", "2"))
except Exception:
    _RAG_EMBED_CONCURRENCY = 2
_RAG_EMBED_CONCURRENCY = max(1, _RAG_EMBED_CONCURRENCY)

_RAG_EMBED_GPU_SEMAPHORE = threading.Semaphore(_RAG_EMBED_CONCURRENCY)

class DocumentIndexer:
    def __init__(
        self,
        files: Optional[List[str]] = None,
        index_filepath: Optional[str] = None,
        user_id: Optional[str] = None,
        client_tag: Optional[str] = None,
        redis_client: Optional[Redis] = None,
        host: str = None,
        port: int = None,
        auto_index_on_load: Optional[bool] = None,
    ):

        if redis_client is None:
            raise ValueError("Se requiere pasar redis_client al DocumentIndexer")
        self.redis_client = redis_client

        # Estado base
        self.files = files
        self.index_filepath = index_filepath
        self.doc_names: List[str] = []
        self.documents: List[str] = []
        self._bm25_index = BM25Index()
        self._metas: List[Dict[str, object]] = []
        self.user_id = user_id or "default_user"
        self.collection_name = f"my_collection_{self.user_id}"
        self.client_tag = client_tag.strip() if client_tag else None

        # -----------------------------
        # Qdrant (HTTP/grpc auto)
        # -----------------------------
        q_host = host or os.getenv("QDRANT_HOST", "localhost")
        http_port = int(port or int(os.getenv("QDRANT_HTTP_PORT", "6333")))
        grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
        prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "1") == "1"

        try:
            self.qdrant_client = QdrantClient(
                host=q_host,
                port=http_port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                timeout=float(os.getenv("QDRANT_TIMEOUT_SEC", "45.0")),
            )
        except TypeError:
            self.qdrant_client = QdrantClient(host=q_host, port=http_port, timeout=45.0)

        # -----------------------------
        # Modelos locales (E5 embeddings) - SINGLETON + device_manager
        # -----------------------------
        global _GLOBAL_EMBED_MODEL, _GLOBAL_EMBED_DEVICE

        if _GLOBAL_EMBED_MODEL is None:
            with _EMBED_INIT_LOCK:
                if _GLOBAL_EMBED_MODEL is None:
                    embed_id = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-large")

                    # Descarga/check SOLO una vez por proceso
                    if not self._dir_has_hf_model(LOCAL_EMBED_DIR):
                        logger.info(
                            f"(embed) modelo incompleto/no encontrado en '{LOCAL_EMBED_DIR}'. "
                            f"Descargando '{embed_id}' y guardando en local…"
                        )
                        ensure_local_with_transformers(embed_id, LOCAL_EMBED_DIR, task="auto")
                    else:
                        logger.info(f"(embed) usando modelo local en: {LOCAL_EMBED_DIR}")

                    # Política de init device (para no “pinnear” VRAM si no quieres)
                    policy = (os.getenv("HF_EMBED_INIT_DEVICE") or os.getenv("RAG_EMBED_INIT_DEVICE") or "auto").strip().lower()
                    min_free = int(os.getenv("HF_EMBED_MIN_FREE_MB", "4096"))

                    embed_device: torch.device
                    if policy in ("cpu", "none", "off"):
                        embed_device = torch.device("cpu")
                    elif policy.startswith("cuda"):
                        # p.ej. "cuda:0"
                        if torch.cuda.is_available():
                            embed_device = torch.device(policy)
                        else:
                            embed_device = torch.device("cpu")
                    else:
                        # auto/best
                        embed_device = gpu_manager.best_device(min_free_mb=min_free)

                    device_str = str(embed_device) if embed_device.type == "cuda" else "cpu"
                    logger.info(f"(embed) inicializando SentenceTransformer en {device_str}")

                    _GLOBAL_EMBED_MODEL = SentenceTransformer(
                        LOCAL_EMBED_DIR,
                        device=device_str,
                    )
                    _GLOBAL_EMBED_DEVICE = embed_device
        else:
            logger.info(f"(embed) reutilizando SentenceTransformer global en {_GLOBAL_EMBED_DEVICE}")

        self.sbert_model = _GLOBAL_EMBED_MODEL
        self.embed_device: torch.device = _GLOBAL_EMBED_DEVICE or torch.device("cpu")

        # -----------------------------
        # Reranker local: asegurar assets UNA vez por proceso (la carga real sigue lazy en rag_rerank)
        # -----------------------------
        self._rerank_dir = LOCAL_RERANK_DIR
        rerank_id = os.getenv("RERANK_MODEL_ID", "BAAI/bge-reranker-v2-m3")

        g = globals()
        if "_RERANK_ASSET_LOCK" not in g:
            g["_RERANK_ASSET_LOCK"] = threading.Lock()
            g["_RERANK_ASSET_READY"] = False

        with g["_RERANK_ASSET_LOCK"]:
            if not g["_RERANK_ASSET_READY"]:
                if not self._dir_has_hf_model(self._rerank_dir):
                    logger.info(
                        f"(rerank) modelo incompleto/no encontrado en '{self._rerank_dir}'. "
                        f"Descargando '{rerank_id}'…"
                    )
                    ensure_local_with_transformers(rerank_id, self._rerank_dir, task="seq-cls")
                else:
                    logger.info(f"(rerank) usando modelo local en: {self._rerank_dir}")
                g["_RERANK_ASSET_READY"] = True

        self._reranker = None  # lazy
        self._use_e5_prefix = bool(USE_E5_PREFIX)
        self._payload_index_ready = False

        # Resolver política de auto-indexado
        auto_index_on_load = self._resolve_auto_index_flag(auto_index_on_load)

        # Validaciones de ruta/argumentos
        if index_filepath and os.path.isdir(index_filepath):
            raise ValueError("La ruta del índice vectorial es un directorio, no un archivo.")
        if files is not None and not index_filepath:
            raise ValueError("Debe proporcionar 'index_filepath' cuando se pasan archivos en 'files'.")

        # Flujo sin archivos
        if files is None:
            self.load_vector_db(index_filepath)
            if auto_index_on_load:
                try:
                    cnt = self.qdrant_client.count(
                        collection_name=self.collection_name,
                        exact=True
                    ).count
                except Exception:
                    cnt = 0
                if cnt == 0 and self.documents:
                    logger.info(
                        f"(auto-index) Colección '{self.collection_name}' vacía; generando embeddings de "
                        f"{len(self.documents)} fragmentos cargados desde pickles…"
                    )
                    self.upsert_documents_to_db(self.documents, self.doc_names)
                else:
                    logger.info(
                        f"(auto-index) Colección '{self.collection_name}' ya contiene {cnt} puntos; no se reindexa."
                    )
            return

        # Con archivos: incremental o desde cero
        base_path = os.path.splitext(index_filepath or "user_index.pkl")[0]
        dn_path   = f"{base_path}_docnames.pkl"
        docs_path = f"{base_path}_documents.pkl"

        if os.path.exists(dn_path) and os.path.exists(docs_path):
            self.load_vector_db(index_filepath)
            new_docs, new_names, _ = self.load_and_fragment_files(files)
            if new_docs:
                with self._get_user_lock():
                    self.upsert_documents_to_db(new_docs, new_names)
                    self.documents += new_docs
                    self.doc_names  += new_names
                    self.save_documents(index_filepath)
                    self.save_doc_names(index_filepath)
                    self._invalidate_bm25_if_needed()
            else:
                logger.info("(init) No se han generado nuevos fragmentos a partir de los archivos proporcionados.")
        else:
            with self._get_user_lock():
                self._metas = []  # reset limpio SOLO en new index
                self.documents, self.doc_names, _ = self.load_and_fragment_files(files)
                self._ensure_collection()
                if self.documents:
                    self.upsert_documents_to_db(self.documents, self.doc_names)
                    self.save_documents(index_filepath)
                    self.save_doc_names(index_filepath)
                    self._invalidate_bm25_if_needed()
                else:
                    logger.warning("(init) No se generaron fragmentos de los archivos proporcionados; no se indexará nada.")


    def _resolve_auto_index_flag(self, auto_index_on_load: Optional[bool]) -> bool:
        if auto_index_on_load is not None:
            return bool(auto_index_on_load)
        raw = os.getenv("COSMOS_AUTO_INDEX", "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if raw in ("1", "true", "yes", "on"):
            return True
        return True

    @torch.inference_mode()
    def _embed_texts(
        self,
        texts: List[str],
        *,
        is_query: bool = False,
        batch_size: Optional[int] = None
    ):
        """
        Embedding unificado:
        - Prefijos E5
        - Batch por env
        - Normaliza
        - DeviceManager con fallback + pool limiter
        """
        import numpy as np

        bs = int(batch_size or int(os.getenv("HF_EMBED_BATCH", "32")))

        if self._use_e5_prefix:
            if is_query:
                texts = [f"{E5_QUERY_PREFIX}{t or ''}" for t in texts]
            else:
                texts = [f"{E5_DOC_PREFIX}{t or ''}" for t in texts]

        global _GLOBAL_EMBED_DEVICE

        used_dev: Optional[torch.device] = None

        with gpu_manager.use_device_with_fallback(
            self.sbert_model,
            prefer_device=self.embed_device,
            min_free_mb=int(os.getenv("HF_EMBED_MIN_FREE_MB", "2048")),
            fallback_to_cpu=True,
            pool="embed",
        ) as dev:
            used_dev = dev
            device_arg = str(dev) if dev.type == "cuda" else "cpu"

            vecs = self.sbert_model.encode(
                texts,
                batch_size=bs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=device_arg,
                show_progress_bar=False,
            )

        # Solo “stick” a CUDA si realmente se usó CUDA (evita quedarse pegado a CPU tras un fallback puntual)
        if used_dev is not None and used_dev.type == "cuda":
            if used_dev != self.embed_device:
                self.embed_device = used_dev
            if _GLOBAL_EMBED_DEVICE is None or _GLOBAL_EMBED_DEVICE.type != "cuda" or _GLOBAL_EMBED_DEVICE != used_dev:
                _GLOBAL_EMBED_DEVICE = used_dev

        try:
            return np.asarray(vecs, dtype="float32")
        except Exception:
            return vecs

    def _build_payload(self, text: str, doc_name: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Construye payload con metadata enriquecida (Docling/MinerU/Excel/Clásico).
        + Evolución: añade campos canónicos del nombre de archivo para:
        - resolver consultas tipo "resumen del archivo X.pdf"
        - filtrar por documento en Qdrant sin depender de onFly
        """
        import hashlib
        import os
        from typing import Any, Dict, List, Optional

        meta = dict(meta or {})
        source_path = meta.get("source_path")
        file_name = meta.get("source_file_name") or os.path.basename(source_path or doc_name) or doc_name
        backend = meta.get("backend") or "classic"

        # doc_id estable por ruta (o file_name si no hay path)
        doc_id_basis = (source_path or file_name).encode("utf-8", errors="ignore")
        doc_id = hashlib.sha1(doc_id_basis).hexdigest()

        # page (si existe)
        page = meta.get("page") or meta.get("page_number")
        try:
            page = int(page) if page is not None else None
        except Exception:
            page = None

        # headings/sections
        def _collect_headings(m: Dict[str, Any]) -> List[str]:
            candidates: List[str] = []
            for k, v in m.items():
                kl = str(k).lower()
                if any(t in kl for t in ("heading", "title", "section")):
                    if isinstance(v, str):
                        if v.strip():
                            candidates.append(v.strip())
                    elif isinstance(v, (list, tuple)):
                        for vv in v:
                            sv = str(vv).strip()
                            if sv:
                                candidates.append(sv)
            seen = set()
            out: List[str] = []
            for t in candidates:
                tl = t.lower()
                if tl not in seen:
                    seen.add(tl)
                    out.append(t)
            return out

        headings = _collect_headings(meta)
        section_path = None
        section_ancestors: List[str] = []
        if headings:
            section_path = " / ".join(headings)
            parts: List[str] = []
            for h in headings:
                parts.append(h)
                section_ancestors.append(" / ".join(parts))

        tags: List[str] = []
        if isinstance(meta.get("tags"), list):
            tags = [str(t).strip() for t in meta["tags"] if str(t).strip()]
        tags += headings

        # de-dup tags
        norm_seen = set()
        norm_tags: List[str] = []
        for t in tags:
            tl = t.lower()
            if tl not in norm_seen:
                norm_seen.add(tl)
                norm_tags.append(t)

        # --- NUEVO: campos canónicos del nombre de archivo para búsqueda/filtrado ---
        stem, ext = os.path.splitext(str(file_name))
        file_ext = ext.lstrip(".").lower().strip() or None

        try:
            file_name_canon = self._canon(str(file_name))
        except Exception:
            file_name_canon = str(file_name).strip().lower()

        try:
            file_stem_canon = self._canon(str(stem))
        except Exception:
            file_stem_canon = str(stem).strip().lower()

        base_payload: Dict[str, Any] = {
            "text": text,
            "file_name": file_name,
            "file_name_canon": file_name_canon,
            "file_stem_canon": file_stem_canon,
            "file_ext": file_ext,

            "source_path": source_path,
            "backend": backend,
            "doc_id": doc_id,
            "page": page,
            "tags": norm_tags if norm_tags else None,
            "section_path": section_path,
            "section_ancestors": section_ancestors if section_ancestors else None,
        }

        # Passthrough tabular / extra
        passthrough_keys = [
            "sheet", "row_idx", "headers", "row_kv", "row_kv_canon",
            "block_type", "bbox", "chunk_index",
            "table_id", "table_cache_path", "row_id",
            "location", "brand", "model", "class", "asset_type",
            "serial", "inventory_id", "asset_id",
            "quantity", "price",
        ]
        for k in passthrough_keys:
            if k in meta and meta[k] is not None:
                base_payload[k] = meta[k]

        # Limpia None
        return {k: v for k, v in base_payload.items() if v is not None}

    def _ensure_payload_indexes(self) -> None:
        """
        Crea índices de payload en Qdrant (perezoso).
        + Evolución: índices para búsqueda por nombre de documento:
        - file_name_canon, file_stem_canon, file_ext, source_path
        """
        if getattr(self, "_payload_index_ready", False):
            return
        try:
            from qdrant_client.models import PayloadSchemaType

            fields_and_types = [
                ("tags", PayloadSchemaType.KEYWORD),
                ("file_name", PayloadSchemaType.KEYWORD),
                ("file_name_canon", PayloadSchemaType.KEYWORD),
                ("file_stem_canon", PayloadSchemaType.KEYWORD),
                ("file_ext", PayloadSchemaType.KEYWORD),
                ("source_path", PayloadSchemaType.KEYWORD),

                ("doc_id", PayloadSchemaType.KEYWORD),
                ("backend", PayloadSchemaType.KEYWORD),
                ("section_ancestors", PayloadSchemaType.KEYWORD),
                ("page", PayloadSchemaType.INTEGER),

                ("sheet", PayloadSchemaType.KEYWORD),
                ("table_id", PayloadSchemaType.KEYWORD),
                ("table_cache_path", PayloadSchemaType.KEYWORD),
                ("row_idx", PayloadSchemaType.INTEGER),
                ("row_id", PayloadSchemaType.INTEGER),

                ("location", PayloadSchemaType.KEYWORD),
                ("brand", PayloadSchemaType.KEYWORD),
                ("model", PayloadSchemaType.KEYWORD),
                ("class", PayloadSchemaType.KEYWORD),
                ("asset_type", PayloadSchemaType.KEYWORD),
                ("serial", PayloadSchemaType.KEYWORD),
                ("inventory_id", PayloadSchemaType.KEYWORD),
                ("asset_id", PayloadSchemaType.KEYWORD),

                ("quantity", PayloadSchemaType.FLOAT),
                ("price", PayloadSchemaType.FLOAT),
            ]

            for field, schema in fields_and_types:
                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=schema,
                    )
                except Exception:
                    pass

            self._payload_index_ready = True
        except Exception:
            # No fallar: el sistema debe seguir funcionando aunque no haya índices
            self._payload_index_ready = True

    def _extract_document_hint_from_query(self, query: str) -> Optional[str]:
        """
        Extrae un posible nombre de archivo del texto del usuario.
        Ejemplos:
        - "resume el archivo pliego_aena.pdf" -> "pliego_aena.pdf"
        - "qué pone en Inventario Valencia.xlsx?" -> "Inventario Valencia.xlsx"
        """
        import re

        q = (query or "").strip()
        if not q:
            return None

        # Captura nombres con extensión típica
        m = re.search(
            r'([^\s"\'<>]+?\.(?:pdf|docx|doc|pptx|ppt|xlsx|xls|csv|txt))\b',
            q,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()

        # Heurística: "archivo/documento <algo>" (sin extensión)
        m2 = re.search(r"\b(?:archivo|documento|fichero)\s+([^\n\r\?]+)", q, flags=re.IGNORECASE)
        if m2:
            cand = m2.group(1).strip()
            # corta por signos típicos
            cand = re.split(r"[\,\;\:\?\!]", cand)[0].strip()
            return cand or None

        return None


    def _is_document_overview_query(self, query: str) -> bool:
        """
        Detecta intención tipo 'resumen del archivo X' / 'de qué trataba X'.
        En esos casos NO debemos depender de similitud semántica del nombre del archivo.
        """
        import re
        q = (query or "").lower()
        if not q:
            return False

        patterns = [
            r"\bresumen\b",
            r"\bsinopsis\b",
            r"\bde\s+qu[eé]\s+trataba\b",
            r"\bqu[eé]\s+pone\b",
            r"\bexplica(?:me)?\b.*\barchivo\b",
            r"\bcontenido\b.*\barchivo\b",
            r"\bdescrib(?:e|eme)\b.*\barchivo\b",
        ]
        return any(re.search(p, q) for p in patterns)


    def _known_file_names(self) -> List[str]:
        """
        Devuelve lista única de file_names conocidos (desde metas/doc_names).
        No falla si faltan metas.
        """
        import os

        seen = set()
        out: List[str] = []

        # 1) metas (si están)
        for m in (self._metas or []):
            if not isinstance(m, dict):
                continue
            sp = m.get("source_path") or m.get("path")
            fn = m.get("file_name") or m.get("source_file_name")
            if not fn:
                fn = os.path.basename(sp) if sp else None
            if fn:
                key = str(fn).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(str(fn))

        # 2) doc_names (fallback)
        for dn in (self.doc_names or []):
            if not dn:
                continue
            base = os.path.basename(str(dn).split("#")[0])
            key = base.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(base)

        return out


    def resolve_document_name(self, hint: str) -> Optional[str]:
        """
        Resuelve un 'hint' (p.ej. 'AENA.pdf', 'pliego aena', 'inventario_valencia.xlsx')
        contra los documentos indexados, con matching robusto (canon + substrings).
        """
        import os

        if not hint:
            return None

        hint = str(hint).strip()
        if not hint:
            return None

        try:
            h_canon = self._canon(hint)
        except Exception:
            h_canon = hint.lower()

        # Si el hint trae ruta, quedarnos con basename para comparar
        hint_base = os.path.basename(hint)
        try:
            hint_base_canon = self._canon(hint_base)
        except Exception:
            hint_base_canon = hint_base.lower()

        candidates = self._known_file_names()
        if not candidates:
            return None

        # Scoring simple (estable y barato)
        def score(cand: str) -> float:
            c_base = os.path.basename(str(cand))
            try:
                c_canon = self._canon(c_base)
            except Exception:
                c_canon = c_base.lower()

            if c_canon == hint_base_canon or c_canon == h_canon:
                return 100.0

            c_stem = os.path.splitext(c_canon)[0]
            h_stem = os.path.splitext(hint_base_canon)[0]

            if c_stem == h_stem:
                return 95.0
            if c_canon.startswith(hint_base_canon) or hint_base_canon.startswith(c_canon):
                return 80.0
            if hint_base_canon in c_canon or c_canon in hint_base_canon:
                return 70.0

            # token overlap
            c_tokens = {t for t in c_stem.replace("-", " ").replace("_", " ").split() if t}
            h_tokens = {t for t in h_stem.replace("-", " ").replace("_", " ").split() if t}
            if not c_tokens or not h_tokens:
                return 0.0
            inter = len(c_tokens & h_tokens)
            union = len(c_tokens | h_tokens)
            return 50.0 * (inter / max(1, union))

        best = None
        best_s = 0.0
        for c in candidates:
            s = score(c)
            if s > best_s:
                best_s = s
                best = c

        # Umbral conservador para evitar falsos positivos
        return best if best and best_s >= 60.0 else None

    def _qdrant_scroll_by_file_name(self, file_name: str, limit: int = 200) -> List[dict]:
        """
        Recupera puntos de Qdrant filtrando por documento.
        Backward compatible:
        - file_name exact
        - OR file_name_canon exact (si existe en puntos nuevos)
        - OR file_stem_canon (para hints sin extensión o variantes)
        """
        if not file_name:
            return []

        try:
            self._ensure_payload_indexes()
        except Exception:
            pass

        import os

        # Valores auxiliares
        fn = str(file_name).strip()
        stem, _ext = os.path.splitext(fn)

        try:
            fn_canon = self._canon(fn)
        except Exception:
            fn_canon = fn.lower()

        try:
            stem_canon = self._canon(stem)
        except Exception:
            stem_canon = stem.lower()

        # Construimos OR (should) para ser tolerantes
        should = []
        try:
            should.append(qmodels.FieldCondition(key="file_name", match=qmodels.MatchValue(value=fn)))
            if fn_canon:
                should.append(qmodels.FieldCondition(key="file_name_canon", match=qmodels.MatchValue(value=fn_canon)))
            if stem_canon:
                should.append(qmodels.FieldCondition(key="file_stem_canon", match=qmodels.MatchValue(value=stem_canon)))
        except Exception:
            # Si por compat de modelos falla, degradamos a file_name exact
            should = [qmodels.FieldCondition(key="file_name", match=qmodels.MatchValue(value=fn))]

        qfilter = qmodels.Filter(should=should)

        points = []
        next_offset = None

        while len(points) < limit:
            batch = min(128, limit - len(points))
            try:
                res = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qfilter,
                    limit=batch,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except TypeError:
                res = self.qdrant_client.scroll(
                    self.collection_name,
                    qfilter,
                    batch,
                    next_offset,
                    with_payload=True,
                    with_vectors=False,
                )

            if not res:
                break

            pts, next_offset = res
            if not pts:
                break

            points.extend(list(pts))
            if next_offset is None:
                break

        out: List[dict] = []
        for p in points:
            payload = getattr(p, "payload", None) or {}
            pid = getattr(p, "id", None)
            txt = payload.get("text") or ""
            out.append(
                {
                    "id": pid,
                    "doc_name": payload.get("file_name") or fn,
                    "text": txt,
                    "score": 1.0,
                    "meta": payload,
                    # por compat con consumers de search_with_similar_blocks
                    "similar_blocks": [],
                }
            )
        return out


    def _document_overview_hits(self, file_name: str, top_k: int = 12) -> List[dict]:
        """
        Recupera un conjunto representativo de fragmentos para resumir el documento entero.
        Estrategia:
        - scroll por file_name
        - ordenar por (page, chunk_index) si existe
        - devolver primeros top_k
        """
        hits = self._qdrant_scroll_by_file_name(file_name, limit=max(80, top_k * 8))
        if not hits:
            return []

        def _key(h: dict):
            m = h.get("meta") or {}
            page = m.get("page")
            chunk_index = m.get("chunk_index")
            try:
                page = int(page) if page is not None else 10**9
            except Exception:
                page = 10**9
            try:
                chunk_index = int(chunk_index) if chunk_index is not None else 10**9
            except Exception:
                chunk_index = 10**9
            return (page, chunk_index)

        hits.sort(key=_key)
        return hits[: max(1, int(top_k))]


    def aggregate_excel(self, query: str, hits: list, top_k_tables: int = 3) -> Optional[Dict[str, Any]]:
        """
        Agregación tabular (conteos) sobre hojas Excel cacheadas (Parquet/CSV).

        - 'hits' debe ser una lista de dicts con al menos meta.table_cache_path y meta.table_id
          (se construyen desde los resultados rerankeados resolviendo el índice del texto
          hacia self._metas[indice]).
        - Devuelve dict con 'count' total y detalle por tabla, o None si no hay datos.

        Mejora clave:
        - Extrae un token de localización desde la query (p.ej. "X ubicacion" en
          "cuántos equipos hay en X ubicacion?") y lo usa como patrón de búsqueda
          en columnas de ubicación/sede/delegación/centro/oficina.
        """
        try:
            import pandas as pd
        except Exception:
            # pandas no disponible: no fallar la búsqueda
            return None

        import re
        import unicodedata

        def _norm_text(s: Any) -> str:
            s = unicodedata.normalize("NFD", str(s))
            s = "".join(c for c in s if unicodedata.category(c) != "Mn")
            s = re.sub(r"\s+", " ", s).strip().lower()
            return s

        raw_q = query or ""
        qn_full = _norm_text(raw_q)
        loc_candidate = None
        m = re.search(r"\ben\s+(?:la|el|los|las)?\s*(.+)", raw_q, flags=re.I)
        if m:
            loc_candidate = m.group(1)
            # Limpiamos puntuación final: ?, !, .
            loc_candidate = re.sub(r"[\?\!\.]+$", "", loc_candidate or "").strip()

        if loc_candidate:
            filter_token = _norm_text(loc_candidate)
        else:
            # Fallback: quitamos palabras típicas de conteo y dejamos el resto
            tmp = re.sub(
                r"\b(cu[aá]nt[oa]s?|n[úu]mero de|total de|cuenta|how many|count|equipos?|impresoras?)\b",
                " ",
                raw_q,
                flags=re.I,
            )
            tmp = re.sub(r"[¿\?]+", " ", tmp)
            tmp = tmp.strip()
            filter_token = _norm_text(tmp) if tmp else qn_full

        if not filter_token:
            # Si por lo que sea nos hemos quedado sin token útil,
            # volvemos al texto normalizado completo.
            filter_token = qn_full

        seen_tables = set()
        total = 0
        details = []

        for h in hits:
            meta = h.get("meta") or {}
            if meta.get("backend") not in {"excel_row", "excel_profile"}:
                continue

            table_path = meta.get("table_cache_path")
            table_id = meta.get("table_id")
            if not table_path or not table_id or table_id in seen_tables:
                continue
            seen_tables.add(table_id)

            # Lee cache (Parquet/CSV)
            try:
                if str(table_path).lower().endswith(".parquet"):
                    df = pd.read_parquet(table_path)
                else:
                    df = pd.read_csv(table_path)
            except Exception:
                continue

            # ------------------------------------------------------------------
            # 2) Columnas candidatas de localización:
            #    buscamos patrones tipo "ubic", "loca", "sede", "deleg", "centro", "oficina"
            # ------------------------------------------------------------------
            cand_cols = [
                c
                for c in df.columns
                if any(k in str(c).lower() for k in ("ubic", "loca", "sede", "deleg", "centro", "oficina"))
            ]

            if not cand_cols:
                # Si no tenemos columnas de ubicación, esta tabla no sirve
                continue

            cnt = 0
            for c in cand_cols:
                try:
                    series = df[c].astype(str).map(_norm_text)
                    # contains(filter_token) sin regex compleja
                    cnt += int(series.str.contains(filter_token, regex=False).sum())
                except Exception:
                    pass

            if cnt:
                total += cnt
                details.append(
                    {
                        "table_id": table_id,
                        "count": int(cnt),
                        "path": table_path,
                        "location_filter": filter_token,
                        "columns_used": cand_cols,
                    }
                )

            if len(details) >= int(top_k_tables):
                break

        return {"count": int(total), "by_table": details} if (total or details) else None
    
    def _get_sparse_metrics(self):
        """
        Devuelve (SPARSE_QUEUE_SIZE, SPARSE_EMBED_DURATION, SPARSE_TASKS_PROCESSED).
        Crea métricas si no existían (modo fallback).
        """
        try:
            # Probar que están registradas
            _ = SPARSE_QUEUE_SIZE.labels(collection="__probe__")
            _ = SPARSE_EMBED_DURATION.labels(collection="__probe__")
            _ = SPARSE_TASKS_PROCESSED.labels(collection="__probe__")
            return SPARSE_QUEUE_SIZE, SPARSE_EMBED_DURATION, SPARSE_TASKS_PROCESSED
        except Exception:
            # Fallback: crear métricas locales si no están
            from prometheus_client import Gauge, Histogram, Counter
            try:
                globals()['SPARSE_QUEUE_SIZE']
                globals()['SPARSE_EMBED_DURATION']
                globals()['SPARSE_TASKS_PROCESSED']
            except KeyError:
                globals()['SPARSE_QUEUE_SIZE'] = Gauge(
                    'rag_sparse_queue_size', 'Tamaño cola SPLADE diferida', ['collection']
                )
                globals()['SPARSE_EMBED_DURATION'] = Histogram(
                    'rag_sparse_embed_seconds', 'Duración embed SPLADE', ['collection']
                )
                globals()['SPARSE_TASKS_PROCESSED'] = Counter(
                    'rag_sparse_tasks_total', 'Tareas SPLADE procesadas', ['collection']
                )
            return SPARSE_QUEUE_SIZE, SPARSE_EMBED_DURATION, SPARSE_TASKS_PROCESSED


#---------------------------- AQUÍ RAG_UTILS.PY --------------------------------------
    def _dir_has_hf_model(self, path: str) -> bool:
        return _dir_has_hf_model(path)

    def _atomic_write_file(self, path: str, data: bytes):
        _atomic_write_file(path, data)

    @staticmethod
    def _canon(s: str) -> str:
        return _canon(s)

    @staticmethod
    def _canon_key(s: str) -> str:
        return _canon_key(s)

    @staticmethod
    def _canon_val(s: str) -> str:
        return _canon_val(s)

    @staticmethod
    def _parse_number_es(s: str) -> Optional[float]:
        return _parse_number_es(s)
    
    #-------------------------------------------A PARTIR DE AQUÍ, RAG_QDRANT---------------------------------
    def build_vector_db(self, documents, doc_names, index_filepath):
        dense_dim = int(self.sbert_model.get_sentence_embedding_dimension())
        create_collection_if_not_exists(
            self.qdrant_client,
            self.collection_name,
            dense_dim,
        )
        self.upsert_documents_to_db(documents, doc_names)

    def _create_qdrant_collection(self, collection_name: Optional[str] = None):
        name = collection_name or self.collection_name
        dim = int(self.sbert_model.get_sentence_embedding_dimension())
        create_collection_if_not_exists(self.qdrant_client, name, dim)

    def _ensure_collection(self):
        dim = int(self.sbert_model.get_sentence_embedding_dimension())
        new_name = ensure_collection_compatible(self.qdrant_client, self.collection_name, dim)
        self.collection_name = new_name

#-------------------------------A PARTIR DE AQUI, RAG_LEXICAL----------------------------
    def lexical_search(self, query: str, limit: int = 100) -> List[Dict[str, object]]:
        if not getattr(self._bm25_index, "ready", False):
            self._bm25_index.build(self.documents)
        return self._bm25_index.search(query, self.documents, self.doc_names, limit)

    def build_lexical_index(self, min_df: int = None) -> None:
        self._bm25_index.build(self.documents)

    def _invalidate_bm25_if_needed(self):
        if getattr(self, "_bm25_index", None) and self._bm25_index.N != len(self.documents):
            self._bm25_index.ready = False

    #---------------------------------------A PARTIR DE AQUÍ, RAG_UPSERT------------------------------------
    def upsert_documents_to_db(self, documents: List[str], doc_names: List[str]):
        # Delegamos en rag_upsert, que usará _build_payload/_ensure_payload_indexes.
        return upsert_documents_to_db_impl(self, documents, doc_names)
    
    def delete_document_by_name(self, filename_or_path: str):
        # Delegación coherente con tu pipeline histórico
        return delete_document_by_name_impl(self, filename_or_path)
    
#----------------------------------A PARTIR DE AQUI, RAG_RERANK---------------------------------
    def _get_reranker(self) -> CrossEncoder:
            return get_reranker_impl(self)
    
    def _get_colbert_encoder(self):
        return get_colbert_encoder_impl(self)

    def _late_interaction_scores(self, query: str, texts: List[str]) -> List[float]:
        return late_interaction_scores_impl(self, query, texts)

    def rerank_results(self, query, retrieved_chunks, model=None, top_k=5, mmr_lambda=0.7):
        return rerank_results_impl(self, query, retrieved_chunks, model, top_k, mmr_lambda)

    def evaluate_results(self, results: List[Tuple], ground_truth: List[str]) -> float:
        return evaluate_results_impl(results, ground_truth)

    def log_query(self, query, results, latency):
        log_query_impl(query, results, latency, logger)

#----------------------------------------A PARTIR DE AQUÍ, RAG_LOADER------------------------------
    def clean_text(self, text):
        return clean_text_impl(text)

    def load_txt(self, path):
        return load_txt_impl(path)

    def load_docx(self, path):
        return load_docx_impl(path)

    def load_pdf(self, path):
        return load_pdf_impl(self, path)

    def load_pptx(self, path):
        return load_pptx_impl(path)

    def load_and_fragment_files(self, files):
        return load_and_fragment_files_impl(self, files)

#----------------------------------------------------------------A PARTIR DE AQUI, RAG_STORAGE------------------------------------
    def load_vector_db(self, index_filepath):
        return load_vector_db_impl(self, index_filepath)

    def save_doc_names(self, index_filepath):
        return save_doc_names_impl(self, index_filepath)

    def save_metas(self, index_filepath):
        return save_metas_impl(self, index_filepath)

    def save_documents(self, index_filepath):
        return save_documents_impl(self, index_filepath)
    
#-------------------------------------------------------------------A PARTIR DE AQUÍ, RAG_TAGS DE REDIS, NO DE QDRANT---------------------------------------
    async def get_collection_tags(self) -> List[str]:
        return await get_collection_tags_impl(self)

    async def get_matched_tags(self, query: str) -> List[str]:
        return await get_matched_tags_impl(self, query)

#--------------------------------------------A PARTIR DE AQUI, RAG_TEXTSPLIT-----------------------------------
    def dynamic_fragment_size(self, total_tokens):
        return dynamic_fragment_size_impl(total_tokens)

    def fragment_text_semantically(self, text, max_tokens=100, overlap_tokens=33):
        return fragment_text_semantically_impl(text, max_tokens, overlap_tokens)

#-----------------------------------------------------------------A PARTIR DE AQUÍ, RAG_SPARSE--------------------------------
    def _update_sparse_vectors_compat(self, point_id: str, indices: List[int], values: List[float]) -> None:
        return update_sparse_vectors_compat_impl(self, point_id, indices, values)

    def _ensure_sparse_worker(self):
        return ensure_sparse_worker_impl(self)

    def _enqueue_sparse_update(self, point_id: str, text: str):
        return enqueue_sparse_update_impl(self, point_id, text)

    def _sparse_worker_loop(self):
        return sparse_worker_loop_impl(self)

    def _get_splade_model(self):
        return get_splade_model_impl(self)

    def _sparse_encode_text(self, text: str):
        return sparse_encode_text_impl(self, text)

    def _sparse_encode_query(self, query: str, prf_topk: int = 5, prf_docs: int = 100):
        return sparse_encode_query_impl(self, query, prf_topk, prf_docs)

    def sparse_search_raw(self, query: str, limit: int = 100, qfilter: Optional[qmodels.Filter] = None):
        return sparse_search_raw_impl(self, query, limit, qfilter)


#-------------------------------------------------- A PARTIR DE AQUI, RAG_SIMILARITY ------------------------------------------
    def retrieve_similar_blocks(
        self,
        doc_name: str,
        relevant_idx: int,
        documents: List[str],
        top_k: int = 4,
        global_search: bool = True
    ) -> List[dict]:
        return retrieve_similar_blocks_impl(self, doc_name, relevant_idx, documents, top_k, global_search)

    def search_with_similar_blocks(
    self,
    query: str,
    top_k_main: int = 5,
    top_k_similars: int = 4,
    matched_tags: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[dict]:
        """
        Devuelve lista de dicts (shape estable):
        { "doc_name": ..., "text": ..., "score": ..., "meta": ..., "similar_blocks": [...] }

        Evolución compatible:
        - Si hay documento objetivo y es overview: devuelve fragmentos representativos del doc.
        - Si hay documento objetivo: devuelve fragmentos del doc como hits principales.
        - Si no: delega a search_with_similar_blocks_impl (intentando pasar filters).
        """
        from rag_similarity import search_with_similar_blocks_impl

        # 1) Detectar documento objetivo
        doc_hint = None
        if isinstance(filters, dict):
            doc_hint = filters.get("documento") or filters.get("doc_name") or filters.get("file_name")
        if not doc_hint:
            doc_hint = self._extract_document_hint_from_query(query)

        resolved = self.resolve_document_name(doc_hint) if doc_hint else None

        def _normalize_hits(hits: List[dict]) -> List[dict]:
            out = []
            for h in hits or []:
                if not isinstance(h, dict):
                    continue
                out.append({
                    "doc_name": h.get("doc_name") or (h.get("meta") or {}).get("file_name"),
                    "text": h.get("text") or "",
                    "score": float(h.get("score", 0.0) or 0.0),
                    "meta": h.get("meta") or {},
                    # ✅ shape estable aunque no calculemos similars aquí
                    "similar_blocks": h.get("similar_blocks") or [],
                })
            return out

        # 2) Overview
        if resolved and self._is_document_overview_query(query):
            hits = self._document_overview_hits(resolved, top_k=max(8, int(top_k_main)))
            return _normalize_hits(hits)[: max(1, int(top_k_main))]

        # 3) Doc concreto (no overview)
        if resolved:
            hits = self._qdrant_scroll_by_file_name(resolved, limit=max(40, int(top_k_main) * 8))
            if hits:
                return _normalize_hits(hits)[: max(1, int(top_k_main))]

        # 4) Fallback: comportamiento original, intentando propagar filters
        try:
            return search_with_similar_blocks_impl(self, query, top_k_main, top_k_similars, matched_tags, filters=filters)
        except TypeError:
            return search_with_similar_blocks_impl(self, query, top_k_main, top_k_similars, matched_tags)


#---------------------------- A PARTIR DE AQUI, RAG_SEARCH ----------------
    def vector_search_raw(self, query: str, limit: int = 100, qfilter: Optional[qmodels.Filter] = None) -> List[Dict[str, object]]:
        return vector_search_raw_impl(self, query, limit, qfilter)

    def hybrid_search_rrf(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
        alpha_vector: float = 1.0,
        alpha_sparse: float = 1.0,
        alpha_lexical: float = 1.0,
        qfilter: Optional[qmodels.Filter] = None
    ) -> List[Tuple[str, str, float, float]]:
        return hybrid_search_rrf_impl(self, query, top_k, rrf_k, alpha_vector, alpha_sparse, alpha_lexical, qfilter)

    def filter_by_score_threshold(self, hits, ratio=0.3):
        return filter_by_score_threshold_impl(self, hits, ratio)

    def search(
    self,
    query: str,
    top_k: int = 5,
    matched_tags: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str, float, float]]:
        """
        Devuelve tuplas: (text, doc_name, rrf_score, rerank_score)

        Evolución compatible y robusta:
        - Si se detecta documento objetivo (filters o texto), resuelve el doc:
            - si es overview: devuelve fragmentos representativos del documento (scroll)
            - si no: devuelve fragmentos del documento (scroll)
        *PERO* solo activamos el "doc-only shortcut" cuando:
            - el filtro es únicamente "documento" (o no hay otros filtros relevantes)
            - y/o la query es claramente overview
        Si hay filtros adicionales (location/brand/etc.), delegamos en search_impl
        para que el filtrado se aplique realmente en Qdrant.
        - Si no hay referencia a documento, delega en search_impl (pasando filters si lo soporta).
        """

        # --- saneo de inputs ---
        q = (query or "").strip()
        if not q:
            return []

        try:
            k = int(top_k) if top_k is not None else 5
        except Exception:
            k = 5
        k = max(1, min(k, 50))  # hard cap razonable

        fdict: Dict[str, Any] = filters if isinstance(filters, dict) else {}

        # --- 1) detectar documento objetivo desde filters o desde texto ---
        doc_hint = None
        if fdict:
            doc_hint = (
                fdict.get("documento")
                or fdict.get("doc_name")
                or fdict.get("file_name")
            )
            if isinstance(doc_hint, str):
                doc_hint = doc_hint.strip() or None
            else:
                doc_hint = None

        if not doc_hint:
            try:
                doc_hint = self._extract_document_hint_from_query(q)
            except Exception:
                doc_hint = None

        # Resolver doc_hint a nombre real (según metas/doc_names)
        resolved = None
        if doc_hint:
            try:
                resolved = self.resolve_document_name(doc_hint)
            except Exception:
                resolved = None

        # --- 2) decidir si hacemos el "shortcut" por documento ---
        # Solo es seguro si:
        #   - es overview, o
        #   - filters no aporta restricciones adicionales más allá de documento
        def _filters_only_document(f: Dict[str, Any]) -> bool:
            if not f:
                return True
            # claves equivalentes a documento
            doc_keys = {"documento", "doc_name", "file_name"}
            for kk, vv in f.items():
                if vv is None:
                    continue
                kkl = str(kk).strip().lower()
                if kkl in doc_keys:
                    continue
                # cualquier otra cosa activa restricciones adicionales
                # (aunque sea "", lo ignoramos)
                if isinstance(vv, str) and not vv.strip():
                    continue
                return False
            return True

        is_overview = False
        if resolved:
            try:
                is_overview = bool(self._is_document_overview_query(q))
            except Exception:
                is_overview = False

        use_doc_shortcut = bool(resolved) and (is_overview or _filters_only_document(fdict))

        # --- 3) shortcut overview: devolver fragmentos representativos ordenados ---
        if use_doc_shortcut and is_overview:
            try:
                hits = self._document_overview_hits(resolved, top_k=max(8, k))
            except Exception:
                hits = []

            out: List[Tuple[str, str, float, float]] = []
            for i, h in enumerate(hits or []):
                try:
                    txt = str((h or {}).get("text") or "").strip()
                except Exception:
                    txt = ""
                if not txt:
                    continue
                s = 1.0 - (i * 0.01)
                # ✅ ORDEN CORRECTO: (text, doc_name, rrf_score, rerank_score)
                out.append((txt, resolved, float(s), float(s)))

            return out[:k] if out else []

        # --- 4) shortcut doc concreto (no overview): devolver chunks del doc ---
        if use_doc_shortcut and resolved:
            try:
                hits = self._qdrant_scroll_by_file_name(resolved, limit=max(40, k * 8))
            except Exception:
                hits = []

            out: List[Tuple[str, str, float, float]] = []
            for i, h in enumerate((hits or [])[: max(1, k)]):
                try:
                    txt = str((h or {}).get("text") or "").strip()
                except Exception:
                    txt = ""
                if not txt:
                    continue
                s = 1.0 - (i * 0.01)
                out.append((txt, resolved, float(s), float(s)))

            if out:
                return out[:k]

        # --- 5) fallback: motor SOTA (Qdrant + filtros + híbrido + rerank + agg) ---
        # Pasamos filters si el impl lo soporta; si no, degradamos.
        try:
            return search_impl(self, q, k, matched_tags, filters=fdict or None)
        except TypeError:
            return search_impl(self, q, k, matched_tags)
        except Exception:
            # nunca romper el endpoint por search_impl
            return []
        

    def _get_user_lock(self):
        """
        Lock reentrante por usuario para proteger operaciones críticas (upserts/pickles/cachés).
        Debe ser GLOBAL por proceso, no por instancia.
        """
        import threading

        g = globals()
        if "_USER_LOCKS" not in g:
            g["_USER_LOCKS"] = {}
        if "_USER_LOCKS_GUARD" not in g:
            g["_USER_LOCKS_GUARD"] = threading.Lock()

        with g["_USER_LOCKS_GUARD"]:
            lk = g["_USER_LOCKS"].get(self.user_id)
            if lk is None:
                lk = threading.RLock()
                g["_USER_LOCKS"][self.user_id] = lk
            return lk


    def _prep_query(self, s: str) -> str:
        if not self._use_e5_prefix:
            return s or ""
        return f"{E5_QUERY_PREFIX}{s or ''}"

    def _prep_doc(self, s: str) -> str:
        if not self._use_e5_prefix:
            return s or ""
        return f"{E5_DOC_PREFIX}{s or ''}"

    def _detect_header_row(self, rows: List[List[str]]) -> Optional[int]:
        for idx, row in enumerate(rows):
            cells = [str(c or "").strip() for c in row]
            non_empty = [c for c in cells if c]
            if not non_empty:
                continue
            with_letters = sum(any(ch.isalpha() for ch in c) for c in non_empty)
            if len(non_empty) > 0 and (with_letters / len(non_empty)) >= 0.5:
                return idx
        return None

    def _standard_field_aliases(self) -> Dict[str, str]:
        """
        Aliases de cabeceras a un esquema canónico para consultas robustas.
        Claves en minúscula y sin tildes (usa self._canon/_canon_key en comparaciones).
        """
        return {
            "clase": "class",
            "tipo": "class",
            "marca": "brand",
            "modelo": "model",
            "número de serie": "serial",
            "numero de serie": "serial",
            "número_de_serie": "serial",
            "num de serie": "serial",
            "ubicacion": "location",
            "ubicación": "location",
            "sede": "location",
            "delegacion": "location",
            "delegación": "location",
            "centro": "location",
            "oficina": "location",
            "emisora": "location",
            "inventario": "asset_id",
            "id": "asset_id",
            "cantidad": "quantity",
            "uds": "quantity",
            "unidades": "quantity",
            "precio": "price",
            "importe": "price",
            "coste": "price",
            "valor": "price",
            "total": "price",
        }


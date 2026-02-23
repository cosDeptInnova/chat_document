# chatdoc_indexer.py

"""
Índice efímero para chatear con UN documento.

Objetivos de este módulo:
- Extraer texto y fragmentos de archivos usando rag_loader (Docling / MinerU / etc).
- Construir un índice en memoria (chunks + embeddings) sin persistencia externa.
- Serializar a JSON para almacenamiento efímero en Redis.
- Soportar búsqueda semántica con SentenceTransformers.

Requisitos específicos implementados:
1) Modelo de embeddings:
   - Se descarga la primera vez en LOCAL_EMBED_DIR si el directorio NO existe.
   - Si LOCAL_EMBED_DIR existe, NO se vuelve a descargar nunca por defecto.
   - Si el directorio existe pero no parece un modelo SentenceTransformers válido,
     se hace fallback a HF en runtime y se loguea cómo corregirlo.
   - Reparación opcional: CHATDOC_EMBED_REPAIR=1 permite redescargar/normalizar.

2) Robustez y trazas:
   - Logging de shape de outputs del loader.
   - Logs de metadatos de chunks y page_count.
   - Embedding dim dinámico.

Variables de entorno:
- LOCAL_EMBED_DIR
- HF_EMBED_MODEL
- HF_EMBED_BATCH
- CHATDOC_USE_E5_PREFIX: "1"/"0" (override)
- CHATDOC_EMBED_REPAIR: "1"/"0"
"""

import os
import logging
import traceback
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import threading
import torch
import re
from typing import Tuple
from contextvars import ContextVar

from rag_loader import (
    clean_text_impl,
    load_txt_impl,
    load_docx_impl,
    load_pdf_impl,
    load_pptx_impl,
    load_and_fragment_files_impl,
)
from rag_textsplit import (
    dynamic_fragment_size_impl,
    fragment_text_semantically_impl,
)
from rag_utils import (
    _atomic_write_file,
    _canon,
    _canon_key,
    _canon_val,
    _parse_number_es,
)
from device_manager import gpu_manager



# Logging
logger = logging.getLogger("chatdoc_indexer")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Embeddings config
LOCAL_EMBED_DIR = os.getenv(
    "LOCAL_EMBED_DIR",
    r"C:\Users\ADM-mcabo\Documents\cosmos_cache\st-e5",
)
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-large")
HF_EMBED_BATCH = int(os.getenv("HF_EMBED_BATCH", "32"))

E5_QUERY_PREFIX = "query: "
E5_DOC_PREFIX = "passage: "

_EMBED_INIT_LOCK = threading.RLock()
_GLOBAL_EMBED_MODEL: Optional[SentenceTransformer] = None
_GLOBAL_EMBED_DEVICE: Optional[torch.device] = None
_EMBED_WORKLOAD_CTX: ContextVar[dict] = ContextVar("chatdoc_embed_workload", default={})

CHATDOC_EMBED_FORCE_DEVICE = os.getenv("CHATDOC_EMBED_FORCE_DEVICE", "").strip()  # "cuda:0" | "cpu" | ""
CHATDOC_EMBED_STICKY_DEVICE = os.getenv("CHATDOC_EMBED_STICKY_DEVICE", "1").strip() == "1"
CHATDOC_CUDA_EMPTY_CACHE = os.getenv("CHATDOC_CUDA_EMPTY_CACHE", "0").strip() == "1"
CHATDOC_EMBED_RELEASE_AFTER_CALL = os.getenv("CHATDOC_EMBED_RELEASE_AFTER_CALL", "0").strip() == "1"
CHATDOC_EMBED_OOM_FALLBACK_CPU = os.getenv("CHATDOC_EMBED_OOM_FALLBACK_CPU", "0").strip() == "1"
CHATDOC_EMBED_STRICT_GPU = os.getenv("CHATDOC_EMBED_STRICT_GPU", "1").strip() == "1"

# Override de prefijos E5
_ENV_E5 = os.getenv("CHATDOC_USE_E5_PREFIX", "").strip().lower()
if _ENV_E5 in ("0", "false", "no"):
    USE_E5_PREFIX = False
elif _ENV_E5 in ("1", "true", "yes"):
    USE_E5_PREFIX = True
else:
    USE_E5_PREFIX = "e5" in (HF_EMBED_MODEL or "").lower()

# Reparación explícita del modelo local
EMBED_REPAIR = os.getenv("CHATDOC_EMBED_REPAIR", "0").strip() == "1"

# --- Semáforo global para embeddings (RAG) ---
try:
    _RAG_EMBED_CONCURRENCY = int(os.getenv("RAG_EMBED_CONCURRENCY", "2"))
except Exception:
    _RAG_EMBED_CONCURRENCY = 2

_RAG_EMBED_CONCURRENCY = max(1, _RAG_EMBED_CONCURRENCY)
_RAG_EMBED_GPU_SEMAPHORE = threading.Semaphore(_RAG_EMBED_CONCURRENCY)
_RE_HEADING = re.compile(
    r"(?m)^(?:\s*(?:\d+(?:\.\d+){0,7})\s+.+|\s*[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ0-9 \t\-\–\—,.;:/()]{6,})\s*$"
)
_RE_BULLET = re.compile(r"^\s*(?:[-*•]|(\d+[\.\)])|([a-zA-Z][\.\)]))\s+")
_RE_MANY_SPACES_COLS = re.compile(r"\S+\s{2,}\S+")
_RE_HAS_DIGIT = re.compile(r"\d")


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("cuda out of memory" in msg) or ("cublas" in msg and "alloc" in msg) or ("unable to allocate" in msg and "cuda" in msg)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _norm_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _looks_like_table(s: str) -> bool:
    if not s:
        return False
    lines = [ln for ln in s.split("\n") if ln.strip()]
    if len(lines) < 6:
        return False

    pipe_like = sum(1 for ln in lines if ln.count("|") >= 2)
    space_cols = sum(1 for ln in lines if _RE_MANY_SPACES_COLS.search(ln) is not None)
    digit_lines = sum(1 for ln in lines if _RE_HAS_DIGIT.search(ln) is not None)

    col_struct = (pipe_like + space_cols) / max(1, len(lines))
    digit_ratio = digit_lines / max(1, len(lines))
    return (col_struct >= 0.35 and digit_ratio >= 0.35)

def _looks_like_list(s: str) -> bool:
    if not s:
        return False
    lines = [ln for ln in s.split("\n") if ln.strip()]
    if len(lines) < 6:
        return False
    bullet_lines = sum(1 for ln in lines if _RE_BULLET.search(ln) is not None)
    return (bullet_lines / max(1, len(lines))) >= 0.45

def _split_by_paragraphs(s: str, max_chars: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n+", s) if p.strip()]
    if not paras:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        txt = "\n\n".join(buf).strip()
        if txt:
            chunks.append(txt)
        buf = []
        buf_len = 0

    for p in paras:
        if not buf:
            buf = [p]
            buf_len = len(p)
            continue
        if (buf_len + 2 + len(p)) <= max_chars:
            buf.append(p)
            buf_len += 2 + len(p)
        else:
            flush()
            buf = [p]
            buf_len = len(p)

    flush()
    return chunks

def _split_table(s: str, *, max_chars: int, max_lines: int, header_max_lines: int) -> list[str]:
    lines = [ln.rstrip() for ln in s.split("\n") if ln.strip()]
    if not lines:
        return []

    header = lines[:header_max_lines]
    rows = lines[header_max_lines:] if len(lines) > header_max_lines else []

    out: list[str] = []
    buf: list[str] = []
    buf_chars = 0

    def flush():
        nonlocal buf, buf_chars
        if not buf:
            return
        chunk = "\n".join((header + buf) if header else buf).strip()
        if chunk:
            out.append(chunk)
        buf = []
        buf_chars = 0

    src = rows if rows else lines
    for ln in src:
        ln2 = ln.strip()
        if not ln2:
            continue
        if buf and (len(buf) >= max_lines or (buf_chars + len(ln2) + 1) > max_chars):
            flush()
        buf.append(ln2)
        buf_chars += len(ln2) + 1

    flush()
    return out or ([s[:max_chars].strip()] if s.strip() else [])

def _split_list(s: str, *, max_chars: int, max_items: int = 18) -> list[str]:
    lines = [ln.rstrip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return []

    # “intro” = primeras líneas antes del primer bullet, si existen
    first_bullet_idx = None
    for i, ln in enumerate(lines):
        if _RE_BULLET.search(ln):
            first_bullet_idx = i
            break

    intro = []
    items = lines
    if first_bullet_idx is not None and first_bullet_idx > 0:
        intro = lines[:first_bullet_idx]
        items = lines[first_bullet_idx:]

    out: list[str] = []
    buf: list[str] = []
    buf_chars = 0
    buf_items = 0

    def flush():
        nonlocal buf, buf_chars, buf_items
        if not buf:
            return
        body = "\n".join(buf).strip()
        if intro:
            body = ("\n".join(intro).strip() + "\n" + body).strip()
        if body:
            out.append(body)
        buf = []
        buf_chars = 0
        buf_items = 0

    for ln in items:
        ln2 = ln.strip()
        if not ln2:
            continue

        is_item = _RE_BULLET.search(ln2) is not None
        if buf and (buf_items >= max_items or (buf_chars + len(ln2) + 1) > max_chars):
            flush()

        buf.append(ln2)
        buf_chars += len(ln2) + 1
        if is_item:
            buf_items += 1

    flush()
    return out or ([s[:max_chars].strip()] if s.strip() else [])

def _split_by_headings(s: str) -> list[Tuple[str, str]]:
    """
    Devuelve lista de (heading, body). heading puede ser "" si no se detecta.
    """
    s = _norm_text(s)
    if not s:
        return []

    lines = s.split("\n")
    blocks: list[Tuple[str, list[str]]] = []
    cur_heading = ""
    cur_lines: list[str] = []

    def push():
        nonlocal cur_heading, cur_lines
        body = "\n".join(cur_lines).strip()
        if body:
            blocks.append((cur_heading.strip(), body))
        cur_heading = ""
        cur_lines = []

    for ln in lines:
        ln2 = ln.strip()
        if ln2 and _RE_HEADING.match(ln2) and len(ln2) <= 160:
            push()
            cur_heading = ln2
        else:
            cur_lines.append(ln)

    push()
    return blocks or [("", s)]

def _apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or len(chunks) <= 1:
        return [c.strip() for c in chunks if c and c.strip()]

    out = [chunks[0].strip()]
    for i in range(1, len(chunks)):
        prev = out[-1]
        tail = prev[-overlap_chars:].strip()
        cur = chunks[i].strip()
        if tail and cur and tail not in cur[: (overlap_chars + 40)]:
            out.append((tail + "\n\n" + cur).strip())
        else:
            out.append(cur)
    return out

def _fragment_advanced(
    text: str,
    *,
    max_chars: int,
    min_chars: int,
    overlap_chars: int,
    table_max_lines: int,
    table_header_max_lines: int,
) -> list[Tuple[str, dict]]:
    """
    Fragmentación avanzada:
    - respeta headings
    - detecta tablas/listas y las trocea con reglas específicas
    - para narrativa usa párrafos + overlap
    Devuelve [(chunk_text, extra_meta), ...]
    """
    t = _norm_text(text)
    if not t:
        return []

    if len(t) <= max_chars:
        kind = "table" if _looks_like_table(t) else ("list" if _looks_like_list(t) else "text")
        return [(t, {"chunk_kind": kind})]

    out: list[Tuple[str, dict]] = []

    for heading, body in _split_by_headings(t):
        body2 = body.strip()
        if not body2:
            continue

        # 1) tablas
        if _looks_like_table(body2):
            parts = _split_table(
                (heading + "\n" + body2).strip() if heading else body2,
                max_chars=max_chars,
                max_lines=table_max_lines,
                header_max_lines=table_header_max_lines,
            )
            for p in parts:
                out.append((p.strip(), {"chunk_kind": "table", "heading": heading or None}))
            continue

        # 2) listas
        if _looks_like_list(body2):
            parts = _split_list(
                (heading + "\n" + body2).strip() if heading else body2,
                max_chars=max_chars,
            )
            for p in parts:
                out.append((p.strip(), {"chunk_kind": "list", "heading": heading or None}))
            continue

        # 3) narrativa
        base = (heading + "\n" + body2).strip() if heading else body2
        parts = _split_by_paragraphs(base, max_chars=max_chars)
        parts = _apply_overlap(parts, overlap_chars)
        for p in parts:
            out.append((p.strip(), {"chunk_kind": "text", "heading": heading or None}))

    # merge de mini-chunks < min_chars (solo para text/list; tablas mejor no mezclar)
    merged: list[Tuple[str, dict]] = []
    buf_txt = ""
    buf_meta: dict = {"chunk_kind": "text"}

    def flush_buf():
        nonlocal buf_txt, buf_meta
        if buf_txt.strip():
            merged.append((buf_txt.strip(), dict(buf_meta)))
        buf_txt = ""
        buf_meta = {"chunk_kind": "text"}

    for chunk_txt, meta in out:
        if not chunk_txt:
            continue
        kind = (meta or {}).get("chunk_kind") or "text"

        if len(chunk_txt) >= min_chars or kind == "table":
            flush_buf()
            merged.append((chunk_txt, meta))
            continue

        # acumular pequeños
        if not buf_txt:
            buf_txt = chunk_txt
            buf_meta = dict(meta or {})
        else:
            buf_txt = (buf_txt + "\n\n" + chunk_txt).strip()

        if len(buf_txt) >= min_chars:
            flush_buf()

    flush_buf()
    return merged

def _get_embed_singleton() -> SentenceTransformer:
    """
    Singleton global del embedder.

    Producción:
    - Inicializa SIEMPRE en CPU (cero VRAM al arrancar).
    - El 'device preferente' lo decide _get_embed_device() (cpu/gpu/auto/force).
    """
    global _GLOBAL_EMBED_MODEL, _GLOBAL_EMBED_DEVICE

    if _GLOBAL_EMBED_MODEL is not None:
        return _GLOBAL_EMBED_MODEL

    with _EMBED_INIT_LOCK:
        if _GLOBAL_EMBED_MODEL is not None:
            return _GLOBAL_EMBED_MODEL

        local_path = _ensure_or_warn_local_model()
        model_id_or_path = local_path or HF_EMBED_MODEL

        logger.info("[chatdoc/embed] init SentenceTransformer='%s' (cpu init)", model_id_or_path)

        try:
            _GLOBAL_EMBED_MODEL = SentenceTransformer(model_id_or_path, device="cpu")
        except TypeError:
            _GLOBAL_EMBED_MODEL = SentenceTransformer(model_id_or_path)
            try:
                _GLOBAL_EMBED_MODEL = _GLOBAL_EMBED_MODEL.to("cpu")
            except Exception:
                pass

        # prefer_device (cpu/gpu/auto/force)
        _GLOBAL_EMBED_DEVICE = _get_embed_device()
        logger.info("[chatdoc/embed] prefer_device=%s", _GLOBAL_EMBED_DEVICE)

        return _GLOBAL_EMBED_MODEL


# Helpers de validación y cache del modelo
def _is_valid_st_model_dir(path: str) -> bool:
    """
    Heurística barata para detectar si una carpeta parece un modelo
    SentenceTransformers válido.
    """
    try:
        p = Path(path)
        if not p.exists() or not p.is_dir():
            return False

        # Señales típicas
        if (p / "modules.json").exists():
            return True
        if (p / "config.json").exists():
            return True
        if (p / "sentence_bert_config.json").exists():
            return True

        # Si hay subcarpetas típicas de ST
        if any((p / sub).exists() for sub in ("0_Transformer", "1_Pooling", "2_Dense")):
            return True

        return False
    except Exception:
        return False

def _acquire_best_effort_lock(lock_path: Path) -> bool:
    """
    Lock best-effort sin dependencias.
    Devuelve True si se adquiere, False si ya existe.
    """
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception:
        return False

def _release_best_effort_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass

def _download_and_save_model(target_dir: Path) -> bool:
    """
    Descarga el modelo HF y lo guarda en target_dir de forma segura.

    Producción:
    - Inicializa SIEMPRE en CPU (evita VRAM durante descarga).
    - Guarda en tmp dentro del MISMO parent.
    - Swap robusto: backup del target actual + move tmp -> target.
    - Limpieza best-effort.
    """
    import shutil

    target_dir = Path(target_dir)

    try:
        logger.info("[chatdoc/embed] Descargando modelo HF: %s (cpu)", HF_EMBED_MODEL)

        # CPU init siempre
        try:
            model = SentenceTransformer(HF_EMBED_MODEL, device="cpu")
        except TypeError:
            model = SentenceTransformer(HF_EMBED_MODEL)
            try:
                model = model.to("cpu")
            except Exception:
                pass

        # tmp dentro del mismo parent => move “barato”
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        pid = os.getpid()

        tmp_dir = target_dir.parent / f"{target_dir.name}.tmp-{pid}-{ts}"
        bak_dir = target_dir.parent / f"{target_dir.name}.bak-{pid}-{ts}"

        # Limpieza tmp residual
        if tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Guardar en tmp
        model.save(str(tmp_dir))

        # Liberar memoria del objeto cuanto antes
        try:
            del model
        except Exception:
            pass

        ok_tmp = _is_valid_st_model_dir(str(tmp_dir))
        if not ok_tmp:
            logger.warning("[chatdoc/embed] Descarga guardada en %s pero no parece ST válido.", str(tmp_dir))

        # Backup del target actual (si existía)
        if target_dir.exists():
            try:
                if bak_dir.exists():
                    shutil.rmtree(bak_dir, ignore_errors=True)
                shutil.move(str(target_dir), str(bak_dir))
            except Exception:
                # Si no podemos moverlo a backup, intentamos borrarlo para permitir el swap
                try:
                    shutil.rmtree(target_dir, ignore_errors=True)
                except Exception:
                    pass

        # Swap: tmp -> target
        try:
            shutil.move(str(tmp_dir), str(target_dir))
        except Exception as e:
            logger.warning("[chatdoc/embed] No se pudo mover tmp->target: %s", e)
            # Intentar rollback desde backup si existe
            try:
                if (not target_dir.exists()) and bak_dir.exists():
                    shutil.move(str(bak_dir), str(target_dir))
            except Exception:
                pass
            return False

        # Validar final
        ok = _is_valid_st_model_dir(str(target_dir))
        if ok:
            logger.info("[chatdoc/embed] Modelo guardado correctamente en %s", str(target_dir))
        else:
            logger.warning("[chatdoc/embed] Guardado en %s pero no parece ST válido.", str(target_dir))

        # Limpieza backup best-effort
        try:
            if bak_dir.exists() and ok:
                shutil.rmtree(bak_dir, ignore_errors=True)
        except Exception:
            pass

        return ok

    except Exception as e:
        logger.warning("[chatdoc/embed] Fallo descargando/guardando modelo: %s", e)
        logger.debug(traceback.format_exc())
        return False


def _ensure_or_warn_local_model() -> Optional[str]:
    """
    Política:
    - Si LOCAL_EMBED_DIR NO existe => descargar primera vez y guardar (con lock).
    - Si LOCAL_EMBED_DIR existe => NO redescargar por defecto.
        * Si es válido => ok.
        * Si NO es válido => warning y no tocar disco (salvo EMBED_REPAIR=1).
    - Si EMBED_REPAIR=1 => reintenta reparación (con lock).
    - Si detecta lock activo de otro proceso => espera y revalida (evita estado corrupto).
    """
    target = Path(LOCAL_EMBED_DIR)

    wait_s = float(os.getenv("CHATDOC_EMBED_DOWNLOAD_WAIT_S", "300"))
    poll_s = float(os.getenv("CHATDOC_EMBED_DOWNLOAD_POLL_S", "2.0"))

    def _wait_for_unlock(lock_path: Path) -> None:
        deadline = time.time() + max(1.0, wait_s)
        while lock_path.exists() and time.time() < deadline:
            time.sleep(max(0.2, poll_s))

    existed_before = target.exists()

    # Si no existía, intentamos crear parent y luego descargar “primera vez”.
    # OJO: no creamos un dir vacío y luego lo tratamos como “corrupto”.
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    lock_path = target / ".download.lock"

    # Si otro proceso está descargando/reparando, esperamos y revalidamos
    if lock_path.exists():
        logger.info("[chatdoc/embed] Lock detectado. Esperando: %s", str(lock_path))
        _wait_for_unlock(lock_path)
        if _is_valid_st_model_dir(str(target)):
            return str(target)

    # Si ya es válido, listo
    if _is_valid_st_model_dir(str(target)):
        return str(target)

    # ---- Caso 1: primera vez (dir no existía) => descargamos sí o sí ----
    if not existed_before:
        have_lock = _acquire_best_effort_lock(lock_path)
        if not have_lock:
            logger.info("[chatdoc/embed] Otro proceso descargando (primera vez). Esperando...")
            _wait_for_unlock(lock_path)
            return str(target) if _is_valid_st_model_dir(str(target)) else None

        try:
            logger.info("[chatdoc/embed] Primera descarga en %s", str(target))
            # Asegura dir destino existe (vacío o no)
            try:
                target.mkdir(parents=True, exist_ok=True)
            except Exception:
                return None
            ok = _download_and_save_model(target)
            return str(target) if ok else None
        finally:
            _release_best_effort_lock(lock_path)

    # ---- Caso 2: existe pero inválido ----
    if not EMBED_REPAIR:
        logger.warning(
            "[chatdoc/embed] LOCAL_EMBED_DIR existe pero NO parece un modelo ST válido: %s. "
            "Por política no se redescarga automáticamente. "
            "Solución: borra la carpeta o activa CHATDOC_EMBED_REPAIR=1.",
            str(target),
        )
        return None

    # ---- Caso 3: reparación explícita ----
    have_lock = _acquire_best_effort_lock(lock_path)
    if not have_lock:
        logger.info("[chatdoc/embed] Reparación en curso por otro proceso. Esperando...")
        _wait_for_unlock(lock_path)
        return str(target) if _is_valid_st_model_dir(str(target)) else None

    try:
        if _is_valid_st_model_dir(str(target)):
            return str(target)

        logger.info("[chatdoc/embed] Reparación activada. Re-descargando en %s", str(target))
        ok = _download_and_save_model(target)
        return str(target) if ok else None
    finally:
        _release_best_effort_lock(lock_path)

def _get_embed_device() -> torch.device:
    """
    Política de device para embeddings (robusta):
    - Respeta CHATDOC_EMBED_FORCE_DEVICE si está.
    - Añade CHATDOC_EMBED_DEVICE_POLICY: 'cpu' | 'gpu' | 'auto' (default gpu)
    - En auto: usa GPU sólo si hay suficiente VRAM libre; si no, CPU.
    - Actualiza _GLOBAL_EMBED_DEVICE para mantener consistencia.
    """
    global _GLOBAL_EMBED_DEVICE

    forced = os.getenv("CHATDOC_EMBED_FORCE_DEVICE", "").strip()
    if forced:
        try:
            d = torch.device(forced)
            _GLOBAL_EMBED_DEVICE = d
            return d
        except Exception:
            _GLOBAL_EMBED_DEVICE = torch.device("cpu")
            return _GLOBAL_EMBED_DEVICE

    policy = os.getenv("CHATDOC_EMBED_DEVICE_POLICY", "gpu").strip().lower()
    if policy in ("cpu", "force_cpu", "only_cpu"):
        _GLOBAL_EMBED_DEVICE = torch.device("cpu")
        return _GLOBAL_EMBED_DEVICE

    if not torch.cuda.is_available():
        if CHATDOC_EMBED_STRICT_GPU and policy in ("gpu", "cuda", "force_gpu", "only_gpu"):
            raise RuntimeError(
                "[chatdoc/embed] CHATDOC_EMBED_STRICT_GPU=1 requiere CUDA, "
                "pero torch.cuda.is_available()=False."
            )
        _GLOBAL_EMBED_DEVICE = torch.device("cpu")
        return _GLOBAL_EMBED_DEVICE

    try:
        min_free_mb_default = int(os.getenv("HF_EMBED_MIN_FREE_MB", "2048"))
    except Exception:
        min_free_mb_default = 2048

    workload = _EMBED_WORKLOAD_CTX.get() or {}
    size_mb = float(workload.get("size_mb") or 0.0)
    hinted_pages = int(workload.get("hinted_pages") or 0)

    large_doc_size_mb = float(os.getenv("CHATDOC_LARGE_DOC_SIZE_MB", "2.0"))
    large_doc_pages = int(os.getenv("CHATDOC_LARGE_DOC_PAGES", "140"))
    is_large_doc = (size_mb >= large_doc_size_mb) or (hinted_pages >= large_doc_pages)

    try:
        min_free_mb_large = int(os.getenv("HF_EMBED_MIN_FREE_MB_LARGE_DOC", "1024"))
    except Exception:
        min_free_mb_large = 1024

    min_free_mb = min_free_mb_large if is_large_doc else min_free_mb_default

    if policy in ("gpu", "cuda", "force_gpu", "only_gpu"):
        try:
            best = gpu_manager.best_device(min_free_mb=min_free_mb)
        except Exception:
            best = torch.device("cpu")
        if CHATDOC_EMBED_STRICT_GPU and best.type != "cuda":
            raise RuntimeError(
                "[chatdoc/embed] CHATDOC_EMBED_STRICT_GPU=1 requiere GPU para embeddings, "
                f"pero no se encontró dispositivo CUDA elegible (min_free_mb={min_free_mb})."
            )
        _GLOBAL_EMBED_DEVICE = best
        return best

    # auto
    cur = _GLOBAL_EMBED_DEVICE or torch.device("cpu")
    if cur.type == "cuda":
        try:
            free_bytes, _ = torch.cuda.mem_get_info(int(cur.index))
            if int(free_bytes / (1024 ** 2)) >= min_free_mb:
                return cur
        except Exception:
            pass

    try:
        best = gpu_manager.best_device(min_free_mb=min_free_mb)
    except Exception:
        best = torch.device("cpu")

    if best.type == "cpu" and is_large_doc:
        # Documento grande: degradar requisito de VRAM para intentar usar GPU si existe.
        try:
            very_low_mb = int(os.getenv("HF_EMBED_MIN_FREE_MB_FLOOR", "256"))
        except Exception:
            very_low_mb = 256
        try:
            best = gpu_manager.best_device(min_free_mb=max(128, very_low_mb))
        except Exception:
            pass

    _GLOBAL_EMBED_DEVICE = best
    return best


def set_embed_workload_context(*, size_mb: float, hinted_pages: int = 0) -> None:
    _EMBED_WORKLOAD_CTX.set(
        {
            "size_mb": float(size_mb or 0.0),
            "hinted_pages": int(hinted_pages or 0),
        }
    )


def clear_embed_workload_context() -> None:
    _EMBED_WORKLOAD_CTX.set({})


def _embedding_dim() -> int:
    """
    Dimensión del embedding usando el MISMO singleton que encode.

    Mejoras:
    - Cachea la dimensión para evitar llamadas repetidas a encode('ping') bajo carga.
    - Thread-safe.
    """
    # Cache global lazy (no requiere tocar globals arriba)
    if "_CHATDOC_EMBED_DIM_CACHE" not in globals():
        globals()["_CHATDOC_EMBED_DIM_CACHE"] = {"value": None}

    cache = globals()["_CHATDOC_EMBED_DIM_CACHE"]

    v = cache.get("value")
    if isinstance(v, int) and v > 0:
        return v

    with _EMBED_INIT_LOCK:
        v2 = cache.get("value")
        if isinstance(v2, int) and v2 > 0:
            return v2

        try:
            model = _get_embed_singleton()
            dim_fn = getattr(model, "get_sentence_embedding_dimension", None)
            if callable(dim_fn):
                d = int(dim_fn())
                if d > 0:
                    cache["value"] = d
                    return d
        except Exception:
            pass

        # Fallback barato (CPU)
        try:
            model = _get_embed_singleton()
            arr = model.encode(
                ["ping"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=1,
                device="cpu",
                show_progress_bar=False,
            )
            if hasattr(arr, "shape") and getattr(arr, "ndim", 0) == 2 and arr.shape[1] > 0:
                d = int(arr.shape[1])
                cache["value"] = d
                return d
        except Exception:
            pass

        cache["value"] = 0
        return 0

# ---------------------- Helper que usa rag_loader ----------------------
class EphemeralRagLoader:
    """
    Objeto mínimo que expone las mismas funciones que espera rag_loader_impl,
    pero SIN Qdrant ni nada persistente. Sólo sirve para:
      - Cargar el archivo.
      - Extraer texto / fragmentos con Docling / MinerU / etc.
    """

    def __init__(self, language: str = "spanish"):
        self.language = language
        self._metas: List[Dict[str, Any]] = []

    # --- hooks que rag_loader_impl usa ---

    def clean_text(self, text: str) -> str:
        return clean_text_impl(text)

    def load_txt(self, path: str):
        return load_txt_impl(path)

    def load_docx(self, path: str):
        return load_docx_impl(path)

    def load_pdf(self, path: str):
        t0 = time.time()
        try:
            out = load_pdf_impl(self, path)
            logger.info("[chatdoc/loader] load_pdf ok path=%s dt=%.3fs", path, time.time() - t0)
            return out
        except Exception as e:
            logger.warning("[chatdoc/loader] load_pdf FAIL path=%s err=%s", path, e)
            logger.debug(traceback.format_exc())
            raise

    def load_pptx(self, path: str):
        return load_pptx_impl(path)

    def load_and_fragment_files(self, files: List[str]):
        t0 = time.time()
        try:
            res = load_and_fragment_files_impl(self, files)
            size = len(res) if isinstance(res, tuple) else None
            logger.info(
                "[chatdoc/loader] load_and_fragment_files ok files=%d tuple_size=%s dt=%.3fs",
                len(files or []), size, time.time() - t0
            )
            return res
        except Exception as e:
            logger.warning("[chatdoc/loader] load_and_fragment_files FAIL files=%s err=%s", files, e)
            logger.debug(traceback.format_exc())
            raise

    def dynamic_fragment_size(self, total_tokens: int) -> int:
        return dynamic_fragment_size_impl(total_tokens)

    def fragment_text_semantically(
        self,
        text: str,
        max_tokens: int = 100,
        overlap_tokens: int = 33,
    ):
        return fragment_text_semantically_impl(text, max_tokens, overlap_tokens)

    # ---- utilidades de rag_utils que pueden usarse desde rag_loader ----

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
    def _parse_number_es(s: str):
        return _parse_number_es(s)

    def _atomic_write_file(self, path: str, data: bytes):
        _atomic_write_file(path, data)

# ---------------------- Índice efímero de chatdoc ----------------------
@dataclass
class ChatdocChunk:
    text: str
    index: int
    char_start: int
    char_end: int
    position_ratio: float
    meta: Optional[Dict[str, Any]] = None

class DocumentChatIndex:
    """
    Índice efímero para chatear con UN documento.

    - Puede construirse:
        * Desde texto plano (full_text).
        * Desde bytes de archivo (usando rag_loader para extracción).
    - No toca Qdrant ni pickles.
    - Se serializa a JSON para guardarlo en Redis.
    """

    def __init__(
        self,
        document_id: str,
        full_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
        language: str = "spanish",
        chunks: Optional[List[ChatdocChunk]] = None,
    ):
        self.document_id = document_id
        self.full_text = full_text or ""
        self.metadata: Dict[str, Any] = metadata or {}
        self.chunk_chars = max(200, int(chunk_chars))
        self.chunk_overlap = max(0, int(chunk_overlap))
        self.language = language

        # -----------------------------
        # Estado del índice (igual que antes)
        # -----------------------------
        self._chunks: List[ChatdocChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self.page_count: int = int(self.metadata.get("page_count") or 0)

        # -----------------------------
        # Embeddings backend (mismo patrón que DocumentIndexer, sin romper nada)
        # - evita AttributeError (_use_e5_prefix)
        # - prepara attrs usados por _embed_texts() (sbert_model / embed_device)
        # - lock para evitar doble cómputo si hay concurrencia
        # -----------------------------
        self._use_e5_prefix: bool = bool(USE_E5_PREFIX)
        self.sbert_model: Optional[SentenceTransformer] = None
        self.embed_device: Optional[torch.device] = None
        self._embeddings_lock = threading.RLock()

        # -----------------------------
        # Construcción de chunks (igual que antes)
        # -----------------------------
        if chunks is not None:
            self._chunks = chunks
        else:
            self._build_chunks_simple()


    # ---------------------- construcción simple (texto plano) ----------------------

    def _ensure_embed_backend(self) -> None:
        """
        Garantiza backend de embeddings:
        - self._use_e5_prefix
        - self.sbert_model (singleton)
        - self.embed_device (preferente, con refresh si procede)
        """
        if getattr(self, "_use_e5_prefix", None) is None:
            self._use_e5_prefix = bool(USE_E5_PREFIX)

        if getattr(self, "sbert_model", None) is None:
            self.sbert_model = _get_embed_singleton()

        # embed_device:
        # - si no existe, se asigna
        # - si es CPU pero ahora hay GPU posible, se refresca
        cur = getattr(self, "embed_device", None)
        if cur is None:
            self.embed_device = _get_embed_device()
            return

        # Si hay force device, siempre manda
        forced = os.getenv("CHATDOC_EMBED_FORCE_DEVICE", "").strip()
        if forced:
            try:
                self.embed_device = torch.device(forced)
            except Exception:
                self.embed_device = torch.device("cpu")
            return

        # Refresh “suave”: si está en CPU y ahora hay GPU razonable, actualizar prefer
        if isinstance(cur, torch.device) and cur.type == "cpu" and torch.cuda.is_available():
            best = _get_embed_device()
            if best.type == "cuda":
                self.embed_device = best

    def _build_chunks_simple(self) -> None:
        """
        Chunking avanzado para texto plano (sin rag_loader):
        - respeta headings/listas/tablas
        - narrativa por párrafos
        """
        text = _norm_text(self.full_text or "")
        if not text:
            self._chunks = []
            return

        MAX_CHUNK_CHARS = _env_int("CHATDOC_MAX_CHUNK_CHARS", 1800)
        MIN_CHUNK_CHARS = _env_int("CHATDOC_MIN_CHUNK_CHARS", 250)
        OVERLAP_CHARS = _env_int("CHATDOC_CHUNK_OVERLAP_CHARS", 160)

        TABLE_MAX_LINES = _env_int("CHATDOC_TABLE_MAX_LINES_PER_CHUNK", 28)
        TABLE_HEADER_MAX_LINES = _env_int("CHATDOC_TABLE_HEADER_MAX_LINES", 3)

        pieces = _fragment_advanced(
            text,
            max_chars=MAX_CHUNK_CHARS,
            min_chars=MIN_CHUNK_CHARS,
            overlap_chars=OVERLAP_CHARS,
            table_max_lines=TABLE_MAX_LINES,
            table_header_max_lines=TABLE_HEADER_MAX_LINES,
        )

        chunks: List[ChatdocChunk] = []
        # construimos full_text alineado a piezas para offsets consistentes
        rebuilt = "\n\n".join([p[0] for p in pieces])
        self.full_text = rebuilt

        total_len = len(self.full_text)
        sep_len = len("\n\n")

        offset = 0
        for idx, (frag, extra_meta) in enumerate(pieces):
            start = offset
            end = start + len(frag)
            pos_ratio = (start / total_len) if total_len > 0 else 0.0
            chunks.append(
                ChatdocChunk(
                    text=frag,
                    index=idx,
                    char_start=start,
                    char_end=end,
                    position_ratio=float(pos_ratio),
                    meta=extra_meta or None,
                )
            )
            offset = end + sep_len

        self._chunks = chunks
        logger.info(
            "[chatdoc] Construido índice texto->chunks advanced doc=%s chunks=%d max_chars=%d overlap=%d",
            self.document_id, len(self._chunks), MAX_CHUNK_CHARS, OVERLAP_CHARS
        )

    # ---------------------- construcción desde bytes de archivo ----------------------
    @classmethod
    def from_bytes(
        cls,
        document_id: str,
        raw_bytes: bytes,
        filename: str,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        language: str = "spanish",
    ) -> "DocumentChatIndex":
        import tempfile

        t0 = time.time()

        if raw_bytes is None or len(raw_bytes) == 0:
            logger.warning("[chatdoc/from_bytes] raw_bytes vacío doc_id=%s filename=%s", document_id, filename)
            return cls(
                document_id=document_id,
                full_text="",
                metadata=dict(metadata or {}),
                chunk_chars=1200,
                chunk_overlap=200,
                language=language,
            )

        # ---- tuning por entorno (sin romper nada) ----
        MAX_CHUNK_CHARS = _env_int("CHATDOC_MAX_CHUNK_CHARS", 1800)
        MIN_CHUNK_CHARS = _env_int("CHATDOC_MIN_CHUNK_CHARS", 250)
        OVERLAP_CHARS = _env_int("CHATDOC_CHUNK_OVERLAP_CHARS", 160)

        TABLE_MAX_LINES = _env_int("CHATDOC_TABLE_MAX_LINES_PER_CHUNK", 28)
        TABLE_HEADER_MAX_LINES = _env_int("CHATDOC_TABLE_HEADER_MAX_LINES", 3)

        # límite defensivo (evita explosión en docs enormes)
        MAX_TOTAL_CHUNKS = _env_int("CHATDOC_MAX_TOTAL_CHUNKS", 6000)

        suffix = Path(filename).suffix or ".bin"
        logger.info(
            "[chatdoc/from_bytes] start doc_id=%s filename=%s suffix=%s mime=%s bytes=%d",
            document_id, filename, suffix, mime_type, len(raw_bytes)
        )

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            loader = EphemeralRagLoader(language=language)

            t_load = time.time()
            res = loader.load_and_fragment_files([tmp_path])
            load_s = time.time() - t_load

            if not isinstance(res, tuple):
                logger.error("[chatdoc/from_bytes] load_and_fragment_files tipo inesperado=%s doc_id=%s", type(res), document_id)
                raise ValueError("load_and_fragment_files_impl debe devolver una tupla.")

            documents: List[str] = []
            metas: List[Dict[str, Any]] = []
            doc_names: List[str] = []

            if len(res) == 3:
                documents, doc_names, metas = res
            elif len(res) == 2:
                documents, doc_names = res
                metas = [{} for _ in (documents or [])]
            else:
                logger.error("[chatdoc/from_bytes] tupla tamaño inesperado=%d doc_id=%s", len(res), document_id)
                raise ValueError("load_and_fragment_files_impl devolvió tupla tamaño inesperado.")

            documents = documents or []
            metas = metas or [{} for _ in documents]

            logger.info(
                "[chatdoc/from_bytes] loader returned docs=%d metas=%d names=%d load_s=%.3f doc_id=%s",
                len(documents), len(metas), len(doc_names or []), load_s, document_id
            )

            # ---- limpieza y normalización ----
            docs_clean: List[str] = []
            metas_clean: List[Dict[str, Any]] = []
            for txt, meta in zip(documents, metas):
                if isinstance(txt, str) and txt.strip():
                    docs_clean.append(_norm_text(txt))
                    metas_clean.append(meta or {})

            if not docs_clean:
                sample_meta = metas[0] if metas else {}
                logger.warning(
                    "[chatdoc/from_bytes] NO_TEXT_EXTRACTED doc_id=%s filename=%s mime=%s raw_docs=%d sample_meta_keys=%s",
                    document_id, filename, mime_type, len(documents),
                    list(sample_meta.keys())[:40] if isinstance(sample_meta, dict) else []
                )

            # ---- POST-CHUNKING avanzado (structure/table/list aware) ----
            chunk_texts: List[str] = []
            chunk_metas: List[Dict[str, Any]] = []

            for src_idx, (txt, meta) in enumerate(zip(docs_clean, metas_clean)):
                base_meta = dict(meta or {})
                base_meta.setdefault("source_fragment_index", src_idx)

                # preserve/normalize page
                page = base_meta.get("page") or base_meta.get("page_number")
                try:
                    if page is not None:
                        base_meta["page"] = int(page)
                except Exception:
                    pass

                pieces = _fragment_advanced(
                    txt,
                    max_chars=MAX_CHUNK_CHARS,
                    min_chars=MIN_CHUNK_CHARS,
                    overlap_chars=OVERLAP_CHARS,
                    table_max_lines=TABLE_MAX_LINES,
                    table_header_max_lines=TABLE_HEADER_MAX_LINES,
                )

                for local_i, (piece_text, extra_meta) in enumerate(pieces):
                    m = dict(base_meta)
                    m.update(extra_meta or {})
                    m["local_chunk_index"] = local_i
                    chunk_texts.append(piece_text)
                    chunk_metas.append(m)

                    if len(chunk_texts) >= MAX_TOTAL_CHUNKS:
                        logger.warning(
                            "[chatdoc/from_bytes] MAX_TOTAL_CHUNKS alcanzado=%d doc_id=%s (truncando chunks)",
                            MAX_TOTAL_CHUNKS, document_id
                        )
                        break
                if len(chunk_texts) >= MAX_TOTAL_CHUNKS:
                    break

            # ---- full_text + offsets ----
            full_text = "\n\n".join(chunk_texts)
            total_len = len(full_text)
            sep_len = len("\n\n")

            chunks: List[ChatdocChunk] = []
            offset = 0
            for idx, (txt, meta) in enumerate(zip(chunk_texts, chunk_metas)):
                start = offset
                end = start + len(txt)
                pos_ratio = (start / total_len) if total_len > 0 else 0.0
                chunks.append(
                    ChatdocChunk(
                        text=txt,
                        index=idx,
                        char_start=start,
                        char_end=end,
                        position_ratio=pos_ratio,
                        meta=meta,
                    )
                )
                offset = end + sep_len

            # page_count desde metas (si existe)
            pages: List[int] = []
            for m in chunk_metas:
                p = None
                if isinstance(m, dict):
                    p = m.get("page") or m.get("page_number")
                try:
                    if p is not None:
                        pages.append(int(p))
                except Exception:
                    pass
            page_count = max(pages) if pages else 0

            meta_out = dict(metadata or {})
            meta_out.setdefault("source_file_name", filename)
            if mime_type:
                meta_out.setdefault("mime_type", mime_type)
            meta_out.setdefault("extraction_engine", "rag_loader")
            if page_count:
                meta_out.setdefault("page_count", page_count)
            meta_out.setdefault("chunking_profile", "advanced_v1")
            meta_out.setdefault("chunking_max_chars", MAX_CHUNK_CHARS)
            meta_out.setdefault("chunking_overlap_chars", OVERLAP_CHARS)

            idx_obj = cls(
                document_id=document_id,
                full_text=full_text,
                metadata=meta_out,
                chunk_chars=len(full_text) + 1,  # evita re-split simple
                chunk_overlap=0,
                language=language,
                chunks=chunks,
            )
            idx_obj.page_count = page_count

            logger.info(
                "[chatdoc/from_bytes] built(advanced) doc_id=%s chunks=%d chars=%d pages=%d total_s=%.3f",
                document_id, len(chunks), len(full_text), page_count, time.time() - t0
            )
            return idx_obj

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # ---------------------- embeddings (prefijo E5) ----------------------
    @torch.inference_mode()
    def _embed_texts(
        self,
        texts: List[str],
        *,
        is_query: bool = False,
        batch_size: Optional[int] = None
    ):
        """
        Embeddings robustos multiusuario (producción):
        - Política de device por env: CHATDOC_EMBED_DEVICE_POLICY=cpu|gpu|auto y/o CHATDOC_EMBED_FORCE_DEVICE
        - Por defecto opera en GPU (CHATDOC_EMBED_DEVICE_POLICY=gpu, strict_gpu=1)
        - Semáforo independiente CPU/GPU (estabiliza p95 y evita oversubscription)
        - Prefijo E5 si aplica
        - Backoff batch en OOM y fallback a CPU opcional
        - Compat SentenceTransformers antiguos (sin param device en encode)
        """
        import numpy as np
        import gc

        self._ensure_embed_backend()

        # --- Sanitizar textos ---
        in_texts = texts or []
        texts = []
        for t in in_texts:
            if t is None:
                continue
            s = str(t).strip()
            if s:
                texts.append(s)

        if not texts:
            dim = _embedding_dim()
            return np.zeros((0, dim if dim > 0 else 0), dtype="float32")

        # Truncado defensivo
        max_chars = _env_int("CHATDOC_EMBED_MAX_CHARS_PER_TEXT", 8000)
        if max_chars > 0:
            texts = [t[:max_chars] for t in texts]

        # Batch base
        bs0 = int(batch_size or int(os.getenv("HF_EMBED_BATCH", "32")))
        bs0 = max(1, min(bs0, len(texts)))

        # Prefijo E5
        if self._use_e5_prefix:
            if is_query:
                texts = [f"{E5_QUERY_PREFIX}{t}" for t in texts]
            else:
                texts = [f"{E5_DOC_PREFIX}{t}" for t in texts]

        # Decide prefer device por política
        prefer = _get_embed_device()

        # min_free_mb para CUDA
        try:
            min_free_mb = int(os.getenv("HF_EMBED_MIN_FREE_MB", "2048"))
        except Exception:
            min_free_mb = 2048

        # Semáforos CPU/GPU (lazy init)
        def _get_sem(name: str, default: int) -> threading.Semaphore:
            sem = globals().get(name)
            if sem is None:
                sem = threading.Semaphore(max(1, int(default)))
                globals()[name] = sem
            return sem

        gpu_cap = int(os.getenv("RAG_EMBED_CONCURRENCY", str(_RAG_EMBED_CONCURRENCY)))
        cpu_cap = int(os.getenv("RAG_EMBED_CPU_CONCURRENCY", os.getenv("CHATDOC_EMBED_CPU_CONCURRENCY", "4")))

        gpu_sem = _get_sem("_RAG_EMBED_GPU_SEMAPHORE_RUNTIME", gpu_cap)
        cpu_sem = _get_sem("_RAG_EMBED_CPU_SEMAPHORE_RUNTIME", cpu_cap)

        sem_timeout = float(os.getenv("RAG_EMBED_SEMAPHORE_TIMEOUT_S", "30"))

        def _maybe_empty_cache():
            if CHATDOC_CUDA_EMPTY_CACHE and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        def _encode(device_arg: str, bs: int):
            """
            Compat:
            - ST moderno: encode(..., device="cuda:0")
            - ST viejo: no acepta device -> movemos el modelo nosotros
            """
            try:
                return self.sbert_model.encode(
                    texts,
                    batch_size=bs,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=device_arg,
                    show_progress_bar=False,
                )
            except TypeError:
                try:
                    if hasattr(self.sbert_model, "to"):
                        self.sbert_model.to(device_arg)
                except Exception:
                    pass
                return self.sbert_model.encode(
                    texts,
                    batch_size=bs,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

        # Elegir semáforo según prefer (si auto cae a CPU, usaremos cpu_sem)
        chosen_sem = gpu_sem if (prefer.type == "cuda") else cpu_sem

        acquired = chosen_sem.acquire(timeout=sem_timeout)
        if not acquired:
            if CHATDOC_EMBED_STRICT_GPU and prefer.type == "cuda":
                raise RuntimeError(
                    "[chatdoc/embed] semaphore timeout en cola GPU con modo estricto activo; "
                    "no se permite fallback a CPU."
                )
            # No bloquear p95: fallback a CPU inmediato
            logger.warning("[chatdoc/embed] semaphore timeout (%.1fs) -> fallback CPU", sem_timeout)
            vecs = _encode("cpu", bs0)
            return np.asarray(vecs, dtype="float32")

        try:
            bs = bs0
            while True:
                try:
                    # CPU directo
                    if prefer.type == "cpu":
                        vecs = _encode("cpu", bs)
                        return np.asarray(vecs, dtype="float32")

                    # CUDA con pool limiter + locks + fallback
                    with gpu_manager.use_device_with_fallback(
                        self.sbert_model,
                        prefer_device=prefer,
                        min_free_mb=min_free_mb,
                        fallback_to_cpu=(not CHATDOC_EMBED_STRICT_GPU),
                        pool="embed",
                    ) as dev:

                        device_arg = str(dev) if getattr(dev, "type", "") == "cuda" else "cpu"
                        vecs = _encode(device_arg, bs)

                    arr = np.asarray(vecs, dtype="float32")

                    # Liberación opcional (ojo: si estás CPU-first, no se ejecuta normalmente)
                    if CHATDOC_EMBED_RELEASE_AFTER_CALL:
                        try:
                            self.sbert_model.to("cpu")
                        except Exception:
                            pass
                        _maybe_empty_cache()

                    # sticky device: si lo usas, no reasignes; si no, deja que se refresque más tarde
                    return arr

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if _is_cuda_oom(e):
                        _maybe_empty_cache()
                        gc.collect()

                        if bs > 1:
                            bs = max(1, bs // 2)
                            logger.warning("[chatdoc/embed] OOM -> reduciendo batch_size a %d", bs)
                            continue

                        if CHATDOC_EMBED_OOM_FALLBACK_CPU:
                            logger.warning("[chatdoc/embed] OOM persistente -> fallback a CPU")
                            try:
                                self.sbert_model.to("cpu")
                            except Exception:
                                pass
                            _maybe_empty_cache()
                            vecs = _encode("cpu", 1)
                            return np.asarray(vecs, dtype="float32")

                    raise
        finally:
            try:
                chosen_sem.release()
            except Exception:
                pass

    def _ensure_embeddings(self) -> None:
        if self._embeddings is not None:
            return

        lock = getattr(self, "_embeddings_lock", None)
        if lock is None:
            lock = threading.RLock()
            setattr(self, "_embeddings_lock", lock)

        with lock:
            if self._embeddings is not None:
                return

            if not self._chunks:
                dim = _embedding_dim()
                self._embeddings = (
                    np.zeros((0, dim), dtype="float32") if dim > 0 else np.zeros((0, 0), dtype="float32")
                )
                return

            dim = _embedding_dim()
            if dim <= 0:
                # No hay dimensión válida => no podemos hacer búsqueda semántica fiable
                self._embeddings = np.zeros((len(self._chunks), 0), dtype="float32")
                logger.warning("[chatdoc] embedding dim=0 -> embeddings vacíos doc_id=%s", self.document_id)
                return

            texts = [c.text for c in self._chunks]
            self._embeddings = self._embed_texts(texts, is_query=False)

            try:
                out_dim = int(self._embeddings.shape[1]) if self._embeddings is not None and self._embeddings.ndim == 2 else 0
            except Exception:
                out_dim = 0

            logger.info(
                "[chatdoc] embeddings ready doc_id=%s chunks=%d dim=%s use_e5_prefix=%s device=%s",
                self.document_id,
                len(self._chunks),
                out_dim or None,
                getattr(self, "_use_e5_prefix", USE_E5_PREFIX),
                getattr(self, "embed_device", None),
            )


    # ---------------------- búsqueda ----------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        window: int = 0,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        query = (query or "").strip()
        if not query:
            raise ValueError("La query no puede estar vacía en DocumentChatIndex.search().")

        self._ensure_embeddings()
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            return []

        # Si no hay dimensión, la similitud no es significativa
        if self._embeddings.ndim != 2 or self._embeddings.shape[1] <= 0:
            return []

        top_k = max(1, int(top_k))
        min_score = float(min_score)
        window = max(0, int(window))

        q_arr = self._embed_texts([query], is_query=True)
        if q_arr is None or getattr(q_arr, "ndim", 0) != 2 or q_arr.shape[1] != self._embeddings.shape[1]:
            return []
        q_vec = q_arr[0]

        sims = np.dot(self._embeddings, q_vec)
        order = np.argsort(-sims)

        base_indices = order[:top_k]

        expanded: set[int] = set()
        num_chunks = len(self._chunks)

        for base_pos in base_indices:
            base_score = float(sims[base_pos])
            if base_score < min_score:
                continue

            start = max(0, int(base_pos) - window)
            end = min(num_chunks - 1, int(base_pos) + window)

            for pos in range(start, end + 1):
                expanded.add(int(pos))

        if not expanded:
            return []

        sorted_positions = sorted(expanded, key=lambda p: float(sims[p]), reverse=True)

        results: List[Dict[str, Any]] = []
        for pos in sorted_positions:
            ch = self._chunks[pos]
            score = float(sims[pos])

            merged_meta: Dict[str, Any] = dict(self.metadata or {})
            if ch.meta and isinstance(ch.meta, dict):
                merged_meta.update(ch.meta)

            results.append(
                {
                    "text": ch.text,
                    "score": score,
                    "chunk_index": ch.index,
                    "char_start": ch.char_start,
                    "char_end": ch.char_end,
                    "position_ratio": ch.position_ratio,
                    "document_id": self.document_id,
                    "metadata": merged_meta,
                    "chunk_metadata": ch.meta or {},
                }
            )

        return results


    # ---------------------- selección de fragmentos para resumen ----------------------
    def select_summary_chunks(
        self,
        max_fragments: int = 12,
        strategy: str = "hybrid",
        min_chars_per_chunk: int = 300,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        if not self._chunks:
            return []

        max_fragments = max(1, int(max_fragments))
        min_chars_per_chunk = max(1, int(min_chars_per_chunk))

        candidate_positions = [
            i for i, c in enumerate(self._chunks)
            if len((c.text or "")) >= min_chars_per_chunk
        ]
        if not candidate_positions:
            candidate_positions = list(range(len(self._chunks)))

        def _uniform_pick_positions(pos_list: List[int], k: int) -> List[int]:
            if k >= len(pos_list):
                return pos_list
            if len(pos_list) <= 1:
                return pos_list
            idxs = np.linspace(0, len(pos_list) - 1, num=k)
            idxs = [int(round(x)) for x in idxs]
            idxs = [max(0, min(len(pos_list) - 1, i)) for i in idxs]
            out: List[int] = []
            seen = set()
            for i in idxs:
                p = pos_list[i]
                if p not in seen:
                    seen.add(p)
                    out.append(p)
            return out

        strategy_norm = (strategy or "hybrid").lower().strip()
        if strategy_norm not in ("uniform", "tfidf", "hybrid"):
            strategy_norm = "hybrid"

        use_dense = strategy_norm in ("tfidf", "hybrid")
        if use_dense:
            self._ensure_embeddings()

        if (not use_dense) or (self._embeddings is None) or (self._embeddings.shape[0] == 0):
            selected_positions = _uniform_pick_positions(candidate_positions, max_fragments)
        else:
            cand_embs = self._embeddings[candidate_positions, :]  # normalizados
            centroid = cand_embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

            base_scores = cand_embs @ centroid  # centralidad real

            mmr_lambda = float(os.getenv("CHATDOC_SUMMARY_MMR_LAMBDA", "0.65"))
            mmr_lambda = max(0.05, min(mmr_lambda, 0.95))

            k = min(max_fragments, len(candidate_positions))
            selected_local: List[int] = []

            seed = int(np.argmax(base_scores))
            selected_local.append(seed)

            N = cand_embs.shape[0]
            precompute = N <= int(os.getenv("CHATDOC_SUMMARY_MMR_PRECOMP_MAX", "1200"))
            sims = (cand_embs @ cand_embs.T) if precompute else None

            while len(selected_local) < k:
                best_j = None
                best_score = -1e9
                for j in range(N):
                    if j in selected_local:
                        continue
                    max_sim = float(np.max(sims[j, selected_local])) if sims is not None else float(
                        np.max(cand_embs[j] @ cand_embs[selected_local].T)
                    )
                    mmr_score = mmr_lambda * float(base_scores[j]) - (1.0 - mmr_lambda) * max_sim
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_j = j
                if best_j is None:
                    break
                selected_local.append(int(best_j))

            dense_positions = [candidate_positions[i] for i in selected_local]

            if strategy_norm == "tfidf":
                selected_positions = dense_positions
            else:
                k_uniform = max_fragments // 2
                k_dense = max_fragments - k_uniform

                uniform_part = _uniform_pick_positions(candidate_positions, min(k_uniform, len(candidate_positions)))
                seen = set(uniform_part)

                dense_part: List[int] = []
                for p in dense_positions:
                    if len(dense_part) >= k_dense:
                        break
                    if p not in seen:
                        seen.add(p)
                        dense_part.append(p)

                selected_positions = uniform_part + dense_part

        out: List[Dict[str, Any]] = []
        for pos in selected_positions:
            ch = self._chunks[pos]
            merged_meta: Dict[str, Any] = dict(self.metadata or {})
            if ch.meta and isinstance(ch.meta, dict):
                merged_meta.update(ch.meta)

            page = None
            if ch.meta and isinstance(ch.meta, dict):
                page = ch.meta.get("page") or ch.meta.get("page_number")

            out.append(
                {
                    "text": ch.text,
                    "chunk_index": ch.index,
                    "char_start": ch.char_start,
                    "char_end": ch.char_end,
                    "position_ratio": ch.position_ratio,
                    "document_id": self.document_id,
                    "metadata": merged_meta,
                    "chunk_metadata": ch.meta or {},
                    "page": page,
                }
            )

        return out

    # ---------------------- serialización ----------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "full_text": self.full_text,
            "metadata": self.metadata,
            "chunk_chars": self.chunk_chars,
            "chunk_overlap": self.chunk_overlap,
            "language": self.language,
            "chunks": [asdict(c) for c in self._chunks],
            "page_count": self.page_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChatIndex":
        if not isinstance(data, dict):
            data = {}

        doc_id = str(data.get("document_id", "") or "")
        full_text = str(data.get("full_text", "") or "")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        try:
            chunk_chars = int(data.get("chunk_chars", 1200))
        except Exception:
            chunk_chars = 1200

        try:
            chunk_overlap = int(data.get("chunk_overlap", 200))
        except Exception:
            chunk_overlap = 200

        language = data.get("language") or "spanish"

        try:
            page_count = int(data.get("page_count") or 0)
        except Exception:
            page_count = 0

        chunks_data = data.get("chunks") or []
        if not isinstance(chunks_data, list):
            chunks_data = []

        chunks: List[ChatdocChunk] = []
        for i, cd in enumerate(chunks_data):
            if not isinstance(cd, dict):
                continue

            text = str(cd.get("text", "") or "")
            try:
                index = int(cd.get("index", i))
            except Exception:
                index = i

            try:
                char_start = int(cd.get("char_start", 0))
            except Exception:
                char_start = 0

            try:
                char_end = int(cd.get("char_end", char_start + len(text)))
            except Exception:
                char_end = char_start + len(text)

            try:
                position_ratio = float(cd.get("position_ratio", 0.0))
            except Exception:
                position_ratio = 0.0

            meta = cd.get("meta") or cd.get("chunk_metadata") or None
            if meta is not None and not isinstance(meta, dict):
                meta = None

            chunks.append(
                ChatdocChunk(
                    text=text,
                    index=index,
                    char_start=max(0, char_start),
                    char_end=max(max(0, char_start), char_end),
                    position_ratio=position_ratio,
                    meta=meta,
                )
            )

        idx_obj = cls(
            document_id=doc_id,
            full_text=full_text,
            metadata=metadata,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            language=language,
            chunks=chunks,
        )
        idx_obj.page_count = page_count
        return idx_obj



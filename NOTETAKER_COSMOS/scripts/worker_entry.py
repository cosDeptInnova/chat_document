from __future__ import annotations

import os
import sys
import platform
from pathlib import Path

# Asegura que el root del repo esté en sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv_best_effort() -> None:
    """
    Carga .env si existe (para local/pro y luego K8s).
    No revienta si python-dotenv no está instalado.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    # 1) .env en CWD
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env, override=False)

    # 2) .env en root del repo
    root_env = ROOT / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)


def _force_offline_env() -> None:
    """
    Fuerza modo OFFLINE para evitar cualquier salida a internet por HF/Transformers.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "1"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "1"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", os.getenv("HF_HUB_DISABLE_TELEMETRY", "1"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", os.getenv("TOKENIZERS_PARALLELISM", "false"))

    # Opcional: fija HF_HOME si lo defines en .env (para cache local controlada)
    # os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", ""))


def _tighten_ctranslate2_cuda_cache() -> None:
    """
    Reduce caché del allocator CUDA de CTranslate2 para minimizar VRAM residual.
    Esto ayuda cuando las GPUs están compartidas con LLMs.
    """
    os.environ.setdefault("CT2_CUDA_ALLOCATOR", os.getenv("CT2_CUDA_ALLOCATOR", "cub_caching"))
    # config: <bin_growth>,<min_bin>,<max_bin>,<max_cached_bytes>
    # 64MB cache
    os.environ.setdefault("CT2_CUDA_CACHING_ALLOCATOR_CONFIG", os.getenv("CT2_CUDA_CACHING_ALLOCATOR_CONFIG", "8,3,7,67108864"))


# Carga .env y fuerza offline ANTES de importar app.*
_load_dotenv_best_effort()
_force_offline_env()
_tighten_ctranslate2_cuda_cache()

from redis import Redis  # noqa: E402
from rq import Queue, Connection  # noqa: E402
from rq.worker import Worker, SimpleWorker  # noqa: E402
from rq.timeouts import TimerDeathPenalty  # noqa: E402

from app.core.settings import get_settings  # noqa: E402
from app.core.logging import setup_logging, logger  # noqa: E402


class WindowsSimpleWorker(SimpleWorker):
    # En Windows, usa timeouts basados en timer (no señales Unix)
    death_penalty_class = TimerDeathPenalty


def main():
    setup_logging()
    if len(sys.argv) < 2:
        print("Usage: python scripts/worker_entry.py <queue_name>")
        sys.exit(2)

    queue_name = sys.argv[1]

    # Etiqueta de observabilidad: útil para meta en resultados y logs
    os.environ["RQ_QUEUE"] = queue_name

    # Cap de threads por proceso (evita que cada worker use 128 hilos)
    # Ajusta en .env con CPU_THREADS_PER_WORKER (recomendado 4 u 8)
    cpu_threads = os.getenv("CPU_THREADS_PER_WORKER", "4")
    os.environ.setdefault("OMP_NUM_THREADS", cpu_threads)
    os.environ.setdefault("MKL_NUM_THREADS", cpu_threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", cpu_threads)

    # Pin GPU por cola + forzar ASR_DEVICE solo en workers CPU
    # IMPORTANTE: así "asr:cpu" realmente usa CPU (y puedes escalar por CPU)
    s = get_settings()

    cuda_env = None
    for q in s.queues:
        if q.name == queue_name:
            cuda_env = q.cuda_visible_devices
            break

    is_cpu_queue = (cuda_env is None) or (str(cuda_env).strip() == "")

    if is_cpu_queue:
        # CPU worker: NO GPU visible + fuerza ASR a CPU (solo para este proceso)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["ASR_DEVICE"] = "cpu"
    else:
        # GPU worker: pin a esa GPU + fuerza ASR a CUDA (solo para este proceso)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_env).strip()
        os.environ["ASR_DEVICE"] = "cuda"

    logger.info(
        "Starting worker pid=%s queue=%s is_cpu=%s CUDA_VISIBLE_DEVICES=%s ASR_DEVICE=%s HF_HUB_OFFLINE=%s",
        os.getpid(),
        queue_name,
        is_cpu_queue,
        os.getenv("CUDA_VISIBLE_DEVICES"),
        os.getenv("ASR_DEVICE"),
        os.getenv("HF_HUB_OFFLINE"),
    )

    redis_conn = Redis.from_url(s.redis_url)

    is_windows = platform.system().lower().startswith("win")
    WorkerClass = WindowsSimpleWorker if is_windows else Worker

    with Connection(redis_conn):
        q = Queue(queue_name)
        w = WorkerClass([q])
        w.work(with_scheduler=False)

if __name__ == "__main__":
    main()

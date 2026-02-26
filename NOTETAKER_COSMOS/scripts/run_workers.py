from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from app.core.settings import get_settings
from app.core.logging import setup_logging, logger


def _is_cpu_queue(cuda_visible_devices: str) -> bool:
    return not (cuda_visible_devices or "").strip()


def _parse_first_gpu_index(cuda_visible_devices: str) -> Optional[int]:
    """
    queues cuda_visible_devices suele ser "0" o "1".
    Si viniera "0,1" tomamos el primero.
    """
    s = (cuda_visible_devices or "").strip()
    if not s:
        return None
    first = s.split(",")[0].strip()
    try:
        return int(first)
    except Exception:
        return None


def _query_gpu_mem_mb(gpu_index: int) -> Optional[Tuple[int, int, int]]:
    """
    Devuelve (total_mb, used_mb, free_mb) usando nvidia-smi.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,nounits,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except FileNotFoundError:
        logger.warning("nvidia-smi no está en PATH. No puedo auto-calcular workers por GPU.")
        return None
    except subprocess.CalledProcessError as e:
        logger.warning("nvidia-smi falló (gpu=%s): %s", gpu_index, str(e.output)[-300:])
        return None

    # salida típica: "46068, 1234"
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 2:
        return None

    try:
        total = int(parts[0])
        used = int(parts[1])
        free = max(0, total - used)
        return total, used, free
    except Exception:
        return None


def _calc_gpu_workers_auto(gpu_index: int, budget_mb: int, headroom_mb: int) -> int:
    """
    Calcula workers por GPU para NO pasar de budget_mb y manteniendo headroom_mb libre.

    Necesita ASR_VRAM_PER_WORKER_MB definido en entorno (medido por ti).
    Si no está, devuelve 1 (seguro).
    """
    per_worker_mb = int(os.getenv("ASR_VRAM_PER_WORKER_MB", "0"))
    if per_worker_mb <= 0:
        logger.warning(
            "GPU_WORKERS_MODE=auto pero ASR_VRAM_PER_WORKER_MB no está definido. "
            "Fallo a 1 worker/GPU (seguro)."
        )
        return 1

    mem = _query_gpu_mem_mb(gpu_index)
    if mem is None:
        return 1

    total_mb, used_mb, free_mb = mem

    # límite duro (budget) y además no invadir lo que ya esté usando la GPU (LLMs)
    # -> target = min(budget, free - headroom)
    target_mb = min(budget_mb, max(0, free_mb - headroom_mb))

    # cuántos workers "caben" dentro del target, mínimo 1
    n = max(1, target_mb // per_worker_mb)

    # techo opcional por seguridad
    max_workers = int(os.getenv("GPU_MAX_WORKERS_PER_QUEUE", "8"))
    n = max(1, min(n, max_workers))

    logger.info(
        "AUTO GPU workers gpu=%s total=%sMB used=%sMB free=%sMB budget=%sMB headroom=%sMB "
        "per_worker=%sMB -> workers=%s",
        gpu_index,
        total_mb,
        used_mb,
        free_mb,
        budget_mb,
        headroom_mb,
        per_worker_mb,
        n,
    )
    return n


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _calc_cpu_workers_auto() -> int:
    """
    Auto-calcula CPU workers para apuntar a ~CPU_TARGET_UTIL del total lógico,
    evitando sobre-suscripción usando CPU_THREADS_PER_WORKER (o OMP_NUM_THREADS).
    """
    logical = os.cpu_count() or 1  # en tu máquina: 128
    target_util = _clamp(_env_float("CPU_TARGET_UTIL", 0.80), 0.10, 1.00)

    threads_per_worker = int(os.getenv("CPU_THREADS_PER_WORKER", "0"))
    if threads_per_worker <= 0:
        # si no lo declaras, intentamos reusar OMP_NUM_THREADS
        threads_per_worker = int(os.getenv("OMP_NUM_THREADS", "4"))

    threads_per_worker = max(1, threads_per_worker)

    # “budget” de threads lógicos a consumir
    budget_threads = max(1, int(logical * target_util))

    # workers = budget / threads_por_worker
    n = max(1, budget_threads // threads_per_worker)

    # techo de seguridad
    max_workers = int(os.getenv("CPU_MAX_WORKERS_PER_QUEUE", "64"))
    n = max(1, min(n, max_workers))

    logger.info(
        "AUTO CPU workers logical=%s target_util=%.2f budget_threads=%s threads_per_worker=%s -> CPU_WORKERS_PER_QUEUE=%s",
        logical, target_util, budget_threads, threads_per_worker, n
    )
    return n


def main():
    setup_logging()
    s = get_settings()
    root = Path(__file__).resolve().parents[1]

    # Privacy-first defaults (offline)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Thread-caps para evitar que cada worker intente usar todos los cores
    # (Esto es CLAVE para poder escalar por “muchos procesos” sin caos)
    os.environ.setdefault("OMP_NUM_THREADS", os.getenv("CPU_THREADS_PER_WORKER", "4"))
    os.environ.setdefault("MKL_NUM_THREADS", os.getenv("CPU_THREADS_PER_WORKER", "4"))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", os.getenv("CPU_THREADS_PER_WORKER", "4"))

    # Concurrency controls (env)
    gpu_workers_mode = os.getenv("GPU_WORKERS_MODE", "fixed").strip().lower()  # fixed|auto
    workers_per_queue = int(os.getenv("WORKERS_PER_QUEUE", "1"))

    cpu_workers_mode = os.getenv("CPU_WORKERS_MODE", "fixed").strip().lower()  # fixed|auto
    if cpu_workers_mode == "auto":
        cpu_workers_per_queue = _calc_cpu_workers_auto()
    else:
        cpu_workers_per_queue = int(os.getenv("CPU_WORKERS_PER_QUEUE", "2"))

    allow_gpu_overcommit = os.getenv("ALLOW_GPU_OVERCOMMIT", "0").strip().lower() in (
        "1", "true", "yes", "y", "on"
    )

    procs: List[subprocess.Popen] = []

    for q in s.queues:
        is_cpu = _is_cpu_queue(q.cuda_visible_devices)

        if is_cpu:
            n = max(1, cpu_workers_per_queue)
        else:
            gpu_index = _parse_first_gpu_index(q.cuda_visible_devices)

            if gpu_workers_mode == "auto" and gpu_index is not None:
                n = _calc_gpu_workers_auto(
                    gpu_index=gpu_index,
                    budget_mb=int(s.asr_vram_budget_mb),
                    headroom_mb=int(s.asr_vram_headroom_mb),
                )
            else:
                n = max(1, workers_per_queue)
                if n > 1 and not allow_gpu_overcommit:
                    logger.warning(
                        "WORKERS_PER_QUEUE=%s pero ALLOW_GPU_OVERCOMMIT=0 -> forzando 1 worker para %s",
                        n, q.name
                    )
                    n = 1

        for i in range(n):
            cmd = [sys.executable, str(root / "scripts" / "worker_entry.py"), q.name]

            # IMPORTANT: fija CUDA_VISIBLE_DEVICES por proceso (no dependas solo del worker_entry)
            child_env = os.environ.copy()
            if is_cpu:
                child_env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                child_env["CUDA_VISIBLE_DEVICES"] = (q.cuda_visible_devices or "").strip()

            logger.info("Launching worker (%s/%s) for queue=%s: %s", i + 1, n, q.name, " ".join(cmd))
            procs.append(subprocess.Popen(cmd, env=child_env))

    logger.info("Workers running. Ctrl+C to stop.")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        logger.info("Stopping workers...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass



if __name__ == "__main__":
    main()

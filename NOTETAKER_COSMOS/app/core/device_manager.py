from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import os
from redis import Redis

from app.core.settings import Settings
from app.core.logging import logger
from app.jobs.queue import get_rq_queue


@dataclass(frozen=True)
class QueueSelection:
    queue_name: str


def _parse_first_cuda_index(cuda_visible_devices: str) -> Optional[int]:
    """
    QueueSpec.cuda_visible_devices is usually "0" or "1" or "".
    We take the first entry if present.
    """
    v = (cuda_visible_devices or "").strip()
    if not v:
        return None
    first = v.split(",")[0].strip()
    if not first:
        return None
    try:
        return int(first)
    except Exception:
        return None


def _nvml_free_mb(gpu_index: int) -> Optional[int]:
    """
    Return free VRAM (MB) for the physical GPU index.
    Uses NVML if available.
    """
    try:
        from pynvml import (  # type: ignore
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
        )
    except Exception:
        return None

    try:
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(int(gpu_index))
        mem = nvmlDeviceGetMemoryInfo(h)
        return int(mem.free // (1024 * 1024))
    except Exception:
        return None


class DeviceManager:
    """
    Selecciona cola (y por extensión GPU) para el job.

    Estrategia (PRO):
    1) Preferir colas GPU con:
       - backlog total más bajo (queued + started)
       - y VRAM libre suficiente (si NVML disponible)
    2) Si no hay VRAM suficiente, y existe cola CPU, usarla (si allow_cpu_fallback).
    3) Fallback final: default_queue.

    Nota: el worker ya se encarga del "VRAM guard" y fallback (WhisperASR),
    aquí solo intentamos distribuir mejor para evitar contention con LLMs.
    """

    def __init__(self, settings: Settings, redis_conn: Redis):
        self.settings = settings
        self.redis = redis_conn

    def _queue_load(self, queue_name: str) -> Tuple[int, int]:
        """
        Returns (queued_count, started_count). Robust to differences in RQ objects.
        """
        q = get_rq_queue(self.settings, self.redis, queue_name)
        queued = 10**9
        started = 0
        try:
            queued = int(q.count)
        except Exception:
            queued = 10**9
        try:
            # started_job_registry exists on rq.Queue
            started = int(q.started_job_registry.count)
        except Exception:
            started = 0
        return queued, started

    def choose_queue(self) -> QueueSelection:
        # If no queues configured, use default
        if not self.settings.queues:
            return QueueSelection(queue_name=self.settings.default_queue)

        # If ASR device is CPU, prefer a CPU queue if one exists
        if str(self.settings.asr_device).lower() != "cuda":
            for qspec in self.settings.queues:
                if not (qspec.cuda_visible_devices or "").strip():
                    return QueueSelection(queue_name=qspec.name)
            return QueueSelection(queue_name=self.settings.default_queue)

        # --- parámetros de ruteo (env) ---
        # Si existe medición real por worker, úsala como umbral de VRAM (mejor que budget=5120)
        per_worker_mb = int(os.getenv("ASR_VRAM_PER_WORKER_MB", "0"))
        if per_worker_mb > 0:
            vram_need_mb = per_worker_mb + int(self.settings.asr_vram_headroom_mb)
        else:
            # fallback conservador
            vram_need_mb = int(self.settings.asr_vram_budget_mb) + int(self.settings.asr_vram_headroom_mb)

        # Derivar a CPU si backlog GPU supera umbral (para 200 reuniones es clave)
        gpu_backlog_to_cpu = int(os.getenv("GPU_BACKLOG_TO_CPU", "0"))  # 0 = desactivado
        prefer_cpu_when_gpu_busy = os.getenv("PREFER_CPU_WHEN_GPU_BUSY", "1").strip().lower() in ("1", "true", "yes", "y", "on")

        # Gather candidates
        gpu_candidates = []
        cpu_candidates = []
        for qspec in self.settings.queues:
            if not (qspec.cuda_visible_devices or "").strip():
                cpu_candidates.append(qspec)
            else:
                gpu_candidates.append(qspec)

        # compute total GPU backlog
        total_gpu_backlog = 0
        for qspec in gpu_candidates:
            queued, started = self._queue_load(qspec.name)
            total_gpu_backlog += (queued + started)

        # Si hay cola CPU y GPUs están muy saturadas, manda a CPU (evita esperas infinitas)
        if (
            prefer_cpu_when_gpu_busy
            and self.settings.asr_allow_cpu_fallback
            and cpu_candidates
            and gpu_backlog_to_cpu > 0
            and total_gpu_backlog >= gpu_backlog_to_cpu
        ):
            best_cpu = None
            best_cpu_score = None
            for qspec in cpu_candidates:
                queued, started = self._queue_load(qspec.name)
                score = (queued + started, queued, started)
                if best_cpu_score is None or score < best_cpu_score:
                    best_cpu_score = score
                    best_cpu = qspec
            logger.warning(
                "GPU backlog high (total=%s >= %s). Routing to CPU queue=%s",
                total_gpu_backlog, gpu_backlog_to_cpu, best_cpu.name if best_cpu else self.settings.default_queue
            )
            return QueueSelection(queue_name=best_cpu.name if best_cpu else self.settings.default_queue)

        # Score GPU candidates: prefer VRAM OK + low started/queued + more free
        best_gpu = None
        best_score = None
        best_free = None
        nvml_seen = False

        for qspec in gpu_candidates:
            queued, started = self._queue_load(qspec.name)
            gpu_idx = _parse_first_cuda_index(qspec.cuda_visible_devices)

            free_mb = None
            if gpu_idx is not None:
                free_mb = _nvml_free_mb(gpu_idx)

            if free_mb is not None:
                nvml_seen = True

            vram_ok = (free_mb is None) or (free_mb >= vram_need_mb)

            # score: vram_ok primero, luego started, luego queued, luego más free
            score = (0 if vram_ok else 1, started, queued, -(free_mb or 0))

            if best_score is None or score < best_score:
                best_score = score
                best_gpu = qspec
                best_free = free_mb

        # Si NVML dice que no hay VRAM suficiente, CPU fallback si existe
        if nvml_seen and best_gpu is not None and best_free is not None and best_free < vram_need_mb:
            if self.settings.asr_allow_cpu_fallback and cpu_candidates:
                best_cpu = None
                best_cpu_score = None
                for qspec in cpu_candidates:
                    queued, started = self._queue_load(qspec.name)
                    score = (queued + started, queued, started)
                    if best_cpu_score is None or score < best_cpu_score:
                        best_cpu_score = score
                        best_cpu = qspec
                logger.warning(
                    "GPU VRAM low (best free=%sMB < need=%sMB). Routing to CPU queue=%s",
                    best_free, vram_need_mb, best_cpu.name if best_cpu else self.settings.default_queue
                )
                return QueueSelection(queue_name=best_cpu.name if best_cpu else self.settings.default_queue)

            logger.warning(
                "GPU VRAM low (best free=%sMB < need=%sMB) and no CPU fallback. Using GPU queue=%s",
                best_free, vram_need_mb, best_gpu.name
            )
            return QueueSelection(queue_name=best_gpu.name)

        if best_gpu is not None:
            logger.debug("Queue selection: %s score=%s free=%s need=%s", best_gpu.name, best_score, best_free, vram_need_mb)
            return QueueSelection(queue_name=best_gpu.name)

        # no GPU -> CPU o default
        if cpu_candidates:
            best_cpu = None
            best_cpu_score = None
            for qspec in cpu_candidates:
                queued, started = self._queue_load(qspec.name)
                score = (queued + started, queued, started)
                if best_cpu_score is None or score < best_cpu_score:
                    best_cpu_score = score
                    best_cpu = qspec
            return QueueSelection(queue_name=best_cpu.name if best_cpu else self.settings.default_queue)

        return QueueSelection(queue_name=self.settings.default_queue)
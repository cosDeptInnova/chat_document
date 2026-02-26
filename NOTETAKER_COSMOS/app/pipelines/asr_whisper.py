from __future__ import annotations

import os
import gc
import logging
from typing import Any, Dict, List, Optional, Tuple

from faster_whisper import WhisperModel

logger = logging.getLogger("asr_service")


def _visible_physical_gpu_index() -> Optional[int]:
    """
    Map CUDA_VISIBLE_DEVICES -> physical GPU index for NVML queries.
    If CUDA_VISIBLE_DEVICES="1" => physical 1.
    If not set, assume 0.
    """
    v = os.getenv("CUDA_VISIBLE_DEVICES")
    if not v:
        return 0
    first = v.split(",")[0].strip()
    if first == "":
        return None
    try:
        return int(first)
    except Exception:
        return None


def _nvml_free_mb() -> Optional[int]:
    """
    Return free VRAM (MB) on the GPU this worker is pinned to (physical index).
    If NVML not available, returns None (no guard).
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
        idx = _visible_physical_gpu_index()
        if idx is None:
            return None
        h = nvmlDeviceGetHandleByIndex(int(idx))
        mem = nvmlDeviceGetMemoryInfo(h)
        return int(mem.free // (1024 * 1024))
    except Exception:
        return None


def _estimated_required_vram_mb(budget_mb: int, headroom_mb: int) -> int:
    """
    Umbral realista: usa ASR_VRAM_PER_WORKER_MB si existe (medido),
    si no, usa budget (conservador).
    """
    per_worker = int(os.getenv("ASR_VRAM_PER_WORKER_MB", "0"))
    base = per_worker if per_worker > 0 else int(budget_mb)
    return int(base + int(headroom_mb))


class WhisperASR:
    """
    Faster-Whisper wrapper with:
      - local model path support
      - low-VRAM compute types (int8_float16 / int8)
      - optional VRAM guard (budget + headroom)
      - optional unload after each job to not pin VRAM (shared GPU with LLMs)
    """

    def __init__(
        self,
        model_path_or_id: str,
        device: str,
        compute_type: str,
        vram_budget_mb: int = 5120,
        vram_headroom_mb: int = 512,
        keep_model_loaded: bool = True,
        allow_cpu_fallback: bool = True,
    ):
        self.model_path_or_id = model_path_or_id
        self.device = device
        self.compute_type = compute_type

        self.vram_budget_mb = int(vram_budget_mb)
        self.vram_headroom_mb = int(vram_headroom_mb)
        self.keep_model_loaded = bool(keep_model_loaded)
        self.allow_cpu_fallback = bool(allow_cpu_fallback)

        self._model: Optional[WhisperModel] = None
        self._active_device: Optional[str] = None
        self._active_compute_type: Optional[str] = None

    def _choose_runtime(self) -> Tuple[str, str]:
        """
        Decide device/compute_type respetando guard de VRAM con umbral REAL.
        """
        dev = self.device
        ctype = self.compute_type

        if dev != "cuda":
            return "cpu", ctype

        free_mb = _nvml_free_mb()
        if free_mb is not None:
            needed = _estimated_required_vram_mb(self.vram_budget_mb, self.vram_headroom_mb)
            if free_mb < needed:
                msg = f"GPU free VRAM too low ({free_mb}MB < {needed}MB)."
                if self.allow_cpu_fallback:
                    logger.warning("%s Falling back to CPU for ASR.", msg)
                    return "cpu", ctype
                raise RuntimeError(msg + " CPU fallback disabled.")

        return "cuda", ctype

    def _load_model(self) -> None:
        if self._model is not None:
            return

        dev, ctype = self._choose_runtime()

        # CPU tuning: no queremos que 1 worker se coma los 128 hilos
        cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", os.getenv("CPU_THREADS_PER_WORKER", "4")))
        cpu_threads = max(1, cpu_threads)

        def _make(device: str, compute_type: str) -> WhisperModel:
            if device == "cuda":
                return WhisperModel(
                    self.model_path_or_id,
                    device="cuda",
                    device_index=0,
                    compute_type=compute_type,
                )

            # En CPU intentamos pasar cpu_threads si la versión lo soporta
            try:
                return WhisperModel(
                    self.model_path_or_id,
                    device="cpu",
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                )
            except TypeError:
                return WhisperModel(
                    self.model_path_or_id,
                    device="cpu",
                    compute_type=compute_type,
                )

        try_order = [(dev, ctype)]
        if dev == "cuda":
            if ctype != "int8_float16":
                try_order.append(("cuda", "int8_float16"))
            if ctype != "int8":
                try_order.append(("cuda", "int8"))
            if self.allow_cpu_fallback:
                try_order.append(("cpu", ctype))

        last_err: Optional[Exception] = None
        for d, ct in try_order:
            try:
                logger.info("Loading WhisperModel model=%s device=%s compute_type=%s", self.model_path_or_id, d, ct)
                self._model = _make(d, ct)
                self._active_device = d
                self._active_compute_type = ct
                return
            except Exception as e:
                last_err = e
                logger.warning("Failed to load WhisperModel on %s/%s: %r", d, ct, e)

        raise RuntimeError(f"Could not load WhisperModel. Last error: {last_err!r}")
    
    
    def unload(self) -> None:
        """
        Release model and (best-effort) free GPU memory.
        """
        self._model = None
        self._active_device = None
        self._active_compute_type = None
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def transcribe(
        self,
        wav_path: str,
        language: str = "es",
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
    ) -> Dict[str, Any]:
        self._load_model()
        assert self._model is not None

        # For shared GPUs: keep beam_size modest to reduce compute pressure.
        # (VRAM impact is mostly model, but this helps overall contention.)
        segments, info = self._model.transcribe(
            wav_path,
            language=language,
            beam_size=int(beam_size),
            vad_filter=bool(vad_filter),
            word_timestamps=bool(word_timestamps),
        )

        out_segments: List[Dict[str, Any]] = []
        for s in segments:
            seg: Dict[str, Any] = {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
            }
            if getattr(s, "words", None):
                seg["words"] = [
                    {"start": float(w.start), "end": float(w.end), "word": w.word}
                    for w in s.words
                ]
            out_segments.append(seg)

        result = {
            "meta": {
                "model": str(self.model_path_or_id),
                "language": info.language,
                "language_probability": float(info.language_probability),
                "device": self._active_device,
                "compute_type": self._active_compute_type,
            },
            "segments": out_segments,
        }

        if not self.keep_model_loaded:
            self.unload()

        return result

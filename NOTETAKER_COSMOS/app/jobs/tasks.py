from __future__ import annotations

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.core.settings import get_settings
from app.core.logging import logger
from app.core.storage import Storage
from app.jobs.queue import get_redis_conn
from app.jobs.repo import JobRepo
from app.utils.audio import ensure_wav_16k_mono
from app.pipelines.asr_whisper import WhisperASR
from app.pipelines.diarization_pyannote import PyannoteDiarizer
from app.pipelines.merge import merge_asr_with_diarization


# Per-process singletons (worker loads once if enabled)
_asr: Optional[WhisperASR] = None
_diarizer: Optional[PyannoteDiarizer] = None


def _get_asr() -> WhisperASR:
    """
    ASR singleton per worker process.
    In PRO we typically set ASR_KEEP_MODEL_LOADED=0 so it unloads after each job
    (shared GPUs with LLM inference).
    """
    global _asr
    if _asr is None:
        s = get_settings()
        _asr = WhisperASR(
            model_path_or_id=s.whisper_model,
            device=s.asr_device,
            compute_type=s.compute_type,
            vram_budget_mb=s.asr_vram_budget_mb,
            vram_headroom_mb=s.asr_vram_headroom_mb,
            keep_model_loaded=s.asr_keep_model_loaded,
            allow_cpu_fallback=s.asr_allow_cpu_fallback,
        )
    return _asr


def _get_diarizer() -> PyannoteDiarizer:
    """
    Diarizer singleton per worker process.
    Forced to CPU in PyannoteDiarizer implementation.
    """
    global _diarizer
    if _diarizer is None:
        s = get_settings()
        _diarizer = PyannoteDiarizer(
            pipeline_path=s.pyannote_pipeline_path,
            device=s.diarization_device,  # will be forced to CPU internally
            enabled=s.diarization_enabled,
        )
    return _diarizer


def _run_asr(wav_path: Path, opts: Dict[str, Any]) -> Dict[str, Any]:
    s = get_settings()
    asr = _get_asr()
    return asr.transcribe(
        wav_path=str(wav_path),
        language=str(opts.get("language", "es")),
        beam_size=int(opts.get("beam_size", 5)),
        vad_filter=bool(opts.get("vad_filter", True)),
        word_timestamps=bool(opts.get("word_timestamps", True)),
    )


def _run_diarization(wav_path: Path, opts: Dict[str, Any]) -> list[dict]:
    diar = _get_diarizer()
    # (Opcional futuro) permitir num_speakers/min/max via opts si lo añades al endpoint
    return diar.diarize(wav_path=str(wav_path))


def process_asr_job(job_id: str, input_path: str, opts: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ worker entrypoint.

    Producción:
    - Valida input (exists + size>0)
    - Normaliza audio
    - Ejecuta ASR (GPU/CPU) y diarización (CPU) en paralelo si se pide
    - Persist result.json
    - Añade timings + info de proceso (pid) para observabilidad
    """
    import time

    s = get_settings()
    storage = Storage(s)
    redis_conn = get_redis_conn(s)
    repo = JobRepo(redis_conn, s)

    # Hard offline defaults en PRO
    if s.env == "pro":
        os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "1"))
        os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "1"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", os.getenv("HF_HUB_DISABLE_TELEMETRY", "1"))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    t0 = time.perf_counter()

    try:
        repo.mark_started(job_id)

        job_dir = storage.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        in_path = Path(input_path)

        # 0) validar input
        if (not in_path.exists()) or (not in_path.is_file()):
            raise FileNotFoundError(f"Input file not found: {in_path}")

        try:
            if in_path.stat().st_size <= 0:
                raise ValueError(f"Input file is empty (0 bytes): {in_path}")
        except Exception as e:
            raise RuntimeError(f"Cannot stat input file: {in_path} ({e})")

        # 1) normalize audio
        t_norm0 = time.perf_counter()
        wav_path = job_dir / "audio_16k_mono.wav"
        ensure_wav_16k_mono(
            input_path=in_path,
            output_path=wav_path,
            ffmpeg_path=s.ffmpeg_path,
        )
        t_norm1 = time.perf_counter()

        diarize_requested = bool(opts.get("diarize", True))
        diarize = diarize_requested and bool(s.diarization_enabled)

        # 2) ASR + diarization
        t_proc0 = time.perf_counter()

        if diarize:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f_asr = ex.submit(_run_asr, wav_path, opts)
                f_diar = ex.submit(_run_diarization, wav_path, opts)

                asr_result: Optional[Dict[str, Any]] = None
                diar_segments: Optional[list[dict]] = None
                first_exc: Optional[BaseException] = None

                for f in as_completed([f_asr, f_diar]):
                    try:
                        r = f.result()
                        if f is f_asr:
                            asr_result = r  # type: ignore[assignment]
                        else:
                            diar_segments = r  # type: ignore[assignment]
                    except BaseException as e:
                        first_exc = e
                        # best-effort cancel
                        try:
                            f_asr.cancel()
                            f_diar.cancel()
                        except Exception:
                            pass
                        break

                if first_exc is not None:
                    raise first_exc

                assert asr_result is not None
                assert diar_segments is not None
                merged = merge_asr_with_diarization(asr_result, diar_segments)

        else:
            merged = _run_asr(wav_path, opts)

        t_proc1 = time.perf_counter()

        # 3) añadir metadatos + timings (no sensibles)
        merged_meta = merged.get("meta") if isinstance(merged, dict) else None
        if not isinstance(merged_meta, dict):
            merged_meta = {}
            if isinstance(merged, dict):
                merged["meta"] = merged_meta

        merged_meta.update(
            {
                "job_id": job_id,
                "pid": os.getpid(),
                "queue_hint": os.getenv("RQ_QUEUE", ""),  # si lo setea tu worker_entry
                "timings_sec": {
                    "normalize": round(t_norm1 - t_norm0, 4),
                    "process": round(t_proc1 - t_proc0, 4),
                    "total": round(time.perf_counter() - t0, 4),
                },
                "opts": {
                    "language": str(opts.get("language", "es")),
                    "diarize": bool(opts.get("diarize", True)),
                    "beam_size": int(opts.get("beam_size", 5)),
                    "vad_filter": bool(opts.get("vad_filter", True)),
                },
            }
        )

        # 4) persist
        storage.write_result_json(job_id, merged)
        repo.mark_finished(job_id)
        return merged

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error("Job failed job_id=%s err=%s\n%s", job_id, err, traceback.format_exc())
        repo.mark_failed(job_id, err)
        raise


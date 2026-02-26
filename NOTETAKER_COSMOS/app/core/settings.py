from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass(frozen=True)
class QueueSpec:
    name: str
    cuda_visible_devices: str  # e.g. "0" or "1" or "" for CPU worker


@dataclass(frozen=True)
class Settings:
    env: str
    redis_url: str
    base_dir: Path
    data_dir: Path

    queues: List[QueueSpec]
    default_queue: str

    # ASR (faster-whisper)
    whisper_model: str          # can be HF id OR local directory path
    compute_type: str           # float16 / int8_float16 / int8
    asr_device: str             # "cuda" or "cpu"
    asr_keep_model_loaded: bool # keep model resident (VRAM) across jobs
    asr_vram_budget_mb: int     # max VRAM we allow for ASR process
    asr_vram_headroom_mb: int   # extra safety free VRAM required beyond budget
    asr_allow_cpu_fallback: bool

    # diarization (pyannote)
    diarization_enabled: bool
    pyannote_pipeline_path: Optional[str]  # local path to pipeline (offline)
    diarization_device: str  # "cuda" or "cpu" (PRO should be cpu)

    # ffmpeg
    ffmpeg_path: str

    # API limits / job config
    allowed_upload_exts: List[str]
    job_timeout_seconds: int
    result_ttl_seconds: int
    failure_ttl_seconds: int
    wait_max_seconds: int


_cached: Optional[Settings] = None


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _try_load_dotenv() -> None:
    """
    Optional .env support for local/prod parity.
    In Kubernetes you'll typically inject env vars, so .env is mainly for local/dev.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    # Prefer CWD .env (when running from repo root)
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env, override=False)

    # Also allow .env next to project base (if running from elsewhere)
    # We don't know base_dir yet here, so just try common parent patterns after base_dir is computed in get_settings.


def get_settings() -> Settings:
    global _cached
    if _cached is not None:
        return _cached

    _try_load_dotenv()

    env = _env("ASR_ENV", "dev")  # dev|pro
    base_dir = Path(_env("ASR_BASE_DIR", str(Path(__file__).resolve().parents[2]))).resolve()

    # second chance: load base_dir/.env if present
    try:
        from dotenv import load_dotenv  # type: ignore

        base_env = base_dir / ".env"
        if base_env.exists():
            load_dotenv(base_env, override=False)
    except Exception:
        pass

    cfg_path = Path(_env("ASR_CONFIG", str(base_dir / "config" / f"{env}.yaml"))).resolve()
    cfg = _load_yaml(cfg_path)

    redis_url = _env("REDIS_URL", cfg.get("redis_url", "redis://localhost:6379/0"))
    data_dir = Path(_env("ASR_DATA_DIR", cfg.get("data_dir", str(base_dir / "data")))).resolve()

    queues_cfg = cfg.get("queues", [])
    queues: List[QueueSpec] = []
    for q in queues_cfg:
        queues.append(
            QueueSpec(
                name=str(q["name"]),
                cuda_visible_devices=str(q.get("cuda_visible_devices", "")),
            )
        )

    default_queue = str(cfg.get("default_queue", queues[0].name if queues else "asr:cpu"))

    # --- ASR settings (defaults safe for shared GPUs)
    whisper_model = str(_env("WHISPER_MODEL", cfg.get("whisper_model", "large-v3")))
    compute_type = str(_env("WHISPER_COMPUTE_TYPE", cfg.get("compute_type", "int8_float16")))
    asr_device = str(_env("ASR_DEVICE", cfg.get("asr_device", "cuda")))

    asr_keep_model_loaded = _env_bool(
        "ASR_KEEP_MODEL_LOADED",
        bool(cfg.get("asr_keep_model_loaded", 0 if env == "pro" else 1)),
    )
    asr_vram_budget_mb = _env_int("ASR_VRAM_BUDGET_MB", int(cfg.get("asr_vram_budget_mb", 5120)))
    asr_vram_headroom_mb = _env_int("ASR_VRAM_HEADROOM_MB", int(cfg.get("asr_vram_headroom_mb", 512)))
    asr_allow_cpu_fallback = _env_bool(
        "ASR_ALLOW_CPU_FALLBACK",
        bool(cfg.get("asr_allow_cpu_fallback", 1)),
    )

    diarization_enabled = bool(int(_env("DIARIZATION_ENABLED", str(int(cfg.get("diarization_enabled", 1))))))
    pyannote_pipeline_path = _env("PYANNOTE_PIPELINE_PATH", cfg.get("pyannote_pipeline_path"))

    # IMPORTANT: production requirement: diarization on CPU (avoid VRAM usage)
    diarization_device = str(_env("DIARIZATION_DEVICE", cfg.get("diarization_device", "cpu" if env == "pro" else "cuda")))

    ffmpeg_path = str(_env("FFMPEG_PATH", cfg.get("ffmpeg_path", "ffmpeg")))

    _cached = Settings(
        env=env,
        redis_url=redis_url,
        base_dir=base_dir,
        data_dir=data_dir,
        queues=queues,
        default_queue=default_queue,
        whisper_model=whisper_model,
        compute_type=compute_type,
        asr_device=asr_device,
        asr_keep_model_loaded=asr_keep_model_loaded,
        asr_vram_budget_mb=asr_vram_budget_mb,
        asr_vram_headroom_mb=asr_vram_headroom_mb,
        asr_allow_cpu_fallback=asr_allow_cpu_fallback,
        diarization_enabled=diarization_enabled,
        pyannote_pipeline_path=pyannote_pipeline_path,
        diarization_device=diarization_device,
        ffmpeg_path=ffmpeg_path,
        allowed_upload_exts=list(
            cfg.get("allowed_upload_exts", [".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".flac", ".webm"])
        ),
        job_timeout_seconds=int(cfg.get("job_timeout_seconds", 60 * 60)),
        result_ttl_seconds=int(cfg.get("result_ttl_seconds", 7 * 24 * 3600)),
        failure_ttl_seconds=int(cfg.get("failure_ttl_seconds", 7 * 24 * 3600)),
        wait_max_seconds=int(cfg.get("wait_max_seconds", 60)),
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    return _cached

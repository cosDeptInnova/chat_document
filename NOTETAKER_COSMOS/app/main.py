from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import ORJSONResponse

from app.core.settings import get_settings
from app.core.logging import setup_logging, logger
from app.core.storage import Storage
from app.core.device_manager import DeviceManager
from app.jobs.queue import get_redis_conn, get_rq_queue
from app.jobs.repo import JobRepo
from app.jobs.schemas import SubmitResponse, StatusResponse
from app.jobs.tasks import process_asr_job
from app.llm_insights.routes import router as insights_router



setup_logging()

app = FastAPI(
    title="Local ASR + Diarization (GPU, Multiuser)",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)
app.include_router(insights_router)

settings = get_settings()
storage = Storage(settings)
redis_conn = get_redis_conn(settings)
job_repo = JobRepo(redis_conn, settings)
device_manager = DeviceManager(settings, redis_conn)

# Content-Type -> extensión (para casos donde filename no trae sufijo)
CONTENT_TYPE_TO_EXT = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "video/mp4": ".mp4",
}


def _pick_extension(upload: UploadFile) -> str:
    """
    Decide la extensión para guardar el archivo:
    1) Si filename trae extensión y está permitida -> úsala
    2) Si no trae o no es válida -> intenta mapear por content-type (audio/mpeg => .mp3)
    3) Si no se puede -> guarda como .bin y deja que ffmpeg detecte por contenido
    """
    filename = upload.filename or ""
    ext = Path(filename).suffix.lower() if filename else ""

    if ext and ext in settings.allowed_upload_exts:
        return ext

    ct = (upload.content_type or "").split(";")[0].strip().lower()
    mapped = CONTENT_TYPE_TO_EXT.get(ct, "")

    if mapped and mapped in settings.allowed_upload_exts:
        return mapped

    # Sin extensión fiable. Aun así, ffmpeg suele poder detectar formato por el stream.
    return ".bin"


@app.post("/v1/asr", response_model=SubmitResponse)
def submit_asr_job(
    file: UploadFile = File(...),
    language: str = Form("es"),
    diarize: bool = Form(True),
    beam_size: int = Form(5),
    vad_filter: bool = Form(True),
    word_timestamps: bool = Form(True),
    wait: bool = Form(False),
):
    # ---- basic validation
    if file is None:
        raise HTTPException(status_code=400, detail="Missing file")

    # Si filename trae una extensión explícita, validamos que esté permitida.
    # Si filename no trae extensión o es genérico, usaremos content-type o .bin.
    filename = file.filename or ""
    explicit_ext = Path(filename).suffix.lower() if filename else ""

    if explicit_ext and explicit_ext not in settings.allowed_upload_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {explicit_ext}")

    # elegimos extensión final (MP3 si content-type audio/mpeg)
    ext = _pick_extension(file)

    job_id = storage.new_job_id()
    job_dir = storage.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{ext}"
    storage.save_upload(file, input_path)

    opts = {
        "language": language,
        "diarize": bool(diarize),
        "beam_size": int(beam_size),
        "vad_filter": bool(vad_filter),
        "word_timestamps": bool(word_timestamps),
    }

    # ---- choose queue/device
    selection = device_manager.choose_queue()
    queue_name = selection.queue_name

    # ---- persist initial status
    job_repo.init_job(job_id=job_id, queue=queue_name, input_path=str(input_path), opts=opts)

    # ---- enqueue
    q = get_rq_queue(settings, redis_conn, queue_name)
    rq_job = q.enqueue(
        process_asr_job,
        job_id,
        str(input_path),
        opts,
        job_timeout=settings.job_timeout_seconds,
        result_ttl=settings.result_ttl_seconds,
        failure_ttl=settings.failure_ttl_seconds,
    )

    logger.info(
        "Enqueued job=%s rq_id=%s queue=%s filename=%s content_type=%s saved_as=%s",
        job_id,
        rq_job.id,
        queue_name,
        filename,
        file.content_type,
        input_path.name,
    )

    if not wait:
        return SubmitResponse(job_id=job_id, status="queued", queue=queue_name)

    # "wait" mode (simple poll loop, still single endpoint)
    import time

    deadline = time.time() + settings.wait_max_seconds
    while time.time() < deadline:
        st = job_repo.get_status(job_id)
        if st.status in ("finished", "failed"):
            break
        time.sleep(0.5)

    st = job_repo.get_status(job_id)
    if st.status == "finished":
        result = storage.read_result_json(job_id)
        return SubmitResponse(job_id=job_id, status="finished", queue=queue_name, result=result)

    if st.status == "failed":
        raise HTTPException(status_code=500, detail={"job_id": job_id, "error": st.error})

    return SubmitResponse(job_id=job_id, status=st.status, queue=queue_name)


@app.get("/v1/asr", response_model=StatusResponse)
def get_asr_status(job_id: str):
    st = job_repo.get_status(job_id)
    if st.status == "not_found":
        raise HTTPException(status_code=404, detail="job_id not found")

    result = None
    if st.status == "finished":
        result = storage.read_result_json(job_id)

    return StatusResponse(
        job_id=job_id,
        status=st.status,
        queue=st.queue,
        created_at=st.created_at,
        started_at=st.started_at,
        finished_at=st.finished_at,
        error=st.error,
        result=result,
    )



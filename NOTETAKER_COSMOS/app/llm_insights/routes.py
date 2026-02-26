from __future__ import annotations
import math
import os
import re
from typing import List, Optional,Literal, Any, Dict, List, Tuple
from fastapi import APIRouter, HTTPException, Body, Request
from fastapi.responses import Response
import base64
from fastapi.responses import FileResponse
import asyncio
import hashlib
from pydantic import ValidationError
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import logging
from collections import Counter
import httpx
from redis import Redis

from .models import (
    RenderPNGRequest,
    NormalizedUtterance,
    MeetingInsights,
    Topic,
    AnalyzeRequest,
    ChunkResult,
    AnalyzeResponse,
)
from .utils import (
    parse_chunk_output_or_fallback,
    parse_meeting_insights_or_fallback,
    extract_segments,
    merge_adjacent,
    compute_talk_stats,
    clamp_handoff_size,
    clip_text,
    make_chart_url,
    save_chart_png,
    coerce_meeting_insights,

    _safe_id_re,

    CONTEXT_WINDOW_TOKENS,
    PROMPT_RESERVED_TOKENS,
    INSIGHTS_CHARTS_DIR
)

from .llm_helpers import (
    resolve_transcript_from_request,
    build_llm_messages_single,
    build_toon_v1,
    call_llama_cpp,
    init_handoff_from_nothing,
    build_toon_summary_v1,
    split_utterances_into_chunks,
    build_llm_messages_chunk,
    build_llm_messages_finalize,
    extract_action_decision_candidates_llm
    
)

logger = logging.getLogger(__name__)

LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "1"))
LLM_SEM = asyncio.Semaphore(max(1, LLM_MAX_CONCURRENCY))
LLAMA_CTX_TOKENS = int(os.getenv("LLAMA_CTX_TOKENS", "18000"))
LLM_CTX_MARGIN = int(os.getenv("LLM_CTX_MARGIN", "256"))
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8014").rstrip("/")
LLAMA_API_MODE: Literal["openai", "llamacpp"] = os.getenv("LLAMA_API_MODE", "openai")  # openai | llamacpp
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "gpt-oss-20b")
LLAMA_TIMEOUT_S = float(os.getenv("LLAMA_TIMEOUT_S", "120"))
_ACTION_CUES = (
    "me pongo un recordatorio", "os mando el correo", "os aviso", "te paso", "te lo paso",
    "haré", "hago", "voy a", "vamos a", "lo implementamos", "lo ponemos en producción",
    "lo habilito", "habilitarlo", "haces pruebas", "hago una batería de pruebas", "hacemos pruebas",
    "te digo", "te digo, fernando", "te envío"
)

_DECISION_CUES = (
    "quedamos en", "para adelante", "decidimos", "acordamos", "empezamos", "empezaríamos",
    "lo ponemos el", "podéis lanzarlo", "empezar con el entero"
)

_ES_WEEKDAYS = {
    "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
    "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6
}

_ws_re = re.compile(r"\s+", re.UNICODE)
_json_object_re = re.compile(r"\{.*\}", re.DOTALL)
_time_re = re.compile(r"\ba\s+las?\s+(\d{1,2})(?:[:\.](\d{2}))?\b", re.IGNORECASE)
_daynum_re = re.compile(r"\bd[ií]a\s+(\d{1,2})\b", re.IGNORECASE)
_weekday_re = re.compile(r"\b(lunes|martes|mi[eé]rcoles|jueves|viernes|s[áa]bado|domingo)\b", re.IGNORECASE)
_menos_re = re.compile(r"\b(\d{1,2})\s+menos\s+(\d{1,2})\b", re.IGNORECASE)

HYBRID_INGEST_ENABLED = os.getenv("HYBRID_INGEST_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
HYBRID_INGEST_URL = os.getenv("HYBRID_INGEST_URL", "http://localhost:8001/ingest").strip()
HYBRID_INGEST_CONNECT_TIMEOUT_S = float(os.getenv("HYBRID_INGEST_CONNECT_TIMEOUT_S", "3"))
HYBRID_INGEST_READ_TIMEOUT_S = float(os.getenv("HYBRID_INGEST_READ_TIMEOUT_S", "30"))
HYBRID_INGEST_RETRIES = max(1, int(os.getenv("HYBRID_INGEST_RETRIES", "3")))
HYBRID_INGEST_RETRY_BACKOFF_S = float(os.getenv("HYBRID_INGEST_RETRY_BACKOFF_S", "0.8"))
HYBRID_INGEST_IDEMPOTENCY_TTL_S = max(300, int(os.getenv("HYBRID_INGEST_IDEMPOTENCY_TTL_S", str(30 * 24 * 3600))))

_INGEST_REDIS: Optional[Redis] = None
try:
    from app.core.settings import get_settings
    from app.jobs.queue import get_redis_conn

    _INGEST_REDIS = get_redis_conn(get_settings())
except Exception as exc:
    logger.warning("Hybrid ingest idempotency with Redis is unavailable: %s", exc)


def _meeting_dedupe_key(meeting_id: str) -> str:
    stable = hashlib.sha256(meeting_id.encode("utf-8", errors="ignore")).hexdigest()[:32]
    return f"insights:hybrid_ingest:meeting:{stable}"


def _extract_meeting_id_for_ingest(analyze_request: AnalyzeRequest, analyze_response: AnalyzeResponse) -> Optional[str]:
    for candidate in (
        analyze_request.meeting_id,
        getattr(analyze_response, "charts_id", None),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _build_hybrid_ingest_payload(
    analyze_request: AnalyzeRequest,
    analyze_response: AnalyzeResponse,
    meeting_id: str,
    *,
    passthrough_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = analyze_response.model_dump(mode="json")
    payload["meeting_id"] = meeting_id
    payload.setdefault("language", analyze_request.language)
    if passthrough_meta:
        payload.update({k: v for k, v in passthrough_meta.items() if v is not None})
    return payload


async def _send_to_hybrid_ingest_once(meeting_id: str, payload: Dict[str, Any]) -> None:
    if not HYBRID_INGEST_ENABLED:
        logger.info("Hybrid ingest disabled. meeting_id=%s", meeting_id)
        return

    dedupe_key = _meeting_dedupe_key(meeting_id)
    if _INGEST_REDIS is not None:
        try:
            acquired = bool(_INGEST_REDIS.set(dedupe_key, "processing", nx=True, ex=HYBRID_INGEST_IDEMPOTENCY_TTL_S))
            if not acquired:
                current_status = _INGEST_REDIS.get(dedupe_key)
                logger.info(
                    "Hybrid ingest skipped by idempotency gate. meeting_id=%s redis_status=%s",
                    meeting_id,
                    current_status.decode("utf-8", errors="ignore") if current_status else "unknown",
                )
                return
        except Exception as exc:
            logger.exception("Hybrid ingest idempotency check failed for meeting_id=%s: %s", meeting_id, exc)

    timeout = httpx.Timeout(
        connect=HYBRID_INGEST_CONNECT_TIMEOUT_S,
        read=HYBRID_INGEST_READ_TIMEOUT_S,
        write=HYBRID_INGEST_READ_TIMEOUT_S,
        pool=HYBRID_INGEST_CONNECT_TIMEOUT_S,
    )

    final_error: Optional[str] = None
    for attempt in range(1, HYBRID_INGEST_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(HYBRID_INGEST_URL, json=payload)

            if resp.status_code in (200, 201):
                logger.info("Hybrid ingest success meeting_id=%s attempt=%d", meeting_id, attempt)
                if _INGEST_REDIS is not None:
                    _INGEST_REDIS.set(dedupe_key, "done", ex=HYBRID_INGEST_IDEMPOTENCY_TTL_S)
                return

            if resp.status_code == 409:
                logger.info("Hybrid ingest already exists for meeting_id=%s (409) attempt=%d", meeting_id, attempt)
                if _INGEST_REDIS is not None:
                    _INGEST_REDIS.set(dedupe_key, "done", ex=HYBRID_INGEST_IDEMPOTENCY_TTL_S)
                return

            final_error = f"http_status={resp.status_code} body={resp.text[:400]}"
            logger.warning(
                "Hybrid ingest non-success response meeting_id=%s attempt=%d detail=%s",
                meeting_id,
                attempt,
                final_error,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            final_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Hybrid ingest transient error meeting_id=%s attempt=%d error=%s", meeting_id, attempt, exc)
        except Exception as exc:
            final_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Hybrid ingest unexpected error meeting_id=%s attempt=%d", meeting_id, attempt)
            break

        if attempt < HYBRID_INGEST_RETRIES:
            await asyncio.sleep(HYBRID_INGEST_RETRY_BACKOFF_S * attempt)

    if _INGEST_REDIS is not None:
        try:
            _INGEST_REDIS.delete(dedupe_key)
        except Exception:
            logger.exception("Failed to release ingest idempotency key for meeting_id=%s", meeting_id)

    logger.error("Hybrid ingest failed meeting_id=%s after retries. last_error=%s", meeting_id, final_error)


async def _trigger_hybrid_ingest(
    analyze_request: AnalyzeRequest,
    analyze_response: AnalyzeResponse,
    *,
    passthrough_meta: Optional[Dict[str, Any]] = None,
) -> None:
    meeting_id = _extract_meeting_id_for_ingest(analyze_request, analyze_response)
    if not meeting_id:
        logger.info("Hybrid ingest skipped: meeting_id unavailable")
        return

    payload = _build_hybrid_ingest_payload(
        analyze_request=analyze_request,
        analyze_response=analyze_response,
        meeting_id=meeting_id,
        passthrough_meta=passthrough_meta,
    )
    await _send_to_hybrid_ingest_once(meeting_id, payload)




def _next_weekday(d: datetime, weekday: int, *, inclusive: bool) -> datetime:
    delta = (weekday - d.weekday()) % 7
    if delta == 0 and not inclusive:
        delta = 7
    return d + timedelta(days=delta)


def parse_due_datetime_es(text: str, meeting_dt: Optional[datetime], tz: ZoneInfo) -> Optional[datetime]:
    """
    Parse determinista de due datetime en español (relativo a meeting_dt).
    Mejoras:
      - Soporta "por la mañana/tarde/noche", "mediodía", "final del día".
      - Heurística: "a las 3" + "tarde" => 15:00.
      - Mantiene comportamiento conservador si no hay fecha.
    """
    if not meeting_dt:
        return None

    base = meeting_dt.astimezone(tz)
    t = (text or "").lower()
    day = base.date()
    due_date = None

    # ---------- fecha ----------
    if "pasado mañana" in t or "pasadomañana" in t:
        due_date = (base + timedelta(days=2)).date()
    elif "mañana" in t:
        due_date = (base + timedelta(days=1)).date()
    elif "hoy" in t:
        due_date = base.date()

    m_daynum = _daynum_re.search(t)
    if due_date is None and m_daynum:
        dn = int(m_daynum.group(1))
        y, mo = day.year, day.month
        try:
            cand = datetime(y, mo, dn, tzinfo=tz).date()
        except Exception:
            cand = None
        if cand is None:
            return None
        if cand < day:
            mo2 = 1 if mo == 12 else mo + 1
            y2 = y + 1 if mo == 12 else y
            try:
                cand = datetime(y2, mo2, dn, tzinfo=tz).date()
            except Exception:
                return None
        due_date = cand

    m_wd = _weekday_re.search(t)
    if due_date is None and m_wd:
        wd = _ES_WEEKDAYS.get(m_wd.group(1).lower().replace("é", "e").replace("á", "a"))
        if wd is not None:
            inclusive = ("este" in t) or ("hoy" in t)
            due_date = _next_weekday(base, wd, inclusive=inclusive).date()

    if due_date is None:
        return None

    # ---------- hora (defaults “profesionales”) ----------
    # Default si hay fecha pero no hora: 09:00 (igual que antes)
    hh, mm = 9, 0

    # “ventanas” semánticas (si no hay hora explícita)
    # Nota: solo aplica si no encontramos "a las ..."
    sem_time = None
    if any(x in t for x in ("por la mañana", "por la manana", "esta mañana", "esta manana", "a primera hora")):
        sem_time = (10, 0) if "primera hora" not in t else (9, 0)
    elif any(x in t for x in ("mediodía", "mediodia", "al mediodía", "al mediodia")):
        sem_time = (12, 0)
    elif any(x in t for x in ("por la tarde", "esta tarde", "por la tarde.")):
        sem_time = (16, 0)
    elif any(x in t for x in ("al final del día", "al final del dia", "fin del día", "fin del dia")):
        sem_time = (18, 0)
    elif any(x in t for x in ("por la noche", "esta noche")):
        sem_time = (20, 0)

    # “menos” (tres menos cuarto)
    m_menos = _menos_re.search(t)
    if m_menos:
        h0 = int(m_menos.group(1)) % 24
        mins = int(m_menos.group(2))
        mm = (60 - mins) % 60
        hh = (h0 - 1) % 24 if mins > 0 else h0
        return datetime(due_date.year, due_date.month, due_date.day, hh, mm, tzinfo=tz)

    # “a las …”
    m_time = _time_re.search(t)
    if m_time:
        hh = max(0, min(23, int(m_time.group(1))))
        mm = int(m_time.group(2) or 0)
        mm = max(0, min(59, mm))

        # Heurística AM/PM por contexto ES si no hay 24h:
        # Si dice “tarde/noche” y la hora es 1..7, asumimos PM (sumar 12).
        if hh <= 7 and any(k in t for k in ("tarde", "noche", "pm")) and hh < 12:
            hh = hh + 12

        return datetime(due_date.year, due_date.month, due_date.day, hh, mm, tzinfo=tz)

    # Sin hora explícita: usa sem_time si hay, si no default 09:00
    if sem_time:
        hh, mm = sem_time

    return datetime(due_date.year, due_date.month, due_date.day, hh, mm, tzinfo=tz)

def extract_facts_deterministic(
    segments_norm: List[Dict[str, Any]],
    *,
    tz_name: str = "Europe/Madrid",
) -> Dict[str, Any]:
    """
    Hechos deterministas (sin LLM):
      - meeting_start_iso (si hay absolute_start_time)
      - participants
      - engagement_talk_seconds_by_hour
      - candidates se rellenan después (LLM extractor) o quedan vacíos
    """
    tz = ZoneInfo(tz_name)

    abs_times = [_parse_iso(s.get("absolute_start_time")) for s in segments_norm]
    abs_times = [x for x in abs_times if x is not None]
    meeting_dt = min(abs_times).astimezone(tz) if abs_times else None

    speakers = sorted({(s.get("speaker") or "").strip() for s in segments_norm if s.get("speaker")})

    talk_by_hour: Dict[str, float] = {}
    if meeting_dt:
        for s in segments_norm:
            abs_s = _parse_iso(s.get("absolute_start_time"))
            abs_e = _parse_iso(s.get("absolute_end_time"))
            if not abs_s or not abs_e:
                continue
            abs_s = abs_s.astimezone(tz)
            abs_e = abs_e.astimezone(tz)
            dur = max(0.0, (abs_e - abs_s).total_seconds())

            # bucket simple por hora de inicio (si quieres precisión, luego lo partimos por horas)
            hour_key = f"{abs_s.hour:02d}:00"
            talk_by_hour[hour_key] = talk_by_hour.get(hour_key, 0.0) + dur

    return {
        "meeting_start_iso": meeting_dt.isoformat() if meeting_dt else None,
        "timezone": tz_name,
        "participants": speakers,
        "engagement_talk_seconds_by_hour": {k: round(v, 1) for k, v in sorted(talk_by_hour.items())},
        "deterministic_action_candidates": [],
        "deterministic_decision_candidates": [],
    }


def _parse_iso(dt_s: Optional[str]) -> Optional[datetime]:
    if not dt_s or not isinstance(dt_s, str):
        return None
    s = dt_s.strip()
    if not s:
        return None
    # soporta Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _normalize_text_key(t: str) -> str:
    t = (t or "").strip().lower()
    t = _ws_re.sub(" ", t)
    # clave corta para dedupe (sin pasarnos)
    return t[:160]

def normalize_segments_preserve_abs(raw_segments: Any) -> List[Dict[str, Any]]:
    """
    - Normaliza speaker/text/start/end
    - Dedup de segmentos muy solapados con mismo speaker+texto
    - Preserva absolute_start_time/absolute_end_time si vienen
    """
    if not isinstance(raw_segments, list):
        return []

    out: List[Dict[str, Any]] = []
    for it in raw_segments:
        if not isinstance(it, dict):
            continue

        speaker = _norm_speaker(it.get("speaker"))
        text = (it.get("text") or "").strip()
        if not text:
            continue

        start = _to_float_time(it.get("start"))
        end = _to_float_time(it.get("end"))

        # algunos payloads vienen con start_time/end_time
        if start is None:
            start = _to_float_time(it.get("start_time") or it.get("startTime"))
        if end is None:
            end = _to_float_time(it.get("end_time") or it.get("endTime"))

        if start is None:
            # si no hay start, lo saltamos (preferible a inventar timeline global)
            continue

        if end is None or end < start:
            dur = _to_float_time(it.get("duration"))
            if dur is not None and dur > 0:
                end = start + dur
            else:
                end = start + _estimate_duration_from_text(text)

        abs_s = _parse_iso(it.get("absolute_start_time"))
        abs_e = _parse_iso(it.get("absolute_end_time"))

        out.append({
            "speaker": speaker,
            "text": text,
            "start": float(start),
            "end": float(end),
            "absolute_start_time": abs_s.isoformat() if abs_s else None,
            "absolute_end_time": abs_e.isoformat() if abs_e else None,
        })

    out.sort(key=lambda s: (s["start"], s["end"]))

    # Dedup: mismo speaker+texto y solape grande => quedarse con el más largo
    deduped: List[Dict[str, Any]] = []
    last_by_key: Dict[str, int] = {}
    for s in out:
        key = f'{s["speaker"]}::{_normalize_text_key(s["text"])}'
        j = last_by_key.get(key)
        if j is None:
            last_by_key[key] = len(deduped)
            deduped.append(s)
            continue

        prev = deduped[j]
        # overlap ratio sobre el menor intervalo
        a0, a1 = prev["start"], prev["end"]
        b0, b1 = s["start"], s["end"]
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        denom = max(1e-6, min(a1 - a0, b1 - b0))
        if inter / denom >= 0.85:
            # mantener el más largo
            if (b1 - b0) > (a1 - a0):
                deduped[j] = s
            continue

        # si no se solapan tanto, se consideran ocurrencias distintas
        last_by_key[key] = len(deduped)
        deduped.append(s)

    return deduped


def compute_talk_stats_deterministic(utterances: List[NormalizedUtterance]) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
    """
    - talk_time_seconds: unión de intervalos por speaker (evita doble conteo por solapes)
    - turns: nº de utterances por speaker (ya post-merge si activado)
    - participation_percent: derivado de talk_time (o turns si total=0), suma 100 estable
    """
    intervals: Dict[str, List[Tuple[float, float]]] = {}
    turns: Dict[str, int] = {}

    for u in utterances or []:
        spk = _norm_speaker(u.speaker)
        st = float(u.start or 0.0)
        en = float(u.end or st)
        if en < st:
            st, en = en, st
        if en - st <= 1e-6:
            continue
        intervals.setdefault(spk, []).append((st, en))
        turns[spk] = turns.get(spk, 0) + 1

    talk: Dict[str, float] = {}
    for spk, ivs in intervals.items():
        ivs.sort()
        merged: List[Tuple[float, float]] = []
        for a, b in ivs:
            if not merged or a > merged[-1][1]:
                merged.append((a, b))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        talk[spk] = sum(max(0.0, b - a) for a, b in merged)

    speakers = sorted(set(list(talk.keys()) + list(turns.keys())))
    if not speakers:
        return {}, {}, {}

    total = sum(talk.get(s, 0.0) for s in speakers)
    if total <= 1e-6:
        total_turns = sum(turns.get(s, 0) for s in speakers) or 1
        pct = {s: 100.0 * (turns.get(s, 0) / total_turns) for s in speakers}
    else:
        pct = {s: 100.0 * (talk.get(s, 0.0) / total) for s in speakers}

    pct = {k: round(float(v), 1) for k, v in pct.items()}
    ssum = round(sum(pct.values()), 1)
    diff = round(100.0 - ssum, 1)
    if abs(diff) >= 0.1:
        kmax = max(pct.keys(), key=lambda k: pct[k])
        pct[kmax] = round(pct[kmax] + diff, 1)

    talk = {k: float(v) for k, v in talk.items()}
    turns = {k: int(v) for k, v in turns.items()}
    return talk, turns, pct

def compute_meeting_signals_deterministic(
    *,
    segments: List[Dict[str, Any]],
    talk_time_seconds: Dict[str, float],
    turns: Dict[str, int],
    participation_percent: Dict[str, float],
    decisions: Any,
    action_items: Any,
) -> Dict[str, Any]:
    """
    Calcula señales SIN LLM:
      - collaboration.score_0_100
      - decisiveness.score_0_100
      - conflict_level_0_100
      - atmosphere.valence + labels (fallback)

    Y genera notes explicables:
      - 1 línea "por qué" (métricas)
      - 1–2 líneas con evidencia: 't=xx.x: ...'
    """

    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _clamp100(x: float) -> float:
        return max(0.0, min(100.0, float(x)))

    def _to_f(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _safe_text(s: Any) -> str:
        return _ws_re.sub(" ", (str(s) if s is not None else "").strip())

    def _gini(values: List[float]) -> float:
        xs = [max(0.0, float(v)) for v in values if v is not None]
        n = len(xs)
        if n <= 1:
            return 0.0
        s = sum(xs)
        if s <= 1e-9:
            return 0.0
        xs.sort()
        cum = 0.0
        for i, x in enumerate(xs, start=1):
            cum += i * x
        g = (2.0 * cum) / (n * s) - (n + 1.0) / n
        return max(0.0, min(1.0, g))

    def _meeting_span_seconds(segs: List[Dict[str, Any]]) -> float:
        if not segs:
            return 0.0
        starts = []
        ends = []
        for s in segs:
            if not isinstance(s, dict):
                continue
            starts.append(_to_f(s.get("start"), 0.0))
            ends.append(_to_f(s.get("end"), 0.0))
        if not starts or not ends:
            return 0.0
        span = max(0.0, max(ends) - min(starts))
        if span <= 1e-6:
            # fallback: suma de habla
            span = sum(max(0.0, _to_f(v)) for v in (talk_time_seconds or {}).values())
        return span

    def _count_speaker_switches(segs: List[Dict[str, Any]]) -> int:
        if not segs:
            return 0
        ordered = sorted(
            [s for s in segs if isinstance(s, dict)],
            key=lambda x: (_to_f(x.get("start"), 0.0), _to_f(x.get("end"), 0.0)),
        )
        prev = None
        switches = 0
        for s in ordered:
            sp = _safe_text(s.get("speaker") or "")
            if not sp:
                continue
            if prev is None:
                prev = sp
                continue
            if sp != prev:
                switches += 1
                prev = sp
        return switches

    def _overlap_seconds(segs: List[Dict[str, Any]]) -> float:
        """
        Aproximación robusta (O(n log n) sin estructuras complejas):
        cuenta solape entre segmentos ADYACENTES ordenados cuando cambia speaker.
        """
        ordered = sorted(
            [s for s in segs if isinstance(s, dict)],
            key=lambda x: (_to_f(x.get("start"), 0.0), _to_f(x.get("end"), 0.0)),
        )
        if len(ordered) < 2:
            return 0.0
        ov = 0.0
        prev = ordered[0]
        for cur in ordered[1:]:
            ps = _safe_text(prev.get("speaker") or "")
            cs = _safe_text(cur.get("speaker") or "")
            p0, p1 = _to_f(prev.get("start")), _to_f(prev.get("end"))
            c0, c1 = _to_f(cur.get("start")), _to_f(cur.get("end"))
            inter = max(0.0, min(p1, c1) - max(p0, c0))
            if inter > 0 and ps and cs and ps != cs:
                ov += inter
            # mantener como "prev" el que termina más tarde (sweep local)
            prev = prev if _to_f(prev.get("end")) >= _to_f(cur.get("end")) else cur
        return ov

    # --- cues léxicos (ES) ---
    ACK = ("vale", "ok", "okay", "perfecto", "de acuerdo", "correcto", "gracias", "me parece", "genial", "bien")
    DECIDE = ("decidimos", "acordamos", "queda", "cerramos", "vamos a", "haré", "hare", "haremos", "lo hago", "lo hacemos")
    PLAN = ("plazo", "deadline", "entrega", "prioridad", "siguiente", "próximo", "proximo", "plan", "tarea", "acción", "accion")
    DISAGREE = ("no estoy de acuerdo", "discrepo", "eso no", "no podemos", "no tiene sentido", "bloqueo", "riesgo", "problema", "tensión", "tension")

    def _hits(txt_low: str, phrases: tuple) -> int:
        n = 0
        for p in phrases:
            if not p:
                continue
            n += txt_low.count(p)
        return n

    full_text = " ".join(
        ((s.get("text") or "") for s in (segments or []) if isinstance(s, dict))
    )
    full_low = full_text.lower()

    pos = sum(1 for w in _POS if w in full_low)
    neg = sum(1 for w in _NEG if w in full_low)
    ack_hits = _hits(full_low, ACK)
    decide_hits = _hits(full_low, DECIDE)
    plan_hits = _hits(full_low, PLAN)
    disagree_hits = _hits(full_low, DISAGREE)

    # también cuenta referencias a tiempo/fecha (ya tienes regex compiladas en el módulo)
    time_mentions = len(_time_re.findall(full_low)) + len(_weekday_re.findall(full_low)) + len(_daynum_re.findall(full_low))

    # --- features temporales ---
    span_s = _meeting_span_seconds(segments)
    minutes = max(1e-6, span_s / 60.0)

    switches = _count_speaker_switches(segments)
    switch_rate = switches / minutes  # cambios de turno por minuto

    ov_s = _overlap_seconds(segments)
    ov_ratio = (ov_s / span_s) if span_s > 1e-6 else 0.0

    # --- balance participación ---
    pvals = [float(v) for v in (participation_percent or {}).values() if v is not None]
    n_part = len(pvals) if pvals else len({(s.get("speaker") or "") for s in (segments or []) if isinstance(s, dict) and s.get("speaker")})
    n_part = max(1, int(n_part))

    top_share = (max(pvals) / 100.0) if pvals else 1.0
    g = _gini(pvals) if pvals else 0.0
    max_g = (n_part - 1.0) / n_part if n_part > 1 else 0.0
    balance_score = 100.0 if max_g <= 1e-9 else 100.0 * (1.0 - (g / max_g))
    balance_score = _clamp100(balance_score)

    # --- COLLABORATION ---
    # normalizaciones razonables (calibrables):
    turn_norm = _clamp01(switch_rate / 6.0)      # ~6 cambios/min = alto
    ack_norm = _clamp01((ack_hits / minutes) / 3.0)  # ~3 asentimientos/min = alto

    collaboration = (
        0.60 * balance_score +
        25.0 * turn_norm +
        15.0 * ack_norm
    )
    collaboration = _clamp100(collaboration)

    # --- DECISIVENESS ---
    def _len_any(xs: Any) -> int:
        if xs is None:
            return 0
        if isinstance(xs, list):
            return len(xs)
        return 0

    n_decisions = _len_any(decisions)
    n_actions = _len_any(action_items)

    dec_act_norm = _clamp01((1.2 * n_decisions + 0.8 * n_actions) / 6.0)
    cue_norm = _clamp01((decide_hits + plan_hits + time_mentions) / 14.0)

    decisiveness = 25.0 + 55.0 * dec_act_norm + 20.0 * cue_norm
    decisiveness = _clamp100(decisiveness)

    # --- CONFLICT ---
    # mezcla desacuerdo explícito + neg + solapes (interrupciones)
    disagree_norm = _clamp01(disagree_hits / 6.0)
    neg_norm = _clamp01(neg / 10.0)
    ov_norm = _clamp01(ov_ratio / 0.06)  # 6% del tiempo solapado ya es “alto” en reuniones

    conflict = 5.0 + 55.0 * disagree_norm + 25.0 * neg_norm + 20.0 * ov_norm
    conflict = _clamp100(conflict)

    # --- ATMOSPHERE (fallback determinista) ---
    denom = max(1, pos + neg)
    valence = _clamp((pos - neg) / denom, -1.0, 1.0)

    labels: List[str] = []
    if abs(valence) < 0.15:
        labels.append("neutro")
    elif valence > 0.2:
        labels.append("positivo")
    else:
        labels.append("negativo")

    if plan_hits + time_mentions > 0:
        labels.append("enfocado")

    if conflict >= 55.0:
        labels.append("tenso")
    elif conflict >= 25.0 and (neg > 0 or disagree_hits > 0):
        labels.append("fricción leve")
    elif conflict < 15.0 and valence > 0.25:
        labels.append("calmado")

    # --- evidencia (líneas t=xx.x) ---
    def _pick_evidence_line(keywords: Tuple[str, ...]) -> Optional[str]:
        for s in segments or []:
            if not isinstance(s, dict):
                continue
            txt = (s.get("text") or "")
            low = txt.lower()
            if any(k in low for k in keywords):
                return f"t={float(s.get('start') or 0.0):.1f}: {clip_text(txt, 120)}"
        return None

    coll_ev = _pick_evidence_line(("de acuerdo", "vale", "ok", "perfecto", "gracias", "me parece", "correcto"))
    dec_ev = _pick_evidence_line(("decidimos", "acordamos", "queda", "cerramos", "vamos a", "plazo", "deadline", "entrega"))
    conf_ev = _pick_evidence_line(("no estoy de acuerdo", "discrepo", "bloqueo", "riesgo", "problema", "tensión", "tension"))

    coll_notes = [
        f"Balance de participación {balance_score:.0f}/100 (top {top_share*100:.1f}%, gini {g:.2f}); turn-taking {switch_rate:.1f} cambios/min; asentimientos {ack_hits}.",
    ]
    if coll_ev:
        coll_notes.append(coll_ev)

    dec_notes = [
        f"Decisiones {n_decisions} y acciones {n_actions}; cues decisión/planificación {decide_hits + plan_hits} y menciones de fecha/hora {time_mentions}.",
    ]
    if dec_ev:
        dec_notes.append(dec_ev)

    conf_notes = [
        f"Desacuerdos explícitos {disagree_hits}, negativos {neg}; solape {ov_ratio*100:.1f}% (~{ov_s:.1f}s) como proxy de interrupciones.",
    ]
    if conf_ev:
        conf_notes.append(conf_ev)

    atmo_notes = []
    atmo_notes.append(f"Valencia {valence:.2f} (pos {pos}, neg {neg}); etiquetas: {', '.join(labels[:3]) if labels else 'neutro'}.")
    if conf_ev:
        atmo_notes.append(conf_ev)

    # cap notes a 3
    coll_notes = coll_notes[:3]
    dec_notes = dec_notes[:3]
    conf_notes = conf_notes[:3]
    atmo_notes = atmo_notes[:3]

    return {
        "collaboration_score": float(collaboration),
        "collaboration_notes": coll_notes,
        "decisiveness_score": float(decisiveness),
        "decisiveness_notes": dec_notes,
        "conflict_score": float(conflict),
        "conflict_notes": conf_notes,  # se mapeará a decisiveness/atmosphere si tu modelo no tiene campo notes para conflicto
        "valence": float(valence),
        "labels": labels,
        "atmosphere_notes": atmo_notes,
    }

def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(lo, min(hi, v))

def _norm_label_es(lbl: str) -> str:
    s = (lbl or "").strip()
    if not s:
        return ""
    low = s.lower()
    mapping = {
        "neutral": "neutro",
        "positive": "positivo",
        "negative": "negativo",
        "tense": "tenso",
        "calm": "calmado",
        "focused": "enfocado",
    }
    return mapping.get(low, s)

def _is_llm_default_metrics(ins: MeetingInsights) -> bool:
    """
    Detecta "métricas default" aunque el parser haya metido notes tipo
    "Estimación por defecto." (caso que ahora bloquea tu fallback).
    """
    def _notes_are_placeholder(notes: Any) -> bool:
        if not notes:
            return True
        if not isinstance(notes, list):
            return True
        bad = (
            "estimación por defecto",
            "estimacion por defecto",
            "default",
            "n/a",
            "sin evidencia",
            "no disponible",
        )
        kept = []
        for n in notes:
            s = (str(n) if n is not None else "").strip().lower()
            if not s:
                continue
            kept.append(s)
        if not kept:
            return True
        # si TODAS son placeholder, cuenta como vacío
        return all(any(b in s for b in bad) for s in kept)

    try:
        c = float(getattr(ins.collaboration, "score_0_100", 0.0) or 0.0)
        d = float(getattr(ins.decisiveness, "score_0_100", 0.0) or 0.0)
        cf = float(getattr(ins, "conflict_level_0_100", 0.0) or 0.0)
        v = float(getattr(ins.atmosphere, "valence", 0.0) or 0.0)
        labels = getattr(ins.atmosphere, "labels", []) or []

        coll_notes_ph = _notes_are_placeholder(getattr(ins.collaboration, "notes", None))
        deci_notes_ph = _notes_are_placeholder(getattr(ins.decisiveness, "notes", None))
        atmo_notes_ph = _notes_are_placeholder(getattr(ins.atmosphere, "notes", None))

        labels_low = [str(x).strip().lower() for x in labels if str(x).strip()]
        labels_is_default = (labels_low == ["neutro"]) or (labels_low == [])  # a veces el modelo lo deja vacío

        looks_like_defaults = (
            (c in (0.0, 50.0)) and
            (d in (0.0, 50.0)) and
            (cf in (0.0, 20.0)) and
            (abs(v) < 1e-6) and
            labels_is_default
        )

        # Si las notas NO aportan evidencia real, lo tratamos como default
        notes_no_evidence = coll_notes_ph and deci_notes_ph and atmo_notes_ph

        return bool(looks_like_defaults and notes_no_evidence)
    except Exception:
        return True


_POS = {"bien", "perfecto", "genial", "ok", "vale", "gracias", "correcto", "buen"}
_NEG = {"problema", "riesgo", "bloqueo", "tensión", "mal", "error", "queja"}
_CONFLICT = {"no estoy de acuerdo", "discrepo", "eso no", "no podemos", "bloqueo", "riesgo"}

def _heuristic_metrics_from_text(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    txt = " ".join(((s.get("text") or "") for s in (segments or []) if isinstance(s, dict))).lower()

    pos = sum(1 for w in _POS if w in txt)
    neg = sum(1 for w in _NEG if w in txt)
    conf_hits = sum(1 for p in _CONFLICT if p in txt)

    # Cues de foco/avance (enfocado)
    focus_hits = sum(
        1 for w in (
            "siguiente", "próximo", "proximo", "plazo", "deadline", "entrega", "prioridad",
            "plan", "tarea", "acción", "accion", "bloqueo", "riesgo", "vamos a", "cerramos", "quedamos"
        )
        if w in txt
    )

    # valence [-1,1]
    denom = max(1, pos + neg)
    valence = _clamp((pos - neg) / denom, -1.0, 1.0)

    # labels (dedupe luego)
    labels: List[str] = []
    if abs(valence) < 0.15:
        labels.append("neutro")
    elif valence > 0.2:
        labels.append("positivo")
    else:
        labels.append("negativo")

    if focus_hits > 0:
        labels.append("enfocado")

    if conf_hits > 0:
        labels.append("tenso")
    elif neg > 0:
        labels.append("fricción leve")

    # conflict score
    conflict = _clamp(8.0 + conf_hits * 20.0 + neg * 7.0, 0.0, 100.0)

    # decisiveness proxy
    decisive_cues = sum(1 for w in ("decidimos", "queda", "acordamos", "vamos a", "plazo", "cerramos", "siguiente") if w in txt)
    decisiveness = _clamp(28.0 + decisive_cues * 14.0 + min(12.0, focus_hits * 3.0), 0.0, 100.0)

    # collaboration proxy (básico por speakers + turnos)
    speakers = []
    for s in segments or []:
        if isinstance(s, dict):
            sp = (s.get("speaker") or "").strip()
            if sp:
                speakers.append(sp)
    unique = len(set(speakers))
    turns_total = len(segments or [])
    collaboration = _clamp(34.0 + min(30.0, unique * 6.5) + min(26.0, turns_total / 9.0), 0.0, 100.0)

    return {
        "collaboration": collaboration,
        "decisiveness": decisiveness,
        "conflict": conflict,
        "valence": valence,
        "labels": labels,
    }

def postprocess_meeting_insights(
    insights: MeetingInsights,
    *,
    segments: List[Dict[str, Any]],
    talk_time_seconds: Dict[str, float],
    turns: Dict[str, int],
    participation_percent: Dict[str, float],
) -> MeetingInsights:
    """
    Profesionaliza señales:
      - Stats deterministas = fuente de verdad.
      - Limpieza fuerte de action_items/decisions (ya la tenías).
      - Señales (colaboración/decisión/conflicto) deterministas + notas explicables.
      - Atmosphere: usa LLM si trae evidencia; si viene default/vacío -> fallback determinista.
      - Summary: si es placeholder/genérico -> se expande con datos reales.
    """
    ins = insights.model_copy(deep=True)

    # 1) stats: sobrescribe siempre
    ins.talk_time_seconds = dict(talk_time_seconds or {})
    ins.turns = dict(turns or {})
    ins.participation_percent = dict(participation_percent or {})

    valid_speakers = list((participation_percent or {}).keys())
    valid_speakers_set = set(valid_speakers)

    def _norm_simple(s: str) -> str:
        s = (s or "").strip().lower()
        s = _ws_re.sub(" ", s)
        s = (s.replace("á", "a").replace("é", "e").replace("í", "i")
               .replace("ó", "o").replace("ú", "u").replace("ü", "u").replace("ñ", "n"))
        return s

    def _best_match_speaker(name: Optional[str]) -> Optional[str]:
        if not isinstance(name, str):
            return None
        n = name.strip()
        if not n:
            return None
        if n in valid_speakers_set:
            return n
        nkey = _norm_simple(n)
        n_tokens = {x for x in re.findall(r"[a-z0-9]+", nkey) if len(x) >= 3}
        best, best_score = None, 0.0
        for cand in valid_speakers:
            ckey = _norm_simple(cand)
            c_tokens = {x for x in re.findall(r"[a-z0-9]+", ckey) if len(x) >= 3}
            if not c_tokens:
                continue
            inter = len(n_tokens & c_tokens)
            denom = max(1, min(len(n_tokens) or 1, len(c_tokens)))
            score = inter / denom
            if score > best_score:
                best_score = score
                best = cand
        return best if (best and best_score >= 0.70) else None

    def _safe_due(due: Any) -> Optional[str]:
        if due is None:
            return None
        if isinstance(due, datetime):
            return due.isoformat()
        if isinstance(due, str):
            s = due.strip()
            if not s:
                return None
            ss = s[:-1] + "+00:00" if s.endswith("Z") else s
            try:
                _ = datetime.fromisoformat(ss)
                return s
            except Exception:
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                    return s
                return None
        return None

    def _clean_text(s: Any, maxlen: int) -> str:
        t = (str(s) if s is not None else "").strip()
        t = _ws_re.sub(" ", t)
        return clip_text(t, maxlen)

    # --- limpieza actions ---
    def _clean_actions() -> None:
        raw_items = getattr(ins, "action_items", None) or []
        cleaned = []
        for it in raw_items:
            owner = getattr(it, "owner", None) if not isinstance(it, dict) else it.get("owner")
            task = getattr(it, "task", None) if not isinstance(it, dict) else it.get("task")
            due = getattr(it, "due_date", None) if not isinstance(it, dict) else it.get("due_date")
            conf = getattr(it, "confidence", None) if not isinstance(it, dict) else it.get("confidence")

            task_s = _clean_text(task, 260)
            if not task_s:
                continue  # elimina placeholder

            owner_s = _best_match_speaker(owner) if owner is not None else None
            due_s = _safe_due(due)

            try:
                cf = float(conf)
            except Exception:
                cf = 0.65
            cf = max(0.0, min(1.0, cf))

            cleaned.append({"owner": owner_s, "task": task_s, "due_date": due_s, "confidence": cf})

        def key_fn(x: Dict[str, Any]) -> str:
            o = (x.get("owner") or "").strip().lower()
            t = _norm_simple(x.get("task") or "")
            return f"{o}::{t}" if o else t

        best: Dict[str, Dict[str, Any]] = {}
        for x in cleaned:
            k = key_fn(x)
            cur = best.get(k)
            if cur is None:
                best[k] = x
                continue
            score_cur = (1 if cur.get("owner") else 0) + (1 if cur.get("due_date") else 0) + float(cur.get("confidence") or 0.0)
            score_new = (1 if x.get("owner") else 0) + (1 if x.get("due_date") else 0) + float(x.get("confidence") or 0.0)
            if score_new > score_cur:
                best[k] = x

        out = list(best.values())
        out.sort(key=lambda z: (0 if z.get("due_date") else 1, -(float(z.get("confidence") or 0.0))))
        ins.action_items = out[:40]

    # --- limpieza decisions ---
    def _clean_decisions() -> None:
        raw_items = getattr(ins, "decisions", None) or []
        cleaned = []
        for it in raw_items:
            text = getattr(it, "text", None) if not isinstance(it, dict) else it.get("text")
            decided_by = getattr(it, "decided_by", None) if not isinstance(it, dict) else it.get("decided_by")
            eff = getattr(it, "effective_date", None) if not isinstance(it, dict) else it.get("effective_date")
            notes = getattr(it, "notes", None) if not isinstance(it, dict) else it.get("notes")

            txt = _clean_text(text, 260)
            if not txt:
                continue

            decided_by_s = _best_match_speaker(decided_by) if decided_by is not None else None
            eff_s = _safe_due(eff)

            if isinstance(notes, list):
                notes2 = [clip_text(_clean_text(n, 160), 160) for n in notes if str(n).strip()]
                notes2 = notes2[:3]
            else:
                notes2 = []

            cleaned.append({"text": txt, "decided_by": decided_by_s, "effective_date": eff_s, "notes": notes2})

        best: Dict[str, Dict[str, Any]] = {}
        for x in cleaned:
            k = _norm_simple(x.get("text") or "")
            if not k:
                continue
            cur = best.get(k)
            if cur is None:
                best[k] = x
                continue
            score_cur = (1 if cur.get("decided_by") else 0) + (1 if cur.get("effective_date") else 0) + len(cur.get("notes") or [])
            score_new = (1 if x.get("decided_by") else 0) + (1 if x.get("effective_date") else 0) + len(x.get("notes") or [])
            if score_new > score_cur:
                best[k] = x

        out = list(best.values())
        out.sort(key=lambda z: (0 if z.get("effective_date") else 1, z.get("text") or ""))
        ins.decisions = out[:30]

    _clean_actions()
    _clean_decisions()

    # main_responsible determinista si falta
    if getattr(ins, "main_responsible", None) in (None, "", "Desconocido"):
        cnt: Dict[str, int] = {}
        for ai in getattr(ins, "action_items", None) or []:
            owner = ai.get("owner") if isinstance(ai, dict) else getattr(ai, "owner", None)
            if owner:
                cnt[owner] = cnt.get(owner, 0) + 1
        if cnt:
            ins.main_responsible = max(cnt.items(), key=lambda kv: kv[1])[0]
        else:
            if valid_speakers:
                ins.main_responsible = max(valid_speakers, key=lambda s: float(participation_percent.get(s, 0.0) or 0.0))

    # labels: castellano + dedupe
    def _dedupe_labels(xs: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in xs or []:
            n = _norm_label_es(str(x))
            if n and n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
        return out

    # 2) clamp base (por seguridad)
    ins.collaboration.score_0_100 = _clamp(getattr(ins.collaboration, "score_0_100", 0.0), 0.0, 100.0)
    ins.decisiveness.score_0_100 = _clamp(getattr(ins.decisiveness, "score_0_100", 0.0), 0.0, 100.0)
    ins.conflict_level_0_100 = _clamp(getattr(ins, "conflict_level_0_100", 0.0), 0.0, 100.0)
    ins.atmosphere.valence = _clamp(getattr(ins.atmosphere, "valence", 0.0), -1.0, 1.0)
    ins.atmosphere.labels = _dedupe_labels(getattr(ins.atmosphere, "labels", []) or [])

    # 3) Señales deterministas (FUENTE DE VERDAD para colaboración/decisión/conflicto)
    det = compute_meeting_signals_deterministic(
        segments=segments or [],
        talk_time_seconds=talk_time_seconds or {},
        turns=turns or {},
        participation_percent=participation_percent or {},
        decisions=getattr(ins, "decisions", None),
        action_items=getattr(ins, "action_items", None),
    )

    # Sobrescribe SIEMPRE estas 3 (para que no dependan del LLM)
    ins.collaboration.score_0_100 = _clamp(det["collaboration_score"], 0.0, 100.0)
    ins.collaboration.notes = det.get("collaboration_notes", [])[:3]

    ins.decisiveness.score_0_100 = _clamp(det["decisiveness_score"], 0.0, 100.0)
    ins.decisiveness.notes = det.get("decisiveness_notes", [])[:3]

    ins.conflict_level_0_100 = _clamp(det["conflict_score"], 0.0, 100.0)

    # Atmosphere: usa LLM SOLO si no viene default y trae algo útil; si no, fallback determinista
    if _is_llm_default_metrics(ins) or not ins.atmosphere.labels:
        ins.atmosphere.valence = _clamp(det.get("valence", 0.0), -1.0, 1.0)
        ins.atmosphere.labels = _dedupe_labels(det.get("labels", []) or [])
        ins.atmosphere.notes = det.get("atmosphere_notes", [])[:3]
    else:
        # si el LLM puso labels pero sin notes, añade un apunte determinista corto
        notes = getattr(ins.atmosphere, "notes", None) or []
        if not notes:
            ins.atmosphere.notes = (det.get("atmosphere_notes", [])[:2])

    # 4) topics: refuerzo determinista si vienen vacíos o genéricos
    _topic_stop = {
        "el", "la", "los", "las", "de", "del", "en", "con", "sin", "para", "por", "que", "como",
        "una", "uno", "unos", "unas", "y", "o", "pero", "muy", "mas", "más", "hay", "sobre",
        "tema", "temas", "reunion", "reunión", "trabajo", "seguimiento", "pasos", "avance", "avances",
        "riesgo", "riesgos", "bloqueo", "bloqueos", "coordinacion", "coordinación", "equipo", "proyecto",
    }

    _generic_topics = {
        "coordinación del trabajo", "seguimiento de avances", "bloqueos y riesgos", "próximos pasos",
        "coordinacion del trabajo", "proximos pasos",
    }

    def _derive_topics_from_segments(max_topics: int = 6) -> List[Topic]:
        counts = Counter()
        for s in segments or []:
            if not isinstance(s, dict):
                continue
            txt = (s.get("text") or "").strip().lower()
            if not txt:
                continue
            words = [w for w in re.findall(r"[a-záéíóúüñ]{4,}", txt) if w not in _topic_stop]
            for a, b in zip(words, words[1:]):
                if a == b:
                    continue
                counts[f"{a} {b}"] += 1

        if not counts:
            return []

        raw = [p for p, c in counts.most_common(16) if c >= 2][:max_topics]
        if not raw:
            raw = [p for p, _ in counts.most_common(max_topics)]

        if not raw:
            return []

        total = float(sum(range(len(raw), 0, -1)))
        out: List[Topic] = []
        for i, name in enumerate(raw):
            w = float((len(raw) - i) / total) if total > 0 else 0.0
            out.append(Topic(name=clip_text(name.strip().title(), 48), weight=w))
        return out

    current_topics = [getattr(t, "name", "") for t in (getattr(ins, "topics", None) or [])]
    normalized_topics = [_norm_simple(t) for t in current_topics if (t or "").strip()]
    only_generic = bool(normalized_topics) and all(t in _generic_topics for t in normalized_topics)
    if not normalized_topics or only_generic:
        derived_topics = _derive_topics_from_segments(max_topics=6)
        if derived_topics:
            ins.topics = derived_topics

    # 5) summary: expande/reemplaza si corto/genérico/placeholder
    def _is_placeholder_summary(s: str) -> bool:
        low = (s or "").strip().lower()
        return low.startswith("sin resumen estructurado") or low.startswith("sin resumen")

    def _summary_needs_help(s: str) -> bool:
        s2 = (s or "").strip().lower()
        if _is_placeholder_summary(s2):
            return True
        if len(s2) < 420:
            return True
        generic = ("reunión de seguimiento" in s2 and "próximos pasos" in s2 and len(s2) < 720)
        return generic

    def _evidence_summary(base: str) -> str:
        base = (base or "").strip()
        if _is_placeholder_summary(base):
            base = ""

        low = base.lower()
        speakers = list((participation_percent or {}).keys())
        n_part = len(speakers)

        total_talk = sum(max(0.0, float(v)) for v in (talk_time_seconds or {}).values())
        total_talk_min = total_talk / 60.0 if total_talk > 0 else 0.0

        top = sorted((participation_percent or {}).items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_txt = ", ".join([f"{k} {v:.1f}%" for k, v in top]) if top else "sin datos"

        n_dec = len(getattr(ins, "decisions", None) or [])
        n_act = len(getattr(ins, "action_items", None) or [])
        topics = [getattr(t, "name", "") for t in (getattr(ins, "topics", None) or []) if getattr(t, "name", "").strip()]
        topics_txt = ", ".join(topics[:6]) if topics else ""
        actions = getattr(ins, "action_items", None) or []

        assigned = 0
        dated = 0
        top_actions: List[str] = []
        for ai in actions[:8]:
            owner = ai.get("owner") if isinstance(ai, dict) else getattr(ai, "owner", None)
            task = ai.get("task") if isinstance(ai, dict) else getattr(ai, "task", None)
            due = ai.get("due_date") if isinstance(ai, dict) else getattr(ai, "due_date", None)
            if owner:
                assigned += 1
            if due:
                dated += 1
            if task:
                if owner and due:
                    top_actions.append(f"{owner}: {task} (plazo {due})")
                elif owner:
                    top_actions.append(f"{owner}: {task}")
                else:
                    top_actions.append(str(task))

        action_preview = "; ".join(top_actions[:3]) if top_actions else "sin acciones suficientemente concretas"

        coll = float(getattr(ins.collaboration, "score_0_100", 0.0) or 0.0)
        decs = float(getattr(ins.decisiveness, "score_0_100", 0.0) or 0.0)
        conf = float(getattr(ins, "conflict_level_0_100", 0.0) or 0.0)
        labels = ", ".join(getattr(ins.atmosphere, "labels", [])[:3]) if getattr(ins.atmosphere, "labels", None) else "no concluyente"

        line1 = (
            f"Participaron {n_part} personas en una sesión de {total_talk_min:.1f} minutos de intervención efectiva. "
            f"La participación estuvo liderada por {top_txt}."
        )

        line2 = (
            f"Se validaron {n_dec} decisiones explícitas y {n_act} acciones con responsable/plazo cuando fue posible."
            if (n_dec or n_act)
            else "No se detectaron decisiones ni acciones explícitas con evidencia suficiente en el transcript disponible."
        )

        line3 = f"Temas de mayor recurrencia: {topics_txt}." if topics_txt else "No se pudieron consolidar temas con recurrencia suficiente."

        line4 = (
            f"En ejecución, {assigned}/{n_act} acciones quedaron asignadas y {dated}/{n_act} incorporaron fecha objetivo."
            if n_act > 0
            else "No hay backlog de acciones verificable para seguimiento operativo."
        )
        line5 = f"Muestra de acciones priorizadas: {action_preview}."
        line6 = (
            f"Señales de reunión: colaboración {coll:.0f}/100, capacidad de decisión {decs:.0f}/100, conflicto {conf:.0f}/100, clima {labels}."
        )
        line7 = (
            "Riesgo operativo principal: tareas sin responsable o sin fecha que pueden degradar la trazabilidad del handoff entre bloques."
            if n_act > 0 and (assigned < n_act or dated < n_act)
            else "Riesgo operativo principal: mantener la actualización del backlog para evitar pérdida de contexto entre bloques."
        )

        if base and len(base) >= 240 and "reunión de seguimiento" not in low:
            return (base + " " + line2 + " " + line3 + " " + line4 + " " + line5 + " " + line6 + " " + line7).strip()
        return (line1 + " " + line2 + " " + line3 + " " + line4 + " " + line5 + " " + line6 + " " + line7).strip()

    if isinstance(ins.summary, str) and _summary_needs_help(ins.summary):
        ins.summary = _evidence_summary(ins.summary)

    # 6) saneo final labels
    ins.atmosphere.labels = _dedupe_labels(ins.atmosphere.labels)

    # Opcional: marca origen determinista para debug
    qf = list(getattr(ins, "quality_flags", None) or [])
    if "signals_deterministic_v2" not in qf:
        qf.append("signals_deterministic_v2")
    ins.quality_flags = qf[:10]

    return ins


def compute_stats_from_segments(
    segments: List[Dict[str, Any]],
    *,
    round_percent_1dp: bool = True,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
    """
    Devuelve (talk_time_seconds, turns, participation_percent) con:
      - speakers como keys EXACTAS (normalizadas solo en espacios)
      - percent estable y sumando 100.0 (si total>0)
    """
    talk: Dict[str, float] = {}
    trn: Dict[str, int] = {}

    # 1) acumula talk + turns
    for s in (segments or []):
        if not isinstance(s, dict):
            continue
        spk = _norm_speaker(s.get("speaker"))
        st = _to_float_time(s.get("start")) or 0.0
        en = _to_float_time(s.get("end"))
        if en is None or en < st:
            # si end falta, estimamos con texto (pero muy conservador)
            en = st + _estimate_duration_from_text((s.get("text") or "").strip())

        dur = max(0.0, float(en) - float(st))
        talk[spk] = talk.get(spk, 0.0) + dur
        trn[spk] = trn.get(spk, 0) + 1

    speakers = list(talk.keys()) or list(trn.keys())
    if not speakers:
        return {}, {}, {}

    # 2) participation percent
    total = sum(max(0.0, v) for v in talk.values())
    if total <= 1e-6:
        # fallback: por turnos si no hay duraciones
        total_turns = sum(trn.values()) or 1
        pct = {k: (100.0 * trn.get(k, 0) / total_turns) for k in speakers}
    else:
        pct = {k: (100.0 * talk.get(k, 0.0) / total) for k in speakers}

    if round_percent_1dp:
        pct = {k: round(float(v), 1) for k, v in pct.items()}
        # fuerza suma 100.0 corrigiendo el mayor
        ssum = round(sum(pct.values()), 1)
        diff = round(100.0 - ssum, 1)
        if abs(diff) >= 0.1:
            kmax = max(pct.keys(), key=lambda k: pct[k])
            pct[kmax] = round(pct[kmax] + diff, 1)

    # 3) redondeos talk
    talk = {k: float(v) for k, v in talk.items()}
    trn = {k: int(v) for k, v in trn.items()}
    return talk, trn, pct


def _adapt_vexa_transcript_if_needed(transcript: Any) -> Any:
    """
    Si transcript ya trae 'segments' (formato actual) -> lo devuelve intacto.
    Si trae 'conversation' (formato Vexa) -> lo convierte a {"segments":[...]} compatible con extract_segments().
    Soporta items con:
      - start_time/end_time
      - start/end
      - duration (si falta end)
    """
    if not isinstance(transcript, dict):
        return transcript

    # Formato actual ya soportado
    if isinstance(transcript.get("segments"), list):
        return transcript

    conv = transcript.get("conversation")
    if not isinstance(conv, list):
        return transcript  # no es vexa

    segments: List[Dict[str, Any]] = []

    for item in conv:
        if not isinstance(item, dict):
            continue

        speaker = item.get("speaker") or "Speaker"
        text = item.get("text") or ""
        if not text:
            continue

        start = item.get("start")
        if start is None:
            start = item.get("start_time") or item.get("startTime")

        end = item.get("end")
        if end is None:
            end = item.get("end_time") or item.get("endTime")

        # Si no hay end pero hay duration, lo calculamos
        if end is None and start is not None:
            dur = item.get("duration")
            if dur is not None:
                try:
                    end = float(start) + float(dur)
                except Exception:
                    end = None

        if start is None or end is None:
            continue

        try:
            start_f = float(start)
            end_f = float(end)
        except Exception:
            continue

        # Normaliza rangos invertidos (por si llegan datos raros)
        if end_f < start_f:
            start_f, end_f = end_f, start_f

        segments.append(
            {"speaker": speaker, "start": start_f, "end": end_f, "text": text}
        )

    segments.sort(key=lambda s: (s["start"], s["end"]))

    # Devolvemos transcript compatible con extract_segments()
    return {"segments": segments}

async def _augment_det_facts_with_llm_candidates(
    det_facts: Dict[str, Any],
    *,
    segments_norm: List[Dict[str, Any]],
    tz_name: str,
) -> Dict[str, Any]:
    """
    Llama a extractor LLM de candidatos y los convierte al formato determinista actual:
      - action: {owner,text,due_iso,confidence_0_1,evidence{speaker,start,end,snippet}}
      - decision: {text,effective_iso,confidence_0_1,evidence{...}}

    Mejoras:
      - Canonicaliza owner a nombres exactos de participants cuando sea posible.
      - Preserva confidence_0_1 del extractor.
      - Añade snippet corto como evidencia (útil para el LLM final sin inventar).
    """
    out = dict(det_facts or {})
    tz = ZoneInfo(tz_name)

    meeting_dt = _parse_iso(out.get("meeting_start_iso")) if out.get("meeting_start_iso") else None
    if meeting_dt:
        meeting_dt = meeting_dt.astimezone(tz)

    participants = out.get("participants") or []
    participants = [p for p in participants if isinstance(p, str) and p.strip()]
    participants_set = set(participants)

    def _norm_simple(s: str) -> str:
        s = (s or "").strip().lower()
        s = _ws_re.sub(" ", s)
        # normaliza tildes frecuente (sin unicodedata)
        s = (s.replace("á", "a").replace("é", "e").replace("í", "i")
               .replace("ó", "o").replace("ú", "u").replace("ü", "u").replace("ñ", "n"))
        return s

    def _best_match_owner(owner: Optional[str], fallback: str) -> Optional[str]:
        """
        Devuelve un owner que sea EXACTAMENTE uno de participants, si encuentra match razonable.
        Si no, fallback (speaker). Si fallback no está, devuelve None.
        """
        if isinstance(owner, str):
            o = owner.strip()
            if o in participants_set:
                return o
            if o:
                okey = _norm_simple(o)
                # match por token overlap (nombre corto vs nombre largo)
                o_tokens = {x for x in re.findall(r"[a-z0-9]+", okey) if len(x) >= 3}
                best = None
                best_score = 0.0
                for cand in participants:
                    ckey = _norm_simple(cand)
                    c_tokens = {x for x in re.findall(r"[a-z0-9]+", ckey) if len(x) >= 3}
                    if not c_tokens:
                        continue
                    inter = len(o_tokens & c_tokens)
                    denom = max(1, min(len(o_tokens) or 1, len(c_tokens)))
                    score = inter / denom
                    if score > best_score:
                        best_score = score
                        best = cand
                if best and best_score >= 0.70:
                    return best

        # fallback speaker si es participante
        fb = (fallback or "").strip()
        if fb in participants_set:
            return fb
        return None

    def _infer_owner_from_text(text: str, speaker: str) -> Optional[str]:
        """
        Inferencia conservadora de asignación explícita dentro de la frase:
          - "Juan, prepara...", "Ana te encargas...", "Pedro puedes..."
        Si no hay evidencia clara, usa speaker como fallback.
        """
        txt = (text or "").strip()
        if not txt:
            return _best_match_owner(None, speaker)

        low = _norm_simple(txt)
        imperative_cues = (
            "puedes", "puede", "encarg", "haz", "haces", "prepara", "revisa", "manda", "envia",
            "sube", "documenta", "organiza", "valida", "cierra", "define", "actualiza",
        )
        has_cue = any(c in low for c in imperative_cues)

        # Busca nombres de participantes en el texto; exige frontera léxica para evitar falsos positivos
        hits: List[Tuple[int, str]] = []
        for p in participants:
            p_norm = _norm_simple(p)
            if not p_norm:
                continue
            pat = r"(?<![a-z0-9])" + re.escape(p_norm) + r"(?![a-z0-9])"
            m = re.search(pat, low)
            if m:
                hits.append((m.start(), p))

        if hits and has_cue:
            hits.sort(key=lambda x: x[0])
            candidate = hits[0][1]

            # Si aparece al inicio con coma/pausa suele ser vocativo de asignación directa.
            first_pos = hits[0][0]
            near_start = first_pos <= 8
            if near_start or any(k in low[:first_pos + 40] for k in ("te encarg", "puedes", "haz", "revisa", "prepara")):
                return candidate

        # fallback seguro
        return _best_match_owner(None, speaker)

    def _dedupe_candidates(
        rows: List[Dict[str, Any]],
        *,
        key_getter,
        cap: int,
    ) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            key_raw = key_getter(row)
            key = _normalize_text_key(str(key_raw or ""))
            if not key:
                continue
            prev = best.get(key)
            if prev is None or float(row.get("confidence_0_1") or 0.0) > float(prev.get("confidence_0_1") or 0.0):
                best[key] = row
        out = list(best.values())
        out.sort(key=lambda x: float(x.get("confidence_0_1") or 0.0), reverse=True)
        return out[:cap]

    try:
        cand = await extract_action_decision_candidates_llm(segments_norm, temperature=0.0, max_tokens=700)
    except Exception:
        return out

    actions: List[Dict[str, Any]] = []
    for a in cand.get("actions", []) or []:
        idx = a.get("segment_idx")
        if not isinstance(idx, int) or idx < 0 or idx >= len(segments_norm):
            continue
        seg = segments_norm[idx]
        spk = seg.get("speaker") or "Desconocido"
        verb = (a.get("verbatim") or seg.get("text") or "").strip()
        if not verb:
            continue

        due_txt = a.get("due_text")
        due_txt = due_txt if isinstance(due_txt, str) else ""
        # parse determinista de due (si no hay meeting_dt, no inventes)
        due = parse_due_datetime_es(due_txt or verb, meeting_dt, tz) if meeting_dt else None

        conf = a.get("confidence_0_1")
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.65
        conf_f = max(0.0, min(1.0, conf_f))

        owner_raw = a.get("owner")
        owner = _best_match_owner(owner_raw if isinstance(owner_raw, str) else None, spk)
        if owner is None:
            owner = _infer_owner_from_text(verb, spk)

        snippet = clip_text(seg.get("text") or "", 140)

        actions.append({
            "owner": owner,
            "text": verb,
            "due_iso": due.isoformat() if due else None,
            "confidence_0_1": conf_f,
            "evidence": {
                "speaker": spk,
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "snippet": snippet,
            }
        })

    decisions: List[Dict[str, Any]] = []
    for d in cand.get("decisions", []) or []:
        idx = d.get("segment_idx")
        if not isinstance(idx, int) or idx < 0 or idx >= len(segments_norm):
            continue
        seg = segments_norm[idx]
        spk = seg.get("speaker") or "Desconocido"
        verb = (d.get("verbatim") or seg.get("text") or "").strip()
        if not verb:
            continue

        eff_txt = d.get("effective_text")
        eff_txt = eff_txt if isinstance(eff_txt, str) else ""
        eff = parse_due_datetime_es(eff_txt, meeting_dt, tz) if (meeting_dt and eff_txt) else None

        conf = d.get("confidence_0_1")
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.65
        conf_f = max(0.0, min(1.0, conf_f))

        snippet = clip_text(seg.get("text") or "", 140)

        decisions.append({
            "text": verb,
            "effective_iso": eff.isoformat() if eff else None,
            "confidence_0_1": conf_f,
            "evidence": {
                "speaker": spk,
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "snippet": snippet,
            }
        })

    out["deterministic_action_candidates"] = _dedupe_candidates(
        actions,
        key_getter=lambda r: f"{(r.get('owner') or '').strip().lower()}|{r.get('text') or ''}",
        cap=40,
    )
    out["deterministic_decision_candidates"] = _dedupe_candidates(
        decisions,
        key_getter=lambda r: r.get("text") or "",
        cap=30,
    )
    return out

async def _analyze_from_transcript(req: AnalyzeRequest, transcript: Any) -> AnalyzeResponse:
    # 0) coerción de transcript (por si llega conversation)
    transcript = _coerce_transcript_to_segments(transcript)

    raw_segments: List[Dict[str, Any]] = []
    if isinstance(transcript, dict) and isinstance(transcript.get("segments"), list):
        raw_segments = normalize_segments_preserve_abs(transcript.get("segments"))
        transcript = {**transcript, "segments": raw_segments}

    # 1) normalize utterances (para TOON + stats)
    utterances = extract_segments(transcript)

    if req.options.merge_adjacent_same_speaker:
        utterances = merge_adjacent(utterances, merge_gap_ms=req.options.merge_gap_ms)

    utterances = [
        NormalizedUtterance(
            speaker=u.speaker,
            start=u.start,
            end=u.end,
            text=clip_text(u.text, req.options.max_chars_per_utterance),
        )
        for u in utterances
        if u.text
    ]

    if not utterances:
        raise HTTPException(status_code=422, detail="No hay segmentos útiles en body['segments'].")

    # 2) stats globales deterministas
    talk_time, turns, participation = compute_talk_stats_deterministic(utterances)

    # segmentos “planos” para heurística/postproceso
    segments_for_pp: List[Dict[str, Any]] = [
        {"speaker": u.speaker, "start": float(u.start), "end": float(u.end), "text": u.text}
        for u in utterances
    ]

    # 3) deterministic facts + candidatos (LLM anclado) => det_facts_json
    segments_for_candidates = raw_segments if raw_segments else segments_for_pp

    det_facts = extract_facts_deterministic(segments_for_candidates, tz_name="Europe/Madrid")
    det_facts = await _augment_det_facts_with_llm_candidates(
        det_facts,
        segments_norm=segments_for_candidates,
        tz_name="Europe/Madrid",
    )
    det_facts_json_full = json.dumps(det_facts, ensure_ascii=False)

    def _det_facts_json_for_window(start_sec: float, end_sec: float) -> str:
        base = dict(det_facts or {})
        actions = base.get("deterministic_action_candidates") or []
        decisions = base.get("deterministic_decision_candidates") or []

        def _in_window(item: Dict[str, Any]) -> bool:
            ev = (item or {}).get("evidence") or {}
            try:
                st = float(ev.get("start") or 0.0)
            except Exception:
                st = 0.0
            return (st >= float(start_sec)) and (st <= float(end_sec))

        base["deterministic_action_candidates"] = [a for a in actions if isinstance(a, dict) and _in_window(a)]
        base["deterministic_decision_candidates"] = [d for d in decisions if isinstance(d, dict) and _in_window(d)]
        return json.dumps(base, ensure_ascii=False)

    def _maybe_make_charts(final_insights: MeetingInsights) -> tuple[Optional[str], Optional[str], Optional[str]]:
        charts_id = None
        charts_png_url = None
        charts_png_base64 = None

        charts_opt = getattr(req.options, "charts", None)
        if not charts_opt or not charts_opt.enabled:
            return charts_id, charts_png_url, charts_png_base64

        mode = getattr(charts_opt, "mode", "url")
        if mode == "none":
            return charts_id, charts_png_url, charts_png_base64

        from app.llm_insights.plots import render_insights_png

        png_bytes = render_insights_png(
            final_insights.model_dump(),
            title=(charts_opt.title or "LLM Meeting Insights"),
        )

        if mode in ("url", "both"):
            charts_id = save_chart_png(png_bytes)
            charts_png_url = make_chart_url(charts_id)

        if mode in ("base64", "both"):
            charts_png_base64 = base64.b64encode(png_bytes).decode("ascii")

        return charts_id, charts_png_url, charts_png_base64

    llm_response_format = {"type": "json_object"}
    llm_reasoning_format = "none"

    # 4) Single pass
    if not req.options.chunking.enabled:
        toon_full = build_toon_v1(
            utterances=utterances,
            fmt=req.options.toon_format,
            include_timestamps=req.options.include_timestamps,
        )

        messages = build_llm_messages_single(
            toon_full,
            talk_time=talk_time,
            turns=turns,
            participation=participation,
            det_facts_json=det_facts_json_full,
        )

        llm_raw = await call_llama_cpp(
            messages,
            temperature=req.options.temperature,
            max_tokens=req.options.max_tokens,
            response_format=llm_response_format,
            reasoning_format=llm_reasoning_format,
        )

        insights, mode = parse_meeting_insights_or_fallback(llm_raw, talk_time, turns, participation)
        final_insights = coerce_meeting_insights(insights.model_dump(), talk_time, turns, participation)

        final_insights = postprocess_meeting_insights(
            final_insights,
            segments=segments_for_pp,
            talk_time_seconds=talk_time,
            turns=turns,
            participation_percent=participation,
        )

        handoff = init_handoff_from_nothing()
        handoff.chunk_index = 0
        handoff.chunks_total = 1
        handoff.processed_until_sec = utterances[-1].end
        handoff.running_summary = final_insights.summary
        handoff.topics = final_insights.topics
        handoff.decisions = final_insights.decisions
        handoff.action_items = final_insights.action_items
        handoff.collaboration = final_insights.collaboration
        handoff.atmosphere = final_insights.atmosphere
        handoff.decisiveness = final_insights.decisiveness
        handoff.conflict_level_0_100 = final_insights.conflict_level_0_100
        handoff = clamp_handoff_size(handoff)

        chunk = ChunkResult(
            chunk_index=0,
            toon=toon_full,
            llm_raw=llm_raw,
            parse_mode=mode,
            insights=final_insights,
            handoff=handoff,
        )

        charts_id, charts_png_url, charts_png_base64 = _maybe_make_charts(final_insights)

        toon_summary = build_toon_summary_v1(
            final_insights,
            talk_time=talk_time,
            turns=turns,
            participation=participation,
        )

        return AnalyzeResponse(
            toon=toon_summary,
            toon_transcript=toon_full,
            normalized=utterances,
            insights=final_insights,
            llm_raw=llm_raw,
            parse_mode=mode,
            chunks_total=1,
            final_handoff=handoff,
            chunks=[chunk],
            charts_id=charts_id,
            charts_png_url=charts_png_url,
            charts_png_base64=charts_png_base64,
        )

    # 5) Chunking
    ch = req.options.chunking
    reserved_tokens = int(ch.prompt_overhead_tokens + ch.reserve_output_tokens + ch.safety_margin_tokens)
    max_tokens_llm = min(req.options.max_tokens, int(ch.reserve_output_tokens))

    try:
        chunks_u = split_utterances_into_chunks(
            utterances=utterances,
            fmt=req.options.toon_format,
            include_timestamps=req.options.include_timestamps,
            context_window_tokens=int(ch.context_window_tokens),
            reserved_tokens=reserved_tokens,
            max_chunks=int(ch.max_chunks),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    chunks_total = len(chunks_u)

    handoff = req.previous_handoff or init_handoff_from_nothing()
    handoff.chunks_total = chunks_total

    chunk_results: List[ChunkResult] = []

    # ---------- helpers de merge determinista de handoff ----------
    def _first_sentences(text: str, n: int = 2, max_chars: int = 420) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        # split muy simple por puntos (suficiente y seguro)
        parts = [p.strip() for p in re.split(r"(?<=[\.\!\?])\s+", s) if p.strip()]
        out = " ".join(parts[:n]).strip()
        return clip_text(out, max_chars)

    def _merge_running_summary(prev: str, chunk_sum: str, *, max_chars: int = 1800) -> str:
        prev = (prev or "").strip()
        delta = _first_sentences(chunk_sum or "", n=3, max_chars=760)
        if not delta:
            return clip_text(prev, max_chars)
        if not prev:
            return clip_text(delta, max_chars)
        low_prev = prev.lower()
        low_delta = delta.lower()
        if low_delta in low_prev:
            return clip_text(prev, max_chars)
        merged = (prev.rstrip() + " " + delta).strip()
        return clip_text(merged, max_chars)

    def _dedupe_by_text(items: List[Any], text_getter, cap: int) -> List[Any]:
        best = {}
        for it in items or []:
            txt = text_getter(it)
            key = _ws_re.sub(" ", (txt or "").strip().lower())
            if not key:
                continue
            if key not in best:
                best[key] = it
        out = list(best.values())
        return out[:cap]

    def _merge_action_items(prev_items: List[Any], cur_items: List[Any], cap: int = 30) -> List[Any]:
        def _as_dict(ai: Any) -> Dict[str, Any]:
            if isinstance(ai, dict):
                return dict(ai)
            return {
                "owner": getattr(ai, "owner", None),
                "task": getattr(ai, "task", None),
                "due_date": getattr(ai, "due_date", None),
                "confidence": getattr(ai, "confidence", None),
            }

        def _task_key(ai: Dict[str, Any]) -> str:
            task = str(ai.get("task") or "").strip().lower()
            task = _ws_re.sub(" ", task)
            return task

        def _score(ai: Dict[str, Any]) -> float:
            c = ai.get("confidence")
            try:
                cf = float(c)
            except Exception:
                cf = 0.0
            return (1.2 if ai.get("owner") else 0.0) + (1.0 if ai.get("due_date") else 0.0) + max(0.0, min(1.0, cf))

        merged: Dict[str, Dict[str, Any]] = {}
        for raw in list(prev_items or []) + list(cur_items or []):
            it = _as_dict(raw)
            key = _task_key(it)
            if not key:
                continue
            cur = merged.get(key)
            if cur is None:
                merged[key] = it
                continue

            # fusiona campos faltantes sin perder el más rico
            best = it if _score(it) > _score(cur) else cur
            other = cur if best is it else it
            if not best.get("owner") and other.get("owner"):
                best["owner"] = other.get("owner")
            if not best.get("due_date") and other.get("due_date"):
                best["due_date"] = other.get("due_date")
            if (best.get("confidence") is None) and (other.get("confidence") is not None):
                best["confidence"] = other.get("confidence")
            merged[key] = best

        out = list(merged.values())
        out.sort(key=lambda x: ((0 if x.get("owner") else 1), (0 if x.get("due_date") else 1), -(float(x.get("confidence") or 0.0))))
        return out[:cap]

    for i, chunk_utterances in enumerate(chunks_u):
        toon_chunk = build_toon_v1(
            utterances=chunk_utterances,
            fmt=req.options.toon_format,
            include_timestamps=req.options.include_timestamps,
        )

        processed_until_sec = chunk_utterances[-1].end if chunk_utterances else None

        if chunk_utterances:
            win_start = float(chunk_utterances[0].start)
            win_end = float(chunk_utterances[-1].end)
            det_facts_json_chunk = _det_facts_json_for_window(win_start, win_end)
        else:
            det_facts_json_chunk = det_facts_json_full

        messages = build_llm_messages_chunk(
            toon_chunk=toon_chunk,
            talk_time=talk_time,
            turns=turns,
            participation=participation,
            prev_handoff=handoff,
            chunk_index=i,
            chunks_total=chunks_total,
            det_facts_json=det_facts_json_chunk,
        )

        llm_raw_i = await call_llama_cpp(
            messages,
            temperature=req.options.temperature,
            max_tokens=max_tokens_llm,
            response_format=llm_response_format,
            reasoning_format=llm_reasoning_format,
        )

        insights_i, handoff_i, mode_i = parse_chunk_output_or_fallback(
            llm_text=llm_raw_i,
            talk_time=talk_time,
            turns=turns,
            participation=participation,
            chunk_index=i,
            chunks_total=chunks_total,
            processed_until_sec=processed_until_sec,
            prev_handoff=handoff,
        )

        insights_i = coerce_meeting_insights(insights_i.model_dump(), talk_time, turns, participation)

        # ✅ postproceso por chunk (incluye limpiar tareas placeholder)
        seg_chunk_pp = [
            {"speaker": u.speaker, "start": float(u.start), "end": float(u.end), "text": u.text}
            for u in chunk_utterances
        ]
        insights_i = postprocess_meeting_insights(
            insights_i,
            segments=seg_chunk_pp,
            talk_time_seconds=talk_time,
            turns=turns,
            participation_percent=participation,
        )

        # ✅ handoff robusto: no dependas de que el LLM lo rellene bien
        # 1) actualiza índices/tiempos
        handoff_i.chunk_index = i
        handoff_i.chunks_total = chunks_total
        handoff_i.processed_until_sec = processed_until_sec

        # 2) running_summary acumulativo (delta por chunk)
        prev_rs = getattr(handoff, "running_summary", "") or ""
        handoff_i.running_summary = _merge_running_summary(prev_rs, insights_i.summary)

        # 3) merge de listas clave para continuidad (dedupe estable)
        # topics: usa lo que venga en insights_i (ya limpio), y conserva anteriores si existían
        prev_topics = getattr(handoff, "topics", None) or []
        cur_topics = getattr(insights_i, "topics", None) or []
        # dedupe por name
        merged_topics = _dedupe_by_text(
            list(prev_topics) + list(cur_topics),
            text_getter=lambda t: (t.get("name") if isinstance(t, dict) else getattr(t, "name", "")),
            cap=14,
        )
        handoff_i.topics = merged_topics

        prev_dec = getattr(handoff, "decisions", None) or []
        cur_dec = getattr(insights_i, "decisions", None) or []
        merged_dec = _dedupe_by_text(
            list(prev_dec) + list(cur_dec),
            text_getter=lambda d: (d.get("text") if isinstance(d, dict) else getattr(d, "text", "")),
            cap=20,
        )
        handoff_i.decisions = merged_dec

        prev_ai = getattr(handoff, "action_items", None) or []
        cur_ai = getattr(insights_i, "action_items", None) or []
        merged_ai = _merge_action_items(prev_ai, cur_ai, cap=30)
        handoff_i.action_items = merged_ai

        # 4) métricas actuales al handoff (para tendencia)
        handoff_i.collaboration = insights_i.collaboration
        handoff_i.atmosphere = insights_i.atmosphere
        handoff_i.decisiveness = insights_i.decisiveness
        handoff_i.conflict_level_0_100 = insights_i.conflict_level_0_100

        # 5) clamp final para no reventar el contexto
        handoff_i = clamp_handoff_size(handoff_i)

        # handoff para el siguiente chunk
        handoff = handoff_i

        chunk_results.append(
            ChunkResult(
                chunk_index=i,
                toon=toon_chunk,
                llm_raw=(llm_raw_i if ch.return_llm_raw_per_chunk else None),
                parse_mode=mode_i,
                insights=insights_i,
                handoff=handoff_i,
            )
        )

    toon_full = build_toon_v1(
        utterances=utterances,
        fmt=req.options.toon_format,
        include_timestamps=req.options.include_timestamps,
    )

    cumulative_stats = {
        "talk_time_seconds": talk_time,
        "turns": turns,
        "participation_percent": participation,
    }

    finalize_messages = build_llm_messages_finalize(
        cumulative_stats=cumulative_stats,
        final_handoff=handoff,
        det_facts_json=det_facts_json_full,
    )

    llm_raw_final = await call_llama_cpp(
        finalize_messages,
        temperature=req.options.temperature,
        max_tokens=max_tokens_llm,
        response_format=llm_response_format,
        reasoning_format=llm_reasoning_format,
    )

    final_insights, final_mode = parse_meeting_insights_or_fallback(llm_raw_final, talk_time, turns, participation)
    final_insights = coerce_meeting_insights(final_insights.model_dump(), talk_time, turns, participation)

    final_insights = postprocess_meeting_insights(
        final_insights,
        segments=segments_for_pp,
        talk_time_seconds=talk_time,
        turns=turns,
        participation_percent=participation,
    )

    charts_id, charts_png_url, charts_png_base64 = _maybe_make_charts(final_insights)

    toon_summary = build_toon_summary_v1(
        final_insights,
        talk_time=talk_time,
        turns=turns,
        participation=participation,
    )

    return AnalyzeResponse(
        toon=toon_summary,
        toon_transcript=toon_full,
        normalized=utterances,
        insights=final_insights,
        llm_raw=llm_raw_final,
        parse_mode=final_mode,
        chunks_total=chunks_total,
        final_handoff=handoff,
        chunks=chunk_results,
        charts_id=charts_id,
        charts_png_url=charts_png_url,
        charts_png_base64=charts_png_base64,
    )


def _norm_speaker(name: Optional[str]) -> str:
    s = (name or "").strip()
    s = _ws_re.sub(" ", s)
    return s or "Desconocido"

def _to_float_time(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            v = float(v)
        elif isinstance(v, (int, float)):
            v = float(v)
        else:
            return None
        # Heurística ms -> s
        if v > 100000:  # 100k ms ~ 100s
            v = v / 1000.0
        return float(v)
    except Exception:
        return None

def _estimate_duration_from_text(text: str) -> float:
    # Aproximación conservadora: 0.42s por palabra (≈ 143 wpm)
    words = len(re.findall(r"\w+", text or "", re.UNICODE))
    return max(0.6, min(18.0, words * 0.42))

def _vexa_conversation_to_segments_list(conversation: Any) -> List[Dict[str, Any]]:
    """
    Convierte conversation[] (Vexa) a segments[] con {speaker,start,end,text}.
    - Soporta start/end o start_time/end_time o duration.
    - Si faltan tiempos, estima duración por texto (conservador).
    - Normaliza ms -> s si detecta magnitudes grandes.
    """
    if not isinstance(conversation, list):
        return []

    def _norm_speaker(name: Optional[str]) -> str:
        s = (name or "").strip()
        s = _ws_re.sub(" ", s)
        return s or "Desconocido"

    def _to_float_time(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            if isinstance(v, str):
                v = v.strip()
                if not v:
                    return None
                v = float(v)
            elif isinstance(v, (int, float)):
                v = float(v)
            else:
                return None
            # ms -> s
            if v > 100000:
                v = v / 1000.0
            return float(v)
        except Exception:
            return None

    def _estimate_duration_from_text(text: str) -> float:
        words = len(re.findall(r"\w+", text or "", re.UNICODE))
        return max(0.6, min(18.0, words * 0.42))

    out: List[Dict[str, Any]] = []
    t_cursor = 0.0

    for item in conversation:
        if not isinstance(item, dict):
            continue

        speaker = _norm_speaker(
            item.get("speaker")
            or item.get("speaker_name")
            or item.get("participant")
            or item.get("name")
        )
        text = (item.get("text") or item.get("utterance") or item.get("content") or "").strip()
        if not text:
            continue

        start = _to_float_time(item.get("start") or item.get("start_time") or item.get("startTime") or item.get("t"))
        end = _to_float_time(item.get("end") or item.get("end_time") or item.get("endTime"))

        if start is None:
            start = t_cursor

        # si hay duration, úsala
        if (end is None or end < start) and item.get("duration") is not None:
            d = _to_float_time(item.get("duration"))
            if d is not None and d > 0:
                end = start + d

        if end is None or end < start:
            end = start + _estimate_duration_from_text(text)

        t_cursor = max(t_cursor, end)

        out.append({"speaker": speaker, "start": float(start), "end": float(end), "text": text})

    out.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    return out


def _coerce_transcript_to_segments(transcript: Any) -> Any:
    """
    Acepta:
      - {"segments":[...]} -> lo devuelve tal cual
      - {"conversation":[...]} -> lo convierte a {"segments":[...]}
      - Cualquier otra cosa -> lo devuelve tal cual
    """
    if not isinstance(transcript, dict):
        return transcript

    if isinstance(transcript.get("segments"), list):
        return transcript

    conv = transcript.get("conversation")
    if isinstance(conv, list):
        return {
            "segments": _vexa_conversation_to_segments_list(conv),
            # meta opcional preservada
            "meeting_id": transcript.get("meeting_id"),
            "native_meeting_id": transcript.get("native_meeting_id"),
            "constructed_meeting_url": transcript.get("constructed_meeting_url"),
            "start_time": transcript.get("start_time"),
            "total_duration_seconds": transcript.get("total_duration_seconds"),
        }

    return transcript

#----------aqui los endpoints--------------
router = APIRouter(prefix="/v1/insights", tags=["insights"])

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    transcript = resolve_transcript_from_request(req)
    result = await _analyze_from_transcript(req, transcript)
    await _trigger_hybrid_ingest(req, result)
    return result

@router.post("/analyze_vexa", response_model=AnalyzeResponse)
async def analyze_vexa(request: Request, body: Any = Body(...)):
    ct = (request.headers.get("content-type") or "").lower()
    logger.info("analyze_vexa content-type=%s", ct)
    logger.info("analyze_vexa body_type=%s", type(body).__name__)

    # 1) Normaliza body -> dict
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")

    if isinstance(body, str):
        logger.warning("analyze_vexa body llegó como string (posible doble stringify). len=%d", len(body))
        s = body.strip()
        try:
            body = json.loads(s)
            logger.info("analyze_vexa body string -> json OK (type=%s)", type(body).__name__)
        except Exception as ex:
            logger.exception("analyze_vexa body string NO es JSON válido: %s", ex)
            raise HTTPException(status_code=422, detail="Body no es JSON válido (parece string).")

    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail=f"Body debe ser objeto JSON (dict). Llegó: {type(body).__name__}")

    body = dict(body)

    # 1b) wrapper típico: {payload:{...}}
    if "payload" in body and isinstance(body["payload"], dict):
        logger.warning("analyze_vexa body trae wrapper 'payload' -> usando body['payload']")
        body = dict(body["payload"])

    # 2) raw_transcript_json puede venir como rawTranscriptJson
    raw = body.get("raw_transcript_json", None)
    if raw is None and "rawTranscriptJson" in body:
        raw = body.get("rawTranscriptJson")
        body["raw_transcript_json"] = raw
        logger.warning("analyze_vexa rawTranscriptJson -> raw_transcript_json")

    logger.info("analyze_vexa raw_type=%s", type(raw).__name__ if raw is not None else "None")

    # 2b) si raw viene como string (stringificado), intenta parsear
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
        body["raw_transcript_json"] = raw

    if isinstance(raw, str):
        rs = raw.strip()
        logger.warning("analyze_vexa raw_transcript_json llegó como string. head=%r", rs[:80])
        if rs and rs[0] in "[{":
            try:
                raw2 = json.loads(rs)
                raw = raw2
                body["raw_transcript_json"] = raw2
                logger.info("analyze_vexa raw_transcript_json string -> json OK (type=%s)", type(raw2).__name__)
            except Exception as ex:
                logger.exception("analyze_vexa raw_transcript_json string NO parseable: %s", ex)

    # 3) Normalizaciones Vexa
    raw = body.get("raw_transcript_json")

    if isinstance(raw, list):
        logger.info("analyze_vexa raw_transcript_json list_len=%d", len(raw))
        # Si ya son turnos {speaker,start,end,text}
        if _looks_like_turn_segments_list(raw):
            logger.info("analyze_vexa raw parece segments -> moviendo a body['segments']")
            body["segments"] = raw
            body.pop("raw_transcript_json", None)

    elif isinstance(raw, dict):
        logger.info("analyze_vexa raw_transcript_json dict_keys=%s", list(raw.keys())[:12])
        if isinstance(raw.get("segments"), list):
            body["segments"] = raw.get("segments") or []
            body.pop("raw_transcript_json", None)
        elif isinstance(raw.get("conversation"), list):
            body["segments"] = _vexa_conversation_to_segments_list(raw.get("conversation") or [])
            body.pop("raw_transcript_json", None)

    ingest_passthrough = {
        "user_id": body.get("user_id"),
        "doc_id": body.get("doc_id"),
        "file_name": body.get("file_name"),
        "source_path": body.get("source_path"),
        "native_meeting_id": body.get("native_meeting_id"),
    }

    # 4) Filtra campos extra si tu modelo los rechaza (recomendado)
    try:
        allowed = set(AnalyzeRequest.model_fields.keys())
        extra = [k for k in body.keys() if k not in allowed]
        if extra:
            logger.warning("analyze_vexa campos extra no esperados: %s", extra[:20])
        body = {k: v for k, v in body.items() if k in allowed}
    except Exception:
        pass

    # 5) Validación final (AQUÍ está tu 422)
    try:
        req = AnalyzeRequest.model_validate(body)
    except ValidationError as e:
        errs = e.errors()
        logger.error("analyze_vexa ValidationError: %s", errs)
        raise HTTPException(status_code=422, detail=errs)

    transcript = resolve_transcript_from_request(req)
    result = await _analyze_from_transcript(req, transcript)
    await _trigger_hybrid_ingest(req, result, passthrough_meta=ingest_passthrough)
    return result


def _looks_like_turn_segments_list(xs: Any) -> bool:
    if not isinstance(xs, list) or not xs:
        return False
    k = 0
    for it in xs[:8]:
        if not isinstance(it, dict):
            continue
        spk = it.get("speaker") or it.get("speaker_name") or it.get("name")
        txt = it.get("text") or it.get("utterance") or it.get("content")
        st = it.get("start") or it.get("start_time") or it.get("t")
        if spk and txt and (st is not None):
            k += 1
    return k >= 2


@router.post("/toon")
async def toon_only(req: AnalyzeRequest):
    transcript = resolve_transcript_from_request(req)

    utterances = extract_segments(transcript)
    if req.options.merge_adjacent_same_speaker:
        utterances = merge_adjacent(utterances, merge_gap_ms=req.options.merge_gap_ms)

    utterances = [
        NormalizedUtterance(
            speaker=u.speaker,
            start=u.start,
            end=u.end,
            text=clip_text(u.text, req.options.max_chars_per_utterance),
        )
        for u in utterances
        if u.text
    ]
    if not utterances:
        raise HTTPException(status_code=422, detail="No hay segmentos útiles en body['segments'].")

    toon = build_toon_v1(utterances, fmt=req.options.toon_format, include_timestamps=req.options.include_timestamps)

    ch = req.options.chunking
    reserved_tokens = int(ch.prompt_overhead_tokens + ch.reserve_output_tokens + ch.safety_margin_tokens)

    try:
        chunks_u = split_utterances_into_chunks(
            utterances=utterances,
            fmt=req.options.toon_format,
            include_timestamps=req.options.include_timestamps,
            context_window_tokens=int(ch.context_window_tokens) if ch.enabled else CONTEXT_WINDOW_TOKENS,
            reserved_tokens=reserved_tokens if ch.enabled else PROMPT_RESERVED_TOKENS,
            max_chunks=int(ch.max_chunks) if ch.enabled else 999,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"toon": toon, "utterances": [u.model_dump() for u in utterances], "chunks_total": len(chunks_u)}

@router.post("/render_png")
async def render_png(req: RenderPNGRequest):
    raw = dict(req.insights or {})

    # Intenta preservar stats si vienen; si no, rellena mínimos con defaults sin machacar
    talk_time = raw.get("talk_time_seconds") or {}
    turns = raw.get("turns") or {}
    participation = raw.get("participation_percent") or {}

    insights_obj = coerce_meeting_insights(raw, talk_time=talk_time, turns=turns, participation=participation)
    insights_dict = insights_obj.model_dump()

    from app.llm_insights.plots import render_insights_png
    png_bytes = render_insights_png(insights_dict, title=req.title or "LLM Meeting Insights")

    return Response(content=png_bytes, media_type="image/png")

@router.get("/charts/{chart_id}.png")
async def get_chart_png(chart_id: str):
    if not _safe_id_re.match(chart_id):
        raise HTTPException(status_code=400, detail="Invalid chart_id")

    path = INSIGHTS_CHARTS_DIR / f"{chart_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chart not found")

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{chart_id}.png",
    )


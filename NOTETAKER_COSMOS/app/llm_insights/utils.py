from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Literal
import httpx
import time
import uuid
from pathlib import Path
import unicodedata
import math

from .models import (
    TopicLedgerItem,
    NormalizedUtterance,
    Topic,
    MeetingInsights,
    HandoffState,

)

#Constantes
_CTX_ERR_MARKERS = (
    "Context size has been exceeded",
    "failed to find free space in the KV cache",
    "failed to find a memory slot",
    "kv cache",
)
INSIGHTS_CHARTS_DIR = Path(os.getenv("INSIGHTS_CHARTS_DIR", str(Path.cwd() / ".cache" / "insights_charts")))
INSIGHTS_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")  # opcional, para URLs absolutas
INSIGHTS_CHARTS_TTL_S = int(os.getenv("INSIGHTS_CHARTS_TTL_S", "86400"))  # 24h
CONTEXT_WINDOW_TOKENS = int(os.getenv("INSIGHTS_CONTEXT_WINDOW_TOKENS", "18000"))
PROMPT_RESERVED_TOKENS = int(os.getenv("PROMPT_RESERVED_TOKENS", "3200"))  # system+schema+overhead
HANDOFF_MAX_TOKENS = int(os.getenv("HANDOFF_MAX_TOKENS", "900"))  # mantener estado compacto

#Variables_regex_limpieza de llm
_punct_no_space_after = {"(", "[", "{"}
_punct_no_space_before = re.compile(r"\s+([,.;:!?%\)\]\}])")
_punct_no_space_after_open = re.compile(r"([\(\[\{¿¡])\s+")
_multi_space = re.compile(r"\s+")
_safe_id_re = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$")


def _strip_llm_wrappers(text: str) -> str:
    text = (text or "").strip()

    # quita fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    # si el modelo usa tokens tipo llama/gpt-oss
    if "<|end|>" in text:
        # normalmente el JSON bueno está tras el último <|end|>
        text = text.split("<|end|>")[-1].strip()

    return text

def _extract_balanced_json_objects(text: str) -> List[str]:
    """
    Extrae objetos JSON {...} balanceando llaves, ignorando llaves dentro de strings.
    Devuelve una lista de candidatos en orden de aparición.
    """
    objs: List[str] = []
    s = text or ""

    in_str = False
    esc = False
    depth = 0
    start: Optional[int] = None

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # fuera de string
        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(s[start : i + 1])
                start = None

    return objs

def try_extract_json_object(text: str) -> Optional[str]:
    """
    Devuelve el 'mejor' objeto JSON (dict) parseable dentro del texto.
    Prioriza el último objeto JSON válido.
    """
    cleaned = _strip_llm_wrappers(text)

    # caso directo
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return cleaned
        except Exception:
            pass

    # candidatos por balanceo
    candidates = _extract_balanced_json_objects(cleaned)

    # prueba desde el final (lo más habitual: el JSON "final" va el último)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return cand
        except Exception:
            continue

    return None

def parse_meeting_insights_or_fallback(
    llm_text: str,
    talk_time: Dict[str, float],
    turns: Dict[str, int],
    participation: Dict[str, float],
) -> Tuple[MeetingInsights, Literal["json", "regex_fallback"]]:
    extracted = try_extract_json_object(llm_text)
    if extracted:
        try:
            obj = json.loads(extracted)
            if isinstance(obj, dict):
                insights = coerce_meeting_insights(obj, talk_time, turns, participation)
                return insights, "json"
        except Exception:
            pass

    partial = regex_fallback_parse(llm_text)
    insights = coerce_meeting_insights(partial, talk_time, turns, participation)
    # marca que ha fallado parseo
    d = insights.model_dump()
    d["quality_flags"] = (d.get("quality_flags") or []) + ["llm_json_parse_failed"]
    return MeetingInsights.model_validate(d), "regex_fallback"

def regex_fallback_parse(text: str) -> Dict[str, Any]:
    """
    Fallback robusto:
    - Intenta rescatar campos desde JSON parcial/truncado.
    - Si participation viene en fracción (0-1), lo pasa a %.
    """
    text = text or ""
    out: Dict[str, Any] = {}

    # ---- participation_percent: intenta pillar "SPEAKER_00": 0.7585 dentro del bloque participation_percent
    part_block = re.search(r'"participation_percent"\s*:\s*\{([^}]*)', text, flags=re.IGNORECASE | re.DOTALL)
    part: Dict[str, float] = {}

    if part_block:
        block = part_block.group(1)
        for spk, num in re.findall(r'"(SPEAKER_\d+)"\s*:\s*([0-9]+(?:\.[0-9]+)?)', block, flags=re.IGNORECASE):
            try:
                part[spk] = float(num)
            except Exception:
                pass
    else:
        # fallback genérico
        for spk, num in re.findall(r"(SPEAKER_\d+)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?", text, flags=re.IGNORECASE):
            try:
                part[spk] = float(num)
            except Exception:
                pass

    if part:
        s = sum(part.values())
        mx = max(part.values())
        # si parece fracción 0..1 => pásalo a %
        if mx <= 1.0 and s <= 1.5:
            part = {k: round(v * 100.0, 4) for k, v in part.items()}
        out["participation_percent"] = part

    topics_m = re.search(r'"topics"\s*:\s*\[(.*?)\]', text, flags=re.IGNORECASE | re.DOTALL)
    if topics_m:
        # intenta extraer {"name":"...", "weight":0.5}
        topics_raw = topics_m.group(1)
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', topics_raw, flags=re.IGNORECASE)
        weights = re.findall(r'"weight"\s*:\s*([0-9]+(?:\.[0-9]+)?)', topics_raw, flags=re.IGNORECASE)
        topics = []
        for idx, n in enumerate(names[:20]):
            w = float(weights[idx]) if idx < len(weights) else 0.3
            topics.append({"name": clean_text(n), "weight": max(0.0, min(1.0, w))})
        out["topics"] = topics

    # ---- helpers para extraer num en JSON truncado
    def find_nested_score(obj_key: str, field: str) -> Optional[float]:
        m = re.search(
            rf'"{re.escape(obj_key)}"\s*:\s*\{{[^{{}}]*"{re.escape(field)}"\s*:\s*([\-]?[0-9]+(?:\.[0-9]+)?)',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return float(m.group(1)) if m else None

    def find_num(key: str) -> Optional[float]:
        m = re.search(rf'"?{re.escape(key)}"?\s*[:=]\s*([\-]?[0-9]+(?:\.[0-9]+)?)', text, flags=re.IGNORECASE)
        return float(m.group(1)) if m else None

    # collaboration / decisiveness / conflict / valence
    coll = find_nested_score("collaboration", "score_0_100") or find_num("collaboration.score_0_100") or find_num("collaboration_score")
    deci = find_nested_score("decisiveness", "score_0_100") or find_num("decisiveness.score_0_100") or find_num("decisiveness_score")
    conf = find_num("conflict_level_0_100") or find_num("conflict_level") or find_num("conflict")
    val = find_nested_score("atmosphere", "valence") or find_num("atmosphere.valence") or find_num("valence")

    if coll is not None:
        out["collaboration"] = {"score_0_100": coll, "notes": []}
        # intenta sacar notes si vienen
        notes_m = re.search(r'"collaboration"\s*:\s*\{.*?"notes"\s*:\s*\[(.*?)\]', text, flags=re.IGNORECASE | re.DOTALL)
        if notes_m:
            notes_raw = notes_m.group(1)
            notes = re.findall(r'"([^"]+)"', notes_raw)
            out["collaboration"]["notes"] = [clean_text(x) for x in notes[:10]]

    if deci is not None:
        out["decisiveness"] = {"score_0_100": deci, "notes": []}

    if conf is not None:
        out["conflict_level_0_100"] = conf

    if val is not None:
        out["atmosphere"] = {"valence": max(-1.0, min(1.0, val)), "labels": [], "notes": []}
        labels_m = re.search(r'"atmosphere"\s*:\s*\{.*?"labels"\s*:\s*\[(.*?)\]', text, flags=re.IGNORECASE | re.DOTALL)
        if labels_m:
            labels = re.findall(r'"([^"]+)"', labels_m.group(1))
            out["atmosphere"]["labels"] = [clean_text(x) for x in labels[:10]]

    # summary en JSON parcial o formato libre
    msum_json = re.search(r'"summary"\s*:\s*"([^"]{1,2000})"', text, flags=re.IGNORECASE)
    if msum_json:
        out["summary"] = clean_text(msum_json.group(1))
    else:
        msum = re.search(r"(?:summary|resumen)\s*[:=]\s*(.+)", text, flags=re.IGNORECASE)
        if msum:
            out["summary"] = clean_text(msum.group(1))

    # decisions / action_items (muy best-effort)
    dec_list = re.search(r'"decisions"\s*:\s*\[(.*?)\]', text, flags=re.IGNORECASE | re.DOTALL)
    if dec_list:
        out["decisions"] = [clean_text(x) for x in re.findall(r'"([^"]+)"', dec_list.group(1))[:20]]

    ai_list = re.search(r'"action_items"\s*:\s*\[(.*?)\]', text, flags=re.IGNORECASE | re.DOTALL)
    if ai_list:
        tasks = re.findall(r'"task"\s*:\s*"([^"]+)"', ai_list.group(1), flags=re.IGNORECASE)
        out["action_items"] = [{"owner": None, "task": clean_text(t), "due_date": None, "confidence": 0.6} for t in tasks[:20]]

    return out

def coerce_meeting_insights(obj: Dict[str, Any], talk_time: Dict[str, float], turns: Dict[str, int], participation: Dict[str, float]) -> MeetingInsights:
    obj = dict(obj or {})

    # Stats: fuente de verdad
    obj["talk_time_seconds"] = talk_time
    obj["turns"] = turns
    obj["participation_percent"] = participation

    # Defaults base (solo si faltan)
    obj.setdefault("collaboration", {"score_0_100": 50.0, "notes": ["Estimación por defecto."]})
    obj.setdefault("atmosphere", {"valence": 0.0, "labels": ["neutral"], "notes": ["Estimación por defecto."]})
    obj.setdefault("decisiveness", {"score_0_100": 50.0, "notes": ["Estimación por defecto."]})
    obj.setdefault("conflict_level_0_100", 20.0)
    obj.setdefault("topics", [])
    obj.setdefault("decisions", [])
    obj.setdefault("action_items", [])
    obj.setdefault("main_responsible", None)
    obj.setdefault("participant_roles", [])
    obj.setdefault("debates", [])
    obj.setdefault("plan", [])
    obj.setdefault("summary", "Sin resumen estructurado.")
    obj.setdefault("quality_flags", [])

    # --- Normalización anti-placeholders (CLAVE) ---
    # 1) Summary vacío o placeholder: mantenlo placeholder, pero evita "" y None
    summary = obj.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        obj["summary"] = "Sin resumen estructurado."

    # 2) Scores a 0 con notas vacías suele ser placeholder => vuelve a defaults
    def _is_placeholder_score(block: Any, key: str) -> bool:
        if not isinstance(block, dict):
            return True
        val = block.get(key, None)
        notes = block.get("notes", [])
        try:
            v = float(val)
        except Exception:
            return True
        return (v == 0.0) and (not notes)

    if _is_placeholder_score(obj.get("collaboration"), "score_0_100"):
        obj["collaboration"] = {"score_0_100": 50.0, "notes": ["Estimación por defecto."]}

    if _is_placeholder_score(obj.get("decisiveness"), "score_0_100"):
        obj["decisiveness"] = {"score_0_100": 50.0, "notes": ["Estimación por defecto."]}

    # Conflict 0 también suele ser placeholder si además el resto viene vacío
    try:
        conf = float(obj.get("conflict_level_0_100", 20.0))
    except Exception:
        conf = 20.0
    if conf == 0.0 and not obj.get("decisions") and not obj.get("action_items"):
        obj["conflict_level_0_100"] = 20.0

    # Atmosphere sin labels y sin notes: rellena neutral
    atm = obj.get("atmosphere")
    if isinstance(atm, dict):
        labels = atm.get("labels") or []
        notes = atm.get("notes") or []
        if not labels and not notes:
            obj["atmosphere"] = {"valence": float(atm.get("valence", 0.0) or 0.0), "labels": ["neutral"], "notes": []}

    return MeetingInsights.model_validate(obj)

def clamp_handoff_size(handoff: HandoffState, max_tokens: Optional[int] = None) -> HandoffState:
    """
    Recorte conservador y "seguro":
    - No borra opening_context
    - Preserva ledger diverso (peso/persistencia + temprano + reciente)
    - Mantiene anclas (salient_mentions) y preguntas abiertas
    - topics SIEMPRE derivado del ledger (campo no autoritativo del LLM)
    """
    if max_tokens is None:
        # usa el global existente para no romper tu pipeline
        max_tokens = HANDOFF_MAX_TOKENS

    raw = json.dumps(handoff.model_dump(), ensure_ascii=False, separators=(",", ":"))
    if estimate_tokens(raw) <= max_tokens:
        # aún así garantizamos derivación pro
        handoff.topics = derive_topics_from_ledger(
            handoff.topic_ledger,
            processed_until_sec=handoff.processed_until_sec,
            k=6,
        )
        return handoff

    h = handoff.model_copy(deep=True)

    # 1) textos protegidos
    h.opening_context = clip_text(h.opening_context, 600)
    h.running_summary = clip_text(h.running_summary, 900)

    # 2) logs
    h.decisions = (h.decisions or [])[-12:]
    h.action_items = (h.action_items or [])[-12:]
    h.open_loops = (h.open_loops or [])[:10]

    # 3) salient mentions: early+late (diversidad temporal)
    if len(h.salient_mentions) > 15:
        early = sorted(h.salient_mentions, key=lambda m: m.t)[:7]
        late = sorted(h.salient_mentions, key=lambda m: m.t, reverse=True)[:8]
        seen = set()
        merged = []
        for m in (early + list(reversed(late))):
            k = (round(m.t, 2), _norm_key(m.text))
            if k in seen:
                continue
            seen.add(k)
            merged.append(m)
        h.salient_mentions = merged[:15]

    # 4) ledger diverso (importancia + temprano + reciente)
    if len(h.topic_ledger) > 18:
        h.topic_ledger = _select_ledger_diverse(h.topic_ledger, 18)

    for t in h.topic_ledger:
        t.bullets = (t.bullets or [])[:3]
        t.evidence = (t.evidence or [])[:2]
        # si tu ledger usa salience/hits, asegúralos aquí
        if hasattr(t, "salience_0_1"):
            t.salience_0_1 = _clamp01(getattr(t, "salience_0_1", 0.5))
        if hasattr(t, "hits"):
            t.hits = max(1, int(getattr(t, "hits", 1) or 1))

    # 5) métricas compactas
    if h.collaboration and len(h.collaboration.notes) > 8:
        h.collaboration.notes = h.collaboration.notes[:8]
    if h.atmosphere:
        h.atmosphere.labels = (h.atmosphere.labels or [])[:8]
        h.atmosphere.notes = (h.atmosphere.notes or [])[:8]
    if h.decisiveness and len(h.decisiveness.notes) > 8:
        h.decisiveness.notes = h.decisiveness.notes[:8]
    h.quality_flags = (h.quality_flags or [])[:10]

    h.topics = derive_topics_from_ledger(
        h.topic_ledger,
        processed_until_sec=h.processed_until_sec,
        k=6,
    )

    return h

def coerce_handoff(
    obj: Dict[str, Any],
    chunk_index: int,
    chunks_total: int,
    processed_until_sec: Optional[float],
) -> HandoffState:
    base = dict(obj or {})
    base.setdefault("chunk_index", chunk_index)
    base.setdefault("chunks_total", chunks_total)
    base.setdefault("processed_until_sec", processed_until_sec)
    base.setdefault("running_summary", "")
    base.setdefault("topics", [])
    base.setdefault("decisions", [])
    base.setdefault("action_items", [])
    base.setdefault("quality_flags", [])
    h = HandoffState.model_validate(base)
    return clamp_handoff_size(h)

def parse_chunk_output_or_fallback(
    llm_text: str,
    talk_time: Dict[str, float],
    turns: Dict[str, int],
    participation: Dict[str, float],
    chunk_index: int,
    chunks_total: int,
    processed_until_sec: Optional[float],
    prev_handoff: HandoffState,
) -> Tuple[MeetingInsights, HandoffState, Literal["json", "regex_fallback"]]:
    """
    Espera {"insights":{...},"handoff":{...}}.
    Si no, intenta usar el JSON como insights directo.
    """
    extracted = try_extract_json_object(llm_text)
    if extracted:
        try:
            root = json.loads(extracted)

            # Caso ideal: root tiene insights/handoff
            if isinstance(root, dict) and ("insights" in root or "handoff" in root):
                insights_obj = root.get("insights", {}) if isinstance(root.get("insights", {}), dict) else {}
                handoff_obj = root.get("handoff", {}) if isinstance(root.get("handoff", {}), dict) else {}

                insights = coerce_meeting_insights(insights_obj, talk_time, turns, participation)

                # Si el modelo no manda handoff, derivamos del anterior + insights
                if not handoff_obj:
                    handoff_obj = prev_handoff.model_dump()
                    # actualiza algunos campos desde insights
                    handoff_obj["running_summary"] = insights.summary
                    handoff_obj["topics"] = [t.model_dump() for t in insights.topics]
                    handoff_obj["decisions"] = insights.decisions
                    handoff_obj["action_items"] = [a.model_dump() for a in insights.action_items]
                    handoff_obj["collaboration"] = insights.collaboration.model_dump()
                    handoff_obj["atmosphere"] = insights.atmosphere.model_dump()
                    handoff_obj["decisiveness"] = insights.decisiveness.model_dump()
                    handoff_obj["conflict_level_0_100"] = insights.conflict_level_0_100

                handoff = coerce_handoff(handoff_obj, chunk_index, chunks_total, processed_until_sec)
                return insights, handoff, "json"

            # Si root parece un MeetingInsights directo
            if isinstance(root, dict):
                insights = coerce_meeting_insights(root, talk_time, turns, participation)
                handoff = coerce_handoff(
                    {
                        "chunk_index": chunk_index,
                        "chunks_total": chunks_total,
                        "processed_until_sec": processed_until_sec,
                        "running_summary": insights.summary,
                        "topics": [t.model_dump() for t in insights.topics],
                        "decisions": insights.decisions,
                        "action_items": [a.model_dump() for a in insights.action_items],
                        "collaboration": insights.collaboration.model_dump(),
                        "atmosphere": insights.atmosphere.model_dump(),
                        "decisiveness": insights.decisiveness.model_dump(),
                        "conflict_level_0_100": insights.conflict_level_0_100,
                        "quality_flags": ["llm_output_missing_root_keys"],
                    },
                    chunk_index,
                    chunks_total,
                    processed_until_sec,
                )
                return insights, handoff, "json"
        except Exception:
            pass

    # Regex fallback
    partial = regex_fallback_parse(llm_text)
    insights = coerce_meeting_insights(partial, talk_time, turns, participation)

    # handoff derivado del anterior + lo nuevo
    derived = prev_handoff.model_dump()
    derived["chunk_index"] = chunk_index
    derived["chunks_total"] = chunks_total
    derived["processed_until_sec"] = processed_until_sec

    # actualiza running summary (simple, se puede mejorar)
    if insights.summary and insights.summary not in (derived.get("running_summary") or ""):
        derived["running_summary"] = clip_text((derived.get("running_summary", "") + " " + insights.summary).strip(), 1500)

    # merge dedupe decisions/action_items
    def dedupe_list(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            k = clean_text(x).lower()
            if k and k not in seen:
                seen.add(k)
                out.append(clean_text(x))
        return out

    derived["decisions"] = dedupe_list((derived.get("decisions") or []) + (insights.decisions or []))

    # action_items dedupe by task
    ai_prev = derived.get("action_items") or []
    ai_new = [a.model_dump() for a in insights.action_items]
    by_task = {clean_text(x.get("task", "")).lower(): x for x in ai_prev if x.get("task")}
    for x in ai_new:
        k = clean_text(x.get("task", "")).lower()
        if k:
            by_task[k] = x
    derived["action_items"] = list(by_task.values())

    # topics: simple append + keep top
    derived["topics"] = [t.model_dump() for t in insights.topics] if insights.topics else (derived.get("topics") or [])

    derived["collaboration"] = insights.collaboration.model_dump()
    derived["atmosphere"] = insights.atmosphere.model_dump()
    derived["decisiveness"] = insights.decisiveness.model_dump()
    derived["conflict_level_0_100"] = insights.conflict_level_0_100
    derived.setdefault("quality_flags", [])
    derived["quality_flags"] = (derived["quality_flags"] + ["llm_json_parse_failed"])[:10]

    handoff = coerce_handoff(derived, chunk_index, chunks_total, processed_until_sec)
    return insights, handoff, "regex_fallback"


def _cleanup_old_charts(dirpath: Path, ttl_s: int) -> None:
    if ttl_s <= 0:
        return
    now = time.time()
    try:
        for p in dirpath.glob("*.png"):
            try:
                if (now - p.stat().st_mtime) > ttl_s:
                    p.unlink(missing_ok=True)
            except Exception:
                continue
    except Exception:
        pass

def save_chart_png(png_bytes: bytes) -> str:
    _cleanup_old_charts(INSIGHTS_CHARTS_DIR, INSIGHTS_CHARTS_TTL_S)
    chart_id = uuid.uuid4().hex  # 32 chars
    path = INSIGHTS_CHARTS_DIR / f"{chart_id}.png"
    path.write_bytes(png_bytes)
    return chart_id

def make_chart_url(chart_id: str) -> str:
    rel = f"/v1/insights/charts/{chart_id}.png"
    return f"{PUBLIC_BASE_URL}{rel}" if PUBLIC_BASE_URL else rel

def estimate_tokens(text: str) -> int:
    """
    Heurística simple: ~4 chars/token (varía por idioma/tokenizer, pero es estable para presupuestar).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)

def chat_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        parts.append(f"{role}:\n{m.get('content','')}\n")
    parts.append("ASSISTANT:\n")
    return "\n".join(parts)

def _safe_json(obj: Any) -> Any:
    try:
        if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
            return obj
        return str(obj)
    except Exception:
        return None

def _extract_http_error(r: httpx.Response) -> Dict[str, Any]:
    # Intenta parsear JSON de error, si no, texto
    try:
        j = r.json()
        return {"status_code": r.status_code, "body": _safe_json(j)}
    except Exception:
        return {"status_code": r.status_code, "body": r.text}

def _httpx_exc_info(e: Exception) -> Dict[str, Any]:
    # str(e) a veces es vacío (p.ej. ReadTimeout). Mejor devolver info útil.
    return {
        "type": e.__class__.__name__,
        "repr": repr(e),
        "str": str(e),
    }


def _is_ctx_error(text: str) -> bool:
    t = (text or "").lower()
    return any(m.lower() in t for m in _CTX_ERR_MARKERS)

def _norm_key(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items or []:
        k = _norm_key(it)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it.strip())
    return out

def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def derive_topics_from_ledger(
    ledger: List[TopicLedgerItem],
    *,
    processed_until_sec: Optional[float],
    k: int = 6,
) -> List["Topic"]:
    """
    Deriva SIEMPRE handoff.topics desde topic_ledger.
    Selección vanguardista:
      - Score = salience * (1 + log1p(hits)) * bonus_recencia * bonus_temprano
      - Diversidad temporal: intenta cubrir early/mid/late si hay material
      - Weights finales normalizados (suman ~1) a partir del score
    """
    if not ledger:
        return []

    # Duración estimada para normalizar tiempos
    tmax = processed_until_sec
    if tmax is None:
        tmax = max((x.last_seen_sec or 0.0) for x in ledger) or 0.0
    tmax = max(1.0, float(tmax))

    def time_bucket(x: TopicLedgerItem) -> str:
        # bucket por last_seen; si falta, usa first_seen; si falta todo, "mid"
        t = x.last_seen_sec if x.last_seen_sec is not None else x.first_seen_sec
        if t is None:
            return "mid"
        r = float(t) / tmax
        if r <= 0.33:
            return "early"
        if r <= 0.66:
            return "mid"
        return "late"

    scored: List[Tuple[float, TopicLedgerItem]] = []
    for it in ledger:
        sal = _clamp01(it.salience_0_1 if it.salience_0_1 is not None else 0.5)
        hits = max(1, int(it.hits or 1))
        last = float(it.last_seen_sec or 0.0)
        first = float(it.first_seen_sec or last)

        recency = _clamp01(last / tmax)  # 0..1
        earlyness = 1.0 - _clamp01(first / tmax)  # 1..0 (más alto si aparece antes)

        bonus_rec = 1.0 + 0.20 * recency
        bonus_early = 1.0 + 0.12 * earlyness

        score = max(0.01, sal) * (1.0 + math.log1p(hits)) * bonus_rec * bonus_early
        scored.append((score, it))

    # Orden base por score desc
    scored.sort(key=lambda x: x[0], reverse=True)

    # Selección con diversidad temporal (early/mid/late)
    buckets: Dict[str, List[Tuple[float, TopicLedgerItem]]] = {"early": [], "mid": [], "late": []}
    for sc, it in scored:
        buckets[time_bucket(it)].append((sc, it))

    chosen: List[Tuple[float, TopicLedgerItem]] = []
    used = set()

    def pick_from(bucket_name: str, n: int):
        nonlocal chosen
        for sc, it in buckets[bucket_name]:
            if len(chosen) >= k:
                return
            key = _norm_key(it.name)
            if not key or key in used:
                continue
            used.add(key)
            chosen.append((sc, it))
            if sum(1 for _, x in chosen if time_bucket(x) == bucket_name) >= n:
                return

    # Garantiza cobertura si existe material
    if buckets["early"]:
        pick_from("early", 1)
    if buckets["mid"]:
        pick_from("mid", 1)
    if buckets["late"]:
        pick_from("late", 1)

    # Rellena el resto por score global
    for sc, it in scored:
        if len(chosen) >= k:
            break
        key = _norm_key(it.name)
        if not key or key in used:
            continue
        used.add(key)
        chosen.append((sc, it))

    chosen = chosen[:k]

    # Normaliza weights ~1
    total = sum(sc for sc, _ in chosen) or 1.0
    topics = [Topic(name=it.name, weight=float(sc / total)) for sc, it in chosen]
    return topics

def merge_handoff_state(prev: HandoffState, nxt: HandoffState) -> HandoffState:
    merged = prev.model_copy(deep=True)

    merged.chunk_index = nxt.chunk_index
    merged.chunks_total = nxt.chunks_total or prev.chunks_total
    merged.processed_until_sec = nxt.processed_until_sec or prev.processed_until_sec

    # opening_context protegido
    if not merged.opening_context.strip() and (nxt.opening_context or "").strip():
        merged.opening_context = nxt.opening_context.strip()

    # summary: acepta nuevo si trae algo
    merged.running_summary = (nxt.running_summary or "").strip() or (prev.running_summary or "").strip()

    # decisiones y open loops: unión/dedup
    merged.decisions = _dedup_keep_order((prev.decisions or []) + (nxt.decisions or []))
    merged.open_loops = _dedup_keep_order((prev.open_loops or []) + (nxt.open_loops or []))

    # action_items: dedup por owner|task normalizado (igual que tenías, resumido)
    ai_map: Dict[str, Any] = {}
    for ai in (prev.action_items or []) + (nxt.action_items or []):
        owner = getattr(ai, "owner", "") or ""
        task = getattr(ai, "task", "") or ""
        k = _norm_key(owner) + "|" + _norm_key(task)
        if not k.strip("|"):
            continue
        ai_map[k] = ai
    merged.action_items = list(ai_map.values())

    # ledger: unión por nombre normalizado, append-only
    ledger_map: Dict[str, TopicLedgerItem] = {}

    def ingest(item: TopicLedgerItem, *, bump_hit: bool):
        key = _norm_key(item.name)
        if not key:
            return
        if key not in ledger_map:
            # copia segura
            base = item.model_copy(deep=True)
            base.hits = max(1, int(base.hits or 1))
            if bump_hit:
                base.hits += 1
            base.salience_0_1 = _clamp01(base.salience_0_1)
            base.bullets = _dedup_keep_order(base.bullets)[:3]
            base.evidence = (base.evidence or [])[:2]
            ledger_map[key] = base
            return

        cur = ledger_map[key]
        # hits
        cur.hits = max(1, int(cur.hits or 1))
        if bump_hit:
            cur.hits += 1
        cur.hits = min(cur.hits, 999)  # sanity

        # salience: max (estabiliza)
        cur.salience_0_1 = max(_clamp01(cur.salience_0_1), _clamp01(item.salience_0_1))

        # tiempos
        if item.first_seen_sec is not None:
            cur.first_seen_sec = item.first_seen_sec if cur.first_seen_sec is None else min(cur.first_seen_sec, item.first_seen_sec)
        if item.last_seen_sec is not None:
            cur.last_seen_sec = item.last_seen_sec if cur.last_seen_sec is None else max(cur.last_seen_sec, item.last_seen_sec)

        # bullets
        cur.bullets = _dedup_keep_order((cur.bullets or []) + (item.bullets or []))[:3]

        # evidence dedup (t|snippet)
        seen_e = {(round(e.t, 2), _norm_key(e.snippet)) for e in (cur.evidence or [])}
        for e in (item.evidence or []):
            ek = (round(e.t, 2), _norm_key(e.snippet))
            if ek in seen_e:
                continue
            cur.evidence.append(e)
            seen_e.add(ek)
        cur.evidence = cur.evidence[:2]

        ledger_map[key] = cur

    # ingesta prev (sin bump) + nxt (con bump)
    for it in (prev.topic_ledger or []):
        ingest(it, bump_hit=False)
    for it in (nxt.topic_ledger or []):
        ingest(it, bump_hit=True)

    merged.topic_ledger = list(ledger_map.values())

    # salient mentions: unión/dedup por (t|text)
    merged.salient_mentions = list(prev.salient_mentions or [])
    seen_m = {(round(m.t, 2), _norm_key(m.text)) for m in merged.salient_mentions}
    for m in (nxt.salient_mentions or []):
        mk = (round(m.t, 2), _norm_key(m.text))
        if mk in seen_m:
            continue
        merged.salient_mentions.append(m)
        seen_m.add(mk)

    # métricas: preferimos nxt si trae
    merged.collaboration = nxt.collaboration or prev.collaboration
    merged.atmosphere = nxt.atmosphere or prev.atmosphere
    merged.decisiveness = nxt.decisiveness or prev.decisiveness
    merged.conflict_level_0_100 = nxt.conflict_level_0_100 if nxt.conflict_level_0_100 is not None else prev.conflict_level_0_100
    merged.quality_flags = _dedup_keep_order((prev.quality_flags or []) + (nxt.quality_flags or []))[:12]

    merged.topics = derive_topics_from_ledger(
        merged.topic_ledger,
        processed_until_sec=merged.processed_until_sec,
        k=6,
    )

    return merged


_ws_re = re.compile(r"\s+")
_control_chars_re = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")

# --- Heurísticas de segmentación/compresión para raw_transcript_json
RAW_SENTENCE_END_RE = re.compile(r"[.!?…]+$")

def _get_rel_seconds(ts_obj: Any) -> Optional[float]:
    """
    ts_obj: {"absolute": "...", "relative": 15.54} o float/int
    """
    if ts_obj is None:
        return None
    if isinstance(ts_obj, (int, float)):
        return float(ts_obj)
    if isinstance(ts_obj, dict):
        rel = ts_obj.get("relative", None)
        try:
            return float(rel) if rel is not None else None
        except Exception:
            return None
    return None

def segments_from_raw_transcript_json_turns(
    raw_transcript_json: Any,
    sort_by_time: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convierte un payload word-level (Teams/meeting) a segments "turn-based" mínimos:
      [{"speaker": <nombre>, "start": <sec>, "end": <sec>, "text": <frase>}, ...]

    Nota:
    - Para minimizar tokens, IGNORAMOS timestamps absolute y cualquier metadata.
    - start/end se calculan desde primer/último word timestamp relativo.
    - speaker usa participant.name si existe.
    """
    # 1) Normaliza "items" (lista de bloques con {words, participant})
    items: List[Dict[str, Any]] = []
    if isinstance(raw_transcript_json, list):
        items = [x for x in raw_transcript_json if isinstance(x, dict)]
    elif isinstance(raw_transcript_json, dict):
        # heurística: intenta claves típicas
        for k in ("utterances", "turns", "segments", "items", "transcript"):
            v = raw_transcript_json.get(k)
            if isinstance(v, list):
                items = [x for x in v if isinstance(x, dict)]
                break
        if not items and "results" in raw_transcript_json and isinstance(raw_transcript_json["results"], list):
            items = [x for x in raw_transcript_json["results"] if isinstance(x, dict)]

    segs: List[Dict[str, Any]] = []

    for it in items:
        words = it.get("words") or []
        participant = it.get("participant") or {}

        # speaker name (mínimo)
        speaker = participant.get("name") or participant.get("displayName") or participant.get("full_name") or "UNKNOWN"

        # text (frase completa)
        text = it.get("text")
        if not text:
            text = _join_words_minimal(words)
        text = (text or "").strip()
        if not text:
            continue

        # timestamps relativos (mínimo)
        start = None
        end = None
        if words:
            start = _pick_relative_ts((words[0] or {}).get("start_timestamp"))
            end = _pick_relative_ts((words[-1] or {}).get("end_timestamp"))

        # fallback si vienen timestamps a nivel del item (a veces pasa)
        if start is None:
            start = _pick_relative_ts(it.get("start_timestamp"))
        if end is None:
            end = _pick_relative_ts(it.get("end_timestamp"))

        # último fallback: 0.0
        if start is None:
            start = 0.0
        if end is None:
            end = max(start, start + 0.01)

        segs.append(
            {
                "speaker": str(speaker),
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )

    if sort_by_time:
        segs.sort(key=lambda x: (x.get("start") or 0.0, x.get("end") or 0.0))

    return segs

def _speaker_from_participant(participant: Dict[str, Any]) -> str:
    """
    Speaker para TOON: usa el nombre (lo que quieres en el prompt).
    Fallback: P{id} si no hay nombre; y si tampoco, SPEAKER_UNKNOWN.
    """
    name = participant.get("name")
    if name:
        return normalize_speaker_label(name)

    pid = participant.get("id", None)
    if pid is not None:
        return normalize_speaker_label(f"P{pid}")

    return "SPEAKER_UNKNOWN"

def _join_tokens_smart(tokens: List[str]) -> str:
    """
    Une tokens intentando no meter espacios antes de puntuación.
    """
    out = ""
    for t in tokens:
        t = (t or "").strip()
        if not t:
            continue
        if not out:
            out = t
            continue
        # no espacio antes de puntuación final
        if t in (".", ",", ";", ":", "!", "?", "…") or RAW_SENTENCE_END_RE.search(t):
            out += t
        # no espacio después de paréntesis/apertura
        elif out.endswith(("(", "[", "{", "«", "“", "\"")):
            out += t
        else:
            out += " " + t
    return out

def segments_from_raw_transcript_json_phrases(
    raw: List[Dict[str, Any]],
    *,
    max_gap_s: float = 1.2,
    flush_on_sentence_end: bool = True,
    max_chars_per_segment: int = 280,
) -> List[Dict[str, Any]]:
    """
    Convierte raw_transcript_json (word-level) a segments compactos tipo:
      {start: float, end: float, speaker: "P300", text: "Frase completa."}

    Reglas de "frase":
      - se acumulan palabras del MISMO speaker
      - se corta si:
          a) speaker cambia
          b) gap entre palabras > max_gap_s
          c) aparece final de frase (., ?, !, …) si flush_on_sentence_end
          d) text supera max_chars_per_segment (para no inflar)
    IMPORTANTE: end = end_timestamp.relative de la ÚLTIMA palabra de esa frase.
    """
    if not raw:
        return []

    segments: List[Dict[str, Any]] = []

    cur_speaker: Optional[str] = None
    cur_tokens: List[str] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None

    def flush():
        nonlocal cur_speaker, cur_tokens, cur_start, cur_end
        if not cur_speaker or not cur_tokens or cur_start is None or cur_end is None:
            cur_speaker, cur_tokens, cur_start, cur_end = None, [], None, None
            return

        text = clean_text(_join_tokens_smart(cur_tokens))
        if text:
            segments.append(
                {
                    "speaker": cur_speaker,
                    "start": float(cur_start),
                    "end": float(cur_end),  # <- timestamp de la última palabra
                    "text": text,
                }
            )
        cur_speaker, cur_tokens, cur_start, cur_end = None, [], None, None

    for block in raw:
        participant = block.get("participant") or {}
        speaker = _speaker_from_participant(participant)

        words = block.get("words") or []
        if not isinstance(words, list) or not words:
            continue

        for w in words:
            if not isinstance(w, dict):
                continue
            token = str(w.get("text", "")).strip()
            if not token:
                continue

            st = _get_rel_seconds(w.get("start_timestamp"))
            en = _get_rel_seconds(w.get("end_timestamp"))

            # si faltan timestamps relativos, no podemos ubicar bien => ignora token
            if st is None and en is None:
                continue
            if st is None:
                st = float(en)
            if en is None:
                en = float(st)

            st = float(st)
            en = float(en)

            # inicia si no hay frase en curso
            if cur_speaker is None:
                cur_speaker = speaker
                cur_tokens = [token]
                cur_start = st
                cur_end = en
            else:
                # si cambia speaker => flush
                if speaker != cur_speaker:
                    flush()
                    cur_speaker = speaker
                    cur_tokens = [token]
                    cur_start = st
                    cur_end = en
                else:
                    # mismo speaker => revisa gap
                    gap = (st - float(cur_end)) if cur_end is not None else 0.0
                    if gap > max_gap_s:
                        flush()
                        cur_speaker = speaker
                        cur_tokens = [token]
                        cur_start = st
                        cur_end = en
                    else:
                        cur_tokens.append(token)
                        cur_end = en  # <- SIEMPRE la última palabra

            # cortes por final de frase
            if flush_on_sentence_end and RAW_SENTENCE_END_RE.search(token):
                flush()
                continue

            # cortes por longitud (para no meter párrafos gigantes en el LLM)
            if max_chars_per_segment > 0:
                approx = len(_join_tokens_smart(cur_tokens))
                if approx >= max_chars_per_segment:
                    flush()

    flush()

    # orden por tiempo
    segments.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    return segments

def clean_text(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = _control_chars_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s

def clip_text(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"

def normalize_speaker_label(speaker: Optional[str]) -> str:
    if not speaker:
        return "SPEAKER_UNKNOWN"
    s = str(speaker).strip()
    if not s:
        return "SPEAKER_UNKNOWN"
    # evita romper TOON "SPEAKER|start-end|text"
    s = s.replace("|", "/")
    s = s.replace("\n", " ").replace("\r", " ").strip()
    return s or "SPEAKER_UNKNOWN"

def _join_words_minimal(words: List[Dict[str, Any]]) -> str:
    """
    Une words -> frase. Minimiza tokens:
    - join por espacios
    - quita espacios antes de puntuación
    - quita espacios después de signos de apertura (¿ ¡ ( [ {)
    """
    toks: List[str] = []
    for w in words or []:
        t = (w or {}).get("text")
        if not t:
            continue
        t = str(t).strip()
        if t:
            toks.append(t)

    if not toks:
        return ""

    s = " ".join(toks)
    s = _punct_no_space_before.sub(r"\1", s)
    s = _punct_no_space_after_open.sub(r"\1", s)
    s = _multi_space.sub(" ", s).strip()
    return s

def _pick_relative_ts(ts_obj: Any) -> Optional[float]:
    """
    Extrae timestamp en segundos relativos si existe.
    Acepta estructuras tipo:
      {"absolute": "...", "relative": 2067.53}
    """
    if ts_obj is None:
        return None
    if isinstance(ts_obj, (int, float)):
        return float(ts_obj)
    if isinstance(ts_obj, dict):
        rel = ts_obj.get("relative")
        if isinstance(rel, (int, float)):
            return float(rel)
    return None

def _join_tokens_smart(tokens: List[str]) -> str:
    out = ""
    for t in tokens:
        if not t:
            continue
        t = str(t)
        if not out:
            out = t
            continue

        if t in _punct_no_space_before:
            out += t
        elif out[-1] in _punct_no_space_after:
            out += t
        else:
            out += " " + t
    return out

def segments_from_raw_transcript_json(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convierte tu raw_transcript_json (lista de bloques con words + participant)
    a una lista de segments compatible con extract_segments():
      {start: float, end: float, speaker: str, text: str}

    Usa timestamps.relative (segundos).
    """
    segments: List[Dict[str, Any]] = []
    if not raw:
        return segments

    def _get_rel(ts_obj: Any) -> Optional[float]:
        # ts_obj: {"absolute": "...", "relative": 15.54} o directo float
        if ts_obj is None:
            return None
        if isinstance(ts_obj, (int, float)):
            return float(ts_obj)
        if isinstance(ts_obj, dict):
            rel = ts_obj.get("relative", None)
            try:
                return float(rel) if rel is not None else None
            except Exception:
                return None
        return None

    for block in raw:
        words = block.get("words") or []
        if not isinstance(words, list) or not words:
            continue

        participant = block.get("participant") or {}
        pid = participant.get("id", None)
        pname = participant.get("name", None)

        speaker = pname or (f"PARTICIPANT_{pid}" if pid is not None else None)
        speaker = normalize_speaker_label(speaker)

        # tokens text
        token_texts: List[str] = []
        starts: List[float] = []
        ends: List[float] = []

        for w in words:
            if not isinstance(w, dict):
                continue
            token_texts.append(str(w.get("text", "")).strip())

            st = _get_rel(w.get("start_timestamp"))
            en = _get_rel(w.get("end_timestamp"))
            if st is not None:
                starts.append(st)
            if en is not None:
                ends.append(en)

        text = clean_text(_join_tokens_smart(token_texts))
        if not text:
            continue

        # timestamps
        if not starts and not ends:
            # sin tiempos: no podemos calcular duraciones => salta
            continue

        start = min(starts) if starts else (min(ends) if ends else 0.0)
        end = max(ends) if ends else (max(starts) if starts else start)

        segments.append(
            {
                "start": float(start),
                "end": float(max(end, start)),
                "speaker": speaker,
                "text": text,
            }
        )

    # orden por tiempo
    segments.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
    return segments

def extract_segments(transcript: Dict[str, Any]) -> List[NormalizedUtterance]:
    segs = transcript.get("segments", [])
    out: List[NormalizedUtterance] = []

    for seg in segs:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            speaker = normalize_speaker_label(seg.get("speaker"))
            text = clean_text(str(seg.get("text", "")))
            if text:
                out.append(NormalizedUtterance(speaker=speaker, start=start, end=end, text=text))
        except Exception:
            continue

    out.sort(key=lambda x: (x.start, x.end))
    return out

def merge_adjacent(utterances: List[NormalizedUtterance], merge_gap_ms: int) -> List[NormalizedUtterance]:
    if not utterances:
        return utterances
    gap_s = max(0, merge_gap_ms) / 1000.0

    merged: List[NormalizedUtterance] = []
    cur = utterances[0]

    for nxt in utterances[1:]:
        same = (nxt.speaker == cur.speaker)
        close = (nxt.start - cur.end) <= gap_s
        if same and close:
            cur = NormalizedUtterance(
                speaker=cur.speaker,
                start=cur.start,
                end=max(cur.end, nxt.end),
                text=clean_text(cur.text + " " + nxt.text),
            )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged

def compute_talk_stats(utterances: List[NormalizedUtterance]) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
    talk_time: Dict[str, float] = {}
    turns: Dict[str, int] = {}
    for u in utterances:
        dur = max(0.0, u.end - u.start)
        talk_time[u.speaker] = talk_time.get(u.speaker, 0.0) + dur
        turns[u.speaker] = turns.get(u.speaker, 0) + 1

    total = sum(talk_time.values()) or 1.0
    participation = {spk: (secs / total) * 100.0 for spk, secs in talk_time.items()}

    talk_time = {k: round(v, 2) for k, v in talk_time.items()}
    participation = {k: round(v, 2) for k, v in participation.items()}
    return talk_time, turns, participation

def _select_ledger_diverse(ledger: List[TopicLedgerItem], k: int) -> List[TopicLedgerItem]:
    if not ledger or k <= 0:
        return []

    # ranking 1: salience/hits (importancia+persistencia)
    by_importance = sorted(
        ledger,
        key=lambda x: (float(x.salience_0_1 or 0.0), int(x.hits or 1)),
        reverse=True,
    )

    keep: List[TopicLedgerItem] = []
    used = set()

    def add(it: TopicLedgerItem):
        key = _norm_key(it.name)
        if not key or key in used:
            return
        used.add(key)
        keep.append(it)

    # 50% por importancia
    for it in by_importance[: max(1, k // 2)]:
        add(it)

    # 25% por temprano
    early = sorted(ledger, key=lambda x: float(x.first_seen_sec or 1e18))
    for it in early[: max(1, k // 4)]:
        add(it)

    # resto por reciente
    recent = sorted(ledger, key=lambda x: float(x.last_seen_sec or -1.0), reverse=True)
    for it in recent:
        if len(keep) >= k:
            break
        add(it)

    return keep[:k]


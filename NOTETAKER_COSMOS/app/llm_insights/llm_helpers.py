from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Literal, Tuple
import httpx
from fastapi import HTTPException
import asyncio
from collections import Counter
from collections import deque

from .models import (
    NormalizedUtterance,
    MeetingInsights,
    AnalyzeRequest,
    HandoffState,

)
from .utils import (
    segments_from_raw_transcript_json_turns,
    clean_text,
    clip_text,
    estimate_tokens,
    chat_messages_to_prompt,
    _extract_http_error,
    _is_ctx_error
)

LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "1"))
LLM_SEM = asyncio.Semaphore(max(1, LLM_MAX_CONCURRENCY))
LLAMA_CTX_TOKENS = int(os.getenv("LLAMA_CTX_TOKENS", "18000"))
LLM_CTX_MARGIN = int(os.getenv("LLM_CTX_MARGIN", "256"))
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8014").rstrip("/")
LLAMA_API_MODE: Literal["openai", "llamacpp"] = os.getenv("LLAMA_API_MODE", "openai")  # openai | llamacpp
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "gpt-oss-20b")
LLAMA_TIMEOUT_S = float(os.getenv("LLAMA_TIMEOUT_S", "120"))
_CANDIDATE_MAX_SEGMENTS = int(os.getenv("LLM_CANDIDATE_MAX_SEGMENTS", "140"))
_CANDIDATE_HINT_RE = re.compile(
    r"\b("
    r"voy\s+a|vamos\s+a|tengo\s+que|tenemos\s+que|hay\s+que|"
    r"me\s+encargo|lo\s+hago|lo\s+hacemos|lo\s+env[ií]o|te\s+env[ií]o|"
    r"queda|acordamos|decidimos|cerramos|definimos|"
    r"plazo|para\s+el|el\s+d[ií]a|mañana|hoy|pasado\s+mañana|jueves|viernes|lunes|martes|miércoles|miercoles|sábado|sabado|domingo"
    r")\b",
    re.IGNORECASE,
)


def _prefilter_segments_for_candidates(segments: List[Dict[str, Any]], max_keep: int) -> List[Dict[str, Any]]:
    """
    Reduce el número de segmentos para el LLM sin depender de cues rígidos.
    Mantiene:
      - los que disparan _CANDIDATE_HINT_RE
      - y una muestra de “contexto” alrededor
    """
    if not segments:
        return []

    n = len(segments)
    if n <= max_keep:
        return segments

    flagged = [i for i, s in enumerate(segments) if _CANDIDATE_HINT_RE.search((s.get("text") or ""))]
    keep = set()

    # ventana de contexto alrededor de flagged
    for i in flagged:
        for j in range(max(0, i - 1), min(n, i + 2)):
            keep.add(j)

    # cobertura de cierre (en reuniones largas las tareas suelen fijarse al final)
    tail_start = max(0, n - max(8, max_keep // 8))
    for j in range(tail_start, n):
        keep.add(j)

    # si aún faltan, prioriza contenido informativo + cobertura uniforme por posición
    if len(keep) < max_keep:
        scored: List[Tuple[float, int]] = []
        for i, seg in enumerate(segments):
            if i in keep:
                continue
            txt = (seg.get("text") or "")
            if not txt.strip():
                continue

            score = 0.0
            if _CANDIDATE_HINT_RE.search(txt):
                score += 6.0

            # tokens accionables / decisionales con peso suave
            low = txt.lower()
            if any(k in low for k in ("decid", "acord", "queda", "cerr", "defin")):
                score += 2.0
            if any(k in low for k in ("voy a", "vamos a", "hay que", "te ", "os ", "mañana", "hoy", "plazo")):
                score += 1.5

            # favorece segmentos más largos (sin pasarnos)
            score += min(2.0, len(txt) / 180.0)

            # ligera preferencia por mitad/final para evitar sesgo al inicio
            score += (i / max(1, n - 1)) * 0.8
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        for _, idx in scored:
            keep.add(idx)
            if len(keep) >= max_keep:
                break

    if len(keep) < max_keep:
        step = max(1, n // max_keep)
        for j in range(0, n, step):
            keep.add(j)
            if len(keep) >= max_keep:
                break

    idxs = sorted(list(keep))[:max_keep]
    return [segments[i] for i in idxs]


def build_llm_messages_extract_candidates_es(segments: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Prompt dedicado para extraer candidatos anclados.
    Salida:
      { "actions":[...], "decisions":[...] }
    Reglas críticas:
      - segment_idx debe existir
      - verbatim debe ser substring EXACTO de segments[segment_idx].text
    """
    # compacta payload
    compact = []
    for i, s in enumerate(segments):
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        compact.append({
            "segment_idx": i,
            "speaker": s.get("speaker") or "Desconocido",
            "start": float(s.get("start") or 0.0),
            "end": float(s.get("end") or 0.0),
            "text": clip_text(txt, 260),
        })

    schema = {
        "actions": [
            {
                "segment_idx": 0,
                "owner": "Speaker",
                "verbatim": "Te envío el correo hoy por la tarde",
                "due_text": "hoy por la tarde",
                "confidence_0_1": 0.85
            }
        ],
        "decisions": [
            {
                "segment_idx": 0,
                "verbatim": "Decidimos empezar con el entero",
                "effective_text": None,
                "confidence_0_1": 0.78
            }
        ]
    }

    system = (
        "Eres un extractor de hechos para reuniones.\n"
        "Tu trabajo: detectar SOLO acciones y decisiones EXPLÍCITAS.\n\n"
        "CRÍTICO (anti-alucinación):\n"
        "- Devuelve SOLO JSON válido (sin markdown).\n"
        "- Cada item DEBE referenciar un segment_idx existente.\n"
        "- verbatim DEBE ser un substring EXACTO (contiguo) del text del segmento.\n"
        "- Si no hay evidencia explícita, NO lo devuelvas.\n"
        "- No reescribas verbatim: cópialo tal cual.\n"
        "- confidence_0_1 en [0,1].\n"
        "- owner: usa el speaker del segmento si es una tarea; si no está claro, null.\n"
    )

    user = (
        "Extrae candidatos.\n\n"
        "SEGMENTS (compacto):\n"
        f"{json.dumps(compact, ensure_ascii=False, separators=(',',':'))}\n\n"
        "ESQUEMA (misma forma, mismas claves):\n"
        f"{json.dumps(schema, ensure_ascii=False, separators=(',',':'))}\n"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_candidates_payload(payload: Any, segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Valida y limpia la salida del LLM.
    Drop sin piedad si:
      - idx fuera de rango
      - verbatim no está en el text
      - confidence fuera de [0,1]

    Mejoras:
      - Canonicaliza owner a uno de los speakers detectados en segments (si match fuerte).
      - Normaliza due_text/effective_text: str o None.
    """
    out = {"actions": [], "decisions": []}
    if not isinstance(payload, dict):
        return out

    seg_texts = [(s.get("text") or "") for s in segments]
    speakers = sorted({(s.get("speaker") or "").strip() for s in segments if (s.get("speaker") or "").strip()})
    speakers_set = set(speakers)

    def clamp01(x: Any, default: float = 0.6) -> float:
        try:
            v = float(x)
        except Exception:
            v = default
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v

    def norm_simple(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = (s.replace("á", "a").replace("é", "e").replace("í", "i")
               .replace("ó", "o").replace("ú", "u").replace("ü", "u").replace("ñ", "n"))
        return s

    def best_match_owner(owner: Optional[str]) -> Optional[str]:
        if not isinstance(owner, str):
            return None
        o = owner.strip()
        if not o:
            return None
        if o in speakers_set:
            return o
        okey = norm_simple(o)
        o_tokens = {x for x in re.findall(r"[a-z0-9]+", okey) if len(x) >= 3}
        best, best_score = None, 0.0
        for cand in speakers:
            ckey = norm_simple(cand)
            c_tokens = {x for x in re.findall(r"[a-z0-9]+", ckey) if len(x) >= 3}
            if not c_tokens:
                continue
            inter = len(o_tokens & c_tokens)
            denom = max(1, min(len(o_tokens) or 1, len(c_tokens)))
            score = inter / denom
            if score > best_score:
                best_score = score
                best = cand
        return best if (best and best_score >= 0.70) else None

    for item in (payload.get("actions") or []):
        if not isinstance(item, dict):
            continue
        idx = item.get("segment_idx")
        if not isinstance(idx, int) or idx < 0 or idx >= len(segments):
            continue

        verbatim = (item.get("verbatim") or "").strip()
        if not verbatim or verbatim not in seg_texts[idx]:
            continue

        owner = item.get("owner", None)
        owner = best_match_owner(owner) if isinstance(owner, str) else None

        due_text = item.get("due_text", None)
        due_text = due_text.strip() if isinstance(due_text, str) and due_text.strip() else None

        out["actions"].append({
            "segment_idx": idx,
            "owner": owner,
            "verbatim": verbatim,
            "due_text": due_text,
            "confidence_0_1": clamp01(item.get("confidence_0_1"), 0.65),
        })

    for item in (payload.get("decisions") or []):
        if not isinstance(item, dict):
            continue
        idx = item.get("segment_idx")
        if not isinstance(idx, int) or idx < 0 or idx >= len(segments):
            continue

        verbatim = (item.get("verbatim") or "").strip()
        if not verbatim or verbatim not in seg_texts[idx]:
            continue

        eff = item.get("effective_text", None)
        eff = eff.strip() if isinstance(eff, str) and eff.strip() else None

        out["decisions"].append({
            "segment_idx": idx,
            "verbatim": verbatim,
            "effective_text": eff,
            "confidence_0_1": clamp01(item.get("confidence_0_1"), 0.65),
        })

    out["actions"].sort(key=lambda x: (x["segment_idx"], -x["confidence_0_1"]))
    out["decisions"].sort(key=lambda x: (x["segment_idx"], -x["confidence_0_1"]))

    out["actions"] = out["actions"][:40]
    out["decisions"] = out["decisions"][:30]
    return out

async def extract_action_decision_candidates_llm(
    segments_norm: List[Dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 650,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extrae candidatos con LLM + validación dura.
    Devuelve:
      { actions:[{segment_idx,owner,verbatim,due_text,confidence_0_1}], decisions:[...] }
    """
    # prefilter para no meter 1000 segmentos al LLM
    reduced = _prefilter_segments_for_candidates(segments_norm, _CANDIDATE_MAX_SEGMENTS)

    messages = build_llm_messages_extract_candidates_es(reduced)

    raw = await call_llama_cpp(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        reasoning_format="none",
    )

    # parse seguro
    try:
        obj = json.loads(_extract_first_json_object(raw))
    except Exception:
        return {"actions": [], "decisions": []}

    # ojo: los idx en reduced son relativos a reduced; necesitamos mapearlos al índice original
    # construimos mapping “posición en reduced” -> “posición en segments_norm”
    mapping: List[Optional[int]] = []
    # reduced suele preservar referencias, pero hacemos fallback robusto para duplicados.
    # clave -> cola de índices originales para no colapsar repeticiones idénticas.
    key_to_orig: Dict[Tuple[str, float, float, str], deque[int]] = {}
    for i, s in enumerate(segments_norm):
        k = (s.get("speaker") or "", float(s.get("start") or 0.0), float(s.get("end") or 0.0), (s.get("text") or "")[:260])
        key_to_orig.setdefault(k, deque()).append(i)

    for s in reduced:
        # fast path por identidad (si reduced comparte referencia con segments_norm)
        oidx = None
        for i, orig in enumerate(segments_norm):
            if s is orig:
                oidx = i
                break
        k = (s.get("speaker") or "", float(s.get("start") or 0.0), float(s.get("end") or 0.0), (s.get("text") or "")[:260])
        if oidx is None:
            q = key_to_orig.get(k)
            if q:
                oidx = q.popleft()
        mapping.append(oidx)

    validated = _validate_candidates_payload(obj, reduced)

    # remapea indices a los originales
    def remap_list(xs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for it in xs:
            ridx = it.get("segment_idx")
            if not isinstance(ridx, int) or ridx < 0 or ridx >= len(mapping):
                continue
            oidx = mapping[ridx]
            if oidx is None:
                continue
            it2 = dict(it)
            it2["segment_idx"] = int(oidx)
            out.append(it2)
        return out

    return {
        "actions": remap_list(validated["actions"]),
        "decisions": remap_list(validated["decisions"]),
    }


def build_toon_v1(
    utterances: List["NormalizedUtterance"],
    fmt: Literal["lines", "json"],
    include_timestamps: bool,
) -> str:
    """
    TOON v1 MIN (para minimizar tokens):
      - include_timestamps=True  => usa SOLO start timestamp (t)
      - include_timestamps=False => omite timestamp (no recomendado si quieres "timestamp por turno")

    lines:
      SPEAKER|t|texto
    json:
      [[SPEAKER,t,text], ...]
    """
    if fmt == "json":
        arr = []
        for u in utterances:
            if include_timestamps:
                arr.append([u.speaker, round(u.start, 2), u.text])
            else:
                arr.append([u.speaker, u.text])
        return json.dumps(arr, ensure_ascii=False, separators=(",", ":"))

    lines: List[str] = []
    for u in utterances:
        if include_timestamps:
            lines.append(f"{u.speaker}|{u.start:.2f}|{u.text}")
        else:
            lines.append(f"{u.speaker}|{u.text}")
    return "\n".join(lines)

def build_toon_summary_v1(
    insights: MeetingInsights,
    talk_time: Dict[str, float],
    turns: Dict[str, int],
    participation: Dict[str, float],
) -> str:
    """
    Brief profesional.
    - No duplica cuantitativos si ya están en insights.summary.
    - Si el summary viene como placeholder ('Sin resumen...'), lo sustituye por un base.
    - Clima y señales formateados en una línea limpia.
    """

    _bad_patterns = (
        "final_handoff",
        "running_summary",
        "cumulative_stats",
        "schema",
        "response_format",
        "we can",
        "we need",
        "is spanish",
        "in spanish",
        "use that",
    )

    _en_stop = {
        "the","and","or","but","so","because","we","you","they","is","are","was","were","be","been","being",
        "this","that","these","those","in","on","at","to","from","for","with","without","as","it","its",
        "can","need","use","using","should","must","will","would","could","if","then","else","there","here",
        "summary","topics","final","handoff","running",
        "due","owner","task","plan","notes","chunk"
    }
    _es_stop = {
        "el","la","los","las","y","o","pero","porque","que","de","del","al","en","con","sin","para","por",
        "es","son","fue","eran","ser","siendo","esto","esa","este","esta","estos","estas","se","lo","su",
        "puede","debe","hay","aquí","allí","resumen","temas","reunión","acuerdos","acciones","puntos",
        "plazo","tarea","plan","notas"
    }

    _word_re_local = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+", re.UNICODE)

    def _looks_like_english(line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return False
        s_low = s.lower()

        for p in _bad_patterns:
            if p in s_low:
                return True

        words = [w.lower() for w in _word_re_local.findall(s_low)]
        if not words:
            return False

        en = sum(1 for w in words if w in _en_stop)
        es = sum(1 for w in words if w in _es_stop)
        return (en >= 2 and en > es * 1.5)

    def _sanitize_spanish_block(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        kept: List[str] = []
        for ln in lines:
            if _looks_like_english(ln):
                continue
            kept.append(ln)
        return clean_text(" ".join(kept)).strip()

    def _norm_label(lbl: str) -> str:
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

    def _labels_clean(labels: List[str], maxn: int = 3) -> List[str]:
        out: List[str] = []
        seen = set()
        for lb in labels or []:
            n = _norm_label(lb)
            if n and n.lower() not in seen:
                seen.add(n.lower())
                out.append(n)
            if len(out) >= maxn:
                break
        return out

    def _fallback_base_summary() -> str:
        return (
            "Reunión de seguimiento orientada a alinear el estado del trabajo y coordinar los próximos pasos. "
            "Se revisaron avances, se identificaron riesgos o bloqueos potenciales y se acordó mantener el foco en el cronograma y las prioridades."
        )

    # -------- Topics ----------
    topics = [t.name for t in (insights.topics or []) if getattr(t, "name", None) and str(getattr(t, "name")).strip()]
    topics_str = ", ".join(topics[:8]) if topics else "No inferidos con confianza suficiente"

    # -------- Top participación ----------
    speakers = list((participation or {}).keys())
    top = sorted((participation or {}).items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_txt = ", ".join([f"{k} {v:.1f}%" for k, v in top]) if top else "sin datos"

    total_talk = sum(max(0.0, float(v)) for v in (talk_time or {}).values())
    total_talk_min = total_talk / 60.0 if total_talk > 0 else 0.0

    # -------- Resumen ejecutivo ----------
    def _expanded_executive_summary() -> str:
        base_raw = insights.summary.strip() if isinstance(insights.summary, str) else ""
        base = _sanitize_spanish_block(base_raw)
        low = base.lower()

        # si viene placeholder, sustitúyelo
        if low.startswith("sin resumen estructurado") or low.startswith("sin resumen") or not base:
            base = _fallback_base_summary()
            low = base.lower()

        parts: List[str] = [base]

        # SOLO añade cuantitativos si no están ya presentes en el summary
        has_quant = any(k in low for k in (
            "participaron", "tiempo total de intervención", "principales intervenciones", "datos cuantitativos"
        ))
        if not has_quant:
            parts.append(
                f"Datos cuantitativos: {len(speakers)} participantes; tiempo total de intervención (habla) ≈ {total_talk_min:.1f} min; "
                f"principales intervenciones: {top_txt}."
            )

        if topics:
            bullets = "\n".join([f"- {t}" for t in topics[:6]])
            parts.append("Puntos clave tratados:\n" + bullets)

        # señales
        coll = float(getattr(insights.collaboration, "score_0_100", 0.0) or 0.0)
        deci = float(getattr(insights.decisiveness, "score_0_100", 0.0) or 0.0)
        conf = float(getattr(insights, "conflict_level_0_100", 0.0) or 0.0)
        labels = _labels_clean(getattr(insights.atmosphere, "labels", []) or [], maxn=3)
        labels_txt = ", ".join(labels) if labels else "neutro"
        parts.append(
            f"Señales: colaboración {coll:.0f}/100, decisión {deci:.0f}/100, conflicto {conf:.0f}/100; "
            f"clima {labels_txt} (valencia {insights.atmosphere.valence:.2f})."
        )

        return "\n\n".join(parts).strip()

    executive_summary = _expanded_executive_summary()

    # -------- Render ----------
    lines: List[str] = []
    lines.append("## Brief de reunión")
    lines.append(f"**Temas tratados:** {topics_str}")
    lines.append("")
    lines.append("### Resumen ejecutivo")
    lines.append(executive_summary)
    lines.append("")
    lines.append("### Carga y roles")

    # roles (si no hay, lista colaboradores)
    if insights.main_responsible:
        lines.append(f"**Responsable principal (mayor carga):** {insights.main_responsible}")

    if speakers:
        for name in sorted(speakers, key=lambda x: participation.get(x, 0.0), reverse=True):
            p = participation.get(name, 0.0)
            tsec = talk_time.get(name, 0.0)
            trn = turns.get(name, 0)
            lines.append(f"- **{name}**: Colaborador — {p:.1f}% · {tsec:.0f}s · {trn} turnos")
    else:
        lines.append("- Sin participantes detectados.")

    lines.append("")
    lines.append("### Señales de reunión (métricas)")
    lines.append(f"- Colaboración: {insights.collaboration.score_0_100:.0f}/100")
    lines.append(f"- Decisión: {insights.decisiveness.score_0_100:.0f}/100")
    lines.append(f"- Conflicto: {insights.conflict_level_0_100:.0f}/100")

    labels_clean = _labels_clean(getattr(insights.atmosphere, "labels", []) or [], maxn=3)
    clima_txt = ", ".join(labels_clean) if labels_clean else "neutro"
    lines.append(f"- Clima: {clima_txt} (valencia {insights.atmosphere.valence:.2f})")

    if insights.quality_flags:
        lines.append("")
        lines.append(f"### Quality flags: {', '.join(insights.quality_flags)}")

    return "\n".join(lines).strip()

# -----------------------------
# Token estimation + chunker
# -----------------------------
_word_re = re.compile(r"\w+", re.UNICODE)
_json_object_re_local = re.compile(r"\{.*\}", re.DOTALL)

_STOP_ES = {
    "el","la","los","las","un","una","unos","unas","y","o","pero","porque","que","de","del","al","en","con","sin",
    "para","por","es","son","fue","eran","ser","siendo","esto","esta","este","estos","estas","se","lo","su","sus",
    "a","e","u","ya","no","sí","si","como","más","menos","muy","también","solo","sólo","hay","ahí","aquí","allí",
    "vale","ok","okay","bueno","pues","entonces","luego","tambien","eso","esa","ese","esas","esos","mi","tu","tú",
    "me","te","nos","os","les","le","yo","tú","tu","él","ella","ellos","ellas","usted","ustedes","vosotros","vosotras",
}

_TOKEN_RE_HINTS = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9._/\-]{1,}", re.UNICODE)
_WORD_RE_ALPHA = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", re.UNICODE)


def _extract_text_from_toon(toon: str) -> str:
    """
    Extrae texto conversacional desde TOON v1 MIN en formato lines o json.
    """
    s = (toon or "").strip()
    if not s:
        return ""
    # Heurística: si parece JSON, intentamos parsear
    if s[0] == "[":
        try:
            arr = json.loads(s)
            parts = []
            for rec in arr:
                if isinstance(rec, list) and rec:
                    parts.append(str(rec[-1]))
            return " ".join(parts)
        except Exception:
            pass

    # Formato lines: SPEAKER|t|text o SPEAKER|text
    parts = []
    for ln in s.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "|" in ln:
            parts.append(ln.split("|")[-1].strip())
        else:
            parts.append(ln)
    return " ".join(parts)


def _salient_hints_from_toon(toon: str, *, max_terms: int = 18, max_phrases: int = 10) -> Dict[str, List[str]]:
    """
    Extrae pistas léxicas locales (sin LLM) para reforzar topics/resumen y evitar alucinaciones.
    Devuelve:
      - terms: tokens (herramientas, acrónimos, productos, nombres, ids...)
      - phrases: bigramas frecuentes (aprox) sin stopwords
    """
    text = _extract_text_from_toon(toon)
    text = clean_text(text)
    if not text:
        return {"terms": [], "phrases": []}

    raw_tokens = _TOKEN_RE_HINTS.findall(text)
    counts = Counter()

    for tok in raw_tokens:
        t = tok.strip()
        if not t:
            continue
        # descartar puramente numéricos
        if t.isdigit():
            continue

        low = t.lower()
        # filtrar stopwords salvo tokens "especiales"
        special = any(ch.isdigit() for ch in t) or any(ch in t for ch in ("_", "-", "/", ".", ":"))
        is_acronym = t.isupper() and len(t) <= 10
        has_caps = any(ch.isupper() for ch in t[1:])  # CamelCase / marcas

        if (low in _STOP_ES) and not (special or is_acronym or has_caps):
            continue
        if len(t) < 3 and not (special or is_acronym):
            continue

        # ponderación: empuja herramientas/acrónimos/tokens con dígitos
        w = 1.0
        if is_acronym:
            w *= 2.3
        if special:
            w *= 1.9
        if has_caps:
            w *= 1.4

        counts[t] += w

    terms = [t for t, _ in counts.most_common(max_terms)]

    # Frases (bigrams) con palabras alfabéticas, sin stopwords
    words = [w.lower() for w in _WORD_RE_ALPHA.findall(text)]
    words = [w for w in words if w not in _STOP_ES and len(w) >= 3]
    bg = Counter()
    for a, b in zip(words, words[1:]):
        if a == b:
            continue
        bg[(a, b)] += 1

    phrases = [" ".join(pair) for pair, _ in bg.most_common(max_phrases)]
    return {"terms": terms, "phrases": phrases}


def _extract_first_json_object(text: str) -> str:
    """
    Extrae el primer objeto JSON { ... } aunque haya basura alrededor.
    - Balancea llaves
    - Respeta strings y escapes
    - Valida con json.loads antes de devolver
    """
    s = (text or "").strip()
    if not s:
        return s

    # Si es JSON dict puro, devuelve tal cual
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return s
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        return s

    in_str = False
    esc = False
    depth = 0

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cand = s[start:i + 1].strip()
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        return cand
                except Exception:
                    # fallback regex (peor, pero a veces salva)
                    m = _json_object_re_local.search(s)
                    return m.group(0).strip() if m else s

    # sin cierre -> regex fallback
    m = _json_object_re_local.search(s)
    return m.group(0).strip() if m else s



def estimate_tokens_rough(text: str) -> int:
    """
    Estimación barata (sin tiktoken).
    Para español suele ir razonable:
      - tokens ~ words * 1.3
      - y/o tokens ~ chars / 4
    usamos el máximo de ambas para ir conservadores.
    """
    text = text or ""
    words = len(_word_re.findall(text))
    by_words = int(words * 1.3)
    by_chars = int(len(text) / 4) + 1
    return max(by_words, by_chars)

def chunk_utterances_by_budget(
    utterances: List[NormalizedUtterance],
    fmt: Literal["lines", "json"],
    include_timestamps: bool,
    token_budget: int,
    max_chunks: int,
) -> List[List[NormalizedUtterance]]:
    """
    Parte por utterances (no corta dentro de un turno).
    token_budget aplica sobre el TOON del chunk (aproximado).
    """
    chunks: List[List[NormalizedUtterance]] = []
    cur: List[NormalizedUtterance] = []
    cur_tokens = 0

    for u in utterances:
        toon_piece = build_toon_v1([u], fmt=fmt, include_timestamps=include_timestamps)
        piece_tokens = estimate_tokens_rough(toon_piece) + 2  # margen

        if cur and (cur_tokens + piece_tokens > token_budget):
            chunks.append(cur)
            if len(chunks) >= max_chunks:
                return chunks
            cur = [u]
            cur_tokens = piece_tokens
        else:
            cur.append(u)
            cur_tokens += piece_tokens

    if cur:
        chunks.append(cur)
    return chunks

def build_llm_messages_single(
    toon: str,
    talk_time: Dict[str, float],
    turns: Dict[str, int],
    participation: Dict[str, float],
    *,
    det_facts_json: Optional[str] = None,
) -> List[Dict[str, str]]:
    hints = _salient_hints_from_toon(toon)
    schema_hint = _schema_hint_with_real_speakers(talk_time, turns, participation)

    system = (
        "Eres analista de reuniones.\n"
        "Entrada: transcript diarizado TOON v1 MIN.\n\n"
        "IDIOMA (CRÍTICO):\n"
        "- Español (castellano) en TODOS los campos de texto.\n"
        "- Excepción: nombres propios/herramientas y participant_roles.role (enum).\n\n"
        "SALIDA (CRÍTICO):\n"
        "- Devuelve SOLO un objeto JSON válido (sin markdown ni texto extra).\n"
        "- Usa EXACTAMENTE las claves del esquema.\n"
        "- No inventes speakers: usa SOLO las KEYS de participation_percent.\n"
        "- COPIA EXACTO talk_time_seconds, turns, participation_percent desde STATS.\n\n"
        "ANTI-ALUCINACIÓN (CRÍTICO):\n"
        "- No inventes decisiones/tareas.\n"
        "- Si hay DATOS_DETERMINISTAS_JSON con deterministic_*_candidates: "
        "decisions/action_items SOLO pueden salir de esos candidates.\n"
        "- Si no hay evidencia explícita: decisions=[] y action_items=[].\n"
        "- PROHIBIDO devolver action_items placeholder (task vacío). Si no hay tareas: [].\n\n"
        "REGLAS DE CALIDAD:\n"
        "- topics: 4–6; name 2–6 palabras; sin comas/puntos; en español; weight desc; suma ~1.\n"
        "- topics deben ser ESPECÍFICOS del contenido; prohibido usar comodines tipo 'Coordinación del trabajo', 'Seguimiento de avances', 'Próximos pasos' salvo cita textual evidente.\n"
        "- Cada topic debe apoyarse en texto de TOON o PISTAS; evita abstracciones genéricas.\n"
        "- summary (MUY IMPORTANTE): 8–12 frases, sin listas, >=900 caracteres.\n"
        "- summary debe incluir 2 datos cuantitativos de STATS (p.ej. nº participantes y top participación%).\n"
        "- summary debe cubrir: contexto, estado por tema, decisiones, acciones, riesgos y próximos hitos.\n"
        "- summary debe mencionar 1–2 términos de PISTAS si existen (sin inventar).\n"
        "- summary debe citar explícitamente decisiones/acciones detectadas; si no existen, indícalo de forma explícita (sin inventar).\n"
        "- atmosphere.labels SOLO: [\"neutro\",\"positivo\",\"negativo\",\"tenso\",\"calmado\",\"enfocado\",\"fricción leve\"].\n"
        "- collaboration/decisiveness/conflict: 0..100. notes: 2–3 evidencias 't=xx.x: ...' o [].\n"
        "- participant_roles.role ∈ {Driver, Co-driver, Contributor, Observer}.\n"
    )

    det_block = ""
    if det_facts_json:
        det_block = (
            "DATOS_DETERMINISTAS_JSON (fuente de verdad):\n"
            f"{det_facts_json}\n\n"
        )

    user = (
        "Devuelve SOLO el JSON del esquema.\n\n"
        "STATS (FUENTE DE VERDAD; COPIA EXACTO):\n"
        f"talk_time_seconds:{json.dumps(talk_time, ensure_ascii=False, separators=(',',':'))}\n"
        f"turns:{json.dumps(turns, ensure_ascii=False, separators=(',',':'))}\n"
        f"participation_percent:{json.dumps(participation, ensure_ascii=False, separators=(',',':'))}\n\n"
        "PISTAS (si aplican; NO inventes):\n"
        f"terms:{json.dumps(hints.get('terms', []), ensure_ascii=False, separators=(',',':'))}\n"
        f"phrases:{json.dumps(hints.get('phrases', []), ensure_ascii=False, separators=(',',':'))}\n\n"
        "ESQUEMA (misma forma y claves; reemplaza Topic 1..5 por topics reales):\n"
        f"{json.dumps(schema_hint, ensure_ascii=False, separators=(',',':'))}\n\n"
        f"{det_block}"
        "TOON v1 MIN:\n"
        f"{toon}"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_llm_messages_finalize(
    cumulative_stats: Dict[str, Any],
    final_handoff: "HandoffState",
    *,
    det_facts_json: Optional[str] = None,
) -> List[Dict[str, str]]:
    final_state = handoff_for_llm(
        final_handoff,
        max_ledger=28,
        max_bullets_per_topic=3,
        max_salient=16,
        max_summary_chars=900,
        max_opening_chars=650,
        max_open_loops=10,
        max_decisions=14,
        max_actions=14,
    )

    talk_time = cumulative_stats.get("talk_time_seconds", {}) or {}
    turns = cumulative_stats.get("turns", {}) or {}
    participation = cumulative_stats.get("participation_percent", {}) or {}
    schema_hint = _schema_hint_with_real_speakers(talk_time, turns, participation)

    system = (
        "Eres analista de reuniones.\n"
        "Generas el INFORME FINAL desde cumulative_stats + handoff.\n\n"
        "IDIOMA (CRÍTICO): español en TODOS los textos.\n\n"
        "SALIDA (CRÍTICO):\n"
        "- Devuelve SOLO JSON válido con EXACTAMENTE las claves del esquema.\n"
        "- No inventes speakers: SOLO KEYS de participation_percent.\n"
        "- COPIA EXACTO talk_time_seconds/turns/participation_percent desde cumulative_stats.\n\n"
        "ANTI-ALUCINACIÓN:\n"
        "- No inventes decisiones/tareas. Si no hay evidencia explícita en handoff/datos deterministas, deja [].\n"
        "- Si se proporcionan DATOS_DETERMINISTAS_JSON con candidates, priorízalos.\n\n"
        "REGLAS:\n"
        "- topics: 4–6; 2–6 palabras, sin comas/puntos; en español; weight desc; suma ~1.\n"
        "- topics NO genéricos; prioriza nombres específicos presentes en handoff/topic_ledger/datos deterministas.\n"
        "- summary: 8–12 frases, sin listas, >=1000 caracteres.\n"
        "- summary debe ser ejecutivo y trazable: avance por tema, decisiones, tareas, riesgos y dependencias.\n"
        "- summary debe incluir qué se decidió y qué quedó como acción; si no hay evidencia, dilo explícitamente.\n"
        "- atmosphere.labels SOLO [\"neutro\",\"positivo\",\"negativo\",\"tenso\",\"calmado\",\"enfocado\",\"fricción leve\"].\n"
        "- notes: 2–3 líneas cortas o [].\n\n"
        "ROLES DE PARTICIPANTES (FINAL):\n"
        "- participant_roles.role ∈ {Driver, Co-driver, Contributor, Observer}.\n"
        "- Exactamente 1 Driver; 0–1 Co-driver.\n"
        "- Si no hay TOON, asigna Driver por combinación de: top participación + más decisiones/acciones en handoff.\n"
        "- Observer si participation_percent < 8.0 o turns muy bajos.\n"
        "- evidence: 1–2 ítems cortos (pueden citar: 'top participación', 'lidera acciones', 'cierra decisiones').\n"
    )

    det_block = ""
    if det_facts_json:
        det_block = (
            "\nDATOS_DETERMINISTAS_JSON (fuente de verdad):\n"
            f"{det_facts_json}\n"
        )

    user = (
        "cumulative_stats (FUENTE DE VERDAD):\n"
        f"{json.dumps(cumulative_stats, ensure_ascii=False, separators=(',',':'))}\n\n"
        "handoff_final_compacto:\n"
        f"{json.dumps(final_state, ensure_ascii=False, separators=(',',':'))}\n\n"
        "ESQUEMA (misma forma y claves; reemplaza Topic 1..5 por topics reales):\n"
        f"{json.dumps(schema_hint, ensure_ascii=False, separators=(',',':'))}\n"
        f"{det_block}"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _estimate_prompt_tokens_for_chat(messages: List[Dict[str, str]]) -> int:
    # aproximación estable: serializa a prompt estilo chat "ROLE:\n..."
    return estimate_tokens(chat_messages_to_prompt(messages))

def _clamp_generation_tokens(messages: List[Dict[str, str]], requested_max_tokens: int) -> int:
    prompt_toks = _estimate_prompt_tokens_for_chat(messages)
    available = LLAMA_CTX_TOKENS - prompt_toks - LLM_CTX_MARGIN
    if available <= 0:
        # Prompt ya excede el contexto: hay que recortar chunk/handoff
        raise HTTPException(
            status_code=413,
            detail={
                "llama_error": "Prompt excede el contexto del modelo",
                "prompt_tokens_est": prompt_toks,
                "ctx_tokens": LLAMA_CTX_TOKENS,
                "hint": "Reduce tamaño del chunk y/o el handoff enviado al LLM",
            },
        )
    return max(32, min(int(requested_max_tokens), int(available)))

async def call_llama_cpp(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    *,
    response_format: Optional[Dict[str, Any]] = None,
    stop: Optional[List[str]] = None,
    reasoning_format: Optional[str] = None,
    max_retries: int = 2,
) -> str:
    """
    Llamada robusta al servidor OpenAI-compatible.
    - clamp max_tokens para no reventar n_ctx
    - retries con backoff en errores transitorios (429/5xx/timeouts)
    - si response_format=json_object, intenta devolver SIEMPRE un dict JSON recortado
    """
    max_tokens = _clamp_generation_tokens(messages, max_tokens)
    want_json_object = bool(response_format and response_format.get("type") == "json_object")

    timeout = httpx.Timeout(
        connect=min(10.0, LLAMA_TIMEOUT_S),
        read=None,
        write=min(30.0, LLAMA_TIMEOUT_S),
        pool=min(10.0, LLAMA_TIMEOUT_S),
    )

    def _is_transient_status(code: int) -> bool:
        return code in (408, 409, 429, 500, 502, 503, 504)

    async def _sleep_backoff(attempt: int) -> None:
        # 0.35, 0.7, 1.4 ... con jitter leve
        base = 0.35 * (2 ** attempt)
        jitter = 0.08 * (attempt + 1)
        await asyncio.sleep(min(2.5, base + jitter))

    async with LLM_SEM:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if LLAMA_API_MODE != "openai":
                raise HTTPException(status_code=500, detail={"llama_error": "LLAMA_API_MODE no soportado aquí"})

            chat_url = f"{LLAMA_BASE_URL}/v1/chat/completions"
            payload: Dict[str, Any] = {
                "model": LLAMA_MODEL,
                "messages": messages,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "stream": False,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            if stop:
                payload["stop"] = stop
            if reasoning_format is not None:
                payload["reasoning_format"] = reasoning_format

            last_err: Optional[Dict[str, Any]] = None

            for attempt in range(max_retries + 1):
                try:
                    r = await client.post(chat_url, json=payload)

                    if r.status_code < 400:
                        data = r.json()
                        content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
                        if not want_json_object:
                            return content

                        # higiene: intenta devolver SIEMPRE un dict JSON
                        trimmed = _extract_first_json_object(content)
                        try:
                            obj = json.loads(trimmed)
                            if isinstance(obj, dict):
                                return trimmed
                        except Exception:
                            # si el modelo devolvió JSON raro, devolvemos el recorte igualmente
                            return trimmed

                    # error HTTP
                    err = _extract_http_error(r)
                    last_err = err

                    body_txt = ""
                    try:
                        body_txt = json.dumps(err.get("body", ""), ensure_ascii=False)
                    except Exception:
                        body_txt = str(err.get("body", ""))

                    if _is_ctx_error(body_txt):
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "llama_error": "Context/KV cache excedido",
                                "status_code": r.status_code,
                                "max_tokens_sent": max_tokens,
                                "prompt_tokens_est": _estimate_prompt_tokens_for_chat(messages),
                                "ctx_tokens": LLAMA_CTX_TOKENS,
                                "server_body": err.get("body"),
                                "fix": "Reduce prompt: baja chunk size y/o compacta el handoff",
                            },
                        )

                    if _is_transient_status(r.status_code) and attempt < max_retries:
                        await _sleep_backoff(attempt)
                        continue

                    raise HTTPException(
                        status_code=502,
                        detail={
                            "llama_error": "LLM devolvió error",
                            "status_code": r.status_code,
                            "server_body": err.get("body"),
                        },
                    )

                except (httpx.TimeoutException, httpx.TransportError) as e:
                    if attempt < max_retries:
                        last_err = {"transport_error": str(e)}
                        await _sleep_backoff(attempt)
                        continue
                    raise HTTPException(
                        status_code=502,
                        detail={"llama_error": "Error de transporte/timeout con LLM", "detail": str(e), "last": last_err},
                    )

            # no debería llegar
            raise HTTPException(status_code=502, detail={"llama_error": "Fallo inesperado LLM", "last": last_err})


def _schema_hint_with_real_speakers(
    talk_time: Dict[str, float],
    turns: Dict[str, int],
    participation: Dict[str, float],
) -> Dict[str, Any]:
    speakers = list(participation.keys())

    return {
        "participation_percent": {k: float(participation.get(k, 0.0)) for k in speakers},
        "talk_time_seconds": {k: float(talk_time.get(k, 0.0)) for k in speakers},
        "turns": {k: int(turns.get(k, 0)) for k in speakers},
        "collaboration": {"score_0_100": 0.0, "notes": []},
        "atmosphere": {"valence": 0.0, "labels": [], "notes": []},
        "decisiveness": {"score_0_100": 0.0, "notes": []},
        "conflict_level_0_100": 0.0,
        "topics": [
            {"name": "Topic 1", "weight": 0.22},
            {"name": "Topic 2", "weight": 0.20},
            {"name": "Topic 3", "weight": 0.20},
            {"name": "Topic 4", "weight": 0.20},
            {"name": "Topic 5", "weight": 0.18},
        ],
        "decisions": [],
        "action_items": [],  # ✅ sin placeholder
        "main_responsible": None,
        "participant_roles": [
            {
                "name": (speakers[0] if speakers else ""),
                "role": "Contributor",
                "responsibility_0_100": 0.0,
                "evidence": [],
            }
        ],
        "debates": [],
        "plan": [],
        "summary": "",
        "quality_flags": [],
    }


def resolve_transcript_from_request(req: AnalyzeRequest) -> Dict[str, Any]:
    """
    Normaliza el input para que el pipeline opere siempre sobre:
      {"segments":[{start,end,speaker,text},...], "meta": {...}}

    Soporta:
      A) req.transcript (antiguo)
      B) req.segments (ASR local)
      C) req.raw_transcript_json (Teams/meeting word-level) -> TURNOS (speaker-change)
    """
    # A) formato antiguo
    if req.transcript and isinstance(req.transcript, dict):
        return req.transcript

    # B) ASR local
    if req.segments is not None:
        t: Dict[str, Any] = {}
        if req.meta is not None:
            t["meta"] = req.meta
        t["segments"] = req.segments
        if req.diarization is not None:
            t["diarization"] = req.diarization
        return t

    # C) Teams/meeting payload (word-level) -> TURNOS (solo cuando cambia speaker)
    if req.raw_transcript_json is not None:
        segs = segments_from_raw_transcript_json_turns(
            req.raw_transcript_json,
            sort_by_time=True,
        )

        return {
            "meta": {
                "source": "raw_transcript_json",
                "meeting_id": req.meeting_id,
                "transcription_id": req.transcription_id,
                "language": req.language,
                "is_final": req.is_final,
            },
            "segments": segs,
        }

    return {"segments": []}

# -----------------------------
# Chunking utilities
# -----------------------------
def _utterance_to_line(u: "NormalizedUtterance", include_timestamps: bool) -> str:
    if include_timestamps:
        return f"{u.speaker}|{u.start:.2f}|{u.text}"
    return f"{u.speaker}|{u.text}"

def split_utterances_into_chunks(
    utterances: List["NormalizedUtterance"],
    fmt: Literal["lines", "json"],
    include_timestamps: bool,
    context_window_tokens: int,
    reserved_tokens: int,
    max_chunks: int = 999,
) -> List[List["NormalizedUtterance"]]:
    """
    Divide en bloques para que el TOON + prompt quepa en ~context_window_tokens.
    Presupuesto: (context_window_tokens - reserved_tokens) para el TOON (aprox).

    Versión robusta: usa estimación de tokens, no longitud en chars.
    """
    if context_window_tokens <= 0:
        raise ValueError("context_window_tokens debe ser > 0")
    if reserved_tokens < 0:
        reserved_tokens = 0

    # Mantén un mínimo razonable para que no salgan chunks microscópicos
    budget_tokens = max(900, int(context_window_tokens) - int(reserved_tokens))

    chunks: List[List["NormalizedUtterance"]] = []
    cur: List["NormalizedUtterance"] = []
    cur_toks = 0

    for u in utterances:
        # estimar tokens incrementales con la misma serialización TOON
        toon_piece = build_toon_v1([u], fmt=fmt, include_timestamps=include_timestamps)
        piece_toks = estimate_tokens_rough(toon_piece) + 6  # margen separadores

        # si un turno aislado ya excede el presupuesto, lo forzamos como chunk propio
        if not cur and piece_toks > budget_tokens:
            chunks.append([u])
            if len(chunks) >= max_chunks:
                raise ValueError(
                    f"El transcript requiere >{max_chunks} chunks. "
                    f"Un turno excede el budget_tokens={budget_tokens}. "
                    f"Reduce max_chars_per_utterance o sube context_window_tokens."
                )
            continue

        if cur and (cur_toks + piece_toks > budget_tokens):
            chunks.append(cur)
            if len(chunks) >= max_chunks:
                raise ValueError(
                    f"El transcript requiere >{max_chunks} chunks con el presupuesto actual. "
                    f"Sube max_chunks o el context_window."
                )
            cur = [u]
            cur_toks = piece_toks
        else:
            cur.append(u)
            cur_toks += piece_toks

    if cur:
        chunks.append(cur)

    return chunks


def init_handoff_from_nothing() -> HandoffState:
    return HandoffState(
        chunk_index=0,
        running_summary="",
        topics=[],
        decisions=[],
        action_items=[],
        collaboration=None,
        atmosphere=None,
        decisiveness=None,
        conflict_level_0_100=None,
        quality_flags=[],
    )


def handoff_for_llm(
    h: HandoffState,
    *,
    max_ledger: int = 18,
    max_bullets_per_topic: int = 2,
    max_salient: int = 12,
    max_summary_chars: int = 650,
    max_opening_chars: int = 450,
    max_open_loops: int = 8,
    max_decisions: int = 10,
    max_actions: int = 10,
) -> Dict[str, Any]:
    """
    Vista compacta del handoff para el LLM.

    Mejora crítica:
      - Clave consistente con prompts de chunk: usa "topic_ledger" (no "ledger").
      - Mantiene "salient_mentions" y otras claves que el prompt espera.
    """

    def ai_min(ai: Any) -> Dict[str, Any]:
        return {
            "owner": getattr(ai, "owner", None),
            "task": getattr(ai, "task", None),
            "due_date": getattr(ai, "due_date", None),
        }

    topic_ledger_min = []
    for t in (getattr(h, "topic_ledger", None) or [])[:max_ledger]:
        bullets = (getattr(t, "bullets", []) or [])[:max_bullets_per_topic]
        evidence = getattr(t, "evidence", []) or []
        ev_min = []
        for ev in evidence[:2]:
            ev_min.append({
                "t": float(getattr(ev, "t", 0.0) or 0.0),
                "speaker": getattr(ev, "speaker", None),
                "snippet": clip_text(getattr(ev, "snippet", "") or "", 160),
            })

        topic_ledger_min.append({
            "name": getattr(t, "name", ""),
            "hits": int(getattr(t, "hits", 1) or 1),
            "salience_0_1": float(getattr(t, "salience_0_1", 0.5) or 0.5),
            "first_seen_sec": getattr(t, "first_seen_sec", None),
            "last_seen_sec": getattr(t, "last_seen_sec", None),
            "bullets": [clip_text(b, 140) for b in bullets],
            "evidence": ev_min,
        })

    salient_min = []
    for m in (getattr(h, "salient_mentions", None) or [])[-max_salient:]:
        salient_min.append({
            "t": float(getattr(m, "t", 0.0) or 0.0),
            "speaker": getattr(m, "speaker", None),
            "kind": getattr(m, "kind", "tema"),
            "text": clip_text(getattr(m, "text", "") or "", 160),
        })

    return {
        "chunk_index": getattr(h, "chunk_index", 0),
        "chunks_total": getattr(h, "chunks_total", None),
        "processed_until_sec": getattr(h, "processed_until_sec", None),

        "opening_context": clip_text(getattr(h, "opening_context", "") or "", max_opening_chars),
        "running_summary": clip_text(getattr(h, "running_summary", "") or "", max_summary_chars),

        "topics": (getattr(h, "topics", None) or [])[:12],
        "decisions": (getattr(h, "decisions", None) or [])[-max_decisions:],
        "action_items": [ai_min(x) for x in (getattr(h, "action_items", None) or [])[-max_actions:]],

        "open_loops": (getattr(h, "open_loops", None) or [])[:max_open_loops],

        # ✅ clave esperada por build_llm_messages_chunk schema
        "topic_ledger": topic_ledger_min,
        "salient_mentions": salient_min,

        "quality_flags": (getattr(h, "quality_flags", None) or [])[:8],
    }


# -----------------------------
# Prompt (chunk-aware): TOON + handoff -> {insights, handoff}
# -----------------------------
def build_llm_messages_chunk(
    toon_chunk: str,
    talk_time: Dict[str, float],
    turns: Dict[str, int],
    participation: Dict[str, float],
    prev_handoff: "HandoffState",
    chunk_index: int,
    chunks_total: int,
    *,
    det_facts_json: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Versión única y coherente (sin placeholders).
    Nota: colaboración/decisión/conflicto se pueden devolver como 0/[] si no hay evidencia;
    luego se recalculan determinísticamente en postproceso.
    """
    hints = _salient_hints_from_toon(toon_chunk)
    real = _schema_hint_with_real_speakers(talk_time, turns, participation)

    system = (
        "Eres analista de reuniones por BLOQUES secuenciales.\n\n"
        "IDIOMA (CRÍTICO): español en todos los textos.\n"
        "SALIDA (CRÍTICO): SOLO JSON válido, sin texto extra.\n"
        "Top-level EXACTO: {\"insights\":{...},\"handoff\":{...}}.\n"
        "No inventes speakers: SOLO KEYS de participation_percent.\n"
        "COPIA EXACTO talk_time_seconds/turns/participation_percent dentro de insights desde STATS.\n\n"
        "ANTI-ALUCINACIÓN:\n"
        "- Si hay DATOS_DETERMINISTAS_JSON con candidates, decisions/action_items SOLO desde ahí.\n"
        "- Si no hay evidencia explícita, decisions=[] y action_items=[].\n"
        "- PROHIBIDO devolver placeholders (task vacío).\n\n"
        "SEÑALES (IMPORTANTE):\n"
        "- collaboration/decisiveness/conflict se recalculan en backend.\n"
        "- Si no tienes evidencia con timestamps, deja score_0_100=0 y notes=[].\n"
        "- atmosphere: si no hay evidencia clara, deja labels=[] y notes=[] (sin 'neutro' por defecto).\n\n"
        "CONTINUIDAD HANDOFF (CRÍTICO PARA TAREAS):\n"
        "- Trata handoff_anterior_compacto.action_items como backlog vivo.\n"
        "- Si una tarea ya existe, ACTUALIZA owner/due_date/confidence en lugar de crear duplicado.\n"
        "- Si en este bloque aparece owner o fecha para tarea previa, rellénalos y súbelos al handoff.\n"
        "- No elimines tareas previas salvo contradicción explícita.\n\n"
        "REGLAS:\n"
        "- atmosphere.labels SOLO [\"neutro\",\"positivo\",\"negativo\",\"tenso\",\"calmado\",\"enfocado\",\"fricción leve\"].\n"
        "- notes: usa 't=xx.x: ...' o [] si no hay.\n"
        "- owner/decided_by: speaker exacto o null.\n"
        "- topics: 4–6; name 2–6 palabras, sin comas/puntos; en español; weight desc, suma ~1.\n"
        "- topics deben salir de contenido específico del bloque/handoff previo; evita etiquetas comodín tipo 'Coordinación del trabajo'.\n"
        "- summary (chunk): 6–10 frases, sin listas, >=650 caracteres.\n"
        "- summary (chunk) debe recoger decisiones/acciones explícitas del bloque o indicar claramente que no hubo evidencia.\n"
        "- handoff conciso.\n"
    )

    schema_hint = {
        "insights": {
            "participation_percent": real["participation_percent"],
            "talk_time_seconds": real["talk_time_seconds"],
            "turns": real["turns"],
            "collaboration": {"score_0_100": 0.0, "notes": []},
            "atmosphere": {"valence": 0.0, "labels": [], "notes": []},
            "decisiveness": {"score_0_100": 0.0, "notes": []},
            "conflict_level_0_100": 0.0,
            "topics": [{"name": "", "weight": 0.0}],
            "decisions": [],
            "action_items": [],
            "main_responsible": None,
            "participant_roles": [{"name": "", "role": "Contributor", "responsibility_0_100": 0.0, "evidence": []}],
            "debates": [],
            "plan": [],
            "summary": "",
            "quality_flags": [],
        },
        "handoff": {
            "chunk_index": chunk_index,
            "chunks_total": chunks_total,
            "processed_until_sec": 0.0,
            "running_summary": "",
            "opening_context": "",
            "salient_mentions": [{"t": 0.0, "speaker": None, "text": "", "kind": "tema"}],
            "topic_ledger": [{
                "name": "",
                "weight": 0.0,
                "first_seen_sec": None,
                "last_seen_sec": None,
                "bullets": [],
                "evidence": [{"t": 0.0, "speaker": None, "snippet": ""}],
            }],
            "open_loops": [],
            "topics": [{"name": "", "weight": 0.0}],
            "decisions": [],
            "action_items": [],
            "collaboration": {"score_0_100": 0.0, "notes": []},
            "atmosphere": {"valence": 0.0, "labels": [], "notes": []},
            "decisiveness": {"score_0_100": 0.0, "notes": []},
            "conflict_level_0_100": 0.0,
            "quality_flags": [],
        },
    }

    det_block = ""
    if det_facts_json:
        det_block = (
            "\n\nDATOS_DETERMINISTAS_JSON (fuente de verdad):\n"
            f"{det_facts_json}\n"
        )

    compress_levels: List[Dict[str, int]] = [
        dict(max_ledger=18, max_salient=12, max_summary_chars=650, max_opening_chars=450, max_open_loops=8),
        dict(max_ledger=12, max_salient=10, max_summary_chars=520, max_opening_chars=360, max_open_loops=6),
        dict(max_ledger=8,  max_salient=8,  max_summary_chars=420, max_opening_chars=280, max_open_loops=4),
        dict(max_ledger=4,  max_salient=5,  max_summary_chars=320, max_opening_chars=220, max_open_loops=2),
        dict(max_ledger=0,  max_salient=0,  max_summary_chars=260, max_opening_chars=180, max_open_loops=0),
    ]

    max_prompt_tokens = max(256, LLAMA_CTX_TOKENS - LLM_CTX_MARGIN - 384)
    final_messages: Optional[List[Dict[str, str]]] = None

    for lvl in compress_levels:
        prev_state = handoff_for_llm(prev_handoff, **lvl)

        user = (
            f"Bloque {chunk_index+1}/{chunks_total}.\n\n"
            "STATS (COPIA EXACTO en insights):\n"
            f"talk_time_seconds:{json.dumps(talk_time, ensure_ascii=False, separators=(',',':'))}\n"
            f"turns:{json.dumps(turns, ensure_ascii=False, separators=(',',':'))}\n"
            f"participation_percent:{json.dumps(participation, ensure_ascii=False, separators=(',',':'))}\n\n"
            "handoff_anterior_compacto:\n"
            f"{json.dumps(prev_state, ensure_ascii=False, separators=(',',':'))}\n\n"
            "PISTAS:\n"
            f"terms:{json.dumps(hints.get('terms', []), ensure_ascii=False, separators=(',',':'))}\n"
            f"phrases:{json.dumps(hints.get('phrases', []), ensure_ascii=False, separators=(',',':'))}\n\n"
            "ESQUEMA JSON requerido:\n"
            f"{json.dumps(schema_hint, ensure_ascii=False, separators=(',',':'))}\n"
            f"{det_block}\n"
            "TOON v1 MIN (este bloque):\n"
            f"{toon_chunk}"
        )

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if _estimate_prompt_tokens_for_chat(msgs) <= max_prompt_tokens:
            final_messages = msgs
            break
        final_messages = msgs

    return final_messages or [{"role": "system", "content": system}, {"role": "user", "content": ""}]


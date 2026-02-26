from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, List, Dict, Any
import os

#Constantes de config
DEFAULT_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "900"))
DEFAULT_MERGE_GAP_MS = int(os.getenv("MERGE_GAP_MS", "800"))

#Utils para models
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v)


#Memoria jerárquica para N handoff
class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    t: float
    speaker: Optional[str] = None
    snippet: str

class SalientMention(BaseModel):
    model_config = ConfigDict(extra="ignore")
    t: float
    speaker: Optional[str] = None
    text: str
    kind: Optional[Literal["tema", "decisión", "tarea", "pregunta", "riesgo", "dato"]] = "tema"

class TopicLedgerItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    weight: float = 0.0
    first_seen_sec: Optional[float] = None
    last_seen_sec: Optional[float] = None
    bullets: List[str] = Field(default_factory=list)       # 0..3, ultra-cortos
    evidence: List[EvidenceItem] = Field(default_factory=list)  # 0..2

class ChunkingOptions(BaseModel):
    # defaults desde .env (para que no tengas que mandar options)
    enabled: bool = Field(default_factory=lambda: _env_bool("INSIGHTS_CHUNKING_ENABLED", False))

    context_window_tokens: int = Field(default_factory=lambda: _env_int("INSIGHTS_CONTEXT_WINDOW_TOKENS", 10_000))
    reserve_output_tokens: int = Field(default_factory=lambda: _env_int("INSIGHTS_RESERVE_OUTPUT_TOKENS", 1_200))
    prompt_overhead_tokens: int = Field(default_factory=lambda: _env_int("INSIGHTS_PROMPT_OVERHEAD_TOKENS", 1_600))
    safety_margin_tokens: int = Field(default_factory=lambda: _env_int("INSIGHTS_SAFETY_MARGIN_TOKENS", 400))

    max_chunks: int = Field(default_factory=lambda: _env_int("INSIGHTS_MAX_CHUNKS", 999))
    return_llm_raw_per_chunk: bool = Field(default_factory=lambda: _env_bool("INSIGHTS_RETURN_LLM_RAW_PER_CHUNK", True))

class RenderPNGRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    insights: Dict[str, Any]
    title: Optional[str] = None

class ChartOptions(BaseModel):
    enabled: bool = Field(default_factory=lambda: _env_bool("INSIGHTS_CHARTS_ENABLED", True))
    mode: Literal["none", "url", "base64", "both"] = Field(default_factory=lambda: _env_str("INSIGHTS_CHARTS_MODE", "url"))  # type: ignore
    title: Optional[str] = Field(default_factory=lambda: os.getenv("INSIGHTS_CHARTS_TITLE"))

class AnalyzeOptions(BaseModel):
    toon_format: Literal["lines", "json"] = Field(default_factory=lambda: _env_str("INSIGHTS_TOON_FORMAT", "lines"))  # type: ignore
    include_timestamps: bool = Field(default_factory=lambda: _env_bool("INSIGHTS_INCLUDE_TIMESTAMPS", True))
    merge_adjacent_same_speaker: bool = Field(default_factory=lambda: _env_bool("INSIGHTS_MERGE_ADJACENT", True))
    merge_gap_ms: int = Field(default_factory=lambda: _env_int("INSIGHTS_MERGE_GAP_MS", DEFAULT_MERGE_GAP_MS))
    max_chars_per_utterance: int = Field(default_factory=lambda: _env_int("INSIGHTS_MAX_CHARS_PER_UTTERANCE", 800))

    # temperatura / tokens también desde env (siguen funcionando tus LLAMA_* si los tienes)
    temperature: float = Field(default_factory=lambda: _env_float("INSIGHTS_TEMPERATURE", DEFAULT_TEMPERATURE))
    max_tokens: int = Field(default_factory=lambda: _env_int("INSIGHTS_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    chunking: ChunkingOptions = Field(default_factory=ChunkingOptions)
    charts: ChartOptions = Field(default_factory=ChartOptions)

# -----------------------------
# Pydantic models (normalized)
# -----------------------------
class NormalizedUtterance(BaseModel):
    speaker: str
    start: float
    end: float
    text: str

# -----------------------------
# Output schema: final insights
# -----------------------------
class ActionItem(BaseModel):
    owner: Optional[str] = None
    task: str
    due_date: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)

class Topic(BaseModel):
    name: str
    weight: float = Field(ge=0.0, le=1.0, default=0.3)

class ParticipantRole(BaseModel):
    """
    Papel del participante en la reunión (conservador, basado en evidencia).
    name debe coincidir con los speakers de participation_percent.
    """
    name: str
    role: str = Field(default="Contributor")  # Driver | Co-driver | Contributor | Observer
    responsibility_0_100: float = Field(ge=0.0, le=100.0, default=50.0)
    evidence: List[str] = Field(default_factory=list)

class DebateItem(BaseModel):
    """
    Debate/tema en discusión. Si no hay resolución clara, resolution puede ser None.
    """
    topic: str
    question: Optional[str] = None
    resolution: Optional[str] = None
    decided_by: Optional[str] = None
    notes: List[str] = Field(default_factory=list)

class PlanStep(BaseModel):
    """
    Planificación/próximos pasos. Puede solaparse con action_items,
    pero suele ser más narrativo (orden / dependencias).
    """
    step: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)

class CollaborationBlock(BaseModel):
    score_0_100: float = Field(ge=0.0, le=100.0)
    notes: List[str] = Field(default_factory=list)

class AtmosphereBlock(BaseModel):
    valence: float = Field(ge=-1.0, le=1.0)
    labels: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

class DecisivenessBlock(BaseModel):
    score_0_100: float = Field(ge=0.0, le=100.0)
    notes: List[str] = Field(default_factory=list)

class MeetingInsights(BaseModel):
    model_config = ConfigDict(extra="ignore")

    participation_percent: Dict[str, float]
    talk_time_seconds: Dict[str, float]
    turns: Dict[str, int]

    collaboration: CollaborationBlock
    atmosphere: AtmosphereBlock
    decisiveness: DecisivenessBlock
    conflict_level_0_100: float = Field(ge=0.0, le=100.0)

    topics: List[Topic] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)

    # NUEVO: roles, debates y plan
    main_responsible: Optional[str] = None
    participant_roles: List[ParticipantRole] = Field(default_factory=list)
    debates: List[DebateItem] = Field(default_factory=list)
    plan: List[PlanStep] = Field(default_factory=list)

    summary: str
    quality_flags: List[str] = Field(default_factory=list)

class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # --- Compatibilidad antigua
    transcript: Optional[Dict[str, Any]] = None  # formato {transcript:{segments:...}}

    # --- Compatibilidad con tu payload actual (ASR local)
    meta: Optional[Dict[str, Any]] = None
    segments: Optional[List[Dict[str, Any]]] = None
    diarization: Optional[List[Dict[str, Any]]] = None

    # --- NUEVO payload (Teams/meeting)
    meeting_id: Optional[str] = None
    has_transcription: Optional[bool] = None
    transcription_id: Optional[str] = None
    language: Optional[str] = None
    is_final: Optional[bool] = None

    # OJO: viene como LISTA (según tu ejemplo)
    raw_transcript_json: Optional[List[Dict[str, Any]]] = None

    # options ahora es opcional para el cliente (pero seguirá pudiendo enviarse si quiere)
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)
    previous_handoff: Optional["HandoffState"] = None

# -----------------------------
# Handoff (estado incremental)
# -----------------------------
class HandoffState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_index: int = 0
    chunks_total: Optional[int] = None
    processed_until_sec: Optional[float] = None

    running_summary: str = ""

    # OJO: este campo ahora ES DERIVADO, no autoridad del LLM
    topics: List["Topic"] = Field(default_factory=list)

    decisions: List[str] = Field(default_factory=list)
    action_items: List["ActionItem"] = Field(default_factory=list)

    collaboration: Optional["CollaborationBlock"] = None
    atmosphere: Optional["AtmosphereBlock"] = None
    decisiveness: Optional["DecisivenessBlock"] = None
    conflict_level_0_100: Optional[float] = None

    quality_flags: List[str] = Field(default_factory=list)

    opening_context: str = ""
    salient_mentions: List[SalientMention] = Field(default_factory=list)
    topic_ledger: List[TopicLedgerItem] = Field(default_factory=list)
    open_loops: List[str] = Field(default_factory=list)

class ChunkResult(BaseModel):
    chunk_index: int
    toon: str
    llm_raw: Optional[str] = None
    parse_mode: Literal["json", "regex_fallback"]
    insights: MeetingInsights
    handoff: HandoffState

class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # ahora toon será el brief/resumen
    toon: str

    # NUEVO: transcript TOON original (para debug/compat)
    toon_transcript: Optional[str] = None

    normalized: List[NormalizedUtterance]
    insights: MeetingInsights
    llm_raw: Optional[str] = None
    parse_mode: Literal["json", "regex_fallback"]

    chunks_total: int
    final_handoff: HandoffState
    chunks: List[ChunkResult] = Field(default_factory=list)

    charts_id: Optional[str] = None
    charts_png_url: Optional[str] = None
    charts_png_base64: Optional[str] = None
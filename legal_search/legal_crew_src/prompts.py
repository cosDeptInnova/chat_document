# legal_crew_src/prompts.py

from typing import List, Dict, Any, Optional
import textwrap
from datetime import datetime, timezone


def build_legal_planner_prompt(*, history_str: str, user_prompt: str, extra_context: str = "") -> str:
    today = datetime.now(timezone.utc).date().isoformat()

    context_block = f"\nContexto documental aportado por usuario:\n---\n{extra_context}\n---\n" if extra_context else ""

    return textwrap.dedent(f"""
    Eres un planificador experto en investigación legal (juristas/abogados internos) para
    un departamento jurídico corporativo de multinacional con actividad inmobiliaria.

    Fecha de referencia (para vigencia/recencia): {today}

    Objetivo:
    - Generar 1 a 4 consultas de búsqueda "atómicas" (auto-contenidas) para localizar
      normativa/jurisprudencia/criterios oficiales y análisis fiable, priorizando fuentes primarias.

    Reglas (estrictas):
    - Determina jurisdicción probable: ES/UE y/o la(s) indicada(s) por el usuario.
    - Si el usuario pide "vigente/actual/último/reciente", incluye el año o rango (ej. 2024–2026)
      y/o términos "vigente", "última modificación", "consolidada".
    - Prioriza fuentes de autoridad (orden):
      1) Diarios oficiales (BOE/DOUE y boletines autonómicos si procede)
      2) EUR-Lex / instituciones UE / tribunales (Curia, etc.)
      3) Reguladores/administraciones
      4) Solo si falta lo anterior: doctrinal/firma/medio (y márcalo como secundario)
    - Evita consultas vagas. Evita referencias "esto/lo anterior".

    Historial reciente:
    ---
    {history_str}
    ---

    Petición actual:
    {user_prompt}
    {context_block}

    Salida estricta:
    - Devuelve SIEMPRE que puedas JSON válido con esta forma EXACTA (keys estables):
      {{
        "jurisdictions": ["ES","EU"],
        "topics": ["..."],
        "queries": ["q1","q2","q3","q4"],
        "source_priorities": ["boe.es","eur-lex.europa.eu","curia.europa.eu"],
        "assumptions": ["si faltan hechos, qué asumes"],
        "constraints": ["límites/ambigüedades detectadas"]
      }}
    - Si no puedes devolver JSON, devuelve 1 a 4 consultas, una por línea,
      sin numeración, sin viñetas, sin comentarios.
    """).strip()


def _sources_text(
    sources: List[Dict[str, Any]],
    *,
    include_content: bool = False,
    max_content_chars: int = 900
) -> str:
    if not sources:
        return "(sin fuentes aún)"

    effective_max = int(max_content_chars or 900)
    n = len(sources)
    if include_content:
        if n >= 20:
            effective_max = max(250, int(effective_max * 0.4))
        elif n >= 12:
            effective_max = max(300, int(effective_max * 0.6))

    chunks: List[str] = []
    for i, s in enumerate(sources, start=1):
        if not isinstance(s, dict):
            continue

        url = (s.get("url") or "").strip()
        title = (s.get("title") or url or "Fuente sin título").strip()
        snippet = (s.get("snippet") if isinstance(s.get("snippet"), str) else "").strip()
        short_snippet = (snippet[:180] + "…") if len(snippet) > 180 else (snippet or "(sin snippet)")

        content_block = ""
        if include_content:
            content = (s.get("content") if isinstance(s.get("content"), str) else "").strip()
            if content:
                if len(content) > effective_max:
                    content = content[:effective_max] + "…"
                content_block = f"\n    Extracto:\n    {content}"

        # info adicional útil (sin forzar)
        meta_bits = []
        src_from = (s.get("from") or "").strip()
        if src_from:
            meta_bits.append(f"from={src_from}")
        q = (s.get("query") or "").strip()
        if q:
            meta_bits.append("query=" + (q[:80] + "…" if len(q) > 80 else q))

        meta_line = f"\n    Meta: {', '.join(meta_bits)}" if meta_bits else ""

        chunks.append(f"[{i}] {title} — {url}\n    {short_snippet}{meta_line}{content_block}")

    return "\n".join(chunks) if chunks else "(sin fuentes aún)"


def build_legal_analyst_prompt(
    *,
    user_prompt: str,
    sources: List[Dict[str, Any]],
    accumulated_sources: List[Dict[str, Any]],
    used_queries: Optional[List[str]] = None,
) -> str:
    used_queries = used_queries or []
    queries_block = "\n".join(f"- {q}" for q in used_queries) if used_queries else "(no hay consultas previas)"

    return (
        "Eres un analista de evidencia legal (estándar 'legal-grade').\n\n"
        "Tarea:\n"
        "- Decide si hay evidencia suficiente para redactar un memo útil y prudente.\n"
        "- Si falta evidencia clave (fuente primaria, vigencia, jurisdicción, órgano, fecha), propone nuevas consultas.\n"
        "- NO repitas consultas ya usadas.\n\n"
        "Petición original:\n"
        f"{user_prompt}\n\n"
        "Consultas ya usadas (NO repetir):\n"
        f"{queries_block}\n\n"
        "Fuentes recién obtenidas:\n"
        + _sources_text(sources)
        + "\n\n"
        "Fuentes acumuladas:\n"
        + _sources_text(accumulated_sources)
        + "\n\n"
        "Criterios:\n"
        "- Preferencia fuerte por fuentes primarias/oficiales.\n"
        "- Si solo hay fuentes secundarias (blogs/medios), probablemente NO basta.\n"
        "- Si hay discrepancias entre fuentes, pide evidencia adicional o indica incertidumbre.\n"
        "- Si ya se puede redactar un memo prudente con citas, decide BASTA.\n\n"
        "Salida estricta:\n"
        "- Devuelve preferentemente JSON válido:\n"
        '  {"decision":"BASTA","rationale":"..."}\n'
        '  {"queries":["consulta 1","consulta 2"],"rationale":"..."}\n'
        "- Si no puedes devolver JSON:\n"
        "  - Si necesitas más búsqueda: SOLO 1 a 3 consultas (una por línea).\n"
        "  - Si ya basta: exactamente BASTA"
    )


def build_legal_writer_prompt(*, user_prompt: str, sources: List[Dict[str, Any]], extra_context: str = "") -> str:
    sources_block = _sources_text(sources, include_content=True, max_content_chars=900)
    context_block = f"\nContexto documental adicional (subido por usuario):\n{extra_context}\n" if extra_context else ""

    return textwrap.dedent(f"""
    Eres el redactor final de un memo legal interno (departamento jurídico corporativo + inmobiliario).

    Reglas estrictas:
    - Usa SOLO la información contenida en los extractos de las fuentes proporcionadas.
    - No añadas conocimiento externo.
    - Si las fuentes no cubren un punto importante, decláralo y propone próximos pasos de investigación.
    - Mantén el tono profesional (abogados internos), sin relleno.

    Petición del usuario:
    {user_prompt}
    {context_block}

    Fuentes disponibles (numeradas):
    {sources_block}

    Formato requerido (memo utilizable):
    1) Resumen ejecutivo (3-6 bullets)
    2) Cuestiones jurídicas / issues (lista)
    3) Hechos asumidos y lagunas (si aplica)
    4) Evidencia encontrada (normativa/jurisprudencia/criterios) con citas [n]
    5) Análisis y conclusión provisional (siempre con soporte)
    6) Riesgos / puntos de atención (probabilidad/impacto cualitativo)
    7) Próximos pasos (qué buscar/confirmar)
    8) Fuentes:
       [1] título — url
       [2] ...

    Reglas de citación:
    - Añade [n] al final de las frases relevantes.
    - No inventes citas. No cites una fuente para algo que no está en su extracto.

    Devuelve ÚNICAMENTE el texto final.
    """).strip()


def build_legal_verifier_prompt(*, user_prompt: str, answer: str, sources: List[Dict[str, Any]]) -> str:
    sources_block = _sources_text(sources, include_content=True, max_content_chars=700)

    return textwrap.dedent(f"""
    Eres un revisor de precisión (QA legal) para un memo basado en fuentes web.

    Objetivo:
    - Verificar que cada afirmación importante está respaldada por extractos de fuentes.
    - Reducir lenguaje categórico si el soporte es insuficiente (evitar overclaiming).
    - Mantener citas [n] coherentes y correctas.
    - Si falta evidencia (jurisdicción/vigencia/fuente primaria), reflejarlo explícitamente y reforzar "Próximos pasos".

    Petición original:
    {user_prompt}

    Memo candidato:
    {answer}

    Fuentes para verificación (extractos):
    {sources_block}

    Reglas:
    - No inventes hechos, fechas, artículos, sentencias o conclusiones no soportadas.
    - Si una frase no está soportada por un extracto, suavízala o muévela a "Limitaciones".
    - Si el memo hace recomendaciones, deben estar condicionadas a confirmación si falta evidencia primaria.

    Instrucciones:
    - Devuelve ÚNICAMENTE la versión final corregida.
    - Conserva la estructura del memo.
    - Si detectas lagunas claras, añade una sección breve dentro del memo:
      "Limitaciones" (si no existe) y refuerza "Próximos pasos".
    """).strip()


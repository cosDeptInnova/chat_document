from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List


def build_query_rewriter_prompt(*, history_str: str, user_query: str, user_id: str, now_iso: str | None = None) -> str:
    now_iso = now_iso or datetime.now(timezone.utc).isoformat()
    return textwrap.dedent(
        f"""
        Eres un especialista en normalización de consultas para búsqueda empresarial en reuniones.

        Fecha-hora de referencia (UTC): {now_iso}
        Usuario autenticado: {user_id}

        Tu tarea:
        1) Reescribir la consulta en una versión clara para embeddings semánticos.
        2) Inferir filtros candidatos para Qdrant (payload metadata), con énfasis en fechas,
           reunión específica, participantes, temas y decisiones.
        3) NUNCA inventes datos no soportados por la consulta/historial.

        Historial reciente:
        ---
        {history_str}
        ---

        Pregunta del usuario:
        {user_query}

        Reglas de seguridad y robustez:
        - Siempre limita por user_id={user_id} para multiusuario.
        - Si la consulta incluye "hace X días/semanas/meses" o fechas explícitas, devuelve rango temporal ISO.
        - Si no hay señal temporal, deja `datetime_range` como null.
        - Si hay ambigüedad, expresa supuestos en `assumptions` sin bloquear salida.
        - Devuelve JSON válido con este esquema exacto:
        {{
          "normalized_query": "...",
          "qdrant_filter": {{
            "must": [
              {{"key": "user_id", "match": "{user_id}"}},
              {{"key": "meeting_id", "match": "..."}}
            ],
            "must_any": [
              {{"key": "topics", "match": "..."}},
              {{"key": "decisions", "match": "..."}}
            ],
            "datetime_range": {{"from": "ISO-8601", "to": "ISO-8601"}} | null
          }},
          "retrieval_hints": {{
            "limit": 8,
            "min_similarity": 0.15,
            "focus": ["...", "..."]
          }},
          "assumptions": ["..."],
          "risk_flags": ["..."],
          "audit_notes": ["..."]
        }}
        """
    ).strip()


def _serialize_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "(sin resultados)"

    rows = []
    for idx, item in enumerate(results, start=1):
        meta = item.get("metadata") or {}
        graph = item.get("graph_context") or {}
        rows.append(
            f"""
            [{idx}] similarity={item.get('similarity_score')}
            text={item.get('text_content', '')[:900]}
            metadata={meta}
            graph_context={graph}
            """.strip()
        )
    return "\n\n".join(rows)


def build_result_evaluator_prompt(
    *,
    user_query: str,
    normalized_query: str,
    filter_summary: Dict[str, Any],
    combined_results: List[Dict[str, Any]],
) -> str:
    context = _serialize_context(combined_results)
    return textwrap.dedent(
        f"""
        Eres un evaluador experto de resultados híbridos RAG + GraphRAG en reuniones corporativas.

        Pregunta original del usuario:
        {user_query}

        Consulta normalizada usada en recuperación:
        {normalized_query}

        Filtros usados para prefiltrado en Qdrant:
        {filter_summary}

        Resultados combinados RAG + GraphRAG:
        ---
        {context}
        ---

        Objetivo:
        - Evaluar calidad de evidencia y consistencia.
        - Entregar respuesta clara y útil al usuario final.
        - Explicar qué proviene de RAG semántico y qué de GraphRAG relacional.
        - Declarar límites/lagunas sin inventar.

        Salida JSON estricta:
        {{
          "final_answer": "respuesta final en español",
          "evidence_summary": [
            {{"source": "RAG|GraphRAG", "detail": "...", "confidence": 0.0}}
          ],
          "quality_assessment": {{
            "coverage": "alta|media|baja",
            "consistency": "alta|media|baja",
            "temporal_alignment": "alta|media|baja"
          }},
          "follow_up_questions": ["..."],
          "audit_trail": ["..."]
        }}
        """
    ).strip()

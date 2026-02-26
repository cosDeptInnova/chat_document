from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from crewai import Agent, Crew, LLM, Process, Task  # type: ignore[import]

from hybrid_crew_src.agents import create_query_rewriter_agent, create_result_evaluator_agent
from hybrid_crew_src.prompts import build_query_rewriter_prompt, build_result_evaluator_prompt
from hybrid_crew_src.utils import clean_llm_text, extract_json_from_text

logger = logging.getLogger(__name__)


class HybridRAGCrewOrchestrator:
    """Orquesta agentes sin tools para preparación de filtros y explicación final."""

    def __init__(self, llm: Optional[LLM] = None) -> None:
        default_base = (os.getenv("CREW_BASE_URL", "http://127.0.0.1:8090/api/v1") or "").rstrip("/")
        default_model = os.getenv("HYBRID_RAG_CREW_MODEL", os.getenv("CREW_MODEL_NAME", "Llama3_8B_Cosmos"))
        api_key = os.getenv("CREW_API_KEY", "dummy-local-key")
        temperature = float(os.getenv("HYBRID_RAG_CREW_TEMPERATURE", os.getenv("CREW_TEMPERATURE", "0.1")))
        verbose = bool(int(os.getenv("CREW_VERBOSE", "0")))

        llm_default = LLM(
            model=default_model,
            api_key=api_key,
            base_url=default_base,
            temperature=temperature,
        )

        self.llm = llm or llm_default
        self.query_agent: Agent = create_query_rewriter_agent(self.llm, verbose=verbose)
        self.eval_agent: Agent = create_result_evaluator_agent(self.llm, verbose=verbose)

        crew_conc = max(1, int(os.getenv("HYBRID_RAG_CREW_CONCURRENCY", "8")))
        self._sem = threading.BoundedSemaphore(value=crew_conc)

    def _format_history(self, history: Optional[List[Dict[str, Any]]], max_turns: int = 6) -> str:
        if not history:
            return "No hay historial relevante."
        lines: List[str] = []
        for item in history[-max_turns:]:
            q = str(item.get("query") or "").strip()
            a = str(item.get("answer") or "").strip()
            if q or a:
                lines.append(f"Usuario: {q}\nAsistente: {a}")
        return "\n\n".join(lines) if lines else "No hay historial relevante."

    def _fallback_query_plan(self, user_query: str, user_id: str) -> Dict[str, Any]:
        return {
            "normalized_query": user_query.strip(),
            "qdrant_filter": {
                "must": [{"key": "user_id", "match": user_id}],
                "must_any": [],
                "datetime_range": None,
            },
            "retrieval_hints": {"limit": 5, "min_similarity": 0.0, "focus": []},
            "assumptions": ["Plan de contingencia por salida no parseable de LLM."],
            "risk_flags": ["low_confidence_filtering"],
            "audit_notes": ["fallback_query_plan"],
        }

    def _relative_time_range(self, text: str) -> Optional[Dict[str, str]]:
        s = (text or "").lower()
        now = datetime.now(timezone.utc)
        if "hace" not in s:
            return None

        def _make(days: int) -> Dict[str, str]:
            dt_from = (now - timedelta(days=days)).replace(microsecond=0)
            return {"from": dt_from.isoformat(), "to": now.replace(microsecond=0).isoformat()}

        tokens = s.split()
        for i, tok in enumerate(tokens):
            if tok == "hace" and i + 2 < len(tokens):
                qty = tokens[i + 1]
                unit = tokens[i + 2]
                if qty.isdigit():
                    n = int(qty)
                    if unit.startswith("día") or unit.startswith("dia"):
                        return _make(n)
                    if unit.startswith("sem"):
                        return _make(n * 7)
                    if unit.startswith("mes"):
                        return _make(n * 30)
                    if unit.startswith("año") or unit.startswith("ano"):
                        return _make(n * 365)
        return None

    def plan_query(self, *, user_query: str, user_id: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        history_str = self._format_history(history)
        prompt = build_query_rewriter_prompt(
            history_str=history_str,
            user_query=user_query,
            user_id=user_id,
        )
        task = Task(
            description=prompt,
            expected_output="JSON con normalized_query, qdrant_filter y hints.",
            agent=self.query_agent,
        )
        crew = Crew(
            agents=[self.query_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        acquired = self._sem.acquire(timeout=30)
        if not acquired:
            logger.warning("No se pudo adquirir semáforo de crew; usando fallback.")
            return self._fallback_query_plan(user_query=user_query, user_id=user_id)

        try:
            raw = crew.kickoff()
            raw_text = clean_llm_text(str(raw))
            obj = extract_json_from_text(raw_text)
            if not isinstance(obj, dict):
                return self._fallback_query_plan(user_query=user_query, user_id=user_id)

            obj.setdefault("normalized_query", user_query.strip())
            qfilter = obj.setdefault("qdrant_filter", {})
            qfilter.setdefault("must", [])
            qfilter.setdefault("must_any", [])
            if not any((x or {}).get("key") == "user_id" for x in qfilter.get("must", []) if isinstance(x, dict)):
                qfilter["must"].append({"key": "user_id", "match": user_id})

            if not qfilter.get("datetime_range"):
                inferred = self._relative_time_range(user_query)
                if inferred:
                    qfilter["datetime_range"] = inferred
                    obj.setdefault("audit_notes", []).append("datetime_range_inferred_from_relative_time")

            return obj
        except Exception as exc:
            logger.exception("Error en plan_query: %s", exc)
            return self._fallback_query_plan(user_query=user_query, user_id=user_id)
        finally:
            self._sem.release()

    def explain_results(
        self,
        *,
        user_query: str,
        normalized_query: str,
        filter_summary: Dict[str, Any],
        combined_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = build_result_evaluator_prompt(
            user_query=user_query,
            normalized_query=normalized_query,
            filter_summary=filter_summary,
            combined_results=combined_results,
        )
        task = Task(
            description=prompt,
            expected_output="JSON con final_answer y evaluación de evidencia.",
            agent=self.eval_agent,
        )
        crew = Crew(
            agents=[self.eval_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        acquired = self._sem.acquire(timeout=30)
        if not acquired:
            logger.warning("No se pudo adquirir semáforo de crew para explicación.")
            return {
                "final_answer": "No fue posible generar explicación avanzada en este momento.",
                "evidence_summary": [],
                "quality_assessment": {"coverage": "baja", "consistency": "media", "temporal_alignment": "media"},
                "follow_up_questions": [],
                "audit_trail": ["fallback_explain_results"],
            }

        try:
            raw = crew.kickoff()
            raw_text = clean_llm_text(str(raw))
            obj = extract_json_from_text(raw_text)
            if isinstance(obj, dict) and obj.get("final_answer"):
                return obj

            return {
                "final_answer": "He recuperado información relevante, pero no se pudo estructurar explicación automática completa.",
                "evidence_summary": [],
                "quality_assessment": {"coverage": "media", "consistency": "media", "temporal_alignment": "media"},
                "follow_up_questions": [],
                "audit_trail": ["non_parseable_evaluation_output"],
            }
        except Exception as exc:
            logger.exception("Error en explain_results: %s", exc)
            return {
                "final_answer": "Se encontraron resultados, pero hubo un error al generar la explicación final.",
                "evidence_summary": [],
                "quality_assessment": {"coverage": "media", "consistency": "media", "temporal_alignment": "media"},
                "follow_up_questions": [],
                "audit_trail": [f"exception:{type(exc).__name__}"],
            }
        finally:
            self._sem.release()

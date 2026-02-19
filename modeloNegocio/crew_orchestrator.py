# crew_orchestrator.py

import os
import logging
from typing import Any, Dict, List, Optional
import httpx
from crewai import Agent, Task, Crew, Process, LLM

from cosmos_crew_src.agents import (
    create_assistant_agent,
    create_planner_agent,
    create_rag_analyst_agent,
    create_files_analyst_agent,
    create_rag_router_agent,
)
from cosmos_crew_src.prompts import (
    build_assistant_system_instructions,
    build_planner_prompt,
    build_rag_router_prompt,
)
from cosmos_crew_src.utils import clean_llm_text, should_use_toon

logger = logging.getLogger(__name__)


class CosmosCrewOrchestrator:
    """
    Orquestador de CrewAI para COSMOS.

    - Multiusuario / paralelo: no guarda estado por usuario, sólo referencias a LLMs y agentes.
    - Multi-LLM: cada agente puede usar un LLM distinto si se pasa en el constructor.
    - TOON: analistas de RAG y de archivos para resumir contexto grande y ahorrar tokens.
    """

    def __init__(
        self,
        llm_assistant: Optional[LLM] = None,
        llm_planner: Optional[LLM] = None,
        llm_rag_analyst: Optional[LLM] = None,
        llm_files_analyst: Optional[LLM] = None,
        llm_rag_router: Optional[LLM] = None,
    ) -> None:
        default_base = "http://127.0.0.1:8090/api/v1"
        base_url = (os.getenv("CREW_BASE_URL", default_base) or default_base).rstrip("/")

        default_model = os.getenv("CREW_MODEL_NAME", "Llama3_8B_Cosmos")
        api_key = os.getenv("CREW_API_KEY", "dummy-local-key")
        temperature = float(os.getenv("CREW_TEMPERATURE", "0.2"))
        verbose = bool(int(os.getenv("CREW_VERBOSE", "0")))

        logger.info(
            "[Crew] Inicializando LLM base: base_url=%s, model=%s",
            base_url,
            default_model,
        )

        llm_default = LLM(
            model=default_model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

        self.llm_assistant: LLM = llm_assistant or llm_default
        self.llm_planner: LLM = llm_planner or llm_default
        self.llm_rag_analyst: LLM = llm_rag_analyst or llm_default
        self.llm_files_analyst: LLM = llm_files_analyst or llm_default
        self.llm_rag_router: LLM = llm_rag_router or self.llm_planner or llm_default

        # Se mantiene para compatibilidad con tu endpoint (crew_orchestrator.llm is not None)
        self.llm: LLM = self.llm_assistant

        self.verbose: bool = verbose
        self.mcp_api_base: str = base_url

        crew_name_env = os.getenv("CREW_NAME", "business_crew")
        self.tools_catalog: List[Dict[str, Any]] = self._discover_tools_for_crew(
            crew_name=crew_name_env
        )

        if self.tools_catalog:
            logger.info(
                "[CrewTools] Tools disponibles para %s: %s",
                crew_name_env,
                [t.get("name") for t in self.tools_catalog],
            )
        else:
            logger.info(
                "[CrewTools] No se han descubierto tools para %s (puede ser normal en desarrollo).",
                crew_name_env,
            )


    #helpers para control de latencia y carga
    def _make_assistant_agent(self) -> Agent:
        return create_assistant_agent(self.llm_assistant, self.verbose)

    def _make_planner_agent(self) -> Agent:
        return create_planner_agent(self.llm_planner, self.verbose)

    def _make_rag_analyst_agent(self) -> Agent:
        return create_rag_analyst_agent(self.llm_rag_analyst, self.verbose)

    def _make_files_analyst_agent(self) -> Agent:
        return create_files_analyst_agent(self.llm_files_analyst, self.verbose)

    def _make_rag_router_agent(self) -> Agent:
        return create_rag_router_agent(self.llm_rag_router, self.verbose)

    #helper de autodescubrimiento de tools desde COSMOS_MCP
    def _discover_tools_for_crew(
        self,
        crew_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Descubre las tools registradas en el MCP para una crew concreta
        (vía endpoint HTTP /api/v1/tools del servicio COSMOS_MCP).

        - No rompe si el MCP no está disponible.
        - Se llama típicamente una vez en __init__ y se cachea en self.tools_catalog.
        """
        url = f"{self.mcp_api_base}/tools"

        timeout_s = float(os.getenv("TOOLS_DISCOVERY_TIMEOUT", "5.0"))

        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.get(url, params={"crew": crew_name})
                resp.raise_for_status()
                data = resp.json() or {}
                tools = data.get("tools") or []
        except Exception as exc:
            logger.warning(
                "[CrewTools] No se pudieron descubrir tools para crew=%s desde %s: %s",
                crew_name,
                url,
                exc,
            )
            return []

        # Validación ligera
        norm_tools: List[Dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            name = t.get("name")
            if not name:
                continue
            norm_tools.append(
                {
                    "name": str(name),
                    "description": str(t.get("description") or ""),
                    "tags": list(t.get("tags") or []),
                    "for_crews": list(t.get("for_crews") or []),
                    "extra": t.get("extra") or {},
                }
            )

        logger.info(
            "[CrewTools] Descubiertas %d tools para crew=%s: %s",
            len(norm_tools),
            crew_name,
            [t["name"] for t in norm_tools],
        )
        return norm_tools
    
    def _format_tools_for_planner(self) -> str:
        if not self.tools_catalog:
            return "No hay herramientas declaradas explícitamente; el orquestador se encargará de las búsquedas necesarias."

        lines: List[str] = []
        for t in self.tools_catalog:
            name = t.get("name")
            desc = t.get("description") or ""
            tags = t.get("tags") or []
            lines.append(f"- {name}: {desc} (tags: {', '.join(tags)})")
        return "\n".join(lines)

    # ----------------------
    # Helpers de formateo
    # ----------------------
    def _format_history(self, history_data: Dict[str, Any], max_turns: int = 6) -> str:
        if not history_data:
            return "No hay historial previo relevante."

        entries = history_data.get("conversation_history", [])[-max_turns:]
        if not entries:
            return "No hay historial previo relevante."

        chunks: List[str] = []
        for item in entries:
            u = (item.get("prompt") or "").strip()
            b = (item.get("response") or "").strip()
            if u or b:
                chunks.append(f"Usuario: {u}\nAsistente: {b}")
        return "\n\n".join(chunks) if chunks else "No hay historial previo relevante."

    def _extract_json_from_text(self, raw: str) -> dict:
        import json

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No se han encontrado llaves JSON en el texto del planner.")

        candidate = raw[start : end + 1]
        return json.loads(candidate)

    def _format_rag_results(
        self,
        rag_results: Optional[List[Dict[str, Any]]],
        max_docs: int = 5,
        max_similar_blocks: int = 3,
        max_chars: int = 12000,
    ) -> str:
        if not rag_results:
            return "No hay resultados RAG relevantes para esta consulta."

        def _safe_score(d: Dict[str, Any]) -> Optional[float]:
            for k in (
                "score",
                "similarity",
                "similarity_score",
                "rrf_score",
                "rerank_score",
            ):
                v = d.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            return None

        def _format_row_kv(
            row_kv: Dict[str, Any],
            indent: str = "  ",
            max_fields: int = 20,
        ) -> List[str]:
            lines: List[str] = []
            items = list(row_kv.items())[:max_fields]
            for k, v in items:
                lines.append(f"{indent}· {k}: {v}")
            if len(row_kv) > max_fields:
                lines.append(
                    f"{indent}… ({len(row_kv) - max_fields} campos adicionales omitidos)"
                )
            return lines

        chunks: List[str] = []
        for idx, r in enumerate(rag_results[:max_docs], start=1):
            meta = r.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}

            doc_name = (
                r.get("doc_name")
                or r.get("source")
                or r.get("title")
                or meta.get("file_name")
                or "Documento sin nombre"
            )

            score = _safe_score(r)
            main_text = str(r.get("text") or r.get("chunk") or "").strip()

            header_lines: List[str] = []
            header_lines.append(f"📄 Documento #{idx}: {doc_name}")
            if score is not None:
                header_lines.append(f"score: {score:.4f}")

            body_lines: List[str] = []
            if main_text:
                body_lines.append("FRAGMENTO PRINCIPAL:")
                body_lines.append(main_text)

            backend = meta.get("backend")
            row_kv = meta.get("row_kv") or {}
            sheet = meta.get("sheet")
            row_idx = meta.get("row_idx")

            if backend or sheet or row_kv:
                body_lines.append("METADATOS DEL REGISTRO PRINCIPAL:")
                if backend:
                    body_lines.append(f"- Origen: {backend}")
                if sheet:
                    body_lines.append(f"- Hoja de Excel: {sheet}")
                if row_idx is not None:
                    body_lines.append(f"- Fila (0-based o índice interno): {row_idx}")
                if isinstance(row_kv, dict) and row_kv:
                    body_lines.append("- Campos de la fila:")
                    body_lines.extend(_format_row_kv(row_kv, indent="  "))

            similar_blocks = r.get("similar_blocks") or []
            if isinstance(similar_blocks, list) and similar_blocks:
                body_lines.append(
                    "FRAGMENTOS SIMILARES (MISMOS O SIMILARES CONJUNTOS DE DATOS):"
                )
                for sb in similar_blocks[:max_similar_blocks]:
                    if not isinstance(sb, dict):
                        continue
                    sb_text = str(sb.get("text") or "").strip()
                    sb_score = _safe_score(sb)
                    sb_meta = sb.get("meta") or {}
                    if not isinstance(sb_meta, dict):
                        sb_meta = {}
                    sb_row_kv = sb_meta.get("row_kv") or {}

                    bullet = "• "
                    if sb_score is not None:
                        bullet += f"({sb_score:.4f}) "
                    bullet += sb_text
                    body_lines.append(bullet)

                    if isinstance(sb_row_kv, dict) and sb_row_kv:
                        body_lines.append("  Campos de la fila:")
                        body_lines.extend(_format_row_kv(sb_row_kv, indent="    "))

            chunk = "\n".join(header_lines + body_lines)
            chunks.append(chunk)

        full_text = "\n\n---\n\n".join(chunks)

        if len(full_text) > max_chars:
            return (
                full_text[:max_chars]
                + "\n\n… (Resto de resultados RAG truncados por longitud)\n"
            )

        return full_text

    def _format_ephemeral_files(
        self,
        ephemeral_files: Optional[List[Dict[str, Any]]],
    ) -> str:
        if not ephemeral_files:
            return "No hay archivos en vuelo actualmente."

        pieces: List[str] = []
        for f in ephemeral_files:
            fname = f.get("filename", "archivo_sin_nombre")
            text = str(f.get("text", ""))[:3000]
            ts = f.get("uploaded_at")
            header = f"➤ Archivo en vuelo: {fname}"
            if ts:
                header += f" (cargado en {ts})"
            pieces.append(f"{header}\n{text}")
        return "\n\n".join(pieces)

    def _build_system_instructions(self, flow: Optional[str]) -> str:
        """
        Construye las instrucciones de sistema para el asistente principal,
        delegando en el builder centralizado de prompts.

        - `flow` puede ser:
            · 'R' → modo respuesta rápida (anclado al doc más relevante)
            · 'C' → modo COSMOS inteligente (combina varias fuentes)
            · None → se asume 'C' por defecto dentro del builder.

        La lógica detallada del prompt vive en `cosmos_crew_src.prompts` para
        evitar superprompts hardcodeados aquí.
        """
        return build_assistant_system_instructions(flow)

    # ----------------------
    # Helper genérico para ejecutar 1 agente + 1 tarea
    # ----------------------
        # ----------------------
    # Helper genérico para ejecutar 1 agente + 1 tarea
    # ----------------------
    def _run_single_agent_task(
        self,
        agent: Agent,
        description: str,
        expected_output: str,
    ) -> str:
        import uuid

        task_id = uuid.uuid4().hex[:8]

        logger.info(
            "[CREW_TASK %s] Iniciando task con agente='%s' (len_desc=%d)",
            task_id,
            agent.role,
            len(description),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[CREW_TASK %s] Descripción completa (truncada a 400 chars):\n%s",
                task_id,
                description[:400],
            )

        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=agent.verbose,
        )

        try:
            result = crew.kickoff()
            logger.info(
                "[CREW_TASK %s] Task completado para agente='%s'",
                task_id,
                agent.role,
            )
        except Exception as exc:
            logger.exception(
                "[CREW_TASK %s] Error ejecutando task con el agente '%s': %s",
                task_id,
                agent.role,
                exc,
            )
            return ""

        # Normalización de salida
        if hasattr(result, "raw") and isinstance(result.raw, str):
            out_str = result.raw
        else:
            tasks_output = getattr(result, "tasks_output", None)
            if tasks_output:
                first = tasks_output[0]
                output = getattr(first, "output", None)
                if isinstance(output, str):
                    out_str = output
                elif output is not None:
                    out_str = str(output)
                else:
                    out_str = str(result)
            else:
                out_str = str(result)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[CREW_TASK %s] Salida bruta del agente '%s' (truncada a 800 chars): %r",
                task_id,
                agent.role,
                out_str[:800],
            )

        return out_str

    def _summarize_rag(
        self,
        user_prompt: str,
        rag_results: Optional[List[Dict[str, Any]]],
        history_data: Optional[Dict[str, Any]],
        flow: Optional[str],
    ) -> str:
        if not rag_results:
            logger.info("[RAG_SUMMARY] Sin resultados RAG: se devuelve mensaje por defecto.")
            return "No hay resultados RAG relevantes para esta consulta."

        history_str = self._format_history(history_data or {})
        rag_str = self._format_rag_results(
            rag_results or [],
            max_docs=5,
            max_similar_blocks=3,
            max_chars=12000,
        )

        logger.info(
            "[RAG_SUMMARY] Contexto RAG formateado (len=%d chars, num_docs=%d)",
            len(rag_str),
            len(rag_results or []),
        )

        if not should_use_toon(rag_str):
            logger.info("[RAG_SUMMARY] Longitud RAG por debajo del umbral; no se usa analista RAG (TOON).")
            return rag_str

        flow_norm = (flow or "").upper()
        system = (
            "Eres el analista de documentos internos (RAG) de COSMOS.\n"
            "Tu misión es LEER los fragmentos de documentos y generar un resumen estructurado y compacto, "
            "eliminando ruido y texto irrelevante.\n"
            "- NO respondas al usuario final.\n"
            "- NO expliques el proceso, solo resume.\n"
        )
        if flow_norm == "R":
            system += "\nModo R (respuesta rápida): prioriza el documento con mejor score y usa máximo 6 bullets.\n"
        else:
            system += "\nModo C (COSMOS inteligente): puedes combinar varios documentos pero máximo 10 bullets.\n"

        description = f"""{system}

    === Historial reciente (para contexto) ===
    {history_str}

    === Resultados RAG BRUTOS ===
    {rag_str}

    === Pregunta del usuario ===
    {user_prompt}

    Devuelve SOLO texto en español con esta estructura:

    INSIGHTS_CLAVE:
    - ...

    DATOS_NUMERICOS:
    - ...

    LIMITACIONES:
    - ...

    Sé muy conciso. No añadas texto fuera de esas secciones.
    """

        logger.info("[RAG_SUMMARY] Invocando analista RAG (TOON) con prompt len=%d chars", len(description))

        # ✅ Agent fresh por llamada (thread-safe)
        rag_analyst_agent = self._make_rag_analyst_agent()

        summary = self._run_single_agent_task(
            agent=rag_analyst_agent,
            description=description,
            expected_output="Un resumen estructurado con secciones INSIGHTS_CLAVE, DATOS_NUMERICOS y LIMITACIONES.",
        )

        summary = clean_llm_text(summary)
        if not summary:
            logger.warning("[RAG_SUMMARY] El analista RAG devolvió vacío o solo ruido; se usa texto RAG recortado.")
            return rag_str[:6000]

        logger.info("[RAG_SUMMARY] Resumen TOON obtenido (len=%d chars).", len(summary))
        return summary


    def _summarize_ephemeral_files(
        self,
        user_prompt: str,
        ephemeral_files: Optional[List[Dict[str, Any]]],
        flow: Optional[str],
    ) -> str:
        if not ephemeral_files:
            logger.info("[EPH_SUMMARY] No hay archivos en vuelo actualmente.")
            return "No hay archivos en vuelo actualmente."

        eph_str = self._format_ephemeral_files(ephemeral_files)

        logger.info(
            "[EPH_SUMMARY] Texto de archivos en vuelo formateado (len=%d chars, num_files=%d)",
            len(eph_str),
            len(ephemeral_files or []),
        )

        if not should_use_toon(eph_str):
            logger.info("[EPH_SUMMARY] Longitud por debajo del umbral; no se usa analista de archivos (TOON).")
            return eph_str

        flow_norm = (flow or "").upper()
        system = (
            "Eres el analista de archivos en vuelo de COSMOS.\n"
            "Lees documentos que el usuario ha subido recientemente (PDF, Word, Excel, correos, etc.) y "
            "extraes SOLO la información que aporta contexto o responde a la pregunta actual.\n"
            "- NO respondas al usuario final.\n"
            "- NO repitas el texto original, solo resume.\n"
        )
        if flow_norm == "R":
            system += "\nModo R: máximo 5 bullets muy directos.\n"
        else:
            system += "\nModo C: máximo 8 bullets.\n"

        description = f"""{system}

    === Archivos en vuelo (texto recortado) ===
    {eph_str}

    === Pregunta del usuario ===
    {user_prompt}

    Devuelve SOLO texto en español con esta estructura:

    ARCHIVOS_RELEVANTES:
    - ...

    LIMITACIONES_ARCHIVOS:
    - ...

    No añadas nada fuera de estas secciones.
    """

        logger.info("[EPH_SUMMARY] Invocando analista de archivos en vuelo (TOON) con prompt len=%d chars", len(description))

        # ✅ Agent fresh por llamada (thread-safe)
        files_analyst_agent = self._make_files_analyst_agent()

        summary = self._run_single_agent_task(
            agent=files_analyst_agent,
            description=description,
            expected_output="Un resumen estructurado con secciones ARCHIVOS_RELEVANTES y LIMITACIONES_ARCHIVOS.",
        )

        summary = clean_llm_text(summary)
        if not summary:
            logger.warning("[EPH_SUMMARY] El analista de archivos devolvió vacío; se usa texto bruto recortado.")
            return eph_str[:4000]

        logger.info("[EPH_SUMMARY] Resumen de archivos en vuelo obtenido (len=%d chars).", len(summary))
        return summary


    # Planner interno (usa self.planner)
    def _plan_query(
        self,
        user_prompt: str,
        history_data: Optional[Dict[str, Any]] = None,
        flow: Optional[str] = None,
    ) -> Dict[str, Any]:
        history_str = self._format_history(history_data or {})
        tools_block = self._format_tools_for_planner()

        description = build_planner_prompt(
            flow=flow,
            history_str=history_str,
            user_prompt=user_prompt,
            tools_block=tools_block,
        )

        logger.debug("Prompt enviado al planner:\n%s", description)

        # ✅ Agent fresh por llamada (thread-safe)
        planner_agent = self._make_planner_agent()

        task = Task(
            description=description,
            expected_output="Un JSON con el plan de consulta descrito en las instrucciones.",
            agent=planner_agent,
        )

        crew = Crew(
            agents=[planner_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=planner_agent.verbose,
        )

        try:
            logger.info(
                "[PLANNER] Ejecutando planner.kickoff (flow=%s, len_prompt=%d)",
                (flow or "").upper(),
                len(description),
            )
            result = crew.kickoff()
            logger.info("[PLANNER] planner.kickoff completado correctamente")
        except Exception as exc:
            logger.exception("Error ejecutando planner.kickoff: %s", exc)
            fallback = {
                "normalized_question": user_prompt,
                "intent": "otro",
                "needs_rag": False,
                "needs_web": False,
                "needs_files": False,
                "needs_history": False,
                "rag_query": user_prompt,
                "filters": {},
            }

            logger.info("[PLANNER] Usando plan de fallback: %s", fallback)
            return fallback

        if hasattr(result, "raw") and isinstance(result.raw, str):
            raw_text = result.raw
        else:
            tasks_output = getattr(result, "tasks_output", None)
            if tasks_output:
                first = tasks_output[0]
                output = getattr(first, "output", None)
                raw_text = output if isinstance(output, str) else str(output)
            else:
                raw_text = str(result)

        raw_text = clean_llm_text(raw_text)

        try:
            plan = self._extract_json_from_text(raw_text)
            if not isinstance(plan, dict):
                raise ValueError("El JSON extraído no es un objeto.")

            logger.info(
                "[PLANNER] Plan generado: intent=%s, needs_rag=%s, needs_web=%s, filters=%s",
                plan.get("intent"),
                plan.get("needs_rag"),
                plan.get("needs_web"),
                plan.get("filters"),
            )
            return plan
        except Exception as exc:
            logger.exception("No se pudo parsear el plan del planner, usando fallback: %s", exc)
            fallback = {
                "normalized_question": user_prompt,
                "intent": "otro",
                "needs_rag": False,
                "needs_web": False,
                "needs_files": False,
                "needs_history": False,
                "rag_query": user_prompt,
                "filters": {},
            }

            logger.info("[PLANNER] Usando plan de fallback: %s", fallback)
            return fallback

        # ---------------------------------------------------------------------
    
    def route_rag_usage(
        self,
        user_prompt: str,
        history_data: Optional[Dict[str, Any]] = None,
        flow: Optional[str] = None,
        planner_plan: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Ejecuta el agente router para decidir si usar RAG.

        Devuelve:
        {"use_rag": bool, "reason": str}
        o None si falla (para que el caller mantenga comportamiento actual).
        """
        history_str = self._format_history(history_data or {})

        description = build_rag_router_prompt(
            flow=flow,
            history_str=history_str,
            user_prompt=user_prompt,
            planner_plan=planner_plan,
        )

        # ✅ Agent fresh por llamada (thread-safe)
        rag_router_agent = self._make_rag_router_agent()

        out = self._run_single_agent_task(
            agent=rag_router_agent,
            description=description,
            expected_output='JSON {"use_rag": true|false, "reason": "..."}',
        )

        out = clean_llm_text(out or "")
        if not out.strip():
            logger.warning("[RAG_ROUTER] Salida vacía del router.")
            return None

        try:
            data = self._extract_json_from_text(out)
            if not isinstance(data, dict):
                return None
            if "use_rag" not in data:
                return None

            use_rag = bool(data.get("use_rag"))
            reason = str(data.get("reason") or "").strip()
            return {"use_rag": use_rag, "reason": reason}
        except Exception as exc:
            logger.warning("[RAG_ROUTER] No se pudo parsear JSON del router: %s", exc)
            return None


    # Método público de planificación (para usar desde FastAPI)
    def plan_query(
        self,
        user_prompt: str,
        history_data: Optional[Dict[str, Any]] = None,
        user_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        flow = None
        if user_ctx:
            flow = user_ctx.get("flow")

        logger.info(
            "[PLANNER] plan_query llamado (flow=%s, len_user_prompt=%d, history_entries=%d)",
            (flow or "C"),
            len(user_prompt or ""),
            len((history_data or {}).get("conversation_history", []) or []),
        )

        return self._plan_query(
            user_prompt=user_prompt,
            history_data=history_data,
            flow=flow,
        )


    # ---------------------------------------------------------------------
    # Método principal de respuesta (usa el asistente)
    # ---------------------------------------------------------------------
    def run_chat(
        self,
        user_prompt: str,
        rag_results: Optional[List[Dict[str, Any]]] = None,
        ephemeral_files: Optional[List[Dict[str, Any]]] = None,
        history_data: Optional[Dict[str, Any]] = None,
        flow: Optional[str] = None,
        plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        import json as _json
        import logging
        import uuid

        logger = logging.getLogger(__name__)
        chat_id = uuid.uuid4().hex[:8]

        MAX_HISTORY_CHARS = 4000
        MAX_RAG_CHARS = 7000
        MAX_EPH_CHARS = 2000
        MAX_PROMPT_CHARS = 15000

        plan = plan or {}

        intent = plan.get("intent") or "otro"
        flow_norm = (flow or "C").upper()

        rag_status = plan.get("rag_status")
        rag_aggregation = plan.get("rag_aggregation")
        rag_used_department = plan.get("rag_used_department")
        rag_cleaned_query = plan.get("rag_cleaned_query")

        normalized_prompt = (
            (plan.get("normalized_prompt") or "").strip()
            or (plan.get("normalized_question") or "").strip()
        )

        must_not_mention_rag = bool(plan.get("must_not_mention_rag", False))
        allow_inherent_fallback = bool(plan.get("allow_inherent_fallback", False))

        history_len = len((history_data or {}).get("conversation_history", []))
        rag_count = len(rag_results or [])
        eph_count = len(ephemeral_files or [])

        attempted_rag = bool(plan.get("needs_rag", False))
        has_rag = bool(rag_results)
        use_rag = attempted_rag and has_rag

        use_files = bool(plan.get("needs_files", True)) and bool(ephemeral_files)
        use_history = bool(plan.get("needs_history", True))

        logger.info(
            "[CREW_RUN %s] run_chat iniciado: intent=%s, flow=%s, rag_status=%s, rag_results=%d, eph_files=%d, "
            "history_entries=%d, use_rag=%s, use_files=%s, use_history=%s, must_not_mention_rag=%s, allow_inherent_fallback=%s",
            chat_id, intent, flow_norm, rag_status, rag_count, eph_count, history_len,
            use_rag, use_files, use_history, must_not_mention_rag, allow_inherent_fallback
        )

        # 2) Historial
        history_str = ""
        if use_history:
            raw_history_str = self._format_history(history_data or {})
            history_str = raw_history_str or ""
            if len(history_str) > MAX_HISTORY_CHARS:
                history_str = history_str[-MAX_HISTORY_CHARS:]

        # 3) RAG
        rag_str = ""
        if use_rag:
            if intent in ("conteo", "consulta"):
                rag_str = self._format_rag_results(
                    rag_results,
                    max_docs=8,
                    max_similar_blocks=2,
                    max_chars=MAX_RAG_CHARS,
                )
            else:
                rag_str = self._summarize_rag(
                    user_prompt=user_prompt,
                    rag_results=rag_results,
                    history_data=history_data,
                    flow=flow,
                )
            if rag_str and len(rag_str) > MAX_RAG_CHARS:
                rag_str = rag_str[:MAX_RAG_CHARS]

        # 4) Ephemeral files
        eph_str = ""
        if use_files:
            eph_raw = self._summarize_ephemeral_files(
                user_prompt=user_prompt,
                ephemeral_files=ephemeral_files,
                flow=flow,
            )
            eph_str = eph_raw or ""
            if len(eph_str) > MAX_EPH_CHARS:
                eph_str = eph_str[:MAX_EPH_CHARS]

        # 5) System instructions base
        system_instructions = self._build_system_instructions(flow)

        if must_not_mention_rag or allow_inherent_fallback:
            system_instructions += (
                "\n\nREGLAS PRIORITARIAS:\n"
                "- NO menciones RAG, Redis, planners, crews, orquestadores, 'documentos internos', ni que faltan archivos.\n"
                "- Si el usuario ha pegado listas/tablas/valores/código/logs, ANALÍZALOS DIRECTAMENTE.\n"
                "- Responde con tu conocimiento general y razonamiento cuando no exista contexto documental.\n"
                "- Si falta información para una conclusión definitiva, da la mejor respuesta posible y explica qué faltaría.\n"
            )

        # 6) Meta-plan (opcional)
        meta_plan_block = ""
        if plan:
            try:
                plan_json = _json.dumps(plan, ensure_ascii=False, indent=2)
            except Exception:
                plan_json = str(plan)
            if len(plan_json) > 2000:
                plan_json = plan_json[:2000] + "... (plan recortado)"
            meta_plan_block = (
                "=== Meta-plan del planner (NO lo cites literal, úsalo solo como guía interna) ===\n"
                f"{plan_json}\n\n"
            )

        # 7) Aggregation
        agg_block = ""
        if use_rag and rag_aggregation and isinstance(rag_aggregation, dict):
            agg_count = rag_aggregation.get("count")
            by_table = rag_aggregation.get("by_table") or []
            lines: List[str] = []
            if agg_count is not None:
                lines.append(f"- Conteo total calculado: {agg_count} elementos.")
            for t in by_table:
                table_id = t.get("table_id")
                cnt = t.get("count")
                loc = t.get("location_filter")
                cols = t.get("columns_used") or []
                lines.append(f"  · Tabla {table_id or '?'} → {cnt} filas que contienen '{loc}' en columnas {cols}")
            agg_block = "=== Resultado de agregación tabular (conteo Excel) ===\n" + "\n".join(lines) + "\n\n"

        # 8) Bloques base: ORIGINAL + NORMALIZADA
        question_block = "=== Pregunta actual del usuario (ORIGINAL) ===\n" f"{user_prompt}\n\n"
        normalized_block = ""
        if normalized_prompt and normalized_prompt != (user_prompt or "").strip():
            normalized_block = (
                "=== Pregunta normalizada (solo para guía interna; no la cites literal) ===\n"
                f"{normalized_prompt}\n\n"
            )

        chat_tag_line = (
            f"[CHAT_ID {chat_id}] (etiqueta interna de trazabilidad; NO la menciones en la respuesta al usuario).\n\n"
        )

        if not attempted_rag:
            rag_status_label = "no_usado_planner"
        else:
            rag_status_label = rag_status or ("ok" if has_rag else "sin_resultados")

        rag_status_line = f"- Estado RAG: {rag_status_label}\n"
        if use_rag and rag_used_department:
            rag_status_line += f"- Índice utilizado: {rag_used_department}\n"
        if use_rag and rag_cleaned_query:
            rag_status_line += f"- Query normalizada usada en RAG: {rag_cleaned_query}\n"

        high_level_block = (
            chat_tag_line
            + f"{system_instructions}\n\n"
            "=== Datos de alto nivel de la consulta ===\n"
            f"- Intent detectado: {intent}\n"
            f"- Flow actual: {flow_norm}\n"
            f"{rag_status_line}\n"
        )

        # 9) Tail por intent (tu lógica)
        if intent == "conteo":
            if (not use_rag) and allow_inherent_fallback:
                tail = (
                    "TU TAREA PRINCIPAL AHORA MISMO ES DE CONTEO.\n"
                    "- NO hay RAG en uso, pero el usuario ha pegado datos/listas/tablas suficientes.\n"
                    "- Cuenta directamente a partir de lo pegado (por ejemplo: filas/entradas/elementos) y devuelve el total.\n"
                    "- Explica brevemente el criterio de conteo.\n"
                )
            else:
                tail = (
                    "TU TAREA PRINCIPAL AHORA MISMO ES DE CONTEO.\n"
                    "- Si NO hay resultados RAG/agregación, NO inventes números: explica qué datos faltan.\n"
                    "- Si hay agregación, úsala como fuente principal.\n"
                    "- Si no hay agregación pero hay filas RAG, cuenta filas relevantes.\n"
                )
        elif intent == "consulta":
            if (not use_rag) and allow_inherent_fallback:
                tail = (
                    "La consulta es de DATOS/VALIDACIÓN.\n"
                    "- NO hay RAG en uso, pero el usuario ha aportado información suficiente en el texto.\n"
                    "- Analiza lo pegado (listas/tablas/valores/código/logs) y responde directamente.\n"
                    "- NO pidas archivos internos ni menciones RAG.\n"
                )
            else:
                tail = (
                    "La consulta es de DATOS CONCRETOS.\n"
                    "- Si hay RAG, responde usando 'Resultados RAG'.\n"
                    "- Si no hay RAG, no inventes datos internos; responde con lo disponible y explica límites.\n"
                )
        else:
            if (not use_rag) and allow_inherent_fallback:
                tail = (
                    "Genera una respuesta final usando conocimiento general y razonamiento.\n"
                    "- Si hay datos pegados por el usuario, analízalos directamente.\n"
                    "- NO menciones RAG ni falta de documentos.\n"
                )
            else:
                tail = (
                    "Genera una respuesta final integrando toda la información útil.\n"
                    "Si algo no está claro, indícalo de forma explícita, pero antes da tu mejor respuesta.\n"
                )

        core = question_block + normalized_block + high_level_block + tail + "\n\n"

        history_block = f"=== Historial reciente ===\n{history_str}\n\n" if history_str else ""
        rag_block = f"=== Resultados RAG (filas de inventario y texto relevante) ===\n{rag_str}\n\n" if rag_str else ""
        eph_block = f"=== Archivos en vuelo (si los hay) ===\n{eph_str}\n\n" if eph_str else ""

        context_sections: List[str] = [history_block, rag_block, agg_block, eph_block, meta_plan_block]

        if len(core) >= MAX_PROMPT_CHARS:
            logger.warning(
                "[CREW_RUN %s] Núcleo del prompt demasiado largo (%d chars). Recortando preservando la pregunta.",
                chat_id, len(core)
            )
            remaining_after_question = MAX_PROMPT_CHARS - len(question_block)
            if remaining_after_question <= 0:
                description = question_block[:MAX_PROMPT_CHARS]
            else:
                other_core = normalized_block + high_level_block + tail + "\n\n"
                description = question_block + (other_core[:remaining_after_question])
        else:
            remaining = MAX_PROMPT_CHARS - len(core)
            description = core
            for block in context_sections:
                if not block:
                    continue
                if len(block) <= remaining:
                    description += block
                    remaining -= len(block)
                else:
                    if remaining > 0:
                        description += block[:remaining]
                    break

        logger.info(
            "[CREW_RUN %s] Prompt ensamblado (len=%d chars). intent=%s, flow=%s, rag_status=%s, use_rag=%s",
            chat_id, len(description), intent, flow_norm, rag_status, use_rag
        )

        # ✅ Agent fresh por llamada (thread-safe)
        assistant_agent = self._make_assistant_agent()

        task = Task(
            description=description,
            expected_output=(
                "Una respuesta en español, bien estructurada. Si hay datos pegados por el usuario, analízalos; "
                "si hay RAG, úsalo; si hay ambigüedad, aclárala sin mencionar sistemas internos."
            ),
            agent=assistant_agent,
        )

        crew = Crew(
            agents=[assistant_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=assistant_agent.verbose,
        )

        try:
            logger.info("[CREW_RUN %s] Ejecutando crew.kickoff()", chat_id)
            result = crew.kickoff()
            logger.info("[CREW_RUN %s] crew.kickoff() completado", chat_id)
        except Exception as exc:
            logger.exception("[CREW_RUN %s] Error ejecutando crew.kickoff: %s", chat_id, exc)
            return (
                "Ha ocurrido un error interno al generar la respuesta con el motor de IA. "
                "Por favor, inténtalo de nuevo en unos instantes. "
                "Si el problema persiste, contacta con el equipo de COSMOS."
            )

        if hasattr(result, "raw") and isinstance(result.raw, str):
            return result.raw

        tasks_output = getattr(result, "tasks_output", None)
        if tasks_output:
            first = tasks_output[0]
            output = getattr(first, "output", None)
            if isinstance(output, str):
                return output
            if output is not None:
                return str(output)

        logger.warning("[CREW_RUN %s] Resultado sin 'raw' ni 'tasks_output'. Fallback a str(result).", chat_id)
        return str(result)


    # ---------------------------------------------------------------------
    # Alias opcional para futuras extensiones MCP-tools
    # ---------------------------------------------------------------------
    def run_chat_with_mcp(
        self,
        user_prompt: str,
        history_data: Optional[Dict[str, Any]] = None,
        user_ctx: Optional[Dict[str, Any]] = None,
        rag_results: Optional[List[Dict[str, Any]]] = None,
        ephemeral_files: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        flow = None
        if user_ctx:
            flow = user_ctx.get("flow")

        # Por ahora no tenemos plan aquí; podríamos volver a llamar al planner si hace falta
        return self.run_chat(
            user_prompt=user_prompt,
            rag_results=rag_results,
            ephemeral_files=ephemeral_files,
            history_data=history_data,
            flow=flow,
            plan=None,
        )

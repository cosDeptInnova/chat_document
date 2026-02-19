# legal_crew_orchestrator.py

import os
import re
import json
import time
import logging
import inspect
import asyncio
import threading
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from time import perf_counter

from crewai import Agent, Task, Crew, Process, LLM  # type: ignore[import]

from legal_crew_src.agents import (
    create_legal_planner_agent,
    create_legal_analyst_agent,
    create_legal_writer_agent,
    create_legal_reviewer_agent,
)
from legal_crew_src.prompts import (
    build_legal_planner_prompt,
    build_legal_analyst_prompt,
    build_legal_writer_prompt,
    build_legal_verifier_prompt,
)
from legal_crew_src.utils import clean_llm_text
from mcp_client import MCPClient

logger = logging.getLogger(__name__)


class LegalSearchCrewOrchestrator:
    """
    Orquestador multi-agente para investigación legal basada en web/tools MCP.

    Pipeline:
      1) Planner: genera plan + queries legales (jurisdicción, vigencia, fuentes prioritarias).
      2) Ejecutor tools (orquestador): search_tool -> fetch_tool (con filtros/ranking).
      3) Analyst: decide si basta o propone nuevas queries.
      4) Writer: memo legal con citas [n] + riesgos + próximos pasos.
      5) Reviewer: valida contra fuentes y corrige overclaiming.

    Nota:
      - Los agentes NO tienen tools; el control de acceso se hace en MCP por token/rol.
    """

    def __init__(
        self,
        llm_legal_planner: Optional[LLM] = None,
        llm_legal_writer: Optional[LLM] = None
    ) -> None:
        # ----------------------------
        # LLM config
        # ----------------------------
        default_base = "http://127.0.0.1:8090/api/v1"
        base_url = (os.getenv("CREW_BASE_URL", default_base) or default_base).rstrip("/")

        default_model = os.getenv(
            "LEGALSEARCH_MODEL_NAME",
            os.getenv("CREW_MODEL_NAME", "Llama3_8B_Cosmos")
        )
        api_key = os.getenv("CREW_API_KEY", "dummy-local-key")
        temperature = float(os.getenv("LEGALSEARCH_TEMPERATURE", os.getenv("CREW_TEMPERATURE", "0.2")))
        verbose = bool(int(os.getenv("CREW_VERBOSE", "0")))

        llm_default = LLM(
            model=default_model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )

        self.llm_planner: LLM = llm_legal_planner or llm_default
        self.llm_writer: LLM = llm_legal_writer or llm_default

        # ----------------------------
        # Agents (SIN tools)
        # ----------------------------
        self.planner_agent: Agent = create_legal_planner_agent(self.llm_planner, verbose)
        self.analyst_agent: Agent = create_legal_analyst_agent(self.llm_writer, verbose)
        self.writer_agent: Agent = create_legal_writer_agent(self.llm_writer, verbose)
        self.reviewer_agent: Agent = create_legal_reviewer_agent(self.llm_writer, verbose)

        # ----------------------------
        # MCP client
        # ----------------------------
        self.mcp = MCPClient()

        # ----------------------------
        # MCP config
        # ----------------------------
        self.crew_name = os.getenv("LEGALSEARCH_MCP_CREW_NAME", "web_search_crew")

        # Nombres “por defecto” (pueden ser sustituidos por auto-pick por tags)
        self.search_tool_name = os.getenv("LEGALSEARCH_SEARCH_TOOL_NAME", "ddg_search_tool")
        self.fetch_tool_name = os.getenv("LEGALSEARCH_FETCH_TOOL_NAME", "ddg_fetch_content_tool")

        # Tokens opcionales por rol (si quieres limitar list_tools vs ejecución).
        # IMPORTANTE: en multiusuario, el auth_token de request debe prevalecer.
        self.mcp_registry_token = os.getenv("LEGALSEARCH_MCP_REGISTRY_TOKEN", "").strip() or None
        self.mcp_executor_token = os.getenv("LEGALSEARCH_MCP_EXECUTOR_TOKEN", "").strip() or None

        # Tracing/metrics
        self.enable_tool_traces = os.getenv("LEGALSEARCH_TRACE_TOOLS", "0") == "1"
        self.search_return_format = "trace_json" if self.enable_tool_traces else "json"
        self.fetch_return_format = "trace_json" if self.enable_tool_traces else "text"

        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self._tools_lock = threading.RLock()
        self._tools_last_refresh = 0.0
        self._tools_ttl_sec = float(os.getenv("LEGALSEARCH_TOOLS_TTL_SEC", "60"))

        # ----------------------------
        # Filtros/ranking de fuentes
        # ----------------------------
        self.allowed_domains = self._parse_domain_list(os.getenv("LEGALSEARCH_ALLOWED_DOMAINS", ""))
        self.blocked_domains = self._parse_domain_list(os.getenv("LEGALSEARCH_BLOCKED_DOMAINS", ""))
        self.preferred_domains = self._parse_domain_list(os.getenv("LEGALSEARCH_PREFERRED_DOMAINS", ""))

        # ----------------------------
        # Tags preferentes para auto-selección (configurable)
        # ----------------------------
        self.search_tag_preferences = [
            {"legal", "search"},
            {"web", "search"},
            {"search"},
        ]
        self.fetch_tag_preferences = [
            {"legal", "fetch"},
            {"web", "fetch"},
            {"fetch"},
        ]

    # ----------------------
    # Domain helpers
    # ----------------------
    def _parse_domain_list(self, raw: str) -> List[str]:
        items = []
        for x in (raw or "").split(","):
            d = x.strip().lower()
            if d:
                items.append(d)
        return items

    def _url_domain(self, url: str) -> str:
        try:
            return (urlparse((url or "").strip()).netloc or "").lower()
        except Exception:
            return ""

    def _domain_allowed(self, url: str) -> bool:
        dom = self._url_domain(url)
        if not dom:
            return False

        # blocked takes precedence
        for bd in self.blocked_domains:
            if dom == bd or dom.endswith("." + bd):
                return False

        # allowlist if provided
        if self.allowed_domains:
            for ad in self.allowed_domains:
                if dom == ad or dom.endswith("." + ad):
                    return True
            return False

        return True

    def _score_source_candidate(self, item: Dict[str, Any]) -> float:
        """
        Ranking simple (no invasivo) orientado a legal:
        - prefer official-like domains (.gov, .gob, europa.eu, eur-lex, boe…)
        - prefer env preferred_domains
        - small bump for longer snippet/title completeness
        """
        url = (item.get("url") or "").strip()
        dom = self._url_domain(url)
        if not dom:
            return -1e9

        score = 0.0

        # preferred domains (configurable)
        for pd in self.preferred_domains:
            if dom == pd or dom.endswith("." + pd):
                score += 50.0
                break

        # heurísticas "oficial"
        official_markers = (
            dom.endswith(".gov")
            or dom.endswith(".gob")
            or dom.endswith(".gov.uk")
            or dom.endswith(".eu")
            or dom.endswith(".int")
            or "eur-lex" in dom
            or "boe.es" in dom
            or "europa.eu" in dom
            or "curia.europa.eu" in dom
        )
        if official_markers:
            score += 35.0

        # pdf/legal docs tend to be useful
        if url.lower().endswith(".pdf"):
            score += 6.0

        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        score += min(8.0, len(title) / 80.0)
        score += min(8.0, len(snippet) / 200.0)

        return score

    # ----------------------
    # JSON extraction helpers
    # ----------------------
    def _extract_json_from_text(self, text: str) -> Optional[Any]:
        """
        Extrae JSON de forma robusta:
        - Soporta ```json ...```, <json>...</json>
        - Intenta detectar el primer bloque {} o [] balanceado.
        - Si falla, intenta heurísticas mínimas sin romper.
        """
        if not text:
            return None

        t = clean_llm_text(text)

        # 1) Bloques explícitos
        m = re.search(r"```json\s*(.*?)\s*```", t, flags=re.S | re.I)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass

        m = re.search(r"<json>\s*(.*?)\s*</json>", t, flags=re.S | re.I)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass

        # 2) Intento de extraer el primer objeto/array JSON balanceado
        #    (evita capturar llaves sueltas dentro de texto largo)
        def _balanced_extract(s: str) -> Optional[str]:
            s = s.strip()
            # Busca primer '{' o '['
            start_idx = None
            start_ch = ""
            for i, ch in enumerate(s):
                if ch in "{[":
                    start_idx = i
                    start_ch = ch
                    break
            if start_idx is None:
                return None
            end_ch = "}" if start_ch == "{" else "]"

            depth = 0
            in_str = False
            esc = False
            for j in range(start_idx, len(s)):
                c = s[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                else:
                    if c == '"':
                        in_str = True
                        continue
                    if c == start_ch:
                        depth += 1
                    elif c == end_ch:
                        depth -= 1
                        if depth == 0:
                            return s[start_idx : j + 1]
            return None

        candidate = _balanced_extract(t)
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # 3) Heurística: lista de strings tipo ["a","b"]
        m = re.search(r"(\[\s*\".*?\"\s*(?:,\s*\".*?\"\s*)*\])", t, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass

        return None




    def _parse_planner_output(self, text: str, fallback_prompt: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Acepta:
        - Lista JSON: ["q1","q2"]
        - Dict JSON: {"queries":[...], "jurisdictions":[...], ...}
        - Fallback: líneas
        Devuelve (queries, plan_meta)
        """
        meta: Dict[str, Any] = {}

        if not text:
            return [fallback_prompt], meta

        obj = self._extract_json_from_text(text)

        if isinstance(obj, dict):
            meta = obj
            qs = obj.get("queries")
            if isinstance(qs, list):
                out = [str(x).strip() for x in qs if str(x).strip()]
                return (out[:4] or [fallback_prompt]), meta

        if isinstance(obj, list):
            out = [str(x).strip() for x in obj if str(x).strip()]
            return (out[:4] or [fallback_prompt]), meta

        lines = self._split_lines(text)[:4]
        return (lines or [fallback_prompt]), meta

    def _parse_analyst_output(self, text: str) -> Dict[str, Any]:
        raw = clean_llm_text(text or "")
        if not raw:
            return {"stop": True, "queries": []}

        obj = self._extract_json_from_text(raw)
        if isinstance(obj, dict):
            decision = str(obj.get("decision") or "").strip().upper()
            if decision == "BASTA":
                return {"stop": True, "queries": []}
            qs = obj.get("queries")
            if isinstance(qs, list):
                out = [str(x).strip() for x in qs if str(x).strip()]
                return {"stop": False, "queries": out[:3]}

        if raw.strip().upper() == "BASTA":
            return {"stop": True, "queries": []}

        lines = self._split_lines(raw)[:3]
        return {"stop": False, "queries": lines}

    # ----------------------
    # Tools discovery / refresh
    # ----------------------
    def _unwrap_tool_block(self, block: Any) -> Any:
        if isinstance(block, dict) and "result" in block:
            if len(block.keys()) == 1:
                return block.get("result")
            inner = block.get("result")
            if isinstance(inner, (dict, list, str)):
                return inner
        return block

    def _refresh_tools(self, *, force: bool = False, user_auth_token: Optional[str] = None) -> None:
        now = time.time()
        with self._tools_lock:
            if not force and self.available_tools and (now - self._tools_last_refresh) < self._tools_ttl_sec:
                return

        crews_to_try: List[Optional[str]] = []
        primary = (self.crew_name or "").strip()
        if primary:
            crews_to_try.append(primary)
        if "web_search_crew" not in crews_to_try:
            crews_to_try.append("web_search_crew")
        crews_to_try.append(None)

        tools_out: List[Dict[str, Any]] = []
        last_err: Optional[Exception] = None

        for crew_name in crews_to_try:
            try:
                tools = self.mcp.list_tools(
                    crew=crew_name,
                    auth_token=self.mcp_registry_token,          # S2S
                    # Si implementas list_tools con user_auth_token, pásalo aquí:
                    # user_auth_token=user_auth_token,
                ) or []

                if isinstance(tools, dict) and "tools" in tools:
                    tools = tools.get("tools") or []

                tools = self._unwrap_tool_block(tools)
                if isinstance(tools, dict) and "tools" in tools:
                    tools = tools.get("tools") or []

                if isinstance(tools, list) and tools:
                    tools_out = tools
                    if crew_name:
                        self.crew_name = crew_name
                    break

            except Exception as e:
                last_err = e
                continue

        with self._tools_lock:
            if not tools_out:
                logger.warning(
                    "[LEGALSEARCH] No se pudo refrescar tools (crew tried=%s). last_err=%s",
                    crews_to_try,
                    last_err,
                )
                if not self.available_tools:
                    self.available_tools = {}
                self._tools_last_refresh = now
                return

            self.available_tools = {
                t.get("name"): t
                for t in tools_out
                if isinstance(t, dict) and t.get("name")
            }
            self._tools_last_refresh = now

        logger.info("[LEGALSEARCH] Tools MCP crew=%s: %s", self.crew_name, list(self.available_tools.keys()) or "(ninguna)")

    def _auto_pick_tool(self, kind: str) -> Optional[str]:
        if not self.available_tools:
            return None

        kind = (kind or "").lower().strip()
        if kind not in ("search", "fetch"):
            return None

        tag_sets = self.search_tag_preferences if kind == "search" else self.fetch_tag_preferences

        # 1) por tags preferentes
        for tag_set in tag_sets:
            for name, meta in self.available_tools.items():
                tags = meta.get("tags") or []
                if not isinstance(tags, list):
                    continue
                tag_norm = {str(t).strip().lower() for t in tags if str(t).strip()}
                if tag_set.issubset(tag_norm):
                    return name

        # 2) fallback por nombre
        if kind == "search":
            for name in self.available_tools.keys():
                if "search" in name.lower():
                    return name
        else:
            for name in self.available_tools.keys():
                low = name.lower()
                if "fetch" in low or "content" in low or "scrape" in low:
                    return name

        return None


    def _validate_tool_names(self) -> None:
        with self._tools_lock:
            if not self.available_tools:
                return

            def pick(kind: str) -> Optional[str]:
                kind = kind.lower().strip()
                if kind == "search":
                    preferred = [{"web", "search"}, {"search"}]
                else:
                    preferred = [{"web", "fetch"}, {"fetch"}]

                for tag_set in preferred:
                    for name, meta in self.available_tools.items():
                        tags = meta.get("tags") or []
                        if not isinstance(tags, list):
                            continue
                        tag_norm = {str(t).strip().lower() for t in tags if str(t).strip()}
                        if tag_set.issubset(tag_norm):
                            return name

                if kind == "search":
                    for n in self.available_tools.keys():
                        if "search" in n.lower():
                            return n
                else:
                    for n in self.available_tools.keys():
                        if "fetch" in n.lower() or "content" in n.lower():
                            return n

                return None

            if self.search_tool_name not in self.available_tools:
                self.search_tool_name = pick("search") or self.search_tool_name

            if self.fetch_tool_name not in self.available_tools:
                self.fetch_tool_name = pick("fetch") or self.fetch_tool_name

    # ----------------------
    # Runner helper (LLM)
    # ----------------------
    def _run_single_agent_task(self, agent: Agent, description: str, expected_output: str) -> str:
        task = Task(description=description, expected_output=expected_output, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=agent.verbose)
        try:
            result = crew.kickoff()
        except Exception:
            logger.exception("[LEGALSEARCH] ERROR agent='%s'", getattr(agent, "role", "unknown"))
            return ""

        out = ""
        try:
            if hasattr(result, "raw") and isinstance(result.raw, str):
                out = result.raw
            else:
                outs = getattr(result, "tasks_output", None)
                if outs and getattr(outs[0], "output", None):
                    out = outs[0].output or ""
                else:
                    out = str(result or "")
        except Exception:
            out = ""

        return clean_llm_text(out)

    # ----------------------
    # URL canonicalization + dedupe
    # ----------------------
    def _canon_url(self, url: str) -> str:
        try:
            p = urlparse((url or "").strip())
            scheme = (p.scheme or "https").lower()
            netloc = (p.netloc or "").lower()
            path = p.path or ""
            if path != "/" and path.endswith("/"):
                path = path[:-1]
            return urlunparse((scheme, netloc, path, "", p.query or "", ""))
        except Exception:
            return (url or "").strip()

    def _dedupe_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_url: Dict[str, Dict[str, Any]] = {}
        for s in sources:
            url = (s.get("url") or "").strip()
            if not url:
                continue
            key = self._canon_url(url)
            existing = by_url.get(key)
            if not existing:
                by_url[key] = s
                continue
            old_len = len((existing.get("content") or "").strip())
            new_len = len((s.get("content") or "").strip())
            if new_len > old_len:
                by_url[key] = s

        out: List[Dict[str, Any]] = []
        seen = set()
        for s in sources:
            url = (s.get("url") or "").strip()
            if not url:
                continue
            key = self._canon_url(url)
            if key in seen:
                continue
            if key in by_url:
                out.append(by_url[key])
                seen.add(key)
        return out

    # ----------------------
    # Parseo search
    # ----------------------
    def _parse_search_result(self, block: Any) -> List[Dict[str, Any]]:
        if not block:
            return []
        block = self._unwrap_tool_block(block)

        def _norm_from_list(raw_list: List[Any]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for r in raw_list:
                if not isinstance(r, dict):
                    continue
                url = (r.get("url") or r.get("link") or r.get("href") or "").strip()
                if not url:
                    continue
                if not self._domain_allowed(url):
                    continue
                title = str(r.get("title") or r.get("name") or url).strip()
                snippet = str(r.get("snippet") or r.get("summary") or r.get("description") or "").strip()
                content = str(r.get("content") or "").strip()
                row: Dict[str, Any] = {"title": title, "url": url, "snippet": snippet}
                if content:
                    row["content"] = content
                out.append(row)
            return out

        if isinstance(block, list):
            return _norm_from_list(block)

        if isinstance(block, dict):
            if "result" in block:
                inner = self._unwrap_tool_block(block)
                if inner is not block:
                    return self._parse_search_result(inner)

            results = block.get("results") or block.get("items") or block.get("data")
            if isinstance(results, list):
                return _norm_from_list(results)

            per_query = block.get("per_query")
            if isinstance(per_query, list):
                out: List[Dict[str, Any]] = []
                for pq in per_query:
                    pq = self._unwrap_tool_block(pq)
                    if isinstance(pq, dict):
                        rr = pq.get("results") or pq.get("items") or pq.get("data")
                        if isinstance(rr, list):
                            out.extend(_norm_from_list(rr))
                return out

        if isinstance(block, str):
            s = block.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    return self._parse_search_result(json.loads(s))
                except Exception:
                    return []
        return []

    # ----------------------
    # Tools execution
    # ----------------------
    def _execute_tools_once(self, queries: List[str], top_k: int, *, user_auth_token: Optional[str] = None) -> Dict[str, Any]:
        import random

        top_k = max(1, int(top_k or 1))

        def _int_env(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        def _float_env(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                return default

        max_chars_per_source = _int_env("LEGALSEARCH_MAX_CHARS_PER_SOURCE", 2200)
        max_sources_per_iter = _int_env("LEGALSEARCH_MAX_SOURCES_PER_ITER", 8)
        max_concurrent_fetch = _int_env("LEGALSEARCH_MAX_CONCURRENT_FETCH", 6)

        search_delay = _float_env("LEGALSEARCH_SEARCH_DELAY_SECONDS", 0.8)
        search_jitter = _float_env("LEGALSEARCH_SEARCH_JITTER_SECONDS", 0.3)

        disable_batch = os.getenv("LEGALSEARCH_DISABLE_SEARCH_BATCH", "0") == "1"

        tool_timeout = _float_env("LEGALSEARCH_TOOL_TIMEOUT_SEC", 60.0)
        # Timeout por iteración de la crew legal.
        # Subimos el default a 5 minutos para casos complejos de investigación
        # (múltiples fuentes + validación) sin cortar la ejecución prematuramente.
        iter_timeout = _float_env("LEGALSEARCH_ITER_TIMEOUT_SEC", 300.0)
        max_retries = _int_env("LEGALSEARCH_TOOL_RETRIES", 2)
        backoff_base = _float_env("LEGALSEARCH_TOOL_BACKOFF_BASE", 0.5)

        clean_queries = [q.strip() for q in (queries or []) if (q or "").strip()]
        if not clean_queries:
            return {"sources": [], "tool_trace": [], "tool_events": []}

        async def _ainvoke_with_backoff(
            tool_name: str,
            payload: Dict[str, Any],
            trace_acc: List[Dict[str, Any]],
            events_acc: List[Dict[str, Any]],
        ):
            last_exc = None
            for attempt in range(max_retries + 1):
                t0 = perf_counter()
                try:
                    coro = self.mcp.ainvoke_tool(
                        tool_name,
                        payload,
                        auth_token=self.mcp_executor_token,      # S2S
                        user_auth_token=user_auth_token,         # USER
                    )
                    out = await asyncio.wait_for(coro, timeout=tool_timeout)
                    out = self._unwrap_tool_block(out)

                    dur = perf_counter() - t0
                    events_acc.append({"tool": tool_name, "status": "ok", "duration": dur, "attempt": attempt})

                    if self.enable_tool_traces and isinstance(out, dict):
                        maybe_trace = out.get("trace") or out.get("tool_trace")
                        if isinstance(maybe_trace, (dict, list)):
                            trace_acc.append({"tool": tool_name, "trace": maybe_trace})

                    return out
                except Exception as e:
                    dur = perf_counter() - t0
                    events_acc.append({"tool": tool_name, "status": "error", "duration": dur, "attempt": attempt})
                    last_exc = e
                    if attempt >= max_retries:
                        break
                    delay = backoff_base * (2 ** attempt) + (random.random() * 0.2)
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        def _search_payload_variants(q: str) -> List[Dict[str, Any]]:
            max_results = max(10, top_k * 4)
            base_opts = {"return_format": self.search_return_format}
            return [
                {"query": q, "max_results": max_results, **base_opts},
                {"query": q, **base_opts},
                {"q": q, "max_results": max_results, **base_opts},
                {"queries": [q], "max_results": max_results, **base_opts},
            ]

        async def _search_single(q: str, trace_acc: List[Dict[str, Any]], events_acc: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
            block = None
            for payload in _search_payload_variants(q):
                try:
                    block = await _ainvoke_with_backoff(self.search_tool_name, payload, trace_acc, events_acc)
                    break
                except Exception:
                    continue
            if block is None:
                return q, []
            items = self._parse_search_result(block)
            return q, items

        async def _search_batch(qs: List[str], trace_acc: List[Dict[str, Any]], events_acc: List[Dict[str, Any]]):
            if disable_batch:
                return None
            payload = {"queries": qs, "max_results": max(10, top_k * 4), "return_format": self.search_return_format}
            try:
                block = await _ainvoke_with_backoff(self.search_tool_name, payload, trace_acc, events_acc)
            except Exception:
                return None
            block = self._unwrap_tool_block(block)
            if not isinstance(block, dict):
                return None
            per_query = block.get("per_query")
            if not isinstance(per_query, list):
                return None

            out: List[Tuple[str, List[Dict[str, Any]]]] = []
            for i, pq in enumerate(per_query):
                pq = self._unwrap_tool_block(pq)
                q = ""
                if isinstance(pq, dict):
                    q = (pq.get("query") or "").strip()
                items = self._parse_search_result(pq)
                out.append((q or (qs[i] if i < len(qs) else ""), items))
            return out

        def _fetch_payload_variants(url: str) -> List[Dict[str, Any]]:
            base_opts = {"return_format": self.fetch_return_format}
            return [
                {"url": url, "max_chars": max_chars_per_source, **base_opts},
                {"url": url, "max_length": max_chars_per_source, **base_opts},
                {"url": url, **base_opts},
                {"url": url},
            ]

        def _normalize_fetched_text(content: Any) -> str:
            content = self._unwrap_tool_block(content)
            if content is None:
                return ""
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, dict):
                for k in ("content", "text", "body", "article", "markdown", "html"):
                    v = content.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                try:
                    return json.dumps(content)[:max_chars_per_source].strip()
                except Exception:
                    return str(content).strip()
            return str(content).strip()

        async def _fetch_one(q: str, it: Dict[str, Any], sem: asyncio.Semaphore,
                            trace_acc: List[Dict[str, Any]], events_acc: List[Dict[str, Any]]):
            async with sem:
                url = (it.get("url") or "").strip()
                if not url or not self._domain_allowed(url):
                    return None

                title = (it.get("title") or url).strip()
                snippet = (it.get("snippet") or "").strip() or None

                pre_content = it.get("content")
                if isinstance(pre_content, str) and pre_content.strip():
                    text = pre_content.strip()
                    if len(text) > max_chars_per_source:
                        text = text[:max_chars_per_source] + "\n… [contenido truncado]"
                    return {"title": title, "url": url, "snippet": snippet, "content": text, "query": q, "from": "search_tool"}

                content = None
                for payload in _fetch_payload_variants(url):
                    try:
                        content = await _ainvoke_with_backoff(self.fetch_tool_name, payload, trace_acc, events_acc)
                        break
                    except Exception:
                        continue

                if content is None:
                    if snippet:
                        return {"title": title, "url": url, "snippet": snippet, "content": "", "query": q, "from": "search_snippet_fallback"}
                    return None

                text = _normalize_fetched_text(content)
                if not text:
                    if snippet:
                        return {"title": title, "url": url, "snippet": snippet, "content": "", "query": q, "from": "search_snippet_fallback"}
                    return None

                if len(text) > max_chars_per_source:
                    text = text[:max_chars_per_source] + "\n… [contenido truncado]"

                return {"title": title, "url": url, "snippet": snippet, "content": text, "query": q, "from": "fetch_tool"}

        async def _run_async():
            tool_trace: List[Dict[str, Any]] = []
            tool_events: List[Dict[str, Any]] = []
            sem_f = asyncio.Semaphore(max_concurrent_fetch)

            # SEARCH
            search_results = None
            if len(clean_queries) > 1:
                search_results = await _search_batch(clean_queries, tool_trace, tool_events)

            if not search_results:
                search_results = []
                for idx, q in enumerate(clean_queries):
                    if idx > 0 and (search_delay > 0 or search_jitter > 0):
                        await asyncio.sleep(search_delay + random.random() * search_jitter)
                    search_results.append(await _search_single(q, tool_trace, tool_events))

            # Preparar candidatos y rankear
            candidates: List[Tuple[str, Dict[str, Any], float]] = []
            for q, items in search_results:
                for it in (items or []):
                    sc = self._score_source_candidate(it)
                    candidates.append((q, it, sc))

            candidates.sort(key=lambda x: x[2], reverse=True)

            sources: List[Dict[str, Any]] = []
            seen_urls: set[str] = set()
            fetch_tasks = []

            for q, it, _ in candidates:
                if len(fetch_tasks) + len(sources) >= max_sources_per_iter:
                    break
                url = (it.get("url") or "").strip()
                if not url:
                    continue
                canon = self._canon_url(url)
                if canon in seen_urls:
                    continue
                seen_urls.add(canon)

                fetch_tasks.append(asyncio.create_task(_fetch_one(q, it, sem_f, tool_trace, tool_events)))
                if len(fetch_tasks) >= max_sources_per_iter:
                    break

            if not fetch_tasks:
                return {"sources": [], "tool_trace": tool_trace, "tool_events": tool_events}

            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for fr in fetch_results:
                if isinstance(fr, Exception) or not fr:
                    continue
                sources.append(fr)

            try:
                sources = self._dedupe_sources(sources)
            except Exception:
                pass

            if len(sources) > max_sources_per_iter:
                sources = sources[:max_sources_per_iter]

            return {"sources": sources, "tool_trace": tool_trace, "tool_events": tool_events}

        async def _run_with_timeout():
            try:
                return await asyncio.wait_for(_run_async(), timeout=iter_timeout)
            except asyncio.TimeoutError:
                logger.warning("[LEGALSEARCH] Iteración excedió timeout=%ss", iter_timeout)
                return {"sources": [], "tool_trace": [], "tool_events": []}

        try:
            asyncio.get_running_loop()
            out_container: Dict[str, Any] = {"value": {"sources": [], "tool_trace": [], "tool_events": []}, "error": None}

            def _worker():
                try:
                    out_container["value"] = asyncio.run(_run_with_timeout())
                except Exception as e:
                    out_container["error"] = e

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            t.join(timeout=max(30.0, iter_timeout * 2.0))

            if out_container["error"]:
                raise out_container["error"]

            return out_container["value"]

        except RuntimeError:
            return asyncio.run(_run_with_timeout())

    # ----------------------
    # Prompt builder wrapper
    # ----------------------
    def _build_analyst_desc(
        self,
        *,
        user_prompt: str,
        sources: List[Dict[str, Any]],
        accumulated_sources: List[Dict[str, Any]],
        used_queries: List[str],
    ) -> str:
        fn = build_legal_analyst_prompt
        kwargs = {
            "user_prompt": user_prompt,
            "sources": sources,
            "accumulated_sources": accumulated_sources,
        }
        try:
            sig = inspect.signature(fn)
            if "used_queries" in sig.parameters:
                kwargs["used_queries"] = used_queries
        except (TypeError, ValueError):
            pass
        return fn(**kwargs)

    # ----------------------
    # Public API
    # ----------------------
    def run_legal_search(
        self,
        user_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 4,
        max_iters: int = 2,
        user_context: str = "",
        *,
        auth_token: Optional[str] = None,  # ✅ USER JWT (con o sin Bearer; se normaliza en MCPClient)
    ) -> Dict[str, Any]:
        """
        API pública del orquestador.

        Multiusuario (compatible con gateway MCP):
        - self.mcp_registry_token / self.mcp_executor_token se usan como S2S en Authorization.
        - auth_token (usuario) se envía como X-User-Authorization (NO sustituye S2S).
        """
        user_prompt = (user_prompt or "").strip()
        if not user_prompt:
            return {
                "final_answer": "No se ha proporcionado un texto de consulta válido.",
                "sources": [],
                "normalized_queries": [],
                "tool_trace": [],
                "agent_steps": [],
                "tool_events": [],
                "plan_meta": {},
            }

        # Tools discovery (con propagación de user token si tu MCPClient/list_tools lo soporta)
        if not self.available_tools or (time.time() - self._tools_last_refresh) > self._tools_ttl_sec:
            self._refresh_tools(force=True, user_auth_token=auth_token)
            self._validate_tool_names()

        # parámetros defensivos
        try:
            max_total_sources = int(os.getenv("LEGALSEARCH_MAX_SOURCES_TOTAL", "12"))
        except ValueError:
            max_total_sources = 12
        max_total_sources = max(6, max_total_sources)

        safe_top_k = max(1, min(8, int(top_k or 4)))
        safe_max_iters = max(1, min(6, int(max_iters or 1)))

        agent_steps: List[Dict[str, Any]] = []
        all_tool_traces: List[Dict[str, Any]] = []
        all_tool_events: List[Dict[str, Any]] = []

        # 1) Planner
        history_str = self._format_history(history or [])
        planner_desc = build_legal_planner_prompt(
            history_str=history_str,
            user_prompt=user_prompt,
            extra_context=(user_context or "")[:6000],
        )

        t0 = perf_counter()
        planner_raw = self._run_single_agent_task(
            self.planner_agent,
            planner_desc,
            expected_output=(
                "Devuelve preferentemente JSON con keys: jurisdictions/topics/queries/source_priorities. "
                "Si no puedes, devuelve 1 a 4 consultas, una por línea, sin comentarios."
            ),
        )
        agent_steps.append({"step": "planner", "duration": perf_counter() - t0})

        norm_queries, plan_meta = self._parse_planner_output(planner_raw, user_prompt)
        norm_queries = [(q or "").strip() for q in norm_queries if (q or "").strip()] or [user_prompt]

        # dedupe queries
        seen_q = set()
        uniq_queries: List[str] = []
        for q in norm_queries:
            if q in seen_q:
                continue
            seen_q.add(q)
            uniq_queries.append(q)
        norm_queries = uniq_queries[:4]

        normalized_queries_acc: List[str] = list(norm_queries)
        all_sources: List[Dict[str, Any]] = []

        # 2) Iteraciones tools + analyst
        for it in range(safe_max_iters):
            logger.info(
                "[LEGALSEARCH] Iter %d/%d queries=%d | search_tool=%s | fetch_tool=%s",
                it + 1, safe_max_iters, len(norm_queries), self.search_tool_name, self.fetch_tool_name
            )

            t_it = perf_counter()
            exec_out = self._execute_tools_once(
                norm_queries,
                top_k=safe_top_k,
                user_auth_token=auth_token,
            )
            agent_steps.append({"step": f"tools_iter_{it+1}", "duration": perf_counter() - t_it})

            iter_sources = exec_out.get("sources") or []
            iter_trace = exec_out.get("tool_trace") or []
            iter_events = exec_out.get("tool_events") or []

            if iter_trace:
                all_tool_traces.extend(iter_trace)
            if iter_events:
                all_tool_events.extend(iter_events)

            if iter_sources:
                all_sources.extend(iter_sources)
                all_sources = self._dedupe_sources(all_sources)

            if len(all_sources) > max_total_sources:
                all_sources = all_sources[:max_total_sources]

            # Stop temprano si no hay progreso
            if it > 0 and not iter_sources:
                logger.info("[LEGALSEARCH] Sin nuevas fuentes en iteración %d; deteniendo.", it + 1)
                break

            # Analyst decide
            analyst_desc = self._build_analyst_desc(
                user_prompt=user_prompt,
                sources=iter_sources,
                accumulated_sources=all_sources,
                used_queries=normalized_queries_acc,
            )

            t_a = perf_counter()
            analyst_raw = self._run_single_agent_task(
                self.analyst_agent,
                analyst_desc,
                expected_output=(
                    "Devuelve JSON: {\"decision\":\"BASTA\"} o {\"queries\":[\"q1\",\"q2\"]}. "
                    "Si no JSON: nuevas consultas (1-3 líneas) o BASTA."
                ),
            )
            agent_steps.append({"step": f"analyst_iter_{it+1}", "duration": perf_counter() - t_a})

            if not analyst_raw:
                break

            parsed = self._parse_analyst_output(analyst_raw)
            if parsed.get("stop") or it == safe_max_iters - 1:
                break

            new_queries = [(q or "").strip() for q in (parsed.get("queries") or []) if (q or "").strip()]
            new_queries = new_queries[:3]
            used_set = set(normalized_queries_acc)
            new_queries = [q for q in new_queries if q not in used_set]

            if not new_queries:
                break

            normalized_queries_acc.extend(new_queries)
            norm_queries = new_queries

        # 3) Writer
        writer_desc = build_legal_writer_prompt(
            user_prompt=user_prompt,
            sources=all_sources,
            extra_context=(user_context or "")[:8000],
        )

        t_w = perf_counter()
        candidate = clean_llm_text(
            self._run_single_agent_task(
                self.writer_agent,
                writer_desc,
                expected_output="Memo legal con citas [n] y sección Fuentes final."
            )
        )
        agent_steps.append({"step": "writer", "duration": perf_counter() - t_w})

        if not candidate:
            fallback_sources = "\n".join(
                f"[{i}] {(s.get('title') or 'Fuente')} — {s.get('url')}"
                for i, s in enumerate(all_sources, start=1)
                if (s.get("url") or "").strip()
            )
            fallback = (
                "No he podido generar un memo fiable en este momento. "
                "Puede haber evidencia insuficiente o un problema temporal con el modelo.\n\n"
                "Fuentes:\n"
                f"{fallback_sources or '[1] (no se pudieron recuperar URLs válidas)'}"
            )
            return {
                "final_answer": fallback,
                "sources": all_sources,
                "normalized_queries": normalized_queries_acc,
                "tool_trace": all_tool_traces if self.enable_tool_traces else [],
                "agent_steps": agent_steps,
                "tool_events": all_tool_events,
                "plan_meta": plan_meta or {},
            }

        # 4) Reviewer
        reviewer_desc = build_legal_verifier_prompt(user_prompt=user_prompt, answer=candidate, sources=all_sources)

        t_r = perf_counter()
        reviewed = clean_llm_text(
            self._run_single_agent_task(
                self.reviewer_agent,
                reviewer_desc,
                expected_output="Versión final verificada contra fuentes."
            )
        )
        agent_steps.append({"step": "reviewer", "duration": perf_counter() - t_r})

        final_answer = reviewed or candidate

        return {
            "final_answer": final_answer,
            "sources": all_sources,
            "normalized_queries": normalized_queries_acc,
            "tool_trace": all_tool_traces if self.enable_tool_traces else [],
            "agent_steps": agent_steps,
            "tool_events": all_tool_events,
            "plan_meta": plan_meta or {},
        }

    # ----------------------
    # Utils internos
    # ----------------------
    def _format_history(self, history: List[Dict[str, Any]], max_turns: int = 6) -> str:
        if not history:
            return "No hay historial previo relevante."
        entries = history[-max_turns:]
        chunks: List[str] = []
        for item in entries:
            u = (item.get("prompt") or item.get("user") or "").strip()
            b = (item.get("response") or item.get("assistant") or "").strip()
            if u or b:
                chunks.append(f"Usuario: {u}\nAsistente: {b}")
        return "\n\n".join(chunks) if chunks else "No hay historial previo relevante."

    def _split_lines(self, block: str) -> List[str]:
        if not block:
            return []
        lines = [re.sub(r"^[\-\*\d\)\.]\s*", "", l).strip() for l in block.splitlines()]
        return [l for l in lines if l]

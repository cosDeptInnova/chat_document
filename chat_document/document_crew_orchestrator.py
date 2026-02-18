# document_crew_orchestrator.py

import os
import logging
from typing import List, Dict, Any, Optional
import json
from crewai import Agent, Task, Crew, Process, LLM  # type: ignore[import]
import threading
import time
import uuid

from utils import _safe_len
from document_crew_src.agents import (
    create_doc_analyst_agent,
    create_doc_writer_agent,
    create_doc_planner_agent,
    create_doc_reviewer_agent,
)
from document_crew_src.utils import clean_llm_text
from document_crew_src.prompts import (
    build_doc_analyst_prompt,
    build_doc_writer_prompt,
    build_doc_planner_prompt,
    build_doc_verifier_prompt,
)
from mcp_client import MCPClient  

logger = logging.getLogger(__name__)


class DocumentCrewOrchestrator:
    """
    Orquestador CrewAI específico para chat con un documento.

    - Planner / rewriter: normaliza la consulta para el índice efímero.
    - Analista: interpreta fragmentos + pregunta normalizada.
    - Redactor: genera respuesta legible para negocio.
    - Revisor: comprueba que la respuesta está soportada por el documento.
    """

    def __init__(
        self,
        llm_doc_analyst: Optional[LLM] = None,
        llm_doc_writer: Optional[LLM] = None,
    ) -> None:
        default_base = "http://127.0.0.1:8090/api/v1"
        base_url = (os.getenv("CREW_BASE_URL", default_base) or default_base).rstrip("/")

        default_model = os.getenv(
            "CHATDOC_MODEL_NAME",
            os.getenv("CREW_MODEL_NAME", "Llama3_8B_Cosmos"),
        )
        api_key = os.getenv("CREW_API_KEY", "dummy-local-key")
        temperature = float(
            os.getenv("CHATDOC_TEMPERATURE", os.getenv("CREW_TEMPERATURE", "0.2"))
        )
        verbose = bool(int(os.getenv("CREW_VERBOSE", "0")))

        logger.info(
            "[ChatDocCrew] Inicializando LLM base: base_url=%s, model=%s",
            base_url,
            default_model,
        )

        llm_default = LLM(
            model=default_model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

        self.llm_doc_analyst: LLM = llm_doc_analyst or llm_default
        self.llm_doc_writer: LLM = llm_doc_writer or llm_default
        self.llm: LLM = self.llm_doc_writer

        # --- Agentes ---
        self.doc_planner: Agent = create_doc_planner_agent(self.llm_doc_analyst, verbose)
        self.doc_analyst: Agent = create_doc_analyst_agent(self.llm_doc_analyst, verbose)
        self.doc_writer: Agent = create_doc_writer_agent(self.llm_doc_writer, verbose)
        self.doc_reviewer: Agent = create_doc_reviewer_agent(self.llm_doc_writer, verbose)

        # --- MCP client (bus real) ---
        self.mcp = MCPClient()

        # --- Autodescubrimiento de tools ---
        self.crew_name = os.getenv("CHATDOC_MCP_CREW_NAME", "document_crew")

        self.chatdoc_query_tool_name = os.getenv("CHATDOC_QUERY_TOOL_NAME", "chatdoc_query_tool")
        self.chatdoc_summary_tool_name = os.getenv("CHATDOC_SUMMARY_TOOL_NAME", "chatdoc_summary_tool")
        self.chatdoc_ingest_tool_name = os.getenv("CHATDOC_INGEST_TOOL_NAME", "chatdoc_ingest_tool")

        self.strict_tools = os.getenv("CHATDOC_STRICT_TOOLS", "1") == "1"

        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self._tools_lock = threading.RLock()
        self._tools_last_refresh = 0.0
        self._tools_ttl_sec = float(os.getenv("CHATDOC_TOOLS_TTL_SEC", "60"))

        self._tools_last_auth_fingerprint: Optional[str] = None

        try:
            crew_conc = int(os.getenv("CHATDOC_CREW_CONCURRENCY", "2"))
        except Exception:
            crew_conc = 2
        crew_conc = max(1, crew_conc)

        self._crew_sem = threading.BoundedSemaphore(value=crew_conc)

        # Timeout “soft” para adquirir (evita deadlocks si algo se queda colgado)
        try:
            self._crew_acquire_timeout_s = float(os.getenv("CHATDOC_CREW_ACQUIRE_TIMEOUT_S", "30"))
        except Exception:
            self._crew_acquire_timeout_s = 30.0


    def _safe_list_str(self, v: Any, max_items: int = 12) -> list[str]:
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for x in v:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
            if len(out) >= max_items:
                break
        return out

    def _dedupe_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicación estable por texto normalizado.
        Mantiene el chunk con mejor score si hay duplicados.
        """
        import hashlib
        best: Dict[str, Dict[str, Any]] = {}

        for ch in chunks or []:
            if not isinstance(ch, dict):
                continue
            txt = (ch.get("text") or ch.get("chunk_text") or ch.get("content") or "").strip()
            if not txt:
                continue

            key = hashlib.sha1(txt.encode("utf-8", errors="ignore")).hexdigest()
            score = ch.get("score")
            try:
                s = float(score) if score is not None else None
            except Exception:
                s = None

            if key not in best:
                best[key] = ch
                continue

            prev = best[key].get("score")
            try:
                p = float(prev) if prev is not None else None
            except Exception:
                p = None

            # Mantener el mejor score si ambos existen
            if s is not None and (p is None or s > p):
                best[key] = ch

        return list(best.values())

    def _fetch_context_multiquery_via_mcp(
        self,
        *,
        document_id: str,
        queries: List[str],
        top_k: int,
        min_score: float,
        window: int,
        access_token: Optional[str],
        mcp_auth_token: Optional[str],
        max_variants: int = 6,
        per_query_topk: int = 3,
        global_cap: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Multi-pass retrieval:
        - ejecuta varias queries (variants)
        - une resultados
        - dedupe
        - ordena por score y recorta
        """
        variants = [q.strip() for q in (queries or []) if isinstance(q, str) and q.strip()]
        if not variants:
            return []

        variants = variants[: max(1, int(max_variants))]
        per_k = max(1, int(per_query_topk))

        all_chunks: List[Dict[str, Any]] = []

        for q in variants:
            out = self._fetch_context_via_mcp(
                document_id=document_id,
                normalized_prompt=q,
                mode="qa",
                detail_level=None,
                summary_profile=None,
                top_k=min(per_k, max(1, int(top_k))),
                min_score=float(min_score or 0.0),
                window=max(0, int(window)),
                access_token=access_token,
                mcp_auth_token=mcp_auth_token,
            ) or {}

            chunks = out.get("chunks") or []
            if isinstance(chunks, list) and chunks:
                # anotamos de dónde viene (útil para debugging interno; no se muestra al usuario)
                for ch in chunks:
                    if isinstance(ch, dict):
                        ch.setdefault("_retrieved_by_query", q[:160])
                all_chunks.extend([c for c in chunks if isinstance(c, dict)])

        merged = self._dedupe_chunks(all_chunks)

        def _score_key(x: Dict[str, Any]) -> float:
            s = x.get("score")
            try:
                return float(s) if s is not None else 0.0
            except Exception:
                return 0.0

        merged = sorted(merged, key=_score_key, reverse=True)
        cap = max(1, int(global_cap))
        return merged[:cap]

    def _unwrap_tool_response(self, obj: Any) -> Any:
        """
        Unwrap robusto para resultados de tools MCP.

        Soporta:
        - JSON-RPC envelope: {"jsonrpc":..., "id":..., "result": {...}}
        - wrappers simples: {"data": ...} / {"result": ...} (única clave)
        - respuestas estilo MCP content/text
        - objetos SDK con atributos .data/.result/.content/.value
        """
        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool, list)):
            return obj

        if isinstance(obj, dict):
            # 1) JSON-RPC envelope
            if "result" in obj and ("jsonrpc" in obj or "id" in obj):
                return self._unwrap_tool_response(obj.get("result"))

            # 2) wrappers simples
            if "data" in obj and len(obj.keys()) == 1:
                return self._unwrap_tool_response(obj.get("data"))
            if "result" in obj and len(obj.keys()) == 1:
                return self._unwrap_tool_response(obj.get("result"))

            # 3) MCP "content" -> a veces viene un JSON string
            content = obj.get("content")
            if isinstance(content, list) and content:
                # caso típico: [{"type":"text","text":"{...json...}"}]
                first = content[0]
                if isinstance(first, dict):
                    text = first.get("text")
                    if isinstance(text, str) and text.strip():
                        txt = text.strip()
                        try:
                            parsed = json.loads(txt)
                            return self._unwrap_tool_response(parsed)
                        except Exception:
                            # no era JSON, devolvemos el texto
                            return txt

            return obj

        # 4) objetos SDK
        for attr in ("data", "result", "content", "value"):
            try:
                v = getattr(obj, attr, None)
                if v is not None:
                    return self._unwrap_tool_response(v)
            except Exception:
                pass

        return obj
    
    def _redact_token(self, token: Optional[str]) -> str:
        if not token:
            return "(none)"
        t = str(token).strip()
        if not t:
            return "(none)"
        if t.lower().startswith("bearer "):
            t = t[7:].strip()
        if len(t) <= 10:
            return "***"
        return f"{t[:4]}...{t[-4:]}"
    
    def _normalize_bearer_token(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        t = str(token).strip()
        if not t:
            return None
        # Evitar "Bearer Bearer ..."
        return t[7:].strip() if t.lower().startswith("bearer ") else t

    def _resolve_mcp_gateway_token(
        self,
        *,
        mcp_auth_token: Optional[str],
        access_token: Optional[str],
    ) -> Optional[str]:
        """
        Resuelve el token para Authorization contra el gateway MCP.

        Política segura:
        1) mcp_auth_token explícito
        2) token S2S de entorno
        3) NO usar access_token del usuario por defecto (evita 401 y fuga de intención)
        salvo que se habilite explícitamente con env.
        """
        env_tok = (
            os.getenv("CHATDOC_MCP_SERVICE_TOKEN")
            or os.getenv("MCP_SERVICE_TOKEN")
            or os.getenv("MCP_AUTH_TOKEN")
        )

        tok = mcp_auth_token or env_tok

        allow_user_fallback = os.getenv("CHATDOC_ALLOW_USER_TOKEN_FOR_MCP", "0") == "1"
        if not tok and allow_user_fallback:
            tok = access_token

        tok = self._normalize_bearer_token(tok)
        return tok

    def _auth_fingerprint(self, token: Optional[str]) -> Optional[str]:
        """
        No guardamos el token; guardamos un fingerprint estable para invalidar cache.
        """
        tok = self._normalize_bearer_token(token)
        if not tok:
            return None
        try:
            import hashlib
            return hashlib.sha256(tok.encode("utf-8")).hexdigest()
        except Exception:
            # fallback ligero
            return tok[:12]
    
    def _refresh_tools(self, *, force: bool = False, auth_token: Optional[str] = None) -> None:
        """
        Refresca lista de tools desde el bus MCP.

        Mejoras:
        - invalidación correcta del cache cuando cambia el auth (incluye None)
        - logs de conteos por crew vs global
        """
        now = time.time()

        with self._tools_lock:
            fp = self._auth_fingerprint(auth_token)

            # ✅ Fix crítico: también detecta cambio a None
            auth_changed = fp != self._tools_last_auth_fingerprint

            if (
                not force
                and not auth_changed
                and self.available_tools
                and (now - self._tools_last_refresh) < self._tools_ttl_sec
            ):
                logger.debug(
                    "[CHATDOC] _refresh_tools cache-hit ttl_ok=True auth_changed=False "
                    "(crew=%s tools=%d)",
                    self.crew_name,
                    len(self.available_tools),
                )
                return

            logger.info(
                "[CHATDOC] _refresh_tools iniciando (force=%s auth_changed=%s crew=%s auth=%s)",
                force,
                auth_changed,
                self.crew_name,
                self._redact_token(auth_token),
            )

            try:
                # 1) Intento filtrado por crew
                tools_by_crew = self.mcp.list_tools(crew=self.crew_name, auth_token=auth_token) or []
                logger.info(
                    "[CHATDOC] list_tools(crew=%s) → %d tools",
                    self.crew_name,
                    len(tools_by_crew),
                )

                tools = tools_by_crew

                # 2) Fallback global
                if not tools:
                    tools_global = self.mcp.list_tools(auth_token=auth_token) or []
                    logger.info(
                        "[CHATDOC] list_tools(global) → %d tools",
                        len(tools_global),
                    )
                    tools = tools_global

                self.available_tools = {
                    t.get("name"): t
                    for t in tools
                    if isinstance(t, dict) and t.get("name")
                }

                self._tools_last_refresh = now
                self._tools_last_auth_fingerprint = fp

                logger.info(
                    "[CHATDOC] Tools MCP cargadas para crew=%s: %s",
                    self.crew_name,
                    list(self.available_tools.keys()) or "(ninguna)"
                )

            except Exception as e:
                logger.warning(
                    "[CHATDOC] No se pudo refrescar tools MCP para crew=%s: %s",
                    self.crew_name, e
                )
                if not self.available_tools:
                    self.available_tools = {}

    def _normalize_query_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normaliza resultados de herramientas tipo chatdoc_query_tool.

        Robusto frente a:
        - envelopes residuales con 'result'
        - claves alternativas: results | chunks | fragments
        - text | chunk_text
        """
        if not isinstance(data, dict):
            return []

        # Tolerancia extra por si algo llega aún envuelto
        if "result" in data and isinstance(data.get("result"), dict):
            data = data["result"]

        results = data.get("results") or data.get("chunks") or data.get("fragments") or []
        if not isinstance(results, list):
            return []

        out: List[Dict[str, Any]] = []
        skipped_no_text = 0

        for r in results:
            if not isinstance(r, dict):
                continue

            text = (r.get("text") or r.get("chunk_text") or r.get("content") or "").strip()
            if not text:
                skipped_no_text += 1
                continue

            meta = r.get("metadata") or r.get("meta") or {}
            if isinstance(meta, dict) and "page" in meta and "page" not in r:
                r["page"] = meta.get("page")

            out.append(r)

        # Logging útil
        logger.info(
            "[CHATDOC] normalize_query_results in=%d out=%d skipped_no_text=%d",
            len(results), len(out), skipped_no_text
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[CHATDOC] normalize_query_results keys=%s",
                list(data.keys())
            )

        return out

    async def _ainvoke_tool_safe(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        *,
        auth_token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Wrapper async con soporte de auth gateway MCP.
        """
        try:
            return await self.mcp.ainvoke_tool(tool_name, payload, auth_token=auth_token, timeout=timeout)
        except TypeError:
            return await self.mcp.ainvoke_tool(tool_name, payload)

    def _invoke_tool_safe(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        *,
        auth_token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Wrapper sync robusto:
        - soporta firmas antiguas de MCPClient
        - aplica unwrap local SIEMPRE
        - logging de entrada/salida
        """
        try:
            raw = self.mcp.invoke_tool(
                tool_name,
                payload,
                auth_token=auth_token,
                timeout=timeout,
            )
        except TypeError:
            raw = self.mcp.invoke_tool(tool_name, payload)

        data = self._unwrap_tool_response(raw)

        logger.info(
            "[CHATDOC_MCP] invoke tool=%s auth=%s timeout=%s payload_keys=%s out_type=%s",
            tool_name,
            self._redact_token(auth_token),
            timeout,
            list(payload.keys()) if isinstance(payload, dict) else [],
            type(data).__name__,
        )

        if logger.isEnabledFor(logging.DEBUG):
            try:
                # cuidado con logs enormes
                preview = str(data)
                logger.debug(
                    "[CHATDOC_MCP] tool=%s out_preview(600)=%r",
                    tool_name,
                    preview[:600],
                )
            except Exception:
                pass

        return data

    def _ensure_tools_ready(self, *, auth_token: Optional[str] = None) -> None:
        """
        Garantiza que:
        - hay tools cargadas
        - los nombres canónicos de chatdoc_* están bien resueltos
        - el cache se invalida correctamente cuando cambia el auth
        """
        # Siempre validamos nombres en base al estado actual,
        # incluso si no refrescamos.
        if not self.available_tools:
            self._refresh_tools(auth_token=auth_token)
            self._validate_tool_names()
            return

        needed = {self.chatdoc_query_tool_name, self.chatdoc_summary_tool_name}
        missing = [n for n in needed if n not in self.available_tools]

        if missing:
            logger.info(
                "[CHATDOC] _ensure_tools_ready detectó tools faltantes=%s → refresh",
                missing,
            )
            self._refresh_tools(auth_token=auth_token)
            self._validate_tool_names()
            return

        # Si no faltan, igual validamos por si tenemos env names raros
        # y existen canónicas mejores en available.
        self._validate_tool_names()

    def _fetch_context_via_mcp(
        self,
        *,
        document_id: str,
        normalized_prompt: str,
        mode: str,
        detail_level: Optional[str] = None,
        summary_profile: Optional[str] = None,
        top_k: int = 8,
        min_score: float = 0.0,
        window: int = 1,
        access_token: Optional[str] = None,
        mcp_auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Obtiene fragments/chunks vía tools MCP.

        Fix clave:
        - Pedir explícitamente return_format="json" a query/summary tools.
        - Aceptar también el campo 'text' en summary cuando el endpoint
        no provea pre_summary_text.
        - Fallback robusto si alguna implementación devuelve str.
        """
        gateway_tok = self._resolve_mcp_gateway_token(
            mcp_auth_token=mcp_auth_token,
            access_token=access_token,
        )
        user_tok = self._normalize_bearer_token(access_token)

        self._ensure_tools_ready(auth_token=gateway_tok)

        mode_norm = (mode or "qa").lower().strip()
        if mode_norm not in ("qa", "summary"):
            mode_norm = "qa"

        try:
            timeout = float(os.getenv("CHATDOC_NLP_QUERY_TIMEOUT_S", "60"))
        except Exception:
            timeout = 60.0

        # ----------------
        # SUMMARY
        # ----------------
        if mode_norm == "summary":
            payload: Dict[str, Any] = {
                "document_id": document_id,
                "strategy": os.getenv("CHATDOC_SUMMARY_STRATEGY", "hybrid"),
                "access_token": user_tok,
                # ✅ IMPORTANTÍSIMO: forzar JSON
                "return_format": "json",
            }
            if detail_level:
                payload["detail_level"] = detail_level
            if summary_profile:
                payload["summary_profile"] = summary_profile

            data = self._invoke_tool_safe(
                self.chatdoc_summary_tool_name,
                payload,
                auth_token=gateway_tok,
                timeout=timeout,
            ) or {}

            # ✅ Fallback si por alguna razón llega texto
            if isinstance(data, str):
                txt = data.strip()
                data = {
                    "status": "ok",
                    "pre_summary_text": txt,
                    "fragments": [],
                    "text": txt,
                }
            elif not isinstance(data, dict):
                data = {}

            # ✅ aceptar múltiples claves de items
            raw_items = (
                data.get("fragments")
                or data.get("results")
                or data.get("chunks")
                or []
            )
            if not isinstance(raw_items, list):
                raw_items = []

            # ✅ normalizamos usando la misma rutina QA
            fragments = self._normalize_query_results({"results": raw_items})

            # ✅ aceptar también 'text' cuando return_format=json
            pre_summary_text = (
                data.get("pre_summary_text")
                or data.get("summary")
                or data.get("pre_summary")
                or data.get("text")
            )

            logger.info(
                "[CHATDOC_FETCH] summary doc_id=%s status=%s raw_items=%d norm_items=%d has_pre_summary=%s",
                document_id,
                data.get("status", "ok"),
                len(raw_items),
                len(fragments),
                bool(isinstance(pre_summary_text, str) and pre_summary_text.strip()),
            )

            return {
                "status": data.get("status", "ok"),
                "chunks": fragments,
                "summary": pre_summary_text,
                "raw": data,
            }

        # ----------------
        # QA
        # ----------------
        payload = {
            "document_id": document_id,
            "query": normalized_prompt,
            "top_k": top_k,
            "min_score": min_score,
            "window": window,
            "access_token": user_tok,
            # ✅ IMPORTANTÍSIMO: forzar JSON
            "return_format": "json",
        }

        data = self._invoke_tool_safe(
            self.chatdoc_query_tool_name,
            payload,
            auth_token=gateway_tok,
            timeout=timeout,
        ) or {}

        # ✅ Fallback si por alguna razón llega texto
        if isinstance(data, str):
            txt = data.strip()

            # Intento best-effort: si parece JSON, parsearlo
            if txt.startswith("{") and txt.endswith("}"):
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict):
                        data = parsed
                    else:
                        data = {}
                except Exception:
                    data = {}
            else:
                # Si es texto plano, lo convertimos a un "chunk" único
                chunks_txt = []
                if txt and "Sin resultados" not in txt:
                    chunks_txt = [{"text": txt}]
                logger.info(
                    "[CHATDOC_FETCH] qa doc_id=%s status=%s (text-fallback) chunks=%d",
                    document_id,
                    "ok",
                    len(chunks_txt),
                )
                return {
                    "status": "ok",
                    "chunks": chunks_txt,
                    "summary": None,
                    "raw": {"text": txt},
                }

        if not isinstance(data, dict):
            data = {}

        chunks = self._normalize_query_results(data)

        logger.info(
            "[CHATDOC_FETCH] qa doc_id=%s status=%s norm_chunks=%d",
            document_id,
            data.get("status", "ok"),
            len(chunks),
        )

        return {
            "status": data.get("status", "ok"),
            "chunks": chunks,
            "summary": None,
            "raw": data,
        }

    def _build_doc_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for idx, ch in enumerate(chunks, start=1):
            if not isinstance(ch, dict):
                continue

            text_chunk = (
                ch.get("text")
                or ch.get("chunk_text")
                or ch.get("content")
                or ch.get("snippet")
                or ""
            ).strip()

            if not text_chunk:
                continue

            page = ch.get("page")
            score = ch.get("score")

            header = f"[Fragmento {idx}"
            if page is not None:
                header += f" - pág {page}"
            if score is not None:
                try:
                    header += f" - score {float(score):.3f}"
                except Exception:
                    pass
            header += "]"

            parts.append(f"{header}\n{text_chunk}")

        return "\n\n".join(parts)


    def _auto_pick_tool(self, kind: str) -> Optional[str]:
        """
        Selección por tags/nombre. kind: query|summary|ingest
        """
        if not self.available_tools:
            return None

        k = (kind or "").lower().strip()
        if k not in ("query", "summary", "ingest"):
            return None

        # Preferencias por tags
        preferred = {
            "query": [{"chatdoc", "query"}, {"document", "query"}, {"rag", "query"}],
            "summary": [{"chatdoc", "summary"}, {"document", "summary"}, {"rag", "summary"}],
            "ingest": [{"chatdoc", "ingest"}, {"document", "ingest"}],
        }

        for tag_set in preferred.get(k, []):
            for name, meta in self.available_tools.items():
                tags = meta.get("tags") or []
                if not isinstance(tags, list):
                    continue
                tag_norm = {str(t).strip().lower() for t in tags if str(t).strip()}
                if tag_set.issubset(tag_norm):
                    return name

        # Fallback por nombre
        for name in self.available_tools.keys():
            ln = name.lower()
            if k == "query" and "query" in ln and "chatdoc" in ln:
                return name
            if k == "summary" and "summary" in ln and "chatdoc" in ln:
                return name
            if k == "ingest" and "ingest" in ln and "chatdoc" in ln:
                return name

        return None

    def _validate_tool_names(self) -> None:
        with self._tools_lock:
            if not self.available_tools:
                return

            available = set(self.available_tools.keys())

            if self.strict_tools:
                # Si existen las tools canónicas, fijarlas
                if "chatdoc_query_tool" in available:
                    self.chatdoc_query_tool_name = "chatdoc_query_tool"
                if "chatdoc_summary_tool" in available:
                    self.chatdoc_summary_tool_name = "chatdoc_summary_tool"
                if "chatdoc_ingest_tool" in available:
                    self.chatdoc_ingest_tool_name = "chatdoc_ingest_tool"

                # Si no existen, mantener env, pero advertir
                if self.chatdoc_query_tool_name not in available:
                    logger.warning(
                        "[CHATDOC] strict_tools activo pero query_tool '%s' no disponible. Disponibles=%s",
                        self.chatdoc_query_tool_name, list(available)
                    )
                if self.chatdoc_summary_tool_name not in available:
                    logger.warning(
                        "[CHATDOC] strict_tools activo pero summary_tool '%s' no disponible. Disponibles=%s",
                        self.chatdoc_summary_tool_name, list(available)
                    )
                return

            # No estricto: autopick
            if self.chatdoc_query_tool_name not in self.available_tools:
                self.chatdoc_query_tool_name = self._auto_pick_tool("query") or "chatdoc_query_tool"
            if self.chatdoc_summary_tool_name not in self.available_tools:
                self.chatdoc_summary_tool_name = self._auto_pick_tool("summary") or "chatdoc_summary_tool"
            if self.chatdoc_ingest_tool_name not in self.available_tools:
                self.chatdoc_ingest_tool_name = self._auto_pick_tool("ingest") or "chatdoc_ingest_tool"

    def _format_history(
        self,
        history: List[Dict[str, Any]],
        max_turns: int = 6,
    ) -> str:
        """
        Historial ligero de este documento en formato compacto.
        Espera una lista de dicts tipo:
          { "prompt": str, "response": str, ... }
        """
        if not history:
            return "No hay historial previo relevante."

        entries = history[-max_turns:]
        chunks: List[str] = []
        for item in entries:
            u = (item.get("prompt") or "").strip()
            b = (item.get("response") or "").strip()
            if u or b:
                chunks.append(f"Usuario: {u}\nAsistente: {b}")
        return "\n\n".join(chunks) if chunks else "No hay historial previo relevante."

    def _run_single_agent_task(
    self,
    agent: Agent,
    description: str,
    expected_output: str,
    clean_output: bool = True,
) -> str:
        """
        Helper genérico:
        - Crea Task + Crew con un solo agente.
        - Ejecuta kickoff() con Process.sequential.
        - DEVUELVE string.
        - Production: limitador de concurrencia (semáforo) para CrewAI.
        """
        import uuid

        task_id = uuid.uuid4().hex[:8]

        logger.info(
            "[CHATDOC_TASK %s] Iniciando task con agente='%s' (len_desc=%d)",
            task_id,
            agent.role,
            len(description),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[CHATDOC_TASK %s] Descripción (trunc 400):\n%s",
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

        acquired = False
        try:
            # --- Límite de concurrencia: evita ejecución simultánea no controlada ---
            try:
                acquired = self._crew_sem.acquire(timeout=getattr(self, "_crew_acquire_timeout_s", 30.0))
            except Exception:
                acquired = False

            if not acquired:
                logger.warning(
                    "[CHATDOC_TASK %s] No se pudo adquirir semáforo de Crew en %.1fs; devolviendo fallback vacío.",
                    task_id,
                    getattr(self, "_crew_acquire_timeout_s", 30.0),
                )
                return ""

            result = crew.kickoff()
            logger.info(
                "[CHATDOC_TASK %s] Task completado para agente='%s'",
                task_id,
                agent.role,
            )

        except Exception as exc:
            logger.exception(
                "[CHATDOC_TASK %s] Error ejecutando task con agente '%s': %s",
                task_id,
                agent.role,
                exc,
            )
            return ""

        finally:
            if acquired:
                try:
                    self._crew_sem.release()
                except Exception:
                    pass

        # --- normalización de salida ---
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
                "[CHATDOC_TASK %s] Salida bruta '%s' (trunc 800): %r",
                task_id,
                agent.role,
                out_str[:800],
            )

        if clean_output:
            out_str = clean_llm_text(out_str)

        return out_str


    def plan_doc_query(
        self,
        user_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        mode: str = "qa",
        doc_meta: Optional[Dict[str, Any]] = None,
        *,
        access_token: Optional[str] = None,
        mcp_auth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        history_str = self._format_history(history or [])
        doc_meta = doc_meta or {}

        document_id = (doc_meta.get("document_id") or "").strip()

        enable_preview = os.getenv("CHATDOC_PLANNER_PREVIEW", "1") == "1"
        doc_preview: Optional[str] = None

        if enable_preview and document_id:
            try:
                preview_out = self._fetch_context_via_mcp(
                    document_id=document_id,
                    normalized_prompt=user_prompt,
                    mode="summary",
                    detail_level="low",
                    summary_profile=None,
                    top_k=4,
                    min_score=0.0,
                    window=0,
                    access_token=access_token,
                    mcp_auth_token=mcp_auth_token,
                )
                doc_preview = preview_out.get("summary")
                if not isinstance(doc_preview, str) or not doc_preview.strip():
                    chunks = preview_out.get("chunks") or []
                    if isinstance(chunks, list) and chunks:
                        doc_preview = self._build_doc_context_from_chunks(chunks)
            except Exception:
                doc_preview = None

        description = build_doc_planner_prompt(
            mode=mode,
            history_str=history_str,
            user_prompt=user_prompt,
            doc_meta=doc_meta,
            doc_preview=doc_preview,
        )

        # salida JSON estricta
        description += """
    FORMATO DE SALIDA (OBLIGATORIO):
    - Devuelve UN ÚNICO objeto JSON válido.
    - No añadas texto antes ni después del JSON.
    - No lo envuelvas en markdown ni en fences.
    """

        raw_out = self._run_single_agent_task(
            agent=self.doc_planner,
            description=description,
            expected_output="Únicamente un objeto JSON con la estructura indicada, sin texto adicional.",
            clean_output=False,
        )

        raw_out = clean_llm_text(raw_out or "").strip()

        mode_norm = (mode or "qa").lower()
        if mode_norm not in ("qa", "summary"):
            mode_norm = "qa"

        try:
            plan_obj = json.loads(raw_out)
            if not isinstance(plan_obj, dict):
                raise ValueError("Planner devolvió un JSON que no es objeto.")
        except Exception:
            # fallback compatible
            return {
                "normalized_question": (raw_out or user_prompt),
                "mode": mode_norm,
                "detail_level": None,
                "query_type": None,
                "summary_profile": None,
            }

        normalized = (plan_obj.get("normalized_question") or "").strip() or user_prompt

        plan_mode = (plan_obj.get("mode") or mode_norm or "qa").lower()
        if plan_mode not in ("qa", "summary"):
            plan_mode = mode_norm

        detail_level = plan_obj.get("detail_level")
        if isinstance(detail_level, str):
            dl = detail_level.strip().lower()
            detail_level = dl if dl in ("low", "medium", "high") else None
        else:
            detail_level = None

        query_type = plan_obj.get("query_type")
        query_type = query_type.strip().lower() if isinstance(query_type, str) else None

        summary_profile = plan_obj.get("summary_profile")
        summary_profile = summary_profile.strip().lower() if isinstance(summary_profile, str) else None

        # ✅ NUEVO (opcional, no rompe):
        query_variants = self._safe_list_str(plan_obj.get("query_variants"), max_items=10)
        subquestions = self._safe_list_str(plan_obj.get("subquestions"), max_items=12)
        must_terms = self._safe_list_str(plan_obj.get("must_terms"), max_items=16)
        should_terms = self._safe_list_str(plan_obj.get("should_terms"), max_items=16)
        focus = self._safe_list_str(plan_obj.get("focus"), max_items=16)

        answer_depth = plan_obj.get("answer_depth")
        answer_depth = answer_depth.strip().lower() if isinstance(answer_depth, str) else None
        if answer_depth not in ("brief", "deep"):
            answer_depth = None

        retrieval_hints = plan_obj.get("retrieval_hints")
        if not isinstance(retrieval_hints, dict):
            retrieval_hints = {}

        # defaults defensivos
        retrieval_hints_norm = {
            "multiquery": bool(retrieval_hints.get("multiquery")) if retrieval_hints else False,
            "max_variants": int(retrieval_hints.get("max_variants") or 6),
            "per_query_topk": int(retrieval_hints.get("per_query_topk") or 3),
            "global_cap": int(retrieval_hints.get("global_cap") or 12),
        }
        retrieval_hints_norm["max_variants"] = max(1, min(retrieval_hints_norm["max_variants"], 10))
        retrieval_hints_norm["per_query_topk"] = max(1, min(retrieval_hints_norm["per_query_topk"], 8))
        retrieval_hints_norm["global_cap"] = max(3, min(retrieval_hints_norm["global_cap"], 32))

        plan = {
            "normalized_question": normalized,
            "mode": plan_mode,
            "detail_level": detail_level,
            "query_type": query_type,
            "summary_profile": summary_profile,

            # nuevos campos (opt)
            "query_variants": query_variants,
            "subquestions": subquestions,
            "must_terms": must_terms,
            "should_terms": should_terms,
            "focus": focus,
            "answer_depth": answer_depth,
            "retrieval_hints": retrieval_hints_norm,
        }

        return plan
    #Método principal
    def run_doc_chat(
    self,
    user_prompt: str,
    doc_context: str,
    history: Optional[List[Dict[str, Any]]] = None,
    mode: str = "qa",
    doc_meta: Optional[Dict[str, Any]] = None,
    precomputed_summary: Optional[str] = None,
    normalized_prompt: Optional[str] = None,
    plan: Optional[Dict[str, Any]] = None,
    verify_answer: bool = True,
    *,
    document_id: Optional[str] = None,
    access_token: Optional[str] = None,
    mcp_auth_token: Optional[str] = None,
    top_k: int = 8,
    min_score: float = 0.0,
    window: int = 1,
) -> str:
        import uuid
        import json

        chat_id = uuid.uuid4().hex[:8]

        mode_norm = (mode or "qa").lower().strip()
        if mode_norm not in ("qa", "summary"):
            mode_norm = "qa"

        try:
            max_history_chars = int(os.getenv("CHATDOC_MAX_HISTORY_CHARS", "3000"))
        except ValueError:
            max_history_chars = 3000

        try:
            max_doc_context_chars = int(os.getenv("CHATDOC_MAX_DOC_CONTEXT_CHARS", "12000"))
        except ValueError:
            max_doc_context_chars = 12000

        try:
            max_pre_summary_chars = int(os.getenv("CHATDOC_MAX_PRE_SUMMARY_CHARS", "4000"))
        except ValueError:
            max_pre_summary_chars = 4000

        raw_history_str = self._format_history(history or [])
        history_str = (
            raw_history_str[-max_history_chars:]
            if raw_history_str and len(raw_history_str) > max_history_chars
            else raw_history_str
        )

        doc_meta = doc_meta or {}
        document_id = (
            (document_id or "").strip()
            or str(doc_meta.get("document_id") or "").strip()
            or None
        )

        norm_prompt = (normalized_prompt or "").strip() or user_prompt
        doc_ctx = (doc_context or "").strip()

        used_fallback_summary = False

        # -------------------------
        # AUTOFETCH CONTEXTO (mejorado)
        # -------------------------
        if not doc_ctx and document_id:
            try:
                detail_level = None
                summary_profile = None
                query_variants: List[str] = []

                if isinstance(plan, dict):
                    detail_level = plan.get("detail_level")
                    summary_profile = plan.get("summary_profile")
                    qv = plan.get("query_variants")
                    if isinstance(qv, list):
                        query_variants = [str(x).strip() for x in qv if isinstance(x, str) and x.strip()]

                # Si QA: intentamos multiquery cuando esté disponible
                if mode_norm == "qa":
                    retrieval_hints = plan.get("retrieval_hints") if isinstance(plan, dict) else {}
                    if not isinstance(retrieval_hints, dict):
                        retrieval_hints = {}

                    multiquery = bool(retrieval_hints.get("multiquery")) or bool(query_variants)

                    if multiquery:
                        max_variants = int(retrieval_hints.get("max_variants") or 6)
                        per_query_topk = int(retrieval_hints.get("per_query_topk") or 3)
                        global_cap = int(retrieval_hints.get("global_cap") or max(8, int(top_k) + 4))

                        queries = query_variants or [norm_prompt]

                        chunks = self._fetch_context_multiquery_via_mcp(
                            document_id=document_id,
                            queries=queries,
                            top_k=top_k,
                            min_score=min_score,
                            window=window,
                            access_token=access_token,
                            mcp_auth_token=mcp_auth_token,
                            max_variants=max_variants,
                            per_query_topk=per_query_topk,
                            global_cap=global_cap,
                        )

                        if chunks:
                            doc_ctx = self._build_doc_context_from_chunks(chunks)
                    else:
                        fetched = self._fetch_context_via_mcp(
                            document_id=document_id,
                            normalized_prompt=norm_prompt,
                            mode="qa",
                            detail_level=detail_level,
                            summary_profile=summary_profile,
                            top_k=top_k,
                            min_score=min_score,
                            window=window,
                            access_token=access_token,
                            mcp_auth_token=mcp_auth_token,
                        ) or {}
                        chunks = fetched.get("chunks") or []
                        if isinstance(chunks, list) and chunks:
                            doc_ctx = self._build_doc_context_from_chunks(chunks)

                    # Salvavidas: si QA no recuperó nada, fallback a summary(low)
                    if not doc_ctx:
                        fb = self._fetch_context_via_mcp(
                            document_id=document_id,
                            normalized_prompt=norm_prompt,
                            mode="summary",
                            detail_level="low",
                            summary_profile=None,
                            top_k=min(int(top_k), 4),
                            min_score=0.0,
                            window=0,
                            access_token=access_token,
                            mcp_auth_token=mcp_auth_token,
                        ) or {}

                        fb_chunks = fb.get("chunks") or []
                        fb_summary = fb.get("summary")
                        used_fallback_summary = True

                        if isinstance(fb_chunks, list) and fb_chunks:
                            doc_ctx = self._build_doc_context_from_chunks(fb_chunks)
                        if not precomputed_summary and isinstance(fb_summary, str) and fb_summary.strip():
                            precomputed_summary = fb_summary.strip()

                # SUMMARY: vía summary tool (como ya hacías)
                else:
                    fetched = self._fetch_context_via_mcp(
                        document_id=document_id,
                        normalized_prompt=norm_prompt,
                        mode="summary",
                        detail_level=detail_level,
                        summary_profile=summary_profile,
                        top_k=top_k,
                        min_score=min_score,
                        window=window,
                        access_token=access_token,
                        mcp_auth_token=mcp_auth_token,
                    ) or {}

                    chunks = fetched.get("chunks") or []
                    summary_txt = fetched.get("summary")

                    if isinstance(chunks, list) and chunks:
                        doc_ctx = self._build_doc_context_from_chunks(chunks)

                    if not precomputed_summary and isinstance(summary_txt, str) and summary_txt.strip():
                        precomputed_summary = summary_txt.strip()

            except Exception:
                doc_ctx = ""

        # -------------------------
        # Recortes defensivos
        # -------------------------
        if doc_ctx and len(doc_ctx) > max_doc_context_chars:
            doc_ctx = doc_ctx[:max_doc_context_chars] + "\n\n… (contexto truncado)\n"

        if precomputed_summary and len(precomputed_summary) > max_pre_summary_chars:
            precomputed_summary = precomputed_summary[:max_pre_summary_chars]

        # Salvavidas final: si no hay contexto pero hay pre_summary, úsalo
        if not doc_ctx and precomputed_summary and isinstance(precomputed_summary, str):
            ps = precomputed_summary.strip()
            if ps:
                doc_ctx = "[Resumen preliminar disponible]\n" + ps

        # Metadatos enriquecidos (sin romper)
        base_meta = dict(doc_meta)
        if plan:
            base_meta["planner_plan"] = plan
            # también propagamos señales útiles al analista
            base_meta.setdefault("detail_level", plan.get("detail_level") if isinstance(plan, dict) else None)
            base_meta.setdefault("summary_profile", plan.get("summary_profile") if isinstance(plan, dict) else None)
            base_meta.setdefault("query_type", plan.get("query_type") if isinstance(plan, dict) else None)
        base_meta["autofetch_fallback_summary"] = used_fallback_summary
        doc_meta = base_meta

        # -------------------------
        # 1) Analista
        # -------------------------
        analyst_description = build_doc_analyst_prompt(
            mode=mode_norm,
            history_str=history_str,
            user_prompt=norm_prompt,
            doc_context=doc_ctx or "No hay fragmentos disponibles del documento.",
            doc_meta=doc_meta,
            precomputed_summary=precomputed_summary,
        )
        analyst_description = (
            f"[CHATDOC_ID {chat_id}] (etiqueta interna, NO la menciones)\n\n"
            + analyst_description
        )

        analyst_raw = self._run_single_agent_task(
            agent=self.doc_analyst,
            description=analyst_description,
            expected_output="Un informe estructurado mining-grade basado SOLO en el documento.",
        )

        # -------------------------
        # 2) Redactor
        # -------------------------
        writer_description = build_doc_writer_prompt(
            user_prompt=user_prompt,
            analyst_output=analyst_raw,
        )

        writer_output = self._run_single_agent_task(
            agent=self.doc_writer,
            description=writer_description,
            expected_output="Una única respuesta final en español para el usuario.",
        )

        candidate_answer = clean_llm_text(writer_output or analyst_raw)
        final_answer = candidate_answer

        # -------------------------
        # 3) Verificador (JSON)
        # -------------------------
        if verify_answer:
            verifier_description = build_doc_verifier_prompt(
                mode=mode_norm,
                user_prompt=user_prompt,
                normalized_prompt=norm_prompt,
                answer=candidate_answer,
                doc_context=doc_ctx or "No hay fragmentos disponibles del documento.",
                doc_meta=doc_meta,
                plan=plan,
            )

            verifier_description += (
                "\n\nINSTRUCCIONES DE SALIDA (MUY IMPORTANTE):\n"
                "Devuelve ÚNICAMENTE un JSON válido con este esquema:\n"
                "{\n"
                '  \"is_supported\": true o false,\n'
                '  \"final_answer\": \"texto de la mejor respuesta final para el usuario\"\n'
                "}\n"
                "No añadas texto fuera del JSON.\n"
            )

            reviewed_output_raw = self._run_single_agent_task(
                agent=self.doc_reviewer,
                description=verifier_description,
                expected_output="Únicamente el JSON final con is_supported y final_answer.",
                clean_output=False,
            )

            if reviewed_output_raw:
                raw_str = reviewed_output_raw.strip()

                if raw_str.startswith("```"):
                    first_nl = raw_str.find("\n")
                    if first_nl != -1:
                        raw_str = raw_str[first_nl + 1:]
                    if raw_str.endswith("```"):
                        raw_str = raw_str[:-3].strip()

                try:
                    reviewed_data = json.loads(raw_str)
                    if isinstance(reviewed_data, dict):
                        fa = reviewed_data.get("final_answer")
                        if isinstance(fa, str) and fa.strip():
                            final_answer = clean_llm_text(fa.strip())
                except Exception:
                    pass

        return final_answer or ""

from __future__ import annotations

import json
import logging
import re
import uuid
import inspect
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status, Query

from api.deps import get_llm_client
from models.inference import ChatCompletionRequest
from services.llm_client import LLMClient
from services.tools_registry import GLOBAL_TOOL_REGISTRY

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])


# -----------------------------
# Models
# -----------------------------
class ToolInvokeRequest(BaseModel):
    # Evita default mutable compartido entre requests
    args: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# Helpers de limpieza / JSON
# -----------------------------
def _clean_llm_text(text: str | None) -> str:
    """
    Limpia texto generado por el LLM antes de intentar extraer JSON:
    - recorta espacios
    - elimina fences ``` y ```json
    - quita encabezados típicos y disclaimers
    """
    if not text:
        return ""

    t = str(text).strip()

    # Quitar fences ```lang ... ```
    if t.startswith("```") and t.endswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_\-]*\s*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)

    # Quitar 'Respuesta:' / 'Assistant:' al inicio
    t = re.sub(
        r"^(respuesta:|respuesta\s*|assistant:|asistente:)\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )

    # Eliminar disclaimers típicos
    disclaimer_patterns = [
        r"como modelo de lenguaje[^.\n]*\.",
        r"como ia(?: de)? lenguaje[^.\n]*\.",
    ]
    for pat in disclaimer_patterns:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)

    # Colapsar más de 2 saltos de línea
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def _extract_json_from_text(raw: str) -> Dict[str, Any]:
    """
    Intenta extraer un objeto JSON desde un texto libre devuelto por el modelo.

    Estrategia robusta:
    - Limpia el texto (fences, disclaimers, cabeceras).
    - Busca la primera '{' y la última '}'.
    - Intenta hacer json.loads() del substring.
    - Si falla, lanza ValueError con un mensaje claro.
    """
    cleaned = _clean_llm_text(raw)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "No se han encontrado llaves JSON en el contenido generado "
            "tras la limpieza inicial."
        )

    candidate = cleaned[start: end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        preview = candidate[:500]
        raise ValueError(
            f"JSON inválido en la respuesta del modelo: {exc}. "
            f"Fragmento analizado (recortado): {preview!r}"
        ) from exc


# -----------------------------
# Normalización OpenAI-compatible
# -----------------------------
def _coerce_choice_to_message(choice: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asegura que cada choice tenga un dict message con role/content mínimos.
    Soporta variantes comunes en backends OpenAI-like.
    """
    if not isinstance(choice, dict):
        return {"message": {"role": "assistant", "content": ""}}

    msg = choice.get("message")

    # Algunos backends ponen 'text' (legacy)
    if not msg and isinstance(choice.get("text"), str):
        msg = {"role": "assistant", "content": choice.get("text")}

    # Algunos streams ponen 'delta' (aunque aquí no soportamos stream,
    # algunos servidores aún pueden incluir algo parecido)
    if not msg and isinstance(choice.get("delta"), dict):
        delta = choice.get("delta") or {}
        msg = {
            "role": delta.get("role") or "assistant",
            "content": delta.get("content") or "",
        }
        # Si hubiera tool_calls dentro delta
        if delta.get("tool_calls"):
            msg["tool_calls"] = delta.get("tool_calls")

    if not isinstance(msg, dict):
        msg = {"role": "assistant", "content": ""}

    # Default role
    if not msg.get("role"):
        msg["role"] = "assistant"

    # Garantiza que exista la clave content (aunque sea None por accidente)
    if "content" not in msg:
        msg["content"] = ""

    choice["message"] = msg
    return choice


def _ensure_non_empty_content_for_crewai(result: Dict[str, Any], req_id: str) -> Dict[str, Any]:
    """
    CrewAI (y algunos wrappers OpenAI) fallan si message.content es None o "".
    Esto es habitual cuando el modelo devuelve tool_calls.

    Esta función:
    - Normaliza cada choice a un message.
    - Si hay tool_calls y content vacío → inserta un sentinel corto.
    - Si no hay tool_calls y content None → lo convierte a "" (sin inventar texto).
    - Si choices vacío → lanza 502 para evitar un 200 engañoso.

    No toca la estructura de tool_calls ni las tools.
    """
    if not isinstance(result, dict):
        logger.error("[MCP_CHAT %s] Respuesta del modelo no es dict.", req_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Respuesta inválida del modelo (formato inesperado).",
        )

    choices = result.get("choices") or []
    if not isinstance(choices, list) or len(choices) == 0:
        logger.error("[MCP_CHAT %s] Respuesta sin choices o choices vacío.", req_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Respuesta inválida del modelo: 'choices' vacío.",
        )

    new_choices: List[Dict[str, Any]] = []
    for idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            choice = {}

        choice = _coerce_choice_to_message(choice)
        msg = choice.get("message") or {}

        content = msg.get("content")
        # tool_calls puede venir en varios sitios según backend
        tool_calls = msg.get("tool_calls") or choice.get("tool_calls")

        # Normaliza None -> ""
        if content is None:
            content = ""
            msg["content"] = ""

        # Caso crítico: tool_calls + content vacío
        if tool_calls and (not isinstance(content, str) or content.strip() == ""):
            # Sentinel mínimo para evitar crash del cliente
            msg["content"] = "Tool call requested."
            choice["message"] = msg

        new_choices.append(choice)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[MCP_CHAT %s] Choice %d normalizado: role=%s content_len=%d has_tool_calls=%s",
                req_id,
                idx,
                (msg.get("role") or "assistant"),
                len((msg.get("content") or "")),
                bool(tool_calls),
            )

    result["choices"] = new_choices
    return result


# -----------------------------
# Endpoint principal
# -----------------------------
@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    llama_client: LLMClient = Depends(get_llm_client),
) -> Dict[str, Any]:
    """
    Endpoint OpenAI-compatible para inferencias de chat.

    Ruta final: POST /api/v1/chat/completions

    - CrewAI (vía cliente `openai`) llamará aquí.
    - Este endpoint delega la inferencia en llama.cpp (ej. 8002/v1/chat/completions)
      usando `LLMClient`.

    Comportamiento especial:

    - Streaming (body.stream == True) NO está soportado: devolvemos HTTP 400.
    - Si `body.response_format == {"type": "json_object"}`:
        * Se intenta normalizar `choices[*].message.content` para que contenga
          únicamente un JSON válido (sin texto extra, sin ```json, etc.).
        * Si no se puede extraer un JSON correcto, devolvemos HTTP 500.
    """
    req_id = uuid.uuid4().hex[:8]

    # 1) Bloqueamos streaming por ahora (no implementado en este MCP).
    if body.stream:
        logger.warning(
            "[MCP_CHAT %s] Streaming solicitado pero no soportado. Rechazando con 400.",
            req_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming no soportado por el MCP /chat/completions.",
        )

    messages = body.messages or []
    msg_count = len(messages)
    model = getattr(body, "model", None)
    response_format = body.response_format or {}

    # Intentamos localizar un CHAT_ID de COSMOS
    chat_id_from_prompt = None
    try:
        for m in messages:
            content = getattr(m, "content", "") or ""
            match = re.search(r"\[CHAT_ID\s+([0-9a-fA-F]{8})\]", content)
            if match:
                chat_id_from_prompt = match.group(1)
    except Exception:
        chat_id_from_prompt = None

    last_user_msg = None
    for m in reversed(messages):
        if getattr(m, "role", None) == "user":
            last_user_msg = (getattr(m, "content", "") or "")[:400]
            break

    logger.info(
        "[MCP_CHAT %s] /chat/completions recibido: model=%s, messages=%d, "
        "response_format=%s, stream=%s, chat_id=%s",
        req_id,
        model,
        msg_count,
        response_format or None,
        body.stream,
        chat_id_from_prompt,
    )

    if last_user_msg and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[MCP_CHAT %s] Último mensaje de usuario (truncado 400 chars): %r",
            req_id,
            last_user_msg,
        )

    # 2) Llamada al llama-server a través de LLMClient
    try:
        logger.info("[MCP_CHAT %s] Llamando a llama_client.chat_completion()", req_id)
        result = await llama_client.chat_completion(body)
        logger.info(
            "[MCP_CHAT %s] Respuesta de llama_client.chat_completion recibida correctamente",
            req_id,
        )
    except HTTPException:
        logger.exception(
            "[MCP_CHAT %s] HTTPException propagada desde llama_client.chat_completion",
            req_id,
        )
        raise
    except Exception as exc:
        logger.exception(
            "[MCP_CHAT %s] Error llamando a llama.cpp desde MCP: %s", req_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error proxying request to llama-server: {exc}",
        ) from exc

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[MCP_CHAT %s] Resultado bruto de llama.cpp (keys=%s)",
            req_id,
            list(result.keys()) if isinstance(result, dict) else type(result),
        )

    # 3) Postprocesado opcional para salidas JSON estructuradas
    if response_format.get("type") == "json_object":
        logger.info(
            "[MCP_CHAT %s] response_format.type='json_object'. Normalizando contenido a JSON.",
            req_id,
        )
        try:
            choices = result.get("choices") or []

            for idx, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    continue

                message = choice.get("message") or {}
                raw_content = message.get("content") or ""

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[MCP_CHAT %s] Choice %d: contenido original (truncado 400 chars): %r",
                        req_id,
                        idx,
                        str(raw_content)[:400],
                    )

                parsed = _extract_json_from_text(str(raw_content))

                message["content"] = json.dumps(parsed, ensure_ascii=False)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[MCP_CHAT %s] Choice %d: contenido normalizado a JSON (truncado 400 chars): %r",
                        req_id,
                        idx,
                        message["content"][:400],
                    )

                choice["message"] = message

            result["choices"] = choices

        except Exception as exc:
            logger.exception(
                "[MCP_CHAT %s] No se pudo normalizar la respuesta JSON del modelo: %s",
                req_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"No se pudo normalizar la respuesta JSON del modelo: {exc}",
            ) from exc

    # 4) Normalización final para compatibilidad con clientes estrictos (CrewAI)
    result = _ensure_non_empty_content_for_crewai(result, req_id)

    # 5) Devolvemos el JSON de llama.cpp (posiblemente postprocesado)
    logger.info(
        "[MCP_CHAT %s] Devolviendo respuesta final al cliente OpenAI-compatible.",
        req_id,
    )
    return result


# -----------------------------
# Tools HTTP registry endpoints
# -----------------------------
@router.get("/tools")
async def list_tools(
    crew: str | None = Query(
        default=None,
        description="Nombre lógico de la crew (p.ej. 'business_crew', 'document_crew').",
    ),
    tags: str | None = Query(
        default=None,
        description="Filtro opcional por tags, separados por coma.",
    ),
) -> Dict[str, Any]:
    """
    Devuelve las tools registradas en el GLOBAL_TOOL_REGISTRY, opcionalmente
    filtradas por nombre de crew y/o por tags.
    """
    tags_any = tags.split(",") if tags else None

    specs = GLOBAL_TOOL_REGISTRY.filter(
        tags_any=tags_any,
        crew_name=crew,
    )

    tools_payload = []
    for spec in specs:
        tools_payload.append(
            {
                "name": spec.name,
                "description": spec.description,
                "tags": spec.tags,
                "for_crews": spec.for_crews,
                "extra": spec.extra,
            }
        )

    logger.info(
        "[MCP_TOOLS] list_tools llamado: crew=%r, tags=%r → %d tools",
        crew,
        tags_any,
        len(tools_payload),
    )

    return {"tools": tools_payload}


@router.post("/tools/{tool_name}/invoke")
async def invoke_tool_http(
    tool_name: str,
    payload: ToolInvokeRequest,
) -> Dict[str, Any]:
    """
    Invoca una tool registrada en GLOBAL_TOOL_REGISTRY por nombre.

    - Busca la función en el registro global.
    - Filtra los args según la firma de la función (ignora 'ctx').
    - Soporta funciones sync y async.
    - Devuelve {"result": <lo_que_devuelva_la_tool>}.
    """
    spec = GLOBAL_TOOL_REGISTRY.get(tool_name)
    if spec is None:
        logger.warning("[MCP_TOOLS] Tool '%s' no encontrada en el registro", tool_name)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' no registrada en MCP",
        )

    func = spec.func
    sig = inspect.signature(func)

    kwargs: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "ctx":
            continue

        if name in payload.args:
            kwargs[name] = payload.args[name]
        else:
            if param.default is inspect._empty:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Falta argumento requerido '{name}' para tool '{tool_name}'",
                )

    try:
        if inspect.iscoroutinefunction(func):
            result = await func(**kwargs)
        else:
            result = func(**kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "[MCP_TOOLS] Error ejecutando tool '%s' vía HTTP: %s",
            tool_name,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ejecutando tool '{tool_name}': {exc}",
        ) from exc

    logger.info("[MCP_TOOLS] Tool '%s' ejecutada correctamente vía HTTP", tool_name)
    return {"result": result}

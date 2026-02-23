# cosmos_crewai/utils.py

import re
from typing import Optional


def clean_llm_text(text: Optional[str]) -> str:
    """
    Limpia texto generado por LLM:
    - recorta espacios
    - quita fences ``` y etiquetas ```json
    - elimina disclaimers típicos
    - colapsa saltos de línea redundantes
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

    # Eliminar disclaimers típicos de LLM en español
    disclaimer_patterns = [
        r"como modelo de lenguaje[^.\n]*\.",
        r"como ia(?: de)? lenguaje[^.\n]*\.",
    ]
    for pat in disclaimer_patterns:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)

    # Colapsar más de 2 saltos de línea
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def should_use_toon(context: str, threshold_chars: int = 4000) -> bool:
    """
    Decide si merece la pena usar el agente 'TOON' (analista/sumario)
    para ahorrar tokens antes de pasar el contexto al asistente principal.
    """
    if not context:
        return False
    return len(context) > threshold_chars

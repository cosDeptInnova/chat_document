import re

_FENCE_OPEN_RE = re.compile(r"^\s*```[a-zA-Z0-9_\-]*\s*$")
_FENCE_CLOSE_RE = re.compile(r"^\s*```\s*$")

_TAG_WRAPPERS = (
    ("<final>", "</final>"),
    ("<answer>", "</answer>"),
    ("<output>", "</output>"),
    ("<response>", "</response>"),
    ("<json>", "</json>"),
)

_PREFIX_RE = re.compile(r"^\s*(respuesta final|respuesta|salida|output)\s*:\s*", flags=re.I)


def clean_llm_text(text: str) -> str:
    """
    Limpieza pensada para orquestación multi-agente:
    - Quita fences ```...``` (aunque haya texto alrededor).
    - Quita wrappers XML-like frecuentes (<final>...</final> etc.).
    - Normaliza saltos de línea/espacios sin romper citas [n] ni secciones.
    - Elimina prefijos tipo "Respuesta:" al inicio.
    """
    if not text:
        return ""

    t = str(text)
    t = t.replace("\r\n", "\n").replace("\r", "\n").strip()

    # 1) Eliminar wrappers tipo <final>...</final> si envuelven TODO
    stripped = t.strip()
    for open_tag, close_tag in _TAG_WRAPPERS:
        if stripped.lower().startswith(open_tag) and stripped.lower().endswith(close_tag):
            inner = stripped[len(open_tag) : len(stripped) - len(close_tag)]
            t = inner.strip()
            break

    # 2) Eliminar fences si vienen como bloque completo
    #    (tanto si el modelo añade "```json" como si solo pone "```")
    #    Lo hacemos por líneas para evitar cortar contenido interno.
    lines = t.split("\n")
    # Quitar fences repetidos al principio
    while lines and _FENCE_OPEN_RE.match(lines[0]):
        lines = lines[1:]
    # Quitar fences repetidos al final
    while lines and _FENCE_CLOSE_RE.match(lines[-1]):
        lines = lines[:-1]
    t = "\n".join(lines).strip()

    # 3) Quitar prefijos basura al inicio
    t = _PREFIX_RE.sub("", t).strip()

    # 4) Colapsar saltos de línea excesivos sin dejar el texto “aplastado”
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()

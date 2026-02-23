# services/tools_registry.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional


@dataclass
class ToolSpec:
    name: str
    func: Callable[..., Any]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    for_crews: List[str] = field(default_factory=list)  # si vacío => genérica
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "for_crews": list(self.for_crews),
            **(self.extra or {}),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
        tags: Optional[List[str]] = None,
        for_crews: Optional[List[str]] = None,
        **extra: Any,
    ) -> ToolSpec:
        spec = ToolSpec(
            name=name,
            func=func,
            description=description or (func.__doc__ or "").strip(),
            tags=list(tags or []),
            for_crews=list(for_crews or []),
            extra=dict(extra or {}),
        )
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_all(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def filter(
        self,
        *,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        crew_name: Optional[str] = None,
    ) -> List[ToolSpec]:
        """
        Filtra tools por tags y/o por crew.

        Semántica:
        - Si crew_name está presente y la tool declara for_crews no vacío,
          debe contener crew_name o "*".
        - Si la tool NO declara for_crews, se considera genérica y se permite.
        - tags_any = OR
        - tags_all = AND
        """
        out: List[ToolSpec] = []
        crew_norm = (crew_name or "").strip().lower() or None
        any_set = {str(t).strip().lower() for t in (tags_any or []) if str(t).strip()}
        all_set = {str(t).strip().lower() for t in (tags_all or []) if str(t).strip()}

        for spec in self._tools.values():
            spec_crews = {str(c).strip().lower() for c in (spec.for_crews or []) if str(c).strip()}
            spec_tags = {str(t).strip().lower() for t in (spec.tags or []) if str(t).strip()}

            if crew_norm and spec_crews:
                if "*" not in spec_crews and crew_norm not in spec_crews:
                    continue

            if any_set:
                if not any(t in spec_tags for t in any_set):
                    continue

            if all_set:
                if not all(t in spec_tags for t in all_set):
                    continue

            out.append(spec)

        return out


# Registro global
GLOBAL_TOOL_REGISTRY = ToolRegistry()


def cosmos_tool(
    name: Optional[str] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
    for_crews: Optional[List[str]] = None,
    **extra: Any,
):
    """
    Decorador para registrar una función como tool de Cosmos
    en el registry interno (metadata + resolución de callable).

    NOTA:
    - Este decorador NO registra en el servidor MCP real por diseño
      (evita ciclos de import).
    - La capa MCP se registra en el módulo de tools (ej. ddg_tools.py).
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        GLOBAL_TOOL_REGISTRY.register(
            name=tool_name,
            func=func,
            description=description,
            tags=tags,
            for_crews=for_crews,
            **extra,
        )
        return func

    return decorator

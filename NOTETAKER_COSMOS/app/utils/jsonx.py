from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import orjson


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def read_json(path: Path) -> Dict[str, Any]:
    return orjson.loads(path.read_bytes())

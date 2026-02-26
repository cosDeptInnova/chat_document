from __future__ import annotations

import json
import re
from typing import Any, Optional


def clean_llm_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\r\n", "\n").strip()


def extract_json_from_text(text: str) -> Optional[Any]:
    if not text:
        return None

    cleaned = clean_llm_text(text)

    code_match = re.search(r"```json\s*(.*?)\s*```", cleaned, flags=re.S | re.I)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip())
        except Exception:
            pass

    xml_match = re.search(r"<json>\s*(.*?)\s*</json>", cleaned, flags=re.S | re.I)
    if xml_match:
        try:
            return json.loads(xml_match.group(1).strip())
        except Exception:
            pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        if start < 0:
            continue

        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break

    return None

from __future__ import annotations

import json
import re
from typing import Any

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json_object(text: str) -> Any:
    cleaned = _CODE_FENCE_RE.sub("", text.strip()).strip()
    if not cleaned:
        raise json.JSONDecodeError("Empty response", text, 0)

    # Fast path: already valid JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Recovery: locate the first {...} block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    return json.loads(cleaned[start : end + 1])


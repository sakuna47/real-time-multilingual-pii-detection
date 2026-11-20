"""Get JSON out of model replies."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S | re.I)


class JSONReplyError(ValueError):
    pass


def parse_json(text: str) -> Any:
    """Parse a reply that should be JSON. Accepts ```json fences and leading/trailing chatter."""
    s = text.strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as first:
        # fall back to the outermost {...} or [...] block
        starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
        if starts:
            start = min(starts)
            end = max(s.rfind("}"), s.rfind("]"))
            if end > start:
                try:
                    return json.loads(s[start:end + 1])
                except json.JSONDecodeError:
                    pass
        raise JSONReplyError(f"reply is not valid JSON: {first.msg} at char {first.pos}") from first

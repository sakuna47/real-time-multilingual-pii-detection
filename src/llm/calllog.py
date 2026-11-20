"""Append-only JSONL log of every LLM call (no keys, no prompt text): cost and latency reporting."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..common.runinfo import utc_now


class CallLog:
    def __init__(self, path: str | Path, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._lock = threading.Lock()

    def write(self, **fields: Any) -> None:
        entry = {"ts": utc_now(), "run_id": self.run_id, **fields}
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def summarize(path: str | Path) -> dict[str, dict[str, float]]:
    """Totals per model: calls, cache hits, errors, tokens, latency."""
    out: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            s = out.setdefault(e.get("model", "?"), {"calls": 0, "cache_hits": 0, "errors": 0,
                                                     "input_tokens": 0, "output_tokens": 0,
                                                     "thinking_tokens": 0, "latency_s": 0.0})
            s["calls"] += 1
            s["cache_hits"] += bool(e.get("cached"))
            s["errors"] += bool(e.get("error"))
            s["input_tokens"] += e.get("input_tokens") or 0
            s["output_tokens"] += e.get("output_tokens") or 0
            s["thinking_tokens"] += e.get("thinking_tokens") or 0   # billed like output
            s["latency_s"] += e.get("latency_s") or 0.0
    return out

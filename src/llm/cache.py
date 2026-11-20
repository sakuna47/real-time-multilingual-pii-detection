"""SQLite response cache. Re-running a pipeline costs nothing for calls already made."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .base import LLMRequest, LLMResponse


def cache_key(provider: str, model: str, req: LLMRequest, params: dict[str, Any]) -> str:
    payload = {
        "provider": provider, "model": model,
        "messages": [m.to_dict() for m in req.messages],
        "params": params, "json_mode": req.json_mode, "extra": req.extra,
        "salt": req.cache_salt,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses ("
                         "key TEXT PRIMARY KEY, model TEXT, response TEXT, created TEXT "
                         "DEFAULT CURRENT_TIMESTAMP)")
        self._db.commit()

    def get(self, key: str) -> LLMResponse | None:
        with self._lock:
            row = self._db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        resp = LLMResponse.from_dict(json.loads(row[0]))
        resp.cached = True
        return resp

    def put(self, key: str, resp: LLMResponse) -> None:
        blob = json.dumps(resp.to_dict(), ensure_ascii=False)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO responses (key, model, response) VALUES (?, ?, ?)",
                             (key, resp.name, blob))
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()

"""manifest.json for a folder of v1 JSONL files (docs/DATA_FORMAT.md §11)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..common.runinfo import git_commit, utc_now
from .record import SCHEMA_VERSION


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_file(path: Path) -> dict[str, Any]:
    c: dict[str, Counter] = {k: Counter() for k in
                             ("by_label", "by_lang", "by_domain", "by_method", "by_status", "by_tag", "by_split")}
    n = spans = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            spans += len(r.get("spans", []))
            c["by_label"].update(s["label"] for s in r.get("spans", []))
            c["by_lang"][r.get("language", {}).get("primary")] += 1
            c["by_domain"][r.get("domain")] += 1
            c["by_method"][(r.get("origin") or {}).get("method")] += 1
            c["by_status"][(r.get("validation") or {}).get("status")] += 1
            c["by_tag"].update(r.get("tags", []))
            c["by_split"][str(r.get("split"))] += 1
    return {"records": n, "spans": spans, "sha256": _sha256(path),
            **{k: dict(v.most_common()) for k, v in c.items()}}


def write_manifest(directory: str | Path, *, dataset: str, config: dict | None = None,
                   extra: dict | None = None, pattern: str = "*.jsonl") -> Path:
    directory = Path(directory)
    files = {p.name: summarize_file(p) for p in sorted(directory.glob(pattern))
             if not p.name.endswith(".rejected.jsonl")}
    totals: dict[str, Counter] = {}
    for info in files.values():
        for k, v in info.items():
            if k.startswith("by_"):
                totals.setdefault(k, Counter()).update(v)
    manifest = {
        "schema_version": SCHEMA_VERSION, "label_set": "v1", "dataset": dataset,
        "created_at": utc_now(), "git_commit": git_commit(),
        "records": sum(f["records"] for f in files.values()),
        "files": files,
        "counts": {k: dict(v.most_common()) for k, v in totals.items()},
        "config": config or {},
        **(extra or {}),
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

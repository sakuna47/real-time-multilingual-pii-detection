"""Fill missing Sinhala/Tamil spellings in resources/sl/{first_names,surnames,places}.csv.

Only empty cells are filled. Each filled row gets `+llm` added to its source, the filled
columns in `llm_filled`, and needs_review=yes, so you can review exactly what the model wrote.
Uses the `transliterator` role in configs/llm.toml.
"""

from __future__ import annotations

import json
import re

from ..llm import get_llm, load_prompt, run_batch
from .resources import load, write_csv

PROMPT_ID = "transliterate_names_v1"
KINDS = {"first_names": "given name", "surnames": "family name", "places": "Sri Lankan place name"}
_SCRIPT = {"sinhala": re.compile(r"^[඀-෿‍ .'\-]+$"), "tamil": re.compile(r"^[஀-௿ .'\-]+$")}
BATCH = 25


def fill_file(name: str, llm, limit: int | None = None) -> dict:
    rows = [dict(r) for r in load(name)]
    fields = list(rows[0].keys()) if rows else []
    if "llm_filled" not in fields:
        fields.append("llm_filled")
    todo = [i for i, r in enumerate(rows) if not r.get("sinhala") or not r.get("tamil")]
    if limit:
        todo = todo[:limit]
    prompt = load_prompt(PROMPT_ID)
    chunks = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    reqs = [prompt.request(kind=KINDS[name], n=len(c), items_json=json.dumps(
        [{"latin": rows[i]["latin"], "sinhala": rows[i].get("sinhala", ""), "tamil": rows[i].get("tamil", "")}
         for i in c], ensure_ascii=False)) for c in chunks]
    filled = rejected = 0
    errors: dict[str, int] = {}
    for chunk, res in zip(chunks, run_batch(llm, reqs, workers=4, json=True, progress_every=0)):
        items = res.data.get("items") if res.ok and isinstance(res.data, dict) else None
        if not isinstance(items, list) or len(items) != len(chunk):
            rejected += len(chunk)
            why = f"{type(res.error).__name__}: {res.error}"[:160] if res.error else "bad reply shape"
            errors[why] = errors.get(why, 0) + 1
            continue
        for i, item in zip(chunk, items):
            r = rows[i]
            if not isinstance(item, dict) or item.get("latin", "").strip().lower() != r["latin"].lower():
                rejected += 1
                continue
            done = []
            for col in ("sinhala", "tamil"):
                val = (item.get(col) or "").strip()
                if not r.get(col) and val and _SCRIPT[col].match(val):
                    r[col] = val
                    done.append(col)
            if done:
                r["source"] = r["source"] if "+llm" in r["source"] else r["source"] + "+llm"
                r["llm_filled"] = ",".join(sorted(set(filter(None, (r.get("llm_filled") or "").split(","))) | set(done)))
                r["needs_review"] = "yes"
                if item.get("unsure"):
                    r["llm_filled"] += ",unsure"
                filled += 1
    if filled:
        write_csv(name, rows, fields)
    return {"rows_needing_fill": len(todo), "filled": filled, "rejected": rejected, "errors": errors}


def fill_all(limit: int | None = None, dry_run: bool = False) -> None:
    llm = get_llm("transliterator", dry_run=dry_run, run_id="fill-scripts")
    for name in KINDS:
        print(name, fill_file(name, llm, limit))

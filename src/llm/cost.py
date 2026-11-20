"""Spend report from the call log, priced per model and per served tier (standard vs Flex).

    python -m src.llm.cost                    # all logged calls
    python -m src.llm.cost --since 2026-11-21 # from a date (UTC)
    python -m src.llm.cost --run tr-nemotron  # runs whose id starts with this

Uses [pricing] and [budget] in configs/llm.toml. Thinking tokens are billed as output.
Cached answers cost nothing and are not counted. This is an estimate from the API's token
counts at list price; the Billing console is the authority (it may also apply promotions).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from ..common.paths import logs_dir
from .config import DEFAULT_CONFIG

FLEX = "ON_DEMAND_FLEX"


def load_pricing(path: Path = DEFAULT_CONFIG) -> tuple[dict, float | None]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("pricing", {}), (data.get("budget") or {}).get("credits_usd")


def call_cost(entry: dict, pricing: dict) -> float | None:
    p = pricing.get(entry.get("model"))
    if p is None or entry.get("cached"):
        return None if p is None else 0.0
    flex = entry.get("traffic_type") == FLEX
    pin = p.get("flex_input", p["input"]) if flex else p["input"]
    pout = p.get("flex_output", p["output"]) if flex else p["output"]
    out_tokens = (entry.get("output_tokens") or 0) + (entry.get("thinking_tokens") or 0)
    return (entry.get("input_tokens") or 0) / 1e6 * pin + out_tokens / 1e6 * pout


def report(log: Path, since: str | None = None, run: str | None = None) -> dict:
    pricing, credits = load_pricing()
    rows: dict = defaultdict(lambda: {"calls": 0, "flex_calls": 0, "input_tokens": 0, "output_tokens": 0,
                                      "usd": 0.0})
    unpriced = set()
    for line in open(log, encoding="utf-8"):
        e = json.loads(line)
        if since and e.get("ts", "") < since:
            continue
        if run and not (e.get("run_id") or "").startswith(run):
            continue
        if e.get("cached") or e.get("error"):
            continue
        c = call_cost(e, pricing)
        if c is None:
            if not e.get("model", "").startswith("mock"):
                unpriced.add(e.get("model"))
            continue
        r = rows[e["model"]]
        r["calls"] += 1
        r["flex_calls"] += e.get("traffic_type") == FLEX
        r["input_tokens"] += e.get("input_tokens") or 0
        r["output_tokens"] += (e.get("output_tokens") or 0) + (e.get("thinking_tokens") or 0)
        r["usd"] += c
    total = sum(r["usd"] for r in rows.values())
    out = {"models": {m: {**r, "usd": round(r["usd"], 2)} for m, r in rows.items()},
           "total_usd": round(total, 2), "unpriced_models": sorted(unpriced)}
    if credits is not None and not since and not run:
        out["credits_usd"] = credits
        out["remaining_usd"] = round(credits - total, 2)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args(argv)
    print(json.dumps(report(args.log or logs_dir() / "llm_calls.jsonl", args.since, args.run), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

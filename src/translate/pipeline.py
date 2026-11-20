"""Placeholder-translation engine: source records -> Sri Lankan Sinhala/Tamil v1 records.

    python -m src.translate.pipeline --source nemotron --limit 50
    python -m src.translate.pipeline --source wildchat --limit 20 --langs si,ta
    python -m src.translate.pipeline --source aya --limit 10
    python -m src.translate.pipeline --source nemotron --limit 20 --dry-run      # offline, fake model
    python -m src.translate.pipeline --source nemotron --new 2500                # N more, skipping done/rejected

Flow per record:
  spans (Nemotron)          mask -> translate -> [code-mix] -> fill
  no spans (WildChat, en)   insert placeholders -> translate -> [code-mix] -> fill
  native (Aya, WildChat si/ta)  insert placeholders -> [code-mix] -> fill
Each source record becomes ONE variety, chosen by a hash of its id (--mix sets the shares).

Output: <data_root>/generated/sri_lankan/<source>/<variety>.jsonl, rejected.jsonl (stage + reason),
manifest.json. Re-running continues: records already written are skipped, and LLM answers
come from the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from ..common.paths import data_root, sources_dir
from ..common.runinfo import run_id as make_run_id
from ..llm import get_llm
from ..llm.mock import MockClient
from ..pii_data.manifest import write_manifest
from ..sl_gen import varieties
from ..sl_gen.personas import PersonaFactory
from . import steps
from .fill import build_record, fill
from .mask import mask

PIPELINE_VERSION = "1.0"
SOURCES = {
    "nemotron": [("nemotron_pii", "train.jsonl"), ("nemotron_pii", "test.jsonl")],
    "wildchat": [("wildchat", "prompts.jsonl"), ("wildchat", "native.jsonl")],
    "aya": [("aya", "si.jsonl"), ("aya", "ta.jsonl")],
}
DEFAULT_MIX = {"native": 0.4, "en": 0.3, "rom": 0.3}


def _h(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def parse_mix(s: str) -> dict[str, float]:
    mix = {k: float(v) for k, v in (p.split(":") for p in s.split(","))}
    if set(mix) - {"native", "en", "rom"}:
        raise SystemExit("--mix keys are native, en, rom")
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def assign_variety(rec: dict, langs: list[str], mix: dict[str, float], seed: int) -> varieties.Variety:
    src_lang = rec["language"]["primary"]
    h = _h(seed, rec["id"])
    lang = src_lang if src_lang in ("si", "ta") else langs[h % len(langs)]
    r, acc, kind = (h >> 20) % 10_000 / 10_000, 0.0, "native"
    for k in ("native", "en", "rom"):
        acc += mix.get(k, 0)
        if r < acc:
            kind = k
            break
    return varieties.get(lang if kind == "native" else f"{lang}-{kind}")


def output_id(rec: dict, v: varieties.Variety) -> str:
    prefix, local = rec["id"].split(":", 1)[0], rec["id"].rsplit(":", 1)[-1]
    return f"{prefix}-tr:{v.lang}:{local}-{v.code}"


def load_source(name: str, *, limit: int | None, seed: int, all_aya: bool = False,
                max_chars: int = 1200) -> list[dict]:
    rows = []
    for folder, fname in SOURCES[name]:
        path = sources_dir(folder) / fname
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if len(r["text"]) > max_chars:
                continue
            if name == "nemotron" and ("has_unmapped_pii" in r["tags"] or not r["spans"]):
                continue
            if name == "aya" and not all_aya and "personal_intent" not in r["tags"]:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"no usable records for {name}; run: python -m src.sources.run {name}")
    rows.sort(key=lambda r: _h(seed, "order", r["id"]))
    return rows[:limit] if limit else rows


class Out:
    def __init__(self, source: str):
        self.dir = data_root() / "generated" / "sri_lankan" / source
        self.dir.mkdir(parents=True, exist_ok=True)
        self.existing = set()
        for p in self.dir.glob("*.jsonl"):
            if p.name != "rejected.jsonl":
                self.existing.update(json.loads(l)["id"] for l in open(p, encoding="utf-8") if l.strip())
        self.files: dict[str, object] = {}
        self.rejected = open(self.dir / "rejected.jsonl", "a", encoding="utf-8", newline="\n")
        self.stats: Counter[str] = Counter()

    def content_rejections(self) -> set[str]:
        """Source ids rejected before for a content reason (not suitable, failed checks...).
        Rejections caused by API errors are not included, so those records are retried."""
        ids = set()
        path = self.dir / "rejected.jsonl"
        if path.exists():
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                if "Error" not in r.get("reason", ""):
                    ids.add(r["source_id"])
        return ids

    def write(self, rec: dict) -> None:
        v = rec["ext"]["variety"]
        if v not in self.files:
            self.files[v] = open(self.dir / f"{v}.jsonl", "a", encoding="utf-8", newline="\n")
        self.files[v].write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.stats[f"written:{v}"] += 1

    def reject(self, rec: dict, v, stage: str, reason: str, run: str) -> None:
        self.rejected.write(json.dumps({"run_id": run, "source_id": rec["id"], "variety": v.code,
                                        "stage": stage, "reason": reason}, ensure_ascii=False) + "\n")
        self.stats[f"rejected:{stage}"] += 1

    def close(self) -> None:
        for f in self.files.values():
            f.close()
        self.rejected.close()


def run(source: str, *, limit: int | None, langs: list[str], mix: dict[str, float], seed: int,
        batch_size: int, workers: int, dry_run: bool, all_aya: bool = False,
        min_budget_usd: float | None = 1.0, new: int | None = None) -> Counter:
    if not dry_run and min_budget_usd is not None:
        from ..common.paths import logs_dir
        from ..llm.cost import report
        log = logs_dir() / "llm_calls.jsonl"
        remaining = report(log).get("remaining_usd") if log.exists() else None
        if remaining is not None and remaining < min_budget_usd:
            raise SystemExit(f"stopping: estimated remaining credit ${remaining} is below ${min_budget_usd} "
                             "(python -m src.llm.cost; change --min-budget to override)")
        print(f"estimated remaining credit: ${remaining}", flush=True)
    run = make_run_id(f"tr-{source}")
    client = MockClient(handler=steps.dry_run_handler) if dry_run else None
    llms = {role: get_llm(role, dry_run=dry_run, run_id=run, provider_client=client)
            for role in ("inserter", "translator", "codemixer")}
    factory = PersonaFactory(seed=seed)
    if len(factory.surnames) < 50:
        print(f"warning: only {len(factory.surnames)} surnames have all three spellings; run "
              "`python -m src.sl_gen.run fill-scripts` and review resources/sl before a real run", flush=True)
    out = Out(source)

    todo = []
    # --new N: walk the whole source in stable hash order and take the first N records that were
    # neither written nor rejected for a content reason before (API errors are retried).
    skip_rejected = out.content_rejections() if new else set()
    for rec in load_source(source, limit=None if new else limit, seed=seed, all_aya=all_aya):
        v = assign_variety(rec, langs, mix, seed)
        if output_id(rec, v) in out.existing:
            out.stats["skipped:already_done"] += 1
            continue
        if rec["id"] in skip_rejected:
            out.stats["skipped:rejected_before"] += 1
            continue
        todo.append({"rec": rec, "v": v, "models": [], "tags": []})
        if new and len(todo) >= new:
            break
    print(f"{source}: {len(todo)} records to process ({out.stats['skipped:already_done']} already done)", flush=True)

    # 1. mask (spans) or insert placeholders (no spans)
    to_insert = [t for t in todo if not t["rec"]["spans"]]
    for t in todo:
        if t["rec"]["spans"]:
            m = mask(t["rec"])
            t["text"], t["slots"] = m.text, {k: s.to_dict() for k, s in m.slots.items()}
    for lang in sorted({t["rec"]["language"]["primary"] for t in to_insert}):
        group = [t for t in to_insert if t["rec"]["language"]["primary"] == lang]
        res = steps.insert(llms["inserter"], [t["rec"]["text"] for t in group], lang, batch_size, workers)
        for t, (val, err) in zip(group, res):
            if err:
                t["fail"] = ("insert", err)
            elif not val["suitable"]:
                t["fail"] = ("insert", "not suitable for personal details")
            else:
                t["text"], t["slots"] = val["text"], val["slots"]
                t["models"].append(llms["inserter"].model_record("insert_v1"))
                t["tags"].append("placeholder_inserted")

    # 2. translate English -> si / ta
    for lang in langs:
        group = [t for t in todo if "fail" not in t and t["rec"]["language"]["primary"] == "en" and t["v"].lang == lang]
        if not group:
            continue
        res = steps.translate(llms["translator"], [t["text"] for t in group], lang, batch_size, workers)
        for t, (val, err) in zip(group, res):
            if err:
                t["fail"] = ("translate", err)
            else:
                t["text"] = val
                t["models"].append(llms["translator"].model_record(steps.TRANSLATE_PROMPT))

    # 3. code-mix / romanise
    for code in ("si-en", "ta-en", "si-rom", "ta-rom"):
        group = [t for t in todo if "fail" not in t and t["v"].code == code]
        if not group:
            continue
        res = steps.codemix(llms["codemixer"], [t["text"] for t in group], varieties.get(code),
                            batch_size, workers, seed=seed)
        for t, (val, err) in zip(group, res):
            if err:
                t["fail"] = ("codemix", err)
            else:
                t["text"] = val
                t["models"].append(llms["codemixer"].model_record("codemix_v1"))

    # 4. fill and write
    for t in todo:
        rec, v = t["rec"], t["v"]
        if "fail" in t:
            out.reject(rec, v, *t["fail"], run)
            continue
        filled, err = fill(t["text"], t["slots"], v, factory, seed_key=f"{seed}:{rec['id']}")
        if err:
            out.reject(rec, v, "fill", err, run)
            continue
        native_src = rec["language"]["primary"] in ("si", "ta")
        method = "llm_generated" if native_src else "translated_placeholder"
        new = build_record(filled=filled, source=rec, variety=v, method=method, masked_target=t["text"],
                           models=t["models"], extra_tags=t["tags"], seed=seed,
                           pipeline_version=PIPELINE_VERSION)
        if new["validation"]["status"] == "auto_failed":
            out.reject(rec, v, "validate", "; ".join(new["validation"]["checks"]["errors"]), run)
            continue
        out.write(new)
    out.close()
    write_manifest(out.dir, dataset=f"sri_lankan/{source}",
                   config={"langs": langs, "mix": mix, "seed": seed, "pipeline_version": PIPELINE_VERSION,
                           "dry_run": dry_run})
    return out.stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--limit", type=int, default=None, help="first N source records (stable order)")
    ap.add_argument("--new", type=int, default=None,
                    help="process N records not yet written or rejected before (use this for later stages)")
    ap.add_argument("--langs", default="si,ta")
    ap.add_argument("--mix", default="native:0.4,en:0.3,rom:0.3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--all-aya", action="store_true", help="Aya: use all prompts, not only personal-intent ones")
    ap.add_argument("--dry-run", action="store_true", help="offline fake model (tests the pipeline only)")
    ap.add_argument("--min-budget", type=float, default=1.0,
                    help="refuse to start if estimated remaining credit (src.llm.cost) is below this, USD")
    args = ap.parse_args(argv)
    stats = run(args.source, limit=args.limit, langs=[l.strip() for l in args.langs.split(",")],
                mix=parse_mix(args.mix), seed=args.seed, batch_size=args.batch_size, workers=args.workers,
                dry_run=args.dry_run, all_aya=args.all_aya, min_budget_usd=args.min_budget,
                new=args.new)
    print(json.dumps(dict(sorted(stats.items())), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

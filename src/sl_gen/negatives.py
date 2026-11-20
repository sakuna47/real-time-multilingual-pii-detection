"""Hard negatives: real PII-free text + a clause with a PII-looking but harmless number.

    python -m src.sl_gen.negatives --n 400            # per variety

The model must learn that a 12-digit invoice number or a 07X tracking number is not a NIC or
phone number when the context says so. Values go into `distractors[]` (never `spans`), and
records are tagged hard_negative.

Base texts (real, PII-free):
  si, ta   Aya prompts without personal intent       (Apache-2.0)
  si-en    NLPC-UOM code-mixed social media comments (MIT)
  si-rom   SinMix2Mono gold romanised Sinhala        (CC BY 4.0)
  ta-en / ta-rom: no real base pool yet (skipped)

The clause wordings below were written for this project and need review by a native speaker.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from datetime import date, timedelta

from ..common.paths import data_root, sources_dir
from ..common.runinfo import git_commit, utc_now
from ..pii_data.manifest import write_manifest
from ..pii_data.record import SCHEMA_VERSION, detect_scripts, validate_record
from .varieties import get as get_variety

# kind -> per-variety clause; {X} is the distractor value
CLAUSES = {
    "invoice_number": {"si": "ඉන්වොයිස් අංකය {X} වේ.", "ta": "விலைப்பட்டியல் எண் {X}.",
                       "si-en": "invoice number එක {X}.", "si-rom": "invoice number eka {X}."},
    "tracking_number": {"si": "පාර්සලයේ ට්‍රැකින් අංකය {X}.", "ta": "பொதியின் கண்காணிப்பு எண் {X}.",
                        "si-en": "parcel එකේ tracking number එක {X}.", "si-rom": "parcel eke tracking number eka {X}."},
    "date_number": {"si": "{X} දින සිට අදාළ වේ.", "ta": "{X} திகதி முதல் பொருந்தும்.",
                    "si-en": "{X} date එකේ ඉඳන් apply වෙනවා.", "si-rom": "{X} date eke idan apply wenawa."},
    "product_code": {"si": "භාණ්ඩ කේතය {X}.", "ta": "பொருள் குறியீடு {X}.",
                     "si-en": "product code එක {X}.", "si-rom": "product code eka {X}."},
}
KIND_TO_SPEC = {"invoice_number": "invoice_number", "tracking_number": "reference_number",
                "date_number": "date_number", "product_code": "product_code"}


def value_for(kind: str, rng: random.Random) -> str:
    if kind == "invoice_number":      # 12 digits starting like a birth year: NIC-shaped on purpose
        return f"{rng.randint(1950, 2024)}{rng.randint(0, 99999999):08d}"
    if kind == "tracking_number":     # 07X + 7 digits: phone-shaped on purpose
        return f"07{rng.choice('0125678')}{rng.randint(0, 9999999):07d}"
    if kind == "date_number":
        d = date(2015, 1, 1) + timedelta(days=rng.randint(0, 4000))
        return d.strftime("%Y%m%d")
    if kind == "product_code":
        return f"{rng.choice(['SKU', 'PRD', 'ITM'])}-{rng.randint(1000, 99999)}"
    raise ValueError(kind)


def base_pool(variety: str) -> list[tuple[str, str, dict]]:
    """[(base_id, text, origin)] of PII-free real texts for a variety."""
    out = []
    if variety in ("si", "ta"):
        path = sources_dir("aya") / f"{variety}.jsonl"
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if "personal_intent" not in r["tags"] and 20 <= len(r["text"]) <= 400 and "\n" not in r["text"].strip():
                out.append((r["id"], r["text"], r["origin"]))
    elif variety == "si-en":
        for line in open(sources_dir("codemix") / "si_en_social.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if {"Sinh", "Latn"} <= set(detect_scripts(r["text"])) and 20 <= len(r["text"]) <= 300:
                out.append((r["id"], r["text"], {"dataset": r["source"], "license": "MIT"}))
    elif variety == "si-rom":
        for line in open(sources_dir("codemix") / "singlish_gold.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if 20 <= len(r["rom"]) <= 300:
                out.append((r["id"], r["rom"], {"dataset": r["source"], "license": "CC BY 4.0"}))
    return out


def make(base_id: str, text: str, origin: dict, variety: str, rng: random.Random) -> dict | None:
    v = get_variety(variety)
    kind = rng.choice(sorted(CLAUSES))
    value = value_for(kind, rng)
    clause = CLAUSES[kind][variety]
    base = unicodedata.normalize("NFC", text.strip())
    sep = " " if base.endswith((".", "?", "!", "।")) else ". "
    prefix = base + sep + clause.split("{X}")[0]
    full = unicodedata.normalize("NFC", prefix + value + clause.split("{X}")[1])
    start = len(prefix)
    if full[start:start + len(value)] != value:
        return None
    scripts = set(detect_scripts(full))
    local = base_id.replace(":", "-")
    rec = {
        "schema_version": SCHEMA_VERSION, "id": f"neg:{v.lang}:{local}-{variety}", "text": full, "spans": [],
        "label_set": "v1",
        "language": v.language_block(has_latin="Latn" in scripts, has_native=v.native_script in scripts),
        "domain": "chat_prompt", "region": "LK",
        "origin": {"method": "template_filled", "dataset": origin.get("dataset"), "source_id": base_id,
                   "parent_id": base_id, "template_id": f"neg:{kind}:{variety}", "license": origin.get("license")},
        "generation": {"pipeline": "src.sl_gen.negatives", "pipeline_version": "1.0", "git_commit": git_commit(),
                       "seed": None, "created_at": utc_now(), "models": []},
        "validation": {"status": "auto_passed", "checks": {"distractor_offsets": True}},
        "implicit": None,
        "distractors": [{"start": start, "end": start + len(value), "kind": KIND_TO_SPEC[kind]}],
        "split": None, "group": base_id, "tags": ["hard_negative"],
        "ext": {"variety": variety, "distractor_kind": kind, "clause_needs_review": True},
    }
    return rec if not validate_record(rec) else None


def run(n: int, seed: int = 42, varieties: tuple[str, ...] = ("si", "ta", "si-en", "si-rom")) -> dict:
    out_dir = data_root() / "generated" / "sri_lankan" / "negatives"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for variety in varieties:
        pool = base_pool(variety)
        rng = random.Random(f"{seed}:{variety}")
        rng.shuffle(pool)
        written = 0
        with open(out_dir / f"{variety}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for base_id, text, origin in pool:
                if written >= n:
                    break
                rec = make(base_id, text, origin, variety, rng)
                if rec:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
        stats[variety] = {"pool": len(pool), "written": written}
    write_manifest(out_dir, dataset="sri_lankan/negatives", config={"n": n, "seed": seed})
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="records per variety")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    print(json.dumps(run(args.n, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

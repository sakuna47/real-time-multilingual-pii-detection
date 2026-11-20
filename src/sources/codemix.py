"""Code-mixing and romanisation reference pools (plain JSONL, not v1 records).

    python -m src.sources.run codemix

Used as few-shot style examples for the codemixer role and as references for the
Code-Mixing Index / romanisation checks. Their content is never inserted as PII.

Writes <data_root>/sources/codemix/:
  singlish_gold.jsonl     SinMix2Mono golden set (manually annotated)   {rom, si}      CC BY 4.0
  singlish_train.jsonl    SinMix2Mono training set                      {rom, si}      CC BY 4.0
  triplets.jsonl          Sinhala/English/Singlish triplets (subtitles;
                          rows with [Unkown] tokens dropped)            {si, en, rom}  MIT
  si_en_social.jsonl      NLPC-UOM code-mixed social media sentences    {text}         MIT
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from ..common.paths import sources_dir
from ..common.runinfo import utc_now
from .common import hf_file, norm_text

SOURCES = {
    "singlish_gold": ("RukDias/SinMix2Mono", "Golden_dataset.csv", "CC BY 4.0"),
    "singlish_train": ("RukDias/SinMix2Mono", "training_dataset.csv", "CC BY 4.0"),
    "triplets": ("Programmer-RD-AI/sinhala-english-singlish-translation", "data.csv", "MIT"),
    "si_en_social": ("NLPC-UOM/Sinhala-English-Code-Mixed-Code-Switched-Dataset",
                     "sentence-level-annotation.csv", "MIT"),
}


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def convert(name: str, row: dict) -> tuple[dict | None, str | None]:
    if name in ("singlish_gold", "singlish_train"):
        rom, si = (row.get("Romanized_Sinhala") or "").strip(), (row.get("Sinhala") or "").strip()
        if not rom or not si:
            return None, "empty"
        return {"rom": norm_text(rom), "si": norm_text(si)}, None
    if name == "triplets":
        si, en, rom = (row.get("Sinhala") or "").strip(), (row.get("English") or "").strip(), \
                      (row.get("Singlish") or "").strip()
        if not (si and en and rom):
            return None, "empty"
        if "[Unkown]" in rom or "[Unknown]" in rom:
            return None, "unknown_tokens"
        return {"si": norm_text(si), "en": en, "rom": rom}, None
    if name == "si_en_social":
        text = (row.get("Sentence") or "").strip()
        if len(text) < 5:
            return None, "too_short"
        return {"text": norm_text(text)}, None
    raise ValueError(name)


def run(limit: int | None = None) -> None:
    out_dir = sources_dir("codemix")
    manifest = {"created_at": utc_now(), "files": {}}
    for name, (repo, filename, license) in SOURCES.items():
        skipped: Counter[str] = Counter()
        seen, n = set(), 0
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for i, row in enumerate(_rows(hf_file(repo, filename))):
                if limit and n >= limit:
                    break
                item, reason = convert(name, row)
                if item is None:
                    skipped[reason] += 1
                    continue
                key = json.dumps(item, ensure_ascii=False)
                if key in seen:
                    skipped["duplicate"] += 1
                    continue
                seen.add(key)
                item.update({"id": f"{name}:{i:06d}", "source": repo})
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
        manifest["files"][f"{name}.jsonl"] = {"source": f"https://huggingface.co/datasets/{repo}",
                                              "file": filename, "license": license, "rows": n,
                                              "skipped": dict(skipped)}
        print(f"codemix/{name}.jsonl: wrote {n}, skipped {dict(skipped)}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

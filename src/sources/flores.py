"""FLORES+ (FLORES-200) sentences aligned across English, Sinhala and Tamil.

Used to pick the translator model on evidence (src/translate/benchmark.py).
Source: openlanguagedata/flores_plus on Hugging Face, CC BY-SA 4.0, gated:
accept the terms on the dataset page and set HF_TOKEN.

    python -m src.sources.flores                 # devtest -> <data_root>/sources/flores/devtest.jsonl
    python -m src.sources.flores --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..common.paths import sources_dir
from ..common.runinfo import utc_now
from ..common.secrets import get_secret

REPO_ID = "openlanguagedata/flores_plus"
LICENSE = "CC BY-SA 4.0"
LANGS = {"en": "eng_Latn", "si": "sin_Sinh", "ta": "tam_Taml"}


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def align(rows_by_lang: dict[str, list[dict]]) -> list[dict]:
    """Join per-language rows on their `id` (falls back to line order if there is no id).
    Returns [{"id", "en", "si", "ta"}] sorted by id, keeping only ids present in every language."""
    tables = {}
    for lang, rows in rows_by_lang.items():
        table = {}
        for i, r in enumerate(rows):
            text = r.get("text") or r.get("sentence")
            if text is None:
                raise ValueError(f"{lang}: row has no 'text' field: {sorted(r)}")
            table[r.get("id", i)] = text.strip()
        tables[lang] = table
    common = set.intersection(*(set(t) for t in tables.values()))
    out = []
    for key in sorted(common, key=lambda k: (str(type(k)), k)):
        row = {"id": str(key)}
        row.update({lang: tables[lang][key] for lang in rows_by_lang})
        out.append(row)
    return out


def download(split: str = "devtest") -> list[dict]:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError as e:
        raise ImportError("pip install huggingface_hub (comes with datasets)") from e
    token = get_secret("HF_TOKEN", required=False)
    rows_by_lang = {}
    for lang, code in LANGS.items():
        try:
            path = hf_hub_download(REPO_ID, f"{split}/{code}.jsonl", repo_type="dataset", token=token)
        except GatedRepoError as e:
            raise SystemExit(
                f"{REPO_ID} is gated. Open https://huggingface.co/datasets/{REPO_ID}, accept the "
                "terms with your account, and set HF_TOKEN (.env or Colab Secrets).") from e
        rows_by_lang[lang] = _read_jsonl(path)
    return align(rows_by_lang)


def output_path(split: str = "devtest") -> Path:
    return sources_dir("flores") / f"{split}.jsonl"


def save(rows: list[dict], split: str) -> Path:
    path = output_path(split)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"dataset": REPO_ID, "split": split, "license": LICENSE, "languages": LANGS,
                "rows": len(rows), "created_at": utc_now()}
    (path.parent / f"{split}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load(split: str = "devtest", limit: int | None = None) -> list[dict]:
    path = output_path(split)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m src.sources.flores --split {split}")
    rows = _read_jsonl(path)
    return rows[:limit] if limit else rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    args = ap.parse_args(argv)
    rows = download(args.split)
    path = save(rows, args.split)
    print(f"saved {len(rows)} aligned en/si/ta sentences -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

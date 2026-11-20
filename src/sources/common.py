"""Shared pieces for importers: record skeletons, span merging, downloads, writing."""

from __future__ import annotations

import hashlib
import json
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..common.paths import download_dir, sources_dir
from ..common.runinfo import git_commit, utc_now
from ..common.secrets import get_secret
from ..pii_data.manifest import write_manifest
from ..pii_data.record import SCHEMA_VERSION, detect_scripts, validate_record


def short_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def norm_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def language_block(primary: str, text: str, other_latin: str = "en") -> dict:
    """language{} from the scripts actually present. Latin letters in a Sinhala/Tamil text
    mark it as code-mixed with English (heuristic; tagged auto_lang by callers)."""
    scripts = detect_scripts(text) or ["Latn"]
    native = {"si": "Sinh", "ta": "Taml"}.get(primary)
    mixed = bool(native and native in scripts and "Latn" in scripts)
    romanized = bool(native and native not in scripts and scripts == ["Latn"])
    langs = [primary] + ([other_latin] if mixed or romanized else [])
    return {"primary": primary, "langs": langs, "scripts": scripts,
            "code_mixed": mixed or (romanized and len(langs) > 1), "romanized": romanized}


def base_record(*, rid: str, text: str, spans: list[dict], lang: dict, domain: str,
                dataset: str, source_id: str, license: str, pipeline: str,
                region: str | None = None, group: str | None = None, tags: list[str] | None = None,
                ext: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "id": rid, "text": text, "spans": spans,
        "label_set": "v1", "language": lang, "domain": domain, "region": region,
        "origin": {"method": "original", "dataset": dataset, "source_id": source_id,
                   "parent_id": None, "template_id": None, "license": license},
        "generation": {"pipeline": pipeline, "pipeline_version": "1.0", "git_commit": git_commit(),
                       "seed": None, "created_at": utc_now(), "models": []},
        "validation": {"status": "auto_passed", "checks": {"import_validated": True}},
        "implicit": None, "distractors": [], "split": None, "group": group or rid,
        "tags": tags or [], "ext": ext or {},
    }


def should_merge(label: str, gap: str) -> bool:
    """Same rule as the AI4Privacy conversion and docs/DATA_FORMAT.md §3."""
    if label == "PERSON":
        return gap == " "
    if label == "ADDRESS":
        return gap.strip(" ,") == ""
    return False


def build_spans(text: str, raw: Iterable[tuple[int, int, str, str | None, str]]) -> list[dict] | None:
    """raw = (start, end, label, subtype, source_label), already mapped to v1.
    Sorts, merges neighbours, trims whitespace. Returns None if spans overlap."""
    spans: list[dict] = []
    for st, en, label, sub, src in sorted(raw, key=lambda x: (x[0], x[1])):
        piece = text[st:en]
        lead = len(piece) - len(piece.lstrip())
        trail = len(piece) - len(piece.rstrip())
        st, en = st + lead, en - trail
        if st >= en:
            continue
        prev = spans[-1] if spans else None
        if prev and st < prev["end"]:
            return None
        if prev and prev["label"] == label and prev["subtype"] == sub \
                and should_merge(label, text[prev["end"]:st]):
            prev["end"] = en
            prev["value"] = text[prev["start"]:en]
            prev["source_labels"].append(src)
            continue
        spans.append({"start": st, "end": en, "label": label, "subtype": sub,
                      "value": text[st:en], "source_labels": [src]})
    return spans


def hf_file(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id, filename, repo_type="dataset",
                                token=get_secret("HF_TOKEN", required=False)))


def url_file(url: str, name: str, subdir: str) -> Path:
    path = download_dir(subdir) / name
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(path)
    return path


class Writer:
    """Validates and writes v1 records to <data_root>/sources/<dataset>/<part>.jsonl.
    Invalid records go to <part>.rejected.jsonl with their errors."""

    def __init__(self, dataset: str, part: str):
        self.dir = sources_dir(dataset)
        self.path = self.dir / f"{part}.jsonl"
        self.rej_path = self.dir / f"{part}.rejected.jsonl"
        self._f = open(self.path, "w", encoding="utf-8", newline="\n")
        self._rej = None
        self.ids: set[str] = set()
        self.written = 0
        self.skipped: Counter[str] = Counter()

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1

    def write(self, rec: dict) -> bool:
        if rec["id"] in self.ids:
            self.skip("duplicate_id")
            return False
        errors = validate_record(rec)
        if errors:
            if self._rej is None:
                self._rej = open(self.rej_path, "w", encoding="utf-8", newline="\n")
            self._rej.write(json.dumps({"errors": errors, "record": rec}, ensure_ascii=False) + "\n")
            self.skip("invalid_record")
            return False
        self.ids.add(rec["id"])
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.written += 1
        return True

    def close(self) -> None:
        self._f.close()
        if self._rej:
            self._rej.close()


def finish(dataset: str, writers: list[Writer], *, license: str, source: str,
           config: dict | None = None, citation: str | None = None) -> Path:
    for w in writers:
        w.close()
        print(f"{dataset}/{w.path.name}: wrote {w.written}, skipped {dict(w.skipped)}")
    extra = {"source": source, "license": license,
             "skipped": {w.path.name: dict(w.skipped) for w in writers}}
    if citation:
        extra["citation"] = citation
    path = write_manifest(sources_dir(dataset), dataset=dataset, config=config, extra=extra)
    print(f"manifest -> {path}")
    return path

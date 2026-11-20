"""PII record format v1: label set, validator and AI4Privacy converter.

Spec: docs/DATA_FORMAT.md. Standard library only, so it runs in Colab and in the
backend unchanged. If `jsonschema` is installed, the JSON Schema is checked too.

    python -m src.pii_data data/processed/ai4privacy/train.jsonl
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = "1.0"

# label -> allowed subtypes (empty = subtype must be null)
LABEL_SETS: dict[str, dict[str, set[str]]] = {
    "v1": {
        "PERSON": set(),
        "ID_NUMBER": {"NIC_OLD", "NIC_NEW", "PASSPORT", "DRIVING_LICENCE",
                      "SOCIAL_SECURITY", "NATIONAL_ID", "OTHER"},
        "PHONE": {"MOBILE", "LANDLINE"},
        "EMAIL": set(),
        "ADDRESS": set(),
        "LOCATION": set(),
        "DOB": set(),
        "USERNAME": set(),
        "PASSWORD": set(),
        "IP_ADDRESS": set(),
    }
}

REQUIRED = ("schema_version", "id", "text", "spans", "label_set", "language",
            "domain", "origin", "validation")
DOMAINS = {"chat_prompt", "email", "form", "letter", "chat_message", "document",
           "news", "wiki", "other"}
METHODS = {"original", "translated_placeholder", "translated_projected",
           "llm_generated", "template_filled", "human_written", "human_edited"}
STATUSES = {"unchecked", "auto_failed", "auto_passed", "judge_failed",
            "judge_passed", "human_rejected", "human_verified"}
SPLITS = {"train", "dev", "test", None}

# Unicode block -> ISO 15924 code, for script detection
_SCRIPT_RANGES = (
    (0x0D80, 0x0DFF, "Sinh"),
    (0x0B80, 0x0BFF, "Taml"),
    (0x0041, 0x024F, "Latn"),
)


def detect_scripts(text: str) -> list[str]:
    """Scripts present in `text`, most frequent first (letters only)."""
    counts: Counter[str] = Counter()
    for ch in text:
        cp = ord(ch)
        for lo, hi, code in _SCRIPT_RANGES:
            if lo <= cp <= hi and (code != "Latn" or ch.isalpha()):
                counts[code] += 1
                break
    return [s for s, _ in counts.most_common()]


# ---------------------------------------------------------------- validation

def _check_offsets(items, text, name, errors, need_value=False):
    for i, s in enumerate(items):
        st, en = s.get("start"), s.get("end")
        if not (isinstance(st, int) and isinstance(en, int) and 0 <= st < en <= len(text)):
            errors.append(f"{name}[{i}]: bad offsets {st}-{en} (text length {len(text)})")
            continue
        if need_value:
            piece = text[st:en]
            if s.get("value") != piece:
                errors.append(f"{name}[{i}]: value {s.get('value')!r} != text {piece!r}")
            if piece != piece.strip():
                errors.append(f"{name}[{i}]: span has leading/trailing whitespace {piece!r}")


def _overlaps(items) -> list[tuple[int, int]]:
    ordered = sorted(range(len(items)), key=lambda i: items[i]["start"])
    return [(a, b) for a, b in zip(ordered, ordered[1:])
            if items[b]["start"] < items[a]["end"]]


def validate_record(rec: dict) -> list[str]:
    """Return a list of problems; empty list means the record is valid."""
    errors: list[str] = []
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        return [f"missing required fields: {missing}"]

    if not str(rec["schema_version"]).startswith("1."):
        errors.append(f"unsupported schema_version {rec['schema_version']!r}")
    parts = str(rec["id"]).split(":", 2)
    if len(parts) != 3 or not all(parts):
        errors.append(f"id {rec['id']!r} is not <source>:<lang>:<local-id>")

    text = rec["text"]
    if not isinstance(text, str) or not text:
        return errors + ["text must be a non-empty string"]
    if unicodedata.normalize("NFC", text) != text:
        errors.append("text is not NFC-normalised")

    labels = LABEL_SETS.get(rec["label_set"])
    if labels is None:
        return errors + [f"unknown label_set {rec['label_set']!r}"]

    spans = rec["spans"]
    _check_offsets(spans, text, "spans", errors, need_value=True)
    if not errors or all(not e.startswith("spans") for e in errors):
        if [s["start"] for s in spans] != sorted(s["start"] for s in spans):
            errors.append("spans are not sorted by start")
        for a, b in _overlaps(spans):
            errors.append(f"spans[{a}] and spans[{b}] overlap")
    for i, s in enumerate(spans):
        label, sub = s.get("label"), s.get("subtype")
        if label not in labels:
            errors.append(f"spans[{i}]: unknown label {label!r}")
        elif sub is not None and sub not in labels[label]:
            errors.append(f"spans[{i}]: subtype {sub!r} not allowed for {label}")

    lang = rec["language"]
    for k in ("primary", "langs", "scripts", "code_mixed", "romanized"):
        if k not in lang:
            errors.append(f"language.{k} missing")
    if lang.get("langs") and lang.get("primary") != lang["langs"][0]:
        errors.append("language.primary must be first in language.langs")
    if lang.get("code_mixed") is False and len(lang.get("langs", [])) > 1:
        errors.append("language.langs has several languages but code_mixed is false")
    found = set(detect_scripts(text))
    declared = set(lang.get("scripts", []))
    if found - declared:
        errors.append(f"scripts in text {sorted(found)} not declared in {sorted(declared)}")

    if rec["domain"] not in DOMAINS:
        errors.append(f"unknown domain {rec['domain']!r}")
    if rec["origin"].get("method") not in METHODS:
        errors.append(f"unknown origin.method {rec['origin'].get('method')!r}")
    if rec["validation"].get("status") not in STATUSES:
        errors.append(f"unknown validation.status {rec['validation'].get('status')!r}")
    if rec.get("split") not in SPLITS:
        errors.append(f"unknown split {rec.get('split')!r}")

    distractors = rec.get("distractors") or []
    _check_offsets(distractors, text, "distractors", errors)
    for i, d in enumerate(distractors):
        if any(s["start"] < d.get("end", -1) and d.get("start", -1) < s["end"] for s in spans):
            errors.append(f"distractors[{i}] overlaps a PII span")

    implicit = rec.get("implicit")
    if implicit:
        _check_offsets(implicit.get("quasi_identifiers", []), text,
                       "implicit.quasi_identifiers", errors)
    return errors


def validate_file(path: str | Path, schema_path: str | Path | None = None,
                  max_errors: int = 20) -> dict:
    """Validate a JSONL file. Returns a summary dict with counts and first errors."""
    schema_validator = None
    if schema_path is None:
        schema_path = Path(__file__).resolve().parents[2] / "schema" / "pii_record.v1.schema.json"
    try:
        import jsonschema
        if Path(schema_path).exists():
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            schema_validator = jsonschema.Draft202012Validator(schema)
    except ImportError:
        pass

    total = bad = 0
    ids: set[str] = set()
    first_errors: list[str] = []
    labels: Counter[str] = Counter()
    langs: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errs = [f"invalid JSON: {e}"]
                rec = None
            else:
                errs = validate_record(rec)
                if schema_validator is not None:
                    errs += [f"schema: {e.message}" for e in schema_validator.iter_errors(rec)]
                if rec.get("id") in ids:
                    errs.append(f"duplicate id {rec.get('id')!r}")
                ids.add(rec.get("id"))
            if errs:
                bad += 1
                for e in errs:
                    if len(first_errors) < max_errors:
                        first_errors.append(f"line {lineno}: {e}")
            if rec:
                langs[rec.get("language", {}).get("primary")] += 1
                statuses[rec.get("validation", {}).get("status")] += 1
                labels.update(s.get("label") for s in rec.get("spans", []))
    return {"records": total, "invalid": bad, "errors": first_errors,
            "by_lang": dict(langs), "by_label": dict(labels), "by_status": dict(statuses),
            "jsonschema_checked": schema_validator is not None}


# ------------------------------------------------------------ AI4Privacy

AI4P_MAP = {
    "GIVENNAME1": ("PERSON", None), "GIVENNAME2": ("PERSON", None),
    "LASTNAME1": ("PERSON", None), "LASTNAME2": ("PERSON", None), "LASTNAME3": ("PERSON", None),
    "IDCARD": ("ID_NUMBER", "NATIONAL_ID"), "SOCIALNUMBER": ("ID_NUMBER", "SOCIAL_SECURITY"),
    "PASSPORT": ("ID_NUMBER", "PASSPORT"), "DRIVERLICENSE": ("ID_NUMBER", "DRIVING_LICENCE"),
    "TEL": ("PHONE", None),
    "EMAIL": ("EMAIL", None),
    "BUILDING": ("ADDRESS", None), "STREET": ("ADDRESS", None),
    "SECADDRESS": ("ADDRESS", None), "POSTCODE": ("ADDRESS", None),
    "CITY": ("LOCATION", None), "GEOCOORD": ("LOCATION", None),
    "BOD": ("DOB", None),
    "USERNAME": ("USERNAME", None),
    "PASS": ("PASSWORD", None),
    "IP": ("IP_ADDRESS", None),
    # dropped -> O: TIME, DATE, SEX, TITLE, STATE, COUNTRY, CARDISSUER
}
AI4P_LANGS = {"English": "en", "French": "fr", "German": "de",
              "Italian": "it", "Spanish": "es", "Dutch": "nl"}


def _should_merge(label: str, gap: str) -> bool:
    if label == "PERSON":
        return gap == " "
    if label == "ADDRESS":
        return gap.strip(" ,") == ""
    return False


def from_ai4privacy_row(row: dict, split: str | None = None,
                        git_commit: str | None = None) -> dict | None:
    """Convert one raw AI4Privacy row (source_text, privacy_mask, id, language)
    to a v1 record. Returns None if the row's offsets don't match its values.
    Same mapping and merge rules as 02_explore_ai4privacy.ipynb, but keeps
    the original labels in `source_labels`."""
    text = row["source_text"]
    raw = sorted(row["privacy_mask"], key=lambda s: int(s["start"]))
    if any(text[int(s["start"]):int(s["end"])] != s["value"] for s in raw):
        return None
    if unicodedata.normalize("NFC", text) != text:
        return None  # offsets would shift; rare in AI4Privacy, skip rather than guess

    spans: list[dict] = []
    for s in raw:
        mapped = AI4P_MAP.get(s["label"])
        if mapped is None:
            continue
        label, sub = mapped
        st, en = int(s["start"]), int(s["end"])
        prev = spans[-1] if spans else None
        if prev and prev["label"] == label and prev["subtype"] == sub \
                and _should_merge(label, text[prev["end"]:st]):
            prev["end"] = en
            prev["value"] = text[prev["start"]:en]
            prev["source_labels"].append(s["label"])
        else:
            spans.append({"start": st, "end": en, "label": label, "subtype": sub,
                          "value": text[st:en], "source_labels": [s["label"]]})
    # trim whitespace that some upstream spans carry
    for sp in spans:
        v = sp["value"]
        sp["start"] += len(v) - len(v.lstrip())
        sp["end"] -= len(v) - len(v.rstrip())
        sp["value"] = text[sp["start"]:sp["end"]]
    spans = [sp for sp in spans if sp["value"]]

    lang = AI4P_LANGS.get(row["language"], row["language"][:2].lower())
    scripts = detect_scripts(text) or ["Latn"]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"ai4p:{lang}:{row['id']}",
        "text": text,
        "spans": spans,
        "label_set": "v1",
        "language": {"primary": lang, "langs": [lang], "scripts": scripts,
                     "code_mixed": False, "romanized": False},
        "domain": "other",  # AI4Privacy mixes emails, forms, letters; not labelled per row
        "region": None,
        "origin": {"method": "original", "dataset": "ai4privacy/pii-masking-300k",
                   "source_id": str(row["id"]), "parent_id": None, "template_id": None,
                   "license": "see upstream dataset card"},
        "generation": {"pipeline": "from_ai4privacy_row", "pipeline_version": "1.0",
                       "git_commit": git_commit, "seed": None, "models": []},
        "validation": {"status": "auto_passed", "checks": {"offsets_match": True}},
        "implicit": None,
        "distractors": [],
        "split": split,
        "group": f"ai4p:{row['id']}",
        "tags": [],
        "ext": {},
    }


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m src.pii_data FILE.jsonl [FILE.jsonl ...]")
        return 2
    status = 0
    for path in argv:
        r = validate_file(path)
        ok = r["invalid"] == 0
        status |= not ok
        print(f"{'OK  ' if ok else 'FAIL'} {path}: {r['records']} records, {r['invalid']} invalid"
              f"{'' if r['jsonschema_checked'] else ' (jsonschema not installed: semantic checks only)'}")
        for e in r["errors"]:
            print("   ", e)
        print("    by_lang:", r["by_lang"])
        print("    by_label:", r["by_label"])
        print("    by_status:", r["by_status"])
    return status


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))

"""Tests for the v1 record format. Run with `python tests/test_record_format.py`
or with pytest."""

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pii_data.record import from_ai4privacy_row, validate_file, validate_record  # noqa: E402

EXAMPLES = ROOT / "examples" / "records.v1.jsonl"


def _examples():
    return [json.loads(l) for l in EXAMPLES.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_examples_file_is_valid():
    r = validate_file(EXAMPLES)
    assert r["invalid"] == 0, r["errors"]


def test_value_must_match_text():
    rec = copy.deepcopy(_examples()[0])
    rec["spans"][0]["value"] = "wrong"
    assert any("!= text" in e for e in validate_record(rec))


def test_overlapping_spans_rejected():
    rec = copy.deepcopy(_examples()[0])
    s = rec["spans"][0]
    rec["spans"].insert(1, {**s, "start": s["start"] + 1, "value": rec["text"][s["start"] + 1:s["end"]]})
    assert any("overlap" in e for e in validate_record(rec))


def test_subtype_must_fit_label():
    rec = copy.deepcopy(_examples()[0])
    rec["spans"][0]["subtype"] = "NIC_NEW"  # on a PERSON span
    assert any("subtype" in e for e in validate_record(rec))


def test_undeclared_script_rejected():
    rec = copy.deepcopy(_examples()[0])
    rec["language"]["scripts"] = ["Sinh"]  # text also has Latin
    assert any("scripts" in e for e in validate_record(rec))


def test_non_nfc_text_rejected():
    rec = copy.deepcopy(_examples()[2])
    rec["text"] = rec["text"].replace("mage", "máge")  # decomposed accent
    assert any("NFC" in e for e in validate_record(rec))


def test_distractor_may_not_overlap_pii():
    rec = copy.deepcopy(_examples()[0])
    s = rec["spans"][1]
    rec["distractors"] = [{"start": s["start"], "end": s["end"], "kind": "invoice_number"}]
    assert any("distractors" in e for e in validate_record(rec))


def test_ai4privacy_merges_names_and_keeps_source_labels():
    text = "Hi Kasun Perera,\nRuwan"
    row = {"id": "X1", "language": "English", "source_text": text, "privacy_mask": [
        {"value": "Kasun", "start": 3, "end": 8, "label": "GIVENNAME1"},
        {"value": "Perera", "start": 9, "end": 15, "label": "LASTNAME1"},
        {"value": "Ruwan", "start": 17, "end": 22, "label": "GIVENNAME1"}]}
    rec = from_ai4privacy_row(row)
    assert validate_record(rec) == []
    assert [s["value"] for s in rec["spans"]] == ["Kasun Perera", "Ruwan"]
    assert rec["spans"][0]["source_labels"] == ["GIVENNAME1", "LASTNAME1"]


def test_ai4privacy_bad_offsets_skipped():
    row = {"id": "X2", "language": "English", "source_text": "abc",
           "privacy_mask": [{"value": "zz", "start": 0, "end": 2, "label": "EMAIL"}]}
    assert from_ai4privacy_row(row) is None


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                failed += 1
                print("FAIL", name, e)
    sys.exit(1 if failed else 0)

"""Offline tests for source importers (small fixtures, no downloads).
Run with `python tests/test_sources.py` or pytest."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PII_DATA_ROOT"] = tempfile.mkdtemp(prefix="pii_test_")

from src.pii_data.record import validate_record  # noqa: E402
from src.sources import aya, codemix, multiner, nemotron_pii, wildchat  # noqa: E402
from src.sources.common import Writer, build_spans  # noqa: E402


def _nem_row(text, spans, **kw):
    return {"uid": "u1", "text": text, "spans": repr(spans), "document_format": "unstructured",
            "locale": "us", "domain": "Banking", "document_type": "Form", **kw}


def test_nemotron_maps_merges_and_fixes_case():
    text = "I am Kasun Perera, black, email kp@example.com, born 1990-01-02."
    spans = [
        {"start": 5, "end": 10, "text": "Kasun", "label": "first_name"},
        {"start": 11, "end": 17, "text": "Perera", "label": "last_name"},
        {"start": 19, "end": 24, "text": "Black", "label": "race_ethnicity"},   # case differs: fine, and O
        {"start": 32, "end": 46, "text": "kp@example.com", "label": "email"},
        {"start": 53, "end": 63, "text": "1990-01-02", "label": "date_of_birth"},
    ]
    rec, reason = nemotron_pii.convert(_nem_row(text, spans), "train")
    assert reason is None and validate_record(rec) == []
    assert [(s["label"], s["value"]) for s in rec["spans"]] == [
        ("PERSON", "Kasun Perera"), ("EMAIL", "kp@example.com"), ("DOB", "1990-01-02")]
    assert rec["spans"][0]["source_labels"] == ["first_name", "last_name"]
    assert rec["tags"] == [] and rec["region"] == "US" and rec["domain"] == "document"


def test_nemotron_numeric_value_and_unmapped_tag():
    text = "Account 12345 for Nimal"
    spans = [{"start": 8, "end": 13, "text": 12345, "label": "account_number"},
             {"start": 18, "end": 23, "text": "Nimal", "label": "first_name"}]
    rec, _ = nemotron_pii.convert(_nem_row(text, spans), "test")
    assert "has_unmapped_pii" in rec["tags"] and rec["ext"]["unmapped_labels"] == ["account_number"]
    assert [s["value"] for s in rec["spans"]] == ["Nimal"]


def test_nemotron_rejects_real_mismatch_and_overlap():
    rec, reason = nemotron_pii.convert(_nem_row("abc def", [{"start": 0, "end": 3, "text": "xyz",
                                                             "label": "first_name"}]), "train")
    assert rec is None and reason == "offset_mismatch"
    assert build_spans("Kasun Perera", [(0, 12, "PERSON", None, "a"), (6, 12, "PERSON", None, "b")]) is None


def test_build_spans_trims_whitespace():
    spans = build_spans("x  Colombo  y", [(1, 11, "LOCATION", None, "city")])
    assert spans[0]["value"] == "Colombo" and spans[0]["start"] == 3


def test_wildchat_filters():
    conv = lambda t: [{"role": "user", "content": t}, {"role": "assistant", "content": "ok"}]
    row = {"conversation_hash": "h1", "language": "English", "toxic": False, "redacted": True,
           "conversation": conv("Write a cover letter for my application to the bank, my name is X.")}
    part, rec, reason = wildchat.convert(row)
    assert part == "prompts" and validate_record(rec) == []
    assert rec["tags"] == ["needs_placeholders", "upstream_redacted"] and rec["spans"] == []
    _, _, reason = wildchat.convert({**row, "conversation": conv("Tell me a joke about cats please, a long one.")})
    assert reason == "intent"
    _, _, reason = wildchat.convert({**row, "toxic": True})
    assert reason == "toxic"
    part, rec, _ = wildchat.convert({**row, "language": "Sinhala",
                                     "conversation": conv("මට ලිපියක් ලියන්න")})
    assert part == "native" and rec["language"]["primary"] == "si" and validate_record(rec) == []


def test_aya_language_and_intent():
    rec, _ = aya.convert({"inputs": "රැකියා අයදුම්පතක් සඳහා ලිපියක් ලියන්න.", "targets": "...",
                          "language_code": "sin", "annotation_type": "original-annotations"})
    assert validate_record(rec) == [] and "personal_intent" in rec["tags"]
    assert rec["language"] == {"primary": "si", "langs": ["si"], "scripts": ["Sinh"],
                               "code_mixed": False, "romanized": False}
    rec, _ = aya.convert({"inputs": "Python என்றால் என்ன? விளக்குங்கள்.", "targets": "",
                          "language_code": "tam", "annotation_type": "x"})
    assert rec["language"]["code_mixed"] and rec["language"]["langs"] == ["ta", "en"]
    assert aya.convert({"inputs": "hello there friend", "language_code": "eng"})[1] == "other_language"


def test_multiner_bio_to_spans():
    conll = ("Mr O\nKasun B-PER\nPerera I-PER\nvisited O\nKandy B-LOC\n( O\nLK B-LOC\n) O\n. O\n\n"
             "Ministry B-ORG\nof I-ORG\nHealth I-ORG\n2013 B-MISC\n")
    sents = multiner.read_conll(conll)
    assert len(sents) == 2
    rec, _ = multiner.convert(0, "en", sents[0])
    assert rec["text"] == "Mr Kasun Perera visited Kandy (LK)."
    assert [(s["label"], s["value"]) for s in rec["spans"]] == [
        ("PERSON", "Kasun Perera"), ("LOCATION", "Kandy"), ("LOCATION", "LK")]
    assert validate_record(rec) == [] and rec["validation"]["status"] == "human_verified"
    rec2, _ = multiner.convert(1, "en", sents[1])
    assert rec2["spans"] == [] and rec2["ext"]["dropped_types"] == ["MISC", "ORG"]


def test_multiner_adjacent_entities_stay_separate():
    text, ents = multiner.detok([("Galle", "B-LOC"), ("Matara", "B-LOC"), ("and", "O"), ("X", "I-PER")])
    assert ents == [(0, 5, "LOC"), (6, 12, "LOC"), (17, 18, "PER")]


def test_codemix_filters_unknown_tokens():
    assert codemix.convert("triplets", {"Sinhala": "අ", "English": "a", "Singlish": "[Unkown] a"})[1] == "unknown_tokens"
    item, _ = codemix.convert("singlish_gold", {"Romanized_Sinhala": "mama yanawa", "Sinhala": "මම යනවා"})
    assert item == {"rom": "mama yanawa", "si": "මම යනවා"}


def test_writer_rejects_invalid_and_duplicates():
    w = Writer("unit", "part")
    rec, _ = aya.convert({"inputs": "රැකියා අයදුම්පතක් සඳහා ලිපියක් ලියන්න.", "language_code": "sin"})
    assert w.write(rec) and not w.write(rec)
    bad = dict(rec, id="aya:si:other", spans=[{"start": 0, "end": 3, "label": "PERSON", "value": "zzz"}])
    assert not w.write(bad)
    w.close()
    assert w.written == 1 and w.skipped == {"duplicate_id": 1, "invalid_record": 1}
    assert w.rej_path.exists()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:
                failed += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if failed else 0)

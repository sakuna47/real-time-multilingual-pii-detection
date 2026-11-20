"""Offline tests for deterministic checks, splits and hard negatives.
Run with `python tests/test_validate.py` or pytest."""

import copy
import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PII_DATA_ROOT"] = tempfile.mkdtemp(prefix="pii_test_")

from src.pii_data.record import validate_record  # noqa: E402
from src.pii_data.split import run as split_run, split_of  # noqa: E402
from src.sl_gen import negatives  # noqa: E402
from src.validate import checks  # noqa: E402

TEXT = "මගේ නම Kasun Perera. NIC 199512304567, phone 077 123 4567. loan එකක් apply කරන්න ඕනේ."


def _rec(text=TEXT, variety="si-en", **kw):
    spans = []
    for value, label, sub in [("Kasun Perera", "PERSON", None), ("199512304567", "ID_NUMBER", "NIC_NEW"),
                              ("077 123 4567", "PHONE", "MOBILE")]:
        if value in text:
            st = text.index(value)
            spans.append({"start": st, "end": st + len(value), "label": label, "subtype": sub, "value": value})
    rec = {"schema_version": "1.0", "id": "t:si:1", "text": text, "spans": spans, "label_set": "v1",
           "language": {"primary": "si", "langs": ["si", "en"], "scripts": ["Sinh", "Latn"],
                        "code_mixed": True, "romanized": False},
           "domain": "chat_prompt", "origin": {"method": "translated_placeholder"},
           "validation": {"status": "auto_passed"}, "group": "g1", "tags": [], "ext": {"variety": variety}}
    rec.update(kw)
    return rec


def test_checks_pass_good_record():
    c = checks.check_record(_rec(), set())
    assert all(v is not False for v in c.values()), c
    assert c["cmi_value"] > 5


def test_checks_catch_problems():
    seen = set()
    checks.check_record(_rec(), seen)
    assert checks.check_record(_rec(), seen)["duplicate_text"] is False
    bad = _rec(text=TEXT.replace("ඕනේ.", "ඕනේ {EMAIL_1}."))
    assert checks.check_record(bad, set())["no_placeholders"] is False
    rom = _rec(variety="si-rom")                       # Sinhala script in a romanised record
    assert checks.check_record(rom, set())["variety_scripts"] is False
    badnic = _rec()                                    # same length, so offsets stay valid
    badnic["text"] = TEXT.replace("199512304567", "199599904567")
    badnic["spans"][1]["value"] = "199599904567"
    assert checks.check_record(badnic, set())["nic_format"] is False   # day 999 does not exist


def test_rerun_keeps_human_verdicts():
    src, dst = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    (src / "x").mkdir()
    (src / "x" / "si-en.jsonl").write_text(json.dumps(_rec(), ensure_ascii=False) + "\n", encoding="utf-8")
    checks.run(src, dst)
    out = dst / "x" / "si-en.jsonl"
    rec = json.loads(out.read_text(encoding="utf-8"))
    rec["validation"]["human"] = [{"annotator": "a1", "verdict": "ok"}]
    rec["validation"]["status"] = "human_verified"
    out.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    checks.run(src, dst)                                   # re-run must not wipe the review
    again = json.loads(out.read_text(encoding="utf-8"))
    assert again["validation"]["status"] == "human_verified" and again["validation"]["human"][0]["verdict"] == "ok"


def test_cmi():
    assert checks.cmi("මම යනවා ගෙදර", "Sinh") == 0
    assert checks.cmi("මම office යනවා", "Sinh") == 33.33
    assert checks.cmi("12345 ...", "Sinh") == 0


def test_split_stable_and_group_safe():
    assert split_of("a") == split_of("a")
    counts = {"train": 0, "dev": 0, "test": 0}
    for i in range(5000):
        counts[split_of(f"g{i}")] += 1
    assert 0.77 < counts["train"] / 5000 < 0.83 and 0.08 < counts["test"] / 5000 < 0.12
    src = Path(tempfile.mkdtemp())
    with open(src / "x.jsonl", "w", encoding="utf-8") as f:
        for i in range(300):
            for variety in ("si", "si-rom"):   # two variants of the same source text
                r = _rec(id=f"t:si:{i}-{variety}", group=f"src{i}")
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps(_rec(id="t:si:failed", group="z", validation={"status": "auto_failed"}),
                           ensure_ascii=False) + "\n")
    out = Path(tempfile.mkdtemp())
    c = split_run(src, out)
    assert c["dropped_status"] == 1 and c["train"] + c["dev"] + c["test"] == 600
    for s in ("train", "dev", "test"):
        groups = [json.loads(l)["group"] for l in open(out / f"{s}.jsonl", encoding="utf-8")]
        assert len(groups) % 2 == 0      # both variants of every group are together


def test_negative_record_is_valid():
    rng = random.Random(0)
    for variety, text in [("si", "අන්නාසි වගා කරන්නේ කොහොම ද?"), ("ta", "மழை எப்போது வரும்?"),
                          ("si-en", "whatsapp status එක load වෙන්නේ නැහැ"), ("si-rom", "exam eka heta")]:
        rec = negatives.make("aya:si:abc", text, {"dataset": "x", "license": "y"}, variety, rng)
        assert rec and validate_record(rec) == [] and rec["spans"] == []
        d = rec["distractors"][0]
        assert rec["text"][d["start"]:d["end"]].replace("-", "").isalnum()
        assert checks.check_record(rec, set()).get("variety_scripts", True)


def test_invoice_values_look_like_nic_on_purpose():
    rng = random.Random(1)
    v = negatives.value_for("invoice_number", rng)
    assert len(v) == 12 and v.isdigit() and 1950 <= int(v[:4]) <= 2024
    assert negatives.value_for("tracking_number", rng).startswith("07")


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

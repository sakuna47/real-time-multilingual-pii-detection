"""Deterministic checks for generated records (no LLM, no cost).

    python -m src.validate.checks data/generated/sri_lankan/nemotron            # one folder
    python -m src.validate.checks data/generated/sri_lankan                     # all sources

Each record gets validation.checks filled in and validation.status set to auto_passed or
auto_failed. Results go to <data_root>/validated/<same relative path>/ plus checks_report.json;
the generated files are not modified.

Checks
  schema            validate_record() (offsets, NFC, labels, declared scripts, ...)
  no_placeholders   no {LABEL_n} token left in the text
  variety_scripts   native script present (si, ta, *-en), absent (*-rom); both scripts in *-en
  nic_format        every NIC span decodes (old or new format)
  nic_dob           tag nic_dob_consistent is backed by a decodable NIC
  phone_format      every PHONE span is a Sri Lankan number (+94 / 0094 / 0 + 9 digits)
  email_format      every EMAIL span looks like an address
  cmi               Code-Mixing Index (Das & Gambäck) for *-en; fails below MIN_CMI (barely mixed)
  duplicate_text    exact duplicate text seen earlier in this run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from ..common.paths import data_root
from ..pii_data.record import detect_scripts, validate_record
from ..sl_gen import nic
from ..sl_gen.varieties import VARIETIES
from ..translate import placeholders as ph

PHONE_RE = re.compile(r"^(?:\+94|0094|0)[\s-]?\d{2}[\s-]?\d{3}[\s-]?\d{4}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
MIN_CMI = 5.0
_TOKEN = re.compile(r"\w+", re.U)
SCRIPT_OF = {"Sinh": re.compile(r"[඀-෿]"), "Taml": re.compile(r"[஀-௿]"),
             "Latn": re.compile(r"[A-Za-z]")}


def _without_spans(rec: dict) -> str:
    """Text with PII values blanked, so names in Latin script don't count as English words."""
    text, out, pos = rec["text"], [], 0
    for s in sorted(rec["spans"], key=lambda s: s["start"]):
        out.append(text[pos:s["start"]])
        out.append(" ")
        pos = s["end"]
    out.append(text[pos:])
    return "".join(out)


def cmi(text: str, native: str) -> float:
    """Code-Mixing Index: 100 * (1 - max_lang_tokens / language_tokens); 0 = monolingual.
    Tokens are assigned to a language by script (native vs Latin); digits etc. are neutral."""
    counts = Counter()
    for tok in _TOKEN.findall(text):
        if SCRIPT_OF[native].search(tok):
            counts[native] += 1
        elif SCRIPT_OF["Latn"].search(tok):
            counts["Latn"] += 1
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return round(100 * (1 - max(counts.values()) / n), 2)


def check_record(rec: dict, seen: set[str]) -> dict:
    c: dict = {}
    errors = validate_record(rec)
    c["schema"] = not errors
    if errors:
        c["schema_errors"] = errors[:3]
    c["no_placeholders"] = not ph.find(rec["text"])

    code = (rec.get("ext") or {}).get("variety")
    v = VARIETIES.get(code)
    if v:
        scripts = set(detect_scripts(_without_spans(rec)))
        if v.romanized:
            c["variety_scripts"] = v.native_script not in scripts and "Latn" in scripts
        elif v.code_mixed:
            c["variety_scripts"] = {v.native_script, "Latn"} <= scripts
        else:
            c["variety_scripts"] = v.native_script in scripts
        if v.code_mixed and not v.romanized:
            c["cmi_value"] = cmi(_without_spans(rec), v.native_script)
            c["cmi"] = c["cmi_value"] >= MIN_CMI

    nics = [s for s in rec["spans"] if s["label"] == "ID_NUMBER" and (s.get("subtype") or "").startswith("NIC")]
    if nics:
        def decodes(value: str) -> bool:
            try:
                nic.decode(value)
                return True
            except ValueError:
                return False
        c["nic_format"] = all(decodes(s["value"]) for s in nics)
    if "nic_dob_consistent" in rec.get("tags", []):
        c["nic_dob"] = bool(nics) and c.get("nic_format", False)
    phones = [s["value"] for s in rec["spans"] if s["label"] == "PHONE"]
    if phones:
        c["phone_format"] = all(PHONE_RE.match(p) for p in phones)
    emails = [s["value"] for s in rec["spans"] if s["label"] == "EMAIL"]
    if emails:
        c["email_format"] = all(EMAIL_RE.match(e) for e in emails)
    key = rec["text"].casefold()
    c["duplicate_text"] = key not in seen   # True = not a duplicate
    seen.add(key)
    return c


def run(src: Path, dst: Path) -> dict:
    report: dict = {"files": {}, "failed_checks": Counter()}
    seen: set[str] = set()
    for path in sorted(src.rglob("*.jsonl")):
        if path.name == "rejected.jsonl":
            continue
        rel = path.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = passed = 0
        with open(path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8", newline="\n") as fout:
            for line in fin:
                if not line.strip():
                    continue
                rec = json.loads(line)
                checks = check_record(rec, seen)
                failed = [k for k, val in checks.items() if val is False]
                val = rec.setdefault("validation", {})
                val.setdefault("checks", {}).update(checks)
                if val.get("status") in (None, "unchecked", "auto_passed", "auto_failed"):
                    val["status"] = "auto_failed" if failed else "auto_passed"
                report["failed_checks"].update(failed)
                n += 1
                passed += not failed
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        report["files"][str(rel)] = {"records": n, "passed": passed}
    report["failed_checks"] = dict(report["failed_checks"].most_common())
    report["records"] = sum(f["records"] for f in report["files"].values())
    report["passed"] = sum(f["passed"] for f in report["files"].values())
    (dst / "checks_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder under <data_root>/generated (or an absolute path)")
    args = ap.parse_args(argv)
    src = args.folder if args.folder.is_absolute() else Path.cwd() / args.folder
    if not src.exists():
        src = data_root() / args.folder
    try:
        rel = src.resolve().relative_to((data_root() / "generated").resolve())
    except ValueError:
        rel = Path(src.name)
    dst = data_root() / "validated" / rel
    r = run(src, dst)
    print(f"{r['passed']}/{r['records']} records passed -> {dst}")
    print("failed checks:", r["failed_checks"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

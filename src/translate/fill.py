"""Masked target text + slots -> v1 record with exact spans.

Placeholders are replaced left to right; each value's start/end is recorded while the
string is built, so spans are correct by construction. The same placeholder always gets
the same text. Each person entity gets its own persona; values not tied to a name use the
first person's persona, so NIC, date of birth and sex agree within a record.
"""

from __future__ import annotations

import hashlib
import random
import unicodedata

from ..common.runinfo import git_commit, utc_now
from ..pii_data.record import SCHEMA_VERSION, detect_scripts, validate_record
from ..sl_gen import nic as nicmod
from ..sl_gen import values
from ..sl_gen.personas import PersonaFactory
from ..sl_gen.varieties import Variety
from . import placeholders as ph


def _k(*parts: str) -> int:
    return int(hashlib.sha256(":".join(parts).encode()).hexdigest()[:12], 16)


def fill(text: str, slots: dict[str, dict], v: Variety, factory: PersonaFactory, *, seed_key: str
         ) -> tuple[dict | None, str | None]:
    """Returns ({text, spans, language, persona_ids, formats, tags}, None) or (None, reason)."""
    found = ph.find(text)
    if not found:
        return None, "fill: no placeholders"
    if ph.stray(text):
        return None, f"fill: invalid tokens {ph.stray(text)[:3]}"
    missing = {t for t, *_ in found} - set(slots)
    if missing:
        return None, f"fill: unknown placeholders {sorted(missing)}"

    # personas: people in order of first appearance; 'primary' = first person (or its own persona)
    order: list[str] = []
    for tok, *_ in found:
        e = slots[tok]["entity"]
        if e not in order:
            order.append(e)
    people = [e for e in order if e != "primary"]
    personas = {e: factory.make(_k(seed_key, e) % 1_000_000) for e in people}
    personas["primary"] = personas[people[0]] if people else factory.make(_k(seed_key, "primary") % 1_000_000)

    rng = random.Random(_k(seed_key, "values"))
    script_of = {e: values.script_for(v, rng) for e in order}   # one script per person per record
    rendered: dict[str, tuple[str, str]] = {}
    out, spans, pos = [], [], 0
    length = 0
    for tok, label, st, en in found:
        seg = unicodedata.normalize("NFC", text[pos:st])
        out.append(seg)
        length += len(seg)
        if tok not in rendered:
            s = slots[tok]
            val, fmt = values.render(label, s.get("subtype"), personas[s["entity"]], v, rng,
                                     s.get("part", "full"), script_of[s["entity"]])
            rendered[tok] = (unicodedata.normalize("NFC", val), fmt)
        val, fmt = rendered[tok]
        s = slots[tok]
        sub = s.get("subtype")
        if label == "ID_NUMBER" and not sub:
            sub = "NIC_OLD" if fmt.startswith("nic:old") else "NIC_NEW" if fmt.startswith("nic:new") else None
        if label == "PHONE" and not sub:
            sub = "LANDLINE" if s.get("subtype") == "LANDLINE" else "MOBILE"
        spans.append({"start": length, "end": length + len(val), "label": label, "subtype": sub,
                      "value": val, "entity_id": s["entity"], "source_labels": [],
                      "script": (detect_scripts(val) or [None])[0]})
        out.append(val)
        length += len(val)
        pos = en
    tail = unicodedata.normalize("NFC", text[pos:])
    out.append(tail)
    final = "".join(out)
    if unicodedata.normalize("NFC", final) != final:
        return None, "fill: text not NFC at a value boundary"
    for sp in spans:   # values are plain strings; make sure offsets really point at them
        if final[sp["start"]:sp["end"]] != sp["value"]:
            return None, "fill: offset mismatch"

    scripts = set(detect_scripts(final))
    tags = []
    p = personas["primary"]
    has_nic = any(sp["label"] == "ID_NUMBER" and (sp["subtype"] or "").startswith("NIC") for sp in spans)
    if has_nic and any(sp["label"] == "DOB" for sp in spans):
        nic_ok = all(nicmod.consistent(sp["value"], p.dob, p.female) for sp in spans
                     if sp["label"] == "ID_NUMBER" and sp["entity_id"] in ("primary", people[0] if people else "primary"))
        if nic_ok:
            tags.append("nic_dob_consistent")
    return {
        "text": final, "spans": spans,
        "language": v.language_block(has_latin="Latn" in scripts, has_native=v.native_script in scripts),
        "persona_ids": sorted({pp.id for pp in personas.values()}),
        "formats": [rendered[t][1] for t in dict.fromkeys(t for t, *_ in found)],
        "tags": tags,
    }, None


def build_record(*, filled: dict, source: dict, variety: Variety, method: str, masked_target: str,
                 models: list[dict], extra_tags: list[str], seed: int, pipeline_version: str) -> dict:
    """Wrap a fill() result as a full v1 record and validate it."""
    src_prefix = source["id"].split(":", 1)[0]
    local = source["id"].rsplit(":", 1)[-1]
    rec = {
        "schema_version": SCHEMA_VERSION,
        "id": f"{src_prefix}-tr:{variety.lang}:{local}-{variety.code}",
        "text": filled["text"], "spans": filled["spans"], "label_set": "v1",
        "language": filled["language"], "domain": source.get("domain", "chat_prompt"), "region": "LK",
        "origin": {"method": method, "dataset": (source.get("origin") or {}).get("dataset"),
                   "source_id": (source.get("origin") or {}).get("source_id"), "parent_id": source["id"],
                   "template_id": None, "license": (source.get("origin") or {}).get("license")},
        "generation": {"pipeline": "src.translate.pipeline", "pipeline_version": pipeline_version,
                       "git_commit": git_commit(), "seed": seed, "created_at": utc_now(), "models": models,
                       "filler": {"name": "src.sl_gen.values", "version": "1.0"},
                       "persona_id": filled["persona_ids"][0] if filled["persona_ids"] else None},
        "validation": {"status": "auto_passed",
                       "checks": {"placeholders_preserved": True, "spans_match": True, "script_ok": True}},
        "implicit": None, "distractors": [], "split": None, "group": source["id"],
        "tags": sorted(set(filled["tags"] + extra_tags)),
        "ext": {"variety": variety.code, "value_formats": filled["formats"],
                "persona_ids": filled["persona_ids"], "masked_target": masked_target},
    }
    errors = validate_record(rec)
    if errors:
        rec["validation"] = {"status": "auto_failed", "checks": {"errors": errors[:5]}}
    return rec

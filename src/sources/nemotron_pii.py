"""nvidia/Nemotron-PII -> v1 records (English, span-labelled). CC BY 4.0.

    python -m src.sources.run nemotron [--limit N] [--max-chars 1500]

Label mapping keeps only labels that exist in v1. Other identifiers (account numbers,
card numbers, medical record numbers, ...) are not in v1; records containing them are
kept but tagged `has_unmapped_pii` and listed in ext.unmapped_labels, so the translation
pipeline can skip them instead of producing records with unlabelled PII.
Attributes that are not identifiers in v1 (dates, company, occupation, gender, ...) are O.
"""

from __future__ import annotations

import ast
from typing import Any

from .common import Writer, base_record, build_spans, finish, hf_file, language_block, norm_text

REPO_ID = "nvidia/Nemotron-PII"
LICENSE = "CC BY 4.0"
FILES = {"train": "data/train-00000-of-00001.parquet", "test": "data/test-00000-of-00001.parquet"}

MAP: dict[str, tuple[str, str | None]] = {
    "first_name": ("PERSON", None), "last_name": ("PERSON", None),
    "ssn": ("ID_NUMBER", "SOCIAL_SECURITY"), "national_id": ("ID_NUMBER", "NATIONAL_ID"),
    "tax_id": ("ID_NUMBER", "OTHER"),
    "phone_number": ("PHONE", None), "fax_number": ("PHONE", "LANDLINE"),
    "email": ("EMAIL", None),
    "street_address": ("ADDRESS", None), "postcode": ("ADDRESS", None),
    "city": ("LOCATION", None), "coordinate": ("LOCATION", None),
    "date_of_birth": ("DOB", None),
    "user_name": ("USERNAME", None),
    "password": ("PASSWORD", None),
    "ipv4": ("IP_ADDRESS", None), "ipv6": ("IP_ADDRESS", None),
}
# Identifiers v1 has no label for (v2 candidates). Their presence makes a record unsafe to translate as-is.
UNMAPPED_PII = {
    "account_number", "credit_debit_card", "bank_routing_number", "swift_bic", "cvv", "pin",
    "customer_id", "employee_id", "medical_record_number", "health_plan_beneficiary_number",
    "biometric_identifier", "certificate_license_number", "vehicle_identifier", "license_plate",
    "device_identifier", "mac_address", "unique_id", "api_key", "http_cookie",
}
# Not identifiers under v1 -> O (text stays as is).
# url: decided 2026-11-20 to treat as O; in Nemotron they are mostly company/service websites.
NOT_PII = {
    "date", "time", "date_time", "company_name", "occupation", "country", "state", "county",
    "employment_status", "education_level", "language", "gender", "age", "race_ethnicity",
    "political_view", "religious_belief", "sexuality", "blood_type", "url",
}


def parse_spans(raw: Any) -> list[dict]:
    if isinstance(raw, str):
        return ast.literal_eval(raw)
    return list(raw)


def convert(row: dict, source_split: str) -> tuple[dict | None, str | None]:
    """One Nemotron row -> (record, None) or (None, skip_reason)."""
    text = row["text"]
    if norm_text(text) != text:
        return None, "not_nfc"
    raw, unmapped, unknown = [], set(), set()
    for s in parse_spans(row["spans"]):
        st, en, label = int(s["start"]), int(s["end"]), s["label"]
        piece = text[st:en]
        # stored values are sometimes lower-cased or numeric; the text at the offsets is the truth
        if piece.casefold() != str(s["text"]).casefold():
            return None, "offset_mismatch"
        if label in MAP:
            v1, sub = MAP[label]
            raw.append((st, en, v1, sub, label))
        elif label in UNMAPPED_PII:
            unmapped.add(label)
        elif label not in NOT_PII:
            unknown.add(label)
    spans = build_spans(text, raw)
    if spans is None:
        return None, "overlapping_spans"
    tags = []
    if unmapped or unknown:
        tags.append("has_unmapped_pii")
    domain = "form" if row.get("document_format") == "structured" else "document"
    rec = base_record(
        rid=f"nemotron:en:{row['uid']}", text=text, spans=spans,
        lang=language_block("en", text), domain=domain, dataset=REPO_ID, source_id=str(row["uid"]),
        license=LICENSE, pipeline="src.sources.nemotron_pii",
        region="US" if row.get("locale") == "us" else None, tags=tags,
        ext={"source_split": source_split, "nemotron_domain": row.get("domain"),
             "document_type": row.get("document_type"), "locale": row.get("locale"),
             "unmapped_labels": sorted(unmapped | unknown)})
    return rec, None


def run(limit: int | None = None, max_chars: int = 1500) -> None:
    import pyarrow.parquet as pq
    writers = []
    for split, filename in FILES.items():
        table = pq.read_table(hf_file(REPO_ID, filename),
                              columns=["uid", "domain", "document_type", "document_format",
                                       "locale", "text", "spans"])
        w = Writer("nemotron_pii", split)
        for i, row in enumerate(table.to_pylist()):
            if limit and i >= limit:
                break
            if max_chars and len(row["text"]) > max_chars:
                w.skip("too_long")
                continue
            rec, reason = convert(row, split)
            if rec is None:
                w.skip(reason)
            else:
                w.write(rec)
        writers.append(w)
    finish("nemotron_pii", writers, license=LICENSE, source=f"https://huggingface.co/datasets/{REPO_ID}",
           config={"limit": limit, "max_chars": max_chars, "map": {k: list(v) for k, v in MAP.items()},
                   "unmapped_pii": sorted(UNMAPPED_PII)})

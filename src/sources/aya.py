"""CohereLabs/aya_dataset -> human-written Sinhala and Tamil prompts. Apache-2.0.

    python -m src.sources.run aya

Writes <data_root>/sources/aya/{si,ta}.jsonl. Prompts (`inputs`) are kept whatever the
topic; tag `personal_intent` marks ones that look like personal tasks, which are the
first candidates for placeholder insertion.
"""

from __future__ import annotations

import re

from .common import Writer, base_record, finish, hf_file, language_block, norm_text, short_hash

REPO_ID = "CohereLabs/aya_dataset"
LICENSE = "Apache-2.0"
FILES = ("data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet")
LANGS = {"sin": "si", "tam": "ta"}

# Sinhala / Tamil / English cue words for personal tasks (letters, applications, forms, names...)
PERSONAL = re.compile(
    # Tasks where a user would naturally include personal details (letters, applications, forms,
    # complaints, emails, CVs/bios). Single attribute words like "name"/"නම" were dropped: නම also
    # matches නම් ("if"), and "form" matched inside "information".
    r"(ලිපිය|ලිපියක්|අයදුම්පත|අයදුම්|ඉල්ලුම්පත|පෝරමය|පෝරමයක්|පැමිණිලි|ඊමේල්|ඊ-මේල්|ජීව දත්ත|"
    r"கடிதம்|கடிதத்தை|விண்ணப்ப|படிவம்|படிவத்தை|புகார்|மின்னஞ்சல்|சுயவிவர|"
    r"\b(?:letter|application|cover letter|form|complaint|e-?mail|resume|cv|bio)\b)", re.I)


def convert(row: dict) -> tuple[dict | None, str | None]:
    code = row.get("language_code")
    if code not in LANGS:
        return None, "other_language"
    text = norm_text((row.get("inputs") or "").strip())
    if len(text) < 10:
        return None, "too_short"
    lang = LANGS[code]
    tags = ["needs_placeholders", "auto_lang"]
    if PERSONAL.search(text):
        tags.append("personal_intent")
    h = short_hash(text)
    rec = base_record(
        rid=f"aya:{lang}:{h}", text=text, spans=[], lang=language_block(lang, text),
        domain="chat_prompt", dataset=REPO_ID, source_id=h, license=LICENSE,
        pipeline="src.sources.aya", region=None, tags=tags,
        ext={"annotation_type": row.get("annotation_type"), "target_chars": len(row.get("targets") or "")})
    return rec, None


def run(limit: int | None = None) -> None:
    import pyarrow.parquet as pq
    writers = {lang: Writer("aya", lang) for lang in LANGS.values()}
    for filename in FILES:
        table = pq.read_table(hf_file(REPO_ID, filename),
                              columns=["inputs", "targets", "language_code", "annotation_type"])
        mask_rows = [r for r in table.to_pylist() if r.get("language_code") in LANGS]
        for row in mask_rows:
            rec, reason = convert(row)
            lang = LANGS[row["language_code"]]
            if rec is None:
                writers[lang].skip(reason)
            elif not limit or writers[lang].written < limit:
                writers[lang].write(rec)
    finish("aya", list(writers.values()), license=LICENSE,
           source=f"https://huggingface.co/datasets/{REPO_ID}", config={"limit": limit})

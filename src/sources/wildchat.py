"""allenai/WildChat-1M -> first-turn user prompts (real chatbot prompts). ODC-BY.

    python -m src.sources.run wildchat [--shards N] [--limit N]

Writes two files to <data_root>/sources/wildchat/:
  prompts.jsonl  English first-turn prompts whose intent suggests personal details
                 (letters, CVs, applications, forms, ...). No spans yet: PII was redacted
                 upstream, so placeholders are inserted later (tag needs_placeholders).
  native.jsonl   every first-turn prompt WildChat labels Sinhala or Tamil, kept whatever
                 the intent: real native prompts are rare and valuable as style references.

Shards are downloaded one at a time (~200 MB each, 14 in total) to keep disk use low.
"""

from __future__ import annotations

import re

from .common import Writer, base_record, finish, hf_file, language_block, norm_text, short_hash

REPO_ID = "allenai/WildChat-1M"
LICENSE = "ODC-BY"
N_SHARDS = 14
NATIVE = {"Sinhala": "si", "Tamil": "ta"}

INTENT = re.compile(
    r"\b(letter|e-?mail|cover letter|cv|resume|résumé|application|apply|form|visa|passport|"
    r"loan|bank|account|landlord|tenant|complaint|appointment|doctor|hospital|insurance|"
    r"invoice|refund|reference letter|recommendation letter|my name is|introduce myself|"
    r"reply to|respond to|message to|write (a|an) (message|note|email|letter)|bio|profile|"
    r"leave request|resignation|job|interview|admission|scholarship|certificate)\b", re.I)
MIN_CHARS, MAX_CHARS = 40, 1500


def first_user_turn(conversation) -> str | None:
    for turn in conversation or []:
        if turn.get("role") == "user":
            return (turn.get("content") or "").strip() or None
    return None


def convert(row: dict) -> tuple[str | None, dict | None, str | None]:
    """Returns (part, record, skip_reason); part is 'prompts' or 'native'."""
    text = first_user_turn(row.get("conversation"))
    if not text:
        return None, None, "no_user_turn"
    text = norm_text(text)
    lang_name = row.get("language")
    if row.get("toxic"):
        return None, None, "toxic"
    if lang_name in NATIVE:
        lang, part = NATIVE[lang_name], "native"
    elif lang_name == "English":
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            return None, None, "length"
        if not INTENT.search(text):
            return None, None, "intent"
        lang, part = "en", "prompts"
    else:
        return None, None, "other_language"
    h = row.get("conversation_hash") or short_hash(text)
    tags = ["needs_placeholders"] + (["upstream_redacted"] if row.get("redacted") else [])
    if part == "native":
        tags.append("auto_lang")
    rec = base_record(
        rid=f"wildchat:{lang}:{h}", text=text, spans=[], lang=language_block(lang, text),
        domain="chat_prompt", dataset=REPO_ID, source_id=str(h), license=LICENSE,
        pipeline="src.sources.wildchat", tags=tags,
        ext={"wildchat_language": lang_name, "model": row.get("model"),
             "country": row.get("country"), "redacted": bool(row.get("redacted"))})
    return part, rec, None


def run(shards: int | None = None, limit: int | None = None) -> None:
    import pyarrow.parquet as pq
    writers = {"prompts": Writer("wildchat", "prompts"), "native": Writer("wildchat", "native")}
    seen_text: set[str] = set()
    total = 0
    for k in range(shards or N_SHARDS):
        path = hf_file(REPO_ID, f"data/train-{k:05d}-of-{N_SHARDS:05d}.parquet")
        table = pq.read_table(path, columns=["conversation_hash", "conversation", "language",
                                             "toxic", "redacted", "model", "country"])
        for row in table.to_pylist():
            part, rec, reason = convert(row)
            if rec is None:
                writers["prompts"].skip(reason)
                continue
            key = rec["text"].casefold()
            if key in seen_text:
                writers[part].skip("duplicate_text")
                continue
            seen_text.add(key)
            writers[part].write(rec)
            total += 1
            if limit and total >= limit:
                break
        print(f"shard {k}: prompts {writers['prompts'].written}, native {writers['native'].written}",
              flush=True)
        if limit and total >= limit:
            break
        try:  # free disk: shards are large and only read once
            path.unlink()
        except OSError:
            pass
    finish("wildchat", list(writers.values()), license=LICENSE,
           source=f"https://huggingface.co/datasets/{REPO_ID}",
           config={"shards": shards or N_SHARDS, "limit": limit, "intent_regex": INTENT.pattern,
                   "min_chars": MIN_CHARS, "max_chars": MAX_CHARS})

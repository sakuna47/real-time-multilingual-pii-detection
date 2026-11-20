"""suralk/multiNER -> parallel en/si/ta sentences with human NER labels. CC0-1.0 (GitHub repo).

    python -m src.sources.run multiner

3,835 sentences per language from Sri Lankan government documents, CoNLL-03 tags.
PER -> PERSON and LOC -> LOCATION; ORG and MISC are not PII and are dropped.
Sentence i is the same sentence in all three languages, so they share group
"multiner:<i>" and always land in the same split.

Note: LOC here means any place name (not only places tied to a person), so these records
carry the tag `loc_generic`; report LOCATION scores on them separately.
"""

from __future__ import annotations

import zipfile

from .common import Writer, base_record, finish, language_block, norm_text, url_file

URL = "https://github.com/suralk/multiNER/raw/main/nerannotateddatasets.zip"
LICENSE = "CC0-1.0"
CITATION = ("Ranathunga, S., Ranasinghe, A., Shamal, J., Dandeniya, A., Galappaththi, R., & "
            "Samaraweera, M. (2025). A Multi-way parallel named entity annotated corpus for English, "
            "Tamil and Sinhala. Natural Language Processing Journal, 100160.")
FILES = {"en": "Final_English.txt", "si": "Final_Sinhala.txt", "ta": "Final_Tamil.txt"}
TAGMAP = {"PER": "PERSON", "LOC": "LOCATION"}
NO_SPACE_BEFORE = set(".,;:!?)]}%'\"")
NO_SPACE_AFTER = set("([{")


def read_conll(text: str) -> list[list[tuple[str, str]]]:
    sents, cur = [], []
    for line in text.splitlines():
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.rsplit(" ", 1)
        tok, tag = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "O")
        # NFC per token, so the joined text is NFC and offsets are computed on the final text
        cur.append((norm_text(tok.strip() or tok), tag.strip() or "O"))
    if cur:
        sents.append(cur)
    return sents


def detok(tokens: list[tuple[str, str]]) -> tuple[str, list[tuple[int, int, str]]]:
    """Join tokens with spaces (none before closing punctuation), returning text and
    character offsets of each BIO entity: (start, end, conll_type)."""
    text = ""
    ents: list[list] = []
    open_ent: list | None = None   # [start, end, type] of the entity being extended
    prev_tok = None
    for tok, tag in tokens:
        if text and tok[0] not in NO_SPACE_BEFORE and (prev_tok is None or prev_tok[-1] not in NO_SPACE_AFTER):
            text += " "
        start = len(text)
        text += tok
        kind = tag[2:] if tag[:2] in ("B-", "I-") else None
        if kind is None:
            open_ent = None
        elif tag.startswith("B-") or open_ent is None or open_ent[2] != kind:
            open_ent = [start, len(text), kind]   # B-, or an I- that starts a new entity
            ents.append(open_ent)
        else:
            open_ent[1] = len(text)
        prev_tok = tok
    return text, [(s, e, t) for s, e, t in ents]


def convert(i: int, lang: str, tokens: list[tuple[str, str]]) -> tuple[dict | None, str | None]:
    text, ents = detok(tokens)
    if norm_text(text) != text:
        return None, "not_nfc"
    spans = []
    for s, e, t in ents:
        if t in TAGMAP:
            spans.append({"start": s, "end": e, "label": TAGMAP[t], "subtype": None,
                          "value": text[s:e], "source_labels": [t]})
    tags = ["loc_generic"] if any(sp["label"] == "LOCATION" for sp in spans) else []
    rec = base_record(
        rid=f"multiner:{lang}:{i:05d}", text=text, spans=spans, lang=language_block(lang, text),
        domain="document", dataset="suralk/multiNER", source_id=f"{i:05d}", license=LICENSE,
        pipeline="src.sources.multiner", region="LK", group=f"multiner:{i:05d}", tags=tags,
        ext={"dropped_types": sorted({t for _, _, t in ents if t not in TAGMAP})})
    rec["validation"]["status"] = "human_verified"  # human-annotated upstream
    rec["validation"]["checks"]["upstream_human_annotation"] = True
    return rec, None


def run(limit: int | None = None) -> None:
    z = zipfile.ZipFile(url_file(URL, "multiner.zip", "multiner"))
    by_lang = {lang: read_conll(z.read(f).decode("utf-8")) for lang, f in FILES.items()}
    counts = {lang: len(s) for lang, s in by_lang.items()}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"multiNER files are not aligned: {counts}")
    writers = []
    for lang, sents in by_lang.items():
        w = Writer("multiner", lang)
        for i, toks in enumerate(sents):
            if limit and i >= limit:
                break
            rec, reason = convert(i, lang, toks)
            w.write(rec) if rec else w.skip(reason)
        writers.append(w)
    finish("multiner", writers, license=LICENSE, source="https://github.com/suralk/multiNER",
           citation=CITATION, config={"limit": limit, "tagmap": TAGMAP})

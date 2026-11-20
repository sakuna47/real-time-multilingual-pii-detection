# PII Record Format (v1)

Every dataset in this project, whether downloaded, translated, LLM-generated or written by hand, is stored as **JSONL with one record per line, using the schema below**. Training, evaluation and analysis code read only this format.

| Artifact | Path |
|---|---|
| This spec | `docs/DATA_FORMAT.md` |
| JSON Schema | `schema/pii_record.v1.schema.json` |
| Validator + label set + converters | `src/pii_data/record.py` |
| Example records | `examples/records.v1.jsonl` |
| Tests | `tests/test_record_format.py` |

Check a file:

```bash
python -m src.pii_data path/to/file.jsonl
```

---

## 1. Design rules

1. **One text, one record.** A record is one prompt or document plus its spans. There are no token-level labels in storage. Tokenisation for XLM-R, GLiNER and the rest happens at training time from character spans.
2. **Offsets are Unicode code points over NFC-normalised text**, end-exclusive (Python `text[start:end]`). JavaScript strings use UTF-16, so convert there. Sinhala conjuncts use ZERO WIDTH JOINER (U+200D), for example in ශ්‍රී. NFC keeps it, so never strip it.
3. **Keep where every record came from.** Every record says how it was made (`origin`), with which models and settings (`generation`), and how it was checked (`validation`). Any number in the thesis must be traceable back to the records behind it.
4. **Keep the source label.** Spans keep the original dataset's label(s) in `source_labels`, so the taxonomy can be remapped later without downloading anything again.
5. **Version everything.** `schema_version` covers the record layout and `label_set` covers the taxonomy. They change independently.
6. **Room to grow without breaking readers.** New optional fields are a minor bump (1.0 → 1.1). Renaming or removing a field, or changing what it means, is a major bump (→ 2.0) and needs a migration script. Experimental fields go in `ext`.

---

## 2. Record layout

```jsonc
{
  "schema_version": "1.0",
  "id": "slgen:si:000042",              // globally unique, stable — see §4
  "text": "මගේ නම කසුන් පෙරේරා. ...",       // NFC
  "spans": [ /* §3 */ ],
  "label_set": "v1",

  "language": {
    "primary": "si",                     // ISO 639-1 where it exists: si, ta, en, fr, de, it, es, nl
    "langs":   ["si", "en"],             // every language present, primary first
    "scripts": ["Sinh", "Latn"],         // ISO 15924: Sinh, Taml, Latn
    "code_mixed": true,
    "romanized": false                   // true for Singlish / Tanglish written in Latin script
  },
  "domain": "chat_prompt",              // §5
  "region": "LK",                        // ISO 3166-1 alpha-2 of the PII formats, or null

  "origin":     { /* §6 */ },
  "generation": { /* §7 */ },
  "validation": { /* §8 */ },

  "implicit":    null,                   // §9
  "distractors": [],                     // §10 hard-negative look-alikes
  "split": null,                         // "train" | "dev" | "test" | null (not assigned)
  "group": "tpl:bank_letter_03",         // leakage key: records sharing it go to the same split
  "tags": ["nic_dob_consistent"],
  "ext": {}                              // free-form, for experiments
}
```

**Required:** `schema_version`, `id`, `text`, `spans`, `label_set`, `language`, `domain`, `origin`, `validation`.
Every other field is optional. When it is absent, readers treat it as `null`, `[]` or `{}`.

---

## 3. Spans

```jsonc
{
  "start": 7, "end": 19,
  "label": "PERSON",
  "subtype": null,
  "value": "කසුන් පෙරේරා",            // must equal text[start:end]
  "entity_id": "p1",                 // same real person/thing across spans → same id (optional)
  "source_labels": ["GIVENNAME1", "LASTNAME1"],   // labels before mapping/merging, [] if native
  "script": "Sinh"                   // script of the value (optional)
}
```

Rules, enforced by the validator:
- `0 <= start < end <= len(text)` and `value == text[start:end]`
- spans sorted by `start` and **not overlapping** (flat NER)
- no leading or trailing whitespace inside a span
- `label` must be in the record's `label_set`, and `subtype` must be allowed for that label, or `null`

### Label set `v1`

| Label | Subtypes | Notes |
|---|---|---|
| `PERSON` | — | Full or partial personal name. Titles (Mr., මහතා, திரு) are outside the span. |
| `ID_NUMBER` | `NIC_OLD`, `NIC_NEW`, `PASSPORT`, `DRIVING_LICENCE`, `SOCIAL_SECURITY`, `NATIONAL_ID`, `OTHER` | Sri Lankan NIC: old = 9 digits + V/X, new = 12 digits |
| `PHONE` | `MOBILE`, `LANDLINE` | |
| `EMAIL` | — | |
| `ADDRESS` | — | Building, street, secondary address, postcode |
| `LOCATION` | — | City, town or geo-coordinates tied to a person |
| `DOB` | — | Date of birth only. Other dates are not PII in v1. |
| `USERNAME` | — | |
| `PASSWORD` | — | |
| `IP_ADDRESS` | — | |

Candidates for `v2` (not valid yet): `BANK_ACCOUNT`, `CARD_NUMBER`, `VEHICLE_NUMBER`, `ORGANIZATION`.

### Span boundary guidelines

- **Inflection stays outside.** Sinhala and Tamil case endings are not part of the span: `[කසුන්]ට`, `[கமல்]உக்கு`. If the ending is fused into the script so it can't be split cleanly, span the whole word and add the tag `fused_suffix`.
- Surrounding punctuation and quotes are outside the span.
- Phone and NIC spans include internal spaces, dashes and the `+94` prefix: `[+94 77 123 4567]`.
- A first name and a surname separated by a single space form one PERSON span. Address parts separated only by spaces or commas form one ADDRESS span. Fields with a form label between them (`Street:`) stay as separate spans.

---

## 4. IDs

Format: `<source>:<lang>:<local-id>`, using only `[a-z0-9_-]` in the first two parts.

| Source prefix | Meaning |
|---|---|
| `ai4p` | AI4Privacy row as published (`ai4p:en:40767A`) |
| `ai4p-tr` | AI4Privacy translated with placeholders (`ai4p-tr:si:40767A`) |
| `slgen` | LLM- or template-generated Sri Lankan prompt |
| `wikiann`, `naamapadam` | external NER sets |
| `gold` | human-written test set |

IDs must be **deterministic**: re-running a pipeline with the same seed produces the same IDs. Never use random UUIDs.

---

## 5. `domain`

`chat_prompt` · `email` · `form` · `letter` · `chat_message` · `document` · `news` · `wiki` · `other`

AI4Privacy is mostly `email`, `form` and `letter`. The system runs on `chat_prompt`. Keeping these apart lets you report per-domain scores.

---

## 6. `origin`: what the record is

```jsonc
{
  "method": "translated_placeholder",
  "dataset": "ai4privacy/pii-masking-300k",   // upstream dataset, or null if native
  "source_id": "40767A",                      // id in the upstream dataset
  "parent_id": "ai4p:en:40767A",              // record this one was derived from
  "template_id": null,                        // template used, if any
  "license": "see upstream"
}
```

`method` is one of:

| method | Meaning |
|---|---|
| `original` | Copied from an upstream dataset (label mapping is allowed) |
| `translated_placeholder` | Masked text translated with placeholders, then filled in code |
| `translated_projected` | Raw text translated, then spans recovered by alignment or markers |
| `llm_generated` | Text written by an LLM, with PII filled in code |
| `template_filled` | Hand-written template filled in code |
| `human_written` | Written by a person |
| `human_edited` | Any of the above, corrected by a person |

---

## 7. `generation`: how it was made

```jsonc
{
  "pipeline": "sl_translate",          // script or notebook name
  "pipeline_version": "0.1.0",
  "git_commit": "c3511e0",
  "seed": 42,
  "created_at": "2026-11-20T10:15:00Z",
  "models": [
    {"role": "translator", "name": "<model id>", "params": {"temperature": 0.2}, "prompt_id": "translate_v2"}
  ],
  "filler": {"name": "sl_values", "version": "0.1.0"},  // deterministic value generator
  "persona_id": "per-000913"           // fake person behind the values, for NIC↔DOB↔sex consistency
}
```

Store prompt text under `prompts/` in the repo and refer to it by `prompt_id`. Don't copy the prompt into every record.

---

## 8. `validation`: how it was checked

```jsonc
{
  "status": "auto_passed",             // see below
  "checks": {                          // deterministic checks: name → true/false or a number
    "spans_match": true, "placeholders_preserved": true,
    "script_ok": true, "langid": "si", "langid_conf": 0.97,
    "backtranslation_chrf": 61.2
  },
  "judges": [                          // LLM judges
    {"model": "<model id>", "prompt_id": "judge_v1", "fluency": 4, "label_correct": true, "verdict": "pass"}
  ],
  "human": [                           // human review
    {"annotator": "a1", "verdict": "pass", "edited": false, "date": "2026-12-01"}
  ]
}
```

`status`, in increasing order of trust:
`unchecked` → `auto_failed` | `auto_passed` → `judge_failed` | `judge_passed` → `human_rejected` | `human_verified`

Annotator IDs are pseudonyms (`a1`, `a2`), never real names. Records with a `*_failed` or `human_rejected` status stay in the raw files, so rejection rates can be reported, but they are filtered out of training files.

---

## 9. `implicit`: record-level implicit PII

`null` means not annotated. Otherwise:

```jsonc
{
  "identifying": true,                         // could this text single out one person?
  "quasi_identifiers": [                       // non-PII spans that together identify
    {"start": 40, "end": 61, "kind": "occupation"},
    {"start": 75, "end": 82, "kind": "small_town"}
  ],
  "rationale": "Only female surgeon in a named small town."
}
```

`kind`: `occupation`, `workplace`, `rare_attribute`, `small_town`, `relationship`, `event`, `age`, `other`. Quasi-identifier offsets follow the same rules as spans, but they may overlap PII spans.

---

## 10. `distractors`: hard negatives

Text that **looks** like PII but isn't. These are never labelled as spans. They are recorded so false positives can be analysed by kind.

```jsonc
[{"start": 8, "end": 20, "kind": "invoice_number"}]
```

`kind`: `invoice_number`, `reference_number`, `date_number`, `public_figure`, `place_name`, `common_word_name`, `product_code`, `other`.
A record with no spans and at least one distractor is a pure hard negative. Also give it the `hard_negative` tag.

---

## 11. Files and manifests

```
data/<stage>/<dataset>/<split or part>.jsonl
data/<stage>/<dataset>/manifest.json
```

`manifest.json` describes the whole file so you don't have to scan it:

```jsonc
{
  "schema_version": "1.0", "label_set": "v1",
  "dataset": "ai4p-tr", "created_at": "...", "git_commit": "...",
  "files": {"si.jsonl": {"records": 12000, "sha256": "..."}},
  "counts": {"by_label": {...}, "by_lang": {...}, "by_status": {...}},
  "config": {...}                     // the pipeline config that produced it
}
```

## 12. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-11-20 | First version |

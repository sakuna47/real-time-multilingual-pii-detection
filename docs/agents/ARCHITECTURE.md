# Architecture

How the dataset pipeline is put together. For the record format itself see `docs/DATA_FORMAT.md`.

## The idea

Build Sri Lankan Sinhala/Tamil PII training data **from real, citable sources** instead of inventing text:
an LLM only translates, localises and code-mixes; every PII value is generated **in code** from official
format rules; spans are exact because code inserts the values. Each source record becomes **one** of six
language varieties, chosen by a stable hash of its id:

| Variety | Meaning |
|---|---|
| `si` / `ta` | Sinhala / Tamil in native script |
| `si-en` / `ta-en` | native script with English words mixed in |
| `si-rom` / `ta-rom` | Singlish / Tanglish: fully romanised, casual |

Default mix: 40% native, 30% mixed, 30% romanised (`--mix`).

## Data flow

```
sources (src/sources)                       values (src/sl_gen)
  Nemotron-PII  (English, spans)              personas: NIC ↔ DOB ↔ sex consistent,
  WildChat      (English prompts, no spans)   phones (TRCSL), names/places (resources/sl),
  Aya           (Sinhala/Tamil prompts)       addresses, emails (example.* domains), IPs
        │                                                 │
        ▼                                                 │
src/translate/pipeline.py  (per chunk of --chunk records) │
  1. mask.py     spans → {PERSON_1}, {PHONE_1} … (same value = same placeholder; "Jason" ↔ "Jason Lee" linked)
     insert      (no spans) LLM adds 1-4 placeholders where a user would put personal data, or says "not suitable"
  2. translate   English → si/ta, keep placeholders, localise foreign context (prompt translate_v2)
  3. codemix     si/ta → si-en, ta-en, si-rom, ta-rom using real style examples (prompt codemix_v1)
  4. fill.py     placeholders → persona values, spans recorded while building the string ◄──┘
        │        one persona per person entity; one script per person per record
        ▼
data/generated/sri_lankan/<source>/<variety>.jsonl   (+ rejected.jsonl with stage + reason)
        │
src/validate/checks.py  → data/validated/…   (schema, leftover placeholders, variety scripts, NIC/phone/email
        │                                     formats, code-mixing index, duplicates; keeps human verdicts)
src/validate/review_export.py → HTML sheet → human marks OK/fix/reject → CSV → review_import.py
        │
src/pii_data/split.py → data/splits/…  (80/10/10 by `group` = source record id; groups never cross splits)
```

Hard negatives (`src/sl_gen/negatives.py`): real PII-free sentences (Aya, NLPC code-mixed social text,
SinMix2Mono Singlish) + an agent-written clause containing a PII-looking but harmless number (12-digit
invoice number shaped like a NIC, 07X tracking number shaped like a phone…). Values go into `distractors[]`.

## LLM layer (`src/llm`)

- `get_llm(role)` builds an `LLM` from `configs/llm.toml`: provider adapter + rate limiter + retry (429/5xx/timeouts
  only) + SQLite response cache + JSONL call log. `--dry-run` swaps in `mock.py`.
- Providers: `vertex` (google-genai on Vertex AI; ADC or express key; **Flex PayGo** via headers when a role sets
  `service_tier = "flex"`, served tier logged as `traffic_type`), `gemini` (AI Studio, `v1beta`), `deepseek` and
  `openrouter` (OpenAI-compatible).
- Roles in use: translator, codemixer, inserter, transliterator → `vertex:gemini-3.5-flash` (thinking off, Flex);
  judge → `vertex:gemini-3.5-flash-lite` (Flex). Judge must differ from generating models (checked on load).
- `prompts.py`: prompt files with front matter; variables use `{{double braces}}` so `{PERSON_1}` passes through.
- `steps.py` (in translate) sends items in batches, validates every answer (placeholders preserved, scripts
  correct), retries failures one at a time with a different cache salt, then rejects with a reason.
- `cost.py` prices the call log per model and per served tier; `pipeline.py` refuses to start / stops between
  chunks below `--min-budget` (default $1).

## Sources and licences

| Source | Use | Licence |
|---|---|---|
| nvidia/Nemotron-PII | English text with spans (main volume); unmapped identifier types → tagged and skipped; URLs treated as non-PII | CC BY 4.0 |
| allenai/WildChat-1M | real first-turn chatbot prompts; placeholders inserted | ODC-BY |
| CohereLabs/aya_dataset (sin, tam) | native prompts; personal-task filter | Apache-2.0 |
| suralk/multiNER | parallel en/si/ta human NER (PER→PERSON, LOC→LOCATION); external test set | CC0-1.0 (repo) |
| SinMix2Mono, Programmer-RD-AI triplets, NLPC-UOM code-mixed | code-mixing style examples and references | CC BY 4.0 / MIT / MIT |
| FLORES+ devtest | translator benchmark only | CC BY-SA 4.0 (gated) |
| Wikidata | Sri Lankan given/family names, 3,035 settlements | CC0 |

Sri Lankan formats: NIC (DRP: old 9 digits + V/X, new 12 digits, day-of-year with fixed 29 Feb, +500 for women;
check digit random, documented as synthetic), phones (TRCSL mobile prefixes 070–078, landline area codes).

## Storage layout (`PII_DATA_ROOT`)

`src/common/paths.py`: `PII_DATA_ROOT` env → `/content/drive/MyDrive/PII_Project` when mounted in Colab →
`./data`. Sub-folders: `sources/`, `generated/sri_lankan/`, `validated/`, `review/`, `splits/`, `results/`,
`logs/llm_calls.jsonl`, `cache/llm_cache.sqlite` (`PII_CACHE_DIR` overrides; on Colab the cache lives on local
disk and is copied to Drive every 5 minutes by `src/common/colab.py`).

## Tests

`tests/test_*.py` (71 tests, offline): record format, LLM layer (mock), benchmark, importers, Sri Lankan values,
translation engine, validation/splits/negatives. Run with the venv Python; pytest is not installed.

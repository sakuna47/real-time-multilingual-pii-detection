# Decisions

Settled with the user. Each has a reason; do not change one without asking the user.

| Date | Decision | Why |
|---|---|---|
| 2026-11-20 | One JSONL record format (v1, now spec 1.1) for every dataset; character spans, not tokens | Any model can tokenise later; provenance and validation fields make every thesis number traceable |
| 2026-11-20 | Build the Sri Lankan data from real sources (translate/localise), not free LLM generation | Grounded, citable data; the LLM only translates, code-mixes and inserts placeholders |
| 2026-11-20 | PII values generated **in code** (NIC, phones, names, addresses), never by the LLM | Exact spans by construction; NIC agrees with DOB and sex; no real people |
| 2026-11-20 | Each source record becomes **one** variety (hash of id), 40/30/30 native/mixed/romanised | More diverse data than six near-copies of one text |
| 2026-11-20 | **No human-written gold test set** for now; the user reviews samples of generated data instead. Test split is held out by `group`; multiNER (real human labels) is the external test set | User's choice; note in the thesis that synthetic test scores may be optimistic |
| 2026-11-20 | Nemotron `url` spans are treated as **not PII** (O) | Mostly company/service websites; excluding them blocked 9% more records |
| 2026-11-20 | Nemotron records with other unmapped identifiers (accounts, cards, medical ids…) are **skipped** | Translating them would create unlabelled PII (label noise) |
| 2026-11-20 | Translate prompt `translate_v2` **localises** foreign context (US states, $, SSN → Sri Lankan districts, rupees, NIC) | US context next to Sri Lankan PII read unnaturally |
| 2026-11-20 | Translator/codemixer/inserter: **`vertex:gemini-3.5-flash`, thinking off**; judge: `vertex:gemini-3.5-flash-lite` | FLORES chrF++ 50.7 si / 52.6 ta, 100% placeholders kept; flash-lite is weaker for Sinhala (43.0); 2.5-flash lost placeholders (33%) |
| 2026-11-20 | Use **Vertex AI Flex PayGo** (half price, best effort) on all Gemini roles | Credit is small ($17.37); Batch mode would need GCS + job orchestration |
| 2026-11-20 | Target ~4,500 more generated records (not 16k) | Budget; the learning curve decides the final size |
| 2026-11-20 | v4-pro / pro-tier models not used | User's instruction (cost) |
| 2026-11-20 | Aya personal-intent filter uses task cues (letter, application, form, complaint, email, CV) | Attribute words like "name"/`නම` matched nearly everything |
| 2026-11-20 | Run full-scale work on Colab with all outputs on Google Drive; the PC is for development and small tests | Bandwidth and long runs; local data was uploaded once as a zip |
| 2026-11-20 | The Colab Google account was granted Vertex AI User on the project (user approved) | Colab runs as a different Google account from the project owner |
| 2026-11-20 | Pipeline writes in chunks of 300 by default; stage-2 Nemotron resume uses `--chunk 0` | Crash safety and progress; chunk 0 keeps cache keys identical to the interrupted run |
| 2026-11-20 | Commits authored by the user only; **no Claude/AI co-author trailers** | User's instruction |

## Planned but not built

- LLM judge (`src/validate/judge.py`, `prompts/judge_v1.md`): fluency, naturalness for the variety, label
  correctness, missed PII → `validation.judges[]`, status `judge_passed/judge_failed` (never overrides human).
- Back-translation chrF check against the English source.
- Phase 2: migrate AI4Privacy (`01_/02_` notebooks) with `from_ai4privacy_row()` and translate it through the
  same engine.

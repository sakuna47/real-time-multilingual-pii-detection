# Status (last updated 2026-11-20)

Update this file whenever a meaningful step finishes. Numbers below are exact unless marked "estimate".

## Phase plan

| # | Step | State |
|---|---|---|
| 0 | v1 record format, JSON Schema, validator (`docs/DATA_FORMAT.md`, spec 1.1) | done |
| 1 | LLM client layer (Vertex/Gemini, DeepSeek, OpenRouter, mock), cache, retry, Flex tier, cost report | done |
| 2 | Translator benchmark on FLORES+ (chose `vertex:gemini-3.5-flash`, thinking off) | done |
| 3 | Source importers (Nemotron-PII, WildChat, Aya, multiNER, code-mix pools, FLORES) | done |
| 4 | Sri Lankan values: NIC/DOB/sex-consistent personas, phones, names, 3,035 places | done; resources partly unreviewed |
| 5 | Placeholder-translation engine (mask/insert → translate → code-mix → fill), chunked writing | done |
| 6 | Pilot (62 records) + human review | done |
| 7 | Checks, splits, hard negatives, review sheet export/import | done; **LLM judge not built** |
| 8 | Stage 1 (~1,000 records, local) + human review | done |
| 9 | **Stage 2 on Colab** (Nemotron 2,500 + WildChat ~900 + Aya) | **in progress / uncertain — see below** |
| 10 | Phase 2: AI4Privacy → v1 migration and translation through the same engine | not started |
| 11 | Detector training (XLM-R stage 2/3, GLiNER), learning curve | not started |

## Data that exists

### On the user's PC (`./data`, not in git)
| Set | Records | Review status |
|---|---:|---|
| `generated/sri_lankan/nemotron` | 885 | 74 human_verified, 807 auto_passed, 4 auto_failed (low code-mixing) |
| `generated/sri_lankan/wildchat` | 91 | 34 human_verified, 57 auto_passed |
| `generated/sri_lankan/aya` | 27 | 10 human_verified, 17 auto_passed |
| `generated/sri_lankan/negatives` (hard negatives) | 1,600 | 19 human_verified, 1,579 auto_passed, 2 auto_failed |
| `sources/` | Nemotron 5k-per-split sample, WildChat 1 shard, Aya, multiNER, code-mix, FLORES | — |

All 137 human-reviewed records were accepted by one reviewer (pseudonym `a1`), no issues flagged.
Review CSVs: `data/review/review_20261120T1006_a1.csv`, `review_20261120T1046_a1.csv`.

### On Google Drive (`MyDrive/PII_Project/`, used by Colab)
- Everything above (uploaded once as `pii_project_upload.zip`, unpacked by the notebook; renamed `.done`).
- Full imports: **Nemotron 87,498 records** (43,435 train + 44,063 test), **WildChat 4,537 English prompts**
  (3 shards; no Sinhala/Tamil prompts in them).
- Possibly the Nemotron stage-2 output (see below).

## Open situation: Colab Nemotron stage-2 run

- Started in Colab with `--source nemotron --new 2500 --batch-size 6 --workers 8` using the **old all-at-once
  code** (records are written only when the whole run finishes). Budget check passed at $10.13 remaining;
  261 stage-1 records were correctly skipped.
- After ~37 minutes the browser page lost its connection ("Resuming execution"). The user confirmed the old
  Colab session still existed; whether the job finished is **unknown**.
- Its paid model answers are in the API cache, which the notebook copies to Drive every 5 minutes, and every
  call is in `MyDrive/PII_Project/logs/llm_calls.jsonl`.

**How to find out:** in Colab (or by looking at Drive), count lines in
`MyDrive/PII_Project/generated/sri_lankan/nemotron/*.jsonl` (excluding `rejected.jsonl`). Stage 1 left 885;
a finished stage 2 adds up to ~2,450 more. Then run `python -m src.llm.cost` with `PII_DATA_ROOT` pointing at Drive.

**If it did not finish:** re-run the notebook's setup cells and the Nemotron cell **as it is now**
(`NEMOTRON_CHUNK = 0`). With chunk 0 the batches are identical to the interrupted run, so cached answers are
reused for free; only the unfinished part is paid. Do not use chunking for this particular resume.

## Budget

- Credit: **$17.37** Google Developer Program credits on the project's billing account (expire Aug/Sep 2027).
- Spent before stage 2 (call log, list price): **$7.23**; remaining then **$10.14**.
- Stage 2 Nemotron estimate: **≈ $5.30** (Flex tier ≈ $2.10 per 1,000 records). Actual: check the Drive call log.
- Planned WildChat (~900 records) ≈ $3.00 and Aya ≈ $0.10 would leave ≈ $1.70. The pipeline refuses to start
  (and stops between chunks) below $1 estimated remaining.
- The Billing console is the authority; the cost report is an estimate from token counts at list prices
  (`[pricing]` in `configs/llm.toml`). DeepSeek has no credit left (402); Gemini AI Studio key is free tier
  (20 requests/day/model) and not usable for runs.

## Immediate next steps (in order)

1. **Confirm Nemotron stage 2** (above). If incomplete, resume it with `NEMOTRON_CHUNK = 0`.
2. In Colab: save the cache to Drive (`colab.save_cache()`), re-run the first cell to pull the latest code.
   The user asked to **stop there** for now: do not start WildChat/Aya until the user says so.
3. When resumed later: WildChat `--new 3000` and Aya `--new 200` (chunked, default `--chunk 300`), then
   checks → review sheet → the user reviews → `review_import` → splits. Report cost after each.
4. Local copy: data generated on Colab lives on Drive; if work moves back to the PC, download it into `./data`.

## Known gaps and backlog

- **LLM judge step not implemented.** Role `judge` (`vertex:gemini-3.5-flash-lite`, Flex) is configured;
  `src/validate/judge.py` and `prompts/judge_v1.md` do not exist yet. Estimated cost ≈ $0.40 per 1,000 records.
- **Back-translation chrF check** (planned in the validation design) not implemented.
- **Unreviewed resources:** `resources/sl/address_words.csv`, `streets.csv`, `months.csv` (agent-written
  Sinhala/Tamil), 3,590 LLM-filled name/place spellings (`source` contains `+llm`), and the hard-negative
  clauses in `src/sl_gen/negatives.py` (`CLAUSES`). All need native-speaker review.
- **Source balance:** most records come from Nemotron (documents/forms); real chat-prompt style (WildChat) is
  under-represented. More WildChat shards would help.
- No hard negatives for `ta-en` / `ta-rom` (no real base corpus); no real Tamil code-mixing reference pool.
- Single reviewer; a second annotator on ~50 records would give Cohen's kappa (`review_import` computes it).
- Phase 2 (AI4Privacy) and all model training not started. README "Project Structure" is aspirational.

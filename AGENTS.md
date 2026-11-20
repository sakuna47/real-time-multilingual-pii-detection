# AGENTS.md

Context for AI coding agents working on this repository. Humans: see `README.md`.
Read this file first, then the documents it points to before changing anything.

| Read next | What it answers |
|---|---|
| [docs/agents/STATUS.md](docs/agents/STATUS.md) | Where the work stands right now, what is unfinished, what to do next |
| [docs/agents/ARCHITECTURE.md](docs/agents/ARCHITECTURE.md) | How the pipeline works, module by module, and where data lives |
| [docs/agents/RUNBOOK.md](docs/agents/RUNBOOK.md) | Exact commands: setup, tests, local runs, Colab runs, resuming, costs, troubleshooting |
| [docs/agents/DECISIONS.md](docs/agents/DECISIONS.md) | Decisions already made and why; do not re-open them without the user |
| [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) | The v1 record format every dataset uses (spec 1.1) |

## Project in one paragraph

Final-year project: a real-time, explainable PII detector for prompts sent to AI chatbots, with Sri Lankan
PII (NIC numbers, +94 phones, local names and addresses) in Sinhala, Tamil, English and code-mixed text
(Singlish, Tanglish). This repository currently contains the **dataset pipeline**, not the detector: it
builds a Sri Lankan Sinhala/Tamil PII dataset by translating and localising existing English PII datasets
with an LLM, filling placeholders with synthetic Sri Lankan values generated in code, validating, and
having a human review samples. Detector training (XLM-R, GLiNER), the FastAPI backend and the Chrome
extension come later (see README).

## Repository map

```
AGENTS.md                 this file
README.md                 human overview of the whole FYP
docs/DATA_FORMAT.md       v1 record spec (read before touching any record code)
docs/agents/              agent context: status, architecture, runbook, decisions
schema/                   JSON Schema for v1 records
configs/llm.toml          LLM providers, roles -> models, prices, credit budget
prompts/*.md              versioned LLM prompts (never edit a used one: copy to a new id)
resources/sl/*.csv        Sri Lankan names, places, phone codes, address words (human-reviewable)
src/common/               data paths, secrets, run metadata, Colab helpers, upload pack
src/llm/                  provider adapters (Vertex/Gemini, DeepSeek, OpenRouter, mock), cache, retry, cost
src/pii_data/             record validator, manifests, train/dev/test splits
src/sources/              importers: Nemotron-PII, WildChat, Aya, multiNER, code-mix pools, FLORES
src/sl_gen/               NIC/phone/persona/value generators, resources builder, hard negatives
src/translate/            placeholder-translation engine + translator benchmark
src/validate/             deterministic checks, review sheet export/import
notebooks/03_sl_dataset.ipynb   Colab runner (outputs to Google Drive)
01_/02_*.ipynb            original AI4Privacy download/exploration notebooks (phase 2 input)
tests/                    script-style tests (no pytest needed)
data/                     NOT in git: all generated data, cache, logs (local or on Drive)
```

## Setup and commands

Windows PC, Python 3.14 in a virtualenv. Always use the venv interpreter:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Run every test suite (71 tests, all offline, no keys needed) before committing:

```bash
for t in tests/test_*.py; do .venv/Scripts/python.exe "$t" | tail -1; done
```

Each test file prints `PASS`/`FAIL` per test and exits non-zero on failure. Set `PYTHONIOENCODING=utf-8`
in Git Bash, or Sinhala/Tamil output can crash the Windows console encoding.

Most-used commands (details in the runbook):

```bash
.venv/Scripts/python.exe -m src.llm.smoke --model vertex:gemini-3.5-flash   # provider works?
.venv/Scripts/python.exe -m src.llm.cost                                    # spend and remaining credit
.venv/Scripts/python.exe -m src.translate.pipeline --source nemotron --new 300 --dry-run
.venv/Scripts/python.exe -m src.validate.checks data/generated/sri_lankan
.venv/Scripts/python.exe -m src.pii_data data/validated/sri_lankan/nemotron/si.jsonl
```

## Rules (follow exactly)

- **Git:** work on branch `Data`. Do not commit to `main`. Commit only when the user asks. **Never add
  `Co-Authored-By: Claude` or any AI attribution trailer** to commits or PR descriptions. Push only when asked.
- **Never commit** `.env`, anything under `data/`, `.venv/`, keys or tokens. The repo is public on GitHub:
  do not write personal emails or names into committed files.
- **Spending:** model calls are paid from a small Google Cloud credit (see STATUS.md for the balance).
  Run `python -m src.llm.cost` before and after any paid run. Use `--dry-run` (mock model) to test
  pipeline changes. Do not start large runs, benchmarks or new providers without the user's OK.
- **Account/billing changes** (IAM roles, enabling APIs, billing links) need the user's explicit OK each time.
- **Prompts are immutable once used:** records cite `prompt_id`. To change a prompt, add `prompts/<name>_vN+1.md`
  and point the code at it.
- **Records:** every file must pass `python -m src.pii_data <file>`. Offsets are code points over NFC text;
  spans must equal `text[start:end]`; spans never overlap. Bump `docs/DATA_FORMAT.md` (changelog) for any
  format change; additive changes are a minor version.
- **Judge model must differ** from translator/codemixer/inserter models (`configs/llm.toml` load fails otherwise).
- **Cache keys** include model, prompt, params and batch composition. Changing batch size, chunking or a prompt
  means old cached answers are not reused (you pay again). Resume an interrupted run with the same flags.
- **Sinhala/Tamil text written by the agent** (curated words, clauses) must be marked `needs_review` and reported
  to the user; the user is the native-speaker reviewer.
- Keep `docs/agents/STATUS.md` current when you finish a meaningful step (what ran, numbers, what is next).

## Where to start

Open [docs/agents/STATUS.md](docs/agents/STATUS.md) → "Immediate next steps". The first job is to find out
whether the Colab Nemotron stage-2 run finished and saved its records to Drive.

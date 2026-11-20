# Runbook

Exact commands. On the PC use Git Bash with `export PYTHONIOENCODING=utf-8` and the venv interpreter
`.venv/Scripts/python.exe` (shown as `py` below).

## 1. Local setup (PC)

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
cp .env.example .env        # then fill keys; never commit .env
```

`.env` keys: `GOOGLE_CLOUD_PROJECT` (Vertex project id), `GOOGLE_CLOUD_LOCATION=global`, optional
`VERTEX_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN` (gated datasets).

Vertex auth on the PC uses Application Default Credentials. gcloud is installed at
`C:\Users\Dell\AppData\Local\Google\google-cloud-sdk\bin\gcloud.cmd` (not on PATH in older shells):

```bash
gcloud auth application-default login
```

## 2. Checks that cost nothing

```bash
for t in tests/test_*.py; do py "$t" | tail -1; done          # all tests
py -m src.llm.smoke --dry-run                                  # mock provider
py -m src.translate.pipeline --source nemotron --new 20 --dry-run
py -m src.llm.cost                                             # spend + remaining credit
py -m src.pii_data <file.jsonl> [...]                          # validate records
```

## 3. Paid steps (ask the user first; check cost before and after)

```bash
py -m src.llm.smoke --model vertex:gemini-3.5-flash
py -m src.sources.run nemotron|wildchat [--shards N]|aya|multiner|codemix|flores
py -m src.translate.pipeline --source nemotron --new 2500 --batch-size 6 --workers 8     # chunked by default (300)
py -m src.translate.pipeline --source wildchat --new 3000 --batch-size 6 --workers 8
py -m src.translate.pipeline --source aya --new 200 --batch-size 6 --workers 8
py -m src.sl_gen.negatives --n 400                                                       # free
py -m src.sl_gen.run fill-scripts                                                        # LLM spelling fill
py -m src.translate.benchmark --models vertex:gemini-3.5-flash --langs si,ta --limit 200
```

`--new N` = next N source records not yet written and not rejected before for a content reason.
`--limit N` = first N in stable order (older stages). Re-running a command resumes; cached answers are free
**only if** batch size, chunk size, prompts and models are unchanged.

## 4. After generation (free)

```bash
py -m src.validate.checks data/generated/sri_lankan              # -> data/validated (keeps review verdicts)
py -m src.validate.review_export data/validated/sri_lankan --per-group 5 --negatives --out data/review/stageN.html
#   user opens the HTML in a browser, marks records, clicks Export CSV
py -m src.validate.review_import data/validated/sri_lankan data/review/review_*.csv
py -m src.pii_data.split data/validated/sri_lankan --out data/splits/sri_lankan
py -m src.pii_data.split data/sources/multiner --out data/splits/multiner --all-test --keep human_verified
```

## 5. Colab (full runs; outputs on Google Drive)

Notebook: `https://colab.research.google.com/github/sakuna47/real-time-multilingual-pii-detection/blob/Data/notebooks/03_sl_dataset.ipynb`
(loads from GitHub branch `Data`, so push code first). CPU runtime is enough.

1. Cell 1 mounts Drive, clones/pulls `Data`, installs requirements. Re-run it to pull new code.
2. Cell 2 sets `PII_DATA_ROOT=/content/drive/MyDrive/PII_Project` and the project id, loads Colab Secrets
   (each secret needs *Notebook access* turned on **for this notebook**), signs in to Google Cloud for Vertex
   (pop-up; the user approves), unpacks a one-time upload zip, restores the cache, starts the 5-minute cache sync.
3. Smoke + cost cell, import cell (skips what exists), stage cells, checks/review/splits cells, finish cell
   (`colab.save_cache()`).

Moving local work to Drive: `py -m src.common.pack` → `data/export/pii_project_upload.zip`; the user uploads it
to `MyDrive/PII_Project/`; cell 2 unpacks it once.

Colab gotchas seen so far:
- The Colab Google account is not the project owner; it was granted **Vertex AI User** on the project. A 403
  `aiplatform.endpoints.predict` means the signed-in account lacks that role (IAM changes take ~1-2 minutes).
- Secrets show `False` until *Notebook access* is enabled for this notebook (GitHub-loaded notebooks count as new).
- The page can lose its connection during long cells ("Resuming execution"). The runtime may keep running;
  check *Runtime → Manage sessions* before starting anything again, so work is not duplicated.
- The Colab terminal panel did not accept typed input through browser automation; use notebook cells instead.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `403 PERMISSION_DENIED aiplatform.endpoints.predict` | signed-in account lacks Vertex AI User on the project |
| `403 API_KEY_SERVICE_BLOCKED` (Gemini) | key restricted; use Vertex or a fresh AI Studio key |
| `429 … FreeTier … limit: 20` | Gemini AI Studio free tier: 20 requests/day/model; use Vertex |
| `400 Developer instruction is not enabled for api version v1` | Gemini adapter must use `v1beta` (already set) |
| `402 Insufficient Balance` | DeepSeek account has no credit |
| Vertex `404 … gemini-2.5-flash` | model not offered to the project on Vertex; use 3.5-flash / flash-lite |
| `UnicodeEncodeError` in console | set `PYTHONIOENCODING=utf-8` |
| `resources/sl has no complete rows…` | run `py -m src.sl_gen.run fill-scripts` (paid) or point `PII_RESOURCES_DIR` at fixtures |
| Wikidata `429` | WDQS limited to ~1 request/minute; queries in `src/sl_gen/resources.py` sleep between calls |

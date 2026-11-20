"""Pick the translator on evidence: score models on FLORES+ English -> Sinhala / Tamil.

For each model and language it reports:
  chrF++          corpus chrF++ against the human FLORES reference (failed sentences count as empty)
  coverage        share of sentences that came back (batch replies with the wrong count fail)
  placeholder_ok  share of probe sentences where {PERSON_1} and {EMAIL_1} come back exactly once
                  and nothing else in braces appears (the pipeline depends on this)
  tokens, calls, latency   for the cost comparison

    python -m src.translate.benchmark --models deepseek:deepseek-flash,gemini:<model> --langs si,ta
    python -m src.translate.benchmark --models translator,judge --limit 50     # role names work too
    python -m src.translate.benchmark --models deepseek:deepseek-flash --dry-run --limit 20

Results: <data_root>/results/translator_benchmark/<run_id>.json (+ .md table and per-model outputs).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

from ..common.paths import results_dir
from ..common.runinfo import git_commit, run_id as make_run_id, utc_now
from ..llm import LLM, get_llm, load_prompt, run_batch
from ..sources import flores

PROMPT_ID = "translate_bench_v1"
LANGS = {"si": ("Sinhala", "Sinhala"), "ta": ("Tamil", "Tamil")}
PROBE_SUFFIX = " Please forward this to {PERSON_1} at {EMAIL_1}."
PROBE_TOKENS = ("{PERSON_1}", "{EMAIL_1}")
_BRACED = re.compile(r"\{[^{}]*\}")


@dataclass
class Score:
    model: str
    lang: str
    n: int
    chrf: float = 0.0
    chrf_ok_only: float = 0.0
    coverage: float = 0.0
    placeholder_ok: float | None = None
    calls: int = 0
    cached_calls: int = 0
    failed_batches: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def chrf_pp(hyps: list[str], refs: list[str]) -> float:
    from sacrebleu.metrics import CHRF
    if not hyps:
        return 0.0
    return round(CHRF(word_order=2).corpus_score(hyps, [refs]).score, 2)


def placeholder_ok(text: str) -> bool:
    found = Counter(_BRACED.findall(text))
    return all(found[t] == 1 for t in PROBE_TOKENS) and set(found) == set(PROBE_TOKENS)


def _batches(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def translate_all(llm: LLM, sentences: list[str], lang: str, batch_size: int, workers: int,
                  score: Score) -> list[str | None]:
    """Translate in batches; returns one output per sentence (None where it failed)."""
    language, script = LANGS[lang]
    prompt = load_prompt(PROMPT_ID)
    chunks = _batches(sentences, batch_size)
    reqs = [prompt.request(language=language, script=script, n=len(c),
                           sentences_json=json.dumps(c, ensure_ascii=False, indent=0))
            for c in chunks]
    results = run_batch(llm, reqs, workers=workers, json=True, progress_every=0)
    out: list[str | None] = []
    for chunk, r in zip(chunks, results):
        if r.response is not None:
            score.calls += 1
            score.cached_calls += r.response.cached
            score.input_tokens += r.response.usage.get("input_tokens") or 0
            score.output_tokens += r.response.usage.get("output_tokens") or 0
            score.latency_s += r.response.latency_s or 0.0
        tr = r.data.get("translations") if r.ok and isinstance(r.data, dict) else None
        if not isinstance(tr, list) or len(tr) != len(chunk) or not all(isinstance(t, str) for t in tr):
            score.failed_batches += 1
            if r.error is not None:
                score.errors.append(f"{type(r.error).__name__}: {r.error}"[:300])
            elif tr is not None:
                score.errors.append(f"expected {len(chunk)} translations, got "
                                    f"{len(tr) if isinstance(tr, list) else type(tr).__name__}")
            else:
                score.errors.append("reply had no 'translations' list")
            out.extend([None] * len(chunk))
        else:
            out.extend(t.strip() for t in tr)
    return out


def evaluate(llm: LLM, rows: list[dict], probe_rows: list[dict], lang: str,
             batch_size: int = 10, workers: int = 4) -> tuple[Score, list[dict]]:
    score = Score(model=llm.name, lang=lang, n=len(rows))
    src = [r["en"] for r in rows]
    refs = [r[lang] for r in rows]
    hyps = translate_all(llm, src, lang, batch_size, workers, score)
    ok = [i for i, h in enumerate(hyps) if h is not None]
    score.coverage = round(len(ok) / len(rows), 4) if rows else 0.0
    score.chrf = chrf_pp([h or "" for h in hyps], refs)
    score.chrf_ok_only = chrf_pp([hyps[i] for i in ok], [refs[i] for i in ok])

    outputs = [{"id": r["id"], "en": r["en"], "ref": r[lang], "hyp": h} for r, h in zip(rows, hyps)]
    if probe_rows:
        probe_src = [r["en"] + PROBE_SUFFIX for r in probe_rows]
        probe_hyps = translate_all(llm, probe_src, lang, batch_size, workers, score)
        score.placeholder_ok = round(sum(bool(h) and placeholder_ok(h) for h in probe_hyps)
                                     / len(probe_rows), 4)
        outputs += [{"id": r["id"], "en": s, "probe": True, "hyp": h,
                     "placeholder_ok": bool(h) and placeholder_ok(h)}
                    for r, s, h in zip(probe_rows, probe_src, probe_hyps)]
    score.latency_s = round(score.latency_s, 2)
    return score, outputs


def markdown_table(scores: list[Score]) -> str:
    lines = ["| model | lang | n | chrF++ | chrF++ (returned only) | coverage | placeholders ok "
             "| calls (cached) | tokens in/out | latency s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in sorted(scores, key=lambda s: (s.lang, -s.chrf)):
        ph = "—" if s.placeholder_ok is None else f"{s.placeholder_ok:.0%}"
        lines.append(f"| {s.model} | {s.lang} | {s.n} | {s.chrf:.1f} | {s.chrf_ok_only:.1f} | "
                     f"{s.coverage:.0%} | {ph} | {s.calls} ({s.cached_calls}) | "
                     f"{s.input_tokens:,}/{s.output_tokens:,} | {s.latency_s:.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma-separated <provider>:<model> or role names")
    ap.add_argument("--langs", default="si,ta")
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--limit", type=int, default=0, help="sentences per language (0 = all)")
    ap.add_argument("--probe", type=int, default=30, help="placeholder-probe sentences (0 = off)")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="offline mock provider")
    args = ap.parse_args(argv)

    all_rows = flores.load(args.split)
    rows = all_rows[:args.limit] if args.limit else all_rows
    # probe sentences come from the end so they don't repeat the scored ones when --limit is small
    probe_rows = all_rows[-args.probe:] if args.probe else []
    run = make_run_id("bench")
    out_dir = results_dir("translator_benchmark")
    detail_dir = out_dir / run
    detail_dir.mkdir(parents=True, exist_ok=True)

    scores: list[Score] = []
    for spec in [m.strip() for m in args.models.split(",") if m.strip()]:
        llm = get_llm(spec, dry_run=args.dry_run, run_id=run)
        for lang in [l.strip() for l in args.langs.split(",") if l.strip()]:
            if lang not in LANGS:
                raise SystemExit(f"unknown lang {lang!r}; use si and/or ta")
            print(f"{llm.name} -> {lang}: {len(rows)} sentences + {len(probe_rows)} probes ...", flush=True)
            score, outputs = evaluate(llm, rows, probe_rows, lang, args.batch_size, args.workers)
            scores.append(score)
            safe = llm.name.replace(":", "_").replace("/", "_")
            with open(detail_dir / f"{safe}_{lang}.jsonl", "w", encoding="utf-8", newline="\n") as f:
                for o in outputs:
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            print(f"   chrF++ {score.chrf:.1f}  coverage {score.coverage:.0%}  "
                  f"placeholders {score.placeholder_ok if score.placeholder_ok is not None else '-'}  "
                  f"failed batches {score.failed_batches}", flush=True)

    table = markdown_table(scores)
    report = {"run_id": run, "created_at": utc_now(), "git_commit": git_commit(),
              "dataset": flores.REPO_ID, "split": args.split, "license": flores.LICENSE,
              "prompt_id": PROMPT_ID, "args": vars(args), "dry_run": args.dry_run,
              "scores": [s.__dict__ for s in scores]}
    (out_dir / f"{run}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{run}.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nsaved {out_dir / (run + '.json')} and per-sentence outputs in {detail_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

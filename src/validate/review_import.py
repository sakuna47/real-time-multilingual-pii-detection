"""Merge review CSVs (from the HTML review sheet) back into validated records.

    python -m src.validate.review_import data/validated/sri_lankan review_*.csv

Verdicts: ok -> human_verified, fix -> human_rejected (issues kept; fix and re-run),
reject -> human_rejected. Each review is appended to validation.human with the pseudonymous
annotator id. If two annotators reviewed the same records, Cohen's kappa on the verdict is
printed (and saved in review_report.json).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..common.runinfo import utc_now

STATUS = {"ok": "human_verified", "fix": "human_rejected", "reject": "human_rejected"}


def read_reviews(paths: list[Path]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("verdict") in STATUS:
                    out[row["id"]].append(row)
    return out


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 3)


def run(folder: Path, csvs: list[Path]) -> dict:
    reviews = read_reviews(csvs)
    updated = 0
    verdicts: Counter = Counter()
    issues: Counter = Counter()
    for p in sorted(folder.rglob("*.jsonl")):
        if p.name == "rejected.jsonl":
            continue
        lines, changed = [], False
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            for rev in reviews.get(r["id"], []):
                v = r.setdefault("validation", {})
                human = v.setdefault("human", [])
                entry = {"annotator": rev.get("annotator") or "a1", "verdict": rev["verdict"],
                         "issues": [i for i in (rev.get("issues") or "").split(";") if i],
                         "notes": rev.get("notes") or "", "sheet": rev.get("sheet"), "date": utc_now()[:10]}
                human[:] = [h for h in human if not (h.get("annotator") == entry["annotator"]
                                                     and h.get("sheet") == entry["sheet"])] + [entry]
                verdicts[rev["verdict"]] += 1
                issues.update(entry["issues"])
                changed = True
            if r["id"] in reviews:
                vs = [h["verdict"] for h in r["validation"]["human"]]
                r["validation"]["status"] = "human_rejected" if any(x != "ok" for x in vs) else "human_verified"
                updated += 1
            lines.append(json.dumps(r, ensure_ascii=False))
        if changed:
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    by_ann: dict[str, dict[str, str]] = defaultdict(dict)
    for rid, revs in reviews.items():
        for rev in revs:
            by_ann[rev.get("annotator") or "a1"][rid] = rev["verdict"]
    kappa = None
    if len(by_ann) >= 2:
        a, b = sorted(by_ann)[:2]
        common = sorted(set(by_ann[a]) & set(by_ann[b]))
        kappa = {"annotators": [a, b], "n": len(common),
                 "kappa": cohen_kappa([(by_ann[a][i], by_ann[b][i]) for i in common])}
    report = {"updated_records": updated, "missing_ids": sorted(set(reviews) - _all_ids(folder))[:20],
              "verdicts": dict(verdicts), "issues": dict(issues.most_common()), "agreement": kappa,
              "csvs": [str(c) for c in csvs], "date": utc_now()}
    (folder / "review_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _all_ids(folder: Path) -> set[str]:
    return {json.loads(l)["id"] for p in folder.rglob("*.jsonl") if p.name != "rejected.jsonl"
            for l in open(p, encoding="utf-8") if l.strip()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("csv", type=Path, nargs="+")
    args = ap.parse_args(argv)
    print(json.dumps(run(args.folder, args.csv), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

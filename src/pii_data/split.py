"""Train/dev/test splits by group, so near-identical records never cross splits.

    python -m src.pii_data.split data/validated/sri_lankan --out data/splits/sri_lankan
    python -m src.pii_data.split data/sources/multiner --out data/splits/multiner --all-test

The split of a record depends only on hash(group): every variant of one source text, and
every language of one multiNER sentence, lands in the same split, and re-running gives the
same answer. Only records whose validation.status is in --keep are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .manifest import write_manifest

KEEP = ("auto_passed", "judge_passed", "human_verified", "human_edited")


def split_of(group: str, train: float = 0.8, dev: float = 0.1, salt: str = "v1") -> str:
    x = int(hashlib.sha256(f"{salt}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if x < train else "dev" if x < train + dev else "test"


def run(src: Path, out: Path, *, keep: tuple[str, ...] = KEEP, train: float = 0.8, dev: float = 0.1,
        all_test: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8", newline="\n") for s in ("train", "dev", "test")}
    counts: Counter = Counter()
    groups: dict[str, set] = defaultdict(set)
    for path in sorted(src.rglob("*.jsonl")):
        if path.name in ("rejected.jsonl",) or path.name.endswith(".rejected.jsonl"):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("validation", {}).get("status") not in keep:
                counts["dropped_status"] += 1
                continue
            group = rec.get("group") or rec["id"]
            s = "test" if all_test else split_of(group, train, dev)
            rec["split"] = s
            groups[group].add(s)
            files[s].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[s] += 1
    for f in files.values():
        f.close()
    leaks = [g for g, ss in groups.items() if len(ss) > 1]
    assert not leaks, f"groups in more than one split: {leaks[:3]}"
    write_manifest(out, dataset=f"splits/{out.name}",
                   config={"source": str(src), "keep": list(keep), "train": train, "dev": dev, "all_test": all_test})
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep", default=",".join(KEEP))
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--dev", type=float, default=0.1)
    ap.add_argument("--all-test", action="store_true", help="put everything in test (external test sets)")
    args = ap.parse_args(argv)
    print(run(args.src, args.out, keep=tuple(args.keep.split(",")), train=args.train, dev=args.dev,
              all_test=args.all_test))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run source importers.

    python -m src.sources.run nemotron [--limit N] [--max-chars 1500]
    python -m src.sources.run wildchat [--shards N] [--limit N]
    python -m src.sources.run aya | multiner | codemix | flores
    python -m src.sources.run all            # everything (full WildChat = ~3.4 GB of downloads)

Outputs: <data_root>/sources/<name>/*.jsonl + manifest.json. Use --limit locally to test.
"""

from __future__ import annotations

import argparse
import sys

NAMES = ("nemotron", "wildchat", "aya", "multiner", "codemix", "flores")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=NAMES + ("all",))
    ap.add_argument("--limit", type=int, default=None, help="max records per output file (testing)")
    ap.add_argument("--shards", type=int, default=None, help="WildChat: number of shards (1-14)")
    ap.add_argument("--max-chars", type=int, default=1500, help="Nemotron: skip longer documents")
    args = ap.parse_args(argv)

    for name in NAMES if args.name == "all" else (args.name,):
        print(f"=== {name}", flush=True)
        if name == "nemotron":
            from . import nemotron_pii
            nemotron_pii.run(limit=args.limit, max_chars=args.max_chars)
        elif name == "wildchat":
            from . import wildchat
            wildchat.run(shards=args.shards, limit=args.limit)
        elif name == "aya":
            from . import aya
            aya.run(limit=args.limit)
        elif name == "multiner":
            from . import multiner
            multiner.run(limit=args.limit)
        elif name == "codemix":
            from . import codemix
            codemix.run(limit=args.limit)
        elif name == "flores":
            from . import flores
            flores.main([])
    return 0


if __name__ == "__main__":
    sys.exit(main())

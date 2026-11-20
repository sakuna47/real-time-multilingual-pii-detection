"""Sri Lankan value resources and personas.

    python -m src.sl_gen.run resources [--cached-json DIR] [--no-wikidata]
    python -m src.sl_gen.run places [--cached-json DIR]          # add all Wikidata towns/villages
    python -m src.sl_gen.run fill-scripts [--limit N] [--dry-run]   # LLM fills missing si/ta forms
    python -m src.sl_gen.run coverage
    python -m src.sl_gen.run personas --n 5 [--variety si]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["resources", "places", "fill-scripts", "coverage", "personas"])
    ap.add_argument("--cached-json", type=Path, default=None)
    ap.add_argument("--no-wikidata", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variety", default="si")
    args = ap.parse_args(argv)

    if args.cmd == "resources":
        from .resources import build, coverage
        print(build(wikidata=not args.no_wikidata, cached_json=args.cached_json))
        print(json.dumps(coverage(), indent=2))
    elif args.cmd == "places":
        from .resources import SPARQL_SETTLEMENTS, build_places, coverage, sparql
        cached = args.cached_json / "wd_settlements.json" if args.cached_json else None
        rows = json.loads(cached.read_text(encoding="utf-8")) if cached and cached.exists() else sparql(SPARQL_SETTLEMENTS)
        print("places:", build_places(rows))
        print(json.dumps(coverage()["places"], indent=2))
    elif args.cmd == "fill-scripts":
        from .fill_scripts import fill_all
        fill_all(limit=args.limit, dry_run=args.dry_run)
    elif args.cmd == "coverage":
        from .resources import coverage
        print(json.dumps(coverage(), indent=2))
    elif args.cmd == "personas":
        from . import values, varieties
        from .personas import PersonaFactory
        f = PersonaFactory(seed=args.seed)
        v = varieties.get(args.variety)
        for k in range(args.n):
            p = f.make(k)
            rng = random.Random(k)
            shown = {lab: values.render(lab, sub, p, v, rng)[0]
                     for lab, sub in [("PERSON", None), ("ID_NUMBER", None), ("PHONE", "MOBILE"),
                                      ("ADDRESS", None), ("DOB", None), ("EMAIL", None)]}
            print(p.id, "F" if p.female else "M", p.dob, json.dumps(shown, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

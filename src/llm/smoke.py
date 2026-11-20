"""Check that each provider works with your keys.

    python -m src.llm.smoke                       # one JSON call per provider that has a key
    python -m src.llm.smoke --list-models         # models your keys can use
    python -m src.llm.smoke --model deepseek:deepseek-flash --model openrouter:<vendor/model>
    python -m src.llm.smoke --dry-run             # offline, mock provider
"""

from __future__ import annotations

import argparse
import sys

from ..common.secrets import MissingSecret, has_secret
from .client import get_llm
from .config import load_config, split_spec
from .prompts import load_prompt
from .providers import build_provider

LIVE = ("gemini", "vertex", "openrouter", "deepseek")


def _default_specs(cfg) -> list[str]:
    """First configured model for each provider that has a key."""
    specs: dict[str, str] = {}
    for role in cfg.roles.values():
        provider, _ = split_spec(role.model)
        if provider in LIVE and provider not in specs:
            specs[provider] = role.model
    out = []
    for provider in LIVE:
        pc = cfg.provider(provider)
        key_env = pc.get("key_env", f"{provider.upper()}_API_KEY")
        if provider == "vertex":
            import os
            if not (has_secret(key_env) or os.environ.get(pc.get("project_env", "GOOGLE_CLOUD_PROJECT"))):
                print("skip vertex: set GOOGLE_CLOUD_PROJECT (with gcloud login) or VERTEX_API_KEY")
                continue
        elif not has_secret(key_env):
            print(f"skip {provider}: {key_env} not set")
            continue
        if provider in specs:
            out.append(specs[provider])
        else:
            print(f"skip {provider}: no role uses it in configs/llm.toml (pass --model {provider}:<id>)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", help="<provider>:<model>, repeatable")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()

    if args.list_models:
        for provider in LIVE:
            try:
                models = build_provider(provider, cfg).list_models()
            except MissingSecret as e:
                print(f"{provider}: {e}")
                continue
            except Exception as e:
                print(f"{provider}: FAILED {e}")
                continue
            print(f"{provider}: {len(models)} models")
            for m in models:
                print(f"  {provider}:{m}")
        return 0

    specs = args.model or (["mock:echo"] if args.dry_run else _default_specs(cfg))
    if not specs:
        print("No providers to test. Add keys to .env (see .env.example).")
        return 1
    prompt = load_prompt("smoke_v1")
    failures = 0
    for spec in specs:
        try:
            llm = get_llm(spec, config=cfg, dry_run=args.dry_run, use_cache=False, run_id="smoke")
            req = prompt.request(word="name")
            req.max_tokens, req.temperature = 1024, 0.0
            data, resp = llm.complete_json(req)
            ok = isinstance(data, dict) and (args.dry_run or
                                            all("{PERSON_1}" in str(data.get(k, "")) for k in ("si", "ta")))
            failures += not ok
            print(f"{'OK  ' if ok else 'BAD '} {resp.name}  {resp.latency_s:.1f}s  "
                  f"tokens in/out {resp.usage.get('input_tokens')}/{resp.usage.get('output_tokens')}  "
                  f"-> {data}")
        except Exception as e:
            failures += 1
            print(f"FAIL {spec}: {type(e).__name__}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

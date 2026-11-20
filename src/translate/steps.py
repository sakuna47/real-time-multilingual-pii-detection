"""LLM steps of the engine: insert placeholders, translate, code-mix.

Each step sends items in batches, checks every answer, and retries failed items one at a
time (with a different cache key) before giving up. Results are (value, None) or
(None, "reason") per item, in input order.
"""

from __future__ import annotations

import difflib
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

from ..common.paths import sources_dir
from ..llm import LLM, load_prompt, run_batch
from ..pii_data.record import detect_scripts
from ..sl_gen.varieties import Variety
from . import placeholders as ph

LANG_NAMES = {"si": "Sinhala", "ta": "Tamil", "en": "English"}
SCRIPT_NAMES = {"si": "Sinhala", "ta": "Tamil"}
SUFFIX_EXAMPLE = {"si": "{PERSON_1}ට or {PERSON_1}ගේ", "ta": "{PERSON_1}க்கு or {PERSON_1}இன்"}
NATIVE = {"si": "Sinh", "ta": "Taml"}
# translate_v2 also localises foreign context (states, currency, SSN -> NIC); v1 kept for old records
TRANSLATE_PROMPT = "translate_v2"

Check = Callable[[Any, Any], tuple[Any, str | None]]   # (input item, model output) -> (value, error)


def _batched(llm: LLM, prompt_id: str, items: list[Any], list_key: str, check: Check, *,
             vars: dict, batch_size: int, workers: int,
             to_json: Callable[[Any], Any] = lambda x: x) -> list[tuple[Any, str | None]]:
    prompt = load_prompt(prompt_id)
    results: list[tuple[Any, str | None]] = [(None, "not run")] * len(items)

    def requests(idx_groups: list[list[int]], salt: str | None):
        reqs = []
        for g in idx_groups:
            req = prompt.request(n=len(g), items_json=json.dumps([to_json(items[i]) for i in g],
                                                                  ensure_ascii=False, indent=1), **vars)
            req.cache_salt = salt
            reqs.append(req)
        return reqs

    def apply(idx_groups, responses) -> list[int]:
        retry = []
        for g, r in zip(idx_groups, responses):
            outs = r.data.get(list_key) if r.ok and isinstance(r.data, dict) else None
            if not isinstance(outs, list) or len(outs) != len(g):
                why = f"{type(r.error).__name__}: {r.error}"[:200] if r.error else \
                      f"bad reply shape ({len(outs) if isinstance(outs, list) else type(outs).__name__})"
                for i in g:
                    results[i] = (None, why)
                retry.extend(g)
                continue
            for i, out in zip(g, outs):
                value, err = check(items[i], out)
                results[i] = (value, err)
                if err:
                    retry.append(i)
        return retry

    groups = [list(range(i, min(i + batch_size, len(items)))) for i in range(0, len(items), batch_size)]
    retry = apply(groups, run_batch(llm, requests(groups, None), workers=workers, json=True, progress_every=0))
    if retry:
        singles = [[i] for i in retry]
        apply(singles, run_batch(llm, requests(singles, "retry1"), workers=workers, json=True, progress_every=0))
    return results


def _strip_placeholders(text: str) -> str:
    for tok, *_ in ph.find(text):
        text = text.replace(tok, " ")
    return text


# ---------------------------------------------------------------- insert

def check_insert(original: str, out: Any) -> tuple[Any, str | None]:
    if not isinstance(out, dict) or "suitable" not in out:
        return None, "insert: bad item"
    if not out.get("suitable"):
        return {"suitable": False}, None
    text = (out.get("text") or "").strip()
    declared = out.get("placeholders") or {}
    found = {t[1:-1] for t in ph.counts(text)}
    if not found:
        return None, "insert: no placeholders"
    if len(found) > 6:
        return None, "insert: too many placeholders"
    if set(declared) != found:
        return None, f"insert: declared {sorted(declared)} != found {sorted(found)}"
    if ph.stray(text):
        return None, f"insert: invalid tokens {ph.stray(text)[:3]}"
    ratio = difflib.SequenceMatcher(None, original, _strip_placeholders(text)).ratio()
    if ratio < 0.5:
        return None, f"insert: rewrote too much (similarity {ratio:.2f})"
    slots = {}
    for name, meta in declared.items():
        label, n = name.rsplit("_", 1)
        sub = (meta or {}).get("subtype") if isinstance(meta, dict) else None
        slots["{" + name + "}"] = {"label": label, "subtype": sub,
                                   "entity": f"p{n}" if label == "PERSON" else "primary",
                                   "part": "full", "original": ""}
    return {"suitable": True, "text": text, "slots": slots, "similarity": round(ratio, 3)}, None


def insert(llm: LLM, texts: list[str], lang: str, batch_size: int = 4, workers: int = 4):
    return _batched(llm, "insert_v1", texts, "items", check_insert,
                    vars={"language": LANG_NAMES[lang]}, batch_size=batch_size, workers=workers)


# ---------------------------------------------------------------- translate

def _script_ok(text: str, lang: str) -> bool:
    return NATIVE[lang] in detect_scripts(_strip_placeholders(text))


def make_check_translate(lang: str) -> Check:
    def check(src: str, out: Any):
        if not isinstance(out, str) or not out.strip():
            return None, "translate: empty"
        out = out.strip()
        ok, why = ph.preserved(src, out)
        if not ok:
            return None, f"translate: placeholders {why}"
        if not _script_ok(out, lang):
            return None, f"translate: no {SCRIPT_NAMES[lang]} script in output"
        return out, None
    return check


def translate(llm: LLM, texts: list[str], lang: str, batch_size: int = 4, workers: int = 4):
    return _batched(llm, TRANSLATE_PROMPT, texts, "translations", make_check_translate(lang),
                    vars={"language": LANG_NAMES[lang], "script": SCRIPT_NAMES[lang],
                          "suffix_example": SUFFIX_EXAMPLE[lang]},
                    batch_size=batch_size, workers=workers)


# ---------------------------------------------------------------- code-mix

STYLES = {
    "si-en": "Sinhala in Sinhala script with the English words Sri Lankans commonly use mixed in, written in "
             "English letters (for example: මට loan එකක් apply කරන්න ඕනේ).",
    "ta-en": "Tamil in Tamil script with the English words Sri Lankan Tamils commonly use mixed in, written in "
             "English letters (for example: எனக்கு ஒரு loan apply பண்ண வேண்டும்).",
    "si-rom": "Singlish: Sinhala written entirely in English letters in casual chat spelling, with English words "
              "mixed in, and no Sinhala script at all (for example: mata loan ekak apply karanna oone).",
    "ta-rom": "Tanglish: Tamil written entirely in English letters in casual chat spelling, with English words "
              "mixed in, and no Tamil script at all (for example: enakku oru loan apply panna venum).",
}


def style_examples(variety: str, k: int = 5, seed: int = 0) -> str:
    """Real examples from the code-mixing pools (src/sources/codemix.py), if imported."""
    files = {"si-rom": ("singlish_gold.jsonl", "rom"), "si-en": ("si_en_social.jsonl", "text")}
    if variety not in files:
        return "(none available; follow the description)"
    path = sources_dir("codemix") / files[variety][0]
    if not Path(path).exists():
        return "(none available; follow the description)"
    rows = [json.loads(l)[files[variety][1]] for l in open(path, encoding="utf-8")]
    if variety == "si-en":   # keep sentences that really mix Sinhala script and Latin
        rows = [r for r in rows if {"Sinh", "Latn"} <= set(detect_scripts(r)) and len(r) < 160]
    rng = random.Random(f"{variety}:{seed}")
    return "\n".join(f"- {r}" for r in rng.sample(rows, min(k, len(rows))))


def make_check_codemix(v: Variety) -> Check:
    native = v.native_script

    def check(src: str, out: Any):
        if not isinstance(out, str) or not out.strip():
            return None, "codemix: empty"
        out = out.strip()
        ok, why = ph.preserved(src, out)
        if not ok:
            return None, f"codemix: placeholders {why}"
        scripts = set(detect_scripts(_strip_placeholders(out)))
        if v.romanized and native in scripts:
            return None, "codemix: native script left in romanised text"
        if v.romanized and "Latn" not in scripts:
            return None, "codemix: no Latin text"
        if not v.romanized and not {native, "Latn"} <= scripts:
            return None, "codemix: not mixed (needs both scripts)"
        return out, None
    return check


def codemix(llm: LLM, texts: list[str], v: Variety, batch_size: int = 4, workers: int = 4, seed: int = 0):
    return _batched(llm, "codemix_v1", texts, "texts", make_check_codemix(v),
                    vars={"language": LANG_NAMES[v.lang], "style": STYLES[v.code],
                          "examples": style_examples(v.code, seed=seed)},
                    batch_size=batch_size, workers=workers)


# ---------------------------------------------------------------- dry run

_ITEMS = re.compile(r"(?:PROMPTS|TEXTS):\n(.*)\Z", re.S)


def dry_run_handler(req) -> str:
    """Offline stand-in for the LLM so the whole pipeline can be exercised with --dry-run."""
    items = json.loads(_ITEMS.search(req.messages[-1].content).group(1))
    pid = req.prompt_id
    if pid == "insert_v1":
        return json.dumps({"items": [{"suitable": True, "text": f"{t} මගේ නම {{PERSON_1}}, දුරකථනය {{PHONE_1}}.",
                                      "placeholders": {"PERSON_1": {"subtype": None},
                                                       "PHONE_1": {"subtype": "MOBILE"}}} for t in items]},
                          ensure_ascii=False)
    if pid.startswith("translate_"):
        native = "පරිවර්තනය" if "Sinhala" in req.messages[0].content else "மொழிபெயர்ப்பு"
        return json.dumps({"translations": [f"{native}: {t}" for t in items]}, ensure_ascii=False)
    if pid == "codemix_v1":
        rom = "entirely in English letters" in req.messages[0].content
        outs = []
        for t in items:
            outs.append(re.sub(r"[඀-෿஀-௿]+", "x", t) + " ok" if rom else t + " please")
        return json.dumps({"texts": outs}, ensure_ascii=False)
    raise ValueError(f"dry run: unknown prompt {pid}")

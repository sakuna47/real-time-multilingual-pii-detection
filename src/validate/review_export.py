"""Human review sheet: a self-contained HTML file to open in any browser.

    python -m src.validate.review_export data/validated/sri_lankan --per-group 10 --out data/review/pilot.html

Shows each record with its PII spans highlighted, and for translated records the original
source text with its spans, so translation and labels can be judged together. Reviewers
mark each record OK / needs fix / reject, tick issue types and add notes; marks are kept in
the browser (localStorage) while working, and "Export CSV" saves them for
`python -m src.validate.review_import`.

Sampling is stratified by (source, variety): up to --per-group records each, chosen by a
stable hash so re-exporting gives the same sample.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

from ..common.paths import data_root, sources_dir
from ..common.runinfo import utc_now

ISSUES = [
    ("translation", "Meaning wrong / mistranslated"),
    ("unnatural", "Unnatural for this variety"),
    ("variety", "Wrong script or not really code-mixed"),
    ("span", "Span boundary or label wrong"),
    ("missed", "PII in text that is not labelled"),
    ("value", "Value implausible (name, address, number)"),
    ("foreign", "Foreign context left (US places, $, SSN)"),
]
SOURCE_FILES = {"nemotron": ["nemotron_pii/train.jsonl", "nemotron_pii/test.jsonl"],
                "wildchat": ["wildchat/prompts.jsonl", "wildchat/native.jsonl"],
                "aya": ["aya/si.jsonl", "aya/ta.jsonl"]}


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def highlight(text: str, spans: list[dict], distractors: list[dict] | None = None) -> str:
    marks = [(s["start"], s["end"], s["label"] + (f":{s['subtype']}" if s.get("subtype") else ""), "pii")
             for s in spans]
    marks += [(d["start"], d["end"], "not PII: " + d["kind"], "neg") for d in distractors or []]
    out, pos = [], 0
    for st, en, label, cls in sorted(marks):
        out.append(html.escape(text[pos:st]))
        out.append(f'<mark class="{cls}" title="{html.escape(label)}">{html.escape(text[st:en])}'
                   f'<sub>{html.escape(label)}</sub></mark>')
        pos = en
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def load_parents(ids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for files in SOURCE_FILES.values():
        for rel in files:
            p = sources_dir() / rel
            if not p.exists():
                continue
            for line in open(p, encoding="utf-8"):
                if not ids - found.keys():
                    return found
                r = json.loads(line)
                if r["id"] in ids:
                    found[r["id"]] = r
    return found


def sample(folder: Path, per_group: int, include_negatives: bool, include_reviewed: bool = False) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in sorted(folder.rglob("*.jsonl")):
        if p.name == "rejected.jsonl":
            continue
        src = p.parent.name
        if src == "negatives" and not include_negatives:
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            v = r.get("validation") or {}
            if v.get("status") == "auto_failed":
                continue
            if v.get("human") and not include_reviewed:
                continue
            groups[(src, (r.get("ext") or {}).get("variety", "?"))].append(r)
    out = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: _h(r["id"]))
        out.extend(rows[:per_group])
    return out


CSS = """
:root{--bg:#f5f6f8;--card:#fff;--fg:#1b2430;--muted:#5d6878;--line:#d8dde5;--pii:#dcefe9;--piib:#11795f;
--neg:#fdeedd;--negb:#a0570f;--ok:#11795f;--bad:#b3261e;--fix:#a0570f;--accent:#2457c5}
@media (prefers-color-scheme:dark){:root{--bg:#11161d;--card:#18202a;--fg:#e3e8ef;--muted:#99a4b3;--line:#2c3746;
--pii:#163a31;--piib:#5fd0ac;--neg:#3b2a17;--negb:#f0b36a;--ok:#5fd0ac;--bad:#ff8a80;--fix:#f0b36a;--accent:#8fb0ff;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,"Segoe UI","Iskoola Pota","Nirmala UI",sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;
display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:5}
header h1{font-size:17px;margin:0 8px 0 0}
main{max-width:980px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.card.ok{border-left:4px solid var(--ok)}.card.fix{border-left:4px solid var(--fix)}.card.reject{border-left:4px solid var(--bad)}
.meta{font:12px ui-monospace,Consolas,monospace;color:var(--muted);display:flex;gap:10px;flex-wrap:wrap}
.text{font-size:17px;margin:8px 0;overflow-wrap:anywhere}
.src{font-size:13px;color:var(--muted);border-top:1px dashed var(--line);margin-top:8px;padding-top:6px;overflow-wrap:anywhere}
mark{border-radius:3px;padding:0 2px;color:inherit}mark.pii{background:var(--pii);border-bottom:2px solid var(--piib)}
mark.neg{background:var(--neg);border-bottom:2px dashed var(--negb)}
mark sub{font:10px ui-monospace,Consolas,monospace;color:var(--muted);margin-left:2px}
.ctl{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font-size:13px}
.issues{display:flex;gap:4px 12px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-top:6px}
textarea{width:100%;box-sizing:border-box;margin-top:6px;font:inherit;font-size:13px;background:var(--bg);color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:6px;min-height:34px}
button,input[type=text]{font:inherit;font-size:13px;padding:5px 10px;border-radius:6px;border:1px solid var(--line);
background:var(--card);color:var(--fg)}button{cursor:pointer}button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.count{font-size:13px;color:var(--muted)}details{font-size:13px;color:var(--muted)}
"""

JS = r"""
const KEY = "pii-review:" + SHEET.id;
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch (e) {} count(); }
function count(){
  const n = Object.values(marks).filter(m => m.verdict).length;
  document.getElementById("count").textContent = n + " / " + SHEET.ids.length + " reviewed";
}
function paint(id){
  const c = document.getElementById("c-" + id); const m = marks[id] || {};
  c.className = "card" + (m.verdict ? " " + m.verdict : "");
}
document.querySelectorAll(".card").forEach(card => {
  const id = card.dataset.id; const m = marks[id] || {};
  card.querySelectorAll("input[type=radio]").forEach(r => { r.checked = m.verdict === r.value;
    r.addEventListener("change", () => { (marks[id] = marks[id] || {}).verdict = r.value; paint(id); save(); }); });
  card.querySelectorAll("input[type=checkbox]").forEach(b => { b.checked = (m.issues || []).includes(b.value);
    b.addEventListener("change", () => { const mm = (marks[id] = marks[id] || {});
      mm.issues = [...card.querySelectorAll("input[type=checkbox]:checked")].map(x => x.value); save(); }); });
  const t = card.querySelector("textarea"); t.value = m.notes || "";
  t.addEventListener("input", () => { (marks[id] = marks[id] || {}).notes = t.value; save(); });
  paint(id);
});
const ann = document.getElementById("annotator");
try { ann.value = localStorage.getItem("pii-review:annotator") || "a1"; } catch (e) { ann.value = "a1"; }
ann.addEventListener("input", () => { try { localStorage.setItem("pii-review:annotator", ann.value); } catch (e) {} });
function csvCell(s){ s = String(s ?? ""); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }
document.getElementById("export").addEventListener("click", () => {
  const rows = [["id","annotator","verdict","issues","notes","sheet"]];
  for (const id of SHEET.ids) { const m = marks[id]; if (!m || !m.verdict) continue;
    rows.push([id, ann.value || "a1", m.verdict, (m.issues || []).join(";"), m.notes || "", SHEET.id]); }
  const blob = new Blob(["﻿" + rows.map(r => r.map(csvCell).join(",")).join("\n")], {type: "text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "review_" + SHEET.id + "_" + (ann.value || "a1") + ".csv"; a.click();
});
count();
"""


def render(records: list[dict], parents: dict[str, dict], sheet_id: str) -> str:
    cards = []
    for r in records:
        rid = html.escape(r["id"])
        variety = (r.get("ext") or {}).get("variety", "?")
        parent = parents.get((r.get("origin") or {}).get("parent_id") or "")
        src_html = ""
        if parent:
            src_html = (f'<div class="src"><b>Source ({html.escape(parent["language"]["primary"])}):</b> '
                        f'{highlight(parent["text"], parent.get("spans", []))}</div>')
        masked = (r.get("ext") or {}).get("masked_target")
        masked_html = f'<details><summary>Text before filling</summary>{html.escape(masked)}</details>' if masked else ""
        issues = "".join(f'<label><input type="checkbox" value="{k}"> {html.escape(v)}</label>' for k, v in ISSUES)
        cards.append(f"""
<section class="card" id="c-{rid}" data-id="{rid}">
  <div class="meta"><span>{rid}</span><span>variety: <b>{html.escape(variety)}</b></span>
    <span>{html.escape((r.get('origin') or {}).get('method') or '')}</span><span>{html.escape(r.get('domain') or '')}</span>
    <span>{', '.join(html.escape(t) for t in r.get('tags', []))}</span></div>
  <div class="text">{highlight(r['text'], r['spans'], r.get('distractors'))}</div>
  {src_html}{masked_html}
  <div class="ctl">
    <label><input type="radio" name="v-{rid}" value="ok"> OK</label>
    <label><input type="radio" name="v-{rid}" value="fix"> Needs fix</label>
    <label><input type="radio" name="v-{rid}" value="reject"> Reject</label>
  </div>
  <div class="issues">{issues}</div>
  <textarea placeholder="Notes (what is wrong, the correct wording, ...)"></textarea>
</section>""")
    sheet = {"id": sheet_id, "ids": [r["id"] for r in records]}
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PII Review {html.escape(sheet_id)}</title>
<style>{CSS}</style></head><body>
<header><h1>PII dataset review</h1><span class="count" id="count"></span>
<label class="count">Annotator <input type="text" id="annotator" size="5" title="Pseudonym, e.g. a1"></label>
<button class="primary" id="export">Export CSV</button>
<span class="count">Green = labelled PII · dashed = deliberate non-PII look-alike</span></header>
<main>{''.join(cards)}</main>
<script>const SHEET = {json.dumps(sheet)};</script><script>{JS}</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="e.g. data/validated/sri_lankan")
    ap.add_argument("--per-group", type=int, default=10, help="records per (source, variety)")
    ap.add_argument("--negatives", action="store_true", help="include hard negatives")
    ap.add_argument("--include-reviewed", action="store_true", help="also show records already reviewed")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    records = sample(args.folder, args.per_group, args.negatives, args.include_reviewed)
    parents = load_parents({(r.get("origin") or {}).get("parent_id") for r in records} - {None})
    sheet_id = utc_now()[:16].replace(":", "").replace("-", "")
    out = args.out or data_root() / "review" / f"review_{sheet_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(records, parents, sheet_id), encoding="utf-8")
    print(f"{len(records)} records -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

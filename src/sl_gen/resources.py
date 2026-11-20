"""Sri Lankan value resources: resources/sl/*.csv (committed, human-reviewable).

Every row carries `source`, `source_ref` and `needs_review`. Build or refresh with
    python -m src.sl_gen.run resources            # Wikidata names + places, curated tables
    python -m src.sl_gen.run fill-scripts         # LLM fills missing Sinhala/Tamil forms (marked source=llm)

Files
  first_names.csv  latin, sinhala, tamil, gender (m/f/u), people   Wikidata: given names (P735) of people with
                                                                    Sri Lankan citizenship (P27=Q854)
  surnames.csv     latin, sinhala, tamil, people                   Wikidata: family names (P734), same people
  places.csv       latin, sinhala, tamil, postcode                 Wikidata: items in Sri Lanka (P17) with a postal code (P281)
  phone_codes.csv  code, type, area                                TRCSL national numbering plan
  streets.csv      latin, sinhala, tamil                           curated street-name stems (review!)
  address_words.csv key, latin, sinhala, tamil                     curated: No./Road/Lane/... (review!)
  months.csv       month, latin, sinhala, tamil                    curated month names (review!)
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from ..common.paths import REPO_ROOT
from ..common.runinfo import utc_now
from . import phone

RES_DIR = REPO_ROOT / "resources" / "sl"
WD = "Wikidata Query Service, accessed {date}"
MALE, FEMALE = "Q6581097", "Q6581072"
_SINH = re.compile(r"[඀-෿]")
_TAML = re.compile(r"[஀-௿]")


def res_dir() -> Path:
    """PII_RESOURCES_DIR overrides the folder (used by tests)."""
    env = os.environ.get("PII_RESOURCES_DIR", "").strip()
    return Path(env) if env else RES_DIR


@lru_cache(maxsize=None)
def load(name: str, directory: str | None = None) -> tuple[dict, ...]:
    path = Path(directory or res_dir()) / f"{name}.csv"
    with open(path, encoding="utf-8", newline="") as f:
        return tuple(csv.DictReader(f))


def write_csv(name: str, rows: list[dict], fields: list[str], directory: Path | None = None) -> Path:
    d = directory or res_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    load.cache_clear()
    return path


# ---------------------------------------------------------------- Wikidata

SPARQL_NAMES = """SELECT ?n ?en ?si ?ta ?g (COUNT(DISTINCT ?p) AS ?people) WHERE {
  ?p wdt:P27 wd:Q854; wdt:P31 wd:Q5; wdt:%s ?n. OPTIONAL { ?p wdt:P21 ?g }
  OPTIONAL { ?n rdfs:label ?en FILTER(LANG(?en)="en") }
  OPTIONAL { ?n rdfs:label ?si FILTER(LANG(?si)="si") }
  OPTIONAL { ?n rdfs:label ?ta FILTER(LANG(?ta)="ta") }
} GROUP BY ?n ?en ?si ?ta ?g"""
SPARQL_PLACES = """SELECT ?p ?en ?si ?ta ?pc WHERE {
  ?p wdt:P17 wd:Q854; wdt:P281 ?pc.
  OPTIONAL { ?p rdfs:label ?en FILTER(LANG(?en)="en") }
  OPTIONAL { ?p rdfs:label ?si FILTER(LANG(?si)="si") }
  OPTIONAL { ?p rdfs:label ?ta FILTER(LANG(?ta)="ta") }
}"""


def sparql(query: str, timeout: int = 120, retries: int = 3) -> list[dict]:
    import time
    import urllib.error
    import urllib.parse
    import urllib.request
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "pii-dataset-research-bot/0.1 (academic research)",
                                               "Accept": "application/sparql-results+json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            return [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(75)  # WDQS can rate-limit to 1 request/minute
    return []


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _clean(label: str | None, script: re.Pattern | None = None) -> str:
    if not label:
        return ""
    label = label.strip()
    if script is not None and not script.search(label):
        return ""   # a "Sinhala" label written in Latin script is not useful
    if label.startswith("Q") and label[1:].isdigit():
        return ""
    return label


def names_from_wikidata(rows: list[dict], with_gender: bool) -> list[dict]:
    agg: dict[str, dict] = {}
    for r in rows:
        q = _qid(r["n"])
        a = agg.setdefault(q, {"qid": q, "latin": _clean(r.get("en")), "sinhala": _clean(r.get("si"), _SINH),
                               "tamil": _clean(r.get("ta"), _TAML), "m": 0, "f": 0, "people": 0})
        n = int(r.get("people", 0))
        a["people"] += n
        g = _qid(r.get("g", ""))
        if g == MALE:
            a["m"] += n
        elif g == FEMALE:
            a["f"] += n
    out = []
    for a in agg.values():
        # single-token Latin names only (skip initials like "K." and multi-word labels)
        if not a["latin"] or not re.fullmatch(r"[A-Za-z][A-Za-z'\-]{1,30}", a["latin"]):
            continue
        row = {"latin": a["latin"], "sinhala": a["sinhala"], "tamil": a["tamil"], "people": a["people"],
               "source": "wikidata", "source_ref": f"wikidata:{a['qid']}",
               "needs_review": "no" if a["sinhala"] and a["tamil"] else "yes"}
        if with_gender:
            m, f = a["m"], a["f"]
            row["gender"] = "m" if m > 2 * f else "f" if f > 2 * m else "u"
        out.append(row)
    out.sort(key=lambda r: (-int(r["people"]), r["latin"]))
    return out


def places_from_wikidata(rows: list[dict]) -> list[dict]:
    agg: dict[str, dict] = {}
    for r in rows:
        q = _qid(r["p"])
        pc = re.sub(r"\D", "", r.get("pc", ""))
        if len(pc) != 5:
            continue
        a = agg.setdefault(q, {"latin": _clean(r.get("en")), "sinhala": _clean(r.get("si"), _SINH),
                               "tamil": _clean(r.get("ta"), _TAML), "postcode": pc,
                               "source": "wikidata", "source_ref": f"wikidata:{q}"})
        a["needs_review"] = "no" if a["sinhala"] and a["tamil"] else "yes"
    out = [a for a in agg.values() if a["latin"] and re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{1,40}", a["latin"])]
    out.sort(key=lambda r: r["latin"])
    return out


# ---------------------------------------------------------------- curated tables

def phone_rows() -> list[dict]:
    ref = "TRCSL National Numbering Plan"
    rows = [{"code": p, "type": "mobile", "area": "", "source": "trcsl", "source_ref": ref, "needs_review": "no"}
            for p in phone.MOBILE_PREFIXES]
    rows += [{"code": c, "type": "landline", "area": a, "source": "trcsl", "source_ref": ref, "needs_review": "no"}
             for c, a in sorted(phone.AREA_CODES.items())]
    return rows


CURATED_ADDRESS_WORDS = [
    # key, latin, sinhala, tamil
    ("no", "No.", "අංක", "இல."),
    ("road", "Road", "පාර", "வீதி"),
    ("mawatha", "Mawatha", "මාවත", "மாவத்தை"),
    ("lane", "Lane", "පටුමග", "ஒழுங்கை"),
    ("street", "Street", "වීදිය", "தெரு"),
]
CURATED_STREETS = [
    # stem: latin, sinhala, tamil (combined with a road/lane word)
    ("Galle", "ගාලු", "காலி"), ("Kandy", "මහනුවර", "கண்டி"), ("Temple", "පන්සල්", "கோயில்"),
    ("Station", "දුම්රිය ස්ථාන", "புகையிரத நிலைய"), ("Hospital", "රෝහල්", "வைத்தியசாலை"),
    ("School", "පාසල්", "பாடசாலை"), ("Church", "පල්ලි", "தேவாலய"), ("Main", "ප්‍රධාන", "பிரதான"),
    ("Negombo", "මීගමු", "நீர்கொழும்பு"), ("High Level", "හයිලෙවල්", "ஹைலெவல்"),
]
CURATED_MONTHS = [
    ("January", "ජනවාරි", "ஜனவரி"), ("February", "පෙබරවාරි", "பிப்ரவரி"), ("March", "මාර්තු", "மார்ச்"),
    ("April", "අප්‍රේල්", "ஏப்ரல்"), ("May", "මැයි", "மே"), ("June", "ජූනි", "ஜூன்"),
    ("July", "ජූලි", "ஜூலை"), ("August", "අගෝස්තු", "ஆகஸ்ட்"), ("September", "සැප්තැම්බර්", "செப்டம்பர்"),
    ("October", "ඔක්තෝබර්", "அக்டோபர்"), ("November", "නොවැම්බර්", "நவம்பர்"),
    ("December", "දෙසැම්බර්", "டிசம்பர்"),
]


def build(wikidata: bool = True, cached_json: Path | None = None) -> dict[str, int]:
    """Write all resource CSVs. cached_json: folder with wd_given.json / wd_family.json /
    wd_places.json from an earlier query (skips the network)."""
    import time
    stamp = WD.format(date=utc_now()[:10])
    counts: dict[str, int] = {}
    if wikidata:
        def get(name, query):
            if cached_json and (cached_json / f"wd_{name}.json").exists():
                return json.loads((cached_json / f"wd_{name}.json").read_text(encoding="utf-8"))
            rows = sparql(query)
            time.sleep(65)  # stay under the WDQS rate limit between queries
            return rows
        given = names_from_wikidata(get("given", SPARQL_NAMES % "P735"), with_gender=True)
        family = names_from_wikidata(get("family", SPARQL_NAMES % "P734"), with_gender=False)
        places = places_from_wikidata(get("places", SPARQL_PLACES))
        cols = ["latin", "sinhala", "tamil", "people", "source", "source_ref", "needs_review"]
        write_csv("first_names", given, cols[:3] + ["gender"] + cols[3:])
        write_csv("surnames", family, cols)
        write_csv("places", places, ["latin", "sinhala", "tamil", "postcode", "source", "source_ref", "needs_review"])
        counts.update(first_names=len(given), surnames=len(family), places=len(places))
    write_csv("phone_codes", phone_rows(), ["code", "type", "area", "source", "source_ref", "needs_review"])
    cur = {"source": "curated", "source_ref": "project glossary", "needs_review": "yes"}
    write_csv("address_words", [{"key": k, "latin": l, "sinhala": s, "tamil": t, **cur}
                                for k, l, s, t in CURATED_ADDRESS_WORDS],
              ["key", "latin", "sinhala", "tamil", "source", "source_ref", "needs_review"])
    write_csv("streets", [{"latin": l, "sinhala": s, "tamil": t, **cur} for l, s, t in CURATED_STREETS],
              ["latin", "sinhala", "tamil", "source", "source_ref", "needs_review"])
    write_csv("months", [{"month": i + 1, "latin": l, "sinhala": s, "tamil": t, **cur}
                         for i, (l, s, t) in enumerate(CURATED_MONTHS)],
              ["month", "latin", "sinhala", "tamil", "source", "source_ref", "needs_review"])
    meta = {"built_at": utc_now(), "wikidata": stamp if wikidata else None, "counts": counts}
    (res_dir() / "SOURCES.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return counts


def complete_rows(name: str, scripts: tuple[str, ...] = ("latin", "sinhala", "tamil"),
                  directory: str | None = None) -> list[dict]:
    """Rows that have every requested script form (what personas can use)."""
    return [r for r in load(name, directory) if all(r.get(s) for s in scripts)]


def coverage(directory: str | None = None) -> dict[str, dict[str, int]]:
    out = defaultdict(dict)
    for name in ("first_names", "surnames", "places"):
        rows = load(name, directory)
        out[name] = {"rows": len(rows), "with_sinhala": sum(bool(r["sinhala"]) for r in rows),
                     "with_tamil": sum(bool(r["tamil"]) for r in rows),
                     "complete": len(complete_rows(name, directory=directory))}
    return dict(out)

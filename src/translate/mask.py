"""v1 record with spans -> masked text + slots.

    m = mask(record)
    m.text   'Hi, I am {PERSON_1}. Call {PHONE_1}.'
    m.slots  {'{PERSON_1}': Slot(label='PERSON', subtype=None, entity='e1', part='full', original='Jason Lee'), ...}

Identical values share one placeholder. PERSON values that are a first or last token of
another PERSON value are linked to the same entity (part='first'/'last'), so "Jason Lee ...
Jason" becomes one person after filling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .placeholders import token


@dataclass
class Slot:
    label: str
    subtype: str | None
    entity: str           # p1, p2, ... per person; 'primary' for values not tied to a name
    part: str = "full"    # PERSON only: full | first | last
    original: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Masked:
    text: str
    slots: dict[str, Slot]

    def to_dict(self) -> dict:
        return {"text": self.text, "slots": {k: v.to_dict() for k, v in self.slots.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Masked":
        return cls(text=d["text"], slots={k: Slot(**v) for k, v in d["slots"].items()})


def _person_links(values: list[str]) -> dict[str, tuple[str, str]]:
    """value -> (entity key, part). Short names that match a token of a longer name join it."""
    full = sorted(set(values), key=lambda v: -len(v.split()))
    out: dict[str, tuple[str, str]] = {}
    for v in full:
        toks = v.split()
        if len(toks) >= 2:
            out.setdefault(v, (v, "full"))
            continue
        host = next((f for f in full if len(f.split()) >= 2 and v in (f.split()[0], f.split()[-1])), None)
        if host:
            out[v] = (host, "first" if host.split()[0] == v else "last")
        else:
            out[v] = (v, "full")
    return out


FIRST_LABELS = {"first_name", "GIVENNAME1", "GIVENNAME2"}
LAST_LABELS = {"last_name", "LASTNAME1", "LASTNAME2", "LASTNAME3"}


def _field_links(spans: list[dict]) -> dict[str, tuple[str, str]]:
    """Separate first-name and last-name fields (forms) -> one entity with parts first/last.
    A last name joins the most recent first name that has no last name yet."""
    out: dict[str, tuple[str, str]] = {}
    open_key = None
    for i, s in enumerate(spans):
        if s["label"] != "PERSON" or s["value"] in out:
            continue
        src = set(s.get("source_labels") or [])
        if src and src <= FIRST_LABELS:
            open_key = f"field{i}"
            out[s["value"]] = (open_key, "first")
        elif src and src <= LAST_LABELS:
            out[s["value"]] = (open_key or f"field{i}", "last")
            open_key = None
    return out


def mask(record: dict) -> Masked:
    text, spans = record["text"], sorted(record["spans"], key=lambda s: s["start"])
    links = _person_links([s["value"] for s in spans if s["label"] == "PERSON"])
    links.update(_field_links(spans))   # source labels say first/last: trust them over token matching
    entity_ids: dict[str, str] = {}
    slot_for_value: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {}
    slots: dict[str, Slot] = {}
    out, pos = [], 0
    for s in spans:
        key = (s["label"], s["value"])
        tok = slot_for_value.get(key)
        if tok is None:
            counters[s["label"]] = counters.get(s["label"], 0) + 1
            tok = token(s["label"], counters[s["label"]])
            slot_for_value[key] = tok
            if s["label"] == "PERSON":
                ekey, part = links[s["value"]]
            else:
                ekey, part = "primary", "full"   # non-name values belong to the main persona
            ent = "primary" if ekey == "primary" else entity_ids.setdefault(ekey, f"p{len(entity_ids) + 1}")
            slots[tok] = Slot(label=s["label"], subtype=s.get("subtype"), entity=ent, part=part,
                              original=s["value"])
        out.append(text[pos:s["start"]])
        out.append(tok)
        pos = s["end"]
    out.append(text[pos:])
    # entity "primary" (non-name values) is given the first person's persona when filling
    return Masked(text="".join(out), slots=slots)

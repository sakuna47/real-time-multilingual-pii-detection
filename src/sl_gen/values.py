"""Render a persona's values as text for one placeholder in one variety.

    text, fmt = render("PHONE", "MOBILE", persona, variety, rng)

`fmt` is a tag recording the surface form (e.g. "phone:intl_spaced", "name:sinhala_full"),
so results can later be broken down by format.
"""

from __future__ import annotations

import random

from . import phone
from .personas import Persona
from .resources import load
from .varieties import Variety

SPACE_NIC_P = 0.15


def _script_col(v: Variety, rng: random.Random) -> str:
    return v.column if rng.random() < v.p_native_names else "latin"


def _word(key: str, col: str, directory: str | None = None) -> str:
    for r in load("address_words", directory):
        if r["key"] == key:
            return r[col]
    raise KeyError(key)


def _month(m: int, col: str, directory: str | None = None) -> str:
    return load("months", directory)[m - 1][col]


def name(p: Persona, v: Variety, rng: random.Random, part: str = "full", col: str | None = None
         ) -> tuple[str, str]:
    col = col or _script_col(v, rng)
    first, last = p.first[col], p.last[col]
    if part == "first":
        return first, f"name:{col}_first"
    if part == "last":
        return last, f"name:{col}_last"
    if col == "latin" and rng.random() < 0.15:     # "K. Perera" style, common in Sri Lanka
        return f"{first[0]}. {last}", "name:latin_initial"
    return f"{first} {last}", f"name:{col}_full"


def nic_value(p: Persona, rng: random.Random, subtype: str | None) -> tuple[str, str]:
    use_old = subtype == "NIC_OLD" or (subtype not in ("NIC_NEW",) and p.nic_old and rng.random() < 0.5)
    if use_old and p.nic_old:
        s = p.nic_old
        if rng.random() < 0.2:
            s = s[:-1] + s[-1].lower()
        return s, "nic:old" + ("_lower" if s[-1].islower() else "")
    s = p.nic_new
    if rng.random() < SPACE_NIC_P:
        return f"{s[:4]} {s[4:8]} {s[8:]}", "nic:new_spaced"
    return s, "nic:new"


def dob_value(p: Persona, v: Variety, rng: random.Random, col: str | None = None) -> tuple[str, str]:
    d = p.dob
    col = "latin" if v.romanized else (col or _script_col(v, rng))
    options = [
        (d.isoformat(), "dob:iso"),
        (f"{d.day:02d}/{d.month:02d}/{d.year}", "dob:dmy_slash"),
        (f"{d.day:02d}.{d.month:02d}.{d.year}", "dob:dmy_dot"),
    ]
    if col == "latin":
        options.append((f"{d.day} {_month(d.month, 'latin')} {d.year}", "dob:long_en"))
    else:
        options.append((f"{d.year} {_month(d.month, col)} {d.day:02d}", f"dob:long_{col}"))
    return rng.choice(options)


def address_value(p: Persona, v: Variety, rng: random.Random, col: str | None = None,
                  include_city: bool = True) -> tuple[str, str]:
    """include_city=False when the text names the city separately (a LOCATION placeholder)."""
    col = col or _script_col(v, rng)
    no = _word("no", col)
    street = f"{p.street[col]} {_word(p.street_word, col)}"
    text = f"{no} {p.house}, {street}"
    with_pc = bool(p.place.get("postcode")) and rng.random() < 0.3
    if include_city:
        text += f", {p.place[col]}" + (f" {p.place['postcode']}" if with_pc else "")
    else:
        with_pc = False
    return text, f"address:{col}" + ("_postcode" if with_pc else "") + ("" if include_city else "_nocity")


def script_for(v: Variety, rng: random.Random) -> str:
    """Pick the script column once per person, so one record writes a person consistently."""
    return _script_col(v, rng)


def render(label: str, subtype: str | None, p: Persona, v: Variety, rng: random.Random,
           part: str = "full", col: str | None = None, include_city: bool = True) -> tuple[str, str]:
    """Text for a placeholder of this label. ID_NUMBER defaults to a NIC.
    `col` fixes the script (latin / sinhala / tamil) for names, places, addresses and dates."""
    if label == "PERSON":
        return name(p, v, rng, part, col)
    if label == "ID_NUMBER":
        if subtype == "PASSPORT":
            return f"N{rng.randint(0, 9999999):07d}", "id:passport"
        if subtype == "DRIVING_LICENCE":
            return f"B{rng.randint(0, 9999999):07d}", "id:driving_licence"
        return nic_value(p, rng, subtype)
    if label == "PHONE":
        nsn = p.landline if subtype == "LANDLINE" else p.mobile
        text, fmt = phone.render(nsn, rng)
        return text, f"phone:{fmt}"
    if label == "EMAIL":
        return p.email, "email"
    if label == "ADDRESS":
        return address_value(p, v, rng, col, include_city)
    if label == "LOCATION":
        col = col or _script_col(v, rng)
        return p.place[col], f"location:{col}"
    if label == "DOB":
        return dob_value(p, v, rng, col)
    if label == "USERNAME":
        return p.username, "username"
    if label == "PASSWORD":
        return p.password, "password"
    if label == "IP_ADDRESS":
        return (p.ipv6, "ip:v6") if rng.random() < 0.25 else (p.ipv4, "ip:v4")
    raise ValueError(f"no value generator for label {label!r}")

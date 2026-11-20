"""Deterministic synthetic Sri Lankan personas.

Each persona's values agree with each other: the NIC encodes the date of birth and sex,
the landline area code matches the city when one is known, the email is built from the
romanised name. Persona k with a given seed is always the same person.

    python -m src.sl_gen.run personas --n 5          # print a few
"""

from __future__ import annotations

import random
import string
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from . import nic, phone
from .resources import complete_rows, load

EMAIL_DOMAINS = ["example.lk", "mail.example.lk", "example.com", "example.net", "example.org"]
# RFC 5737 / RFC 3849 documentation ranges mixed with random addresses
DOC_V4 = ["192.0.2", "198.51.100", "203.0.113"]


@dataclass
class Persona:
    id: str
    female: bool
    dob: date
    first: dict           # {"latin", "sinhala", "tamil"}
    last: dict
    nic_old: str | None
    nic_new: str
    place: dict           # {"latin", "sinhala", "tamil", "postcode" (may be ""), "district"}
    street: dict          # {"latin", "sinhala", "tamil"}
    street_word: str      # "road" | "mawatha" | "lane" | "street"
    house: str            # "25/3"
    mobile: str           # national significant number, 9 digits
    landline: str
    email: str
    username: str
    password: str
    ipv4: str
    ipv6: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dob"] = self.dob.isoformat()
        return d


def _dob(rng: random.Random, year_min: int = 1945, year_max: int = 2006) -> date:
    start, end = date(year_min, 1, 1), date(year_max, 12, 31)
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _password(rng: random.Random) -> str:
    chars = string.ascii_letters + string.digits + "!@#$%&*?"
    pw = [rng.choice(string.ascii_uppercase), rng.choice(string.digits), rng.choice("!@#$%&*?")]
    pw += [rng.choice(chars) for _ in range(rng.randint(5, 9))]
    rng.shuffle(pw)
    return "".join(pw)


def _ipv4(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return f"{rng.choice(DOC_V4)}.{rng.randint(1, 254)}"
    return ".".join(str(x) for x in (rng.randint(11, 223), rng.randint(0, 255), rng.randint(0, 255), rng.randint(1, 254)))


def _ipv6(rng: random.Random) -> str:
    return "2001:db8:" + ":".join(f"{rng.randint(0, 0xffff):x}" for _ in range(6))


class PersonaFactory:
    def __init__(self, seed: int = 42, resources_dir: str | None = None):
        self.seed = seed
        self.dir = resources_dir
        firsts = complete_rows("first_names", directory=resources_dir)
        self.first_by_gender = {
            True: [r for r in firsts if r.get("gender") in ("f", "u")],
            False: [r for r in firsts if r.get("gender") in ("m", "u")],
        }
        self.surnames = complete_rows("surnames", directory=resources_dir)
        self.places = complete_rows("places", directory=resources_dir)
        self.streets = complete_rows("streets", directory=resources_dir)
        missing = [n for n, rows in (("first names (female)", self.first_by_gender[True]),
                                     ("first names (male)", self.first_by_gender[False]),
                                     ("surnames", self.surnames), ("places", self.places),
                                     ("streets", self.streets)) if not rows]
        if missing:
            raise RuntimeError(f"resources/sl has no complete rows (latin+sinhala+tamil) for: {missing}. "
                               "Run: python -m src.sl_gen.run resources, then fill-scripts.")
        self.area_by_place = {r["area"].lower(): r["code"] for r in load("phone_codes", resources_dir)
                              if r["type"] == "landline"}

    def make(self, k: int) -> Persona:
        rng = random.Random(f"{self.seed}:{k}")
        female = rng.random() < 0.5
        dob = _dob(rng)
        first = dict(rng.choice(self.first_by_gender[female]))
        last = dict(rng.choice(self.surnames))
        # people born before 2000 may hold an old-format NIC (or its converted new form)
        if dob.year < 2000:
            old = nic.make_old(dob, female, rng)
            new = nic.old_to_new(old) if rng.random() < 0.5 else nic.make_new(dob, female, rng)
        else:
            old, new = None, nic.make_new(dob, female, rng)
        place = dict(rng.choice(self.places))
        area = self.area_by_place.get(place["latin"].lower())
        house = str(rng.randint(1, 450)) + (f"/{rng.randint(1, 12)}" if rng.random() < 0.35 else "")
        fl = first["latin"].lower()
        ll = last["latin"].lower()
        local = rng.choice([f"{fl}.{ll}", f"{fl}{ll[:1]}", f"{fl}{rng.randint(1, 99)}", f"{fl[:1]}{ll}"])
        return Persona(
            id=f"per-{self.seed}-{k:06d}", female=female, dob=dob,
            first={s: first[s] for s in ("latin", "sinhala", "tamil")},
            last={s: last[s] for s in ("latin", "sinhala", "tamil")},
            nic_old=old, nic_new=new,
            place={s: place.get(s, "") for s in ("latin", "sinhala", "tamil", "postcode", "district")},
            street={s: v for s, v in rng.choice(self.streets).items() if s in ("latin", "sinhala", "tamil")},
            street_word=rng.choice(["road", "road", "mawatha", "lane", "street"]),
            house=house, mobile=phone.mobile(rng), landline=phone.landline(rng, area),
            email=f"{local}@{rng.choice(EMAIL_DOMAINS)}",
            username=rng.choice([f"{fl}{rng.randint(10, 9999)}", f"{fl}_{ll}", f"{ll}.{fl[:1]}{rng.randint(1, 99)}"]),
            password=_password(rng), ipv4=_ipv4(rng), ipv6=_ipv6(rng),
            meta={"area_code_matches_city": area is not None})

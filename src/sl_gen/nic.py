"""Sri Lankan NIC numbers (synthetic, format-valid).

Format (Department for Registration of Persons):
  old  YY DDD SSS C L   9 digits + letter (V, or X for non-voters)
  new  YYYY DDD SSSS C  12 digits; an old NIC converts as 19YY DDD 0SSS C
  DDD  day of the year of birth, counted as if every year had 29 February
       (1 March is always day 61); women add 500.
  C    check digit. The official algorithm is not published, so it is random here:
       these numbers are synthetic and must not be presented as real ones.
"""

from __future__ import annotations

import random
import re
from datetime import date

# cumulative days before each month in a 366-day year (Feb always has 29 days)
_CUM = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
OLD_RE = re.compile(r"^(\d{2})(\d{3})(\d{3})(\d)([VvXx])$")
NEW_RE = re.compile(r"^(\d{4})(\d{3})(\d{4})(\d)$")


def day_code(dob: date, female: bool) -> int:
    d = _CUM[dob.month - 1] + dob.day
    return d + 500 if female else d


def decode_day(ddd: int) -> tuple[int, int, bool]:
    """DDD -> (month, day, female). Raises ValueError for impossible codes."""
    female = ddd > 500
    d = ddd - 500 if female else ddd
    if not 1 <= d <= 366:
        raise ValueError(f"day code {ddd} out of range")
    month = max(i for i, c in enumerate(_CUM) if c < d) + 1
    return month, d - _CUM[month - 1], female


def make_old(dob: date, female: bool, rng: random.Random, letter: str | None = None) -> str:
    if not 1900 <= dob.year <= 1999:
        raise ValueError("old-format NICs exist only for people born 1900-1999")
    letter = letter or ("V" if rng.random() < 0.9 else "X")
    return f"{dob.year % 100:02d}{day_code(dob, female):03d}{rng.randint(0, 999):03d}{rng.randint(0, 9)}{letter}"


def old_to_new(old: str) -> str:
    m = OLD_RE.match(old)
    if not m:
        raise ValueError(f"not an old NIC: {old}")
    yy, ddd, sss, c, _ = m.groups()
    return f"19{yy}{ddd}0{sss}{c}"


def make_new(dob: date, female: bool, rng: random.Random) -> str:
    return f"{dob.year}{day_code(dob, female):03d}{rng.randint(0, 9999):04d}{rng.randint(0, 9)}"


def decode(nic: str) -> dict:
    """NIC (either format, spaces allowed) -> {format, year, month, day, female}."""
    s = re.sub(r"\s+", "", nic)
    m = OLD_RE.match(s)
    if m:
        year, ddd, fmt = 1900 + int(m.group(1)), int(m.group(2)), "old"
    else:
        m = NEW_RE.match(s)
        if not m:
            raise ValueError(f"not a Sri Lankan NIC: {nic!r}")
        year, ddd, fmt = int(m.group(1)), int(m.group(2)), "new"
    month, day, female = decode_day(ddd)
    return {"format": fmt, "year": year, "month": month, "day": day, "female": female}


def consistent(nic: str, dob: date | None = None, female: bool | None = None) -> bool:
    """True if the NIC encodes this date of birth and sex (Feb 29 is valid in any year)."""
    try:
        d = decode(nic)
    except ValueError:
        return False
    if dob and (d["year"], d["month"], d["day"]) != (dob.year, dob.month, dob.day):
        return False
    if female is not None and d["female"] != female:
        return False
    return True

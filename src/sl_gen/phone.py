"""Sri Lankan phone numbers (synthetic) following the TRCSL national numbering plan.

National number = 9 digits after the trunk prefix 0 (country code +94):
  mobile    7X XXX XXXX     X in {0,1,2,4,5,6,7,8}
  landline  AA X XXX XXX    AA = geographic area code (resources/sl/phone_codes.csv)
"""

from __future__ import annotations

import random

MOBILE_PREFIXES = ["70", "71", "72", "74", "75", "76", "77", "78"]

# Area codes from the TRCSL numbering plan (source_ref kept in resources/sl/phone_codes.csv).
AREA_CODES = {
    "11": "Colombo", "21": "Jaffna", "23": "Mannar", "24": "Vavuniya", "25": "Anuradhapura",
    "26": "Trincomalee", "27": "Polonnaruwa", "31": "Negombo", "32": "Chilaw", "33": "Gampaha",
    "34": "Kalutara", "35": "Kegalle", "36": "Avissawella", "37": "Kurunegala", "38": "Panadura",
    "41": "Matara", "45": "Ratnapura", "47": "Hambantota", "51": "Hatton", "52": "Nuwara Eliya",
    "54": "Nawalapitiya", "55": "Badulla", "57": "Bandarawela", "63": "Ampara",
    "65": "Batticaloa", "66": "Matale", "67": "Kalmunai", "81": "Kandy", "91": "Galle",
}


def mobile(rng: random.Random) -> str:
    """National significant number, 9 digits, e.g. '771234567'."""
    return rng.choice(MOBILE_PREFIXES) + f"{rng.randint(0, 9999999):07d}"


def landline(rng: random.Random, area: str | None = None) -> str:
    area = area or rng.choice(sorted(AREA_CODES))
    return area + f"{rng.randint(2000000, 9999999):07d}"


def is_mobile(nsn: str) -> bool:
    return len(nsn) == 9 and nsn[:2] in MOBILE_PREFIXES


# (format tag, function nsn -> text). Mobile groups 2-3-4, landline 2-3-4 as well (0AA XXX XXXX).
FORMATS = {
    "local":        lambda n: f"0{n}",
    "local_spaced": lambda n: f"0{n[:2]} {n[2:5]} {n[5:]}",
    "local_dash":   lambda n: f"0{n[:2]}-{n[2:]}",
    "intl":         lambda n: f"+94{n}",
    "intl_spaced":  lambda n: f"+94 {n[:2]} {n[2:5]} {n[5:]}",
    "intl_00":      lambda n: f"0094{n}",
}
FORMAT_WEIGHTS = {"local": 3, "local_spaced": 3, "local_dash": 1, "intl": 2, "intl_spaced": 2, "intl_00": 1}


def render(nsn: str, rng: random.Random, fmt: str | None = None) -> tuple[str, str]:
    fmt = fmt or rng.choices(list(FORMAT_WEIGHTS), weights=list(FORMAT_WEIGHTS.values()))[0]
    return FORMATS[fmt](nsn), fmt

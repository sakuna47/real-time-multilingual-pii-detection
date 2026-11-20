"""Placeholder tokens: {PERSON_1}, {PHONE_2}, ... one name per v1 label."""

from __future__ import annotations

import re
from collections import Counter

from ..pii_data.record import LABEL_SETS

LABELS = tuple(LABEL_SETS["v1"])
# longest labels first so ID_NUMBER / IP_ADDRESS match before shorter prefixes
_TOKEN = re.compile(r"\{(" + "|".join(sorted(LABELS, key=len, reverse=True)) + r")_(\d{1,2})\}")
_ANY_BRACED = re.compile(r"\{[^{}\s]{1,40}\}")


def token(label: str, n: int) -> str:
    return f"{{{label}_{n}}}"


def find(text: str) -> list[tuple[str, str, int, int]]:
    """[(token, label, start, end)] in order of appearance."""
    return [(m.group(0), m.group(1), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def counts(text: str) -> Counter:
    return Counter(t for t, *_ in find(text))


def stray(text: str) -> list[str]:
    """Brace tokens that are not valid placeholders (e.g. a translated {පුද්ගලයා_1})."""
    valid = {t for t, *_ in find(text)}
    return [m.group(0) for m in _ANY_BRACED.finditer(text) if m.group(0) not in valid]


def preserved(source: str, target: str) -> tuple[bool, str]:
    """Same set of placeholders in both texts, each present at least once, nothing invented.
    (A placeholder may legitimately appear a different number of times after translation.)"""
    a, b = set(counts(source)), set(counts(target))
    if a != b:
        return False, f"missing {sorted(a - b)} extra {sorted(b - a)}"
    bad = stray(target)
    if bad:
        return False, f"invalid tokens {bad[:3]}"
    return True, ""

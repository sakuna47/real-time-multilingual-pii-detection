"""The six target language varieties and how values are written in each."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variety:
    code: str                 # "si", "si-en", "si-rom", ...
    lang: str                 # primary language
    native_script: str        # ISO 15924 of the language
    romanized: bool
    code_mixed: bool
    # probability that a name / place / address is written in the native script (else Latin)
    p_native_names: float

    @property
    def column(self) -> str:   # resource CSV column for the native script
        return {"Sinh": "sinhala", "Taml": "tamil"}[self.native_script]

    def language_block(self, has_latin: bool, has_native: bool) -> dict:
        scripts = ([self.native_script] if has_native else []) + (["Latn"] if has_latin else [])
        mixed = self.code_mixed or self.romanized or (has_latin and has_native)
        return {"primary": self.lang, "langs": [self.lang, "en"] if mixed else [self.lang],
                "scripts": scripts or ["Latn"], "code_mixed": mixed, "romanized": self.romanized}


VARIETIES = {
    "si":     Variety("si", "si", "Sinh", romanized=False, code_mixed=False, p_native_names=0.9),
    "ta":     Variety("ta", "ta", "Taml", romanized=False, code_mixed=False, p_native_names=0.9),
    "si-en":  Variety("si-en", "si", "Sinh", romanized=False, code_mixed=True, p_native_names=0.6),
    "ta-en":  Variety("ta-en", "ta", "Taml", romanized=False, code_mixed=True, p_native_names=0.6),
    "si-rom": Variety("si-rom", "si", "Sinh", romanized=True, code_mixed=True, p_native_names=0.0),
    "ta-rom": Variety("ta-rom", "ta", "Taml", romanized=True, code_mixed=True, p_native_names=0.0),
}


def get(code: str) -> Variety:
    if code not in VARIETIES:
        raise ValueError(f"unknown variety {code!r}; use one of {sorted(VARIETIES)}")
    return VARIETIES[code]

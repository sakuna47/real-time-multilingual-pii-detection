"""Offline tests for NIC, phones, personas and value rendering (fixture resources).
Run with `python tests/test_sl_gen.py` or pytest."""

import os
import random
import re
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PII_DATA_ROOT"] = tempfile.mkdtemp(prefix="pii_test_")
RES = Path(tempfile.mkdtemp(prefix="pii_res_"))
os.environ["PII_RESOURCES_DIR"] = str(RES)

from src.sl_gen import nic, phone, resources, values, varieties  # noqa: E402
from src.sl_gen.personas import PersonaFactory  # noqa: E402


def _fixture_resources():
    cur = {"source": "test", "source_ref": "", "needs_review": "no", "people": 1}
    resources.write_csv("first_names", [
        {"latin": "Kasun", "sinhala": "කසුන්", "tamil": "கசுன்", "gender": "m", **cur},
        {"latin": "Nimali", "sinhala": "නිමාලි", "tamil": "நிமாலி", "gender": "f", **cur},
        {"latin": "Ravi", "sinhala": "", "tamil": "ரவி", "gender": "m", **cur}],          # incomplete: unused
        ["latin", "sinhala", "tamil", "gender", "people", "source", "source_ref", "needs_review"])
    resources.write_csv("surnames", [{"latin": "Perera", "sinhala": "පෙරේරා", "tamil": "பெரேரா", **cur}],
                        ["latin", "sinhala", "tamil", "people", "source", "source_ref", "needs_review"])
    resources.write_csv("places", [{"latin": "Kandy", "sinhala": "මහනුවර", "tamil": "கண்டி", "postcode": "20000", **cur}],
                        ["latin", "sinhala", "tamil", "postcode", "source", "source_ref", "needs_review"])
    resources.build(wikidata=False)   # curated tables: phone codes, streets, address words, months


_fixture_resources()


def test_nic_day_code_uses_leap_calendar():
    assert nic.day_code(date(1995, 3, 1), female=False) == 61      # always 61, even in non-leap years
    assert nic.day_code(date(1996, 3, 1), female=True) == 561
    assert nic.decode_day(60) == (2, 29, False)
    assert nic.decode_day(866) == (12, 31, True)


def test_nic_roundtrip_both_formats():
    rng = random.Random(0)
    start = date(1940, 1, 1)
    for i in range(10000):
        dob = start + timedelta(days=rng.randint(0, 365 * 66))
        female = rng.random() < 0.5
        new = nic.make_new(dob, female, rng)
        assert len(new) == 12 and nic.consistent(new, dob, female)
        if dob.year < 2000:
            old = nic.make_old(dob, female, rng)
            assert re.fullmatch(r"\d{9}[VX]", old) and nic.consistent(old, dob, female)
            conv = nic.old_to_new(old)
            assert nic.consistent(conv, dob, female) and conv[7] == "0"
    assert not nic.consistent("951234567V", date(1995, 1, 1))
    assert not nic.consistent("not a nic")
    assert nic.decode("1995 0610 1234")["month"] == 3


def test_phone_formats():
    rng = random.Random(1)
    n = phone.mobile(rng)
    assert phone.is_mobile(n) and len(n) == 9
    assert phone.FORMATS["intl_spaced"](n).startswith("+94 7")
    assert phone.FORMATS["local"](n) == "0" + n
    assert phone.landline(rng, "81").startswith("81")


def test_personas_are_deterministic_and_consistent():
    f = PersonaFactory(seed=7)
    a, b = f.make(3), PersonaFactory(seed=7).make(3)
    assert a.to_dict() == b.to_dict()
    for k in range(500):
        p = f.make(k)
        assert nic.consistent(p.nic_new, p.dob, p.female)
        if p.nic_old:
            assert nic.consistent(p.nic_old, p.dob, p.female)
        assert p.first["latin"] in (("Nimali",) if p.female else ("Kasun",))   # Ravi is incomplete
        assert p.email.split("@")[1].endswith(("example.lk", "example.com", "example.net", "example.org"))
        if p.place["latin"] == "Kandy":
            assert p.landline.startswith("81")


def test_render_scripts_follow_variety():
    p = PersonaFactory(seed=1).make(0)
    rng = random.Random(0)
    for _ in range(50):
        text, fmt = values.render("PERSON", None, p, varieties.get("si-rom"), rng)
        assert re.fullmatch(r"[A-Za-z. ]+", text) and fmt.startswith("name:latin")
    sinh = [values.render("ADDRESS", None, p, varieties.get("si"), rng)[0] for _ in range(50)]
    assert sum(bool(re.search(r"[඀-෿]", t)) for t in sinh) > 30          # mostly Sinhala script
    assert any(t.startswith("අංක ") for t in sinh)
    ta = values.render("LOCATION", None, p, varieties.get("ta"), random.Random(5))[0]
    assert ta in ("கண்டி", "Kandy")


def test_render_all_labels():
    p = PersonaFactory(seed=2).make(1)
    rng = random.Random(3)
    v = varieties.get("ta-en")
    for label, sub in [("PERSON", None), ("ID_NUMBER", "NIC_NEW"), ("ID_NUMBER", "PASSPORT"), ("PHONE", "MOBILE"),
                       ("PHONE", "LANDLINE"), ("EMAIL", None), ("ADDRESS", None), ("LOCATION", None),
                       ("DOB", None), ("USERNAME", None), ("PASSWORD", None), ("IP_ADDRESS", None)]:
        text, fmt = values.render(label, sub, p, v, rng)
        assert text and text == text.strip(), (label, text)
        assert ":" in fmt or fmt in ("email", "username", "password"), fmt
    nic_text, _ = values.render("ID_NUMBER", "NIC_NEW", p, v, rng)
    assert nic.consistent(nic_text, p.dob, p.female)


def test_wikidata_rows_to_names():
    rows = [
        {"n": "http://www.wikidata.org/entity/Q1", "en": "Kasun", "si": "කසුන්", "ta": "கசுன்",
         "g": "http://www.wikidata.org/entity/Q6581097", "people": "5"},
        {"n": "http://www.wikidata.org/entity/Q2", "en": "K.", "people": "3"},                  # initial: skipped
        {"n": "http://www.wikidata.org/entity/Q3", "en": "Amaya", "si": "Amaya",                 # Latin "si": dropped
         "g": "http://www.wikidata.org/entity/Q6581072", "people": "2"},
    ]
    out = resources.names_from_wikidata(rows, with_gender=True)
    assert [(r["latin"], r["sinhala"], r["gender"], r["needs_review"]) for r in out] == [
        ("Kasun", "කසුන්", "m", "no"), ("Amaya", "", "f", "yes")]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:
                failed += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if failed else 0)

"""Offline tests for the placeholder-translation engine (fixture resources, mock LLM).
Run with `python tests/test_translate.py` or pytest."""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PII_DATA_ROOT"] = tempfile.mkdtemp(prefix="pii_test_")
os.environ["PII_RESOURCES_DIR"] = tempfile.mkdtemp(prefix="pii_res_")

from src.llm import LLM  # noqa: E402
from src.llm.mock import MockClient  # noqa: E402
from src.pii_data.record import validate_record  # noqa: E402
from src.sl_gen import nic, resources, varieties  # noqa: E402
from src.sl_gen.personas import PersonaFactory  # noqa: E402
from src.translate import placeholders as ph  # noqa: E402
from src.translate import steps  # noqa: E402
from src.translate.fill import build_record, fill  # noqa: E402
from src.translate.mask import mask  # noqa: E402
from src.translate.pipeline import assign_variety, parse_mix  # noqa: E402

cur = {"source": "fixture", "source_ref": "", "needs_review": "no", "people": 1}
resources.write_csv("first_names", [
    {"latin": "Kasun", "sinhala": "කසුන්", "tamil": "கசுன்", "gender": "m", **cur},
    {"latin": "Nimali", "sinhala": "නිමාලි", "tamil": "நிமாலி", "gender": "f", **cur}],
    ["latin", "sinhala", "tamil", "gender", "people", "source", "source_ref", "needs_review"])
resources.write_csv("surnames", [{"latin": "Perera", "sinhala": "පෙරේරා", "tamil": "பெரேரா", **cur}],
                    ["latin", "sinhala", "tamil", "people", "source", "source_ref", "needs_review"])
resources.write_csv("places", [{"latin": "Kandy", "sinhala": "මහනුවර", "tamil": "கண்டி", "postcode": "20000", **cur}],
                    ["latin", "sinhala", "tamil", "postcode", "source", "source_ref", "needs_review"])
resources.build(wikidata=False)
FACTORY = PersonaFactory(seed=1)

SRC = {
    "id": "nemotron:en:abc", "domain": "form", "language": {"primary": "en"},
    "origin": {"dataset": "nvidia/Nemotron-PII", "source_id": "abc", "license": "CC BY 4.0"},
    "text": "Jason Lee (born 1990-01-02) can be reached at 555-0101. Jason lives in Austin. Call 555-0101.",
    "spans": [
        {"start": 0, "end": 9, "label": "PERSON", "subtype": None, "value": "Jason Lee"},
        {"start": 16, "end": 26, "label": "DOB", "subtype": None, "value": "1990-01-02"},
        {"start": 46, "end": 54, "label": "PHONE", "subtype": None, "value": "555-0101"},
        {"start": 56, "end": 61, "label": "PERSON", "subtype": None, "value": "Jason"},
        {"start": 71, "end": 77, "label": "LOCATION", "subtype": None, "value": "Austin"},
        {"start": 84, "end": 92, "label": "PHONE", "subtype": None, "value": "555-0101"},
    ],
}


def test_placeholder_helpers():
    assert ph.find("a {PERSON_1}ට {ID_NUMBER_2}")[1][:2] == ("{ID_NUMBER_2}", "ID_NUMBER")
    assert ph.preserved("{PERSON_1} {PHONE_1}", "{PHONE_1} x {PERSON_1} {PERSON_1}") == (True, "")
    ok, why = ph.preserved("{PERSON_1} {PHONE_1}", "{PERSON_1}")
    assert not ok and "PHONE_1" in why
    assert not ph.preserved("{PERSON_1}", "{PERSON_1} {නම_1}")[0]          # translated token
    assert not ph.preserved("{PERSON_1}", "{PERSON_1} {EMAIL_1}")[0]        # invented placeholder


def test_mask_reuses_tokens_and_links_first_names():
    m = mask(SRC)
    assert m.text == ("{PERSON_1} (born {DOB_1}) can be reached at {PHONE_1}. {PERSON_2} lives in "
                      "{LOCATION_1}. Call {PHONE_1}.")
    assert m.slots["{PERSON_2}"].part == "first" and m.slots["{PERSON_2}"].entity == m.slots["{PERSON_1}"].entity
    assert m.slots["{PHONE_1}"].entity == "primary"


def test_fill_spans_exact_and_consistent():
    m = mask(SRC)
    targets = {"si": "{PERSON_1} ({DOB_1} උපන්) {PHONE_1} අමතන්න. {PERSON_2}ට {LOCATION_1} නිවසක් ඇත. {PHONE_1}",
               "ta": "{PERSON_1} ({DOB_1} பிறந்தவர்) {PHONE_1} அழைக்கவும். {PERSON_2}க்கு {LOCATION_1} வீடு உண்டு. {PHONE_1}"}
    slots = {k: s.to_dict() for k, s in m.slots.items()}
    for code in varieties.VARIETIES:
        v = varieties.get(code)
        target = targets[v.lang]
        if v.romanized:
            target = "{PERSON_1} ({DOB_1} upan) {PHONE_1} call karanna. {PERSON_2}ta {LOCATION_1} gedarak. {PHONE_1}"
        filled, err = fill(target, slots, v, FACTORY, seed_key=f"t:{code}")
        assert err is None, err
        for sp in filled["spans"]:
            assert filled["text"][sp["start"]:sp["end"]] == sp["value"]
        phones = [s["value"] for s in filled["spans"] if s["label"] == "PHONE"]
        assert len(phones) == 2 and phones[0] == phones[1]                 # same placeholder, same text
        names = [s["value"] for s in filled["spans"] if s["label"] == "PERSON"]
        # same person, same script: the first-name mention matches the full name's first token or initial
        assert names[0].startswith(names[1]) or (names[0][1] == "." and names[0][0] == names[1][0]), names
        rec = build_record(filled=filled, source=SRC, variety=v, method="translated_placeholder",
                           masked_target=target, models=[], extra_tags=[], seed=1, pipeline_version="t")
        assert validate_record(rec) == [], validate_record(rec)
        assert rec["id"] == f"nemotron-tr:{v.lang}:abc-{code}" and rec["group"] == "nemotron:en:abc"


def test_fill_nic_matches_dob():
    slots = {"{PERSON_1}": {"label": "PERSON", "subtype": None, "entity": "p1", "part": "full"},
             "{ID_NUMBER_1}": {"label": "ID_NUMBER", "subtype": None, "entity": "primary", "part": "full"},
             "{DOB_1}": {"label": "DOB", "subtype": None, "entity": "primary", "part": "full"}}
    for k in range(30):
        filled, err = fill("{PERSON_1} NIC {ID_NUMBER_1} DOB {DOB_1}", slots, varieties.get("si-rom"),
                           FACTORY, seed_key=f"nic:{k}")
        assert err is None and "nic_dob_consistent" in filled["tags"]
        nic_text = next(s["value"] for s in filled["spans"] if s["label"] == "ID_NUMBER")
        assert nic.decode(nic_text)


def test_form_first_and_last_name_fields_are_one_person():
    text = "First name: Jason\nLast name: Lee\nCity: Austin"
    rec = {"text": text, "spans": [
        {"start": 12, "end": 17, "label": "PERSON", "value": "Jason", "source_labels": ["first_name"]},
        {"start": 29, "end": 32, "label": "PERSON", "value": "Lee", "source_labels": ["last_name"]},
        {"start": 39, "end": 45, "label": "LOCATION", "value": "Austin", "source_labels": ["city"]}]}
    m = mask(rec)
    a, b = m.slots["{PERSON_1}"], m.slots["{PERSON_2}"]
    assert a.entity == b.entity and (a.part, b.part) == ("first", "last")
    slots = {k: s.to_dict() for k, s in m.slots.items()}
    filled, err = fill("නම: {PERSON_1}\nවාසගම: {PERSON_2}\nනගරය: {LOCATION_1}", slots,
                       varieties.get("si"), FACTORY, seed_key="form")
    assert err is None
    first, last = [s["value"] for s in filled["spans"] if s["label"] == "PERSON"]
    assert " " not in first and " " not in last and len(filled["persona_ids"]) == 1


def test_address_omits_city_when_location_present():
    slots = {"{ADDRESS_1}": {"label": "ADDRESS", "subtype": None, "entity": "primary"},
             "{LOCATION_1}": {"label": "LOCATION", "subtype": None, "entity": "primary"}}
    for k in range(10):
        filled, err = fill("Address {ADDRESS_1}, {LOCATION_1}", slots, varieties.get("si-rom"),
                           FACTORY, seed_key=f"addr:{k}")
        addr, city = [s["value"] for s in filled["spans"]]
        assert city not in addr and "_nocity" in filled["formats"][0]


def test_ssn_slot_becomes_nic_subtype():
    slots = {"{ID_NUMBER_1}": {"label": "ID_NUMBER", "subtype": "SOCIAL_SECURITY", "entity": "primary"},
             "{ID_NUMBER_2}": {"label": "ID_NUMBER", "subtype": "PASSPORT", "entity": "primary"}}
    for k in range(20):
        filled, err = fill("NIC {ID_NUMBER_1}, passport {ID_NUMBER_2}", slots, varieties.get("si-rom"),
                           FACTORY, seed_key=f"ssn:{k}")
        assert err is None
        nic_span, pp_span = filled["spans"]
        assert nic_span["subtype"] in ("NIC_OLD", "NIC_NEW") and nic.decode(nic_span["value"])
        assert pp_span["subtype"] == "PASSPORT"


def test_fill_rejects_unknown_or_stray_tokens():
    assert fill("{PERSON_1} {EMAIL_1}", {"{PERSON_1}": {"label": "PERSON", "subtype": None, "entity": "p1"}},
                varieties.get("si"), FACTORY, seed_key="x")[1].startswith("fill: unknown")
    assert "invalid tokens" in fill("{PERSON_1} {නම}", {"{PERSON_1}": {"label": "PERSON", "subtype": None,
                                                                      "entity": "p1"}},
                                    varieties.get("si"), FACTORY, seed_key="x")[1]


def test_check_insert():
    ok, err = steps.check_insert("Write a letter to my landlord.",
                                 {"suitable": True, "text": "Write a letter to my landlord {PERSON_1}. My phone is {PHONE_1}.",
                                  "placeholders": {"PERSON_1": {"subtype": None}, "PHONE_1": {"subtype": "MOBILE"}}})
    assert err is None and ok["slots"]["{PHONE_1}"]["subtype"] == "MOBILE" and ok["slots"]["{PERSON_1}"]["entity"] == "p1"
    _, err = steps.check_insert("Write a letter.", {"suitable": True, "text": "Totally different text {PERSON_1} here now",
                                                    "placeholders": {"PERSON_1": {}}})
    assert "rewrote too much" in err
    _, err = steps.check_insert("x", {"suitable": True, "text": "x {PERSON_1}", "placeholders": {"EMAIL_1": {}}})
    assert "declared" in err
    assert steps.check_insert("x", {"suitable": False}) == ({"suitable": False}, None)


def test_codemix_checks_script():
    rom = steps.make_check_codemix(varieties.get("si-rom"))
    assert rom("{PERSON_1} ට", "mata {PERSON_1}ta call karanna")[1] is None
    assert "native script" in rom("{PERSON_1}", "{PERSON_1} මට")[1]
    mixed = steps.make_check_codemix(varieties.get("si-en"))
    assert "both scripts" in mixed("{PERSON_1} මට", "{PERSON_1} මට ඕනේ")[1]
    assert mixed("{PERSON_1}", "{PERSON_1}ට loan එකක් ඕනේ")[1] is None


def test_translate_step_retries_bad_batch_singly():
    calls = []
    def handler(req):
        items = json.loads(req.messages[-1].content.split("TEXTS:\n", 1)[1])
        calls.append(len(items))
        if len(items) > 1:                       # batch reply drops a placeholder -> retried one by one
            return json.dumps({"translations": ["මම" for _ in items]})
        return json.dumps({"translations": [items[0].replace("Hello", "ආයුබෝවන්")]}, ensure_ascii=False)
    llm = LLM(MockClient(handler=handler), "m", sleep=lambda s: None)
    res = steps.translate(llm, ["Hello {PERSON_1}", "Hello {PHONE_1}"], "si", batch_size=2, workers=1)
    assert [r[0] for r in res] == ["ආයුබෝවන් {PERSON_1}", "ආයුබෝවන් {PHONE_1}"] and calls == [2, 1, 1]


def test_variety_assignment_is_stable_and_follows_mix():
    mix = parse_mix("native:0.5,en:0.25,rom:0.25")
    recs = [{"id": f"x:en:{i}", "language": {"primary": "en"}} for i in range(2000)]
    codes = [assign_variety(r, ["si", "ta"], mix, 42).code for r in recs]
    assert codes == [assign_variety(r, ["si", "ta"], mix, 42).code for r in recs]
    native = sum(c in ("si", "ta") for c in codes) / len(codes)
    assert 0.45 < native < 0.55
    assert assign_variety({"id": "aya:ta:1", "language": {"primary": "ta"}}, ["si"], mix, 1).lang == "ta"


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

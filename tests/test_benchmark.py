"""Offline tests for FLORES alignment, the translator benchmark and Colab cache sync.
Run with `python tests/test_benchmark.py` (needs sacrebleu) or pytest."""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PII_DATA_ROOT"] = tempfile.mkdtemp(prefix="pii_test_")

from src.llm import LLM  # noqa: E402
from src.llm.mock import MockClient  # noqa: E402
from src.sources import flores  # noqa: E402
from src.translate import benchmark as bm  # noqa: E402

ROWS = [
    {"id": "1", "en": "The bank opens at nine.", "si": "බැංකුව නවයට විවෘත වේ.", "ta": "வங்கி ஒன்பது மணிக்கு திறக்கும்."},
    {"id": "2", "en": "Rain is expected tomorrow.", "si": "හෙට වැසි අපේක්ෂා කෙරේ.", "ta": "நாளை மழை எதிர்பார்க்கப்படுகிறது."},
    {"id": "3", "en": "She works in Kandy.", "si": "ඇය මහනුවර වැඩ කරයි.", "ta": "அவள் கண்டியில் வேலை செய்கிறாள்."},
]


def _sentences(req):
    return json.loads(req.messages[-1].content.split("SENTENCES:", 1)[1])


def _perfect(lang):
    """Mock translator that returns the FLORES reference (and keeps probe placeholders)."""
    lookup = {r["en"]: r[lang] for r in ROWS}
    def handler(req):
        out = []
        for s in _sentences(req):
            if s.endswith(bm.PROBE_SUFFIX):
                out.append(lookup[s[:-len(bm.PROBE_SUFFIX)]] + " {PERSON_1} {EMAIL_1}")
            else:
                out.append(lookup[s])
        return json.dumps({"translations": out}, ensure_ascii=False)
    return handler


def _llm(handler):
    return LLM(MockClient(handler=handler), "m", sleep=lambda s: None)


def test_align_joins_on_id_and_drops_missing():
    rows = flores.align({
        "en": [{"id": 2, "text": "b"}, {"id": 1, "text": "a"}, {"id": 3, "text": "c"}],
        "si": [{"id": 1, "text": "ා"}, {"id": 2, "text": "ි"}],
        "ta": [{"id": 1, "text": "அ"}, {"id": 2, "text": "இ "}],
    })
    assert rows == [{"id": "1", "en": "a", "si": "ා", "ta": "அ"},
                    {"id": "2", "en": "b", "si": "ි", "ta": "இ"}]


def test_placeholder_check():
    assert bm.placeholder_ok("x {PERSON_1} y {EMAIL_1}")
    assert not bm.placeholder_ok("x {PERSON_1} y")                      # lost one
    assert not bm.placeholder_ok("{PERSON_1} {PERSON_1} {EMAIL_1}")      # duplicated
    assert not bm.placeholder_ok("{PERSON_1} {EMAIL_1} {පුද්ගලයා_1}")     # invented/translated token
    assert not bm.placeholder_ok("{ PERSON_1 } {EMAIL_1}")               # altered


def test_perfect_translator_scores_100():
    for lang in ("si", "ta"):
        score, outputs = bm.evaluate(_llm(_perfect(lang)), ROWS, ROWS[:2], lang, batch_size=2, workers=2)
        assert score.chrf == 100.0 and score.coverage == 1.0 and score.placeholder_ok == 1.0, score
        assert score.calls == 3 and score.failed_batches == 0
        assert len(outputs) == 5


def test_wrong_count_fails_batch_not_run():
    good = _perfect("si")
    def handler(req):
        sents = _sentences(req)
        if any(s.startswith("She") for s in sents):
            return json.dumps({"translations": ["only one"] * (len(sents) + 1)})
        return good(req)
    score, outputs = bm.evaluate(_llm(handler), ROWS, [], "si", batch_size=1, workers=1)
    assert score.failed_batches == 1 and abs(score.coverage - 2 / 3) < 1e-3
    assert score.chrf < 100 and score.chrf_ok_only == 100.0
    assert outputs[2]["hyp"] is None and "expected 1" in score.errors[0]


def test_markdown_table_lists_models():
    s = bm.Score(model="mock:m", lang="si", n=3, chrf=55.5, coverage=1.0, placeholder_ok=0.9)
    assert "| mock:m | si | 3 | 55.5 |" in bm.markdown_table([s])


def test_colab_cache_backup_roundtrip():
    from src.common import colab
    from src.llm.base import LLMRequest, Message
    from src.llm.cache import ResponseCache, cache_key
    from src.llm.base import LLMResponse
    local = Path(tempfile.mkdtemp(prefix="local_cache_"))
    os.environ["PII_CACHE_DIR"] = str(local)
    try:
        c = ResponseCache(local / "llm_cache.sqlite")
        req = LLMRequest(messages=[Message("user", "x")])
        key = cache_key("mock", "m", req, {})
        c.put(key, LLMResponse(text="y", provider="mock", model="m"))
        assert colab.save_cache() == ["llm_cache.sqlite"]          # cache stays open while copied
        c.close()
        (local / "llm_cache.sqlite").unlink()
        assert colab.restore_cache() == ["llm_cache.sqlite"]
        assert ResponseCache(local / "llm_cache.sqlite").get(key).text == "y"
    finally:
        del os.environ["PII_CACHE_DIR"]


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

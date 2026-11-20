"""Offline tests for the LLM layer (mock provider, no network, no keys).
Run with `python tests/test_llm.py` or pytest."""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PII_DATA_ROOT"] = tempfile.mkdtemp(prefix="pii_test_")  # keep cache/logs out of the repo

from src.llm import LLM, LLMError, LLMRequest, Message, get_llm, parse_json, run_batch  # noqa: E402
from src.llm.cache import ResponseCache  # noqa: E402
from src.llm.calllog import CallLog, summarize  # noqa: E402
from src.llm.config import ConfigError, load_config  # noqa: E402
from src.llm.jsonout import JSONReplyError  # noqa: E402
from src.llm.mock import MockClient  # noqa: E402
from src.llm.prompts import PromptError, load_prompt, parse_prompt  # noqa: E402
from src.llm.ratelimit import RateLimiter  # noqa: E402

TMP = Path(os.environ["PII_DATA_ROOT"])


def _req(text="hello", **kw):
    return LLMRequest(messages=[Message("user", text)], **kw)


def _llm(mock=None, cache=True, **kw):
    cache_obj = ResponseCache(TMP / f"c_{os.urandom(4).hex()}.sqlite") if cache else None
    return LLM(mock or MockClient(), "m1", role="test", cache=cache_obj, sleep=lambda s: None, **kw)


# ------------------------------------------------------------ config

def test_config_loads_and_roles_resolve():
    cfg = load_config()
    for role in ("translator", "codemixer", "inserter", "judge"):
        assert ":" in cfg.role(role).model


def test_judge_must_differ_from_generator():
    p = TMP / "bad.toml"
    p.write_text('[providers.mock]\nrpm=0\n[roles.translator]\nmodel="mock:x"\n'
                 '[roles.judge]\nmodel="mock:x"\n', encoding="utf-8")
    try:
        load_config(p)
    except ConfigError as e:
        assert "judge" in str(e)
    else:
        raise AssertionError("expected ConfigError")


def test_role_override():
    cfg = load_config(overrides={"translator": "deepseek:deepseek-chat", "judge": "gemini:x"})
    assert cfg.role("translator").model == "deepseek:deepseek-chat"


# ------------------------------------------------------------ cache, retry

def test_cache_hit_skips_provider():
    mock = MockClient()
    llm = _llm(mock)
    a = llm.complete(_req("hi", temperature=0.1))
    b = llm.complete(_req("hi", temperature=0.1))
    assert len(mock.calls) == 1 and not a.cached and b.cached and b.text == "hi"


def test_cache_key_changes_with_params_and_salt():
    mock = MockClient()
    llm = _llm(mock)
    llm.complete(_req("hi", temperature=0.1))
    llm.complete(_req("hi", temperature=0.9))
    llm.complete(_req("hi", temperature=0.1, cache_salt="2"))
    assert len(mock.calls) == 3


def test_retry_on_429_then_success():
    mock = MockClient(fail_with=[LLMError("rate", status=429, retryable=True),
                                 LLMError("busy", status=503, retryable=True)])
    resp = _llm(mock, cache=False).complete(_req("x"))
    assert resp.text == "x" and len(mock.calls) == 3


def test_no_retry_on_401():
    mock = MockClient(fail_with=[LLMError("bad key", status=401, retryable=False)])
    try:
        _llm(mock, cache=False).complete(_req("x"))
    except LLMError as e:
        assert e.status == 401 and len(mock.calls) == 1
    else:
        raise AssertionError("expected LLMError")


def test_gives_up_after_max_attempts():
    mock = MockClient(fail_with=[LLMError("rate", status=429, retryable=True)] * 10)
    try:
        _llm(mock, cache=False, max_attempts=3).complete(_req("x"))
    except LLMError:
        assert len(mock.calls) == 3
    else:
        raise AssertionError("expected LLMError")


def test_role_defaults_and_request_override():
    seen = {}
    class Spy(MockClient):
        def complete(self, model, req, params):
            seen.update(params)
            return super().complete(model, req, params)
    llm = LLM(Spy(), "m", defaults={"temperature": 0.2, "max_tokens": 100})
    llm.complete(_req("a", max_tokens=50))
    assert seen == {"temperature": 0.2, "max_tokens": 50}


# ------------------------------------------------------------ JSON

def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert parse_json("[1, 2]") == [1, 2]
    try:
        parse_json("no json here")
    except JSONReplyError:
        pass
    else:
        raise AssertionError("expected JSONReplyError")


def test_complete_json_repairs_once():
    replies = iter(["not json at all", '{"ok": true}'])
    mock = MockClient(handler=lambda req: next(replies))
    data, _ = _llm(mock, cache=False).complete_json(_req("give json"))
    assert data == {"ok": True} and len(mock.calls) == 2
    assert mock.calls[1].messages[-1].content.startswith("That reply was not valid JSON")


# ------------------------------------------------------------ prompts

def test_prompt_keeps_single_brace_placeholders():
    p = parse_prompt("---\nid: t\njson: true\n---\n=== system ===\nS {{lang}}\n=== user ===\n"
                     "Translate: {{text}}\nKeep {PERSON_1} as is.")
    msgs = p.render(lang="si", text="Hi {PERSON_1}")
    assert msgs[0].content == "S si"
    assert msgs[1].content == "Translate: Hi {PERSON_1}\nKeep {PERSON_1} as is."
    assert p.request(lang="si", text="x").json_mode and p.request(lang="si", text="x").prompt_id == "t"


def test_prompt_missing_variable():
    p = parse_prompt("---\nid: t\n---\n=== user ===\n{{a}} {{b}}")
    try:
        p.render(a=1)
    except PromptError as e:
        assert "b" in str(e)
    else:
        raise AssertionError("expected PromptError")


def test_smoke_prompt_file_loads():
    p = load_prompt("smoke_v1")
    assert p.json and p.variables == {"word"}


# ------------------------------------------------------------ batch, limiter, log, get_llm

def test_batch_keeps_order_and_captures_errors():
    def handler(req):
        text = req.messages[-1].content
        if text == "boom":
            raise LLMError("bad request", status=400, retryable=False)
        return json.dumps({"t": text})
    reqs = [_req(t) for t in ["a", "b", "boom", "d"]]
    res = run_batch(_llm(MockClient(handler=handler)), reqs, workers=3, json=True, progress_every=0)
    assert [r.data["t"] if r.ok else None for r in res] == ["a", "b", None, "d"]
    assert isinstance(res[2].error, LLMError)


def test_rate_limiter_spaces_requests():
    t = [0.0]
    waits = []
    lim = RateLimiter(60, clock=lambda: t[0], sleep=lambda s: waits.append(s))
    for _ in range(3):
        lim.acquire()
    assert waits == [1.0, 2.0]  # 60 rpm -> one per second


def test_calllog_and_model_record():
    log = CallLog(TMP / "calls.jsonl", run_id="r1")
    llm = _llm(calllog=log, defaults={"temperature": 0.2})
    resp = llm.complete(_req("x", prompt_id="p1"))
    llm.complete(_req("x", prompt_id="p1"))
    s = summarize(TMP / "calls.jsonl")["mock:m1"]
    assert s["calls"] == 2 and s["cache_hits"] == 1
    rec = resp.as_model_record("translator")
    assert rec == {"role": "translator", "name": "mock:m1", "params": {"temperature": 0.2},
                   "prompt_id": "p1"}


def test_get_llm_dry_run_uses_mock():
    llm = get_llm("translator", dry_run=True, run_id="t")
    assert llm.provider == "mock" and llm.model.startswith("dryrun-")
    data, resp = llm.complete_json(load_prompt("smoke_v1").request(word="name"))
    assert "text" in data and resp.prompt_id == "smoke_v1"


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

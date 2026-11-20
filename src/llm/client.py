"""The LLM object pipelines use: provider adapter + rate limit + retry + cache + call log."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..common.paths import cache_dir, logs_dir
from .base import LLMError, LLMRequest, LLMResponse, Message, ProviderClient
from .cache import ResponseCache, cache_key
from .calllog import CallLog
from .config import LLMConfig, RoleSpec, load_config, split_spec
from .jsonout import JSONReplyError, parse_json
from .providers import build_provider
from .ratelimit import RateLimiter, limiter_for
from .retry import call_with_retry


class LLM:
    def __init__(self, provider_client: ProviderClient, model: str, *, role: str | None = None,
                 defaults: dict[str, Any] | None = None, extra: dict[str, Any] | None = None,
                 limiter: RateLimiter | None = None, cache: ResponseCache | None = None,
                 calllog: CallLog | None = None, max_attempts: int = 5,
                 sleep=time.sleep):
        self.client = provider_client
        self.provider = provider_client.provider
        self.model = model
        self.role = role
        self.defaults = dict(defaults or {})
        self.extra = dict(extra or {})
        self.limiter = limiter or RateLimiter(0)
        self.cache = cache
        self.calllog = calllog
        self.max_attempts = max_attempts
        self._sleep = sleep

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def _params(self, req: LLMRequest) -> dict[str, Any]:
        params = dict(self.defaults)
        for k in ("temperature", "max_tokens", "seed"):
            v = getattr(req, k)
            if v is not None:
                params[k] = v
        return {k: v for k, v in params.items() if v is not None}

    def complete(self, req: LLMRequest) -> LLMResponse:
        if self.extra:
            req.extra = {**self.extra, **req.extra}
        params = self._params(req)
        key = cache_key(self.provider, self.model, req, params)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit:
                self._log(hit, req, cached=True)
                return hit

        def attempt() -> LLMResponse:
            self.limiter.acquire()
            return self.client.complete(self.model, req, params)

        try:
            resp = call_with_retry(attempt, max_attempts=self.max_attempts, sleep=self._sleep)
        except LLMError as e:
            self._log(None, req, error=f"{e.status}: {e}")
            raise
        resp.prompt_id = req.prompt_id
        resp.params = params
        if self.cache is not None:
            self.cache.put(key, resp)
        self._log(resp, req, cached=False)
        return resp

    def complete_json(self, req: LLMRequest, max_repairs: int = 1) -> tuple[Any, LLMResponse]:
        """Call with json_mode and parse the reply. If it is not valid JSON, show the model
        its reply and the parse error and ask again (up to max_repairs times)."""
        req.json_mode = True
        resp = self.complete(req)
        for repair in range(max_repairs + 1):
            try:
                return parse_json(resp.text), resp
            except JSONReplyError as e:
                if repair == max_repairs:
                    raise
                fix = LLMRequest(
                    messages=req.messages + [
                        Message("assistant", resp.text),
                        Message("user", f"That reply was not valid JSON ({e}). "
                                        "Reply again with only the corrected JSON, nothing else.")],
                    temperature=req.temperature, max_tokens=req.max_tokens, json_mode=True,
                    seed=req.seed, prompt_id=req.prompt_id, extra=req.extra,
                    cache_salt=req.cache_salt)
                resp = self.complete(fix)
        raise AssertionError("unreachable")

    def model_record(self, prompt_id: str | None = None) -> dict:
        """generation.models[] entry for records made with this LLM."""
        return {"role": self.role, "name": self.name, "params": dict(self.defaults),
                "prompt_id": prompt_id}

    def _log(self, resp: LLMResponse | None, req: LLMRequest, cached: bool = False,
             error: str | None = None) -> None:
        if self.calllog is None:
            return
        self.calllog.write(role=self.role, model=self.name, prompt_id=req.prompt_id, cached=cached,
                           input_tokens=(resp.usage or {}).get("input_tokens") if resp else None,
                           output_tokens=(resp.usage or {}).get("output_tokens") if resp else None,
                           latency_s=resp.latency_s if resp and not cached else None,
                           finish_reason=resp.finish_reason if resp else None, error=error)


_shared_cache: dict[Path, ResponseCache] = {}


def get_llm(role_or_spec: str, *, config: LLMConfig | None = None, dry_run: bool = False,
            use_cache: bool = True, run_id: str | None = None,
            provider_client: ProviderClient | None = None) -> LLM:
    """Build an LLM for a role from configs/llm.toml ("translator", "judge", ...) or for an
    explicit "<provider>:<model>" spec. dry_run swaps in the offline mock provider."""
    cfg = config or load_config()
    if ":" in role_or_spec:
        role = RoleSpec(name="adhoc", model=role_or_spec)
    else:
        role = cfg.role(role_or_spec)
    provider, model = split_spec(role.model)
    if dry_run:
        provider, model = "mock", f"dryrun-{model}"
    pcfg = cfg.provider(provider)
    client = provider_client or build_provider(provider, cfg)

    cache = None
    if use_cache:
        path = cache_dir() / ("llm_cache_dryrun.sqlite" if dry_run else "llm_cache.sqlite")
        cache = _shared_cache.get(path)
        if cache is None:
            cache = _shared_cache[path] = ResponseCache(path)
    calllog = CallLog(logs_dir() / "llm_calls.jsonl", run_id=run_id)
    return LLM(client, model, role=role.name, defaults=role.params, extra=role.extra,
               limiter=limiter_for(provider, float(pcfg.get("rpm", 0))), cache=cache,
               calllog=calllog)

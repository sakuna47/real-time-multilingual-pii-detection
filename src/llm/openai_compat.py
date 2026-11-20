"""Adapter for OpenAI-compatible chat APIs: OpenRouter and DeepSeek."""

from __future__ import annotations

import time
from typing import Any

from .base import LLMError, LLMRequest, LLMResponse, is_retryable_status

# Generic app name for OpenRouter's optional attribution header. No personal details.
OPENROUTER_HEADERS = {"X-Title": "multilingual-pii-dataset-research"}


class OpenAICompatClient:
    def __init__(self, provider: str, api_key: str, base_url: str, timeout_s: float = 120,
                 json_response_format: bool = False, default_headers: dict | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(f"{provider} needs the openai package: pip install openai") from e
        self.provider = provider
        self.json_response_format = json_response_format
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s,
                              max_retries=0,  # retries are ours (src/llm/retry.py)
                              default_headers=default_headers)

    def complete(self, model: str, req: LLMRequest, params: dict[str, Any]) -> LLMResponse:
        messages = [m.to_dict() for m in req.messages]
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if params.get("temperature") is not None:
            kwargs["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            kwargs["max_tokens"] = params["max_tokens"]
        if params.get("seed") is not None:
            kwargs["seed"] = params["seed"]
        if req.json_mode and self.json_response_format:
            kwargs["response_format"] = {"type": "json_object"}
            # DeepSeek's JSON mode requires the word "json" somewhere in the prompt
            if not any("json" in m["content"].lower() for m in messages):
                messages.insert(0, {"role": "system", "content": "Reply with valid JSON only."})
        if req.extra:
            kwargs["extra_body"] = dict(req.extra)

        start = time.monotonic()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise _to_llm_error(e, self.provider) from e
        latency = time.monotonic() - start

        if not resp.choices:
            raise LLMError(f"{self.provider} returned no choices", retryable=True, provider=self.provider)
        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {"input_tokens": resp.usage.prompt_tokens or 0,
                     "output_tokens": resp.usage.completion_tokens or 0}
        return LLMResponse(text=choice.message.content or "", provider=self.provider,
                           model=getattr(resp, "model", None) or model, usage=usage,
                           latency_s=round(latency, 3), finish_reason=choice.finish_reason,
                           prompt_id=req.prompt_id, params=params)

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except Exception as e:
            raise _to_llm_error(e, self.provider) from e


def _to_llm_error(e: Exception, provider: str) -> LLMError:
    status = getattr(e, "status_code", None)
    retry_after = None
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            ra = resp.headers.get("retry-after")
            retry_after = float(ra) if ra else None
        except Exception:
            retry_after = None
    return LLMError(f"{provider} error ({status}): {e}", status=status,
                    retryable=is_retryable_status(status), retry_after=retry_after,
                    provider=provider)

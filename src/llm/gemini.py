"""Google Gemini adapter (google-genai SDK)."""

from __future__ import annotations

import time
from typing import Any

from .base import LLMError, LLMRequest, LLMResponse, is_retryable_status


class GeminiClient:
    provider = "gemini"

    def __init__(self, api_key: str, timeout_s: float = 120):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError("Gemini needs the google-genai package: pip install google-genai") from e
        self._types = types
        # v1beta: system instructions and JSON mode are not enabled on v1 for every model
        self._client = genai.Client(api_key=api_key,
                                    http_options=types.HttpOptions(api_version="v1beta",
                                                                   timeout=int(timeout_s * 1000)))

    def _contents(self, req: LLMRequest):
        t = self._types
        system = "\n\n".join(m.content for m in req.messages if m.role == "system") or None
        contents = [t.Content(role="model" if m.role == "assistant" else "user",
                              parts=[t.Part(text=m.content)])
                    for m in req.messages if m.role != "system"]
        return system, contents

    def complete(self, model: str, req: LLMRequest, params: dict[str, Any]) -> LLMResponse:
        t = self._types
        system, contents = self._contents(req)
        cfg: dict[str, Any] = {
            "system_instruction": system,
            "temperature": params.get("temperature"),
            "max_output_tokens": params.get("max_tokens"),
            "seed": params.get("seed"),
        }
        if req.json_mode:
            cfg["response_mime_type"] = "application/json"
        if "thinking_budget" in req.extra:
            cfg["thinking_config"] = t.ThinkingConfig(thinking_budget=req.extra["thinking_budget"])
        config = t.GenerateContentConfig(**{k: v for k, v in cfg.items() if v is not None})

        start = time.monotonic()
        try:
            resp = self._client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:  # map SDK errors to LLMError
            raise _to_llm_error(e) from e
        latency = time.monotonic() - start

        finish = None
        if getattr(resp, "candidates", None):
            fr = resp.candidates[0].finish_reason
            finish = getattr(fr, "name", None) or (str(fr) if fr is not None else None)
        um = getattr(resp, "usage_metadata", None)
        usage = {"input_tokens": getattr(um, "prompt_token_count", None) or 0,
                 "output_tokens": getattr(um, "candidates_token_count", None) or 0}
        thoughts = getattr(um, "thoughts_token_count", None)
        if thoughts:
            usage["thinking_tokens"] = thoughts
        text = resp.text or ""
        if not text and finish not in (None, "STOP"):
            raise LLMError(f"Gemini returned no text (finish_reason={finish})",
                           retryable=False, provider=self.provider)
        return LLMResponse(text=text, provider=self.provider, model=model, usage=usage,
                           latency_s=round(latency, 3), finish_reason=finish,
                           prompt_id=req.prompt_id, params=params)

    def list_models(self) -> list[str]:
        try:
            names = []
            for m in self._client.models.list():
                actions = getattr(m, "supported_actions", None) or []
                if not actions or "generateContent" in actions:
                    names.append(m.name.removeprefix("models/"))
            return sorted(names)
        except Exception as e:
            raise _to_llm_error(e) from e


def _to_llm_error(e: Exception) -> LLMError:
    status = getattr(e, "code", None) or getattr(e, "status_code", None)
    if not isinstance(status, int):
        status = None
    # timeouts / connection problems have no status and are retryable
    return LLMError(f"Gemini error ({status}): {e}", status=status,
                    retryable=is_retryable_status(status), provider="gemini")

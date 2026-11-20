"""Offline provider for tests and --dry-run.

Models:
  mock:echo   returns the last user message (as {"text": ...} in JSON mode)
  mock:<any>  same as echo unless a handler is set
Set MockClient.handler(req) -> str to script replies, and fail_with to inject errors.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .base import LLMError, LLMRequest, LLMResponse


class MockClient:
    provider = "mock"

    def __init__(self, handler: Callable[[LLMRequest], str] | None = None,
                 fail_with: list[LLMError] | None = None):
        self.handler = handler
        self.fail_with = list(fail_with or [])  # raised one per call, in order, before succeeding
        self.calls: list[LLMRequest] = []

    def complete(self, model: str, req: LLMRequest, params: dict[str, Any]) -> LLMResponse:
        self.calls.append(req)
        if self.fail_with:
            raise self.fail_with.pop(0)
        if self.handler:
            text = self.handler(req)
        else:
            last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            text = json.dumps({"text": last}, ensure_ascii=False) if req.json_mode else last
        return LLMResponse(text=text, provider=self.provider, model=model,
                           usage={"input_tokens": 0, "output_tokens": 0}, latency_s=0.0,
                           finish_reason="stop", prompt_id=req.prompt_id, params=params)

    def list_models(self) -> list[str]:
        return ["echo"]

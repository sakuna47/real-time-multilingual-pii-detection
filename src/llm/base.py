"""Provider-neutral request/response types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMRequest:
    messages: list[Message]
    # None = use the role's default from configs/llm.toml
    temperature: float | None = None
    max_tokens: int | None = None
    json_mode: bool = False
    seed: int | None = None
    prompt_id: str | None = None
    # provider-specific options, e.g. {"thinking_budget": 0} for Gemini
    extra: dict[str, Any] = field(default_factory=dict)
    # change this to get a different sample for an otherwise identical cached request
    cache_salt: str | None = None


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)  # input_tokens, output_tokens
    latency_s: float = 0.0
    finish_reason: str | None = None
    cached: bool = False
    prompt_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def as_model_record(self, role: str) -> dict:
        """Entry for record['generation']['models'] (docs/DATA_FORMAT.md §7)."""
        return {"role": role, "name": self.name, "params": dict(self.params),
                "prompt_id": self.prompt_id}

    def to_dict(self) -> dict:
        return {"text": self.text, "provider": self.provider, "model": self.model,
                "usage": self.usage, "latency_s": self.latency_s,
                "finish_reason": self.finish_reason, "prompt_id": self.prompt_id,
                "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "LLMResponse":
        return cls(**{k: d.get(k) for k in ("text", "provider", "model", "usage", "latency_s",
                                             "finish_reason", "prompt_id", "params")})


class LLMError(RuntimeError):
    """Error from a provider. `retryable` is True for rate limits, 5xx and timeouts."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False,
                 retry_after: float | None = None, provider: str | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after
        self.provider = provider


def is_retryable_status(status: int | None) -> bool:
    return status is None or status == 408 or status == 429 or status >= 500


class ProviderClient(Protocol):
    provider: str

    def complete(self, model: str, req: LLMRequest, params: dict[str, Any]) -> LLMResponse:
        """Send one request. `params` holds the resolved temperature/max_tokens/seed."""
        ...

    def list_models(self) -> list[str]:
        ...

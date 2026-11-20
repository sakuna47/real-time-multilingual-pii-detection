"""Build provider adapters from configs/llm.toml."""

from __future__ import annotations

from typing import Any

from ..common.secrets import get_secret
from .base import ProviderClient
from .config import ConfigError, LLMConfig
from .mock import MockClient
from .openai_compat import OPENROUTER_HEADERS, OpenAICompatClient

PROVIDERS = ("gemini", "openrouter", "deepseek", "mock")


def build_provider(name: str, cfg: LLMConfig) -> ProviderClient:
    p: dict[str, Any] = cfg.provider(name)
    if name == "mock":
        return MockClient()
    key = get_secret(p.get("key_env") or f"{name.upper()}_API_KEY")
    timeout = float(p.get("timeout_s", 120))
    if name == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(api_key=key, timeout_s=timeout)
    if name in ("openrouter", "deepseek"):
        if not p.get("base_url"):
            raise ConfigError(f"provider {name} needs base_url in configs/llm.toml")
        return OpenAICompatClient(
            provider=name, api_key=key, base_url=p["base_url"], timeout_s=timeout,
            json_response_format=bool(p.get("json_response_format", False)),
            default_headers=OPENROUTER_HEADERS if name == "openrouter" else None)
    raise ConfigError(f"unknown provider {name!r}; supported: {PROVIDERS}")

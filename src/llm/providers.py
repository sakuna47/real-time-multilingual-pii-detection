"""Build provider adapters from configs/llm.toml."""

from __future__ import annotations

from typing import Any

from ..common.secrets import get_secret
from .base import ProviderClient
from .config import ConfigError, LLMConfig
from .mock import MockClient
from .openai_compat import OPENROUTER_HEADERS, OpenAICompatClient

PROVIDERS = ("gemini", "vertex", "openrouter", "deepseek", "mock")


def build_provider(name: str, cfg: LLMConfig) -> ProviderClient:
    p: dict[str, Any] = cfg.provider(name)
    if name == "mock":
        return MockClient()
    if name == "vertex":   # Google Cloud project credits; no API key needed with ADC
        import os
        from .gemini import GeminiClient
        express_key = get_secret(p.get("key_env", "VERTEX_API_KEY"), required=False)
        project = os.environ.get(p.get("project_env", "GOOGLE_CLOUD_PROJECT"), "").strip() or p.get("project")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip() or p.get("location", "global")
        return GeminiClient(api_key=express_key, timeout_s=float(p.get("timeout_s", 120)), vertex=True,
                            project=project, location=location)
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

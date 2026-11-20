"""Load configs/llm.toml: provider settings and role -> model mapping."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common.paths import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "llm.toml"
GENERATING_ROLES = ("translator", "codemixer", "inserter")
PARAM_KEYS = ("temperature", "max_tokens", "seed")


class ConfigError(ValueError):
    pass


def split_spec(spec: str) -> tuple[str, str]:
    """'gemini:gemini-2.5-flash' -> ('gemini', 'gemini-2.5-flash'). Model ids may contain ':'."""
    if ":" not in spec:
        raise ConfigError(f"model spec {spec!r} must look like '<provider>:<model>'")
    provider, model = spec.split(":", 1)
    if not provider or not model:
        raise ConfigError(f"model spec {spec!r} must look like '<provider>:<model>'")
    return provider, model


@dataclass
class RoleSpec:
    name: str
    model: str                      # "<provider>:<model>"
    params: dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, seed
    extra: dict[str, Any] = field(default_factory=dict)   # anything else, passed to the adapter


@dataclass
class LLMConfig:
    providers: dict[str, dict[str, Any]]
    roles: dict[str, RoleSpec]

    def provider(self, name: str) -> dict[str, Any]:
        if name not in self.providers:
            raise ConfigError(f"provider {name!r} is not in configs/llm.toml "
                              f"(known: {sorted(self.providers)})")
        return self.providers[name]

    def role(self, name: str) -> RoleSpec:
        if name not in self.roles:
            raise ConfigError(f"role {name!r} is not in configs/llm.toml (known: {sorted(self.roles)})")
        return self.roles[name]


def _check(cfg: LLMConfig) -> None:
    for r in cfg.roles.values():
        provider, _ = split_spec(r.model)
        cfg.provider(provider)
    judge = cfg.roles.get("judge")
    if judge:
        for g in GENERATING_ROLES:
            if g in cfg.roles and cfg.roles[g].model == judge.model:
                raise ConfigError(f"judge and {g} both use {judge.model}; "
                                  "the judge must be a different model")


def load_config(path: str | Path | None = None, overrides: dict[str, str] | None = None) -> LLMConfig:
    """Read the TOML config. `overrides` maps role -> model spec (e.g. from a CLI flag)."""
    data = tomllib.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    providers = data.get("providers", {})
    roles = {}
    for name, body in data.get("roles", {}).items():
        body = dict(body)
        model = (overrides or {}).get(name) or body.pop("model", None)
        body.pop("model", None)
        if not model:
            raise ConfigError(f"role {name!r} has no model")
        params = {k: body.pop(k) for k in PARAM_KEYS if k in body}
        roles[name] = RoleSpec(name=name, model=model, params=params, extra=body)
    cfg = LLMConfig(providers=providers, roles=roles)
    _check(cfg)
    return cfg

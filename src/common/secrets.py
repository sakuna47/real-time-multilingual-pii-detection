"""API keys: environment -> .env in the repo -> Colab Secrets. Values are never printed."""

from __future__ import annotations

import os

from .paths import REPO_ROOT

_dotenv_loaded = False


class MissingSecret(RuntimeError):
    pass


def _load_dotenv() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    except ImportError:  # minimal fallback parser
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_secret(name: str, required: bool = True) -> str | None:
    value = os.environ.get(name, "").strip()
    if not value:
        _load_dotenv()
        value = os.environ.get(name, "").strip()
    if not value:
        try:
            from google.colab import userdata  # type: ignore
            value = (userdata.get(name) or "").strip()
        except Exception:
            value = ""
    if not value and required:
        raise MissingSecret(
            f"{name} is not set. Add it to .env (see .env.example) or to Colab Secrets.")
    return value or None


def has_secret(name: str) -> bool:
    return get_secret(name, required=False) is not None

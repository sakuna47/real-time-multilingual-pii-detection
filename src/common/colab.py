"""Colab helpers: keep the API cache on fast local disk and copy it to Drive.

In the notebook:
    from src.common import colab
    colab.restore_cache()            # Drive -> /content/cache (if a saved copy exists)
    colab.start_cache_sync(every_min=5)
    ... run pipelines ...
    colab.save_cache()               # final copy back to Drive

Copies use SQLite's backup API, so they are consistent even while the cache is in use.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading
from pathlib import Path

from .paths import cache_dir, data_root

LOCAL_CACHE = Path("/content/cache")


def drive_cache_dir() -> Path:
    d = data_root() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_env(local_cache: str | Path = LOCAL_CACHE, hf_home: str = "/content/hf_cache") -> None:
    """Point the API cache and Hugging Face downloads at Colab's local disk."""
    os.environ.setdefault("PII_CACHE_DIR", str(local_cache))
    os.environ.setdefault("HF_HOME", hf_home)
    Path(os.environ["PII_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


SECRET_NAMES = ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "HF_TOKEN",
                "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "VERTEX_API_KEY")


def load_secrets(names: tuple[str, ...] = SECRET_NAMES) -> dict[str, bool]:
    """Copy Colab Secrets into os.environ so `!python -m ...` subprocesses can see them
    (Colab Secrets are only readable from the notebook kernel). Values are never printed."""
    from google.colab import userdata  # type: ignore
    found = {}
    for name in names:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = value
        found[name] = bool(value)
    return found


def vertex_login() -> None:
    """Log this Colab session in to Google Cloud so Vertex AI uses your project's credits.
    Needs GOOGLE_CLOUD_PROJECT (Colab Secret). Not needed when VERTEX_API_KEY is set."""
    if os.environ.get("VERTEX_API_KEY"):
        return
    from google.colab import auth  # type: ignore
    auth.authenticate_user(project_id=os.environ.get("GOOGLE_CLOUD_PROJECT") or None)


def _backup(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    s = sqlite3.connect(str(src))
    d = sqlite3.connect(str(tmp))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    os.replace(tmp, dst)


def restore_cache() -> list[str]:
    """Copy saved caches from Drive to the local cache dir (only files not already there)."""
    local, drive = cache_dir(), drive_cache_dir()
    if local.resolve() == drive.resolve():
        return []
    restored = []
    for f in drive.glob("*.sqlite"):
        target = local / f.name
        if not target.exists():
            shutil.copy2(f, target)
            restored.append(f.name)
    return restored


def save_cache() -> list[str]:
    """Copy local caches to Drive."""
    local, drive = cache_dir(), drive_cache_dir()
    if local.resolve() == drive.resolve():
        return []
    saved = []
    for f in local.glob("*.sqlite"):
        _backup(f, drive / f.name)
        saved.append(f.name)
    return saved


def start_cache_sync(every_min: float = 5) -> threading.Event:
    """Save the cache to Drive every few minutes in a background thread.
    Returns an Event; call .set() on it to stop."""
    stop = threading.Event()

    def loop():
        while not stop.wait(every_min * 60):
            try:
                save_cache()
            except Exception as e:  # keep the run going; report and retry next round
                print(f"[cache sync] failed: {e}")

    threading.Thread(target=loop, daemon=True, name="cache-sync").start()
    return stop

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


def unpack_upload(name: str = "pii_project_upload.zip") -> dict | None:
    """Unpack a zip made on a PC with `python -m src.common.pack` and uploaded to
    MyDrive/PII_Project/. Runs once: the zip is renamed to *.done afterwards.
    Files that already exist on Drive are kept (Drive is newer), except the call log,
    which is merged so the budget guard sees all past spending."""
    import json
    import zipfile
    root = data_root()
    zpath = root / name
    if not zpath.exists():
        return None
    added = skipped = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir() or info.filename == "UPLOAD_MANIFEST.json":
                continue
            target = root / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if info.filename == "logs/llm_calls.jsonl" and target.exists():
                old = target.read_text(encoding="utf-8").splitlines()
                seen = set(old)
                new_lines = [l for l in z.read(info).decode("utf-8").splitlines() if l and l not in seen]
                target.write_text("\n".join(new_lines + old) + "\n", encoding="utf-8")
                added += 1
                continue
            if target.exists():
                skipped += 1
                continue
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            added += 1
        manifest = json.loads(z.read("UPLOAD_MANIFEST.json")) if "UPLOAD_MANIFEST.json" in z.namelist() else {}
    zpath.rename(zpath.with_name(zpath.name + ".done"))
    return {"added": added, "kept_existing": skipped, "packed_at": manifest.get("created_at")}


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

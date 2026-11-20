"""Where data lives.

Order: PII_DATA_ROOT env var -> Google Drive project folder (when mounted in
Colab) -> ./data in the repo. Sub-folders follow 00_folder_creation.ipynb.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVE_ROOT = Path("/content/drive/MyDrive/PII_Project")


def data_root() -> Path:
    env = os.environ.get("PII_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    if DRIVE_ROOT.exists():
        return DRIVE_ROOT
    return REPO_ROOT / "data"


def path(*parts: str, mkdir: bool = False) -> Path:
    """data_root()/parts... ; with mkdir=True the parent folder is created."""
    p = data_root().joinpath(*parts)
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    """API response cache. PII_CACHE_DIR overrides it; in Colab point it at local disk
    (/content/cache) because SQLite on mounted Drive is slow and can lock, then copy it
    to Drive with src.common.colab."""
    env = os.environ.get("PII_CACHE_DIR", "").strip()
    d = Path(env) if env else data_root() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = data_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(*parts: str) -> Path:
    d = data_root().joinpath("results", *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def sources_dir(*parts: str) -> Path:
    d = data_root().joinpath("sources", *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_dir(*parts: str) -> Path:
    """Raw downloads that are not on the Hugging Face Hub (e.g. GitHub zips). Lives next to
    HF_HOME when it is set (Colab local disk), otherwise under <data_root>/raw/downloads."""
    hf_home = os.environ.get("HF_HOME", "").strip()
    base = Path(hf_home) / "downloads" if hf_home else data_root() / "raw" / "downloads"
    d = base.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d

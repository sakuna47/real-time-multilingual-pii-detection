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
    d = data_root() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = data_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d

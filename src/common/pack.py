"""Pack local work into one zip to upload to Google Drive (for Colab to continue from).

    python -m src.common.pack                      # -> data/export/pii_project_upload.zip

Upload the zip to MyDrive/PII_Project/ in your browser. The Colab notebook unpacks it
once (src.common.colab.unpack_upload) and renames it to *.done.

Included: generated/, validated/, review/, results/, logs/ (the call log, needed for the
budget guard), cache/*.sqlite (saved model answers: nothing is paid twice), and the small
sources (aya, codemix, multiner, flores). Nemotron and WildChat sources are left out:
Colab re-imports them in full.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

from .paths import cache_dir, data_root
from .runinfo import git_commit, utc_now

FOLDERS = ["generated", "validated", "review", "results", "logs",
           "sources/aya", "sources/codemix", "sources/multiner", "sources/flores"]
UPLOAD_NAME = "pii_project_upload.zip"


def _sqlite_snapshot(src: Path, dst: Path) -> None:
    s, d = sqlite3.connect(str(src)), sqlite3.connect(str(dst))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()


def pack(out: Path | None = None) -> Path:
    root = data_root()
    out = out or root / "export" / UPLOAD_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for folder in FOLDERS:
            base = root / folder
            if not base.exists():
                continue
            n = 0
            for p in sorted(base.rglob("*")):
                if p.is_file() and not p.name.endswith((".tmp", ".part")):
                    z.write(p, p.relative_to(root).as_posix())
                    n += 1
            counts[folder] = n
        with tempfile.TemporaryDirectory() as tmp:
            for db in sorted(cache_dir().glob("*.sqlite")):
                if "dryrun" in db.name:
                    continue
                snap = Path(tmp) / db.name
                _sqlite_snapshot(db, snap)           # consistent copy even if the cache is open
                z.write(snap, f"cache/{db.name}")
                counts[f"cache/{db.name}"] = 1
        z.writestr("UPLOAD_MANIFEST.json", json.dumps(
            {"created_at": utc_now(), "git_commit": git_commit(), "files": counts}, indent=2))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    path = pack(args.out)
    with zipfile.ZipFile(path) as z:
        info = json.loads(z.read("UPLOAD_MANIFEST.json"))
    print(json.dumps(info["files"], indent=2))
    print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB) -> upload it to MyDrive/PII_Project/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

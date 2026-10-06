#!/usr/bin/env python3
"""kaggle/push_code.py — upload the code the notebook needs as a PRIVATE Kaggle dataset.

Keeps work-in-progress off GitHub: the notebook reads src/, the dataset
builder and notebook_helpers.py from /kaggle/input/aerial-guardian-code.

  python kaggle/push_code.py --user <kaggle-user> --new     # first time
  python kaggle/push_code.py --user <kaggle-user>           # later updates
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.kaggle_cli import kaggle_command  # noqa: E402
STAGE = ROOT / "build" / "kaggle-code"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True)
    p.add_argument("--new", action="store_true", help="create the dataset instead of versioning it")
    args = p.parse_args()

    shutil.rmtree(STAGE, ignore_errors=True)
    shutil.copytree(ROOT / "src", STAGE / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (STAGE / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "01_build_dataset.py", STAGE / "scripts")
    shutil.copy2(ROOT / "kaggle" / "notebook_helpers.py", STAGE)  # dataset root; the notebook puts it on sys.path
    meta = {"title": "aerial-guardian-code", "id": f"{args.user}/aerial-guardian-code",
            "licenses": [{"name": "MIT"}]}
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    if args.new:
        cmd = [*kaggle_command(), "datasets", "create", "-p", str(STAGE), "--dir-mode", "zip"]
    else:
        cmd = [*kaggle_command(), "datasets", "version", "-p", str(STAGE), "--dir-mode", "zip", "-m", "update code"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

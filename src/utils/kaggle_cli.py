"""src/utils/kaggle_cli.py — locate the `kaggle` CLI even when the venv is not activated.

We normally run `.venv/Scripts/python script.py` without activating the venv, so the
venv's Scripts/ (bin/) directory is not on PATH and a bare ["kaggle", ...] subprocess
fails with FileNotFoundError. The CLI is installed next to the interpreter, so look
there too.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def kaggle_command() -> list[str]:
    """Return the argv prefix that runs the kaggle CLI (PATH first, then next to sys.executable)."""
    found = shutil.which("kaggle")
    if found:
        return [found]
    exe_dir = Path(sys.executable).resolve().parent
    for name in ("kaggle.exe", "kaggle"):
        sibling = exe_dir / name
        if sibling.is_file():
            return [str(sibling)]
    raise FileNotFoundError(
        "kaggle CLI not found on PATH or next to the Python interpreter "
        f"({exe_dir}). Run `pip install kaggle` in this environment or activate the project venv."
    )

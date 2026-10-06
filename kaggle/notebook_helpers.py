"""kaggle/notebook_helpers.py — small pure helpers used by the Kaggle training notebook.

They live in a module (shipped in the code dataset by push_code.py, imported from the
notebook via the code dir) rather than in notebook cells so that unit tests can import
them. Their job is to turn silent degradations into loud failures: a wrong MOT path
must not quietly become a DET-only run, and a late start must not run into Kaggle's
hard session kill.
"""
from __future__ import annotations

import os
from pathlib import Path

MOT_DIR_NAME = "VisDrone2019-MOT-train"


def _is_mot_train_dir(path: Path) -> bool:
    return path.is_dir() and (path / "annotations").is_dir() and (path / "sequences").is_dir()


def find_mot_train_dir(input_root: Path) -> Path | None:
    """First (sorted) directory named VisDrone2019-MOT-train under input_root holding annotations/ and sequences/."""
    for dirpath, dirnames, _ in os.walk(input_root):
        dirnames.sort()
        if Path(dirpath).name == MOT_DIR_NAME and _is_mot_train_dir(Path(dirpath)):
            return Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in ("annotations", "sequences")]  # skip the ~34k files
    return None


def list_top_levels(input_root: Path, depth: int = 2) -> list[str]:
    """Paths (relative to input_root) of the first `depth` directory levels, for error messages."""
    root = Path(input_root)
    if not root.is_dir():
        return [f"(missing: {root})"]
    out: list[str] = []
    for a in sorted(p for p in root.iterdir() if p.is_dir()):
        out.append(a.name)
        if depth >= 2:
            out += [f"{a.name}/{b.name}" for b in sorted(p for p in a.iterdir() if p.is_dir())]
    return out


def resolve_mot_dir(configured: str | None, input_root: str | Path = "/kaggle/input") -> str | None:
    """Resolve the MOT-train directory.

    None            -> None (explicit DET-only opt-out, --no-mot).
    configured ok   -> configured.
    configured bad  -> search input_root for a valid MOT-train dir (Kaggle has changed mount
                       layouts before); raise with a listing of input_root if none is found.
    """
    if configured is None:
        return None
    if _is_mot_train_dir(Path(configured)):
        return configured
    found = find_mot_train_dir(Path(input_root))
    if found is not None:
        print(f"MOT_TRAIN_DIR {configured} not found; using {found}")
        return str(found)
    raise FileNotFoundError(
        f"MOT_TRAIN_DIR {configured!r} is not a valid VisDrone2019-MOT-train directory (needs annotations/ and "
        f"sequences/) and none was found under {input_root}. Contents of {input_root} (top two levels): "
        f"{list_top_levels(Path(input_root))}. Rebuild the notebook with --mot-train-dir <path>, or --no-mot "
        f"to train on DET data only."
    )


def compute_time_budget(time_hours: float, elapsed_h: float, session_limit_h: float = 11.0,
                        reserve_h: float = 1.0, min_h: float = 0.5) -> float:
    """Training hours that still fit: min(requested, session limit - time already used - reserve)."""
    budget = min(time_hours, session_limit_h - elapsed_h - reserve_h)
    if budget <= min_h:
        raise RuntimeError(
            f"only {budget:.2f} h of training budget left (session limit {session_limit_h} h, "
            f"{elapsed_h:.2f} h already used by setup, {reserve_h} h reserved for validation/export); "
            f"refusing to start a pointless run"
        )
    return budget

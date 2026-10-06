#!/usr/bin/env python3
"""kaggle/build_notebook.py — render the Kaggle training notebook + kernel metadata.

Kaggle kernels take no runtime arguments, so parameters are baked into the
notebook's PARAMS cell before `kaggle kernels push`.

  python kaggle/build_notebook.py --user <kaggle-user> --mot-dataset <owner/slug> --scale n --smoke
  kaggle kernels push -p kaggle
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jupytext

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "train_notebook.py"
DEFAULT_TIME_HOURS = 9.5  # keep equal to TIME_HOURS in train_notebook.py (pinned by a test)


def _literal(value) -> str:
    """Python source literal: JSON quoting for str, repr for None/bool/numbers (never JSON null/true)."""
    return json.dumps(value) if isinstance(value, str) else repr(value)


def render(params: dict) -> str:
    src = SOURCE.read_text(encoding="utf-8")
    for key, value in params.items():
        src, n = re.subn(rf"^{key} = .*$", f"{key} = {_literal(value)}", src, count=1, flags=re.M)
        if n != 1:
            raise KeyError(f"parameter {key} not found in {SOURCE.name}")
    return src


def default_mot_train_dir(mot_dataset: str) -> str:
    """Where the MOT train split sits under /kaggle/input for the chosen dataset.

    This is the layout of akhileswarelluru/visdrone2019-mot (extracted files, not zips):
    VisDrone_Zips/VisDrone2019-MOT-train/VisDrone2019-MOT-train/{annotations,sequences}.
    Datasets laid out differently need an explicit --mot-train-dir.
    """
    name = mot_dataset.split("/")[-1]
    return f"/kaggle/input/{name}/VisDrone_Zips/VisDrone2019-MOT-train/VisDrone2019-MOT-train"


def mot_dir_param(mot_dataset: str, mot_train_dir: str | None, no_mot: bool) -> str | None:
    """MOT_TRAIN_DIR value for the notebook; None only as an explicit --no-mot opt-out."""
    if no_mot:
        return None
    return mot_train_dir or default_mot_train_dir(mot_dataset)


def kernel_metadata(user: str, mot_dataset: str, scale: str) -> dict:
    return {
        "id": f"{user}/aerial-guardian-train-{scale}",
        "title": f"aerial-guardian-train-{scale}",
        "code_file": "02_train_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{user}/aerial-guardian-code", mot_dataset],
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--user", required=True)
    p.add_argument("--mot-dataset", required=True, help="Kaggle dataset slug owner/name with VisDrone MOT")
    p.add_argument("--mot-train-dir", default=None,
                   help="Path of VisDrone2019-MOT-train inside /kaggle/input "
                        "(default derived from the slug, assuming the akhileswarelluru/visdrone2019-mot layout)")
    p.add_argument("--no-mot", action="store_true",
                   help="explicit opt-out: train on DET data only (otherwise a missing MOT dir is an error)")
    p.add_argument("--scale", choices=["n", "s"], default="s")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--time-hours", type=float, default=DEFAULT_TIME_HOURS,
                   help="training-time cap; the notebook also subtracts setup time to stay under Kaggle's 12 h kill")
    p.add_argument("--resume-from", default=None)
    args = p.parse_args()

    if args.no_mot and args.mot_train_dir:
        p.error("--no-mot and --mot-train-dir are mutually exclusive")
    mot_dir = mot_dir_param(args.mot_dataset, args.mot_train_dir, args.no_mot)
    src = render({"SCALE": args.scale, "SMOKE": args.smoke, "TIME_HOURS": args.time_hours,
                  "MOT_TRAIN_DIR": mot_dir, "RESUME_FROM": args.resume_from})
    jupytext.write(jupytext.reads(src, fmt="py:percent"), HERE / "02_train_kaggle.ipynb")
    meta = kernel_metadata(args.user, args.mot_dataset, args.scale)
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote 02_train_kaggle.ipynb + kernel-metadata.json for {meta['id']}")


if __name__ == "__main__":
    main()

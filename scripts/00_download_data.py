#!/usr/bin/env python3
"""
scripts/00_download_data.py — fetch the VisDrone splits used locally.

Local disk only needs the evaluation splits: DET-val (detector metrics),
MOT-val (tracker tuning) and MOT-test-dev (reported tracking metrics).
Training splits are downloaded on Kaggle by the notebook instead.

The Kaggle MOT dataset (akhileswarelluru/visdrone2019-mot, ~12 GB) stores ~34k
already-extracted files rather than zips, so fetching single files with
`kaggle datasets download -f` is impractical. Instead the whole dataset zip is
downloaded once, only the requested folders are extracted (SRC_PREFIX=DEST_NAME,
prefix stripped), and the zip is deleted again unless --keep-zip is given —
disk is the scarce resource.

  python scripts/00_download_data.py --det val
  python scripts/00_download_data.py --mot-kaggle akhileswarelluru/visdrone2019-mot \\
      --mot-extract VisDrone_Zips/VisDrone2019-MOT-val/VisDrone2019-MOT-val=VisDrone2019-MOT-val \\
      --mot-extract VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev=VisDrone2019-MOT-test-dev
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "visdrone"


def parse_extract_spec(spec: str) -> tuple[str, str]:
    """Split 'SRC_PREFIX=DEST_NAME'; DEST_NAME must be a plain relative path under --out."""
    src, sep, dest = spec.partition("=")
    src, dest = src.strip("/"), dest.strip()
    if not sep or not src or not dest:
        raise ValueError(f"expected SRC_PREFIX=DEST_NAME, got {spec!r}")
    if dest.startswith(("/", "\\")) or ".." in Path(dest).parts or Path(dest).is_absolute():
        raise ValueError(f"DEST_NAME must be a relative path without '..': {dest!r}")
    return src, dest


def extract_prefix(zip_path: Path, src_prefix: str, dest_dir: Path) -> int:
    """Extract members under `src_prefix/` into `dest_dir`, with the prefix stripped.

    Returns the number of files written. Directory entries are skipped (parents are
    created on demand), members outside the prefix are ignored, and any member whose
    resolved path would escape `dest_dir` (zip-slip, e.g. '<prefix>/../../x') is skipped.
    """
    prefix = src_prefix.strip("/") + "/"
    dest_root = Path(dest_dir).resolve()
    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            rel = name[len(prefix):]
            target = (dest_root / rel).resolve()
            if target == dest_root or dest_root not in target.parents:
                print(f"skipping unsafe zip member: {info.filename!r}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            written += 1
    return written


def download_kaggle_zip(slug: str, out: Path) -> Path:
    """Download the full dataset zip (no --unzip) into <out>/_downloads and return its path."""
    dl_dir = out / "_downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dl_dir / f"{slug.split('/')[-1]}.zip"
    if zip_path.exists():
        print(f"reusing existing {zip_path}")
        return zip_path
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dl_dir)], check=True)
    if not zip_path.exists():
        found = sorted(dl_dir.glob("*.zip"))
        if len(found) != 1:
            raise FileNotFoundError(f"expected {zip_path.name} in {dl_dir}, found {[f.name for f in found]}")
        zip_path = found[0]
    return zip_path


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--det", nargs="*", default=[], choices=["train", "val", "test-dev"])
    p.add_argument("--mot-kaggle", default=None, help="Kaggle dataset slug holding VisDrone MOT")
    p.add_argument("--mot-extract", action="append", default=[], metavar="SRC_PREFIX=DEST_NAME",
                   help="Folder inside the dataset zip to extract into <out>/DEST_NAME (repeatable)")
    p.add_argument("--keep-zip", action="store_true", help="Keep the downloaded dataset zip")
    p.add_argument("--out", type=Path, default=DATA)
    args = p.parse_args()
    if args.mot_extract and not args.mot_kaggle:
        p.error("--mot-extract requires --mot-kaggle")
    if args.mot_kaggle and not args.mot_extract:
        p.error("--mot-kaggle requires at least one --mot-extract SRC_PREFIX=DEST_NAME")
    try:
        specs = [parse_extract_spec(s) for s in args.mot_extract]
    except ValueError as e:
        p.error(str(e))
    args.out.mkdir(parents=True, exist_ok=True)

    if args.det:
        from ultralytics.utils import ASSETS_URL
        from ultralytics.utils.downloads import download
        download([f"{ASSETS_URL}/VisDrone2019-DET-{s}.zip" for s in args.det], dir=args.out, threads=2)

    if args.mot_kaggle:
        zip_path = download_kaggle_zip(args.mot_kaggle, args.out)
        for src, dest in specs:
            n = extract_prefix(zip_path, src, args.out / dest)
            if n == 0:
                raise SystemExit(f"no files under '{src}/' in {zip_path.name}; "
                                 f"check the prefix (zip kept at {zip_path})")
            print(f"extracted {n} files -> {args.out / dest}")
        if not args.keep_zip:
            zip_path.unlink()
            try:
                zip_path.parent.rmdir()
            except OSError:
                pass
    print(f"data in {args.out}: {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    main()

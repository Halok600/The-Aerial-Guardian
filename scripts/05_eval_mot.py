#!/usr/bin/env python3
"""
scripts/05_eval_mot.py — Tracking evaluation on VisDrone MOT (HOTA / MOTA / IDF1).

Modes
-----
  report (default): on --split dir (MOT-test-dev), compare
      * Ultralytics' built-in ByteTrack with its own default config
      * Ultralytics' ByteTrack with OUR tuned thresholds (isolates the effect of
        our tracker code from the effect of threshold tuning)
      * ours without camera-motion compensation
      * ours with ECC camera-motion compensation
    per evaluation group (person, vehicle) -> outputs/eval/mot/{mot_table.md, mot_per_seq.md,
    mot_metrics.json}
  --tune: grid-search tracker thresholds on MOT-val using cached detections (CMC on)
    -> outputs/eval/mot_tune/tuning.md (copy the winner into configs/tracker.yaml).
    Refuses any --split whose directory name contains "test": never tune on test-dev.

All baselines use the same weights, imgsz, conf floor, NMS IoU and max_det as our detector.
Detections + ECC warps are cached per sequence, so tracker variants replay in seconds.
--max-seqs / --max-frames shrink the run (plumbing smoke tests); GT is cut to the same frames.
Output directories are only wiped when they live inside the repo's outputs/ (or with --force-out).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.visdrone import parse_mot_file  # noqa: E402
from src.eval.mot_format import (  # noqa: E402
    EVAL_GROUPS, filter_ignored, gt_lines, ignore_regions, limit_frames, pred_lines,
    write_gt, write_seqmap, write_tracker,
)
from src.eval.trackeval_runner import run_trackeval  # noqa: E402
from src.pipeline import Detector, DetectorSettings, TrackRow, cache_sequence, track_cached  # noqa: E402
from src.tracker.byte_tracker import ByteTracker, TrackerConfig  # noqa: E402
from src.tracker.ecc_compensator import ECCCompensator  # noqa: E402
from src.utils.classes import group_of  # noqa: E402
from src.utils.video_io import FramePrefetcher  # noqa: E402

GRID = {"track_high": [0.3, 0.4, 0.5], "new_track": [0.4, 0.5, 0.6], "max_time_lost": [30, 60],
        "match_thresh": [0.7, 0.8, 0.9], "fuse_score": [True, False]}
GRID_SMALL = {"track_high": [0.3, 0.5], "new_track": [0.5], "max_time_lost": [30],
              "match_thresh": [0.8], "fuse_score": [True]}
DET_CONF = 0.05     # shared by our detector and the Ultralytics baselines (a fair reference)
DET_IOU = 0.6
MAX_DET = 1000
CACHE_VERSION = 1   # bump when the ECC / detector-pipeline code changes so old caches are not reused
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
BASELINE_NAMES = {"default": "ultralytics_default", "ours": "ultralytics_ours"}
TABLE_ORDER = ("ultralytics_default", "ultralytics_ours", "ours_nocmc", "ours_cmc")
TABLE_NAMES = {"ultralytics_default": "Ultralytics ByteTrack (default config)",
               "ultralytics_ours": "Ultralytics ByteTrack (our thresholds)",
               "ours_nocmc": "Ours (no camera-motion comp.)",
               "ours_cmc": "Ours + ECC camera-motion comp."}


# ---- arguments and filesystem safety --------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", default="models/weights/yolo26s-p2-visdrone.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--split", type=Path, default=Path("data/visdrone/VisDrone2019-MOT-test-dev"))
    p.add_argument("--tracker-config", default="configs/tracker.yaml")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: outputs/eval/mot, or outputs/eval/mot_tune with --tune)")
    p.add_argument("--tune", action="store_true", help="grid search (only on a validation split)")
    p.add_argument("--grid-small", action="store_true", help="with --tune: a 2-variant grid for smoke tests")
    p.add_argument("--baseline", choices=("default", "ours", "both"), default="both",
                   help="Ultralytics ByteTrack baseline(s): its default config, our thresholds, or both")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--max-seqs", type=int, default=None, help="use only the first N sequences")
    p.add_argument("--max-frames", type=int, default=None, help="use only the first N frames of each sequence")
    p.add_argument("--force-out", action="store_true",
                   help="allow wiping an output dir outside the repo's outputs/ directory")
    args = p.parse_args(argv)
    if args.tune and "test" in args.split.resolve().name.lower():
        p.error(f"--tune refuses test splits ({args.split.name}); tune on MOT-val only, never on test-dev")
    if args.out is None:
        args.out = Path("outputs/eval/mot_tune" if args.tune else "outputs/eval/mot")
    return args


def _strictly_inside(target: Path, root: Path) -> bool:
    return target != root and root in target.parents


def check_out_dir(out: Path, outputs_root: Path | None = None, force: bool = False) -> None:
    """Raise ValueError unless `out` (relative paths: against the repo root) is inside outputs/."""
    root = Path(outputs_root or OUTPUTS_ROOT).resolve()
    target = (Path(out) if Path(out).is_absolute() else PROJECT_ROOT / out).resolve()
    if not force and not _strictly_inside(target, root):
        raise ValueError(f"refusing to use {target} as an output dir: it is not inside {root} "
                         "(pass --force-out to override)")


def safe_rmtree(path: Path, outputs_root: Path | None = None, force: bool = False) -> None:
    """shutil.rmtree that only ever deletes inside outputs/ (symlinks resolved) unless forced."""
    root = Path(outputs_root or OUTPUTS_ROOT).resolve()
    target = Path(path).resolve()
    if not force and not _strictly_inside(target, root):
        raise ValueError(f"refusing to delete {target}: it is not inside {root} (pass --force-out to override)")
    if target.exists():
        shutil.rmtree(target)


# ---- detection cache -------------------------------------------------------------------------

def limited(frames, max_frames: int | None):
    """Yield (frame_id, frame) pairs, at most `max_frames` of them, and always release the reader."""
    with frames:
        for frame_id, frame in frames:
            if max_frames is not None and frame_id > max_frames:
                return
            yield frame_id, frame


def cache_dir_for(weights, settings: DetectorSettings, split: Path, max_frames: int | None) -> Path:
    """Cache folder keyed by everything that changes the cached detections / warps."""
    w = Path(weights)
    stamp = f"{w.stat().st_size}_{int(w.stat().st_mtime)}" if w.exists() else "na"   # retrained weights -> new cache
    key = (f"{w.stem}_{stamp}_{settings.imgsz}_c{settings.conf}_i{settings.iou}_m{settings.max_det}"
           f"_v{CACHE_VERSION}_{Path(split).name}_f{max_frames or 'all'}")
    return PROJECT_ROOT / "outputs" / "cache" / key


def load_cache(seq_dir: Path, get_detector, cache_dir: Path, max_frames: int | None = None):
    """(dets, warps) for a sequence; unreadable caches are recomputed, writes are atomic."""
    path = cache_dir / f"{seq_dir.name}.npz"
    if path.exists():
        try:
            with np.load(path) as z:
                return z["dets"], z["warps"]
        except Exception as e:  # noqa: BLE001 - corrupt/partial file: recompute below
            print(f"[WARN] ignoring unreadable cache {path.name}: {e}")
    dets, warps = cache_sequence(limited(FramePrefetcher(seq_dir), max_frames), get_detector(), ECCCompensator())
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, dets=dets, warps=warps)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return dets, warps


# ---- Ultralytics baseline --------------------------------------------------------------------

def ultralytics_tracker_config(cfg: TrackerConfig) -> dict:
    """Our thresholds in Ultralytics' bytetrack.yaml vocabulary (same semantics, verified in 8.4.174:
    match_thresh is a 1-IoU cost threshold, track_buffer is in frames at 30 fps, fuse_score fuses
    IoU with detection score). Its stage-2 (0.5) and unconfirmed (0.7) thresholds are hard-coded and
    equal our defaults; we do not tune those."""
    return {"tracker_type": "bytetrack", "track_high_thresh": cfg.track_high,
            "track_low_thresh": cfg.track_low, "new_track_thresh": cfg.new_track,
            "track_buffer": cfg.max_time_lost, "match_thresh": cfg.match_thresh,
            "fuse_score": cfg.fuse_score}


def write_ultralytics_yaml(cfg: TrackerConfig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(ultralytics_tracker_config(cfg), sort_keys=False), encoding="utf-8")
    return path


def ultralytics_rows(weights: str, seq_dir: Path, imgsz: int, max_frames: int | None = None,
                     tracker: str = "bytetrack.yaml", conf: float = DET_CONF) -> list[TrackRow]:
    """Ultralytics' own ByteTrack on the same weights and NMS settings as ours (`tracker` = yaml name/path)."""
    import torch
    from ultralytics import YOLO

    cuda = torch.cuda.is_available()
    model = YOLO(weights)                     # fresh instance = fresh tracker state per sequence
    rows: list[TrackRow] = []
    for frame_id, frame in limited(FramePrefetcher(seq_dir), max_frames):
        r = model.track(frame, persist=True, tracker=tracker, imgsz=imgsz, conf=conf,
                        iou=DET_IOU, max_det=MAX_DET, device=0 if cuda else "cpu", verbose=False,
                        **({"quantize": 16} if cuda else {}))[0]    # fp16 on GPU, like our Detector
        if r.boxes is None or r.boxes.id is None:
            continue
        for (x1, y1, x2, y2), s, c, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                                r.boxes.cls.cpu().numpy(), r.boxes.id.cpu().numpy()):
            rows.append(TrackRow(frame_id, int(tid), x1, y1, x2, y2, float(s), group_of(int(c))))
    return rows


# ---- ground truth, variants, tracking --------------------------------------------------------

def prepare_gt(mot_dir: Path, seqs: list[str], gt_root: Path, max_frames: int | None = None) -> dict:
    """Write per-group GT + seqinfo/seqmap (caller clears gt_root); returns each sequence's ignore regions."""
    ignores = {}
    for group, cats in EVAL_GROUPS.items():
        for seq in seqs:
            frames = limit_frames(parse_mot_file(mot_dir / "annotations" / f"{seq}.txt"), max_frames)
            length = len(list((mot_dir / "sequences" / seq).glob("*.jpg")))
            if max_frames is not None:
                length = min(length, max_frames)
            write_gt(gt_root, group, seq, gt_lines(frames, cats), length)
            ignores[seq] = ignore_regions(frames)
        write_seqmap(gt_root, group, seqs)
    return ignores


def write_rows(trk_root: Path, name: str, seq: str, rows: list[TrackRow], ignore: dict) -> None:
    for group in EVAL_GROUPS:
        sel = filter_ignored([r for r in rows if r.group == group], ignore)
        write_tracker(trk_root, group, name, seq, pred_lines(sel))


def build_variants(args: argparse.Namespace, base_cfg: TrackerConfig) -> dict[str, tuple[TrackerConfig, bool]]:
    """{variant name: (tracker config, use camera-motion compensation)}; names are only labels."""
    if not args.tune:
        return {"ours_cmc": (base_cfg, True), "ours_nocmc": (base_cfg, False)}
    grid = GRID_SMALL if args.grid_small else GRID
    variants = {}
    for values in itertools.product(*grid.values()):
        point = dict(zip(grid, values))
        if point["new_track"] < point["track_high"]:
            continue    # no-op: only detections >= track_high can start tracks, so this equals new_track == track_high
        name = "_".join(f"{k}{v}" for k, v in zip(grid, values))
        variants[name] = (replace(base_cfg, **dict(zip(grid, values))), True)
    return variants


def track_variants(variants: dict[str, tuple[TrackerConfig, bool]], caches: dict, ignores: dict,
                   trk_root: Path) -> None:
    for name, (cfg, use_cmc) in tqdm(variants.items(), desc="track"):
        for seq, (dets, warps) in caches.items():
            rows = track_cached(dets, warps, ByteTracker(cfg), use_cmc=use_cmc)
            write_rows(trk_root, name, seq, rows, ignores[seq])


# ---- tables ------------------------------------------------------------------------------------

def _ordered(keys) -> list[str]:
    return [k for k in TABLE_ORDER if k in keys] + [k for k in keys if k not in TABLE_ORDER]


def table(results: dict[str, dict[str, dict]], names: dict[str, str]) -> str:
    out = []
    for group, res in results.items():
        out += [f"**{group}**", "", "| Tracker | HOTA | MOTA | IDF1 | ID switches |", "|---|---|---|---|---|"]
        out += [f"| {names.get(k, k)} | {res[k]['HOTA']:.1f} | {res[k]['MOTA']:.1f} | {res[k]['IDF1']:.1f} "
                f"| {res[k]['IDSW']} |" for k in _ordered(res)]
        out.append("")
    return "\n".join(out)


def per_seq_table(results: dict[str, dict[str, dict]], names: dict[str, str]) -> str:
    """HOTA per sequence (rows) and tracker (columns), per evaluation group."""
    out = []
    for group, res in results.items():
        keys = _ordered(res)
        seqs = sorted({s for k in keys for s in res[k].get("per_seq", {})})
        out += [f"**{group}** (HOTA per sequence)", "",
                "| Sequence | " + " | ".join(names.get(k, k) for k in keys) + " |",
                "|---|" + "---|" * len(keys)]
        cells = {k: res[k].get("per_seq", {}) for k in keys}
        out += [f"| {s} | " + " | ".join(
            "-" if s not in cells[k] else "n/a" if cells[k][s]["HOTA"] is None else f"{cells[k][s]['HOTA']:.1f}"
            for k in keys) + " |" for s in seqs]
        if any(cells[k][s]["HOTA"] is None for k in keys for s in cells[k]):
            out += ["", "n/a: the sequence has no ground-truth objects of this group"]
        out.append("")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    try:
        check_out_dir(out, force=args.force_out)
    except ValueError as e:
        sys.exit(f"error: {e}")

    seqs = sorted(d.name for d in (args.split / "sequences").iterdir() if d.is_dir())[:args.max_seqs]
    gt_root, trk_root = out / "gt", out / "trackers"
    safe_rmtree(gt_root, force=args.force_out)
    safe_rmtree(trk_root, force=args.force_out)
    ignores = prepare_gt(args.split, seqs, gt_root, args.max_frames)

    settings = DetectorSettings(imgsz=args.imgsz, conf=DET_CONF, iou=DET_IOU, max_det=MAX_DET)
    detector = None

    def get_detector() -> Detector:
        nonlocal detector
        if detector is None:
            detector = Detector(args.weights, settings)
        return detector

    cache_dir = cache_dir_for(args.weights, settings, args.split, args.max_frames)
    caches = {s: load_cache(args.split / "sequences" / s, get_detector, cache_dir, args.max_frames)
              for s in tqdm(seqs, desc="detect")}
    base_cfg = TrackerConfig.from_yaml(args.tracker_config)

    variants = build_variants(args, base_cfg)
    track_variants(variants, caches, ignores, trk_root)
    if not args.tune and not args.no_baseline:
        baselines = {"default": "bytetrack.yaml"}
        if args.baseline in ("ours", "both"):
            baselines["ours"] = str(write_ultralytics_yaml(base_cfg, out / "ultralytics_ours_bytetrack.yaml"))
        if args.baseline == "ours":
            baselines.pop("default")
        for key, tracker in baselines.items():
            for seq in tqdm(seqs, desc=f"ultralytics bytetrack ({key})"):
                write_rows(trk_root, BASELINE_NAMES[key], seq,
                           ultralytics_rows(args.weights, args.split / "sequences" / seq, args.imgsz,
                                            args.max_frames, tracker=tracker),
                           ignores[seq])

    names = sorted(p.name for p in (trk_root / "person-test").iterdir())
    results = {g: run_trackeval(gt_root, trk_root, g, names) for g in EVAL_GROUPS}
    out.mkdir(parents=True, exist_ok=True)
    (out / "mot_metrics.json").write_text(json.dumps(results, indent=2))

    if args.tune:
        score = {n: float(np.mean([results[g][n]["HOTA"] for g in EVAL_GROUPS])) for n in names}
        idsw = {n: sum(results[g][n]["IDSW"] for g in EVAL_GROUPS) for n in names}
        best = max(score, key=score.get)
        lines = ["| Variant | mean HOTA | ID switches |", "|---|---|---|"] + [
            f"| {n} | {score[n]:.2f} | {idsw[n]} |" for n in sorted(score, key=score.get, reverse=True)]
        (out / "tuning.md").write_text("\n".join(lines) + f"\n\nBest: {best}\n{asdict(variants[best][0])}\n")
        print("\n".join(lines[:8]), f"\nBEST {best}: {asdict(variants[best][0])}")
    else:
        md = table(results, TABLE_NAMES)
        (out / "mot_table.md").write_text(md)
        (out / "mot_per_seq.md").write_text(per_seq_table(results, TABLE_NAMES))
        print(md)


if __name__ == "__main__":
    main()

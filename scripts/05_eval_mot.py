#!/usr/bin/env python3
"""
scripts/05_eval_mot.py — Tracking evaluation on VisDrone MOT (HOTA / MOTA / IDF1).

Modes
-----
  report (default): on --split dir (MOT-test-dev), compare
      * Ultralytics' built-in ByteTrack (same weights/imgsz/NMS settings, its default config)
      * ours without camera-motion compensation
      * ours with ECC camera-motion compensation
    per evaluation group (person, vehicle) -> outputs/eval/mot/mot_table.md
  --tune: grid-search tracker thresholds on MOT-val using cached detections
    -> outputs/eval/mot_tune/tuning.md (copy the winner into configs/tracker.yaml)

Detections + ECC warps are cached per sequence, so tracker variants replay in seconds.
--max-seqs / --max-frames shrink the run (plumbing smoke tests); GT is cut to the same frames.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
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

GRID = {"track_high": [0.3, 0.4, 0.5], "new_track": [0.4, 0.5, 0.6], "max_time_lost": [30, 60]}
GRID_SMALL = {"track_high": [0.3, 0.5], "new_track": [0.5], "max_time_lost": [30]}
DET_CONF = 0.05     # shared by our detector and the Ultralytics baseline (a fair reference)
DET_IOU = 0.6
TABLE_ORDER = ("ultralytics_bytetrack", "ours_nocmc", "ours_cmc")
TABLE_NAMES = {"ultralytics_bytetrack": "Ultralytics ByteTrack (default)",
               "ours_nocmc": "Ours (no camera-motion comp.)",
               "ours_cmc": "Ours + ECC camera-motion comp."}


def limited(frames, max_frames: int | None):
    """Yield (frame_id, frame) pairs, at most `max_frames` of them, and always release the reader."""
    with frames:
        for frame_id, frame in frames:
            if max_frames is not None and frame_id > max_frames:
                return
            yield frame_id, frame


def load_cache(seq_dir: Path, get_detector, cache_dir: Path, max_frames: int | None = None):
    path = cache_dir / f"{seq_dir.name}.npz"
    if path.exists():
        z = np.load(path)
        return z["dets"], z["warps"]
    dets, warps = cache_sequence(limited(FramePrefetcher(seq_dir), max_frames), get_detector(), ECCCompensator())
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dets=dets, warps=warps)
    return dets, warps


def ultralytics_rows(weights: str, seq_dir: Path, imgsz: int, max_frames: int | None = None,
                     tracker: str = "bytetrack.yaml", conf: float = DET_CONF) -> list[TrackRow]:
    """Ultralytics' own ByteTrack (bytetrack.yaml defaults) on the same weights and NMS settings as ours."""
    import torch
    from ultralytics import YOLO

    cuda = torch.cuda.is_available()
    model = YOLO(weights)                     # fresh instance = fresh tracker state per sequence
    rows: list[TrackRow] = []
    for frame_id, frame in limited(FramePrefetcher(seq_dir), max_frames):
        r = model.track(frame, persist=True, tracker=tracker, imgsz=imgsz, conf=conf,
                        iou=DET_IOU, max_det=1000, device=0 if cuda else "cpu", verbose=False,
                        **({"quantize": 16} if cuda else {}))[0]    # fp16 on GPU, like our Detector
        if r.boxes is None or r.boxes.id is None:
            continue
        for (x1, y1, x2, y2), s, c, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                                r.boxes.cls.cpu().numpy(), r.boxes.id.cpu().numpy()):
            rows.append(TrackRow(frame_id, int(tid), x1, y1, x2, y2, float(s), group_of(int(c))))
    return rows


def prepare_gt(mot_dir: Path, seqs: list[str], gt_root: Path, max_frames: int | None = None) -> dict:
    """Write per-group GT + seqinfo/seqmap; returns each sequence's ignore regions."""
    shutil.rmtree(gt_root, ignore_errors=True)
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


def table(results: dict[str, dict[str, dict]], names: dict[str, str]) -> str:
    out = []
    for group, res in results.items():
        out += [f"**{group}**", "", "| Tracker | HOTA | MOTA | IDF1 | ID switches |", "|---|---|---|---|---|"]
        order = [k for k in TABLE_ORDER if k in res] + [k for k in res if k not in TABLE_ORDER]
        out += [f"| {names.get(k, k)} | {res[k]['HOTA']:.1f} | {res[k]['MOTA']:.1f} | {res[k]['IDF1']:.1f} "
                f"| {res[k]['IDSW']} |" for k in order]
        out.append("")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", default="models/weights/yolo26s-p2-visdrone.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--split", type=Path, default=Path("data/visdrone/VisDrone2019-MOT-test-dev"))
    p.add_argument("--tracker-config", default="configs/tracker.yaml")
    p.add_argument("--out", type=Path, default=Path("outputs/eval/mot"))
    p.add_argument("--tune", action="store_true", help="grid search (use with --split ...MOT-val)")
    p.add_argument("--grid-small", action="store_true", help="with --tune: a 2-variant grid for smoke tests")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--max-seqs", type=int, default=None, help="use only the first N sequences")
    p.add_argument("--max-frames", type=int, default=None, help="use only the first N frames of each sequence")
    args = p.parse_args()

    seqs = sorted(d.name for d in (args.split / "sequences").iterdir() if d.is_dir())[:args.max_seqs]
    gt_root, trk_root = args.out / "gt", args.out / "trackers"
    shutil.rmtree(trk_root, ignore_errors=True)
    ignores = prepare_gt(args.split, seqs, gt_root, args.max_frames)

    detector = None

    def get_detector() -> Detector:
        nonlocal detector
        if detector is None:
            detector = Detector(args.weights, DetectorSettings(imgsz=args.imgsz, conf=DET_CONF, iou=DET_IOU))
        return detector

    w = Path(args.weights)
    stamp = f"{w.stat().st_size}_{int(w.stat().st_mtime)}" if w.exists() else "na"   # retrained weights -> new cache
    cache_dir = (PROJECT_ROOT / "outputs" / "cache"
                 / f"{w.stem}_{stamp}_{args.imgsz}_{args.split.name}_f{args.max_frames or 'all'}")
    caches = {s: load_cache(args.split / "sequences" / s, get_detector, cache_dir, args.max_frames)
              for s in tqdm(seqs, desc="detect")}
    base_cfg = TrackerConfig.from_yaml(args.tracker_config)

    if args.tune:
        grid = GRID_SMALL if args.grid_small else GRID
        variants = {}
        for values in itertools.product(*grid.values()):
            cfg = replace(base_cfg, **dict(zip(grid, values)))
            variants["_".join(f"{k}{v}" for k, v in zip(grid, values))] = cfg
    else:
        variants = {"ours_cmc": base_cfg, "ours_nocmc": base_cfg}

    for name, cfg in tqdm(variants.items(), desc="track"):
        for seq in seqs:
            rows = track_cached(*caches[seq], ByteTracker(cfg), use_cmc=(name != "ours_nocmc"))
            write_rows(trk_root, name, seq, rows, ignores[seq])
    if not args.tune and not args.no_baseline:
        for seq in tqdm(seqs, desc="ultralytics bytetrack"):
            write_rows(trk_root, "ultralytics_bytetrack", seq,
                       ultralytics_rows(args.weights, args.split / "sequences" / seq, args.imgsz, args.max_frames),
                       ignores[seq])

    names = sorted(p.name for p in (trk_root / "person-test").iterdir())
    results = {g: run_trackeval(gt_root, trk_root, g, names) for g in EVAL_GROUPS}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mot_metrics.json").write_text(json.dumps(results, indent=2))

    if args.tune:
        score = {n: np.mean([results[g][n]["HOTA"] for g in EVAL_GROUPS]) for n in names}
        best = max(score, key=score.get)
        lines = ["| Variant | mean HOTA |", "|---|---|"] + [
            f"| {n} | {score[n]:.2f} |" for n in sorted(score, key=score.get, reverse=True)]
        (args.out / "tuning.md").write_text("\n".join(lines) + f"\n\nBest: {best}\n{asdict(variants[best])}\n")
        print("\n".join(lines[:8]), f"\nBEST {best}: {asdict(variants[best])}")
    else:
        md = table(results, TABLE_NAMES)
        (args.out / "mot_table.md").write_text(md)
        print(md)


if __name__ == "__main__":
    main()

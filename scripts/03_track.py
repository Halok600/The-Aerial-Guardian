#!/usr/bin/env python3
"""
scripts/03_track.py — The Aerial Guardian: detect + track + render a video.

Pipeline per frame: YOLO26-P2 (low conf, 10 VisDrone classes) -> ECC camera
motion compensation -> ByteTrack v3 -> render (class-coloured boxes,
"car ID:12" labels, trails, per-group unique counts).

Usage
-----
  python scripts/03_track.py --weights models/weights/yolo26s-p2-visdrone.pt \\
      --source data/visdrone/VisDrone2019-MOT-test-dev/sequences/uav0000009_03358_v
  python scripts/03_track.py --weights ... --source clip.mp4 --no-cmc
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Detector, DetectorSettings, run_tracking  # noqa: E402
from src.tracker.byte_tracker import ByteTracker, TrackerConfig  # noqa: E402
from src.tracker.ecc_compensator import ECCCompensator  # noqa: E402
from src.utils.classes import GROUP_LABELS, GROUP_ORDER  # noqa: E402
from src.utils.video_io import AsyncWriter, FramePrefetcher  # noqa: E402
from src.utils.visualizer import TrackVisualizer  # noqa: E402


def run(args: argparse.Namespace) -> Path:
    source = Path(args.source)
    src = FramePrefetcher(source, buffer_size=8)
    out_path = Path(args.out) if args.out else PROJECT_ROOT / "outputs" / "videos" / f"{source.stem}_tracked.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    detector = Detector(args.weights, DetectorSettings(imgsz=args.imgsz, conf=args.det_conf))
    tracker = ByteTracker(TrackerConfig.from_yaml(args.tracker_config), frame_rate=src.fps_native)
    cmc = None if args.no_cmc else ECCCompensator()
    renderer = TrackVisualizer(trail_len=args.trail_len)
    writer = AsyncWriter(out_path, src.fps_native, (src.width, src.height))

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    times: list[float] = []
    t0 = time.perf_counter()
    try:
        for frame_id, frame, tracks, dets in run_tracking(src, detector, tracker, cmc):
            if stop["flag"] or (args.max_frames and frame_id > args.max_frames):
                break
            t1 = time.perf_counter()
            times.append(t1 - t0)
            t0 = t1
            window = times[-30:]
            fps = len(window) / sum(window) if sum(window) > 0 else 0.0
            writer.write(renderer.draw(frame, tracks, tracker.unique_counts(), fps, frame_id))
            if frame_id == 1 or frame_id % 30 == 0:
                print(f"  frame {frame_id:5d}/{src.total_frames} | fps {fps:5.1f} | "
                      f"dets {len(dets):4d} | active {len(tracks):3d} | ids {tracker.total_ids}")
    finally:
        writer.release()

    counts = tracker.unique_counts()
    print("\nUnique objects: " + ", ".join(f"{GROUP_LABELS[g]} {counts[g]}" for g in GROUP_ORDER))
    print(f"Output video  : {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="The Aerial Guardian — detect, track and render.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", default="models/weights/yolo26s-p2-visdrone.pt")
    p.add_argument("--source", required=True, help="Video file or folder of frames.")
    p.add_argument("--out", default=None, help="Output .mp4 (default outputs/videos/<name>_tracked.mp4)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--det-conf", type=float, default=0.05)
    p.add_argument("--tracker-config", default=str(PROJECT_ROOT / "configs" / "tracker.yaml"))
    p.add_argument("--trail-len", type=int, default=40)
    p.add_argument("--no-cmc", action="store_true", help="Disable ECC camera-motion compensation.")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""
scripts/07_make_showcase.py — Render demo clips for the site/README.

OpenCV writes mp4v, which browsers won't play, so every clip is re-encoded to
H.264 (yuv420p, faststart) with the ffmpeg binary bundled by imageio-ffmpeg.

  python scripts/07_make_showcase.py --sequence <MOT seq dir> --name demo1 --gif
  python scripts/07_make_showcase.py --sequence <MOT seq dir> --name sample-raw --raw
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tests.helpers import load_script  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
DEMO = PROJECT_ROOT / "web" / "assets" / "demo"
MEDIA = PROJECT_ROOT / "docs" / "media"
FPS = 30  # VisDrone MOT sequences are image folders shot at 30 fps
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def list_frames(seq: Path) -> list[Path]:
    """Frame images of an image-sequence folder, in playback (name) order."""
    if not seq.is_dir():
        raise FileNotFoundError(f"Sequence folder not found: {seq}")
    frames = sorted(p for p in seq.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not frames:
        raise FileNotFoundError(f"No .jpg/.png frames in {seq}")
    return frames


def run_ffmpeg(args: list[str]) -> None:
    """Run the bundled ffmpeg; on failure raise with ffmpeg's own message."""
    cmd = [FFMPEG, "-y", "-loglevel", "error", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}")


def h264_args(src: Path, dst: Path, width: int = 1280) -> list[str]:
    # Never upscale; keep the width even (required by yuv420p).
    scale = f"scale='trunc(min({width},iw)/2)*2':-2"
    return ["-i", str(src), "-vf", scale, "-c:v", "libx264", "-crf", "28", "-preset", "slow",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]


def h264(src: Path, dst: Path, width: int = 1280) -> None:
    run_ffmpeg(h264_args(src, dst, width))


def gif_vf(width: int = 600, fps: int = 10, max_colors: int = 64, bayer_scale: int = 5) -> str:
    """ffmpeg filter graph for the README GIF. With gif()'s 6 s, these defaults reproduce the
    committed docs/media/hero.gif from demo1.mp4 (about 5 MB, under the 6 MB budget)."""
    return (f"fps={fps},scale='min({width},iw)':-1:flags=lanczos,split[a][b];"
            f"[a]palettegen=max_colors={max_colors}[p];"
            f"[b][p]paletteuse=dither=bayer:bayer_scale={bayer_scale}")


def gif(src: Path, dst: Path, seconds: float = 6.0, width: int = 600, fps: int = 10,
        max_colors: int = 64, bayer_scale: int = 5) -> None:
    vf = gif_vf(width, fps, max_colors, bayer_scale)
    run_ffmpeg(["-t", str(seconds), "-i", str(src), "-vf", vf, str(dst)])


def frames_to_mp4(seq: Path, out: Path, max_frames: int) -> None:
    """Encode an image sequence (first max_frames frames; 0 = all) as a web-playable mp4."""
    frames = list_frames(seq)
    if max_frames > 0:
        frames = frames[:max_frames]
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Cannot read frame {frames[0]}")
    h, w = first.shape[:2]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {tmp} (size={w}x{h})")
    try:
        for p in frames:
            img = cv2.imread(str(p))
            if img is None:
                raise RuntimeError(f"Cannot read frame {p}")
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
        writer.release()
        h264(tmp, out)
    finally:
        writer.release()
        tmp.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render showcase clips for the site and README.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sequence", type=Path, required=True, help="Folder of frames (MOT sequence).")
    p.add_argument("--name", required=True, help="Output name, written as <name>.mp4.")
    p.add_argument("--weights", default="models/weights/yolo26s-p2-visdrone.pt")
    p.add_argument("--max-frames", type=int, default=450)
    p.add_argument("--raw", action="store_true", help="untracked clip for the live demo's sample button")
    p.add_argument("--gif", action="store_true", help="also write hero.gif (6 s) from the clip")
    p.add_argument("--demo-dir", type=Path, default=DEMO, help="Where the mp4 is written.")
    p.add_argument("--media-dir", type=Path, default=MEDIA, help="Where hero.gif is written.")
    return p


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if not NAME_RE.match(args.name):
        raise SystemExit(f"--name must be a plain file stem (letters, digits, - and _), got {args.name!r}")
    list_frames(args.sequence)  # fail fast on a bad sequence, before any heavy work
    args.demo_dir.mkdir(parents=True, exist_ok=True)
    out = args.demo_dir / f"{args.name}.mp4"

    if args.raw:
        frames_to_mp4(args.sequence, out, args.max_frames)
    else:
        weights = Path(args.weights)
        if not weights.is_absolute() and not weights.exists():
            weights = PROJECT_ROOT / weights
        if not weights.exists():
            raise SystemExit(f"Weights not found: {args.weights}")
        render = out.with_suffix(".render.mp4")  # mp4v intermediate, removed after re-encoding
        track = load_script("03_track.py")
        try:
            rendered = track.run(SimpleNamespace(
                weights=str(weights), source=str(args.sequence), out=str(render), imgsz=1280, det_conf=0.05,
                tracker_config=str(PROJECT_ROOT / "configs" / "tracker.yaml"), trail_len=40,
                no_cmc=False, max_frames=args.max_frames))
            h264(rendered, out)
        finally:
            render.unlink(missing_ok=True)
    if args.gif:
        args.media_dir.mkdir(parents=True, exist_ok=True)
        gif(out, args.media_dir / "hero.gif")
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    main()

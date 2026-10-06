"""src/utils/video_io.py — Threaded frame reading and video writing.

Profiling showed the GPU loop idled ~50ms/frame waiting on cv2.imread and
VideoWriter.write; a prefetch thread and an async writer keep it busy.
"""
import queue
import threading
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Threaded Frame Prefetcher
# Reads frames from disk in a background thread so the GPU loop never waits
# for cv2.imread (which blocks on JPEG decompression for ~8-15ms per frame).
# ---------------------------------------------------------------------------

class FramePrefetcher:
    """
    Reads frames ahead of the inference loop in a daemon thread.
    Maintains a fixed-size buffer so memory stays bounded.
    Yields (frame_id, bgr_ndarray) tuples.
    """

    def __init__(self, source: Path, buffer_size: int = 8):
        self._q: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.width = self.height = 0
        self.fps_native = 30.0
        self.total_frames = 0
        self._img_paths: list = []
        self._cap = None
        self._done = False

        if source.is_dir():
            self._img_paths = sorted(
                list(source.glob("*.jpg")) + list(source.glob("*.png"))
            )
            if not self._img_paths:
                raise FileNotFoundError(f"No images in {source}")
            sample = cv2.imread(str(self._img_paths[0]))
            self.height, self.width = sample.shape[:2]
            self.total_frames = len(self._img_paths)
        elif source.is_file():
            self._cap = cv2.VideoCapture(str(source))
            if not self._cap.isOpened():
                raise FileNotFoundError(f"Cannot open: {source}")
            self.width      = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height     = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps_native = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        if self._cap is not None:
            idx = 0
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                idx += 1
                self._q.put((idx, frame))
            self._cap.release()
        else:
            for idx, p in enumerate(self._img_paths, 1):
                frame = cv2.imread(str(p))
                self._q.put((idx, frame))
        self._done = True
        self._q.put(None)  # sentinel

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item


# ---------------------------------------------------------------------------
# Threaded Video Writer
# Compresses and writes frames in a background thread so the GPU loop
# never blocks waiting for VideoWriter (JPEG encoding takes ~8-12ms).
# ---------------------------------------------------------------------------

class AsyncWriter:
    """Writes annotated frames to MP4 in a background thread."""

    def __init__(self, path: Path, fps: float, size: tuple):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        self._q: queue.Queue = queue.Queue(maxsize=16)
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self):
        while True:
            frame = self._q.get()
            if frame is None:
                break
            self._writer.write(frame)

    def write(self, frame: np.ndarray):
        self._q.put(frame.copy())  # copy to avoid mutation while writing

    def release(self):
        self._q.put(None)   # sentinel
        self._thread.join(timeout=30)
        self._writer.release()

"""src/utils/video_io.py — Threaded frame reading and video writing.

Profiling showed the GPU loop idled ~50ms/frame waiting on cv2.imread and
VideoWriter.write; a prefetch thread and an async writer keep it busy.
"""
import math
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

DEFAULT_FPS = 30.0
MAX_SANE_FPS = 240.0


def sane_fps(value) -> float:
    """Container-reported frame rate, or 30 when it is missing, NaN/inf, non-positive or absurd (> 240)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return DEFAULT_FPS
    return v if math.isfinite(v) and 0 < v <= MAX_SANE_FPS else DEFAULT_FPS


class FramePrefetcher:
    """
    Reads frames ahead of the inference loop in a daemon thread.
    Maintains a fixed-size buffer so memory stays bounded.
    Yields (frame_id, bgr_ndarray) tuples.
    """

    _END = object()  # end-of-stream marker; any stored error is raised when it is reached

    def __init__(self, source: Path, buffer_size: int = 8):
        self._q: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.width = self.height = 0
        self.fps_native = 30.0
        self.total_frames = 0
        self._img_paths: list = []
        self._cap = None
        self._done = False
        self._stop = threading.Event()
        self._error: BaseException | None = None

        if source.is_dir():
            self._img_paths = sorted(
                list(source.glob("*.jpg")) + list(source.glob("*.png"))
            )
            if not self._img_paths:
                raise FileNotFoundError(f"No images in {source}")
            sample = cv2.imread(str(self._img_paths[0]))
            if sample is None:
                raise ValueError(f"Cannot read image: {self._img_paths[0]}")
            self.height, self.width = sample.shape[:2]
            self.total_frames = len(self._img_paths)
        elif source.is_file():
            self._cap = cv2.VideoCapture(str(source))
            if not self._cap.isOpened():
                raise FileNotFoundError(f"Cannot open: {source}")
            self.width      = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height     = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps_native = sane_fps(self._cap.get(cv2.CAP_PROP_FPS))
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _put(self, item) -> bool:
        """Blocking put that gives up once close() has been called; False if abandoned."""
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _read_loop(self):
        try:
            if self._cap is not None:
                idx = 0
                while not self._stop.is_set():
                    ok, frame = self._cap.read()
                    if not ok:
                        break
                    idx += 1
                    if not self._put((idx, frame)):
                        return
            else:
                for idx, p in enumerate(self._img_paths, 1):
                    frame = cv2.imread(str(p))
                    if frame is None:
                        raise ValueError(f"Cannot read image: {p}")
                    if not self._put((idx, frame)):
                        return
        except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
            self._error = e
        finally:
            if self._cap is not None:
                self._cap.release()
            self._done = True
            self._put(self._END)

    def close(self) -> None:
        """Stop the reader thread, drop buffered frames and release the video file."""
        self._stop.set()
        self._drain()
        self._thread.join(timeout=2.0)
        self._drain()
        if self._cap is not None and not self._thread.is_alive():
            self._cap.release()

    def _drain(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def __enter__(self) -> "FramePrefetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    raise StopIteration
                continue
            if item is self._END:
                err, self._error = self._error, None
                if err is not None:
                    raise err
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
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {path} (size={size}, fps={fps})")
        self._size = (int(size[0]), int(size[1]))  # (width, height)
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
        if frame.shape[:2] != (self._size[1], self._size[0]):
            raise ValueError(f"Frame is {frame.shape[1]}x{frame.shape[0]} but the writer was "
                             f"opened for {self._size[0]}x{self._size[1]}")
        self._q.put(frame.copy())  # copy to avoid mutation while writing

    def release(self):
        self._q.put(None)   # sentinel
        self._thread.join(timeout=30)
        self._writer.release()

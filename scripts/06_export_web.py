#!/usr/bin/env python3
"""
scripts/06_export_web.py — Export the browser detector to ONNX + metadata.

YOLO26 heads can be exported two ways, and web/js/inference.js has to decode
whichever one it is given, so the layout is detected from the exported graph
and recorded in model-meta.json next to the model:

  end2end  [1, K, 6]    NMS-free top-K rows (x1, y1, x2, y2, score, cls) in
                        letterboxed input pixels (0..imgsz), K = max_det = 300.
                        Needs `nms=False` at export: Ultralytics only switches
                        the head to its one-to-one branch when nms is exactly
                        False — the default (None) exports the raw head.
  raw      [1, 4+nc, N] (cx, cy, w, h, class scores...) per anchor; the browser
                        must run NMS itself. N = 76500 for a 960 P2 model.

After exporting, ONNX Runtime is compared against the PyTorch weights on a few
real images (raw output tensors, plus Ultralytics prediction counts) so a bad
export fails loudly instead of silently shipping to the demo.

  python scripts/06_export_web.py --weights models/weights/yolo26n-p2-visdrone.pt
  python scripts/06_export_web.py --weights ... --out-dir outputs/web_dev_model   # dry run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.classes import VISDRONE_NAMES  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "web" / "model"
ONNX_NAME = "aerial-guardian.onnx"
META_NAME = "model-meta.json"
FORMATS = ("end2end", "raw")

# Parity tolerances (PyTorch fused model vs ONNX Runtime CPU, fp32).
SCORE_TOL = 1e-4   # sigmoid scores (fp32 runtimes agree to ~1e-6)
BOX_TOL_PX = 0.5   # box coordinates, letterboxed input pixels
TOP_K = 50         # end2end rows compared per image
MIN_MATCH = 1.0    # fraction of compared end2end rows that must match


def detect_output_format(output_shape) -> str:
    """Classify an ONNX output shape as 'end2end' ([B, K, 6]) or 'raw' ([B, 4+nc, N]).

    Dynamic dims arrive as strings or None from onnxruntime and are ignored; only
    the last dim matters. A raw head always has at least 4 box + 1 class rows in
    dim 1 and its last dim is the anchor count (thousands), never exactly 6.
    """
    shape = list(output_shape)
    if len(shape) != 3:
        raise ValueError(f"expected a rank-3 detection output, got shape {shape}")
    return "end2end" if shape[2] == 6 else "raw"


def build_meta(input_size: int, output_format: str, weights: str, ultralytics_version: str,
               names=VISDRONE_NAMES) -> dict:
    """The model-meta.json contract read by web/js/inference.js."""
    if output_format not in FORMATS:
        raise ValueError(f"output_format must be one of {FORMATS}, got {output_format!r}")
    return {
        "input_size": int(input_size),
        "output_format": output_format,
        "names": list(names),
        "source_weights": Path(weights).name,
        "exported_with": f"ultralytics {ultralytics_version}",
    }


def export_onnx(weights: Path, imgsz: int, layout: str, opset: int, simplify: bool,
                work_dir: Path) -> Path:
    """Export `weights` to ONNX inside `work_dir` (Ultralytics writes next to the .pt)."""
    from ultralytics import YOLO

    staged = work_dir / weights.name
    shutil.copy2(weights, staged)
    kwargs = dict(format="onnx", imgsz=imgsz, opset=opset, simplify=simplify, dynamic=False)
    if layout == "end2end":
        kwargs["nms"] = False  # NMS-free one-to-one head; nms=None would export the raw head
    return Path(YOLO(str(staged)).export(**kwargs))


def _torch_output(weights: Path, layout: str, chw: np.ndarray, imgsz: int) -> np.ndarray:
    """Run the PyTorch weights the way the exporter does (fused, same head mode)."""
    import copy

    import torch
    from ultralytics import YOLO

    model = copy.deepcopy(YOLO(str(weights)).model).float().eval()
    if hasattr(model, "end2end"):
        model.end2end = layout == "end2end"
    model.fuse(verbose=False)
    head = model.model[-1]
    head.export = True  # return the decoded tensor only, like the exported graph
    head.max_det = min(300, int(sum((imgsz // s) ** 2 for s in model.stride.tolist())))  # exporter default K
    with torch.no_grad():
        out = model(torch.from_numpy(chw))
    out = out[0] if isinstance(out, (tuple, list)) else out
    return out.numpy()


def compare_outputs(torch_out: np.ndarray, onnx_out: np.ndarray, layout: str) -> tuple[float, float, float]:
    """(max |box diff| px, max |score diff|, matched fraction) between PyTorch and ONNX outputs.

    raw: the whole [1, 4+nc, N] tensors are compared elementwise (fraction is 1.0).
    end2end: the rows are a top-K selection and near-tied scores (an early or weak
    model scores almost everything alike) can swap order or straddle the K cut-off
    between runtimes, so rows are matched by content instead of position: each of
    the top-TOP_K rows of one side must have a same-class row on the other side with
    score within SCORE_TOL and boxes within BOX_TOL_PX. Rows scoring at or below the
    other side's K-th (lowest) score are exempt: either side may have dropped them.
    The fraction of matched rows is returned for the caller to gate.
    """
    if layout == "raw":
        d = np.abs(torch_out - onnx_out)
        return float(d[:, :4].max()), float(d[:, 4:].max()), 1.0

    box_d = score_d = 0.0
    checked = matched = 0
    for a, b in ((torch_out[0], onnx_out[0]), (onnx_out[0], torch_out[0])):
        cutoff = b[:, 4].min()
        for row in a[np.argsort(-a[:, 4], kind="stable")[:TOP_K]]:
            if row[4] <= cutoff:
                continue
            checked += 1
            cand = b[(b[:, 5] == row[5]) & (np.abs(b[:, 4] - row[4]) <= SCORE_TOL)]
            if len(cand) == 0:
                continue
            dists = np.abs(cand[:, :4] - row[:4]).max(1)
            best = dists.argmin()
            if dists[best] > BOX_TOL_PX:
                continue
            matched += 1
            box_d = max(box_d, float(dists[best]))
            score_d = max(score_d, float(abs(cand[best, 4] - row[4])))
    return box_d, score_d, (matched / checked if checked else 1.0)


def parity_check(weights: Path, onnx_path: Path, layout: str, imgsz: int, images: list[Path]) -> None:
    """Fail (AssertionError) when ONNX Runtime and PyTorch disagree on real images."""
    import cv2
    import onnxruntime as ort
    from ultralytics import YOLO

    from src.eval.onnx_person import letterbox

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    pt, ox = YOLO(str(weights)), YOLO(str(onnx_path), task="detect")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        chw, *_ = letterbox(img, imgsz)
        d_box, d_score, frac = compare_outputs(_torch_output(weights, layout, chw, imgsz),
                                         sess.run(None, {in_name: chw})[0], layout)
        counts = []
        for conf in (0.25, 0.001):  # the weak/early models fire almost nothing at 0.25
            a = len(pt.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0].boxes)
            b = len(ox.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0].boxes)
            counts.append((conf, a, b))
        print(f"  {img_path.name}: max|d box|={d_box:.4f}px max|d score|={d_score:.2e} matched={frac:.0%}  "
              + "  ".join(f"conf{c}: pt={a} onnx={b}" for c, a, b in counts))
        assert frac >= MIN_MATCH and d_score <= SCORE_TOL and d_box <= BOX_TOL_PX, (
            f"ONNX output disagrees with PyTorch on {img_path.name} "
            f"(box {d_box:.4f}px, score {d_score:.2e}, matched {frac:.0%})")
        for conf, a, b in counts:
            assert abs(a - b) <= max(2, 0.1 * a), (
                f"ONNX/PyTorch detection counts differ on {img_path.name} at conf {conf}: pt={a} onnx={b}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", default="models/weights/yolo26n-p2-visdrone.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--layout", choices=FORMATS, default="end2end",
                   help="requested head layout; the layout actually exported is detected and recorded")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-simplify", dest="simplify", action="store_false",
                   help="skip onnxslim (it is also skipped, with a warning, when onnxslim is not installed)")
    p.add_argument("--sample", default="data/visdrone/VisDrone2019-DET-val/images",
                   help="folder of images for the parity sanity check")
    p.add_argument("--n-images", type=int, default=5)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR.relative_to(PROJECT_ROOT)),
                   help="where to write aerial-guardian.onnx and model-meta.json")
    args = p.parse_args()

    import onnxruntime as ort
    import ultralytics

    weights = Path(args.weights)
    out_dir = Path(args.out_dir)
    images = sorted(Path(args.sample).glob("*.jpg"))[:args.n_images]
    if not images:
        sys.exit(f"no .jpg images found in {args.sample} for the parity check")
    if args.simplify:
        try:
            import onnxslim  # noqa: F401
        except ImportError:
            print("warning: onnxslim is not installed; Ultralytics will export WITHOUT simplification")

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(weights, args.imgsz, args.layout, args.opset, args.simplify, Path(tmp))
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out_shape = sess.get_outputs()[0].shape
        fmt = detect_output_format(out_shape)
        print(f"ONNX output {out_shape} -> {fmt}; size {onnx_path.stat().st_size / 1e6:.1f} MB")
        if fmt != args.layout:
            print(f"warning: requested layout {args.layout} but the exported graph is {fmt}")

        print(f"Parity check on {len(images)} images (PyTorch vs ONNX Runtime CPU):")
        parity_check(weights, onnx_path, fmt, args.imgsz, images)

        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, out_dir / ONNX_NAME)

    meta = build_meta(args.imgsz, fmt, str(weights), ultralytics.__version__)
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"-> {out_dir / ONNX_NAME}\n-> {out_dir / META_NAME}")


if __name__ == "__main__":
    main()

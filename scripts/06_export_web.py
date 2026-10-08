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
import importlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

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
UNSAFE_WEIGHT_WORDS = ("smoke", "dev")  # weights that must never overwrite the committed web model


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


def simplify_status(simplify_requested: bool, onnxslim_available: bool) -> str:
    """'fail' when simplification was requested but onnxslim is missing.

    Ultralytics turns an onnxslim failure into a warning and still writes an
    unsimplified model, so the script has to check after the export (never before:
    with autoinstall on, Ultralytics installs onnxslim itself during the export).
    """
    return "fail" if simplify_requested and not onnxslim_available else "ok"


def layout_problems(requested: str, detected: str, metadata_end2end: str | None) -> list[str]:
    """Reasons the exported graph cannot be trusted as `requested`; empty list when consistent.

    `detected` comes from the output shape; `metadata_end2end` is the value of the
    `end2end` entry Ultralytics writes into the ONNX metadata_props ('True'/'False').
    """
    problems = []
    if detected != requested:
        problems.append(f"requested {requested} but the exported graph is {detected}")
    if metadata_end2end is None:
        problems.append("the ONNX metadata has no 'end2end' entry to cross-check the output shape against")
    elif (str(metadata_end2end).lower() == "true") != (detected == "end2end"):
        problems.append(f"the output shape says {detected} but the ONNX metadata says end2end={metadata_end2end}")
    return problems


def out_dir_problem(weights, out_dir, force: bool) -> str | None:
    """Refuse to put a smoke/dev model into the committed web/model unless forced."""
    name = Path(weights).name.lower()
    unsafe = [w for w in UNSAFE_WEIGHT_WORDS if w in name]
    if force or not unsafe:
        return None
    if Path(out_dir).resolve() != DEFAULT_OUT_DIR.resolve():
        return None
    return (f"refusing to write '{Path(weights).name}' (looks like a {unsafe[0]} model) into the committed "
            f"{DEFAULT_OUT_DIR}; pass --out-dir elsewhere, or --force if this is really intended")


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


class ParityResult(NamedTuple):
    box_diff: float         # max |box diff| in px over matched rows (raw: over the whole tensor)
    score_diff: float       # max |score diff| over matched rows (raw: over the whole tensor)
    matched_frac: float     # fraction of compared end2end rows that matched; 0.0 if none could be compared
    n_unmatched: int
    worst_unmatched: float  # nearest-box distance (px) of the worst unmatched row; inf = no counterpart


def compare_outputs(torch_out: np.ndarray, onnx_out: np.ndarray, layout: str) -> ParityResult:
    """Compare PyTorch and ONNX outputs of one image.

    raw: the whole [1, 4+nc, N] tensors are compared elementwise (fraction is 1.0).
    end2end: the rows are a top-K selection and near-tied scores (an early or weak
    model scores almost everything alike) can swap order or straddle the K cut-off
    between runtimes, so rows are matched by content instead of position: each of
    the top-TOP_K rows of one side must have a same-class row on the other side with
    score within SCORE_TOL and boxes within BOX_TOL_PX. Rows scoring at or below the
    other side's K-th (lowest) score are exempt: either side may have dropped them.
    If every row is exempt nothing was verified, so the fraction is 0.0 (a failure).
    """
    if layout == "raw":
        d = np.abs(torch_out - onnx_out)
        return ParityResult(float(d[:, :4].max()), float(d[:, 4:].max()), 1.0, 0, 0.0)

    box_d = score_d = worst = 0.0
    checked = matched = 0
    for a, b in ((torch_out[0], onnx_out[0]), (onnx_out[0], torch_out[0])):
        cutoff = b[:, 4].min()
        for row in a[np.argsort(-a[:, 4], kind="stable")[:TOP_K]]:
            if row[4] <= cutoff:
                continue
            checked += 1
            cand = b[(b[:, 5] == row[5]) & (np.abs(b[:, 4] - row[4]) <= SCORE_TOL)]
            if len(cand) == 0:
                worst = float("inf")
                continue
            dists = np.abs(cand[:, :4] - row[:4]).max(1)
            best = dists.argmin()
            if dists[best] > BOX_TOL_PX:
                worst = max(worst, float(dists[best]))
                continue
            matched += 1
            box_d = max(box_d, float(dists[best]))
            score_d = max(score_d, float(abs(cand[best, 4] - row[4])))
    return ParityResult(box_d, score_d, matched / checked if checked else 0.0, checked - matched, worst)


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
        res = compare_outputs(_torch_output(weights, layout, chw, imgsz),
                              sess.run(None, {in_name: chw})[0], layout)
        counts = []
        # A freshly loaded checkpoint predicts with its one-to-many head + NMS, while the
        # end2end ONNX runs the one-to-one head; predict(nms=False) selects that same head.
        # rect=False: pytorch predict otherwise letterboxes to the minimal rectangle (960x544) while the
        # static ONNX and the browser use the full 960x960 square, which changes edge activations.
        pt_kwargs = {"nms": False} if layout == "end2end" else {}
        for conf in (0.25, 0.5, 0.001):  # weak/early models fire almost nothing at the higher ones
            a = len(pt.predict(str(img_path), imgsz=imgsz, conf=conf, device="cpu", rect=False, verbose=False,
                               **pt_kwargs)[0].boxes)
            b = len(ox.predict(str(img_path), imgsz=imgsz, conf=conf, device="cpu", rect=False,
                               verbose=False)[0].boxes)
            counts.append((conf, a, b))
        print(f"  {img_path.name}: max|d box|={res.box_diff:.4f}px max|d score|={res.score_diff:.2e} "
              f"matched={res.matched_frac:.0%}  " + "  ".join(f"conf{c}: pt={a} onnx={b}" for c, a, b in counts))
        worst = "has no counterpart" if res.worst_unmatched == float("inf") else f"is {res.worst_unmatched:.2f}px off"
        assert res.matched_frac >= MIN_MATCH and res.matched_frac > 0 and res.score_diff <= SCORE_TOL \
            and res.box_diff <= BOX_TOL_PX, (
            f"ONNX output disagrees with PyTorch on {img_path.name}: matched {res.matched_frac:.0%} of compared "
            f"rows ({res.n_unmatched} unmatched, worst unmatched {worst}), "
            f"box diff {res.box_diff:.4f}px, score diff {res.score_diff:.2e}"
            + (" (nothing could be compared: all rows tie)" if res.matched_frac == 0 and res.n_unmatched == 0 else ""))
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
                   help="do not simplify; without this flag the export FAILS if onnxslim is unavailable")
    p.add_argument("--sample", default="data/visdrone/VisDrone2019-DET-val/images",
                   help="folder of images for the parity sanity check")
    p.add_argument("--n-images", type=int, default=5)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="where to write aerial-guardian.onnx and model-meta.json")
    p.add_argument("--force", action="store_true",
                   help="allow a smoke/dev weights file to be written into the committed web/model")
    args = p.parse_args()

    import onnxruntime as ort
    import ultralytics

    weights = Path(args.weights)
    out_dir = Path(args.out_dir)
    images = sorted(Path(args.sample).glob("*.jpg"))[:args.n_images]
    if not images:
        sys.exit(f"no .jpg images found in {args.sample} for the parity check")
    if msg := out_dir_problem(weights, out_dir, args.force):
        sys.exit(msg)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(weights, args.imgsz, args.layout, args.opset, args.simplify, Path(tmp))
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out_shape = sess.get_outputs()[0].shape
        fmt = detect_output_format(out_shape)
        print(f"ONNX output {out_shape} -> {fmt}; size {onnx_path.stat().st_size / 1e6:.1f} MB")
        import onnx
        meta_props = {m.key: m.value for m in onnx.load(str(onnx_path), load_external_data=False).metadata_props}
        if problems := layout_problems(args.layout, fmt, meta_props.get("end2end")):
            sys.exit("export layout check failed: " + "; ".join(problems))
        importlib.invalidate_caches()  # Ultralytics may have just auto-installed onnxslim
        if simplify_status(args.simplify, importlib.util.find_spec("onnxslim") is not None) == "fail":
            sys.exit("onnxslim is not available, so Ultralytics skipped simplification and exported an "
                     "UNSIMPLIFIED model. Install onnxslim (or allow Ultralytics to auto-install it), "
                     "or pass --no-simplify to export unsimplified on purpose.")

        print(f"Parity check on {len(images)} images (PyTorch vs ONNX Runtime CPU):")
        parity_check(weights, onnx_path, fmt, args.imgsz, images)

        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, out_dir / ONNX_NAME)

    meta = build_meta(args.imgsz, fmt, str(weights), ultralytics.__version__)
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"-> {out_dir / ONNX_NAME}\n-> {out_dir / META_NAME}")


if __name__ == "__main__":
    main()

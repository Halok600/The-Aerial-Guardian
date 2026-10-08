#!/usr/bin/env python3
"""
scripts/04_eval_det.py — Detector evaluation on VisDrone2019-DET-val.

1. New model: Ultralytics val -> mAP50, mAP50-95, per-class AP50 (10 classes).
2. Old vs new, same yardstick: person recall / precision at conf 0.25, IoU 0.5,
   pedestrian+people GT, predictions inside ignored regions dropped.

The evaluation logic lives in functions that take already-loaded, duck-typed
models (anything with .predict(path, imgsz, conf, max_det, verbose) and .val(...)),
so it is unit-tested with fakes and needs no weights.

  python scripts/04_eval_det.py --weights models/weights/yolo26s-p2-visdrone.pt --imgsz 1280
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.visdrone import parse_det_file  # noqa: E402
from src.eval.det_metrics import greedy_tp, keep_outside_ignore  # noqa: E402
from src.pipeline import parse_detections  # noqa: E402

OLD_LABEL = "v1 YOLOv8n-P2 (person only)"
PERSON_GT_CATEGORIES = (1, 2)  # VisDrone pedestrian, people


def person_scores(model, det_val: Path, imgsz: int, conf: float, person_classes: set[int],
                  desc: str = "model") -> dict:
    """Person recall/precision of `model` at `conf` over every annotated DET-val image."""
    det_val = Path(det_val)
    tp = n_gt = n_pred = 0
    for ann in tqdm(sorted((det_val / "annotations").glob("*.txt")), desc=desc):
        boxes = parse_det_file(ann)
        gt = np.array([[b.x, b.y, b.x + b.w, b.y + b.h] for b in boxes
                       if not b.is_ignore and b.category in PERSON_GT_CATEGORIES]).reshape(-1, 4)
        ignore = np.array([[b.x, b.y, b.x + b.w, b.y + b.h] for b in boxes if b.is_ignore]).reshape(-1, 4)
        r = model.predict(str(det_val / "images" / f"{ann.stem}.jpg"), imgsz=imgsz, conf=conf,
                          max_det=1000, verbose=False)[0]
        dets = parse_detections(r)  # (N, 6) [x1, y1, x2, y2, score, cls]; accepts tensors or arrays
        dets = dets[np.isin(dets[:, 5].astype(int), list(person_classes))]
        dets = dets[keep_outside_ignore(dets[:, :4], ignore)]
        tp += greedy_tp(dets[:, :4], dets[:, 4], gt)
        n_gt += len(gt)
        n_pred += len(dets)
    return {"recall": tp / max(n_gt, 1), "precision": tp / max(n_pred, 1), "tp": tp,
            "gt": n_gt, "pred": n_pred}


def require_person_only(old, path: str = "") -> None:
    """Exit loudly unless `old` is the one-class (person-only) v1 model.

    Scoring any other model with person_classes={0} and the v1 label would silently mislabel it.
    """
    names = getattr(old, "names", None)
    if not names or len(names) != 1:
        n = len(names) if names else 0
        sys.exit(f"--old-onnx {path or '<model>'} has {n} classes; it must be the one-class person-only v1 model "
                 f"(extract it from commit b23778a, see the README) or pass --old-onnx '' to skip the v1 row")


def build_result(new, old, *, weights: str, imgsz: int, data: str, det_val: Path, conf: float,
                 old_onnx: str = "", old_imgsz: int = 960, label: str = "v2 YOLO26s-P2") -> dict:
    """Run mAP validation + person scoring for the new model, and person scoring for `old` if given."""
    m = new.val(data=data, imgsz=imgsz, batch=4, max_det=1000, plots=False, verbose=False)
    result = {
        "new": {
            "label": label, "weights": weights, "imgsz": imgsz,
            "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
            "per_class_AP50": {m.names[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)},
            "person": person_scores(new, det_val, imgsz, conf, {0, 1}, desc=Path(weights).name),
        }
    }
    if old is not None:
        require_person_only(old, old_onnx)
        result["old"] = {"weights": old_onnx, "imgsz": old_imgsz,
                         "person": person_scores(old, det_val, old_imgsz, conf, {0},
                                                 desc=Path(old_onnx).name or "old")}
    return result


def render_table(result: dict) -> str:
    rows = ["| Model | Person recall @0.25 | Person precision @0.25 | mAP50 (10 cls) | mAP50-95 |",
            "|---|---|---|---|---|"]
    if "old" in result:
        o = result["old"]["person"]
        rows.append(f"| {OLD_LABEL} | {o['recall']:.1%} | {o['precision']:.1%} | n/a | n/a |")
    n = result["new"]
    rows.append(f"| {n['label']} @{n['imgsz']} | {n['person']['recall']:.1%} | {n['person']['precision']:.1%} "
                f"| {n['mAP50']:.3f} | {n['mAP50_95']:.3f} |")
    rows += ["", "| Class | AP50 |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in n["per_class_AP50"].items()]
    return "\n".join(rows)


def write_outputs(result: dict, out: Path) -> str:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "det_metrics.json").write_text(json.dumps(result, indent=2))
    table = render_table(result)
    (out / "det_table.md").write_text(table + "\n")
    return table


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", default="models/weights/yolo26s-p2-visdrone.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--label", default="v2 YOLO26s-P2", help="Row label for the new model in the table")
    p.add_argument("--data", default="configs/data/visdrone10.yaml")
    p.add_argument("--det-val", type=Path, default=Path("data/visdrone/VisDrone2019-DET-val"))
    p.add_argument("--old-onnx", default="",
                   help="Previous one-class person-only model, e.g. extracted from commit b23778a "
                        "(see the README); '' (default) skips the old-vs-new row")
    p.add_argument("--old-imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("outputs/eval/det"))
    args = p.parse_args()

    from ultralytics import YOLO

    new = YOLO(args.weights)
    if args.old_onnx and not Path(args.old_onnx).exists():
        sys.exit(f"--old-onnx {args.old_onnx} does not exist")
    old = YOLO(args.old_onnx, task="detect") if args.old_onnx else None
    if old is not None:
        require_person_only(old, args.old_onnx)  # fail before the slow new-model validation
    result = build_result(new, old, weights=args.weights, imgsz=args.imgsz, data=args.data,
                          det_val=args.det_val, conf=args.conf, old_onnx=args.old_onnx,
                          old_imgsz=args.old_imgsz, label=args.label)
    print(write_outputs(result, args.out))


if __name__ == "__main__":
    main()

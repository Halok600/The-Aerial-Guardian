# Detector, YOLO26n-P2 at 960 px (browser model)

Same protocol as [det_s1280.md](det_s1280.md). Produced by `python scripts/04_eval_det.py --weights models/weights/yolo26n-p2-visdrone.pt --imgsz 960 --label "v2 YOLO26n-P2" --out outputs/eval/det_n --old-onnx outputs/old-model.onnx` (`outputs/eval/det_n/det_table.md`). The v1 model is the ONNX file at commit `b23778a`, extracted as described in the [README](../../README.md#reproduce).

| Model | Person recall @0.25 | Person precision @0.25 | mAP50 (10 cls) | mAP50-95 |
|---|---|---|---|---|
| v1 YOLOv8n-P2 (person only) | 2.8% | 55.7% | n/a | n/a |
| v2 YOLO26n-P2 @960 | 48.8% | 73.5% | 0.429 | 0.246 |

| Class | AP50 |
|---|---|
| pedestrian | 0.520 |
| people | 0.406 |
| bicycle | 0.145 |
| car | 0.837 |
| van | 0.471 |
| truck | 0.377 |
| tricycle | 0.299 |
| awning-tricycle | 0.155 |
| bus | 0.584 |
| motor | 0.495 |

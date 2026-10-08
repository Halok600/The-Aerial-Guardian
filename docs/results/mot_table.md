# Tracking, VisDrone2019-MOT-test-dev (17 sequences)

Detections from YOLO26s-P2 at 1280 px, confidence floor 0.05, identical for every row. Sequence-combined TrackEval scores. Groups: person = pedestrian + people; vehicle = car + van + truck + bus. Ground-truth boxes and predictions inside VisDrone ignored regions (IoA > 0.5) are both dropped. Thresholds from `configs/tracker.yaml`, tuned on MOT-val only ([mot_tuning.md](mot_tuning.md)). Per-sequence HOTA: [mot_per_seq.md](mot_per_seq.md). Produced by `python scripts/05_eval_mot.py` (`outputs/eval/mot/mot_table.md`).

**person**

| Tracker | HOTA | MOTA | IDF1 | ID switches |
|---|---|---|---|---|
| Ultralytics ByteTrack (default config) | 34.3 | 11.1 | 41.4 | 2482 |
| Ultralytics ByteTrack (our thresholds) | 35.8 | 27.9 | 45.5 | 1636 |
| Ours (no camera-motion comp.) | 36.1 | 28.2 | 45.9 | 1443 |
| Ours + ECC camera-motion comp. | 40.2 | 28.8 | 52.4 | 577 |

**vehicle**

| Tracker | HOTA | MOTA | IDF1 | ID switches |
|---|---|---|---|---|
| Ultralytics ByteTrack (default config) | 53.8 | 56.2 | 66.7 | 1273 |
| Ultralytics ByteTrack (our thresholds) | 55.9 | 65.5 | 70.0 | 1121 |
| Ours (no camera-motion comp.) | 56.2 | 66.0 | 70.5 | 1046 |
| Ours + ECC camera-motion comp. | 61.7 | 68.1 | 80.4 | 181 |

# The Aerial Guardian 🛸
### Production-Grade Multi-Object Tracking Pipeline for Drone Footage

A production-oriented multi-object tracking pipeline for drone footage. Detects and tracks people (single class: person) using a custom YOLOv8n-P2 detector + ByteTrack. Optimized for aerial, small-person detection at high resolution.

Quick Start (in-project)

Activate the project virtualenv (Windows PowerShell):
& .venv\Scripts\Activate.ps1

Run inference on a sequence folder or a video file:
python scripts/03_track.py --weights models/weights/best.pt --source "path\to\video.mp4" --conf 0.25

Or for an image-sequence directory:
python scripts/03_track.py --weights models/weights/best.pt --source "data\raw\sequences\uav0000086_00000_v" --conf 0.25

Output video is written to outputs/videos/<source>_tracked.mp4
Recommended Command-Line Flags (tuning)

--conf: detector confidence threshold (default tightened to 0.20; raise to 0.25–0.30 to reduce false positives).
--track-thresh and --low-thresh: ByteTrack association buckets (defaults: 0.25 / 0.10). Increase --track-thresh to make tracker require higher-confidence detections.
--no-cmc: disable ECC camera-motion compensation if it causes alignment problems for your footage.
Example stricter run for noisy external videos:
python scripts/03_track.py --weights models/weights/best.pt --source "D:\Projects\dronefootage.mp4" --conf 0.30 --track-thresh 0.35 --low-thresh 0.15
Notes on Performance & Inputs

Accepts either a folder of frames (JPEG/PNG) or a single video file. Frame folders with thousands of images are supported and expected for VisDrone sequences.
The detector uses a single high-res forward pass (imgsz=1280 by default). Use a GPU (CUDA) for realtime or near-realtime performance.
The model only detects the class “person” (trained on VisDrone person annotations). It will not detect vehicles or other object classes.
Why false positives happen (and quick fixes)

Domain shift: model trained on VisDrone (open aerial dataset). City drone footage with different altitude, viewpoint, resolution, or lighting can produce more false positives.
Short-term fixes:
Raise --conf and --track-thresh.
Post-filter using detection size (very small boxes) or aspect ratio if appropriate.
Disable ECC (--no-cmc) to check whether CMC warping introduces artifacts.
Retraining / Improving Accuracy

Best datasets to improve small-person aerial detection:
VisDrone2019 (DET, MOT, VID) — closest match and already used here.
Okutama-Action — additional aerial human examples.
Collect and label a small set of representative city drone clips (strongest improvement for domain-specific false positives).
To prepare data: use 01_preprocess.py (converts VisDrone MOT -> YOLO format) and inspect dataset.yaml.
Train with the included script:
python scripts/02_train.py          # default: 100 epochs, imgsz=1280, batch=4 (VRAM-safe)
# For quick smoke test:
python scripts/02_train.py --epochs 5 --imgsz 640 --batch 8

Recommended hyper-choices are already encoded in 02_train.py (mosaic augmentation, FP16/AMP, AdamW, etc.). Add your city-labeled images into the processed dataset and rerun preprocessing/training.
Project Layout (important files)

03_track.py — inference / main tracking pipeline (accepts video or frame folder).
02_train.py — training entrypoint (YOLOv8n-P2 fine-tune).
01_preprocess.py — VisDrone -> YOLO preprocessing.
dataset.yaml — dataset paths used for training.
model.yaml — custom YOLOv8n-P2 architecture.
best.pt — inference weights (place best checkpoint here).
outputs/videos — annotated video outputs.
Environment & Dependencies

Python 3.10+ recommended.
Key Python packages (examples)
pip install ultralytics==8.0.236
pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow tqdm

Adjust the Torch wheel to match your CUDA version.
Troubleshooting

If the script errors on import, activate the .venv and install dependencies above.
If GPU out-of-memory during training, reduce --batch to 2 or lower --imgsz.
If many false positives remain after thresholding, retrain with mixed VisDrone + your labeled city data.
Next Steps I can help with

Create a small labeling/spec and assist combining VisDrone + your city clips for retraining.
Add a quick post-filter for size/aspect ratio to reduce obvious false positives.
Run a short diagnostic pass on a sample video and produce suggested --conf/--track-thresh values.

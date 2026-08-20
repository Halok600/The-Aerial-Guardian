# The Aerial Guardian 🛸
### Multi-Object Tracking Pipeline for Drone Footage

A person-tracking pipeline for aerial drone video: a custom **YOLOv8n-P2** detector
(an extra stride-4 detection head grafted onto YOLOv8n, tuned for 8–16px targets)
feeding a from-scratch **ByteTrack** implementation (Kalman filter + 3-stage
Hungarian IoU association) with OpenCV **ECC** camera-motion compensation for a
panning drone camera. Trained on VisDrone MOT footage on a laptop RTX 3050.

**Live browser demo (free, no server, no signup):** run the same detector +
tracker entirely client-side via ONNX Runtime Web. See [`web/`](web/) and the
Deployment section below.

---

## Pipeline

```
frame ──▶ YOLOv8n-P2 detector ──▶ ECC camera-motion compensation ──▶ ByteTrack ──▶ render
          (single 1280px pass)     (stabilises Kalman predictions)   (Kalman +      (boxes,
                                                                       Hungarian)     trails, HUD)
```

| Component | File | Notes |
|---|---|---|
| Detector architecture | [`configs/model.yaml`](configs/model.yaml) | YOLOv8n backbone + P2 head → 4 detection scales (P2/P3/P4/P5) |
| Preprocessing | [`scripts/01_preprocess.py`](scripts/01_preprocess.py) | VisDrone MOT → YOLO label format, person class only |
| Training | [`scripts/02_train.py`](scripts/02_train.py) | VRAM-safe defaults for a 6GB GPU (batch=4, imgsz=1280, AMP) |
| Tracking | [`src/tracker/byte_tracker.py`](src/tracker/byte_tracker.py), [`kalman_filter.py`](src/tracker/kalman_filter.py) | ByteTrack with corrected cost-threshold semantics |
| Camera-motion compensation | [`src/tracker/ecc_compensator.py`](src/tracker/ecc_compensator.py) | OpenCV ECC on a downscaled 320×180 frame |
| Rendering | [`src/utils/visualizer.py`](src/utils/visualizer.py) | O(1)-blend trail rendering, golden-angle ID coloring |
| Inference | [`scripts/03_track.py`](scripts/03_track.py) | Threaded I/O pipeline (prefetch + async writer) |
| Browser demo | [`web/`](web/) | ONNX Runtime Web port — detector + tracker, no server |

---

## Local setup (Python pipeline)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install ultralytics==8.0.236 torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-contrib-python scipy tqdm pillow

python scripts/01_preprocess.py                       # VisDrone -> YOLO labels
python scripts/02_train.py                             # train (RTX 3050 6GB defaults)
python scripts/03_track.py --weights models/weights/best.pt --source path/to/video.mp4
```

Outputs land in `outputs/videos/<name>_tracked.mp4`.

---

## Model status — honest assessment

The current checkpoint is genuinely undertrained: **mAP50 ≈ 0.17, recall ≈ 15%**
after ~59 epochs on a small dataset (1,978 train / 275 val frames drawn from 6
VisDrone sequences — the *validation* split of VisDrone-MOT, not its much larger
56-sequence training split). Detecting 8–16px pedestrians from altitude is one of
the hardest regimes in object detection, and this dataset is small for training a
randomly-initialized detection head from scratch.

Training is being continued (see `runs/train/aerial_guardian_p2_v1/`, resumable
via `python scripts/02_train.py --resume`). The two highest-leverage further
improvements, in order:

1. **More data** — swap in the full VisDrone2019-MOT-train split (56 sequences)
   instead of just the val split's 6/1 sequences. This is almost certainly the
   single biggest lever; VisDrone is a well-known public dataset (free to
   download from the [official VisDrone GitHub](https://github.com/VisDrone/VisDrone-Dataset)).
2. **More epochs** — loss was still decreasing with no plateau at epoch 59;
   100+ epochs with `close_mosaic` tapering is standard for small custom heads.

Re-export ONNX for the web demo after any retrain:

```bash
python -c "from ultralytics import YOLO; YOLO('models/weights/best.pt').export(format='onnx', imgsz=960, simplify=True, opset=12)"
cp models/weights/best.onnx web/model/aerial-guardian.onnx
```

---

## Deployment — 100% free, no server

**HuggingFace Spaces' free tier no longer fits this project, so the browser
demo runs the model entirely client-side instead of on a hosted backend.**
The detector is exported to ONNX (12MB) and the ByteTrack + Kalman-filter
tracker is ported to plain JavaScript ([`web/js/tracker.js`](web/js/tracker.js),
[`web/js/inference.js`](web/js/inference.js)) — inference runs in the visitor's
own browser via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
(WebGPU, falling back to WASM). That means:

- **No server to pay for, rent-free forever** — it's a static site (HTML/CSS/JS
  + a 12MB model file). Any static host works.
- **No cold starts, no request limits, no API keys.**
- **Nothing is uploaded** — video/webcam frames never leave the visitor's device.

### Deploy to GitHub Pages (already wired up)

A workflow at [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)
publishes [`web/`](web/) to GitHub Pages on every push to `main`. One-time setup:

1. Push this repo to GitHub (already has a remote: `Halok600/JOB_PROJECT_II_DRONE`).
2. Repo **Settings → Pages → Source → GitHub Actions**.
3. Push to `main` (or run the workflow manually) — the site deploys to
   `https://halok600.github.io/JOB_PROJECT_II_DRONE/`.

### Alternative free static hosts (same `web/` folder, zero config)

- **Vercel** / **Netlify** / **Cloudflare Pages** — free tier, connect the GitHub
  repo, set the root directory to `web/`, no build command needed.
- **Google Colab (temporary/demo only)** — not a real "deployment," but if you
  want a quick shareable link without any static host: open a Colab notebook,
  `!pip install gradio`, wrap `scripts/03_track.py`'s detect+track loop in a
  Gradio `Interface`, and call `.launch(share=True)` for a temporary public
  URL. Useful for a live walkthrough; the link dies when the notebook stops,
  so it's not a substitute for the static site above.

### Local preview

```bash
python -m http.server 8000 --directory web
# open http://localhost:8000
```
(A plain `file://` open won't work — ES modules and the ONNX fetch require HTTP.)

---

## Known limitations of the browser demo

- ECC camera-motion compensation is Python/OpenCV-only; the browser demo runs
  detector + Kalman + ByteTrack without it, so expect more ID switches on
  fast-panning footage than the reference Python clips.
- Inference resolution is fixed at 960×960 (letterboxed) in the browser vs.
  1280×1280 in the Python pipeline, trading a little recall on the smallest
  targets for tractable in-browser latency.
- Speed depends entirely on the visitor's device (WebGPU when available, WASM
  otherwise) — there's no guaranteed frame rate.

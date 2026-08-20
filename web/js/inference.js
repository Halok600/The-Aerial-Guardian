/**
 * inference.js — ONNX Runtime Web wrapper for the YOLOv8n-P2 "Aerial Guardian"
 * detector. Handles letterbox preprocessing, session creation with a
 * WebGPU -> WASM fallback, and NMS postprocessing matching the Python
 * pipeline's parse_detections() contract: output is [cx, cy, w, h, score]
 * in ORIGINAL image pixel coordinates.
 */

const MODEL_URL = new URL('../model/aerial-guardian.onnx', import.meta.url).href;
export const INPUT_SIZE = 960;

let ortModule = null;
async function loadOrt() {
  if (ortModule) return ortModule;
  // Loaded from CDN in index.html as a module; re-exported on window.ort.
  if (!window.ort) throw new Error('onnxruntime-web failed to load from CDN');
  ortModule = window.ort;
  ortModule.env.wasm.numThreads = 1; // avoid COOP/COEP requirement on static hosts
  ortModule.env.wasm.simd = true;
  return ortModule;
}

export class Detector {
  constructor() {
    this.session = null;
    this.backend = null;
  }

  async load(onStatus) {
    const ort = await loadOrt();
    const attempts = [
      { name: 'webgpu', providers: ['webgpu'] },
      { name: 'wasm', providers: ['wasm'] },
    ];
    let lastErr = null;
    for (const attempt of attempts) {
      try {
        onStatus?.(`Initialising ${attempt.name} backend…`);
        this.session = await ort.InferenceSession.create(MODEL_URL, {
          executionProviders: attempt.providers,
          graphOptimizationLevel: 'all',
        });
        this.backend = attempt.name;
        return this.backend;
      } catch (e) {
        lastErr = e;
      }
    }
    throw lastErr || new Error('No available ONNX Runtime backend');
  }

  /**
   * Letterbox-resize a source (canvas/video/image) into a square INPUT_SIZE
   * tensor. Returns { tensor, scale, padX, padY } for un-letterboxing boxes.
   */
  _preprocess(source, srcW, srcH) {
    const size = INPUT_SIZE;
    const scale = Math.min(size / srcW, size / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = Math.floor((size - newW) / 2);
    const padY = Math.floor((size - newH) / 2);

    if (!this._padCanvas) {
      this._padCanvas = document.createElement('canvas');
      this._padCanvas.width = size;
      this._padCanvas.height = size;
      this._padCtx = this._padCanvas.getContext('2d', { willReadFrequently: true });
    }
    const ctx = this._padCtx;
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(source, 0, 0, srcW, srcH, padX, padY, newW, newH);

    const { data } = ctx.getImageData(0, 0, size, size);
    const chw = new Float32Array(3 * size * size);
    const plane = size * size;
    for (let i = 0; i < plane; i++) {
      const o = i * 4;
      chw[i] = data[o] / 255;             // R
      chw[plane + i] = data[o + 1] / 255; // G
      chw[2 * plane + i] = data[o + 2] / 255; // B
    }
    return { data: chw, scale, padX, padY };
  }

  /**
   * Run detection. `source` must be canvas/video/img-like and drawable via
   * drawImage. Returns Array<[cx, cy, w, h, score]> in source pixel space.
   */
  async detect(source, srcW, srcH, { confThresh = 0.25, iouThresh = 0.45 } = {}) {
    const ort = await loadOrt();
    const { data, scale, padX, padY } = this._preprocess(source, srcW, srcH);
    const tensor = new ort.Tensor('float32', data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const feeds = { [this.session.inputNames[0]]: tensor };
    const results = await this.session.run(feeds);
    const output = results[this.session.outputNames[0]];
    // output: [1, 5, N] -> (cx, cy, w, h, score) x N, in INPUT_SIZE model space
    const [, C, N] = output.dims;
    const buf = output.data;

    const boxes = [];
    for (let i = 0; i < N; i++) {
      const score = buf[4 * N + i];
      if (score < confThresh) continue;
      const cx = buf[i];
      const cy = buf[N + i];
      const w = buf[2 * N + i];
      const h = buf[3 * N + i];
      boxes.push({
        x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score,
      });
    }

    const kept = nms(boxes, iouThresh);

    // Un-letterbox back to source pixel space
    return kept.map((b) => {
      const x1 = (b.x1 - padX) / scale;
      const y1 = (b.y1 - padY) / scale;
      const x2 = (b.x2 - padX) / scale;
      const y2 = (b.y2 - padY) / scale;
      const w = x2 - x1, h = y2 - y1;
      return [x1 + w / 2, y1 + h / 2, w, h, b.score];
    });
  }
}

function nms(boxes, iouThresh) {
  boxes.sort((a, b) => b.score - a.score);
  const kept = [];
  const suppressed = new Array(boxes.length).fill(false);
  for (let i = 0; i < boxes.length; i++) {
    if (suppressed[i]) continue;
    const a = boxes[i];
    kept.push(a);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    for (let j = i + 1; j < boxes.length; j++) {
      if (suppressed[j]) continue;
      const b = boxes[j];
      const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
      const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
      const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
      const iou = inter / (areaA + areaB - inter);
      if (iou > iouThresh) suppressed[j] = true;
    }
  }
  return kept;
}

/**
 * inference.js — ONNX Runtime Web wrapper for the YOLO26n-P2 VisDrone detector.
 *
 * Letterbox preprocessing, WebGPU -> WASM fallback, and decoding of either
 * export layout recorded in model/model-meta.json:
 *   end2end: [1, K, 6] rows (x1, y1, x2, y2, score, cls) in letterboxed pixels — NMS-free
 *   raw:     [1, 4 + nc, N] (cx, cy, w, h, class scores) — class-aware NMS runs here
 * Output of Detector.detect: Array<[x1, y1, x2, y2, score, cls]> in SOURCE pixels.
 *
 * Everything except the Detector class is pure (no window/document access at
 * import time) so web/tests/inference.test.mjs can exercise it under Node.
 */
const MODEL_URL = new URL('../model/aerial-guardian.onnx', import.meta.url).href;
const META_URL = new URL('../model/model-meta.json', import.meta.url).href;
const PAD_GREY = 114;             // Ultralytics letterbox fill
const MAX_RAW_CANDIDATES = 3000;  // cap before the O(n^2) NMS on the 76,500-anchor raw layout

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/** Letterbox geometry: uniform scale to fit `size`x`size`, content centred, pad floored. */
export function letterboxParams(srcW, srcH, size) {
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale), newH = Math.round(srcH * scale);
  return { scale, newW, newH, padX: Math.floor((size - newW) / 2), padY: Math.floor((size - newH) / 2) };
}

/** Map one [x1,y1,x2,y2,score,cls] row from letterboxed pixels back to source pixels, clipped to the frame. */
export function unletterbox([x1, y1, x2, y2, s, c], { scale, padX, padY }, srcW, srcH) {
  const clip = (v, hi) => Math.min(hi, Math.max(0, v));
  return [
    clip((x1 - padX) / scale, srcW), clip((y1 - padY) / scale, srcH),
    clip((x2 - padX) / scale, srcW), clip((y2 - padY) / scale, srcH), s, c,
  ];
}

/** unletterbox every row and drop boxes that have no area left inside the source frame. */
export function mapToSource(boxes, lb, srcW, srcH) {
  const out = [];
  for (const b of boxes) {
    const m = unletterbox(b, lb, srcW, srcH);
    if (m[2] > m[0] && m[3] > m[1]) out.push(m);
  }
  return out;
}

/** Rows (x1,y1,x2,y2,score,cls) with score >= conf from an end2end [1,K,6] tensor. */
export function decodeEnd2End(t, conf) {
  const K = t.dims[1], d = t.data, out = [];
  for (let i = 0; i < K; i++) {
    const o = 6 * i;
    if (d[o + 4] >= conf) out.push([d[o], d[o + 1], d[o + 2], d[o + 3], d[o + 4], Math.round(d[o + 5])]);
  }
  return out;
}

export function iou(a, b) {
  const inter = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
    * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/** Class-aware greedy NMS over [x1,y1,x2,y2,score,cls] rows; returns survivors by descending score. */
export function nms(boxes, iouThresh) {
  const order = [...boxes].sort((a, b) => b[4] - a[4]);
  const kept = [], dead = new Array(order.length).fill(false);
  for (let i = 0; i < order.length; i++) {
    if (dead[i]) continue;
    kept.push(order[i]);
    for (let j = i + 1; j < order.length; j++) {
      if (!dead[j] && order[j][5] === order[i][5] && iou(order[i], order[j]) > iouThresh) dead[j] = true;
    }
  }
  return kept;
}

/** Rows from a raw channel-first [1, 4+nc, N] tensor: best class per anchor, conf filter, class-aware NMS. */
export function decodeRaw(t, conf, iouThresh) {
  const C = t.dims[1], N = t.dims[2], d = t.data, nc = C - 4, cand = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < nc; c++) { const s = d[(4 + c) * N + i]; if (s > best) { best = s; cls = c; } }
    if (best < conf) continue;
    const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
    cand.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, cls]);
  }
  cand.sort((a, b) => b[4] - a[4]);
  return nms(cand.slice(0, MAX_RAW_CANDIDATES), iouThresh);
}

/** Fail loudly (with a message fit for the status line) on a malformed model-meta.json. */
export function validateMeta(meta) {
  if (!meta || typeof meta !== 'object') throw new Error('model-meta.json is not a JSON object');
  if (!(Number.isInteger(meta.input_size) && meta.input_size > 0)) throw new Error('model-meta.json: bad input_size');
  if (meta.output_format !== 'end2end' && meta.output_format !== 'raw') {
    throw new Error(`model-meta.json: output_format must be "end2end" or "raw", got ${JSON.stringify(meta.output_format)}`);
  }
  if (!Array.isArray(meta.names) || meta.names.length !== 10) throw new Error('model-meta.json: names must list the 10 VisDrone classes');
  return meta;
}

// ---------------------------------------------------------------------------
// Browser-only: session handling
// ---------------------------------------------------------------------------

function getOrt() {
  if (!globalThis.window?.ort) throw new Error('onnxruntime-web failed to load from the CDN (check your connection)');
  globalThis.window.ort.env.wasm.numThreads = 1;   // avoids the COOP/COEP requirement on static hosts
  return globalThis.window.ort;
}

async function fetchMeta() {
  let res;
  try { res = await fetch(META_URL, { cache: 'no-cache' }); }
  catch (e) { throw new Error(`Could not fetch model/model-meta.json: ${e.message}`); }
  if (!res.ok) throw new Error(`model/model-meta.json not found (HTTP ${res.status}) — the detector weights have not been deployed`);
  let meta;
  try { meta = await res.json(); }
  catch (e) { throw new Error(`model/model-meta.json is not valid JSON: ${e.message}`); }
  return validateMeta(meta);
}

export class Detector {
  constructor() { this.session = null; this.backend = null; this.meta = null; this.canvas = null; this.ctx = null; }

  async load(onStatus) {
    this.meta = await fetchMeta();
    const ort = getOrt();
    let lastErr = null;
    for (const name of ['webgpu', 'wasm']) {
      try {
        onStatus?.(`Initialising ${name} backend…`);
        this.session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: [name], graphOptimizationLevel: 'all' });
        this.backend = name;
        return name;
      } catch (e) { lastErr = e; }
    }
    throw new Error(`Could not load model/aerial-guardian.onnx: ${lastErr?.message || 'no available ONNX Runtime backend'}`);
  }

  preprocess(source, srcW, srcH) {
    const size = this.meta.input_size;
    const lb = letterboxParams(srcW, srcH, size);
    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      this.canvas.width = size; this.canvas.height = size;
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }
    this.ctx.fillStyle = `rgb(${PAD_GREY},${PAD_GREY},${PAD_GREY})`;
    this.ctx.fillRect(0, 0, size, size);
    this.ctx.drawImage(source, 0, 0, srcW, srcH, lb.padX, lb.padY, lb.newW, lb.newH);
    const { data } = this.ctx.getImageData(0, 0, size, size);
    const plane = size * size, chw = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      chw[i] = data[4 * i] / 255; chw[plane + i] = data[4 * i + 1] / 255; chw[2 * plane + i] = data[4 * i + 2] / 255;
    }
    return { chw, lb };
  }

  /** conf: detector score floor (keep low — ByteTrack's low-score stage needs those boxes). iou: raw layout only. */
  async detect(source, srcW, srcH, { conf = 0.05, iou: iouThresh = 0.6 } = {}) {
    const ort = getOrt();
    const size = this.meta.input_size;
    const { chw, lb } = this.preprocess(source, srcW, srcH);
    const out = await this.session.run({ [this.session.inputNames[0]]: new ort.Tensor('float32', chw, [1, 3, size, size]) });
    const t = out[this.session.outputNames[0]];
    let boxes;
    if (this.meta.output_format === 'end2end') {
      if (t.dims.length !== 3 || t.dims[2] !== 6) throw new Error(`model output ${JSON.stringify(t.dims)} does not match the end2end layout [1,K,6]`);
      boxes = decodeEnd2End(t, conf);
    } else {
      const nc = this.meta.names.length;
      if (t.dims.length !== 3 || t.dims[1] !== 4 + nc) throw new Error(`model output ${JSON.stringify(t.dims)} does not match the raw layout [1,${4 + nc},N]`);
      boxes = decodeRaw(t, conf, iouThresh);
    }
    return mapToSource(boxes, lb, srcW, srcH);
  }
}

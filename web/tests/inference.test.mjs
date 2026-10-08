import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decodeEnd2End, decodeRaw, nms, iou, letterboxParams, unletterbox, mapToSource, validateMeta }
  from '../js/inference.js';

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} !~ ${b}`);
const closeRow = (row, want) => { assert.equal(row.length, want.length); row.forEach((v, i) => close(v, want[i])); };

// ---------------------------------------------------------------- letterbox
test('letterbox: landscape 1920x1080 -> 960 scales by 0.5 and pads top/bottom', () => {
  assert.deepEqual(letterboxParams(1920, 1080, 960), { scale: 0.5, newW: 960, newH: 540, padX: 0, padY: 210 });
});

test('letterbox: portrait 1080x1920 pads left/right', () => {
  assert.deepEqual(letterboxParams(1080, 1920, 960), { scale: 0.5, newW: 540, newH: 960, padX: 210, padY: 0 });
});

test('letterbox: square 500x500 upscales with no padding', () => {
  const p = letterboxParams(500, 500, 960);
  assert.deepEqual(p, { scale: 1.92, newW: 960, newH: 960, padX: 0, padY: 0 });
});

test('letterbox: 640x480 upscales by 1.5 with 120px vertical pad', () => {
  assert.deepEqual(letterboxParams(640, 480, 960), { scale: 1.5, newW: 960, newH: 720, padX: 0, padY: 120 });
});

test('letterbox: rounded size and floored pad (333x500)', () => {
  // scale 1.92 -> newW = round(639.36) = 639, pad = floor((960 - 639) / 2) = 160
  const p = letterboxParams(333, 500, 960);
  assert.equal(p.newW, 639); assert.equal(p.newH, 960); assert.equal(p.padX, 160); assert.equal(p.padY, 0);
});

test('unletterbox: maps a box in the 960 canvas back to source pixels', () => {
  const lb = letterboxParams(1920, 1080, 960);        // scale .5, padY 210
  // source box (200,100)-(300,160) lands at (100,260)-(150,290) in the canvas
  closeRow(unletterbox([100, 260, 150, 290, 0.9, 3], lb, 1920, 1080), [200, 100, 300, 160, 0.9, 3]);
});

test('unletterbox: portrait pad is applied on x', () => {
  const lb = letterboxParams(1080, 1920, 960);        // scale .5, padX 210
  closeRow(unletterbox([310, 100, 410, 200, 0.5, 0], lb, 1080, 1920), [200, 200, 400, 400, 0.5, 0]);
});

test('unletterbox: boxes sticking into the padding are clipped to the source frame', () => {
  const lb = letterboxParams(1920, 1080, 960);
  closeRow(unletterbox([-20, 200, 100, 300, 0.7, 1], lb, 1920, 1080), [0, 0, 200, 180, 0.7, 1]);
  const r = unletterbox([900, 740, 980, 800, 0.7, 1], lb, 1920, 1080);
  assert.equal(r[2], 1920); assert.equal(r[3], 1080);
});

test('mapToSource drops boxes with no area inside the source after clipping', () => {
  const lb = letterboxParams(1920, 1080, 960);
  // entirely inside the top padding band (y < 210)
  const out = mapToSource([[10, 10, 60, 60, 0.9, 0], [100, 260, 150, 290, 0.9, 3]], lb, 1920, 1080);
  assert.equal(out.length, 1);
  assert.equal(out[0][5], 3);
});

// ---------------------------------------------------------------- end2end
test('decodeEnd2End keeps rows >= conf, in order, with rounded integer classes', () => {
  const data = Float32Array.from([
    10, 20, 110, 220, 0.90, 3,
    0, 0, 0, 0, 0.04, 0,          // padding row below conf
    30, 40, 60, 90, 0.05, 9.0,    // exactly the threshold -> kept
    5, 5, 15, 15, 0.30, 5.0000001,
  ]);
  const out = decodeEnd2End({ dims: [1, 4, 6], data }, 0.05);
  assert.equal(out.length, 3);
  closeRow(out[0], [10, 20, 110, 220, Math.fround(0.90), 3]);
  assert.equal(out[1][5], 9);
  assert.equal(out[2][5], 5);
});

test('decodeEnd2End: a higher confidence threshold removes low-score rows', () => {
  const data = Float32Array.from([1, 2, 3, 4, 0.9, 0, 1, 2, 3, 4, 0.2, 1]);
  assert.equal(decodeEnd2End({ dims: [1, 2, 6], data }, 0.5).length, 1);
  assert.equal(decodeEnd2End({ dims: [1, 2, 6], data }, 0.1).length, 2);
  assert.equal(decodeEnd2End({ dims: [1, 2, 6], data }, 0.95).length, 0);
});

// ---------------------------------------------------------------- nms
test('iou of identical, disjoint and half-overlapping boxes', () => {
  close(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1);
  close(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0);
  close(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3);   // inter 50 / union 150
});

test('nms is class-aware: same-class overlap suppressed, other class kept', () => {
  const boxes = [
    [0, 0, 100, 100, 0.9, 3],
    [2, 2, 102, 102, 0.8, 3],      // same class, IoU ~0.92 -> suppressed
    [2, 2, 102, 102, 0.7, 0],      // different class, same place -> kept
    [300, 300, 400, 400, 0.6, 3],  // disjoint -> kept
  ];
  const kept = nms(boxes, 0.6);
  assert.deepEqual(kept.map((b) => b[4]), [0.9, 0.7, 0.6]);
});

test('nms keeps the highest score first regardless of input order', () => {
  const kept = nms([[0, 0, 10, 10, 0.4, 1], [0, 0, 10, 10, 0.9, 1]], 0.5);
  assert.equal(kept.length, 1);
  assert.equal(kept[0][4], 0.9);
});

test('nms: boxes below the IoU threshold both survive', () => {
  // IoU = 1/3 < 0.6
  assert.equal(nms([[0, 0, 10, 10, 0.9, 2], [5, 0, 15, 10, 0.8, 2]], 0.6).length, 2);
});

// ---------------------------------------------------------------- raw
/** Build a channel-first [1, 14, N] tensor from [cx,cy,w,h,{cls:score}] candidates. */
function rawTensor(cands, N = cands.length) {
  const C = 14, data = new Float32Array(C * N);
  cands.forEach(([cx, cy, w, h, scores], i) => {
    data[i] = cx; data[N + i] = cy; data[2 * N + i] = w; data[3 * N + i] = h;
    for (const [c, s] of Object.entries(scores)) data[(4 + Number(c)) * N + i] = s;
  });
  return { dims: [1, C, N], data };
}

test('decodeRaw converts cx,cy,w,h to x1,y1,x2,y2 and takes the best class', () => {
  // N = 6 anchors, the real candidate sits at anchor 2: only a channel-first reader finds it
  // (an anchor-major reader would read cx from anchor 0 channel 2, etc.).
  const empty = [0, 0, 0, 0, {}];
  const t = rawTensor([empty, empty, [100, 200, 40, 20, { 3: 0.2, 4: 0.8, 9: 0.1 }], empty, empty, empty]);
  const out = decodeRaw(t, 0.05, 0.6);
  assert.equal(out.length, 1);
  closeRow(out[0], [80, 190, 120, 210, Math.fround(0.8), 4]);
});

test('decodeRaw: confidence threshold filters candidates', () => {
  const t = rawTensor([[50, 50, 20, 20, { 0: 0.04 }], [200, 200, 20, 20, { 0: 0.5 }]]);
  assert.equal(decodeRaw(t, 0.05, 0.6).length, 1);
  assert.equal(decodeRaw(t, 0.6, 0.6).length, 0);
});

test('decodeRaw: class-aware NMS suppresses same-class duplicates, keeps different class', () => {
  const t = rawTensor([
    [100, 100, 50, 50, { 3: 0.9 }],
    [102, 101, 50, 50, { 3: 0.7 }],   // duplicate car -> suppressed
    [101, 100, 50, 50, { 0: 0.6 }],   // pedestrian at the same place -> kept
    [400, 400, 30, 30, { 3: 0.5 }],   // separate car -> kept
  ]);
  const out = decodeRaw(t, 0.05, 0.6);
  assert.deepEqual(out.map((b) => [Math.round(b[4] * 10), b[5]]), [[9, 3], [6, 0], [5, 3]]);
});

// ---------------------------------------------------------------- meta
test('validateMeta accepts the documented formats and rejects anything else', () => {
  const ok = { input_size: 960, output_format: 'end2end', names: Array(10).fill('x') };
  assert.equal(validateMeta(ok), ok);
  assert.equal(validateMeta({ ...ok, output_format: 'raw' }).output_format, 'raw');
  assert.throws(() => validateMeta({ ...ok, output_format: 'nms' }), /output_format/);
  assert.throws(() => validateMeta({ ...ok, input_size: 0 }), /input_size/);
  assert.throws(() => validateMeta({ ...ok, names: ['a'] }), /names/);
  assert.throws(() => validateMeta(null), /model-meta/);
});

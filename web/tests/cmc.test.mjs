import { test } from 'node:test';
import assert from 'node:assert/strict';
import { estimateShift } from '../js/cmc.js';

function pattern(w, h, dx = 0, dy = 0) {
  const img = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const u = x - dx, v = y - dy;
    img[y * w + x] = Math.max(0, Math.min(255, Math.round(128 + 60 * Math.sin(u / 5) + 60 * Math.cos(v / 7) + 30 * Math.sin((u + v) / 11))));
  }
  return img;
}

test('recovers an integer shift', () => {
  const w = 160, h = 90;
  assert.deepEqual(estimateShift(pattern(w, h), pattern(w, h, 3, -2), w, h), [3, -2]);
});

test('zero motion', () => {
  const w = 160, h = 90, a = pattern(w, h);
  assert.deepEqual(estimateShift(a, a, w, h), [0, 0]);
});

test('recovers a larger in-radius shift on a 160x90 texture', () => {
  const w = 160, h = 90;
  assert.deepEqual(estimateShift(pattern(w, h), pattern(w, h, 6, -4), w, h), [6, -4]);
});

test('shift beyond the search radius is clamped, not a crash', () => {
  const w = 160, h = 90, radius = 8;
  const [dx, dy] = estimateShift(pattern(w, h), pattern(w, h, 12, 0), w, h, radius);
  assert.ok(Number.isInteger(dx) && Number.isInteger(dy));
  assert.ok(Math.hypot(dx, dy) <= radius * Math.SQRT2 + 1e-9);
  assert.ok(Math.abs(dx) <= radius && Math.abs(dy) <= radius);
});

test('estimateShift on 160x90 averages well under 50 ms per call', () => {
  const w = 160, h = 90, a = pattern(w, h), b = pattern(w, h, 5, 3);
  estimateShift(a, b, w, h);   // warm-up
  const t0 = performance.now();
  for (let i = 0; i < 20; i++) estimateShift(a, b, w, h);
  const avg = (performance.now() - t0) / 20;
  assert.ok(avg < 50, `average ${avg.toFixed(2)} ms per call`);
});

test('GlobalMotion returns null first, then a prev->curr translation in source pixels', async () => {
  const { GlobalMotion } = await import('../js/cmc.js');
  const w = 160, h = 90, scale = 4;                 // source is 640x360
  let current = null;
  globalThis.document = {
    createElement: () => ({
      width: 0, height: 0,
      getContext: () => ({
        drawImage() {},
        getImageData: () => {
          const data = new Uint8ClampedArray(w * h * 4);
          for (let i = 0; i < w * h; i++) data.fill(current[i], 4 * i, 4 * i + 3);
          return { data };
        },
      }),
    }),
  };
  try {
    const gm = new GlobalMotion(w);
    current = pattern(w, h);
    assert.equal(gm.update({}, w * scale, h * scale), null);
    current = pattern(w, h, 3, -2);
    assert.deepEqual(gm.update({}, w * scale, h * scale), [[1, 0, 12], [0, 1, -8]]);
    gm.reset();
    assert.equal(gm.update({}, w * scale, h * scale), null);
  } finally {
    delete globalThis.document;
  }
});

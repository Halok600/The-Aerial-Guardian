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


// Deterministic PRNG so the noise tests are reproducible (never Math.random).
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function noiseImage(w, h, seed) {
  const rnd = mulberry32(seed), img = new Uint8Array(w * h);
  for (let i = 0; i < img.length; i++) img[i] = Math.floor(rnd() * 256);
  return img;
}

// Soft Gaussian blobs on a mid-grey background, shifted by (dx, dy), plus optional
// seeded noise of +-noise grey levels. `contrast` scales the blob amplitude.
const BLOBS = [[30, 20, 8], [80, 45, 11], [120, 25, 7], [50, 70, 9], [135, 68, 10], [100, 80, 6], [15, 55, 7]];
function blobImage(w, h, dx = 0, dy = 0, { noise = 0, seed = 1, contrast = 1 } = {}) {
  const rnd = mulberry32(seed), img = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 128;
    BLOBS.forEach(([bx, by, r], i) => {
      const d2 = (x - dx - bx) ** 2 + (y - dy - by) ** 2;
      v += (i % 2 ? -1 : 1) * 90 * contrast * Math.exp(-d2 / (2 * r * r));
    });
    v += noise * (2 * rnd() - 1);
    img[y * w + x] = Math.max(0, Math.min(255, Math.round(v)));
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

test('a shift beyond the search radius saturates at the boundary (documented limitation)', () => {
  // True shift 12 > radius 8: it is NOT recovered. The best candidate is the one
  // at the search boundary, so the result saturates at 8 along that axis. Larger
  // camera motion needs the Python pipeline's ECC compensator.
  const w = 160, h = 90;
  assert.deepEqual(estimateShift(pattern(w, h), pattern(w, h, 12, 0), w, h), [8, 0]);
  assert.deepEqual(estimateShift(blobImage(w, h), blobImage(w, h, 12, 0), w, h), [8, 0]);
});

test('independent pure-noise pairs give no motion (confidence gate)', () => {
  const w = 160, h = 90;
  for (let seed = 1; seed <= 10; seed++) {
    const r = estimateShift(noiseImage(w, h, seed), noiseImage(w, h, 1000 + seed), w, h);
    assert.deepEqual(r, [0, 0], `seed ${seed}`);
  }
});

test('a genuine shift is still recovered through mild noise', () => {
  const w = 160, h = 90;
  const a = blobImage(w, h, 0, 0, { noise: 6, seed: 11 });
  const b = blobImage(w, h, 5, -3, { noise: 6, seed: 12 });
  assert.deepEqual(estimateShift(a, b, w, h), [5, -3]);
});

test('flat image pair gives no motion', () => {
  const w = 160, h = 90, a = new Uint8Array(w * h).fill(120), b = new Uint8Array(w * h).fill(120);
  assert.deepEqual(estimateShift(a, b, w, h), [0, 0]);
});

test('static low-contrast scene with independent sensor noise does not drift', () => {
  const w = 160, h = 90;
  for (let seed = 1; seed <= 10; seed++) {
    const a = blobImage(w, h, 0, 0, { noise: 8, seed, contrast: 0.08 });
    const b = blobImage(w, h, 0, 0, { noise: 8, seed: 500 + seed, contrast: 0.08 });
    assert.deepEqual(estimateShift(a, b, w, h), [0, 0], `seed ${seed}`);
  }
});

test('the gate can be disabled and then returns the raw best candidate', () => {
  const w = 160, h = 90;
  const moved = [];
  for (let seed = 1; seed <= 10; seed++) {
    moved.push(estimateShift(noiseImage(w, h, seed), noiseImage(w, h, 1000 + seed), w, h, 8, Infinity));
  }
  assert.ok(moved.some(([dx, dy]) => dx !== 0 || dy !== 0), 'ungated noise should produce some spurious motion');
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

test('estimateShift: a frame too small for the search radius returns [0, 0] instead of NaN scores', () => {
  const r = 8;
  for (const [w, h] of [[16, 90], [160, 16], [10, 10], [0, 0], [2 * r - 1, 2 * r - 1]]) {
    const a = new Uint8Array(Math.max(1, w * h)).fill(100), b = new Uint8Array(Math.max(1, w * h)).fill(100);
    assert.deepEqual(estimateShift(a, b, w, h, r), [0, 0], `${w}x${h}`);
  }
});

// Fake canvas/DOM for GlobalMotion: records allocations; pixel source is `fake.current`.
function fakeDom(w, hOf) {
  const fake = { current: null, canvases: 0 };
  globalThis.document = {
    createElement: () => {
      fake.canvases += 1;
      const canvas = { width: 0, height: 0 };
      canvas.getContext = () => ({
        drawImage() {},
        getImageData: () => {
          const h = canvas.height, data = new Uint8ClampedArray(w * h * 4);
          for (let i = 0; i < w * h; i++) data.fill(fake.current(i % w, Math.floor(i / w), h), 4 * i, 4 * i + 3);
          return { data };
        },
      });
      return canvas;
    },
  };
  return fake;
}

test('GlobalMotion.update is a no-op returning null when the source size is not known yet (0 or non-finite)', async () => {
  const { GlobalMotion } = await import('../js/cmc.js');
  const fake = fakeDom(160);
  try {
    fake.current = () => 128;
    const gm = new GlobalMotion(160);
    for (const [sw, sh] of [[0, 0], [640, 0], [0, 360], [NaN, 360], [640, NaN], [undefined, undefined]]) {
      assert.equal(gm.update({}, sw, sh), null, `${sw}x${sh}`);
    }
    assert.equal(fake.canvases, 0, 'no canvas allocated, no state touched');
    assert.equal(gm.prev, null);
  } finally {
    delete globalThis.document;
  }
});

test('GlobalMotion drops its previous thumbnail when the source size / aspect changes', async () => {
  const { GlobalMotion } = await import('../js/cmc.js');
  const fake = fakeDom(160);
  try {
    const px = (dx, dy) => (x, y) => Math.max(0, Math.min(255, Math.round(128 + 60 * Math.sin((x - dx) / 5) + 60 * Math.cos((y - dy) / 7))));
    const gm = new GlobalMotion(160);
    fake.current = px(0, 0);
    assert.equal(gm.update({}, 640, 360), null);          // first frame: 160x90 thumbnail
    fake.current = px(3, 0);
    assert.ok(gm.update({}, 640, 360));                   // same source size: estimate
    // Different source size with the SAME thumbnail height (160x90 from 1280x720) must still reset.
    fake.current = px(3, 0);
    assert.equal(gm.update({}, 1280, 720), null);
    fake.current = px(5, 0);
    assert.ok(gm.update({}, 1280, 720));
    // Aspect change: thumbnail height differs, again no diff against the stale frame.
    fake.current = px(5, 0);
    assert.equal(gm.update({}, 640, 480), null);
  } finally {
    delete globalThis.document;
  }
});

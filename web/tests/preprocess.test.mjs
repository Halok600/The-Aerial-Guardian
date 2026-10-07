import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Detector } from '../js/inference.js';

/** A fake 2D canvas: records draw calls and serves a scripted RGBA image from getImageData. */
function fakeEnvironment(size, rgba) {
  const calls = [];
  const ctx = {
    fillStyle: '',
    fillRect: (...a) => calls.push(['fillRect', ctx.fillStyle, ...a]),
    drawImage: (...a) => calls.push(['drawImage', ...a]),
    getImageData: (...a) => { calls.push(['getImageData', ...a]); return { data: Uint8ClampedArray.from(rgba) }; },
  };
  const canvas = { width: 0, height: 0, getContext: () => ctx };
  globalThis.document = { createElement: () => canvas };
  return { calls, canvas };
}

function detectorWith(size) {
  const d = new Detector();
  d.meta = { input_size: size, output_format: 'end2end', names: Array(10).fill('x') };
  return d;
}

test('preprocess: letterbox draw call, grey fill, NCHW planar RGB divided by 255', () => {
  // 2x2 input canvas; pixels (R,G,B,A): p0=(255,0,0) p1=(0,255,0) p2=(0,0,255) p3=(51,102,153)
  const rgba = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 51, 102, 153, 255];
  const { calls, canvas } = fakeEnvironment(2, rgba);
  const d = detectorWith(2);
  const { chw, lb } = d.preprocess({ tag: 'src' }, 4, 1);          // 4x1 source -> scale .5, 2x1 content, padY 0 (floor(0.5))
  assert.equal(canvas.width, 2); assert.equal(canvas.height, 2);
  assert.deepEqual(lb, { scale: 0.5, newW: 2, newH: 1, padX: 0, padY: 0 });
  assert.deepEqual(calls[0], ['fillRect', 'rgb(114,114,114)', 0, 0, 2, 2]);
  assert.deepEqual(calls[1], ['drawImage', { tag: 'src' }, 0, 0, 4, 1, 0, 0, 2, 1]);
  assert.equal(chw.length, 3 * 4);
  assert.ok(chw instanceof Float32Array);
  const f = Math.fround;
  // plane 0 = R, plane 1 = G, plane 2 = B, each in pixel order
  assert.deepEqual([...chw.slice(0, 4)], [1, 0, 0, f(51 / 255)]);
  assert.deepEqual([...chw.slice(4, 8)], [0, 1, 0, f(102 / 255)]);
  assert.deepEqual([...chw.slice(8, 12)], [0, 0, 1, f(153 / 255)]);
});

test('preprocess reuses one tensor buffer across frames and still overwrites every value', () => {
  const first = [10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255];
  const second = first.map((v, i) => (i % 4 === 3 ? 255 : 255 - v));
  const env = fakeEnvironment(2, first);
  const d = detectorWith(2);
  const a = d.preprocess({}, 2, 2).chw;
  const snapshot = [...a];
  env.canvas.getContext = () => ({
    fillStyle: '', fillRect() {}, drawImage() {},
    getImageData: () => ({ data: Uint8ClampedArray.from(second) }),
  });
  d.ctx = env.canvas.getContext();
  const b = d.preprocess({}, 2, 2).chw;
  assert.equal(a, b, 'same Float32Array instance');
  assert.notDeepEqual([...b], snapshot);
  assert.equal(b[0], Math.fround((255 - 10) / 255));
  assert.equal(b[4], Math.fround((255 - 20) / 255));
});

test('preprocess allocates a new buffer only if the input size changes', () => {
  fakeEnvironment(2, new Array(16).fill(0));
  const d = detectorWith(2);
  const a = d.preprocess({}, 2, 2).chw;
  d.meta.input_size = 4;
  d.canvas = null;
  fakeEnvironment(4, new Array(64).fill(0));
  const b = d.preprocess({}, 2, 2).chw;
  assert.notEqual(a, b);
  assert.equal(b.length, 3 * 16);
});

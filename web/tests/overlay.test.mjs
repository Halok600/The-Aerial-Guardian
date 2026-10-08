import { test } from 'node:test';
import assert from 'node:assert/strict';
import { overlayMetrics, trailSegments, labelRect } from '../js/render.js';

test('overlayMetrics: canvas 1920 shown at 640 CSS px -> scale 3, sizes scale with it', () => {
  const m = overlayMetrics(1920, 640);
  assert.equal(m.scale, 3);
  assert.ok(m.lineWidth >= 6, `lineWidth ${m.lineWidth}`);
  assert.ok(m.fontPx >= 33, `fontPx ${m.fontPx}`);
  assert.ok(m.pad >= 12 && m.trailWidth >= 3);
});

test('overlayMetrics: 1:1 display keeps >= 2 CSS px lines and >= 11 CSS px text', () => {
  const m = overlayMetrics(960, 960);
  assert.equal(m.scale, 1);
  assert.ok(m.lineWidth >= 2);
  assert.ok(m.fontPx >= 11);
});

test('overlayMetrics: phone case (961 px source in a 343 CSS px stage) stays legible', () => {
  const m = overlayMetrics(961, 343);
  const k = 961 / 343;
  assert.ok(m.lineWidth / k >= 2 - 1e-9, 'box line >= 2 CSS px on screen');
  assert.ok(m.fontPx / k >= 11 - 1e-9, 'label text >= 11 CSS px on screen');
});

test('overlayMetrics: a zero / missing clientWidth does not divide by zero', () => {
  for (const cw of [0, undefined, NaN, -5]) {
    const m = overlayMetrics(1280, cw);
    assert.ok(Number.isFinite(m.lineWidth) && Number.isFinite(m.fontPx) && m.lineWidth > 0 && m.fontPx > 0);
  }
  assert.ok(Number.isFinite(overlayMetrics(0, 400).fontPx));
});

test('overlayMetrics: object-fit contain uses the limiting axis (4:3 source in a 16:9 box)', () => {
  // 1600x1200 canvas in a 640x360 element: content is shown at 360/1200 = 0.3 scale (not 640/1600 = 0.4)
  const m = overlayMetrics(1600, 640, 1200, 360);
  assert.ok(Math.abs(m.scale - 1 / 0.3) < 1e-9);
});

test('trailSegments widens strokes with the display scale (never below 1 CSS px)', () => {
  const hist = [[0, 0], [1, 0], [2, 0]];
  const segs = trailSegments(hist, 3, 3);
  assert.ok(Math.abs(segs[1].width - 9) < 1e-9);
  assert.ok(Math.abs(segs[0].width - 4.5) < 1e-9);
  assert.ok(trailSegments(hist, 3, 2).every((s) => s.width >= 2));
});

test('labelRect honours an explicit padding', () => {
  const r = labelRect(100, 80, 60, 36, 12);
  assert.deepEqual(r, { x: 100, y: 80 - 54, w: 60 + 24, h: 36 + 18, textX: 112, textY: 80 - 54 + 54 - 9 });
});

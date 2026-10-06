import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ByteTracker, DEFAULT_CONFIG, pyRound } from '../js/tracker.js';

const golden = JSON.parse(readFileSync(new URL('../../tests/fixtures/tracker_golden.json', import.meta.url)));

test('DEFAULT_CONFIG equals the config recorded in the golden fixture', () => {
  assert.deepEqual(DEFAULT_CONFIG, golden.config);
});

test('JS tracker reproduces Python IDs, classes and boxes frame by frame', () => {
  const tracker = new ByteTracker(golden.config);
  golden.frames.forEach((frame, i) => {
    const got = tracker.update(frame.dets)
      .map((t) => [t.trackId, ...t.tlbr, t.cls])
      .sort((a, b) => a[0] - b[0]);
    assert.equal(got.length, frame.tracks.length, `frame ${i + 1}: track count`);
    got.forEach((row, j) => {
      const want = frame.tracks[j];
      assert.equal(row[0], want[0], `frame ${i + 1}: id`);
      assert.equal(row[5], want[5], `frame ${i + 1}: cls`);
      for (let k = 1; k <= 4; k++) assert.ok(Math.abs(row[k] - want[k]) < 1e-6, `frame ${i + 1}: box`);
    });
  });
  assert.deepEqual(tracker.uniqueCounts(), golden.unique_counts);
});

test('translation warp shifts tracks', () => {
  const tracker = new ByteTracker(DEFAULT_CONFIG);
  tracker.update([[0, 0, 20, 40, 0.9, 0]]);
  const [t] = tracker.update([[10, -5, 30, 35, 0.9, 0]], [[1, 0, 10], [0, 1, -5]]);
  assert.equal(t.trackId, 1);
  assert.ok(Math.abs(t.tlbr[0] - 10) < 0.5);
});

const box = (x, y, w, h, cls = 0, score = 0.9) => [x, y, x + w, y + h, score, cls];

// Mirrors tests/test_byte_tracker.py::test_degenerate_boxes_are_dropped_before_tracking.
test('degenerate boxes are dropped before tracking', () => {
  const zeroHeight = [300, 400, 340, 400, 0.9, 0];
  const inverted = [500, 600, 480, 640, 0.9, 0];
  const nan = [NaN, 10, 30, 50, 0.9, 0];
  const tracker = new ByteTracker(DEFAULT_CONFIG);
  let out;
  assert.doesNotThrow(() => {
    tracker.update([box(100, 200, 30, 60), zeroHeight, inverted, nan]);
    out = tracker.update([box(103, 200, 30, 60), zeroHeight, inverted, nan]);
  });
  assert.deepEqual(out.map((t) => t.trackId), [1]);
  assert.equal(tracker.totalIds, 1);
  assert.equal(tracker.tracked.length, 1);
});

test('pyRound rounds half to even like Python round()', () => {
  assert.equal(pyRound(12.5), 12);
  assert.equal(pyRound(13.5), 14);
  assert.equal(pyRound(0.5), 0);
  assert.equal(pyRound(1.5), 2);
  assert.equal(pyRound(2.4), 2);
  assert.equal(pyRound(2.6), 3);
  assert.equal(pyRound(-0.5), 0);
  assert.equal(pyRound(-1.5), -2);
});

test('maxTimeLost scales with frame rate using banker rounding', () => {
  assert.equal(new ByteTracker({ ...DEFAULT_CONFIG, max_time_lost: 15 }, 25).maxTimeLost, 12); // 12.5 -> 12
  assert.equal(new ByteTracker(DEFAULT_CONFIG, 30).maxTimeLost, 30);
  assert.equal(new ByteTracker(DEFAULT_CONFIG, 15).maxTimeLost, 15);
});

test('class vote: majority wins, ties go to the most recent class', () => {
  const tracker = new ByteTracker(DEFAULT_CONFIG);
  const seq = [0, 0, 1, 1, 1, 0, 1, 0]; // counts 0:4, 1:4 -> tie, last vote is 0
  let out;
  seq.forEach((c, i) => { out = tracker.update([box(100 + 2 * i, 100, 30, 60, c)]); });
  assert.equal(out[0].cls, 0);
  out = tracker.update([box(116, 100, 30, 60, 1)]); // 1:5 vs 0:4
  assert.equal(out[0].cls, 1);
});

test('one-frame false positive burns no ID', () => {
  const tracker = new ByteTracker(DEFAULT_CONFIG);
  tracker.update([box(100, 100, 30, 60)]);
  tracker.update([box(102, 100, 30, 60), box(500, 500, 20, 40)]);
  tracker.update([box(104, 100, 30, 60)]);
  assert.equal(tracker.totalIds, 1);
});

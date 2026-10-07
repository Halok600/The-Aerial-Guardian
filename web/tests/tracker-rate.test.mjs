import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ByteTracker, DEFAULT_CONFIG } from '../js/tracker.js';

test('setFrameRate rescales maxTimeLost (banker rounding, clamped to >= 3) and nothing else', () => {
  const t = new ByteTracker(DEFAULT_CONFIG, 30);
  const before = JSON.stringify({ cfg: t.cfg, frameId: t.frameId });
  t.setFrameRate(10);
  assert.equal(t.maxTimeLost, 10);
  t.setFrameRate(25);
  assert.equal(t.maxTimeLost, 25);
  t.setFrameRate(0.1);
  assert.equal(t.maxTimeLost, 3);
  t.setFrameRate(30);
  assert.equal(t.maxTimeLost, 30);
  assert.equal(JSON.stringify({ cfg: t.cfg, frameId: t.frameId }), before);
  const t2 = new ByteTracker({ ...DEFAULT_CONFIG, max_time_lost: 15 });
  t2.setFrameRate(25);
  assert.equal(t2.maxTimeLost, 12);          // 12.5 -> 12 like Python round()
});

test('setFrameRate ignores nonsense and survives reset()', () => {
  const t = new ByteTracker(DEFAULT_CONFIG, 30);
  t.setFrameRate(15);
  t.setFrameRate(NaN); t.setFrameRate(-4); t.setFrameRate(undefined);
  assert.equal(t.maxTimeLost, 15);
  t.reset();
  assert.equal(t.maxTimeLost, 15);
});

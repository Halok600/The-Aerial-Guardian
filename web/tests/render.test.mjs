import { test } from 'node:test';
import assert from 'node:assert/strict';
import { labelText, trailSegments, labelRect, visibleTracks } from '../js/render.js';
import { FpsMeter, statsView, groupCounts } from '../js/stats.js';

test('labelText uses the class name and track id', () => {
  assert.equal(labelText(0, 1), 'pedestrian ID:1');
  assert.equal(labelText(3, 12), 'car ID:12');
  assert.equal(labelText(7, 3), 'awning-tricycle ID:3');
});

test('trailSegments: fewer than 2 points or trailLen < 2 gives nothing', () => {
  assert.deepEqual(trailSegments([[1, 1]], 30), []);
  assert.deepEqual(trailSegments([[1, 1], [2, 2], [3, 3]], 0), []);
  assert.deepEqual(trailSegments([[1, 1], [2, 2], [3, 3]], 1), []);
});

test('trailSegments keeps only the newest trailLen points and fades old -> new', () => {
  const hist = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]];
  const segs = trailSegments(hist, 3);          // points [2,3,4] -> 2 segments
  assert.equal(segs.length, 2);
  assert.deepEqual([segs[0].x0, segs[0].x1, segs[1].x0, segs[1].x1], [2, 3, 3, 4]);
  // newest segment: a = 1 -> alpha .9, width 3 ; oldest: a = .5 -> alpha .525, width 1.5
  assert.ok(Math.abs(segs[1].alpha - 0.9) < 1e-9 && Math.abs(segs[1].width - 3) < 1e-9);
  assert.ok(Math.abs(segs[0].alpha - 0.525) < 1e-9 && Math.abs(segs[0].width - 1.5) < 1e-9);
  assert.ok(segs[0].alpha < segs[1].alpha);
});

test('trailSegments width never drops below 1px', () => {
  const hist = Array.from({ length: 60 }, (_, i) => [i, 0]);
  assert.ok(trailSegments(hist, 60).every((s) => s.width >= 1));
});

test('labelRect sits above the box and clamps to the top edge', () => {
  const r = labelRect(100, 80, 60, 18);          // text 60px wide, 18px font
  assert.deepEqual(r, { x: 100, y: 56, w: 68, h: 24, textX: 104, textY: 77 });
  assert.equal(labelRect(10, 5, 60, 18).y, 0);
});

test('visibleTracks filters by enabled class group', () => {
  const tracks = [{ group: 'person', trackId: 1 }, { group: 'vehicle', trackId: 2 }, { group: 'tricycle', trackId: 3 }];
  assert.deepEqual(visibleTracks(tracks, new Set(['person', 'tricycle'])).map((t) => t.trackId), [1, 3]);
  assert.deepEqual(visibleTracks(tracks, new Set()), []);
});

test('FpsMeter averages over a sliding window of the last N frame times', () => {
  const m = new FpsMeter(3);
  assert.equal(m.fps(), 0);
  m.push(0.1); m.push(0.1); m.push(0.1);
  assert.ok(Math.abs(m.fps() - 10) < 1e-9);
  m.push(0.5); m.push(0.5); m.push(0.5);        // the 0.1s have rolled out
  assert.ok(Math.abs(m.fps() - 2) < 1e-9);
  m.reset();
  assert.equal(m.fps(), 0);
});

test('statsView formats the live-stats text and per-group counters', () => {
  const v = statsView({ tracks: 5, fps: 12.345, detections: 17, totalIds: 9,
    counts: { person: 4, vehicle: 3, 'two-wheeler': 0, tricycle: 1 } });
  assert.equal(v.fps, '12.3');
  assert.equal(v.hudFps, '12.3 FPS');
  assert.equal(v.tracks, '5');
  assert.equal(v.hudTracks, '5 tracked');
  assert.equal(v.detections, '17');
  assert.equal(v.totalIds, '9');
  assert.deepEqual(v.counts, { person: '4', vehicle: '3', 'two-wheeler': '0', tricycle: '1' });
  assert.equal(statsView({}).counts.person, '0');
});

test('groupCounts tallies tracks per class group (used for single images, where unique-ID counting needs >1 frame)', () => {
  const c = groupCounts([{ group: 'person' }, { group: 'person' }, { group: 'vehicle' }]);
  assert.deepEqual(c, { person: 2, vehicle: 1, 'two-wheeler': 0, tricycle: 0 });
  assert.deepEqual(groupCounts([]), { person: 0, vehicle: 0, 'two-wheeler': 0, tricycle: 0 });
});

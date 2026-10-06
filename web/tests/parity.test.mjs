// Frame-by-frame parity against the real Python tracker on compact scenarios that exercise
// the branches the single golden scenario misses (see scripts/dev/make_tracker_golden.py).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ByteTracker } from '../js/tracker.js';

const { scenarios } = JSON.parse(readFileSync(new URL('../../tests/fixtures/tracker_parity.json', import.meta.url)));
const TOL = 1e-6;

for (const sc of scenarios) {
  test(`parity: ${sc.name} (frame_rate ${sc.frame_rate}, ${sc.frames.length} frames)`, () => {
    const tracker = new ByteTracker(sc.config, sc.frame_rate);
    sc.frames.forEach((frame, i) => {
      const at = `${sc.name} frame ${i + 1}`;
      const got = tracker.update(frame.dets, frame.warp)
        .map((t) => [t.trackId, ...t.tlbr, t.cls])
        .sort((a, b) => a[0] - b[0]);
      assert.deepEqual(got.map((r) => r[0]), frame.tracks.map((r) => r[0]), `${at}: track ids`);
      assert.deepEqual(got.map((r) => r[5]), frame.tracks.map((r) => r[5]), `${at}: classes`);
      got.forEach((row, j) => {
        for (let k = 1; k <= 4; k++) {
          assert.ok(Math.abs(row[k] - frame.tracks[j][k]) < TOL, `${at}: box coord ${k} of id ${row[0]} (${row[k]} vs ${frame.tracks[j][k]})`);
        }
      });
      assert.equal(tracker.tracked.length, frame.n_tracked, `${at}: n_tracked`);
      assert.equal(tracker.lost.length, frame.n_lost, `${at}: n_lost`);
    });
    assert.deepEqual(tracker.uniqueCounts(), sc.unique_counts, `${sc.name}: unique counts`);
    assert.equal(tracker.totalIds, sc.total_ids, `${sc.name}: total ids`);
  });
}

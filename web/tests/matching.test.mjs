import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GATED_COST, fuseScore, gateGroups, iouCost, iouMatrix, linearAssignment } from '../js/matching.js';
import { hungarianSolve } from '../js/hungarian.js';
import { groupOf } from '../js/classes.js';

test('iouMatrix / iouCost basics', () => {
  const m = iouMatrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]);
  assert.ok(Math.abs(m[0][0] - 1) < 1e-12);
  assert.ok(Math.abs(m[0][1] - 1 / 3) < 1e-12);
  assert.equal(m[0][2], 0);
  assert.ok(Math.abs(iouCost([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0][0]) < 1e-12);
});

test('fuseScore and gateGroups', () => {
  assert.deepEqual(fuseScore([[0.2, 0.4]], [0.5, 1]).map((r) => r.map((v) => +v.toFixed(6))), [[0.6, 0.4]]);
  assert.deepEqual(gateGroups([[0.1, 0.2]], ['person'], ['person', 'vehicle']), [[0.1, GATED_COST]]);
});

test('groupOf maps VisDrone classes and rejects unknown ones', () => {
  assert.equal(groupOf(0), 'person');
  assert.equal(groupOf(8), 'vehicle');
  assert.equal(groupOf(9), 'two-wheeler');
  assert.equal(groupOf(7), 'tricycle');
  assert.throws(() => groupOf(10));
});

// Parity with scipy's linear_sum_assignment depends on matches being row-sorted.
test('linearAssignment returns row-sorted matches: more rows than columns', () => {
  const cost = [[0.1, 0.9], [0.8, 0.2], [0.5, 0.6]];
  const r = linearAssignment(cost, 0.8, 2);
  assert.deepEqual(r.matches, [[0, 0], [1, 1]]);
  assert.deepEqual(r.unmatchedRows, [2]);
  assert.deepEqual(r.unmatchedCols, []);
});

test('linearAssignment returns row-sorted matches: more columns than rows', () => {
  const cost = [[0.9, 0.1, 0.8], [0.2, 0.9, 0.7]];
  const r = linearAssignment(cost, 0.8, 3);
  assert.deepEqual(r.matches, [[0, 1], [1, 0]]);
  assert.deepEqual(r.unmatchedRows, []);
  assert.deepEqual(r.unmatchedCols, [2]);
});

test('linearAssignment applies the threshold with <= and handles empties', () => {
  let r = linearAssignment([[0.8, 0.95]], 0.8, 2);
  assert.deepEqual(r.matches, [[0, 0]]);
  assert.deepEqual(r.unmatchedCols, [1]);
  r = linearAssignment([[0.85]], 0.8, 1);
  assert.deepEqual(r.matches, []);
  assert.deepEqual([r.unmatchedRows, r.unmatchedCols], [[0], [0]]);
  r = linearAssignment([], 0.8, 3);
  assert.deepEqual([r.matches, r.unmatchedRows, r.unmatchedCols], [[], [], [0, 1, 2]]);
  r = linearAssignment([[], []], 0.8, 0);
  assert.deepEqual([r.matches, r.unmatchedRows, r.unmatchedCols], [[], [0, 1], []]);
});

test('hungarianSolve finds the optimal assignment and is deterministic on ties', () => {
  assert.deepEqual(hungarianSolve([[4, 1, 3], [2, 0, 5], [3, 2, 2]]), [[0, 1], [1, 0], [2, 2]]);
  const ties = [[1, 1], [1, 1]];
  assert.deepEqual(hungarianSolve(ties), hungarianSolve(ties));
  assert.deepEqual(hungarianSolve(ties), [[0, 0], [1, 1]]);
});

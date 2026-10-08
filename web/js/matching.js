/** matching.js — port of src/tracker/matching.py. */
import { hungarianSolve } from './hungarian.js';

export const GATED_COST = 1e6;

export function iouMatrix(a, b) {
  return a.map(([ax1, ay1, ax2, ay2]) => {
    const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
    return b.map(([bx1, by1, bx2, by2]) => {
      const inter = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1)) *
                    Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
      const union = areaA + Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1) - inter;
      return union > 0 ? inter / Math.max(union, 1e-12) : 0;
    });
  });
}
export const iouCost = (a, b) => iouMatrix(a, b).map((r) => r.map((v) => 1 - v));
export const fuseScore = (cost, scores) => cost.map((r) => r.map((c, j) => 1 - (1 - c) * scores[j]));
export const gateGroups = (cost, tg, dg) => cost.map((r, i) => r.map((c, j) => (tg[i] !== dg[j] ? GATED_COST : c)));

/**
 * nCols must be passed explicitly: a 0-row matrix can't tell how many detections went unmatched.
 * Matches are returned sorted by row (as scipy's linear_sum_assignment does) so that
 * track-update order, and therefore ID assignment, is identical to the Python tracker.
 */
export function linearAssignment(cost, thresh, nCols) {
  const nRows = cost.length;
  const matches = [];
  if (nRows && nCols) for (const [r, c] of hungarianSolve(cost)) if (cost[r][c] <= thresh) matches.push([r, c]);
  matches.sort((p, q) => p[0] - q[0]);
  const mr = new Set(matches.map(([r]) => r)), mc = new Set(matches.map(([, c]) => c));
  return {
    matches,
    unmatchedRows: [...Array(nRows).keys()].filter((i) => !mr.has(i)),
    unmatchedCols: [...Array(nCols).keys()].filter((j) => !mc.has(j)),
  };
}

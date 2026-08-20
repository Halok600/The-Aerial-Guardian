/**
 * hungarian.js — Kuhn–Munkres assignment algorithm (rectangular, O(n^3)).
 * Port of the classic Munkres method. Minimises total cost.
 * Track/detection counts here are always small (<100), so O(n^3) is fine.
 */

export function hungarianSolve(costMatrixIn) {
  const nRows = costMatrixIn.length;
  const nCols = nRows > 0 ? costMatrixIn[0].length : 0;
  if (nRows === 0 || nCols === 0) return [];

  const n = Math.max(nRows, nCols);
  const BIG = 1e6;
  // Pad to square with a large cost so padded pairs are never chosen unless forced.
  const cost = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(BIG);
    if (i < nRows) {
      for (let j = 0; j < nCols; j++) row[j] = costMatrixIn[i][j];
    }
    cost.push(row);
  }

  // Standard Jonker-Volgenant-free Munkres via potentials (Hungarian algorithm, O(n^3)).
  const u = new Array(n + 1).fill(0);
  const v = new Array(n + 1).fill(0);
  const p = new Array(n + 1).fill(0); // p[j] = row assigned to column j (1-indexed)
  const way = new Array(n + 1).fill(0);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(n + 1).fill(Infinity);
    const used = new Array(n + 1).fill(false);

    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = -1;
      for (let j = 1; j <= n; j++) {
        if (!used[j]) {
          const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
        else { minv[j] -= delta; }
      }
      j0 = j1;
    } while (p[j0] !== 0);

    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  // p[j] = row (1-indexed) assigned to column j (1-indexed). Build row->col map.
  const rowToCol = new Array(n).fill(-1);
  for (let j = 1; j <= n; j++) {
    if (p[j] !== 0) rowToCol[p[j] - 1] = j - 1;
  }

  const pairs = [];
  for (let i = 0; i < nRows; i++) {
    const j = rowToCol[i];
    if (j >= 0 && j < nCols) pairs.push([i, j]);
  }
  return pairs;
}

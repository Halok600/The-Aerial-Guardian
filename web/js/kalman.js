/** kalman.js — port of src/tracker/kalman_filter.py (xyah state, height-scaled noise). */
const WP = 1 / 20, WV = 1 / 160;

export function tlbrToXyah([x1, y1, x2, y2]) {
  const w = x2 - x1, h = y2 - y1;
  return [x1 + w / 2, y1 + h / 2, w / Math.max(h, 1e-6), h];
}
export function xyahToTlbr([cx, cy, a, h]) {
  const w = a * h;
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}

const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const diag = (v) => { const m = zeros(v.length, v.length); v.forEach((x, i) => { m[i][i] = x; }); return m; };
const T = (A) => A[0].map((_, j) => A.map((row) => row[j]));
function mul(A, B) {
  const out = zeros(A.length, B[0].length);
  for (let i = 0; i < A.length; i++)
    for (let k = 0; k < B.length; k++) {
      const a = A[i][k];
      if (a === 0) continue;
      for (let j = 0; j < B[0].length; j++) out[i][j] += a * B[k][j];
    }
  return out;
}
const add = (A, B) => A.map((r, i) => r.map((x, j) => x + B[i][j]));
const sub = (A, B) => A.map((r, i) => r.map((x, j) => x - B[i][j]));
function inv(A) {
  const n = A.length;
  const M = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const pv = M[c][c];
    for (let j = 0; j < 2 * n; j++) M[c][j] /= pv;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (f !== 0) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((row) => row.slice(n));
}

const F = (() => { const m = diag(new Array(8).fill(1)); for (let i = 0; i < 4; i++) m[i][i + 4] = 1; return m; })();
const H = (() => { const m = zeros(4, 8); for (let i = 0; i < 4; i++) m[i][i] = 1; return m; })();
const sq = (v) => v.map((x) => x * x);

export class KalmanFilterXYAH {
  initiate(m) {
    const h = m[3];
    const std = [2 * WP * h, 2 * WP * h, 1e-2, 2 * WP * h, 10 * WV * h, 10 * WV * h, 1e-5, 10 * WV * h];
    return { mean: [...m, 0, 0, 0, 0], cov: diag(sq(std)) };
  }
  predict(mean, cov) {
    const h = mean[3];
    const Q = diag(sq([WP * h, WP * h, 1e-2, WP * h, WV * h, WV * h, 1e-5, WV * h]));
    const m = mul(F, mean.map((x) => [x])).map((r) => r[0]);
    return { mean: m, cov: add(mul(mul(F, cov), T(F)), Q) };
  }
  update(mean, cov, z) {
    const h = mean[3];
    const R = diag(sq([WP * h, WP * h, 1e-1, WP * h]));
    const S = add(mul(mul(H, cov), T(H)), R);
    const K = mul(mul(cov, T(H)), inv(S));                       // 8x4
    const y = z.map((v, i) => v - mean[i]);
    const m = mean.map((v, i) => v + K[i].reduce((s, k, j) => s + k * y[j], 0));
    return { mean: m, cov: sub(cov, mul(mul(K, S), T(K))) };
  }
}

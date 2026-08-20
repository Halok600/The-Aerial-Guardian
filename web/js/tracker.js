/**
 * tracker.js — ByteTrack MOT, ported from src/tracker/byte_tracker.py and
 * src/tracker/kalman_filter.py. Same math, same 3-stage association, same
 * cost-threshold semantics (cost = 1 - IoU; accept if cost <= match_thresh).
 *
 * Note: camera-motion compensation (ECC) is Python/OpenCV-only in this repo
 * and is intentionally omitted here — the browser demo runs detector +
 * Kalman + ByteTrack only. See README for details.
 */
import { hungarianSolve } from './hungarian.js';

// ---------------------------------------------------------------------
// Constant-velocity Kalman filter over [cx, cy, w, h, vcx, vcy, vw, vh]
// ---------------------------------------------------------------------
class KalmanBoxFilter {
  constructor() {
    this.mean = new Float64Array(8);
    // P diagonal only is tracked as a full 8x8 for fidelity with the Python version.
    this.P = KalmanBoxFilter._diag([10, 10, 10, 10, 10000, 10000, 10000, 10000]);
    this.Q = KalmanBoxFilter._diag([1, 1, 1, 1, 0.01, 0.01, 0.0001, 0.0001]);
    this.R = KalmanBoxFilter._diag([1, 1, 10, 10]);
  }

  static _diag(vals) {
    const n = vals.length;
    const m = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) m[i][i] = vals[i];
    return m;
  }

  static _matMul(A, B) {
    const n = A.length, k = B.length, m = B[0].length;
    const out = Array.from({ length: n }, () => new Float64Array(m));
    for (let i = 0; i < n; i++)
      for (let kk = 0; kk < k; kk++) {
        const a = A[i][kk];
        if (a === 0) continue;
        for (let j = 0; j < m; j++) out[i][j] += a * B[kk][j];
      }
    return out;
  }

  static _transpose(A) {
    const n = A.length, m = A[0].length;
    const out = Array.from({ length: m }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j][i] = A[i][j];
    return out;
  }

  static _add(A, B) {
    const n = A.length, m = A[0].length;
    const out = Array.from({ length: n }, () => new Float64Array(m));
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[i][j] = A[i][j] + B[i][j];
    return out;
  }

  static _inv(A) {
    // Gauss-Jordan inversion, small matrices only (4x4 here).
    const n = A.length;
    const M = A.map((row, i) => {
      const r = new Float64Array(2 * n);
      r.set(row, 0);
      r[n + i] = 1;
      return r;
    });
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
      [M[col], M[pivot]] = [M[pivot], M[col]];
      const pv = M[col][col] || 1e-9;
      for (let j = 0; j < 2 * n; j++) M[col][j] /= pv;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
      }
    }
    return M.map((row) => row.slice(n, 2 * n));
  }

  initiate(bbox) {
    this.mean.set(bbox, 0);
    this.mean.set([0, 0, 0, 0], 4);
    const std = Math.max(bbox[2], bbox[3], 1);
    this.P = KalmanBoxFilter._diag([2 * std, 2 * std, 2 * std, 2 * std, 10 * std, 10 * std, 10 * std, 10 * std]);
  }

  predict() {
    // F: position += velocity (dt = 1)
    const next = new Float64Array(8);
    for (let i = 0; i < 4; i++) next[i] = this.mean[i] + this.mean[i + 4];
    for (let i = 4; i < 8; i++) next[i] = this.mean[i];
    this.mean = next;

    const F = KalmanBoxFilter._diag([1, 1, 1, 1, 1, 1, 1, 1]);
    for (let i = 0; i < 4; i++) F[i][i + 4] = 1;
    this.P = KalmanBoxFilter._add(KalmanBoxFilter._matMul(KalmanBoxFilter._matMul(F, this.P), KalmanBoxFilter._transpose(F)), this.Q);

    this.mean[2] = Math.max(this.mean[2], 1);
    this.mean[3] = Math.max(this.mean[3], 1);
  }

  update(z) {
    // H = [I4 | 0]
    const H = Array.from({ length: 4 }, (_, i) => {
      const r = new Float64Array(8);
      r[i] = 1;
      return r;
    });
    const Ht = KalmanBoxFilter._transpose(H);
    const S = KalmanBoxFilter._add(KalmanBoxFilter._matMul(KalmanBoxFilter._matMul(H, this.P), Ht), this.R);
    const Sinv = KalmanBoxFilter._inv(S);
    const K = KalmanBoxFilter._matMul(KalmanBoxFilter._matMul(this.P, Ht), Sinv);

    const y = new Float64Array(4);
    for (let i = 0; i < 4; i++) y[i] = z[i] - this.mean[i];

    const dx = new Float64Array(8);
    for (let i = 0; i < 8; i++) {
      let s = 0;
      for (let j = 0; j < 4; j++) s += K[i][j] * y[j];
      dx[i] = s;
    }
    for (let i = 0; i < 8; i++) this.mean[i] += dx[i];

    const KH = KalmanBoxFilter._matMul(K, H);
    const I = KalmanBoxFilter._diag([1, 1, 1, 1, 1, 1, 1, 1]);
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) I[i][j] -= KH[i][j];
    this.P = KalmanBoxFilter._matMul(I, this.P);
  }

  get bbox() { return [this.mean[0], this.mean[1], this.mean[2], this.mean[3]]; }

  get tlbr() {
    const [cx, cy, w, h] = this.bbox;
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  }
}

// ---------------------------------------------------------------------
// IoU helpers
// ---------------------------------------------------------------------
function iouMatrix(a, b) {
  const n = a.length, m = b.length;
  const out = Array.from({ length: n }, () => new Float64Array(m));
  for (let i = 0; i < n; i++) {
    const [ax1, ay1, ax2, ay2] = a[i];
    const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
    for (let j = 0; j < m; j++) {
      const [bx1, by1, bx2, by2] = b[j];
      const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
      const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
      const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
      const union = areaA + areaB - inter;
      out[i][j] = union > 0 ? inter / union : 0;
    }
  }
  return out;
}

function cxywhToTlbr(d) {
  const [cx, cy, w, h] = d;
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}

function associate(tracks, detections, costThresh) {
  if (tracks.length === 0 || detections.length === 0) {
    return { matched: [], uTracks: tracks.map((_, i) => i), uDets: detections.map((_, i) => i) };
  }
  const tBoxes = tracks.map((t) => t.kalman.tlbr);
  const dBoxes = detections.map((d) => cxywhToTlbr(d));
  const iou = iouMatrix(tBoxes, dBoxes);
  const cost = iou.map((row) => row.map((v) => 1 - v));

  const pairs = hungarianSolve(cost);
  const matched = [];
  const matchedT = new Set(), matchedD = new Set();
  for (const [r, c] of pairs) {
    if (cost[r][c] <= costThresh) {
      matched.push([r, c]);
      matchedT.add(r);
      matchedD.add(c);
    }
  }
  const uTracks = tracks.map((_, i) => i).filter((i) => !matchedT.has(i));
  const uDets = detections.map((_, i) => i).filter((i) => !matchedD.has(i));
  return { matched, uTracks, uDets };
}

// ---------------------------------------------------------------------
// STrack
// ---------------------------------------------------------------------
const TrackState = { NEW: 0, TRACKED: 1, LOST: 2, REMOVED: 3 };

class STrack {
  constructor(det) {
    this.kalman = new KalmanBoxFilter();
    this.kalman.initiate(det.slice(0, 4));
    this.score = det[4];
    this.state = TrackState.NEW;
    this.trackId = -1;
    this.age = 1;
    this.hits = 1;
    this.timeSinceUpdate = 0;
    this.history = [[Math.round(det[0]), Math.round(det[1])]];
  }

  predict() {
    this.kalman.predict();
    this.age += 1;
    this.timeSinceUpdate += 1;
  }

  update(det, nextId) {
    this.kalman.update(det.slice(0, 4));
    this.score = det[4];
    this.hits += 1;
    this.timeSinceUpdate = 0;
    const [cx, cy] = this.kalman.bbox;
    this._pushHistory(cx, cy);
    if (this.state === TrackState.NEW) {
      this.state = TrackState.TRACKED;
      this.trackId = nextId();
    }
  }

  reActivate(det) {
    this.kalman.update(det.slice(0, 4));
    this.score = det[4];
    this.hits += 1;
    this.timeSinceUpdate = 0;
    this.state = TrackState.TRACKED;
    const [cx, cy] = this.kalman.bbox;
    this._pushHistory(cx, cy);
  }

  _pushHistory(cx, cy) {
    this.history.push([Math.round(cx), Math.round(cy)]);
    if (this.history.length > 50) this.history.shift();
  }

  markLost() { this.state = TrackState.LOST; }
  get isConfirmed() { return this.state === TrackState.TRACKED && this.trackId >= 0; }
}

// ---------------------------------------------------------------------
// ByteTracker
// ---------------------------------------------------------------------
export class ByteTracker {
  constructor({ trackThresh = 0.25, lowThresh = 0.10, matchThresh = 0.80, maxTimeLost = 60 } = {}) {
    this.trackThresh = trackThresh;
    this.lowThresh = lowThresh;
    this.matchThresh = matchThresh;
    this.maxTimeLost = maxTimeLost;
    this.trackedStracks = [];
    this.lostStracks = [];
    this.frameId = 0;
    this._idCounter = 0;
  }

  reset() {
    this.trackedStracks = [];
    this.lostStracks = [];
    this.frameId = 0;
    this._idCounter = 0;
  }

  _nextId() { this._idCounter += 1; return this._idCounter; }

  /** detections: Array<[cx, cy, w, h, score]> */
  update(detections) {
    this.frameId += 1;
    const nextId = () => this._nextId();

    const highDets = detections.filter((d) => d[4] >= this.trackThresh);
    const lowDets = detections.filter((d) => d[4] >= this.lowThresh && d[4] < this.trackThresh);

    for (const t of [...this.trackedStracks, ...this.lostStracks]) t.predict();

    // Stage 1: high-conf <-> tracked
    const s1 = associate(this.trackedStracks, highDets, this.matchThresh);
    const matchedT1 = new Set(s1.matched.map(([r]) => r));
    for (const [ti, di] of s1.matched) this.trackedStracks[ti].update(highDets[di], nextId);

    // Stage 2: low-conf <-> unmatched tracked
    const remainingTracked = s1.uTracks.map((i) => this.trackedStracks[i]);
    const s2 = associate(remainingTracked, lowDets, 0.5);
    for (const [ti, di] of s2.matched) remainingTracked[ti].update(lowDets[di], nextId);

    const matchedT2 = new Set(s2.matched.map(([r]) => r));
    const newlyLost = [];
    for (let i = 0; i < remainingTracked.length; i++) {
      if (!matchedT2.has(i)) {
        remainingTracked[i].markLost();
        newlyLost.push(remainingTracked[i]);
      }
    }

    // Stage 3: unmatched high-conf <-> lost (re-activation)
    const unmatchedHigh = s1.uDets.map((i) => highDets[i]);
    const s3 = associate(this.lostStracks, unmatchedHigh, this.matchThresh);
    const recoveredSet = new Set(s3.matched.map(([r]) => r));
    for (const [ti, di] of s3.matched) this.lostStracks[ti].reActivate(unmatchedHigh[di]);

    // New tracks from unmatched-after-stage-3 high-conf detections
    const newStracks = [];
    for (const di of s3.uDets) {
      const det = unmatchedHigh[di];
      if (det[4] >= this.trackThresh) {
        const t = new STrack(det);
        t.state = TrackState.TRACKED;
        t.trackId = nextId();
        newStracks.push(t);
      }
    }

    const stage1Matched = s1.matched.map(([ti]) => this.trackedStracks[ti]);
    const stage2Matched = s2.matched.map(([ti]) => remainingTracked[ti]);
    const recovered = s3.matched.map(([ti]) => this.lostStracks[ti]);

    this.trackedStracks = [...stage1Matched, ...stage2Matched, ...recovered, ...newStracks];

    const survivingLost = this.lostStracks.filter(
      (t, i) => !recoveredSet.has(i) && t.timeSinceUpdate <= this.maxTimeLost
    );
    this.lostStracks = [...newlyLost, ...survivingLost];

    return this.trackedStracks.filter((t) => t.isConfirmed);
  }
}

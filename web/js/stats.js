/**
 * stats.js — FPS smoothing and the text shown in the live-stats panel / counters,
 * kept pure so it can be tested without a DOM.
 */
import { GROUP_ORDER } from './classes.js';

export class FpsMeter {
  constructor(windowSize = 30) { this.n = windowSize; this.times = []; }
  push(seconds) { this.times.push(seconds); if (this.times.length > this.n) this.times.shift(); }
  reset() { this.times = []; }
  fps() {
    if (!this.times.length) return 0;
    const avg = this.times.reduce((a, b) => a + b, 0) / this.times.length;
    return avg > 0 ? 1 / avg : 0;
  }
}

/** Strings for every stat element; missing inputs render as zero. */
export function statsView({ tracks = 0, fps = 0, detections = 0, totalIds = 0, counts = {} }) {
  return {
    fps: fps.toFixed(1),
    hudFps: `${fps.toFixed(1)} FPS`,
    tracks: String(tracks),
    hudTracks: `${tracks} tracked`,
    detections: String(detections),
    totalIds: String(totalIds),
    counts: Object.fromEntries(GROUP_ORDER.map((g) => [g, String(counts[g] ?? 0)])),
  };
}

/** Per-group tally of the tracks visible in one frame. */
export function groupCounts(tracks) {
  const out = Object.fromEntries(GROUP_ORDER.map((g) => [g, 0]));
  for (const t of tracks) out[t.group] += 1;
  return out;
}

/**
 * Effective PROCESSED frame rate (EMA of 1/dt over processed frames) and how far apart, in video
 * frames, consecutive processed frames are. poll() publishes at most once per `publishMs` so the
 * tracker's frame-rate setting does not jitter. `slow` = processed frames > slowFactor video
 * frames apart on average (inference slower than the video).
 */
export class ProcessingRate {
  constructor({ alpha = 0.2, publishMs = 1000, slowFactor = 1.5, minSamples = 3 } = {}) {
    Object.assign(this, { alpha, publishMs, slowFactor, minSamples });
    this.reset();
  }
  reset() { this.fps = null; this.apart = null; this.samples = 0; this.lastPublish = null; }
  push(dtSeconds, framesApart = 1) {
    if (!Number.isFinite(dtSeconds) || dtSeconds <= 0) return;
    const a = this.alpha, x = 1 / dtSeconds;
    this.fps = this.fps === null ? x : a * x + (1 - a) * this.fps;
    this.apart = this.apart === null ? framesApart : a * framesApart + (1 - a) * this.apart;
    this.samples += 1;
  }
  /** { fps, apart, slow } when there are enough samples and >= publishMs since the last publish, else null. */
  poll(nowMs) {
    if (this.samples < this.minSamples) return null;
    if (this.lastPublish !== null && nowMs - this.lastPublish < this.publishMs) return null;
    this.lastPublish = nowMs;
    return { fps: this.fps, apart: this.apart, slow: this.apart > this.slowFactor };
  }
}

/** Reports each distinct error message once (so a failing loop does not flood the console/status line). */
export class ErrorOnce {
  constructor() { this.seen = new Set(); }
  report(message) {
    if (this.seen.has(message)) return false;
    this.seen.add(message);
    return true;
  }
  reset() { this.seen.clear(); }
}

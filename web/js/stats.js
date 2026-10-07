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

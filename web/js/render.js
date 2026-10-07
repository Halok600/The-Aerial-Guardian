/**
 * render.js — pure geometry/text helpers for the overlay (no canvas, no DOM),
 * so the label/trail/filter logic is unit-tested under Node and app.js only
 * has to issue the actual canvas calls.
 */
import { CLASS_NAMES } from './classes.js';

/** "car ID:7" — class name from the track's voted class, plus its persistent ID. */
export function labelText(cls, trackId) {
  return `${CLASS_NAMES[cls]} ID:${trackId}`;
}

export function fontPx(frameHeight) {
  return Math.max(12, Math.round(frameHeight / 60));
}

/** Fading trail segments (oldest -> newest) from a track's centre history. */
export function trailSegments(history, trailLen) {
  if (trailLen < 2) return [];
  const pts = history.slice(-trailLen);
  const segs = [];
  for (let i = 1; i < pts.length; i++) {
    const a = i / (pts.length - 1);
    segs.push({
      x0: pts[i - 1][0], y0: pts[i - 1][1], x1: pts[i][0], y1: pts[i][1],
      alpha: 0.15 + 0.75 * a, width: Math.max(1, 3 * a),
    });
  }
  return segs;
}

/** Label background + text anchor for a box whose top-left is (x1, y1). */
export function labelRect(x1, y1, textWidth, font) {
  const w = textWidth + 8, h = font + 6, y = Math.max(0, y1 - h);
  return { x: x1, y, w, h, textX: x1 + 4, textY: y + h - 3 };
}

/** Tracks whose class group is currently switched on in the filter chips. */
export function visibleTracks(tracks, enabledGroups) {
  return tracks.filter((t) => enabledGroups.has(t.group));
}


/**
 * render.js — pure geometry/text helpers for the overlay (no canvas, no DOM),
 * so the label/trail/filter/sizing logic is unit-tested under Node and app.js only
 * has to issue the actual canvas calls.
 */
import { CLASS_NAMES } from './classes.js';

// On-screen (CSS pixel) targets; multiplied by the canvas->screen factor so the overlay is
// equally legible for a 640 px clip and a 4K clip, on a desktop and on a phone.
const LINE_CSS = 2;
const FONT_CSS = 12;
const PAD_CSS = 4;
const TRAIL_CSS = 3;

/** "car ID:7" — class name from the track's voted class, plus its persistent ID. */
export function labelText(cls, trackId) {
  return `${CLASS_NAMES[cls]} ID:${trackId}`;
}

/**
 * Canvas-pixel sizes that come out at fixed CSS-pixel sizes on screen.
 * canvasW/clientW: drawing-buffer width vs displayed element width. Optionally also the heights:
 * the stage uses object-fit: contain, so the limiting axis decides the real on-screen scale.
 * Missing / zero / non-finite sizes fall back to 1:1 instead of dividing by zero.
 */
export function overlayMetrics(canvasW, clientW, canvasH, clientH) {
  const ratio = (c, d) => (c > 0 && d > 0 && Number.isFinite(c) && Number.isFinite(d) ? c / d : null);
  const rw = ratio(canvasW, clientW), rh = ratio(canvasH, clientH);
  const k = rw === null ? 1 : rh === null ? rw : Math.max(rw, rh);   // canvas px per CSS px
  return { scale: k, lineWidth: LINE_CSS * k, fontPx: FONT_CSS * k, pad: PAD_CSS * k, trailWidth: TRAIL_CSS * k };
}

/** Fading trail segments (oldest -> newest) from a track's centre history; widths scale with `scale` (canvas px per CSS px). */
export function trailSegments(history, trailLen, scale = 1) {
  if (trailLen < 2) return [];
  const pts = history.slice(-trailLen);
  const segs = [];
  for (let i = 1; i < pts.length; i++) {
    const a = i / (pts.length - 1);
    segs.push({
      x0: pts[i - 1][0], y0: pts[i - 1][1], x1: pts[i][0], y1: pts[i][1],
      alpha: 0.15 + 0.75 * a, width: Math.max(1, 3 * a) * scale,
    });
  }
  return segs;
}

/** Label background + text anchor for a box whose top-left is (x1, y1). */
export function labelRect(x1, y1, textWidth, font, pad = 4) {
  const w = textWidth + 2 * pad, h = font + 1.5 * pad, y = Math.max(0, y1 - h);
  return { x: x1, y, w, h, textX: x1 + pad, textY: y + h - 0.75 * pad };
}

/** Tracks whose class group is currently switched on in the filter chips. */
export function visibleTracks(tracks, enabledGroups) {
  return tracks.filter((t) => enabledGroups.has(t.group));
}

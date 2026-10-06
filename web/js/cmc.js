/**
 * cmc.js — lightweight camera-motion compensation for the browser.
 * The Python pipeline uses OpenCV ECC; shipping OpenCV.js (~8MB) for one
 * function isn't worth it, so the demo estimates a global TRANSLATION by
 * sum-of-absolute-differences search on a 160px-wide grayscale thumbnail.
 * Panning (the dominant drone motion) is handled; rotation is not.
 *
 * Convention: estimateShift returns [dx, dy] such that curr(x) ~= prev(x - d),
 * i.e. image content moved by +d from prev to curr. GlobalMotion.update turns
 * that into a prev -> curr affine [[1,0,tx],[0,1,ty]] in source pixels.
 */
export function estimateShift(prev, curr, w, h, radius = 8) {
  let best = Infinity, bx = 0, by = 0;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      let sad = 0, n = 0;
      for (let y = radius; y < h - radius; y += 2) {
        const rowC = y * w, rowP = (y - dy) * w;
        for (let x = radius; x < w - radius; x += 2) { sad += Math.abs(curr[rowC + x] - prev[rowP + x - dx]); n++; }
      }
      const score = sad / n + 1e-3 * (dx * dx + dy * dy);   // tie-break toward small motion
      if (score < best) { best = score; bx = dx; by = dy; }
    }
  }
  return [bx, by];
}

export class GlobalMotion {
  constructor(width = 160) { this.w = width; this.prev = null; }
  reset() { this.prev = null; }
  update(source, srcW, srcH) {
    const w = this.w, h = Math.max(1, Math.round(srcH * w / srcW));
    if (!this.canvas || this.canvas.height !== h) {
      this.canvas = document.createElement('canvas');
      this.canvas.width = w; this.canvas.height = h;
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
      this.prev = null;
    }
    this.ctx.drawImage(source, 0, 0, srcW, srcH, 0, 0, w, h);
    const { data } = this.ctx.getImageData(0, 0, w, h);
    const gray = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) gray[i] = (data[4 * i] * 77 + data[4 * i + 1] * 150 + data[4 * i + 2] * 29) >> 8;
    const prev = this.prev;
    this.prev = gray;
    if (!prev) return null;
    const [dx, dy] = estimateShift(prev, gray, w, h);
    const s = srcW / w;
    return [[1, 0, dx * s], [0, 1, dy * s]];
  }
}

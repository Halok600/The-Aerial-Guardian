/**
 * frameGate.js — decide when a video frame is genuinely NEW, so the tracker sees each
 * presented frame exactly once. Pure (no DOM), unit-tested in tests/frameGate.test.mjs.
 *
 * Why: a rAF-driven loop runs at the display rate, not the video rate, so a 30 fps clip on a
 * 60 Hz screen is processed ~2x per frame (measured 78% duplicates in headless Chrome). Duplicates
 * inflate `hits` (counts confirm in one real frame), shorten max_time_lost in real time and feed
 * the Kalman filter zero-motion measurements. NOTE video.currentTime cannot be used for this in
 * Chrome: it is interpolated and advances every rAF. Use requestVideoFrameCallback's
 * metadata.mediaTime (or presentedFrames), else getVideoPlaybackQuality().totalVideoFrames.
 *
 * A "stamp" is { mediaTime?, presentedFrames?, totalVideoFrames?, currentTime? } (seconds / counts).
 */
const finite = (v) => typeof v === 'number' && Number.isFinite(v);
const WRAP_JUMP_S = 0.25;   // currentTime jumping back by more than this means the clip restarted

/** Compare one counter-like field; null when either side lacks it. */
function compare(prev, curr, key) {
  return finite(prev?.[key]) && finite(curr?.[key]) ? Math.sign(curr[key] - prev[key]) : null;
}

/**
 * Should `curr` be processed given the last PROCESSED stamp `prev`?
 * Returns { process, wrapped }; wrapped = playback went backwards (loop restart or seek back),
 * so the caller should start a fresh tracking pass. Missing fields => process (caller throttles).
 */
export function shouldProcess(prev, curr) {
  if (!prev) return { process: true, wrapped: false };
  for (const key of ['mediaTime', 'presentedFrames', 'totalVideoFrames']) {
    const c = compare(prev, curr, key);
    if (c === null) continue;
    let wrapped = c < 0;
    // Without a media timestamp a loop restart only shows up as currentTime jumping back
    // (the frame counters keep climbing across the loop).
    if (key !== 'mediaTime' && finite(prev.currentTime) && finite(curr.currentTime)
        && prev.currentTime - curr.currentTime > WRAP_JUMP_S) wrapped = true;
    return { process: c !== 0 || wrapped, wrapped };
  }
  const jumpedBack = finite(prev.currentTime) && finite(curr?.currentTime) && prev.currentTime - curr.currentTime > WRAP_JUMP_S;
  return { process: true, wrapped: jumpedBack };
}

/**
 * Time and frame spacing between two processed stamps.
 * dt: seconds of video time (mediaTime) when both stamps have it, else the wall-clock dt.
 * apart: how many video frames lie between them (presentedFrames delta, else dt * nominalFps), >= 1.
 */
export function describeStep(prev, curr, wallDt, nominalFps = 30) {
  const media = finite(prev?.mediaTime) && finite(curr?.mediaTime);
  const dt = media ? curr.mediaTime - prev.mediaTime : wallDt;
  let apart;
  if (finite(prev?.presentedFrames) && finite(curr?.presentedFrames)) apart = curr.presentedFrames - prev.presentedFrames;
  else apart = Math.round(dt * nominalFps);
  return { dt, apart: Math.max(1, apart) };
}

/** Last-resort wall-clock throttle (~30 fps) for browsers with no frame counters at all. */
export function wallClockDue(lastMs, nowMs, fps = 30) {
  return lastMs == null || nowMs - lastMs >= 1000 / fps - 2;
}

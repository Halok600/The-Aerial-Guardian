import { Detector } from './inference.js';
import { ByteTracker, DEFAULT_CONFIG } from './tracker.js';
import { GROUP_COLORS, GROUP_ORDER } from './classes.js';
import { GlobalMotion } from './cmc.js';
import { labelText, trailSegments, labelRect, visibleTracks, overlayMetrics } from './render.js';
import { FpsMeter, ProcessingRate, ErrorOnce, statsView, groupCounts, slowHintAction, errorMessage } from './stats.js';
import { shouldProcess, describeStep, wallClockDue } from './frameGate.js';

// ---------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------
const root = document.documentElement;
const themeBtn = document.getElementById('themeToggle');
function systemPrefersDark() {
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
}
function applyTheme(mode) {
  if (mode) {
    root.setAttribute('data-theme', mode);
    localStorage.setItem('ag-theme', mode);
  } else {
    root.removeAttribute('data-theme');
    localStorage.removeItem('ag-theme');
  }
}
const savedTheme = localStorage.getItem('ag-theme');
if (savedTheme) applyTheme(savedTheme);
themeBtn?.addEventListener('click', () => {
  const current = root.getAttribute('data-theme') || (systemPrefersDark() ? 'dark' : 'light');
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

// ---------------------------------------------------------------------
// Reveal-on-scroll
// ---------------------------------------------------------------------
const io = new IntersectionObserver(
  (entries) => entries.forEach((e) => { if (e.isIntersecting) e.target.classList.add('in'); }),
  { threshold: 0.12 }
);
document.querySelectorAll('.reveal').forEach((el) => io.observe(el));

// ---------------------------------------------------------------------
// ID -> color (golden-angle hue); the offline renders use a similar but not identical palette.
// Used for trails only; boxes and labels take their colour from the class group.
// ---------------------------------------------------------------------
const colorCache = new Map();
function colorForId(id) {
  if (!colorCache.has(id)) {
    const hue = (id * 137.508) % 360;
    colorCache.set(id, `hsl(${hue.toFixed(1)}, 78%, 58%)`);
  }
  return colorCache.get(id);
}

// ---------------------------------------------------------------------
// Elements
// ---------------------------------------------------------------------
const bootOverlay = document.getElementById('bootOverlay');
const bootText = document.getElementById('bootText');
const stage = document.getElementById('stage');
const stageEmpty = document.getElementById('stageEmpty');
const videoEl = document.getElementById('sourceVideo');
const imgEl = document.getElementById('sourceImage');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');

const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const webcamBtn = document.getElementById('webcamBtn');
const resetBtn = document.getElementById('resetBtn');
const sampleBtn = document.getElementById('sampleBtn');
const cmcToggle = document.getElementById('cmcToggle');
const filterChips = document.querySelectorAll('#filterChips .chip');

const fpsVal = document.getElementById('fpsVal');
const backendVal = document.getElementById('backendVal');
const trackVal = document.getElementById('trackVal');
const idVal = document.getElementById('idVal');
const detVal = document.getElementById('detVal');
const statusLine = document.getElementById('statusLine');
const hudFps = document.getElementById('hudFps');
const hudTracks = document.getElementById('hudTracks');
const recPill = document.getElementById('recPill');

const confSlider = document.getElementById('confSlider');
const confVal = document.getElementById('confVal');
const trailSlider = document.getElementById('trailSlider');
const trailVal = document.getElementById('trailVal');

const modeTabs = document.querySelectorAll('.mode-tab');

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------
let detector = null;
let modelReady = false;
const tracker = new ByteTracker(DEFAULT_CONFIG);
const motion = new GlobalMotion();
const enabledGroups = new Set(GROUP_ORDER);
const fpsMeter = new FpsMeter(30);
const rate = new ProcessingRate();       // effective processed-frame rate -> tracker.setFrameRate
const errorOnce = new ErrorOnce();
let mode = 'video';
let running = false;
let busy = false;
let rafId = null;
let rvfcHandle = null;
let webcamStream = null;
let currentObjectUrl = null;
let lastTracks = [];
let lastTrailLen = 30;
let slowShown = false;

// Bumped on every reset / source change / stop: a frame whose inference resolves after that
// belongs to an older generation and is discarded (no phantom tracks after a reset).
let generation = 0;
// Identifies the active video loop; callbacks of an older loop stop themselves.
let loopId = 0;
// Stamp of the last PROCESSED video frame (see frameGate.js) and its wall-clock time.
let lastStamp = null;
let lastRunWall = null;

const SLOW_HINT = 'Inference is slower than the video — tracking is coarser; try a faster device or a browser with WebGPU, or a shorter, low-motion clip.';

let statusKind = '';
function setStatus(msg, kind = '') {
  statusKind = kind;
  statusLine.textContent = msg;
  statusLine.className = 'status-line' + (kind ? ' ' + kind : '');
}

// The detector always runs at a low score floor so ByteTrack's low-score stage has input;
// the slider only controls the tracker's new-track threshold.
const DETECTOR_CONF = 0.05;
const RAW_NMS_IOU = 0.6;   // used only for the 'raw' export layout

function getSettings() {
  return { newTrack: parseFloat(confSlider.value), trailLen: parseInt(trailSlider.value, 10) };
}

confSlider.value = DEFAULT_CONFIG.new_track;
confVal.textContent = Number(confSlider.value).toFixed(2);
confSlider.addEventListener('input', () => { confVal.textContent = Number(confSlider.value).toFixed(2); });
trailSlider.addEventListener('input', () => { trailVal.textContent = trailSlider.value; });

/** Fresh tracking pass on the SAME source (Reset button, loop restart, seek back). */
function resetTracking() {
  generation += 1;
  tracker.reset();
  motion.reset();
  fpsMeter.reset();
  lastTracks = [];
  lastStamp = null;
}

function resetPass() {
  resetTracking();
  octx.clearRect(0, 0, overlay.width, overlay.height);
  updateStats([], 0);
}

/** A new source (file, webcam, mode switch): also forget the learned frame rate and error history. */
function beginSource() {
  resetTracking();
  rate.reset();
  errorOnce.reset();
  tracker.setFrameRate(30);
  slowShown = false;
  setStatus('');          // a previous source's error / hint must not outlive it
}

// Group filter chips hide boxes/labels/trails of a class group; counting continues.
filterChips.forEach((chip) => chip.addEventListener('click', () => {
  const g = chip.dataset.group;
  if (enabledGroups.has(g)) enabledGroups.delete(g); else enabledGroups.add(g);
  chip.classList.toggle('active', enabledGroups.has(g));
  chip.setAttribute('aria-pressed', String(enabledGroups.has(g)));
  repaint();   // still image / paused video
}));

// Toggling camera-motion compensation mid-stream must not diff against a stale thumbnail.
cmcToggle.addEventListener('change', () => motion.reset());

function repaint() { render(visibleTracks(lastTracks, enabledGroups), lastTrailLen); }
window.addEventListener('resize', repaint);   // overlay sizes are in display pixels

// ---------------------------------------------------------------------
// Boot: load model
// ---------------------------------------------------------------------
async function boot() {
  detector = new Detector();
  try {
    const backend = await detector.load((msg) => { bootText.textContent = msg; });
    backendVal.textContent = backend.toUpperCase();
    modelReady = true;
    setStatus(`Model loaded on ${backend.toUpperCase()}. Choose a source to begin.`, 'ok');
    probeSample();
  } catch (e) {
    console.error(e);
    bootText.textContent = 'Failed to load model.';
    setStatus(`Model failed to load: ${errorMessage(e)}`, 'err');
  } finally {
    bootOverlay.classList.add('hidden');
  }
}
boot();

// ---------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------
modeTabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    modeTabs.forEach((t) => t.classList.remove('active'));
    tab.classList.add('active');
    switchMode(tab.dataset.mode);
  });
});

function releaseObjectUrl() {
  if (currentObjectUrl) { URL.revokeObjectURL(currentObjectUrl); currentObjectUrl = null; }
}

function stopAll() {
  running = false;
  loopId += 1;            // stops any rVFC / rAF loop
  generation += 1;        // discards an in-flight frame
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (rvfcHandle !== null && videoEl.cancelVideoFrameCallback) videoEl.cancelVideoFrameCallback(rvfcHandle);
  rvfcHandle = null;
  // Stop whichever stream is attached (srcObject beats src, so it must be cleared before a file can play).
  const stream = videoEl.srcObject || webcamStream;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  webcamStream = null;
  videoEl.onloadedmetadata = null;
  imgEl.onload = null;
  videoEl.pause();
  videoEl.srcObject = null;
  videoEl.removeAttribute('src');
  videoEl.load();
  releaseObjectUrl();
  videoEl.style.display = 'none';
  imgEl.style.display = 'none';
  overlay.width = 0; overlay.height = 0;
  recPill.style.display = 'none';
  stageEmpty.style.display = 'flex';
  webcamBtn.textContent = 'Start Webcam';
}

function switchMode(next) {
  stopAll();
  mode = next;
  beginSource();
  updateStats([], 0);
  if (mode === 'webcam') {
    stageEmpty.innerHTML = emptyMsg('Click "Start Webcam" to begin live tracking.');
    dropzone.style.display = 'none';
    webcamBtn.style.display = 'inline-flex';
  } else {
    dropzone.style.display = 'inline-flex';
    webcamBtn.style.display = 'none';
    stageEmpty.innerHTML = emptyMsg(
      mode === 'video' ? 'Drop a video file here, or click to browse.' : 'Drop an image here, or click to browse.'
    );
  }
}

function emptyMsg(text) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M4 7l8-4 8 4-8 4-8-4z"/><path d="M4 7v10l8 4 8-4V7"/><path d="M12 11v10"/></svg><div>${text}</div>`;
}

// ---------------------------------------------------------------------
// File input / drag-drop
// ---------------------------------------------------------------------
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});
['dragover', 'dragenter'].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('drag'); })
);
['dragleave', 'drop'].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove('drag'); })
);
dropzone.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (f) loadFile(f);
});

function loadFile(file) {
  if (!modelReady) { setStatus('The detector is not loaded, so nothing can be processed.', 'err'); return; }
  stopAll();
  beginSource();
  updateStats([], 0);
  currentObjectUrl = URL.createObjectURL(file);
  const url = currentObjectUrl;

  if (file.type.startsWith('video/') || mode === 'video') {
    mode = 'video';
    setModeTab('video');
    videoEl.src = url;
    videoEl.style.display = 'block';
    videoEl.muted = true;
    videoEl.loop = true;
    stageEmpty.style.display = 'none';
    videoEl.onloadedmetadata = () => {
      overlay.width = videoEl.videoWidth;
      overlay.height = videoEl.videoHeight;
      videoEl.play();
      running = true;
      recPill.style.display = 'flex';
      startVideoLoop(false);
    };
  } else {
    mode = 'image';
    setModeTab('image');
    imgEl.src = url;
    imgEl.style.display = 'block';
    stageEmpty.style.display = 'none';
    imgEl.onload = () => {
      overlay.width = imgEl.naturalWidth;
      overlay.height = imgEl.naturalHeight;
      runOnce(imgEl, imgEl.naturalWidth, imgEl.naturalHeight);
    };
  }
}

function setModeTab(m) {
  modeTabs.forEach((t) => t.classList.toggle('active', t.dataset.mode === m));
}

// ---------------------------------------------------------------------
// Webcam
// ---------------------------------------------------------------------
webcamBtn.addEventListener('click', async () => {
  if (running) { stopAll(); switchMode('webcam'); return; }
  if (!modelReady) { setStatus('The detector is not loaded, so nothing can be processed.', 'err'); return; }
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
  } catch (e) {
    setStatus('Webcam permission denied or unavailable.', 'err');
    return;
  }
  beginSource();
  videoEl.srcObject = webcamStream;
  videoEl.style.display = 'block';
  videoEl.muted = true;
  stageEmpty.style.display = 'none';
  videoEl.onloadedmetadata = () => {
    overlay.width = videoEl.videoWidth;
    overlay.height = videoEl.videoHeight;
    videoEl.play();
    running = true;
    recPill.style.display = 'flex';
    webcamBtn.textContent = 'Stop Webcam';
    startVideoLoop(true);
  };
});

resetBtn.addEventListener('click', () => {
  resetPass();
  setStatus('Tracker state reset.', 'ok');
});

// ---------------------------------------------------------------------
// Sample clip (assets/demo/sample-raw.mp4 may not be deployed: hide the button then)
// ---------------------------------------------------------------------
const SAMPLE_URL = 'assets/demo/sample-raw.mp4';

async function probeSample() {
  try {
    const res = await fetch(SAMPLE_URL, { method: 'HEAD' });
    const type = res.headers.get('content-type') || '';
    sampleBtn.style.display = res.ok && !type.includes('text/html') ? 'inline-flex' : 'none';
  } catch (e) {
    sampleBtn.style.display = 'none';
  }
}

sampleBtn.addEventListener('click', async () => {
  sampleBtn.disabled = true;
  setStatus('Fetching sample clip…');
  try {
    const res = await fetch(SAMPLE_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    loadFile(new File([blob], 'sample-raw.mp4', { type: 'video/mp4' }));
  } catch (e) {
    console.warn('sample clip unavailable', e);
    setStatus(`Sample clip unavailable (${errorMessage(e)}). Choose your own file instead.`, 'err');
    sampleBtn.style.display = 'none';
  } finally {
    sampleBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------
// Processing loops: every PRESENTED video frame is processed at most once (see frameGate.js)
// ---------------------------------------------------------------------
function startVideoLoop(isWebcam) {
  const id = ++loopId;
  lastStamp = null;
  lastRunWall = null;
  if (typeof videoEl.requestVideoFrameCallback === 'function') {
    const onFrame = (now, meta) => {
      if (id !== loopId || !running) return;
      rvfcHandle = videoEl.requestVideoFrameCallback(onFrame);
      // Webcam streams: use the presented-frame counter only (their mediaTime is not a reliable clock).
      onVideoFrame(isWebcam
        ? { presentedFrames: meta.presentedFrames }
        : { mediaTime: meta.mediaTime, presentedFrames: meta.presentedFrames });
    };
    rvfcHandle = videoEl.requestVideoFrameCallback(onFrame);
  } else {
    // No rVFC: poll on rAF but gate on decoded-frame counters, never on currentTime (interpolated in Chrome).
    const poll = () => {
      if (id !== loopId || !running) return;
      rafId = requestAnimationFrame(poll);
      if (videoEl.readyState < 2 || videoEl.paused) return;
      const q = videoEl.getVideoPlaybackQuality?.();
      const stamp = { totalVideoFrames: q?.totalVideoFrames, currentTime: isWebcam ? undefined : videoEl.currentTime };
      // No counters at all: throttle to the nominal ~30 fps by wall clock.
      if (!Number.isFinite(stamp.totalVideoFrames) && !wallClockDue(lastRunWall, performance.now())) return;
      onVideoFrame(stamp);
    };
    rafId = requestAnimationFrame(poll);
  }
}

// A loop restart / seek back starts a fresh tracking pass (counters back to 0), like the Reset button.
videoEl.addEventListener('seeked', () => {
  if (!running || webcamStream || !lastStamp) return;
  const prevT = Number.isFinite(lastStamp.mediaTime) ? lastStamp.mediaTime : lastStamp.currentTime;
  if (Number.isFinite(prevT) && videoEl.currentTime < prevT - 0.5) resetPass();
});

function onVideoFrame(stamp) {
  if (busy) return;                                   // keep the busy gate: skip frames while inferring
  const gate = shouldProcess(lastStamp, stamp);
  if (!gate.process) return;                          // duplicate presentation of the same frame
  if (gate.wrapped) resetPass();                      // loop restart: fresh pass (also clears lastStamp)
  const now = performance.now();
  if (lastStamp && lastRunWall !== null) {
    const step = describeStep(lastStamp, stamp, (now - lastRunWall) / 1000);
    rate.push(step.dt, step.apart);
    const p = rate.poll(now);
    if (p) applyRate(p);
  }
  lastStamp = stamp;
  lastRunWall = now;
  busy = true;
  processFrame(videoEl, overlay.width, overlay.height)
    .catch(reportLoopError)
    .finally(() => { busy = false; });
}

/** Feed the effective processed frame rate to the tracker and tell the user when inference lags the video. */
function applyRate(p) {
  tracker.setFrameRate(Math.min(60, Math.max(1, p.fps)));
  // The hint needs >= minSamples timing samples (ProcessingRate.poll) and never replaces an error.
  const d = slowHintAction(p.slow, slowShown, statusKind);
  slowShown = d.slowShown;
  if (d.action === 'show') setStatus(SLOW_HINT, 'warn');
  else if (d.action === 'clear') setStatus('');
}

function reportLoopError(e) {
  const msg = errorMessage(e);
  if (errorOnce.report(msg)) {
    console.error(e);
    setStatus(`Inference error: ${msg}`, 'err');
  }
}

const RUNNING_MSG = 'Running detection…';
let runToken = 0;
let statusBeforeRun = null;
async function runOnce(source, w, h) {
  const token = ++runToken;
  // Remember what the status line said before (not our own "Running" text from an overlapping run).
  if (statusLine.textContent !== RUNNING_MSG) statusBeforeRun = { msg: statusLine.textContent, kind: statusKind };
  setStatus(RUNNING_MSG);
  try {
    resetTracking();   // a still image is always frame 1, however often the slider re-runs it
    if (await processFrame(source, w, h)) {
      setStatus('Detection complete.', 'ok');
    } else if (token === runToken && statusLine.textContent === RUNNING_MSG) {
      // Result discarded (reset / source change mid-inference) and nothing else has spoken since:
      // do not leave "Running detection…" up forever.
      setStatus(statusBeforeRun?.msg ?? '', statusBeforeRun?.kind ?? '');
    }
  } catch (e) {
    console.error(e);
    setStatus(`Inference error: ${errorMessage(e)}`, 'err');
  }
}
confSlider.addEventListener('change', () => {
  if (mode === 'image' && imgEl.style.display !== 'none') runOnce(imgEl, overlay.width, overlay.height);
});

// One ONNX session.run at a time (a second concurrent run on the same session throws), so a still-image
// re-run or a new source waits for an in-flight video frame instead of overlapping it.
let inference = Promise.resolve();
function processFrame(source, w, h) {
  const run = inference.then(() => processFrameNow(source, w, h));
  inference = run.catch(() => {});
  return run;
}

/** Returns false when the result was discarded because a reset / source change happened mid-inference. */
async function processFrameNow(source, w, h) {
  const g = generation;
  const t0 = performance.now();
  const { newTrack, trailLen } = getSettings();
  // Camera-motion compensation only makes sense between consecutive video frames. Estimated BEFORE the
  // await so it sees the same frame the detector preprocesses (detect() reads the frame synchronously).
  const warp = cmcToggle.checked && mode !== 'image' ? motion.update(source, w, h) : null;
  const dets = await detector.detect(source, w, h, { conf: DETECTOR_CONF, iou: RAW_NMS_IOU });
  if (g !== generation) return false;
  tracker.cfg.new_track = newTrack;
  const tracks = tracker.update(dets, warp);
  lastTracks = tracks;
  lastTrailLen = trailLen;
  render(visibleTracks(tracks, enabledGroups), trailLen);

  fpsMeter.push((performance.now() - t0) / 1000);
  // A single image is one frame: unique-ID counting needs several, so show the per-group tally instead.
  updateStats(tracks, fpsMeter.fps(), dets.length, mode === 'image' ? groupCounts(tracks) : tracker.uniqueCounts());
  return true;
}

// ---------------------------------------------------------------------
// Render — fading trails (per-ID tint), class-group boxes, "class ID:n" labels.
// Sizes follow the on-screen scale so lines stay >= 2 CSS px and text >= 11 CSS px on any display.
// ---------------------------------------------------------------------
function render(tracks, trailLen) {
  const w = overlay.width, h = overlay.height;
  octx.clearRect(0, 0, w, h);
  const m = overlayMetrics(w, overlay.clientWidth, h, overlay.clientHeight);
  for (const t of tracks) {                       // trails under boxes
    octx.strokeStyle = colorForId(t.trackId);
    for (const s of trailSegments(t.history, trailLen, m.scale)) {
      octx.globalAlpha = s.alpha;
      octx.lineWidth = s.width;
      octx.beginPath(); octx.moveTo(s.x0, s.y0); octx.lineTo(s.x1, s.y1); octx.stroke();
    }
  }
  octx.globalAlpha = 1;
  octx.font = `600 ${m.fontPx}px "JetBrains Mono", monospace`;
  octx.textBaseline = 'bottom';
  for (const t of tracks) {
    const [x1, y1, x2, y2] = t.tlbr;
    const color = GROUP_COLORS[t.group];
    octx.strokeStyle = color; octx.lineWidth = m.lineWidth;
    octx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = labelText(t.cls, t.trackId);
    const r = labelRect(x1, y1, octx.measureText(label).width, m.fontPx, m.pad, w);
    octx.fillStyle = color; octx.fillRect(r.x, r.y, r.w, r.h);
    octx.fillStyle = '#ffffff'; octx.fillText(label, r.textX, r.textY);
  }
}

// ---------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------
function updateStats(tracks, fps, detCount = 0, counts = {}) {
  const v = statsView({ tracks: tracks.length, fps, detections: detCount, totalIds: tracker.totalIds, counts });
  fpsVal.textContent = v.fps;
  hudFps.textContent = v.hudFps;
  trackVal.textContent = v.tracks;
  hudTracks.textContent = v.hudTracks;
  detVal.textContent = v.detections;
  idVal.textContent = v.totalIds;
  for (const g of GROUP_ORDER) document.getElementById(`count-${g}`).textContent = v.counts[g];
}

// Init default mode UI
switchMode('video');

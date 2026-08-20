import { Detector } from './inference.js';
import { ByteTracker } from './tracker.js';

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
// ID -> color (golden-angle hue), mirrors src/utils/visualizer.py
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
let tracker = new ByteTracker();
let mode = 'video';
let running = false;
let busy = false;
let rafId = null;
let webcamStream = null;
let trails = new Map();
let fpsWindow = [];
let lastTick = 0;
let totalIdsSeen = 0;

function setStatus(msg, kind = '') {
  statusLine.textContent = msg;
  statusLine.className = 'status-line' + (kind ? ' ' + kind : '');
}

function getSettings() {
  const conf = parseFloat(confSlider.value);
  return {
    confThresh: conf,
    iouThresh: 0.45,
    trailLen: parseInt(trailSlider.value, 10),
  };
}

confSlider.addEventListener('input', () => { confVal.textContent = confSlider.value; });
trailSlider.addEventListener('input', () => { trailVal.textContent = trailSlider.value; });

// ---------------------------------------------------------------------
// Boot: load model
// ---------------------------------------------------------------------
async function boot() {
  detector = new Detector();
  try {
    const backend = await detector.load((msg) => { bootText.textContent = msg; });
    backendVal.textContent = backend.toUpperCase();
    setStatus(`Model loaded on ${backend.toUpperCase()}. Choose a source to begin.`, 'ok');
  } catch (e) {
    console.error(e);
    bootText.textContent = 'Failed to load model — see console.';
    setStatus('Model failed to load. Try a different browser (Chrome/Edge recommended).', 'err');
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

function stopAll() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  videoEl.pause();
  videoEl.removeAttribute('src');
  videoEl.style.display = 'none';
  imgEl.style.display = 'none';
  overlay.width = 0; overlay.height = 0;
  recPill.style.display = 'none';
  stageEmpty.style.display = 'flex';
}

function switchMode(next) {
  stopAll();
  mode = next;
  tracker.reset();
  trails.clear();
  totalIdsSeen = 0;
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
  stopAll();
  tracker.reset();
  trails.clear();
  totalIdsSeen = 0;
  const url = URL.createObjectURL(file);

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
      loopVideo();
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
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
  } catch (e) {
    setStatus('Webcam permission denied or unavailable.', 'err');
    return;
  }
  tracker.reset();
  trails.clear();
  totalIdsSeen = 0;
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
    loopVideo();
  };
});

resetBtn.addEventListener('click', () => {
  tracker.reset();
  trails.clear();
  totalIdsSeen = 0;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  updateStats([], 0);
  setStatus('Tracker state reset.', 'ok');
});

// ---------------------------------------------------------------------
// Processing loops
// ---------------------------------------------------------------------
async function loopVideo() {
  if (!running) return;
  if (!busy && videoEl.readyState >= 2 && !videoEl.paused) {
    busy = true;
    try {
      await processFrame(videoEl, overlay.width, overlay.height);
    } catch (e) {
      console.error(e);
    }
    busy = false;
  }
  rafId = requestAnimationFrame(loopVideo);
}

async function runOnce(source, w, h) {
  setStatus('Running detection…');
  try {
    await processFrame(source, w, h);
    setStatus('Detection complete.', 'ok');
  } catch (e) {
    console.error(e);
    setStatus('Inference error — see console.', 'err');
  }
}
confSlider.addEventListener('change', () => {
  if (mode === 'image' && imgEl.style.display !== 'none') runOnce(imgEl, overlay.width, overlay.height);
});

async function processFrame(source, w, h) {
  const t0 = performance.now();
  const { confThresh, iouThresh, trailLen } = getSettings();
  const dets = await detector.detect(source, w, h, { confThresh, iouThresh });
  // Keep the tracker's high/low confidence buckets in lockstep with the
  // user-facing confidence slider, otherwise a low slider value surfaces
  // raw detections that the tracker's (higher, hardcoded) thresholds would
  // silently refuse to ever promote into a confirmed track.
  tracker.trackThresh = confThresh;
  tracker.lowThresh = confThresh * 0.4;
  const tracks = tracker.update(dets);
  render(tracks, trailLen);

  const dt = (performance.now() - t0) / 1000;
  fpsWindow.push(dt);
  if (fpsWindow.length > 30) fpsWindow.shift();
  const avg = fpsWindow.reduce((a, b) => a + b, 0) / fpsWindow.length;
  const fps = avg > 0 ? 1 / avg : 0;

  updateStats(tracks, fps, dets.length);
}

// ---------------------------------------------------------------------
// Render — boxes, fading trails, ID labels (mirrors visualizer.py)
// ---------------------------------------------------------------------
function render(tracks, trailLen) {
  const w = overlay.width, h = overlay.height;
  octx.clearRect(0, 0, w, h);

  const activeIds = new Set();
  for (const t of tracks) {
    activeIds.add(t.trackId);
    if (!trails.has(t.trackId)) trails.set(t.trackId, []);
    const arr = trails.get(t.trackId);
    const [cx, cy] = t.kalman.bbox;
    arr.push([cx, cy]);
    while (arr.length > trailLen) arr.shift();
  }
  for (const id of Array.from(trails.keys())) {
    if (!activeIds.has(id)) trails.delete(id);
  }

  // Trails first (under boxes)
  for (const t of tracks) {
    const pts = trails.get(t.trackId);
    if (!pts || pts.length < 2) continue;
    const color = colorForId(t.trackId);
    for (let i = 1; i < pts.length; i++) {
      const alpha = i / pts.length;
      octx.strokeStyle = color;
      octx.globalAlpha = 0.15 + 0.65 * alpha;
      octx.lineWidth = Math.max(1, 3 * alpha);
      octx.beginPath();
      octx.moveTo(pts[i - 1][0], pts[i - 1][1]);
      octx.lineTo(pts[i][0], pts[i][1]);
      octx.stroke();
    }
  }
  octx.globalAlpha = 1;

  // Boxes + labels
  octx.font = '600 15px "JetBrains Mono", monospace';
  octx.textBaseline = 'bottom';
  for (const t of tracks) {
    const [cx, cy, tw, th] = t.kalman.bbox;
    const x1 = cx - tw / 2, y1 = cy - th / 2;
    const color = colorForId(t.trackId);

    octx.strokeStyle = color;
    octx.lineWidth = 2.5;
    octx.strokeRect(x1, y1, tw, th);

    const label = `ID ${t.trackId}`;
    const metrics = octx.measureText(label);
    const lw = metrics.width + 12, lh = 20;
    const ly = Math.max(0, y1 - lh);
    octx.fillStyle = color;
    octx.fillRect(x1, ly, lw, lh);
    octx.fillStyle = '#04141a';
    octx.fillText(label, x1 + 6, ly + lh - 4);
  }
}

// ---------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------
function updateStats(tracks, fps, detCount = 0) {
  fpsVal.textContent = fps.toFixed(1);
  hudFps.textContent = `${fps.toFixed(1)} FPS`;
  trackVal.textContent = tracks.length;
  hudTracks.textContent = `${tracks.length} tracked`;
  detVal.textContent = detCount;
  const maxId = tracks.reduce((m, t) => Math.max(m, t.trackId), 0);
  totalIdsSeen = Math.max(totalIdsSeen, maxId, tracker._idCounter || 0);
  idVal.textContent = totalIdsSeen;
}

// Init default mode UI
switchMode('video');

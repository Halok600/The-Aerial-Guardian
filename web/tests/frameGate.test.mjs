import { test } from 'node:test';
import assert from 'node:assert/strict';
import { shouldProcess, describeStep, wallClockDue } from '../js/frameGate.js';
import { ProcessingRate, ErrorOnce } from '../js/stats.js';

// ------------------------------------------------------------ shouldProcess
test('first frame is always processed and is not a wrap', () => {
  assert.deepEqual(shouldProcess(null, { mediaTime: 0.5 }), { process: true, wrapped: false });
  assert.deepEqual(shouldProcess(undefined, {}), { process: true, wrapped: false });
});

test('same mediaTime is a duplicate: not processed', () => {
  assert.deepEqual(shouldProcess({ mediaTime: 1.2 }, { mediaTime: 1.2 }), { process: false, wrapped: false });
});

test('advanced mediaTime is processed', () => {
  assert.deepEqual(shouldProcess({ mediaTime: 1.2 }, { mediaTime: 1.2333 }), { process: true, wrapped: false });
});

test('mediaTime going backwards (loop wrap / seek back) is processed and flagged wrapped', () => {
  assert.deepEqual(shouldProcess({ mediaTime: 9.9 }, { mediaTime: 0.03 }), { process: true, wrapped: true });
});

test('mediaTime wins over presentedFrames when both are present', () => {
  assert.equal(shouldProcess({ mediaTime: 2, presentedFrames: 10 }, { mediaTime: 2, presentedFrames: 11 }).process, false);
  assert.equal(shouldProcess({ mediaTime: 2, presentedFrames: 10 }, { mediaTime: 2.04, presentedFrames: 10 }).process, true);
});

test('presentedFrames-only metadata (webcam streams): equal -> skip, advanced -> process', () => {
  assert.deepEqual(shouldProcess({ presentedFrames: 7 }, { presentedFrames: 7 }), { process: false, wrapped: false });
  assert.deepEqual(shouldProcess({ presentedFrames: 7 }, { presentedFrames: 9 }), { process: true, wrapped: false });
});

test('totalVideoFrames (getVideoPlaybackQuality fallback) is used when nothing better exists', () => {
  assert.equal(shouldProcess({ totalVideoFrames: 40 }, { totalVideoFrames: 40 }).process, false);
  assert.equal(shouldProcess({ totalVideoFrames: 40 }, { totalVideoFrames: 41 }).process, true);
});

test('fallback path detects a loop wrap from currentTime jumping back (frame counter keeps climbing)', () => {
  const r = shouldProcess({ totalVideoFrames: 300, currentTime: 9.9 }, { totalVideoFrames: 301, currentTime: 0.02 });
  assert.deepEqual(r, { process: true, wrapped: true });
  // ordinary playback: no wrap
  assert.equal(shouldProcess({ totalVideoFrames: 300, currentTime: 1.0 }, { totalVideoFrames: 301, currentTime: 1.03 }).wrapped, false);
});

test('no frame counters at all: a currentTime jump back still flags the loop wrap', () => {
  assert.deepEqual(shouldProcess({ currentTime: 9.9 }, { currentTime: 0.04 }), { process: true, wrapped: true });
  assert.deepEqual(shouldProcess({ currentTime: 1.0 }, { currentTime: 1.02 }), { process: true, wrapped: false });
});

test('missing fields on either side: process (the caller throttles by wall clock)', () => {
  assert.deepEqual(shouldProcess({}, {}), { process: true, wrapped: false });
  assert.deepEqual(shouldProcess({ mediaTime: 1 }, {}), { process: true, wrapped: false });
  assert.deepEqual(shouldProcess({ mediaTime: NaN }, { mediaTime: NaN }), { process: true, wrapped: false });
});

// ------------------------------------------------------------ describeStep
test('describeStep: file video uses mediaTime for dt and presentedFrames for frames-apart', () => {
  const s = describeStep({ mediaTime: 1.0, presentedFrames: 30 }, { mediaTime: 1.1, presentedFrames: 33 }, 0.2);
  assert.ok(Math.abs(s.dt - 0.1) < 1e-9);
  assert.equal(s.apart, 3);
});

test('describeStep: mediaTime only -> frames apart from the nominal 30 fps', () => {
  const s = describeStep({ mediaTime: 1.0 }, { mediaTime: 1.1333 }, 0.2);
  assert.ok(Math.abs(s.dt - 0.1333) < 1e-9);
  assert.equal(s.apart, 4);
});

test('describeStep: webcam (presentedFrames only) uses the wall-clock dt', () => {
  const s = describeStep({ presentedFrames: 100 }, { presentedFrames: 102 }, 0.0667);
  assert.equal(s.dt, 0.0667);
  assert.equal(s.apart, 2);
});

test('describeStep: nothing usable -> wall-clock dt, at least 1 frame apart', () => {
  const s = describeStep({}, {}, 0.01);
  assert.equal(s.dt, 0.01);
  assert.equal(s.apart, 1);
});

// ------------------------------------------------------------ wallClockDue
test('wallClockDue throttles to ~30 fps by default', () => {
  assert.equal(wallClockDue(null, 1000), true);
  assert.equal(wallClockDue(1000, 1016), false);
  assert.equal(wallClockDue(1000, 1034), true);
  assert.equal(wallClockDue(1000, 1045, 20), false);   // 20 fps -> 50 ms (2 ms rAF-jitter tolerance)
  assert.equal(wallClockDue(1000, 1050, 20), true);
});

// ------------------------------------------------------------ ProcessingRate
test('ProcessingRate: nothing is published until it has a few samples', () => {
  const r = new ProcessingRate();
  assert.equal(r.poll(0), null);
  r.push(0.1, 3); r.push(0.1, 3);
  assert.equal(r.poll(5000), null);
});

test('ProcessingRate: EMA of 1/dt, flags slow when frames are > 1.5 apart on average', () => {
  const r = new ProcessingRate({ alpha: 0.5 });
  for (let i = 0; i < 6; i++) r.push(0.1, 3);
  const p = r.poll(2000);
  assert.ok(Math.abs(p.fps - 10) < 1e-9);
  assert.equal(p.apart, 3);
  assert.equal(p.slow, true);
});

test('ProcessingRate: real-time processing (1 frame apart) is not slow', () => {
  const r = new ProcessingRate();
  for (let i = 0; i < 5; i++) r.push(1 / 30, 1);
  const p = r.poll(2000);
  assert.ok(Math.abs(p.fps - 30) < 1e-9);
  assert.equal(p.slow, false);
});

test('ProcessingRate: publishes at most once per second', () => {
  const r = new ProcessingRate();
  for (let i = 0; i < 5; i++) r.push(0.04, 1);
  assert.ok(r.poll(2000));
  r.push(0.04, 1);
  assert.equal(r.poll(2500), null);
  assert.ok(r.poll(3001));
});

test('ProcessingRate ignores non-positive / non-finite dt and resets cleanly', () => {
  const r = new ProcessingRate();
  r.push(0, 1); r.push(-1, 1); r.push(NaN, 1);
  for (let i = 0; i < 5; i++) r.push(0.05, 1);
  assert.ok(Math.abs(r.poll(2000).fps - 20) < 1e-9);
  r.reset();
  assert.equal(r.poll(9000), null);
});

test('ProcessingRate EMA moves toward new rates', () => {
  const r = new ProcessingRate({ alpha: 0.5 });
  for (let i = 0; i < 4; i++) r.push(0.05, 1);      // 20 fps
  for (let i = 0; i < 6; i++) r.push(0.1, 3);       // 10 fps
  const p = r.poll(2000);
  assert.ok(p.fps > 10 && p.fps < 11, `fps ${p.fps}`);
});

// ------------------------------------------------------------ ErrorOnce
test('ErrorOnce reports each distinct message once until reset', () => {
  const e = new ErrorOnce();
  assert.equal(e.report('boom'), true);
  assert.equal(e.report('boom'), false);
  assert.equal(e.report('other'), true);
  assert.equal(e.report('other'), false);
  e.reset();
  assert.equal(e.report('boom'), true);
});

// ------------------------------------------------------------ slow-hint gating / error text
test('ProcessingRate: a slow first step (model warm-up) does not flash the slow hint', () => {
  const r = new ProcessingRate();
  r.push(0.4, 12);                                    // first step: warm-up outlier
  r.push(1 / 30, 1); r.push(1 / 30, 1);
  const p = r.poll(2000);
  assert.ok(p);
  assert.equal(p.slow, false);
  assert.ok(Math.abs(p.fps - 30) < 1e-9);
});

test('ProcessingRate: no slow verdict before minSamples, and a genuinely slow source is still flagged after', () => {
  const r = new ProcessingRate();
  r.push(0.3, 9); r.push(0.3, 9);
  assert.equal(r.poll(5000), null);
  r.push(0.3, 9);
  assert.equal(r.poll(5000).slow, true);
});

test('slowHintAction shows / clears the hint and never overwrites an error status', async () => {
  const { slowHintAction } = await import('../js/stats.js');
  assert.deepEqual(slowHintAction(true, false, ''), { slowShown: true, action: 'show' });
  assert.deepEqual(slowHintAction(true, true, 'warn'), { slowShown: true, action: null });
  assert.deepEqual(slowHintAction(false, true, 'warn'), { slowShown: false, action: 'clear' });
  assert.deepEqual(slowHintAction(false, true, 'ok'), { slowShown: false, action: null });   // someone else's status stays
  assert.deepEqual(slowHintAction(false, false, ''), { slowShown: false, action: null });
  // error status: refuse to show, and do not latch slowShown so the hint can appear once the error is gone
  assert.deepEqual(slowHintAction(true, false, 'err'), { slowShown: false, action: null });
  assert.deepEqual(slowHintAction(false, true, 'err'), { slowShown: false, action: null });
});

test('errorMessage copes with Errors, strings, null and message-less values', async () => {
  const { errorMessage } = await import('../js/stats.js');
  assert.equal(errorMessage(new Error('boom')), 'boom');
  assert.equal(errorMessage('plain string'), 'plain string');
  assert.equal(errorMessage(null), 'null');
  assert.equal(errorMessage(undefined), 'undefined');
  assert.equal(errorMessage(42), '42');
  assert.equal(errorMessage({ message: 'from object' }), 'from object');
  assert.equal(errorMessage({ code: 7 }), '[object Object]');
});

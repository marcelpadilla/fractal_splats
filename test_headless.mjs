// Checks for everything in the renderer that does not need a GPU: invariant
// moments, descent targets, the kernel normalization, the refinement cut, the
// Mandelbrot field, rebasing, and the whole descent driven through the real
// camera loop.
//
// Runs against the real script realm, not a vm sandbox. See harness.mjs: a
// contextified global made every builtin lookup go through an interceptor and
// made the refinement kernel look fifteen times slower than it is, which is how
// a wrong bottleneck ended up in the written record.
import { api, get } from './harness.mjs';

const {
  loadPreset, buildCut, advanceCamera, pickTarget, rebaseAll, rebaseInto, logZoom,
  measureDimension, kernelConst, mandelField, projectToBoundary, camFromAngles,
  cfg, cam, anchor, stats, pool, job, emitted, fld, PRESETS,
} = api;

let fails = 0;
const ok = (label, cond, detail = '') => {
  console.log((cond ? 'PASS  ' : 'FAIL  ') + label + (detail ? '   ' + detail : ''));
  if (!cond) fails++;
};
const W = 1600, H = 900;
// `instances` is reallocated when the budget grows, so it has to be read live out
// of the script scope rather than captured once.
const totalLit = n => {
  const inst = get('instances');
  let s = 0;
  for (let q = 0, o = 12; q < n; q++, o += get('FLOATS')) s += inst[o];
  return s;
};
const head = s => console.log('\n--- ' + s + ' ' + '-'.repeat(Math.max(0, 66 - s.length)));

/* ---------- 1. the Sierpinski tetrahedron, which is known exactly ---------- */
head('1. invariant measure against closed form');
loadPreset('sierpinski', false);
cfg.autopilot = false;
const root = get('root');
ok('mean is the origin', Math.max(...[0, 1, 2].map(i => Math.abs(root.mu[i]))) < 1e-12,
  'mu = [' + [0, 1, 2].map(i => root.mu[i].toExponential(2)).join(', ') + ']');
ok('covariance is exactly I/3',
  Math.abs(root.cov[0] - 1 / 3) < 1e-6 && Math.abs(root.cov[3] - 1 / 3) < 1e-6 &&
  Math.abs(root.cov[5] - 1 / 3) < 1e-6 && Math.abs(root.cov[1]) < 1e-9,
  'diag = ' + [root.cov[0], root.cov[3], root.cov[5]].map(v => v.toFixed(9)).join(', '));
ok('information dimension is exactly 2', Math.abs(get('mDim') - 2) < 1e-12,
  'D = ' + get('mDim').toFixed(12));
// The fixed point of x -> (x+v)/2 is the vertex v.
const tg = get('targets').slice(0, 4).map(t => get('fixWord')(get('maps'), t.word));
const isVertex = t => t.every(c => Math.abs(Math.abs(c) - 1) < 1e-12);
ok('single map fixed points are the four vertices', tg.every(isVertex),
  JSON.stringify(tg[0].map(v => +v.toFixed(12))));

/* ------------------------- 2. the splat kernel --------------------------- */
head('2. super Gaussian kernel normalization');
// Numerically integrate the kernel in the plane and check both the area factor
// and the second moment, which is the property that makes sharpening honest.
function kernelMoments(beta) {
  const k = kernelConst(beta);
  let area = 0, m2 = 0;
  const dr = 1e-4, rmax = k.extent * 1.4;
  for (let r = dr / 2; r < rmax; r += dr) {
    const v = Math.exp(-0.5 * Math.pow(k.shape * r * r, beta / 2));
    area += v * 2 * Math.PI * r * dr;
    m2 += v * r * r * 2 * Math.PI * r * dr;
  }
  return { area, meanSq: m2 / area, knorm: k.knorm, extent: k.extent };
}
for (const beta of [2, 3, 4, 6]) {
  const m = kernelMoments(beta);
  // peak = knorm / sqrt(det); with Sigma = I the integral must come to 1
  ok('beta ' + beta + ': unit integral and E[q] = 2',
    Math.abs(m.area * m.knorm - 1) < 2e-3 && Math.abs(m.meanSq - 2) < 6e-3,
    'integral ' + (m.area * m.knorm).toFixed(4) + ', E[q] ' + m.meanSq.toFixed(4) +
    ', support ' + m.extent.toFixed(2) + ' sigma');
}

/* --------------------------- 3. the cut ---------------------------------- */
head('3. the refinement cut');
loadPreset('sierpinski', false);
cfg.autopilot = false;
let n = buildCut(W, H);
ok('opening cut is non-empty', n > 5000, n + ' splats, depth ' + stats.dmin + '-' + stats.dmax);
// The weight that was UPLOADED, not the node's branch weight. A node inside the cross fade
// band is drawn alongside its children, so summing branch weights counts that part of the
// measure twice: the invariant is about the drawn weight and it is still exact. Read out of the
// instance buffer and divided by the baked exposure, which is what makes it order one.
const instW = get('instances');
const wsum = Array.from(emitted.subarray(0, get('nEmit')))
  .reduce((a, i, q) => a + instW[q * FLOATS + 12], 0) / get('job').norm;
ok('drawn weights sum to 1 (a cut partitions the measure, cross fade included)',
  Math.abs(wsum - 1) < 1e-6,
  'sum = ' + wsum.toFixed(12) + ', culled ' + stats.culled);
// The invariant refinement maintains is on VISIBLE error, footprint times the
// depth of field attenuation, not on footprint alone. Splats whose raw footprint
// exceeds the threshold but whose visible error does not are the whole point of the
// change: they are the far field the shader has already dimmed, left coarse on
// purpose so the budget goes to what can be seen.
const idx = Array.from(emitted.subarray(0, get('nEmit')));
// A splat mid cross fade is ABOVE the threshold by construction: it is splitting, and it is
// drawn at the complementary weight while its children fade in. So the invariant is about
// splats that are oversized and drawn at FULL weight.
//
// The fade fraction is not `pool.bl`, which is the blend a node INHERITED from its ancestors: a
// faded parent inherits 1 and is drawn at 1-t, so reading bl reported every one of them as
// unfaded. Read what was actually uploaded and divide out the branch weight and the baked
// exposure, which is the fraction that was really drawn.
const drawnFrac = q => instW[q * FLOATS + 12] / (pool.w[idx[q]] * get('job').norm);
// And the threshold to compare against is the TOP of the fade band, not the split threshold.
// The fade fraction is `t = (prio/splitPx - 1) / (FADE_BAND - 1)`, so a node whose priority
// only just exceeds splitPx has t near zero: it is splitting, its children are drawn at
// almost nothing, and it is itself drawn at almost its full weight. That is not a violation,
// it is the first frame of the dissolve. Comparing against splitPx reported 23 of 302 301
// splats as oversized and unexplained for a week; every one of them was inside 0.045 percent
// of the threshold, which is exactly the band where `1 - t > 0.999`. Above splitPx *
// FADE_BAND the parent is not drawn at all, so that is where the invariant has teeth.
const fadeTop = cfg.splitPx * get('FADE_BAND');
const overs = idx.filter((i, q) => pool.prio[i] > fadeTop * 1.0001 && drawnFrac(q) > 0.999).length;
const oversRaw = idx.filter(i => pool.fp[i] > cfg.splitPx * 1.0001).length;
const fading = idx.filter((i, q) => drawnFrac(q) < 0.999).length;
console.log('    left coarse by the depth of field: ' + oversRaw + ' of ' + n +
  ' exceed ' + cfg.splitPx + ' px of footprint, ' + overs +
  ' exceed the top of the fade band in visible error while drawn whole, ' +
  fading + ' are visibly mid cross fade');
// The budget binds in two places and either explains an oversized splat. The per node
// reservation refuses a split when the cut is already full; the level threshold from
// prepareLevel refuses one when the level as a whole cannot fit. Comparing against the emitted
// count instead is wrong twice over: culled children and cross faded parents both make the
// emitted count and the cut size diverge legitimately.
const jb = get('job');
const budgetBound = jb.cutSize + jb.nmaps > jb.cap || jb.bLow !== -Infinity;
ok('oversized splats only where the budget binds', overs === 0 || budgetBound,
  overs + ' over in visible error, of ' + n + ', budget ' + cfg.budget);
// Children must nest inside their parent. Wrong composition order still looks
// right in a still frame, so this is checked by containment and by descent.
const bad = (() => {
  let worst = 0;
  for (let q = 0; q < get('nEmit'); q++) {
    const i = emitted[q], p = pool.par[i];
    if (p < 0) continue;
    const d = Math.hypot(pool.mu[i * 3] - pool.mu[p * 3],
      pool.mu[i * 3 + 1] - pool.mu[p * 3 + 1], pool.mu[i * 3 + 2] - pool.mu[p * 3 + 2]);
    const rp = Math.sqrt(pool.cov[p * 6] + pool.cov[p * 6 + 3] + pool.cov[p * 6 + 5]);
    worst = Math.max(worst, d / rp);
  }
  return worst;
})();
ok('every child centre lies inside its parent', bad < 1.5,
  'worst |mu_child - mu_parent| / sigma_parent = ' + bad.toFixed(3));
ok('budget respected for every object', ['sierpinski', 'dragon', 'cantor', 'mandelbrot'].every(p => {
  loadPreset(p, false); cfg.autopilot = false; cfg.budget = 60000;
  return buildCut(W, H) <= 60000;
}), 'all four under a 60k cap');

/* ------------------------ 4. the Mandelbrot field ------------------------ */
head('4. the Mandelbrot field');
// The distance estimate must bracket the true distance. Koebe gives
// de/4 <= d <= de, so test against a brute force nearest boundary search along
// the ray from a known exterior point.
loadPreset('mandelbrot', false);
mandelField(0.35, 0, 4000, 0, 1, 0, 0);
// The nearest boundary point to 0.35+0i on the real axis is the cusp at 0.25, so
// the true distance is 0.1. The estimate is only asymptotically a distance: the
// practical guarantee is a factor of four either way, and the cusp is a parabolic
// point where it is at its worst.
ok('distance estimate is within a factor of four at the cusp',
  fld.de >= 0.1 / 4 && fld.de <= 0.1 * 4,
  'de = ' + fld.de.toFixed(5) + ', true distance 0.1, ratio ' + (fld.de / 0.1).toFixed(2));
mandelField(-0.5, 0, 4000, 0, 1, 0, 0);
ok('a point in the main cardioid is interior', fld.inside === 1 && fld.h === 0,
  'inside = ' + fld.inside);
mandelField(-1.0, 0.05, 4000, 0, 1, 0, 0);
ok('a point in the period two bulb is interior', fld.inside === 1, 'inside = ' + fld.inside);
// Every target must land on the boundary: distance estimate below the pixel the
// camera could ever resolve there.
const tgs = get('targets');
let worstDE = 0;
for (const t of tgs) {
  mandelField(t[0], t[1], 40000, 0, 1, 0, 0);
  if (!fld.inside) worstDE = Math.max(worstDE, fld.de);
}
// The Newton step uses the estimate, so it converges to the accuracy of the
// estimate and not further. 1e-8 is four decades below the deepest zoom the
// terrain is good for, so it is on the boundary as far as any frame can tell.
ok('all ' + tgs.length + ' descent targets are on the boundary', worstDE < 1e-8,
  'worst distance estimate ' + worstDE.toExponential(2));
// The analytic normal must agree with a finite difference of the height.
{
  const c = [-0.9, 0.28], e = 1e-7, hc = 1.4;
  mandelField(c[0], c[1], 4000, 0, 0.42, 0, hc);
  const gx = fld.gx, gz = fld.gz;
  mandelField(c[0] + e, c[1], 4000, 0, 0.42, 0, hc); const hxp = fld.h;
  mandelField(c[0] - e, c[1], 4000, 0, 0.42, 0, hc); const hxm = fld.h;
  mandelField(c[0], c[1] + e, 4000, 0, 0.42, 0, hc); const hzp = fld.h;
  mandelField(c[0], c[1] - e, 4000, 0, 0.42, 0, hc); const hzm = fld.h;
  const fx = (hxp - hxm) / (2 * e), fz = (hzp - hzm) / (2 * e);
  // The analytic gradient assumes |grad de| = 1, which is exact for a true
  // distance function and only asymptotically true of the estimate. So the
  // DIRECTION is right, which is what carries the shading, while the magnitude is
  // idealized to a uniform slope. Test the direction and report the magnitude.
  const ca = (gx * fx + gz * fz) / Math.max(1e-30, Math.hypot(gx, gz) * Math.hypot(fx, fz));
  ok('analytic normal points the same way as a central difference', ca > 0.985,
    'cosine ' + ca.toFixed(5) + '; |analytic| ' + Math.hypot(gx, gz).toFixed(4) +
    ' against |difference| ' + Math.hypot(fx, fz).toFixed(4) +
    ' (slope idealized to a uniform cone field, by design)');
}
// The derivative bailout must never change a verdict that matters: a cell it
// rejects must genuinely be within rTol of the set.
{
  let checked = 0, wrong = 0;
  for (let k = 0; k < 3000; k++) {
    const cx = -0.75 + 0.02 * ((k * 0.6180339887) % 1 - 0.5);
    const cz = 0.13 + 0.02 * ((k * 0.3819660113) % 1 - 0.5);
    const rTol = 1e-4;
    mandelField(cx, cz, 20000, rTol, 1, 0, 0);
    if (!fld.edge) continue;
    checked++;
    mandelField(cx, cz, 200000, 0, 1, 0, 0);       // no bailout, ground truth
    if (!fld.inside && fld.de > rTol * 1.5) wrong++;
  }
  ok('derivative bailout never rejects a cell that was not near the set',
    checked > 20 && wrong === 0, checked + ' cells took the bailout, ' + wrong + ' wrongly');
}

/* ----------------------- 5. rebasing and precision ---------------------- */
head('5. rebasing');
// Without rebasing the position error reaches a pixel at a definite zoom. With
// it, the error does not grow at all, because no coordinate does.
function wallDecade(mode) {
  cfg.precision = mode; cfg.rebase = false;
  loadPreset('sierpinski', false);
  cfg.precision = mode; cfg.rebase = false; cfg.autopilot = false;
  cam.target.set(get('targets')[0].word ? get('fixWord')(get('maps'), [0]) : [1, 1, 1]);
  let lo = 0, hi = 20;
  for (let it = 0; it < 60; it++) {
    const mid = (lo + hi) / 2;
    cam.dist = cam.startDist / Math.pow(10, mid);
    buildCut(W, H);
    if (stats.errPx > 1) hi = mid; else lo = mid;
  }
  return lo;
}
const w32 = wallDecade('f32'), w64 = wallDecade('f64');
console.log('  one pixel of position error at 1600x900, 45 degree fov, single frame:');
console.log('    float32 world positions: zoom 1e' + w32.toFixed(1));
console.log('    float64 world positions: zoom 1e' + w64.toFixed(1));
ok('float32 wall between 1e3 and 1e6 without rebasing', w32 > 3 && w32 < 6, '1e' + w32.toFixed(2));
ok('float64 wall past 1e12 without rebasing', w64 > 12, '1e' + w64.toFixed(2));

// A rebase must not change the picture. It drops content in the parent but
// outside the new walk root, and the trigger is set so that content is already
// past the depth of field, so the emitted set must agree either side.
{
  loadPreset('sierpinski', false);
  cfg.autopilot = false; cfg.precision = 'f64'; cfg.budget = 60000;
  cfg.rebase = false;                                  // hold it off, then step by hand
  const zFar = 1 + 8 / cfg.fog;
  cam.dist = get('suppE') / (zFar * 4) * 0.98;         // just past the trigger
  cam.target.set(cam.goal);
  const before = buildCut(W, H);
  // What the shader receives is w * norm, so that is what a rebase has to preserve.
  // The raw weights cannot be compared either side: after a rebase they are
  // relative to the new walk root, which carries p_anchor of the old measure.
  const litBefore = totalLit(before);
  const depthBefore = anchor.depth;
  cfg.rebase = true;
  rebaseAll();
  const after = buildCut(W, H);
  const litAfter = totalLit(after);
  ok('a rebase actually fired', anchor.depth > depthBefore,
    'anchor depth ' + depthBefore + ' -> ' + anchor.depth);
  ok('splat count survives a rebase', Math.abs(after - before) / before < 0.10,
    before + ' -> ' + after + ' splats');
  ok('the light the shader receives survives a rebase',
    Math.abs(litAfter - litBefore) / litBefore < 0.06,
    litBefore.toFixed(4) + ' -> ' + litAfter.toFixed(4) +
    ' (' + ((litAfter / litBefore - 1) * 100).toFixed(2) + ' percent)');
}

/* ---------- 6. the whole descent, through the real camera loop ---------- */
// Requirement one of the project. advanceCamera runs at 60 Hz and the cut is
// rebuilt every third step, which is what the frame loop actually does. This is
// the only test that has ever caught composition order, the depth of field cull,
// the aim invariant, or the aim point drifting under rebasing: every one of them
// left still frames looking perfectly correct.
head('6. autopilot descent, 60 Hz camera, cut rebuilt every third step');
function descend(preset, rebase, seconds, rate, budget) {
  loadPreset(preset, false);
  cfg.autopilot = true; cfg.rate = rate; cfg.budget = budget;
  cfg.precision = 'f64'; cfg.rebase = rebase;
  let minN = 1e9, empty = 0, maxLZ = -1e9, maxDepth = 0, worstErr = 0, prev = -1e9;
  // A turnaround is the descent reaching the precision wall and climbing back out. It
  // replaces parking, which replaced restarting: a restart faded the whole frame to black
  // on the way out, and parking stopped, which is not something a demo should ever do.
  let climbs = 0, wasClimb = false;
  for (let step = 0; step < 60 * seconds; step++) {
    advanceCamera(1 / 60);
    if (step % 3) continue;
    const cnt = buildCut(W, H);
    const c = get('climbing');
    if (c && !wasClimb) climbs++;
    wasClimb = c;
    prev = stats.logZoom;
    if (stats.logZoom > 1) {
      if (cnt < minN) minN = cnt;
      if (cnt < 500) empty++;
      if (stats.errPx > worstErr) worstErr = stats.errPx;
    }
    if (stats.logZoom > maxLZ) { maxLZ = stats.logZoom; maxDepth = stats.dmax; }
  }
  return { minN, empty, maxLZ, maxDepth, worstErr, climbs };
}
for (const p of ['sierpinski', 'dragon', 'cantor']) {
  const off = descend(p, false, 12, 3.0, 30000);
  const on = descend(p, true, 12, 3.0, 30000);
  console.log('  ' + p);
  console.log('    single frame  deepest 1e' + off.maxLZ.toFixed(1).padStart(6) +
    '   worst err ' + off.worstErr.toExponential(1) + '   turnarounds ' + off.climbs);
  console.log('    rebasing      deepest 1e' + on.maxLZ.toFixed(1).padStart(6) +
    '   worst err ' + on.worstErr.toExponential(1) + '   tree depth ' + on.maxDepth +
    '   rebases ' + stats.rebases);
  ok('  ' + p + ' descends with the frame populated', on.empty === 0 && on.minN > 2000,
    'min ' + on.minN + ' splats, ' + on.empty + ' empty frames');
  // Both runs fly for the same simulated time, so the depths only differ by whatever the
  // single frame run lost at the precision wall. The real claim is that rebasing never
  // hits a wall at all, which is section 7. The single frame run must reach one and turn
  // around there; the rebasing run must never reach one. Turning around replaced parking,
  // which replaced restarting: a restart faded the frame to black, and parking stopped,
  // and a demo that stops looks broken.
  ok('  ' + p + ' single frame turns around at its wall, rebasing never reaches one',
    off.climbs > 0 && on.climbs === 0,
    'turnarounds: single frame ' + off.climbs + ', rebasing ' + on.climbs);
  ok('  ' + p + ' rebasing holds the position error flat', on.worstErr < 1e-8,
    'worst ' + on.worstErr.toExponential(2) + ' px against ' + off.worstErr.toExponential(2));
}
// The terrain has no self similarity to rebase into, so it is only checked for a
// populated descent inside its own float64 range.
{
  const t = descend('mandelbrot', false, 10, 2.0, 40000);
  console.log('  mandelbrot terrain (no rebasing: not self similar)');
  console.log('    deepest 1e' + t.maxLZ.toFixed(1) + '   quadtree depth ' + t.maxDepth +
    '   min splats ' + t.minN + '   max iterations ' + stats.maxIter);
  ok('  terrain descends past six decades with the frame populated',
    t.maxLZ > 6 && t.empty === 0 && t.minN > 2000,
    'reached 1e' + t.maxLZ.toFixed(1) + ', ' + t.empty + ' empty frames');
}

/* ----------------------- 7. an unbounded descent ------------------------ */
head('7. how far rebasing actually goes');
{
  loadPreset('sierpinski', false);
  cfg.autopilot = true; cfg.rate = 24; cfg.budget = 25000; cfg.rebase = true;
  const marks = [10, 100, 1000], seen = [];
  let empty = 0, minN = 1e9, worst = 0;
  for (let step = 0; step < 60 * 200; step++) {
    advanceCamera(1 / 60);
    if (step % 6) continue;
    const cnt = buildCut(W, H);
    if (stats.logZoom > 1) {
      if (cnt < minN) minN = cnt;
      if (cnt < 500) empty++;
      if (stats.errPx > worst) worst = stats.errPx;
    }
    while (seen.length < marks.length && stats.logZoom >= marks[seen.length]) {
      seen.push('1e' + marks[seen.length] + ': ' + cnt + ' splats, local dist ' +
        cam.dist.toExponential(2) + ', anchor depth ' + anchor.depth +
        ', err ' + stats.errPx.toExponential(2) + ' px');
    }
  }
  for (const s of seen) console.log('    ' + s);
  console.log('    final 1e' + stats.logZoom.toFixed(0) + ' after ' + stats.rebases + ' rebases');
  ok('descends past 1e1000 with the frame populated',
    stats.logZoom > 1000 && empty === 0 && minN > 2000,
    'reached 1e' + stats.logZoom.toFixed(0) + ', min ' + minN + ' splats, ' + empty + ' empty');
  ok('local coordinates never leave order one',
    Math.hypot(cam.target[0], cam.target[1], cam.target[2]) < 10 &&
    cam.dist / cam.startDist > 1e-4,
    '|target| = ' + Math.hypot(cam.target[0], cam.target[1], cam.target[2]).toFixed(4) +
    ', dist/startDist = ' + (cam.dist / cam.startDist).toExponential(2));
  ok('position error does not grow with depth', worst < 1e-6,
    'worst ' + worst.toExponential(2) + ' px over the whole flight');
}

/* --------------------------- 8. what it costs --------------------------- */
head('8. cost of a rebuild, best of eight');
console.log('  node ' + process.version + ', single thread, ' + W + 'x' + H);
console.log('  object        splats   split    walked    rebuild   per splat');
for (const p of ['mandelbrot', 'sierpinski', 'dragon', 'cantor']) {
  loadPreset(p, false); cfg.autopilot = false;
  let best = 1e9, cnt = 0;
  for (let k = 0; k < 8; k++) { cnt = buildCut(W, H); best = Math.min(best, stats.buildMs); }
  console.log('  ' + PRESETS[p].name.slice(0, 13).padEnd(14) + String(cnt).padStart(7) +
    cfg.splitPx.toFixed(2).padStart(8) + String(stats.visited).padStart(10) +
    (best.toFixed(1) + ' ms').padStart(11) + (best * 1e6 / cnt).toFixed(0).padStart(8) + ' ns');
}
{
  let it = 0;
  const t0 = performance.now();
  for (let k = 0; k < 240000; k++) {
    mandelField(-2.2 + 3.0 * ((k * 0.6180339887) % 1), -1.5 + 3.0 * ((k * 0.3819660113) % 1),
      420, 1e-3, 0.42, 0, 1.4);
    it += fld.iters;
  }
  const ms = performance.now() - t0;
  console.log('  Mandelbrot iteration: ' + (it / ms / 1e3).toFixed(0) + ' M iterations/s (' +
    it + ' iterations in ' + ms.toFixed(1) + ' ms)');
}

/* ------------- 9. the frame that actually reaches the screen ------------- */
// Everything above tests the CUT. This section tests the IMAGE, through the CPU
// rasterizer in frames.mjs, which reproduces both GL passes including the surface
// depth prepass and its 24 bit depth buffer.
//
// It exists because a terrain frame went completely black with 250 000 splats sitting
// in the buffer while every check above passed. The prepass writes the front surface
// pushed back by a slab and the colour pass keeps what is in front of that; when the
// push rounded to zero in the depth buffer, every surfel failed its own test. Nothing
// that inspects the cut can see that, and neither could an unquantized model of the
// depth buffer.
head('9. the image, both GL passes, prepass and 24 bit depth included');
{
  const { rasterize, frameStats, upload, runFrames, BW, BH } = await import('./frames.mjs');
  const reprojection = get('reprojection');
  // The per pixel GPU objects are skipped here and in the descent below, and it is not a gap in
  // the coverage: they have no cut, no splats and no instance buffer, so `buildCut` and this
  // whole rasterizer model are measuring a code path they never take. What they do have that can
  // be wrong is the image to plane homography, and section 9b checks that directly.
  for (const p of Object.keys(PRESETS)) {
    if (PRESETS[p].direct) continue;
    loadPreset(p, false);
    cfg.autopilot = false;
    buildCut(BW, BH);
    upload();
    reprojection();
    const r = rasterize();
    const st = frameStats();
    const rej = r.occluded / Math.max(r.frags + r.occluded, 1);
    console.log('    ' + PRESETS[p].name.padEnd(24) + 'cover ' + (st.cover * 100).toFixed(1) +
      '%  mean ' + st.mean.toFixed(4) + '  prepass rejected ' + (rej * 100).toFixed(1) + '%');
    // Coverage floors are per object because they are set by DIMENSION, not by the
    // renderer. The folded Koch curve is dimension 1.262 and covers well under one
    // percent of the frame however well it is drawn; asserting four percent of it would
    // be asserting that it is not a curve. Brightness is the real "is it black" test.
    const floor = { koch: 0.003 }[p] || 0.04;
    ok('  ' + p + ' opening frame is lit', st.cover > floor && st.mean > 0.02,
      'cover ' + (st.cover * 100).toFixed(1) + '%, mean ' + st.mean.toFixed(4) +
      ', floor ' + (floor * 100).toFixed(1) + '%');
    ok('  ' + p + ' depth prepass does not reject the whole surface', rej < 0.75,
      (rej * 100).toFixed(1) + '% of fragments rejected');
  }
  // And through a descent at the coarse time step the browser actually runs at under a
  // virtual time budget, which is where the black terrain frame appeared. A frame whose
  // brightness falls under a fifth of the median has gone dark.
  // rate is set explicitly: earlier sections leave cfg.rate at 24 e-folds a second,
  // which at dt 0.1 is 2.4 e-folds PER FRAME and flies the terrain straight through its
  // precision wall before the park can see it. loadPreset does not reset rate, because
  // no preset view sets it.
  for (const p of Object.keys(PRESETS)) {
    if (PRESETS[p].direct) continue;
    const rows = runFrames(p, 90, { dt: 0.1, cfg: { rate: 0.8 } });
    const mean = rows.map(r => r.mean).sort((a, b) => a - b);
    const med = mean[mean.length >> 1];
    const dark = rows.filter(r => r.mean < med * 0.2).length;
    const empty = rows.filter(r => r.splats < 500).length;
    console.log('    ' + PRESETS[p].name.padEnd(24) + 'descent to 10^' +
      rows[rows.length - 1].zoom.toFixed(2) + '  median brightness ' + med.toFixed(4) +
      '  darkest ' + Math.min(...rows.map(r => r.mean)).toFixed(4));
    ok('  ' + p + ' stays lit through a coarse step descent', dark === 0 && empty === 0,
      dark + ' dark frames, ' + empty + ' empty cuts of ' + rows.length);
  }
  // A flat field has no self similarity to rebase into, so its precision wall is real and
  // it must TURN AROUND there: climb back out, re-aim, and dive again, without ever fading,
  // restarting, or flying on into an empty frame. Long enough to get there, at a sane rate.
  //
  // The previous version of this check asserted that the zoom HELD STILL near the wall, and
  // it was flaky between runs for a good reason: refinement is sliced against the wall
  // clock, so the cut differs run to run and so does the exact frame the wall is reached
  // on. A turnaround is a structural event, not a threshold on a noisy quantity.
  for (const p of ['mandelbrot', 'mandel2d']) {
    const rows = runFrames(p, 900, { dt: 0.05, cfg: { rate: 1.6 } });
    const deepest = Math.max(...rows.map(r => r.zoom));
    const iDeep = rows.findIndex(r => r.zoom === deepest);
    const after = rows.slice(iDeep + 1);
    const climbed = after.length ? deepest - Math.min(...after.map(r => r.zoom)) : 0;
    const med = [...rows.map(r => r.mean)].sort((a, b) => a - b)[rows.length >> 1];
    const dark = rows.filter(r => r.mean < med * 0.2).length;
    const empty = rows.filter(r => r.splats < 500).length;
    console.log('    ' + PRESETS[p].name.padEnd(18) + ' turns around at 10^' +
      deepest.toFixed(1) + ', climbs back ' + climbed.toFixed(1) + ' decades, darkest ' +
      Math.min(...rows.map(r => r.mean)).toFixed(4));
    ok('  ' + p + ' turns around at its wall instead of stopping or going dark',
      deepest > 8 && climbed > 1 && dark === 0 && empty === 0,
      'deepest 10^' + deepest.toFixed(1) + ', climbed ' + climbed.toFixed(2) +
      ', ' + dark + ' dark, ' + empty + ' empty');
  }
}

// The per pixel GPU path, which shares nothing with the cut but the camera. Two things about it
// can be wrong in a way no picture reveals, and both are checked here rather than looked at.
head('9b. the per pixel path: its image to plane map is exactly the inverse of place()');
{
  const directMap = get('directMap');
  const directPlaneAt = get('directPlaneAt');
  const basis = get('basis');
  const updateBasis = get('updateBasis');
  const advanceCamera = api.advanceCamera;
  const W = 1101, H = 731;                       // deliberately not square and not a round number
  for (const p of ['mandelgpu', 'juliagpu']) {
    for (const decades of [0, 3, 6]) {
      loadPreset(p, false);
      cfg.autopilot = true;
      cfg.rate = 0.8;
      let guard = 0;
      while (api.logZoom() < decades && guard++ < 20000) advanceCamera(0.05);
      cfg.autopilot = false;
      updateBasis();
      const m = directMap(W, H);
      // Compose the two directions and demand the identity. Forward is exactly what `place`
      // computes: eye = R (world - pos), then sx = W/2 + focal ex/ez, sy = H/2 - focal ey/ez.
      const R = basis.R, pos = basis.pos;
      let worst = 0;
      for (let iy = 0; iy <= 8; iy++) for (let ix = 0; ix <= 8; ix++) {
        const sx = (ix + 0.5) / 9 * W, sy = (iy + 0.5) / 9 * H;
        const q = directPlaneAt(m, sx, sy);
        const dx = q[0] - pos[0], dy = 0 - pos[1], dz = q[1] - pos[2];
        const ex = R[0] * dx + R[1] * dy + R[2] * dz;
        const ey = R[3] * dx + R[4] * dy + R[5] * dz;
        const ez = -(R[6] * dx + R[7] * dy + R[8] * dz);
        const bx = W * 0.5 + m.focal * ex / ez, by = H * 0.5 - m.focal * ey / ez;
        worst = Math.max(worst, Math.hypot(bx - sx, by - sy));
      }
      // And the pixel size the shader measures its bailouts against, checked against a finite
      // difference of the same map rather than against the formula that produced it.
      const a = directPlaneAt(m, W * 0.5, H * 0.5);
      const b = directPlaneAt(m, W * 0.5 + 1, H * 0.5);
      const fd = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const rel = Math.abs(fd - m.pix) / Math.max(fd, 1e-300);
      console.log('    ' + p.padEnd(10) + ' 10^' + api.logZoom().toFixed(2).padStart(5) +
        '  round trip ' + worst.toExponential(1) + ' px   pixel size ' + m.pix.toExponential(3) +
        ' vs finite difference ' + fd.toExponential(3) + '  (' + (rel * 100).toFixed(3) + '%)');
      ok('  ' + p + ' at 10^' + decades + ': pixel to plane inverts place to under 1e-6 px',
        worst < 1e-6, 'worst round trip ' + worst.toExponential(2) + ' px');
      ok('  ' + p + ' at 10^' + decades + ': reported pixel size matches the map',
        rel < 1e-6, (rel * 100).toFixed(5) + '% off the finite difference');
    }
  }
  // And the precision wall. This path's arithmetic is coarser than the CPU path's, so it must turn
  // around SOONER, and it must still turn around rather than fly on into a frame where every pixel
  // is the same quantized value. Driven through advanceCamera alone, with no cut and no raster,
  // which is exactly what the browser does for this object.
  for (const p of ['mandelgpu', 'mandelbrot']) {
    loadPreset(p, false);
    cfg.autopilot = true; cfg.rate = 1.6;
    let deepest = 0, climbed = 0, turned = false;
    for (let k = 0; k < 4000; k++) {
      advanceCamera(0.05);
      const z = api.logZoom();
      if (z > deepest) deepest = z;
      if (z < deepest - 0.2) turned = true;
      if (turned) climbed = Math.max(climbed, deepest - z);
      if (climbed > 1.5) break;
    }
    console.log('    ' + p.padEnd(10) + ' turns around at 10^' + deepest.toFixed(1) +
      ', climbs back ' + climbed.toFixed(1) + ' decades');
    ok('  ' + p + ' turns around at its own wall', turned && deepest > 8 && climbed > 1,
      'deepest 10^' + deepest.toFixed(1) + ', climbed ' + climbed.toFixed(2));
  }
}

console.log('\n' + (fails ? fails + ' CHECK(S) FAILED' : 'all checks passed'));
process.exitCode = fails ? 1 : 0;

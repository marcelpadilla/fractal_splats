/*
 * Top level state, visible to every other src/ file in the concatenated bundle: the flat node
 * arena the refinement walks, the camera and anchor frames, cfg and preset loading, the palette
 * tables, and the anchor rebasing.
 *
 * Two frames are in play. World is the preset's own; anchor-local puts the walk root at scale
 * one, with anchor.M, anchor.t and anchor.logScale mapping back, and everything below the anchor
 * block is anchor-local. src/35_perf.js comes first because the arena is sized from the device
 * tier before any frame is drawn.
 */

/* ============================== node pool =============================== */
// One rebuild's worth of tree nodes in flat arrays, addressed by node index. Reset per rebuild,
// never reallocated, so refinement does no garbage collection work.
//
// IFS: a node has a branch word u = (i1 ... in) and carries F_u = w_i1 o ... o w_in, with M the
// linear part, mu = M mu_root + t and Sigma = M Sigma_root M^T. The new map composes on the
// inside, which is what makes P_{u.i} = F_u(w_i(A)) a subset of P_u = F_u(A); composing outside
// breaks containment and with it frustum culling.
//
// Terrain: the same arrays hold a quadtree over the c plane, M as the scratch slots
// [cx, cz, r, de, nu, inside|edge, nx, ny, nz] and `cc` as the unboosted visible error.
//
// f64 for `mu`, a world position, for `cov` and `M`, which accumulate a product down the branch,
// and for `w`, a branch weight reaching 8^-45 on the Cantor cube, under the smallest normal
// Float32. The rest is pixel space against thresholds of order one. 204 bytes a node.

// MAXCAP is the node arena, MAXBUDGET the splats a cut may draw. Interior nodes are never freed,
// so an n map tree with L leaves holds about L*n/(n-1) nodes: 1.14x for the eight map Cantor
// cube, 2x for the two map dragon, 1.34x for a four way quadtree.
const MAXCAP = Math.round(perfArenaBudget() * 1.34);
// Splat budget, sized from the device at load by perfArenaBudget in src/35_perf.js: the arena is
// allocated once, cannot be resized, and 204 bytes a node is 273 MB at the desktop tier. Measured
// at 1e2 on a 1100x1280 seahorse valley frame against a 4x4 supersampled per pixel reference, a
// 500 000 budget gave 177 059 splats and rms 0.068 and 1 200 000 gave 750 043 and rms 0.057, and
// only the second separates the arms of the spiral. A cut can be rationed by prepareLevel's
// priority threshold long before it reaches the hard cap, so "budget limited: no" in the readout
// does not mean the cut was unconstrained.
const MAXBUDGET = perfArenaBudget();
// What speed 1 on the slider means, in e-folds a second. The slider's own number is unitless: a
// per second here is the exponent of the camera distance, not frames, zoom factor or distance.
const RATE1 = 0.32;
// Linear up to SPD_KNEE, the range a descent is watched in, and geometric above it, where the
// useful values are 8, 20 and 64. Measured with `speed_probe.mjs` at handed in dt: the dragon
// reaches 10^133 at speed 64 and 10^2084 at speed 1000, both with 0.000 px position error and
// the full budget standing, since the rebasing is exact whatever a frame covers. The Mandelbrot
// stops at 10^13.9 at any speed, its own double-single wall.
const SPD_MIN = 0.10;        // slowest the slider offers. Not zero: stopping is the run button
const SPD_KNEE = 2;          // linear below here, geometric above
const SPD_MAX = 64;          // top of the slider
const SPD_KNEE_X = 0.62;     // fraction of the track given to the linear part
const SPD_TYPED_MAX = 1000;  // top of the typed value, which is the measured limit
// Largest projected 1-sigma extent, in pixels, a measure splat may keep: refinement splits
// anything over it however dim, and the shader fades out what is over it anyway. A surface is
// exempt, since a big surfel near the camera is the ground. See `job.attFloor` in startJob.
let SIZE_CAP = 5.0;
// Added to the hue of every splat, in turns. Zero in every shipped frame; it exists for a looping
// zoom video, where one contraction deeper rotates the frame by that map's hue increment, -0.06
// turns on the folded dragon, and ramping HUE_OFF over the loop cancels the drift exactly.
// Nothing in the viewer sets it; loopgen.mjs computes the ramp from mapHue.
let HUE_OFF = 0;
// How far past the exp(-8) fog attenuation the depth of field reaches. Exactly 1 in every shipped
// frame; an evaluation knob, so the same pose at 1 and at 2 differences to what the cull discards.
let ZF_MUL = 1;
// Width of the cross fade band across a split, as a ratio of priority above the split threshold.
// Nodes inside it are drawn twice, once as themselves at 1-t and once as their children at t, so
// it costs the fraction of the cut that sits in the band. Wider is smoother and dearer.
const FADE_BAND = 1.45;
// The same cross fade at the budget threshold, which prepareLevel expresses as a priority
// threshold bLow. bLow can sit far above splitPx when the budget is tight, so without this a node
// near it flips between itself and its children, at full weight either way, whenever bLow moves.
const BUD_BAND = 1.7;

// exp(-x) for the aerial attenuation, tabulated: every node placed evaluates it, and on the
// folded dragon Math.exp was 21 percent of a rebuild, 10.5 ms against 8.3. At ZF_MUL = 1 the
// argument is bounded by 8, since a node past zFar = 1 + 8/fog is culled and fog*(zFar-1) is
// exactly 8 whatever the fog, so one table covers every object and every zoom. Linear
// interpolation over a step of 8/1024 leaves 8e-6 relative error.
const ATT_N = 1024, ATT_SC = ATT_N / 8;
const ATT = new Float64Array(ATT_N + 2);
for (let i = 0; i <= ATT_N + 1; i++) ATT[i] = Math.exp(-i / ATT_SC);
function attOf(x) {
  if (!(x > 0)) return 1;
  const u = x * ATT_SC;
  if (u >= ATT_N) return 0;
  const j = u | 0, f = u - j;
  return ATT[j] + (ATT[j + 1] - ATT[j]) * f;
}
const pool = {
  M: new Float64Array(MAXCAP * 9),
  mu: new Float64Array(MAXCAP * 3),
  cov: new Float64Array(MAXCAP * 6),
  rel: new Float32Array(MAXCAP * 3),   // camera-relative position, scaled units
  w: new Float64Array(MAXCAP),
  cc: new Float32Array(MAXCAP),        // colour coordinate, mixed down the branch
  depth: new Int32Array(MAXCAP),
  sx: new Float32Array(MAXCAP),        // projected centre, pixels from top left
  sy: new Float32Array(MAXCAP),
  fp: new Float32Array(MAXCAP),        // projected 1-sigma extent, pixels
  prio: new Float32Array(MAXCAP),      // visible error: footprint times attenuation
  ls: new Float32Array(MAXCAP),        // log10 of the piece's own diameter, anchor relative
  bl: new Float32Array(MAXCAP),        // cross fade weight through a split, see stepIFS
  par: new Int32Array(MAXCAP),         // parent node, for reconstructing a word
  mi: new Int32Array(MAXCAP),          // which map made this node
  n: 0,
};
let frontier = new Int32Array(MAXCAP);
let nextF = new Int32Array(MAXCAP);
const emitted = new Int32Array(MAXBUDGET + 8);
let nEmit = 0;

/* =============================== state ================================= */
const cfg = {
  preset: 'dragon',
  budget: 250000,
  splitPx: 1.1,
  density: 1.0,        // optical depth scale
  glow: 0.15,          // 0 = emission absorption, 1 = Draves log density
  kernel: 3.4,         // super Gaussian exponent beta
  sigma: 1.0,          // covariance scale, on top of the exact covariance
  fog: 0.55,
  precision: 'f64',    // f32 emulates every shipping splat renderer
  autopilot: true,
  // Descent speed, in e-folds per second. The slider shows a unitless number with 1 at the
  // default; RATE1 is what speed 1 means. See the slider table in src/70_ui.js.
  rate: 0.32,
  rebase: true,        // exact anchor rebasing: unbounded zoom
  relief: 0.42,        // terrain slope
  iters: 420,          // terrain base iteration count
  contour: 0.55,       // terrain equipotential contours
  height: 0,           // 0 = distance, 1 = potential
  power: 2,            // plane: the exponent p of z -> z^p + c, the family parameter
  cx: -0.8,            // plane, Julia: the family parameter c, live
  cy: 0.156,
  gauss: 0,            // show the individual Gaussians. See GAUSS_PX
};

/* --------------------------- show the Gaussians -------------------------- */
// At a working split threshold a splat is under about a pixel by construction, so the primitives
// are not separable. This mode raises the split threshold to GAUSS_PX, taking a cut from a few
// hundred thousand splats to a few thousand tens of pixels across, keeps the outline one neutral
// colour so a splat reads on any ground, and tints the interior with the splat's own colour so an
// ellipse says which piece it came from. The refinement itself is unchanged, only told that a
// splat may be large.
//
// Absolute pixels, not a multiple of the object's own threshold, which runs 0.34 px (Koch) to
// 1.4 px (plane). At 2.5 px a cut is 24.5k splats on the dragon, 31.8k on the Sierpinski
// tetrahedron, 32.8k on the carpet and 33.8k on the Cantor cube at 989x1082. Finer fails on the
// outline, which cannot be narrower than the pixel grid: on a six pixel ellipse, half widths of
// 0.25, 0.35, 0.45 and 0.75 px all came back a white texture. So the full frame shows the
// distribution and the magnified squares show one primitive. Sweep it with `gs_probe.mjs` on the
// CPU, since through a headless capture the autopilot descends and each run reads a new depth.
// `gp=` in the query sets it.
let GAUSS_PX = 2.5;
const GAUSS_CAP = 70;
// The plane objects need their own size, and they need the split test to be about size at all. A
// plane cell splits on how much of the colour ramp it spans, with size only as the outer cap
// `maxPx`, so here maxPx has to be the primitive size or a flat region, which on a Julia set is
// most of the frame, is tiled with GAUSS_CAP sized discs: at 70 px the cut was 1 376 splats
// against the 24k to 34k the iterated function systems draw.
//
// Eight times coarser than the measure objects, because an IFS cut covers a thin set while a plane
// cut tiles the whole frame. Each cell carries a one sigma outline about a pixel wide, so the
// outline's share of the frame is about 4/cell in pixels: a fifth at 20 px, a third at 12, over
// half at 7, where the capture comes back a white mesh with the object gone. At 20 px both plane
// objects give 15 984 splats, quantized by quadtree level and so not monotone in the threshold.
// Measured with `gs_probe.mjs --plane` at 989x1082; frames in `evaluation/2026-08-12_gauss_*`.
// `gpp=` in the query sets it.
let GAUSS_PX_PLANE = 20;
// The outline's half width, in pixels, measured in the shader with a screen space derivative; see
// the note on `dp` in src/60_gl.js. It cannot be a kernel value, which is a fixed number of
// sigmas: that is 0.19 px on the six pixel ellipses this mode draws and aliases to a dotted crawl.
let GAUSS_RING = 0.20;
// How much of the splat's own colour the interior carries; the outline stays neutral. The tone
// map divides accumulated colour by accumulated alpha, so at 0.022 the ring owns every pixel it
// touches and the tint is invisible. 0.13 reads as its own hue and still loses to the ring.
const GAUSS_FILL = 0.13;

// Two fixed magnified squares in a corner, aimed at `cam.goal` and not at the frame centre: the
// camera looks at `cam.target`, which eases in from the object's centre of mass, and for an open
// curve that centre is off the curve, while `cam.goal` is the fixed point of a branch word for an
// IFS and a boundary point for a field, so it lies on the object. The second magnification is the
// square of the first, so frame to square one is the same step as square one to square two.
const LOUPE_MAG = [3.2, 10.24];
const MULT = String.fromCharCode(215);   // the times sign, for the squares' labels
const LOUPE_GAP = 0.020;    // between the two squares, as a fraction of the shorter side
const LOUPE_FRAC = 0.30;    // side of the square, as a fraction of the shorter side of the frame
const LOUPE_MIN = 100;      // and its bounds in device pixels, so it survives a phone and a 4K panel
const LOUPE_MAX = 340;
const LOUPE_PAD = 0.022;    // margin from the corner, same units as LOUPE_FRAC
// How many ellipses a square has to show across its width to be worth drawing, which is a
// different question from whether it fits: on a 366x473 iframe both fitted and the second showed
// three ellipses on a black ground. A square of side s at magnification m over a cut split at t
// pixels shows s/(m t) ellipses across, and 6.5 is the fewest that reads as a field rather than
// as two blobs, so at t = GAUSS_PX that is 52 px for the x3.2 square and 166 px for the x10.24
// one. Stated against GAUSS_PX, since the plane objects do not split on size; see GAUSS_PX_PLANE.
const LOUPE_ROOM = 6.5;
// The square also has to clear the controls, pinned to the other top corner and a fixed 178 CSS
// px wide on a narrow page: 178 for the panel, 100 for the square at its floor, 8 for the margin
// and a little air. CSS pixels, because that is the unit the collision happens in: the panel is
// laid out by the stylesheet while the canvas backing store is a tier decision. See loupeRect.
const LOUPE_CLEAR = 292;
// The frame around the magnified square and the box around the region it came from, both a white
// line between two black ones, the only marking that reads on a dense field of saturated colour.
const MARK_COL = new Float32Array([0.98, 0.94, 0.86]);
const BLACK_COL = new Float32Array([0.0, 0.0, 0.0]);
// The outline's colour. Where a splat stops being drawn here is three line widths outside the
// contour, in pixels, and lives in the shader with the line it belongs to; a kernel value is a
// fixed number of sigmas, so on a small splat that cut falls inside the line and clips it.
// `uCore` is still a uniform because the surface depth prepass has its own use for it.
const RING_COL = new Float32Array([1.0, 0.96, 0.88]);
function ringWidth() { return cfg.gauss ? GAUSS_RING : 0; }

const cam = {
  target: new Float64Array(3),
  goal: new Float64Array(3),
  // World to eye rotation, rows = right, up, back. A full matrix rather than a yaw and a pitch,
  // because a rebase multiplies the camera by the linear part of an IFS map and two Euler angles
  // cannot represent the result; the leftover roll would show as a twitch at every rebase.
  R: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
  dist: 3,
  startDist: 3,
  fov: 45 * Math.PI / 180,
  level: false,        // keep the horizon level (terrain) or free orbit (IFS)
  // The aim point, named symbolically. `goal` is only a cached fixWord(maps, rotateWord(goalWord,
  // goalPhase)); see fixWord for why it is recomputed rather than transported through a rebase.
  goalWord: [],
  goalPhase: 0,
};

// The anchor. Below here everything is in anchor-local coordinates, where the walk root is the
// whole attractor at scale one and the camera sits at a distance of order the framing distance.
// `M` and `t` are the composed affine map back to the world frame, kept so a view can be named
// and reproduced. `logScale` accumulates log10 of the scale, a sum of small numbers with no ceiling.
const anchor = {
  M: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
  t: new Float64Array(3),
  logScale: 0,
  depth: 0,
  word: [],
  // What the letters the anchor has absorbed are worth in hue. A rebase into child `best` adds
  // a(best) here and every local sum below restarts one letter shorter; frac is additive, so
  // frac(anchor.hue + local) is unchanged and the colour field is exactly rebase invariant.
  hue: 0,
};

// suppR is a Mahalanobis support radius, suppE a Euclidean one, both in anchor-local units.
// `targets` holds [x, z] for a field preset and { word } for an IFS, discriminated in setGoal by
// the preset kind.
let maps = null, root = null, targets = [], targetIdx = 0, aimIdx = 0, suppR = 4, suppE = 1;
let mapA = null, mapD = null, mapP = null, mapS = null, mapSc = null, mapCC = null, mapLS = null;
let mapHue = null;
let mDim = 2, conformal = false, terrHMax = 1;

// ========================== the colour law ==============================
// Hue is an increment per map, summed down the address and taken mod one:
//
//     u(i1 i2 ... ik) = frac( a(i1) + a(i2) + ... + a(ik) )
//
// with the increments chosen to have zero measure weighted mean, sum p_i a_i = 0. Everything the
// law is for follows. u is a function of the address alone, so no node's colour ever moves and
// neighbouring pieces, sharing a long prefix, differ by the last increment or two. Zero mean
// makes the children of a piece straddle its hue symmetrically, so a region's measure weighted
// mean hue is preserved exactly by a refinement at every level, while new colour still arrives
// from within at +-hueStep; a law based on size cannot do this, since refining a region
// necessarily recolours it. The sum is a random walk on the hue circle with step sd
// sqrt(sum p_i a_i^2), about 0.6 hueStep, so regions wander independently and there is no whole
// frame drift. The largest step anywhere is one increment, and the cross fade band dissolves even
// that. The increments are the presets' per map `cc`, centred on the measure weighted mean and
// scaled so the largest is exactly hueStep.
function buildMapHue(P) {
  const k = maps.length;
  mapHue = new Float64Array(k);
  const step = P.hueStep || 0;
  if (!step) return;
  let mean = 0;
  for (let i = 0; i < k; i++) mean += mapP[i] * mapCC[i];
  let big = 0;
  for (let i = 0; i < k; i++) {
    mapHue[i] = mapCC[i] - mean;
    const a = Math.abs(mapHue[i]);
    if (a > big) big = a;
  }
  if (!(big > 0)) return;
  const sc = step / big;
  for (let i = 0; i < k; i++) mapHue[i] *= sc;
}
// Hypsometric span, in units of the camera distance so a change of zoom needs no reconvergence,
// and measured from the cut rather than assumed: the terrain's height relative to the view
// depends on the local gap structure of the boundary and varies by an order of magnitude between
// locations. In the seahorse valley a constant tuned at 1e4 left the frame eight times too dark
// at 1e8.
let terrSpanRel = 0;
// The ramp's ends, as log2 of height over camera distance, from the cut's own height histogram
// every rebuild. Equalized between two percentiles of log height, because the height is a
// distance to the set and a distance near a fractal boundary is log distributed: with a single
// span anchored at zero, 16 to 43 percent of the land fell in the darkest tenth of the ramp.
let terrLo = 0, terrHi = 0, terrEq = false;
const PAL = 256;
const palette = new Float32Array(PAL * 3);
const bg0 = new Float32Array(3), bg1 = new Float32Array(3);
let lakeCol = new Float32Array(3);
const sheenCol = new Float32Array(3);      // what the lake reflects at grazing angles
const edgeCol = new Float32Array(3);       // plane: where the ramp cannot be resolved
const palMean = new Float32Array(3);       // the ramp's own mean, in linear light
const palInt = new Float64Array((PAL + 1) * 3);   // its integral, for exact box filtering
const rampOut = new Float64Array(3);
// Bumped by loadPalette. The per pixel path samples the same table through a texture, so it needs
// to know when to re-upload; see palUpload in src/62_direct.js.
let palGen = 0;

// The exact average of the periodic ramp over an interval of width `s` turns centred on `u`,
// which is the correct antialiased colour of a cell whose colour coordinate is not constant
// across it. s -> 0 gives the ramp itself, so a resolved cell is unaffected; s >= 1 gives the
// ramp's mean, which is what a cell spanning whole periods looks like when it is filtered rather
// than point sampled; in between a cell desaturates smoothly with no threshold anywhere. Two
// table lookups and a subtraction, against an iteration count per sample for supersampling.
function rampBox(u, s) {
  if (!(s > 1e-4)) {
    // Interpolated between adjacent entries, not the nearest: the table is a cyclic sampling of a
    // continuous loop, and the nearest entry both quantizes the ramp to 256 steps and disagrees
    // with the GPU path's texture lookup, which reads the same stops.
    let x = (u - Math.floor(u)) * PAL;
    let j = x | 0; const f = x - j;
    if (j >= PAL) j = PAL - 1;
    const j1 = j + 1 === PAL ? 0 : j + 1;
    for (let c = 0; c < 3; c++) {
      rampOut[c] = palette[j * 3 + c] + (palette[j1 * 3 + c] - palette[j * 3 + c]) * f;
    }
    return rampOut;
  }
  if (s >= 1) { rampOut[0] = palMean[0]; rampOut[1] = palMean[1]; rampOut[2] = palMean[2]; return rampOut; }
  let a = u - 0.5 * s, b = u + 0.5 * s;
  a -= Math.floor(a); b = a + s;                  // b may exceed 1, hence the wrap below
  const inv = 1 / s;
  for (let c = 0; c < 3; c++) {
    let v = palIntAt(b, c) - palIntAt(a, c);
    rampOut[c] = v * inv;
  }
  return rampOut;
}
// F extended past one period by adding whole period integrals, one per turn, which is exactly
// palMean. b < 2 always here, so one wrap is enough.
function palIntAt(t, c) {
  if (t >= 1) return palMean[c] + palIntAt(t - 1, c);
  const x = t * PAL, j = x | 0, f = x - j;
  const j0 = j < PAL ? j : PAL - 1;
  return palInt[j0 * 3 + c] + (palInt[(j0 + 1) * 3 + c] - palInt[j0 * 3 + c]) * f;
}
let dirty = true, kern = kernelConst(3.4);
// Milliseconds of refinement per frame. Shared, because startJob needs it to predict how long
// the cut it is about to build will take to arrive.
let WORK_MS = 5.0;      // reset every frame from the resource tier; see perfNow
// Signed d(log dist)/dt, and how many seconds a cut takes to reach the screen: together they say
// where the camera will be when the cut being started now lands. See startJob.
let zoomRate = 0, buildLead = 0;
// Progressive refinement while the camera holds still; see the latency ceiling in planeRoot.
// `stillCuts` is the level, 0 while moving, `stillT` the seconds of stillness it comes from, and
// both reset the moment anything moves. Level k is reached after STILL_T0 * 2^(k-1) seconds and
// the ceiling doubles per level, so work done is about proportional to time spent. Eight levels
// take 25 seconds, and the cut converges several levels earlier in every case measured.
let stillCuts = 0, stillT = 0;
const STILL_T0 = 0.2, STILL_MAX = 8;
// Evaluation only: pin the level so a capture is reproducible, -1 is off, set by `st=` in the
// query. It pins rather than seeds because the ramp reads a wall clock a headless page barely
// advances; seeding it made every `st=8` capture a capture of level zero.
let stillPin = -1;
const stats = {
  splats: 0, buildMs: 0, dmin: 0, dmax: 0, visited: 0, rebuilds: 0, hz: 0,
  errPx: 0, format: '?', gpu: '?', logZoom: 0, culled: 0, evals: 0, iters: 0,
  rebases: 0, maxIter: 0, fieldHit: 0, itersReal: 0,
  gfQueries: 0, gfBatches: 0, gfMs: 0,
};
// Iterations per cell, smoothed, from the last terrain rebuild. The field cache is gated on it
// because a cache only pays when a cell is expensive: at the opening view a cell costs 13
// iterations, about 95 ns, less than a hash, a key compare and a writeback, and the cache turned
// a 43 ms rebuild into 85 ms. At 1e7 a cell costs 1541 iterations and it turns 961 ms into 344 ms.
let terrItEma = 0;
const FC_MIN_ITERS = 80;
// The same gate for the plane's field cache, at a lower threshold because a plane cell's hash key
// carries three doubles rather than two, so a lookup is dearer while the iteration it avoids is
// the same. Measured 17 iterations a cell at the opening view of the Mandelbrot, against a cache
// wrapper that was 30 percent of the whole cut.
let planeItEma = 0;
const PFC_MIN_ITERS = 60;

// Where a plane descent aims. A Julia target is built from the live c, the only way to name a
// point on the set for every c; the Mandelbrot's are named coordinates projected onto the
// boundary, since eighteen digits is only approximately on a fractal boundary and the descent
// goes far past eighteen digits of framing. Both are recomputed when c or the exponent moves.
function planeTargets(P) {
  targets = [];
  const it = Math.max(4000, Math.round((P.view && P.view.iters) || 500));
  const pw = Math.max(2, Math.min(8, Math.round(cfg.power)));
  if (P.julia) {
    for (let s = 0; s < 4; s++) targets.push(juliaPoint(cfg.cx, cfg.cy, pw, s * 7 + 1));
  } else if (pw === 2) {
    for (const t of P.targets) {
      targets.push(projectToPlane(t[0], t[1], pw, cfg.cx, cfg.cy, false, it));
    }
  } else {
    // A named coordinate belongs to one exponent: the quadratic set's landmarks are deep inside
    // the cubic set, so at p = 3 the descent aimed into solid interior and the frame came back one
    // flat colour. mandelBoundaryRay bisects along a ray to find a boundary point of the set drawn.
    for (const a of [0.62, 1.90, 2.74, 4.10]) {
      targets.push(mandelBoundaryRay(a, pw, it));
    }
  }
  if (!targets.length) targets.push([P.center[0], P.center[1]]);
}

function isTerrain() { return PRESETS[cfg.preset].kind === 'terrain'; }
function isPlane() { return PRESETS[cfg.preset].kind === 'plane'; }
function isField() { const k = PRESETS[cfg.preset].kind; return k === 'terrain' || k === 'plane'; }

/* ----------------------------- colour ramp ------------------------------ */
// Mixing a scalar coordinate down the branch and looking it up in a ramp keeps blended splats on
// the ramp, so overlapping branches stay saturated; mixing RGB averages complementary hues toward
// grey, which is why fractal flames use a one dimensional palette too. Stops are authored as they
// look on screen and linearized here, since accumulation is in linear light and the tone map
// encodes gamma at the end. A preset gives either a list of stops or a `ramp` spec; see rampFill.
function loadPalette(P) {
  if (P.ramp) rampFill(P.ramp);
  else {
    const stops = P.palette;
    for (let i = 0; i < PAL; i++) {
      const t = i / (PAL - 1) * (stops.length - 1);
      const j = Math.min(stops.length - 2, Math.floor(t)), f = t - j;
      for (let c = 0; c < 3; c++) {
        palette[i * 3 + c] = Math.pow(stops[j][c] * (1 - f) + stops[j + 1][c] * f, 2.2);
      }
    }
  }
  for (let c = 0; c < 3; c++) {
    let m = 0;
    for (let i = 0; i < PAL; i++) m += palette[i * 3 + c];
    palMean[c] = m / PAL;
  }
  // The ramp's integral, F(v) = integral of pal from 0 to v for v in [0,1], so the exact average
  // over any interval is two lookups and a subtraction. Trapezoid, because `rampBox` reads the
  // table as a piecewise linear loop and the integral has to be of the thing being looked up. The
  // last panel wraps to entry zero, which closes the loop.
  for (let c = 0; c < 3; c++) {
    palInt[c] = 0;
    let a = 0;
    for (let i = 0; i < PAL; i++) {
      const j1 = i + 1 === PAL ? 0 : i + 1;
      a += 0.5 * (palette[i * 3 + c] + palette[j1 * 3 + c]) / PAL;
      palInt[(i + 1) * 3 + c] = a;
    }
  }
  for (let c = 0; c < 3; c++) {
    bg0[c] = Math.pow(P.bg[0][c], 2.2);
    bg1[c] = Math.pow(P.bg[1][c], 2.2);
    lakeCol[c] = Math.pow((P.lake || [0, 0, 0])[c], 2.2);
    sheenCol[c] = Math.pow((P.sheen || P.lake || [0, 0, 0])[c], 2.2);
    edgeCol[c] = Math.pow((P.edge || [0, 0, 0])[c], 2.2);
  }
  palGen++;
}

// A closed colour loop from a formula, in Oklch, filled straight into the 256 entry table. Hue
// advances linearly through the whole circle once, so the loop closes by construction and no stop
// is a decision. Lightness runs `cyc` full cycles per turn of hue, and cyc is 2: measured at 1e12
// the escape count across a whole frame spans as little as 0.2 turns of the ramp, so at one cycle
// per turn which of the single bright and dark bands a deep frame lands on is luck, while at two
// a fifth of the loop always contains a lightness extremum. It also doubles the band count
// without doubling the rate of hue change. Chroma is the largest the display can show at that
// lightness and hue, tapered asymmetrically near the extremes since the gamut runs out faster at
// the top; see oklchToRgb for why the chroma and never the channels gives way. L stays in
// [0.42, 0.84], so no frame lands entirely on a near black stretch and the darkest thing in the
// picture is the set itself.
function rampFill(o) {
  const cyc = o.cyc === undefined ? 2 : o.cyc;
  const dir = o.dir === undefined ? 1 : o.dir;
  for (let i = 0; i < PAL; i++) {
    const u = i / PAL;                       // PAL and not PAL-1: entry PAL would repeat entry 0
    const t = Math.sin(2 * Math.PI * (cyc * u + o.phase));
    const L = o.L0 + o.amp * t;
    const taper = t > 0 ? 1 - o.taperHi * Math.pow(t, 1.4)
                        : 1 - o.taperLo * Math.pow(-t, 1.4);
    const c = oklchToRgb(L, o.C * taper, o.h0 + 360 * u * dir);
    palette[i * 3] = c[0]; palette[i * 3 + 1] = c[1]; palette[i * 3 + 2] = c[2];
  }
}

/* ---------------------------- camera helpers ---------------------------- */
function camFromAngles(yaw, pitch) {
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const f = [sy * cp, sp, cy * cp];
  const u = [-sy * sp, cp, -cy * sp];
  const r = [f[1] * u[2] - f[2] * u[1], f[2] * u[0] - f[0] * u[2], f[0] * u[1] - f[1] * u[0]];
  const R = cam.R;
  R[0] = r[0]; R[1] = r[1]; R[2] = r[2];
  R[3] = u[0]; R[4] = u[1]; R[5] = u[2];
  R[6] = -f[0]; R[7] = -f[1]; R[8] = -f[2];
}

// Re-orthonormalize the eye basis. Free orbit composes small rotations forever and a rebase
// multiplies by the linear part of a map, so drift has to be swept up somewhere. With `level` set
// the up axis is pulled back toward world up, which a landscape wants and a fractal does not.
function orthoBasis() {
  const R = cam.R;
  let bx = R[6], by = R[7], bz = R[8];
  let n = Math.hypot(bx, by, bz) || 1;
  bx /= n; by /= n; bz /= n;
  let ux = R[3], uy = R[4], uz = R[5];
  if (cam.level) {
    const d = by;                       // world up projected out of the back axis
    ux = -bx * d; uy = 1 - by * d; uz = -bz * d;
    if (Math.hypot(ux, uy, uz) < 1e-6) { ux = R[3]; uy = R[4]; uz = R[5]; }
  }
  const d2 = ux * bx + uy * by + uz * bz;
  ux -= bx * d2; uy -= by * d2; uz -= bz * d2;
  n = Math.hypot(ux, uy, uz) || 1;
  ux /= n; uy /= n; uz /= n;
  R[3] = ux; R[4] = uy; R[5] = uz;
  R[6] = bx; R[7] = by; R[8] = bz;
  R[0] = uy * bz - uz * by;             // right = up x back
  R[1] = uz * bx - ux * bz;
  R[2] = ux * by - uy * bx;
}

// Rotate the view. `ax` turns about the eye's own up axis and `ay` about its right axis, so
// R <- Q R with Q built in eye coordinates. With `level` set the turn goes about world up
// instead, which keeps a horizon.
function turnCamera(ax, ay) {
  const R = cam.R;
  if (cam.level) {
    const c = Math.cos(ax), s = Math.sin(ax);     // about world up: R <- R Ry
    for (let r = 0; r < 3; r++) {
      const x = R[r * 3], z = R[r * 3 + 2];
      R[r * 3] = x * c - z * s;
      R[r * 3 + 2] = x * s + z * c;
    }
  } else {
    const c = Math.cos(ax), s = Math.sin(ax);     // about eye up: rows 0 and 2
    for (let k = 0; k < 3; k++) {
      const a = R[k], b = R[6 + k];
      R[k] = a * c - b * s;
      R[6 + k] = a * s + b * c;
    }
  }
  const c2 = Math.cos(ay), s2 = Math.sin(ay);     // about eye right: rows 1 and 2
  for (let k = 0; k < 3; k++) {
    const a = R[3 + k], b = R[6 + k];
    R[3 + k] = a * c2 - b * s2;
    R[6 + k] = a * s2 + b * c2;
  }
  orthoBasis();
}

/* =========================== camera basis =============================== */
// One place only, so the cull math and the render math cannot drift apart.
const basis = {
  R: new Float64Array(9),
  Rgl: new Float32Array(9),    // same matrix, column major, for GLSL
  fwd: new Float64Array(3),
  right: new Float64Array(3),
  up: new Float64Array(3),
  pos: new Float64Array(3),
};

function updateBasis() {
  const R = cam.R, B = basis.R;
  for (let i = 0; i < 9; i++) B[i] = R[i];
  const f = basis.fwd, r = basis.right, u = basis.up;
  r[0] = R[0]; r[1] = R[1]; r[2] = R[2];
  u[0] = R[3]; u[1] = R[4]; u[2] = R[5];
  f[0] = -R[6]; f[1] = -R[7]; f[2] = -R[8];
  const G = basis.Rgl;                                   // column major transpose
  G[0] = B[0]; G[1] = B[3]; G[2] = B[6];
  G[3] = B[1]; G[4] = B[4]; G[5] = B[7];
  G[6] = B[2]; G[7] = B[5]; G[8] = B[8];
  basis.pos[0] = cam.target[0] - f[0] * cam.dist;
  basis.pos[1] = cam.target[1] - f[1] * cam.dist;
  basis.pos[2] = cam.target[2] - f[2] * cam.dist;
}

/* ============================ loading =================================== */
function loadPreset(id, keepView) {
  cfg.preset = id;
  const P = PRESETS[id];
  anchor.M.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  anchor.t.fill(0);
  anchor.logScale = 0; anchor.depth = 0; anchor.word = []; anchor.hue = 0;
  stats.rebases = 0;
  cam.level = P.kind !== 'ifs';        // a field is looked at, not orbited
  // Screen space size cap, per object rather than global: it measures how well the budget can
  // resolve this object, and fades out the tail of oversized splats a budget limited cut leaves.
  SIZE_CAP = (P.view && P.view.sizeCap) || 5.0;
  cam.fov = (P.fov || 45) * Math.PI / 180;
  loadPalette(P);
  targets = [];

  if (P.kind === 'plane') {
    // A flat field. No maps, no rebasing, no exposure law: the camera looks straight down at a
    // plane and the only thing that changes with distance is the scale.
    maps = null; root = null; conformal = false; mDim = 2;
    suppR = 2.6;
    // The plane's own half width. The opening view frames the square of side 2*radius.
    suppE = P.radius;
    planeTargets(P);
    cam.goalWord = []; cam.goalPhase = 0;
  } else if (P.kind === 'terrain') {
    maps = null; root = null; conformal = false; mDim = 2;
    suppR = 2.6;
    const rel = (P.view && P.view.relief) || 0.42;
    const it = Math.round((P.view && P.view.iters) || 420);
    // Measure how tall the terrain gets rather than guessing. A 28x28 grid at 240 iterations is
    // enough: the height is capped and smooth away from the boundary.
    const hc = P.radius * 0.9;
    let hmax = 0;
    for (let i = 0; i < 28; i++) for (let j = 0; j < 28; j++) {
      const cx = P.center[0] + P.radius * (2 * (i + 0.5) / 28 - 1);
      const cz = P.center[1] + P.radius * (2 * (j + 0.5) / 28 - 1);
      mandelField(cx, cz, 240, 0, rel, 0, hc);
      if (fld.h > hmax) hmax = fld.h;
    }
    terrHMax = hmax;
    terrSpanRel = 0;      // remeasured from the first cut
    terrEq = false;
    suppE = Math.hypot(P.radius, P.radius, hmax * 0.5);
    for (const t of P.targets) targets.push(projectToBoundary(t[0], t[1], Math.max(it, 3000)));
    cam.goalWord = []; cam.goalPhase = 0;
  } else {
    maps = P.maps;
    root = rootMoments(maps);
    const sup = supportRadius(maps, root);
    suppR = sup.maha; suppE = sup.euclid;
    mDim = measureDimension(maps);
    conformal = maps.every(m => m.conformal && m.inv);
    // Flatten the maps. The refinement kernel touches these millions of times
    // per second and object property loads show up in the frame time.
    const k = maps.length;
    mapA = new Float64Array(k * 9);
    mapD = new Float64Array(k * 3);
    mapP = new Float64Array(k);
    mapS = new Float64Array(k);
    mapLS = new Float64Array(k);
    mapSc = new Uint8Array(k);
    mapCC = new Float64Array(k);
    for (let i = 0; i < k; i++) {
      const m = maps[i];
      mapA.set(m.A, i * 9);
      // mu_{u.i} = mu_u + M_u d_i, with d_i = w_i(mu_root) - mu_root.
      const q = mat3Vec(m.A, root.mu);
      mapD[i * 3] = q[0] + m.b[0] - root.mu[0];
      mapD[i * 3 + 1] = q[1] + m.b[1] - root.mu[1];
      mapD[i * 3 + 2] = q[2] + m.b[2] - root.mu[2];
      mapP[i] = m.p;
      mapS[i] = m.s;
      // log10 of the map's uniform scale factor, |det A|^(1/3) rather than A[0], because that is
      // the contraction of a general conformal map and A[0] is its scale only when the map has
      // no rotation. This is what the colour coordinate integrates.
      mapLS[i] = Math.log10(m.scale);
      mapSc[i] = m.scalar ? 1 : 0;
      mapCC[i] = m.cc === undefined ? (i + 0.5) / k : m.cc;
    }
    // Needs mapP and mapCC, so it happens here rather than with the palette.
    buildMapHue(P);
    // Descent targets. The fixed point of a map, or of a short word of maps, is exactly on the
    // attractor, so the camera can fall toward it forever; a chaos game point is only nearly on it.
    for (let i = 0; i < maps.length; i++) {
      if (fixWord(maps, [i])) targets.push({ word: [i] });
    }
    for (let i = 0; i + 1 < maps.length; i++) {
      if (fixWord(maps, [i, i + 1])) targets.push({ word: [i, i + 1] });
    }
  }
  // Which target a descent starts on. The fixed point of a single map is a vertex or corner of
  // most of these attractors, and zooming into a corner frames badly: the piece hangs off one side
  // and most of the frame stays empty. A two letter word sits between two pieces, so the structure
  // surrounds the aim. Named per object, since the folded dragon's single map fixed point is
  // inside its sheet.
  aimIdx = 0;
  if (P.aim !== undefined) {
    for (let i = 0; i < targets.length; i++) {
      if (targets[i].word && targets[i].word.join('.') === P.aim) { aimIdx = i; break; }
    }
  }

  if (!keepView) {
    // Only the escape time fields open still. Their cut costs hundreds of milliseconds and goes on
    // sharpening for seconds while the camera holds, so an opening frame in flight is the coarse
    // one. On the kind and not on is2D, which is true of a flat IFS too, and set before the view
    // so a preset can still say otherwise.
    cfg.autopilot = P.kind !== 'plane';
    const v = P.view;
    if (v) for (const k in v) if (k in cfg) cfg[k] = v[k];
    const terr = P.kind === 'terrain';
    const yaw = (v && v.yaw !== undefined) ? v.yaw : (terr ? 0.22 : 0.6);
    // The terrain wants a steep view: a distance field terrain is locally a smooth cone flank, so
    // a low raking angle shows one featureless slope, while its fractal character is in the plan
    // of the ridge network, which only reads from above.
    const pitch = (v && v.pitch !== undefined) ? v.pitch : (terr ? -0.98 : -0.24);
    if (!v || v.budget === undefined) cfg.budget = 250000;
    kern = kernelConst(cfg.kernel);
    // Frame the measured extent, not a guess in sigmas: a sphere of radius suppE exactly fills the
    // vertical field, and 0.98 leaves a hair of margin. The field of view is vertical, so a
    // portrait window sees less of the object across than down and cropped the Pythagoras tree in
    // half; frameAspect() pulls back by the ratio. On the plane presets the field is 15 degrees,
    // where the distance varies by 0.9 percent from frame centre to corner, so it is orthographic.
    cam.dist = 0.98 * suppE / Math.sin(0.5 * cam.fov) * frameAspect();
    cam.startDist = cam.dist;
    if (P.kind === 'terrain') {
      cam.target[0] = P.center[0];
      cam.target[1] = terrHMax * 0.22;
      cam.target[2] = P.center[1];
      camFromAngles(yaw, pitch);
    } else if (P.kind === 'plane') {
      cam.target.set([P.center[0], 0, P.center[1]]);
      // Straight at the plane, from -y, and a hair off the pole: 1.5703 is pi/2 - 5e-4, which
      // keeps orthoBasis's level branch above its 1e-6 degeneracy floor. The plane's coordinates
      // are the real and imaginary parts of a complex number, drawn real to the right and
      // imaginary up. With a right handed basis, forward = -y and up = +z force right = -x, so
      // looking down at the card draws the set rotated by half a turn against a per pixel
      // reference. From -y, right = +x and up = +z, and a Gaussian looks the same from either side.
      camFromAngles(Math.PI, 1.5703);
    } else {
      cam.target.set(root.mu);
      camFromAngles(yaw, pitch);
    }
    setGoal(aimIdx);
  }
  orthoBasis();
  dirty = true;
  syncUI();
}

function setGoal(i) {
  if (!targets.length) { cam.goal.set(cam.target); return; }
  targetIdx = ((i % targets.length) + targets.length) % targets.length;
  const t = targets[targetIdx];
  if (isPlane()) {
    cam.goal[0] = t[0]; cam.goal[1] = 0; cam.goal[2] = t[1];
    cam.goalWord = []; cam.goalPhase = 0;
  } else if (isTerrain()) {
    cam.goal[0] = t[0]; cam.goal[2] = t[1];
    cam.goal[1] = terrainAimHeight();
    cam.goalWord = []; cam.goalPhase = 0;
  } else {
    cam.goalWord = t.word.slice(); cam.goalPhase = 0;
    refreshGoal();
  }
}

// How high above the boundary point to aim, as a fraction of the camera distance so the clearance
// is the same at every depth. The relief is proportional to the camera distance once the camera is
// down among it, and the tallest ridge inside the depth of field reaches about 2.1 * relief camera
// distances, so 1.1 * relief * dist puts the eye at about 1.7 and clears it. Bounded by the
// object's own height so the opening view still aims at the object.
function terrainAimHeight() {
  return Math.min(terrHMax * 0.22, 1.1 * cfg.relief * cam.dist);
}

// Recompute the aim point from its word at the current phase. Exact at any depth.
function refreshGoal() {
  if (!maps || !cam.goalWord.length) return;
  const x = fixWord(maps, rotateWord(cam.goalWord, cam.goalPhase));
  if (x && isFinite(x[0] + x[1] + x[2])) cam.goal.set(x);
}

function pickTarget(i, snapOut) {
  if (!targets.length) return;
  if (snapOut) {
    loadPreset(cfg.preset, false);
    setGoal(i);
  } else setGoal(i);
  dirty = true;
}

/* ============================== rebasing ================================ */
// Why there is no zoom limit for a self similar object.
//
// A node's piece is an exact affine copy of the whole attractor, P_u = F_u(A), so the scene can be
// re-expressed in that node's own coordinates at any moment by applying F_u^-1 to the object and
// to the camera together: the object comes back identical, the camera at a distance of order one,
// and the tree walk starts from the root again. The descent is then a loop rather than a ramp, one
// rebase per factor of 1/s of zoom, with every coordinate permanently of order one. The zoom lives
// in `anchor.logScale`, a sum of logarithms, so there is no precision wall because there is no
// large number anywhere.
//
// Two conditions have to hold. Every map must be conformal, A = s R, or it would shear the camera
// as well as the object. And the walk root has to stay above the camera's own scale, since content
// outside it is dropped: REBASE_MARGIN keeps the walk root's piece four times larger than the
// depth of field the shader already fades to nothing, so a rebase discards exactly what the depth
// of field cull was discarding anyway, which is why it does not pop.
const REBASE_MARGIN = 4;

function rebaseNeeded() {
  if (!cfg.rebase || !conformal || isTerrain()) return false;
  const zFar = 1 + 8 / Math.max(cfg.fog, 0.02);
  return cam.dist * zFar * REBASE_MARGIN < suppE;
}

function rebaseStep() {
  // Which child piece holds the aim point. A worded aim says so exactly: its address is that word
  // repeated, so the first letter names the child. The geometric fallback is for a wordless aim.
  if (cam.goalWord.length) {
    return rebaseInto(cam.goalWord[cam.goalPhase % cam.goalWord.length]);
  }
  const inv6 = sym3Inv(root.cov);
  let best = -1, bestQ = Infinity;
  const x = new Float64Array(3);
  for (let i = 0; i < maps.length; i++) {
    const m = maps[i];
    if (!m.inv) continue;
    const g0 = cam.goal[0] - m.b[0], g1 = cam.goal[1] - m.b[1], g2 = cam.goal[2] - m.b[2];
    mat3Vec(m.inv, [g0, g1, g2], x);
    const dx = x[0] - root.mu[0], dy = x[1] - root.mu[1], dz = x[2] - root.mu[2];
    const q = inv6
      ? dx * (inv6[0] * dx + inv6[1] * dy + inv6[2] * dz) +
        dy * (inv6[1] * dx + inv6[3] * dy + inv6[4] * dz) +
        dz * (inv6[2] * dx + inv6[4] * dy + inv6[5] * dz)
      : dx * dx + dy * dy + dz * dz;
    if (q < bestQ) { bestQ = q; best = i; }
  }
  if (best < 0 || bestQ > suppR * suppR * 1.5) return false;
  return rebaseInto(best);
}

// Transport one cut's frame record into the new coordinates. `pos`, `R` and `dist` describe where
// the camera was when the cut was built, expressed in the current frame, so they move exactly as
// the camera just did. `Rq` and `sSince` accumulate the residual rotation and scale that the cut's
// own splat coordinates still carry and that `reprojection` has to undo.
function rebaseRecord(rec, m, s, iv) {
  const p = new Float64Array(3);
  mat3Vec(iv, [rec.pos[0] - m.b[0], rec.pos[1] - m.b[1], rec.pos[2] - m.b[2]], p);
  rec.pos.set(p);
  rec.dist /= s;
  const Rb = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    let acc = 0;
    for (let k = 0; k < 3; k++) acc += rec.R[r * 3 + k] * m.A[k * 3 + c];
    Rb[r * 3 + c] = acc / s;
  }
  rec.R.set(Rb);
  const As = new Float64Array(9);
  for (let i = 0; i < 9; i++) As[i] = m.A[i] / s;        // the map's rotation
  rec.Rq.set(mat3Mul(rec.Rq, As));
  rec.sSince *= s;
}

// Re-express the whole scene in child `best`'s own coordinates.
function rebaseInto(best) {
  const m = maps[best], s = m.scale, iv = m.inv;
  if (!iv) return false;
  // The zoom and the colour accumulators are sums over the letters the anchor has swallowed, each
  // exactly compensated by every local value below restarting one letter shorter.

  // The aim point is recomputed from its word, never transported: the inverse map expands, so
  // transporting multiplies its error by 1/s every step. The camera target is transported, which
  // is safe because the aim invariant holds it within 0.4 camera distances of the aim and the map
  // scales both sides by 1/s, so the invariant survives exactly.
  const t = new Float64Array(3);
  mat3Vec(iv, [cam.target[0] - m.b[0], cam.target[1] - m.b[1], cam.target[2] - m.b[2]], t);
  cam.target.set(t);
  if (cam.goalWord.length) {
    cam.goalPhase = (cam.goalPhase + 1) % cam.goalWord.length;
    refreshGoal();
  } else {
    const g = new Float64Array(3);
    mat3Vec(iv, [cam.goal[0] - m.b[0], cam.goal[1] - m.b[1], cam.goal[2] - m.b[2]], g);
    cam.goal.set(g);
  }
  cam.dist /= s;

  // The camera has to absorb the same map. A world vector v becomes
  // v' = A^-1 v, so the eye basis becomes R' = R A / s for R' v' = R v.
  const Rn = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    let acc = 0;
    for (let k = 0; k < 3; k++) acc += cam.R[r * 3 + k] * m.A[k * 3 + c];
    Rn[r * 3 + c] = acc / s;
  }
  cam.R.set(Rn);
  orthoBasis();

  // Both cuts come along, for the same reason: each was built in the frame just superseded and
  // stays valid there, since the maps and the walk root are identical in every anchor frame.
  //
  //  - `built`, the cut on screen, keeps being drawn for the few frames the next rebuild takes.
  //    Untransported it drew at scale s with the new frame's exposure: a measured 63 percent whole
  //    frame flash on the folded dragon, once every 0.35 decades of descent.
  //  - the record the job in flight hands over. Its node data stays in the old frame by design,
  //    but labelling the result current-frame in `finishJob` moves the flash one rebuild later:
  //    measured 53 percent over eight frames, with k wrong by exactly s.
  if (built.valid) rebaseRecord(built, m, s, iv);
  if (job.active) rebaseRecord(job.out, m, s, iv);

  // Bookkeeping back to the original frame, for naming a view. anchor <- anchor o w.
  const t2 = new Float64Array(3);
  mat3Vec(anchor.M, m.b, t2);
  anchor.t[0] += t2[0]; anchor.t[1] += t2[1]; anchor.t[2] += t2[2];
  anchor.M.set(mat3Mul(anchor.M, m.A));
  anchor.logScale += Math.log10(s);
  anchor.hue += mapHue[best];
  anchor.depth++;
  anchor.word.push(best);
  stats.rebases++;
  return true;
}

// At most 64 contractions in one frame, which bounds the work one frame can be asked to do. More
// than that is a teleport rather than a descent.
function rebaseAll() {
  let n = 0;
  while (n < 64 && rebaseNeeded() && rebaseStep()) n++;
  if (n) dirty = true;
  return n;
}

// Total zoom, as log10. anchor.logScale is negative and unbounded below, so the
// readout keeps climbing long after a float would have run out of exponent.
function logZoom() {
  return Math.log10(cam.startDist / Math.max(cam.dist, 1e-300)) - anchor.logScale;
}

// Measure in a ball of radius r about a point of the attractor scales as r^D with D the
// information dimension, so the light landing in the frame scales as dist^D and the gain is its
// inverse. In closed form it depends on the camera distance and nothing else, so rotation cannot
// change the exposure; normalizing against the weight actually in view does, and the image
// visibly breathed. Passed a distance rather than reading the camera, so a cut is exposed at draw
// time instead of carrying a two frame old exposure baked into its weights.
//
// The crossover is suppE, the object's own radius, and not the framing distance: above it the
// whole attractor is in frame and closing in only magnifies, so the gain stays flat. Measuring
// from startDist brightened the first e-fold of a descent by (startDist/suppE)^D, a factor of 6.5
// on these objects, which the saturating opacity mostly hid.
function measureNormAt(dist) {
  if (isField()) return 1;
  const z = Math.max(1, suppE / Math.max(dist, 1e-300));
  return Math.pow(z, mDim);
}
function measureNorm() { return measureNormAt(cam.dist); }

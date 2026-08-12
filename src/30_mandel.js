/* ========================== the Mandelbrot field ========================= */
// z -> z^2 + c from z = 0 with dz/dc carried alongside, a cache of that field, and
// the preset tables for the plane and terrain objects. One evaluation gives
//
//   escape count    nu = n - log2(log|z| / log B)
//   distance        de = 2 |z| log|z| / |dz|              Douady and Hubbard
//   surface normal  from d(log|z|)/dc = conj(dz/z)
//
// so a cell costs one evaluation and not the three a finite difference needs, and
// that evaluation is the whole cost of the terrain. Height is de itself: a distance
// scales like the horizontal coordinate, so the relief is invariant under zoom.

const LOGB = Math.log(1e6);
const BAIL2 = 1e12;                             // |z|^2 bailout, B = 1e6
const LN2 = Math.LN2;

const fld = {
  h: 0, gx: 0, gz: 0,          // height and its gradient in the c plane
  nx: 0, ny: 1, nz: 0,         // world normal, world = (Re c, h, Im c)
  de: 0, nu: 0, inside: 0, iters: 0, edge: 0,
};

// Analytic interior tests for the main cardioid and the period two bulb, most of the
// interior area of the opening view; without them every cell inside runs the full cap.
function inMainBody(x, y) {
  const q = (x - 0.25) * (x - 0.25) + y * y;
  if (q * (q + (x - 0.25)) < 0.25 * y * y) return true;
  return (x + 1) * (x + 1) + y * y < 0.0625;
}

function flatCell() {
  fld.h = 0; fld.gx = 0; fld.gz = 0;
  fld.nx = 0; fld.ny = 1; fld.nz = 0;
  fld.de = 0; fld.nu = 0; fld.inside = 1; fld.edge = 0;
}

// rTol is the cell's tolerance, half its half size (50_cut.js passes r * 0.5). Two
// early exits use it. de = 2|z|log|z|/|dz| and |dz| only grows once |z| > 1, so |dz|
// past 2 B log B / rTol bounds de under rTol while the orbit stays outside |z| = 0.5,
// and on a cell straddling the boundary that fires near iteration 100 instead of 3000.
// Periodicity against a reference refreshed on a doubling schedule catches an interior
// point in one pass.
// hc soft caps the height: de overshoots the true distance by about three at |c| = 2,
// so an uncapped exterior is a wall several object radii tall. de*hc/(de+hc) leaves the
// near boundary alone, and hc is a length in the c plane, so the terrain does not move
// when the camera does.
function mandelField(cx, cz, maxIter, rTol, slope, mode, hc) {
  if (inMainBody(cx, cz)) { flatCell(); fld.iters = 0; return; }
  const dbail = rTol > 0 ? 2 * 1e6 * LOGB / rTol : 1e300;
  const dbail2 = dbail < 1e150 ? dbail * dbail : 1e300;
  let zx = 0, zz = 0, dx = 0, dz = 0;
  let rx = 0, rz = 0, per = 8, chk = 8;
  let n = 0, esc = false, m = 0, cyc = false;
  for (; n < maxIter; n++) {
    const ndx = 2 * (zx * dx - zz * dz) + 1;         // dz' = 2 z dz + 1
    const ndz = 2 * (zx * dz + zz * dx);
    dx = ndx; dz = ndz;
    const nzx = zx * zx - zz * zz + cx;              // z' = z^2 + c
    zz = 2 * zx * zz + cz; zx = nzx;
    m = zx * zx + zz * zz;
    if (m > BAIL2) { n++; esc = true; break; }
    // Three of the twelve multiplies in this loop, and |dz| grows by orders of
    // magnitude per iteration once it grows at all, so it runs one iteration in eight.
    if ((n & 7) === 7 && dx * dx + dz * dz > dbail2) {   // proves de < rTol
      fld.iters = n;
      fld.inside = 0; fld.edge = 1;
      fld.de = rTol; fld.nu = 0;
      fld.h = mode === 0 ? slope * rTol : 0;   // hc never binds this close in
      fld.gx = 0; fld.gz = 0;
      fld.nx = 0; fld.ny = 1; fld.nz = 0;
      return;
    }
    if (n > 40) {
      if (Math.abs(zx - rx) + Math.abs(zz - rz) < 1e-16) { cyc = true; break; }   // on a cycle
      if (--chk <= 0) { rx = zx; rz = zz; per += per; chk = per; }
    }
  }
  fld.iters = n;
  // Interior is claimed only on a proof, a cycle or the body test above; a starved cap
  // is not one. Measured in seahorse valley, from 1e8 down every cell called interior
  // that way was really exterior, 23 percent of the frame at 1e11, each one a hole in
  // the background colour. A starved cell gets the bailout's verdict: unresolved.
  if (!esc) {
    if (cyc || n < maxIter) { flatCell(); return; }
    fld.inside = 0; fld.edge = 1;
    fld.de = rTol; fld.nu = 0;
    fld.h = mode === 0 ? slope * rTol : 0;
    fld.gx = 0; fld.gz = 0;
    fld.nx = 0; fld.ny = 1; fld.nz = 0;
    return;
  }

  // |dz| can overflow to Infinity before the point escapes; the unit gradient is then
  // Inf/Inf = NaN, and a NaN colour in the additive buffer poisons every pixel the
  // splat touches. Treat an overflowed derivative as unresolved.
  const dm = Math.sqrt(dx * dx + dz * dz);
  if (!(dm < 1e300) || !(m < 1e300)) {
    fld.inside = 0; fld.edge = 1;
    fld.de = rTol; fld.nu = 0;
    fld.h = mode === 0 ? slope * rTol : 0;
    fld.gx = 0; fld.gz = 0;
    fld.nx = 0; fld.ny = 1; fld.nz = 0;
    return;
  }
  const lr = 0.5 * Math.log(m);                      // log|z|
  const de = 2 * Math.sqrt(m) * lr / (dm > 1e-300 ? dm : 1e-300);
  const nu = n - Math.log(lr / LOGB) * Math.LOG2E;   // Math.log2 is a slow builtin
  // grad log|z| = (Re(dz/z), -Im(dz/z)), pointing away from the set
  const wr = (dx * zx + dz * zz) / m;
  const wi = (dz * zx - dx * zz) / m;
  let ux = wr, uz = -wi;
  const gm = Math.sqrt(ux * ux + uz * uz);           // Math.hypot guards overflow
  if (gm > 1e-300) { ux /= gm; uz /= gm; } else { ux = 0; uz = 0; }

  fld.de = de; fld.nu = nu; fld.inside = 0; fld.edge = 0;
  if (mode === 0) {
    // Distance height. |grad de| = 1 almost everywhere, so the terrain is a cone field
    // over the boundary: uniform slopes creasing along the medial axis of the
    // complement, with the filigree appearing as a network of valleys.
    const k = hc > 0 ? hc / (de + hc) : 1;      // soft cap, and its derivative is k^2
    fld.h = slope * de * k;
    fld.gx = slope * k * k * ux; fld.gz = slope * k * k * uz;
  } else {
    // Potential height. nu rises without bound at the boundary, so this is the terraced
    // citadel and it is not scale invariant, the walls steepening without limit under
    // zoom. The terraces are the true equipotentials of the set.
    const k = slope * 0.11;
    let g = gm / (LN2 * lr);                          // |grad nu|
    if (g > 40) g = 40;
    fld.h = k * nu;
    fld.gx = -k * g * ux; fld.gz = -k * g * uz;       // nu grows inward
  }
  if (!(Math.abs(fld.gx) < 1e150) || !(Math.abs(fld.gz) < 1e150)) { fld.gx = 0; fld.gz = 0; }
  const inv = 1 / Math.sqrt(1 + fld.gx * fld.gx + fld.gz * fld.gz);
  fld.nx = -fld.gx * inv; fld.ny = inv; fld.nz = -fld.gz * inv;
  if (!(fld.h > -1e300 && fld.h < 1e300)) { fld.h = 0; fld.nx = 0; fld.ny = 1; fld.nz = 0; }
}

/* ------------------------- field evaluation cache ------------------------- */
// A cell's field depends only on the cell, and the quadtree always starts from the same
// root, so (cx, cz) names a cell of a fixed grid exactly and the same cells recur across
// rebuilds, of which only 4 percent of camera drift triggers one. Before the cache, at
// 1e7 in seahorse valley: 961 ms per rebuild, 1388 iterations per cell.
// Direct mapped, full key stored and compared, so it can only fail to help, never return
// a wrong answer; fcStamp is the one hashed comparison. The cap is part of the key, since
// a cell starved at one cap escapes at a higher one, quantized to steps of 1.3 so a nine
// decade descent invalidates six times instead of every frame.
const FC_BITS = 20, FC_N = 1 << FC_BITS, FC_MASK = FC_N - 1;
const fcX = new Float64Array(FC_N), fcZ = new Float64Array(FC_N);
const fcH = new Float64Array(FC_N);          // height, kept f64: it is a coordinate
const fcG = new Float32Array(FC_N * 4);      // gx gz de nu, all shading or thresholds
const fcIt = new Float32Array(FC_N);
const fcKind = new Int8Array(FC_N);          // 0 exterior, 1 interior, 2 unresolved
const fcGen = new Int32Array(FC_N);
let fcCur = 1, fcHits = 0, fcMiss = 0;
// Iterations served from the cache, so stats.itersReal stays consistent with the ms
// beside it: at 1e7 the nominal 436 M against a 346 ms rebuild would imply an impossible
// 1.3 G iterations a second.
let fcSaved = 0;
const fcBuf = new Float64Array(2);
const fcInt = new Int32Array(fcBuf.buffer);

function fcHash(cx, cz) {
  fcBuf[0] = cx; fcBuf[1] = cz;
  let h = Math.imul(fcInt[0], 0x9e3779b1) ^ Math.imul(fcInt[1], 0x85ebca6b) ^
          Math.imul(fcInt[2], 0xc2b2ae35) ^ Math.imul(fcInt[3], 0x27d4eb2f);
  h ^= h >>> 15; h = Math.imul(h, 0x2545f491); h ^= h >>> 13;
  return h & FC_MASK;
}

// Everything the output depends on besides the cell itself: quantized cap, relief scale,
// height mode, soft cap length, folded into one non zero stamp and compared per slot, so
// a change invalidates without clearing 45 MB.
function fcStamp(mstep, relief, hmode, hc) {
  let g = Math.imul(mstep + 1, 1000003) ^ (hmode ? 0x5bd1e995 : 0x1b873593);
  fcBuf[0] = relief; g = Math.imul(g ^ fcInt[0], 0x85ebca6b) ^ fcInt[1];
  fcBuf[0] = hc;     g = Math.imul(g ^ fcInt[0], 0xc2b2ae35) ^ fcInt[1];
  return g | 1;
}

// Same contract as mandelField, `fld` included. Only the quadtree uses it;
// projectToBoundary calls mandelField directly because it evaluates at rTol = 0.
// Four ways by linear probing: at 283 000 live cells in 1 048 576 slots the load is only
// 0.27, but one slot per hash measured 73.1 percent hits with the camera frozen and four
// ways gives 97, for no extra memory; doubling the table reaches 87 for another 49 MB.
const FC_WAYS = 4;
function mandelFieldCached(cx, cz, maxIter, rTol, slope, mode, hc) {
  const h0 = fcHash(cx, cz);
  for (let p = 0; p < FC_WAYS; p++) {
    const s = (h0 + p) & FC_MASK;
    if (fcGen[s] === fcCur && fcX[s] === cx && fcZ[s] === cz) {
      const g4 = s * 4;
      const gx = fcG[g4], gz = fcG[g4 + 1];
      const k = fcKind[s];
      fld.h = fcH[s]; fld.gx = gx; fld.gz = gz;
      fld.de = fcG[g4 + 2]; fld.nu = fcG[g4 + 3];
      fld.inside = k === 1 ? 1 : 0; fld.edge = k === 2 ? 1 : 0;
      fld.iters = fcIt[s];
      const inv = 1 / Math.sqrt(1 + gx * gx + gz * gz);
      fld.nx = -gx * inv; fld.ny = inv; fld.nz = -gz * inv;
      fcHits++; fcSaved += fld.iters;
      return;
    }
  }
  mandelField(cx, cz, maxIter, rTol, slope, mode, hc);
  fcMiss++;
  // Prefer a slot this generation has not claimed, otherwise evict the first of the
  // group. The key is written with the value, so a probe only matches this cell.
  let s = h0;
  for (let p = 0; p < FC_WAYS; p++) {
    const t = (h0 + p) & FC_MASK;
    if (fcGen[t] !== fcCur) { s = t; break; }
  }
  const g4 = s * 4;
  fcX[s] = cx; fcZ[s] = cz; fcGen[s] = fcCur;
  fcH[s] = fld.h; fcG[g4] = fld.gx; fcG[g4 + 1] = fld.gz;
  fcG[g4 + 2] = fld.de; fcG[g4 + 3] = fld.nu;
  fcIt[s] = fld.iters;
  fcKind[s] = fld.inside ? 1 : (fld.edge ? 2 : 0);
}

// Push a point onto the boundary. de gives distance and direction, so c <- c - de * u is
// a Newton step whose fixed points are the boundary. Three or four steps land far inside
// the visible pixel, and the projection is repeated as the camera descends, for the
// terrain whenever the camera distance halves.
function projectToBoundary(cx, cz, maxIter) {
  for (let k = 0; k < 6; k++) {
    mandelField(cx, cz, maxIter, 0, 1, 0, 0);      // hc = 0: no cap, raw estimate
    if (fld.inside || fld.de <= 0) break;
    const step = fld.de;
    if (!isFinite(step) || step > 1) break;
    cx -= step * fld.gx;                    // gx,gz is the unit outward normal
    cz -= step * fld.gz;                    // when slope = 1 and mode = 0
  }
  return [cx, cz];
}

/* ----------------------------- preset tables ------------------------------ */
// Presets for the terrain above and for the plane objects implemented in
// src/32_plane.js and src/62_direct.js.
const TERRAIN = {
  mandelbrot: {
    hidden: true,                    // not in the menu on any tier; reached by ?p=mandelbrot
    name: 'Mandelbrot terrain',
    kind: 'terrain',
    blurb: 'Height is the distance to the Mandelbrot set, so the set itself is a flat lake exactly its own ' +
           'shape and the exterior is a cone field rising out of it. Because a distance scales like the ' +
           'coordinate, the relief is invariant under zoom: the landscape keeps its character at every ' +
           'depth, with no reference level and nothing to renormalize. Splats are surfels, oriented by the ' +
           'analytic normal, opaque rather than emissive.',
    center: [-0.7, 0],
    radius: 1.55,
    // Hypsometric: lake, shore, lowland, upland, snow. Authored in display space,
    // linearized at load, because accumulation is linear light.
    palette: [[0.11, 0.25, 0.36], [0.15, 0.47, 0.48], [0.40, 0.56, 0.38], [0.80, 0.66, 0.35], [0.97, 0.94, 0.88]],
    // The interior seen as water: `lake` straight down, `sheen` at a grazing angle,
    // Fresnel between them in emitTerr. A single near black constant made every interior
    // dominated view a black frame.
    lake: [0.055, 0.135, 0.215],
    sheen: [0.48, 0.66, 0.74],
    bg: [[0.020, 0.028, 0.044], [0.055, 0.072, 0.105]],
    view: {
      splitPx: 1.4, budget: 300000, density: 1.0, glow: 0.0, kernel: 3.0, sigma: 1.25, fog: 1.30,
      relief: 0.42, iters: 420, contour: 0.55,
    },
    // Approximate literals, projected onto the boundary at load with the distance
    // estimate, so a point that is merely close still lands exactly on the set.
    targets: [
      [-0.743643887037158704, 0.131825904205311970],   // seahorse valley
      [-0.775683770000000000, 0.136467370000000000],   // Misiurewicz, seahorse
      [-1.401155189092050000, 0.000000000000000000],   // Feigenbaum point
      [-0.101096363845622100, 0.956286510809141500],   // north spiral
      [0.2500000000000000000, 0.000000000000000000],   // the cardioid cusp
      [-1.768610000000000000, 0.001850000000000000],   // west antenna
    ],
  },
};

// The escape time field drawn flat. The smooth escape count is continuous in the point
// and grows without bound towards the boundary, so a periodic ramp cycles forever with no
// seam; see src/32_plane.js. Oklch loop built by rampFill: hue turns once, lightness
// twice, and no stop is near black, because a deep frame can span as little as 0.2 turns
// of the ramp (at 1e12 u landed at 0.07 and the whole image sat on [0.02, 0.04, 0.22]) so
// every fifth of the loop has to carry a lightness extremum. nuCycle is the period in
// units of 3*cbrt(escape count); see planeRoot for the power and pramp.mjs for the cube
// root. The loop stays clear of the Ultra Fractal default, deep blue through white to
// orange and back through dark plum, so the render is read on its own terms.
const PLANE_RAMP = { h0: 196, dir: 1, cyc: 2, phase: 0.06, L0: 0.57, amp: 0.31,
                     C: 0.190, taperHi: 0.35, taperLo: 0.05 };
// The same construction with the hue started 96 degrees round and the lightness a third
// of a cycle out of phase, so the two objects look related and not identical.
const JULIA_RAMP = { h0: 292, dir: 1, cyc: 2, phase: 0.31, L0: 0.56, amp: 0.33,
                     C: 0.185, taperHi: 0.32, taperLo: 0.04 };
// Fallback for anything that asks for stops; the shipped presets take the `ramp` branch.
const PLANE_PALETTE = [
  [0.13, 0.20, 0.52], [0.07, 0.36, 0.76], [0.20, 0.62, 0.92],
  [0.64, 0.88, 0.98], [0.95, 0.91, 0.80], [0.98, 0.70, 0.26],
  [0.80, 0.32, 0.28], [0.42, 0.16, 0.44], [0.13, 0.20, 0.52],
];
const PLANE = {
  mandel2d: {
    name: 'Mandelbrot set',
    kind: 'plane',
    blurb: 'The escape time field of z -> z^p + c, drawn flat: one Gaussian per quadtree cell, ' +
           'coloured by the smooth escape count. Cells split while they span more than a fraction of ' +
           'one turn of the colour ramp AND more than a pixel, so the smooth exterior costs almost ' +
           'nothing and the budget goes to the boundary. Where a cell cannot resolve the ramp at all, ' +
           'it fades to the edge colour instead of showing one arbitrary sample of it, which is what ' +
           'turns the usual rainbow noise along the boundary into a clean edge.',
    center: [-0.6, 0], radius: 1.45,
    palette: PLANE_PALETTE,
    ramp: PLANE_RAMP,
    lake: [0.09, 0.05, 0.16],          // the set itself: dark, but plainly a colour
    edge: [0.02, 0.02, 0.05],          // unresolved: darker than any ramp stop
    bg: [[0.010, 0.011, 0.024], [0.020, 0.022, 0.046]],
    nuCycle: 2.5,
    fov: 15,          // the distance to the plane varies 0.9 percent across the frame
    // splitPx is a cost, not a taste. At 0.55 px a cut is 90 000 to 300 000 cells and
    // 170 to 1000 ms, which against a 5 ms refinement slice is one to three seconds of
    // latency, so the frame on screen is always the stale coarser one. Safe to raise
    // because the ramp is box filtered over the cell.
    // iters is the base cap, before the depth growth and the stationary boost. 1500 and
    // not 500: at 1e6 on a two megapixel window a cap of 500 leaves 63 percent of the
    // frame out of iterations and therefore on one flat colour, 1500 leaves 24 percent,
    // and costs 45 000 cells down to 30 000 while moving. See planeRoot.
    view: { splitPx: 1.4, budget: 1000000, density: 1.0, glow: 0.0, kernel: 3.0, sigma: 1.732,
            fog: 0.10, iters: 1500 },
    targets: [
      [-0.743643887037158704, 0.131825904205311970],   // seahorse valley
      [-0.775683770000000000, 0.136467370000000000],   // Misiurewicz
      [ 0.360240443437614363, 0.641313061064803410],   // the classic spiral
      [-1.768610000000000000, 0.001850000000000000],   // west antenna
      [-0.101096363845622100, 0.956286510809141500],   // north spiral
    ],
  },
  julia2d: {
    name: 'Julia set',
    kind: 'plane',
    julia: true,
    blurb: 'The same field with the roles swapped: the point is the starting value and c is held ' +
           'fixed, so every c gives a different set. c is live, and the whole family is one drag ' +
           'apart. Same adaptive quadtree, same escape count colouring, same edge treatment.',
    center: [0, 0], radius: 1.6,
    palette: PLANE_PALETTE,
    ramp: JULIA_RAMP,
    lake: [0.13, 0.05, 0.15],
    edge: [0.02, 0.02, 0.05],
    bg: [[0.012, 0.010, 0.026], [0.024, 0.020, 0.050]],
    // 3.0 and not 2.1 because of the ramp rather than this object: the generated loop
    // runs its lightness twice per turn of the hue, so at 2.1 octaves of escape count per
    // turn the exterior came out as hard bands. See rampFill in src/40_state.js.
    nuCycle: 3.0,
    fov: 15,
    // Base iteration cap, as in mandel2d above.
    view: { splitPx: 1.4, budget: 1000000, density: 1.0, glow: 0.0, kernel: 3.0, sigma: 1.732,
            fog: 0.10, iters: 1500 },
    // No coordinates: a Julia set moves with c, so any literal is on the set for one c
    // only. The aim is built from c by inverse iteration from the repelling fixed point,
    // which lands on the set for every c. See juliaPoint in src/32_plane.js.
    targets: [],
  },
};

// The same two objects with one orbit per pixel in a fragment shader, for comparison
// against the splat path. `direct: true` means there is no cut: no quadtree, no budget,
// no refinement, no instance buffer. Everything else is shared, from the camera and the
// targets down to the box filter; see src/62_direct.js.
for (const [id, base, nm] of [['mandelgpu', 'mandel2d', 'Mandelbrot set, per pixel'],
                              ['juliagpu', 'julia2d', 'Julia set, per pixel']]) {
  PLANE[id] = Object.assign({}, PLANE[base], {
    name: nm,
    direct: true,
    // Kept out of the menu: this is the same object as its splat twin, and a menu pick
    // reframes the camera, which is what the comparison cannot afford. Reached through
    // swapRepresentation, which swaps the representation and nothing else.
    hidden: true,
    // Tone, against a supersampled reference at the opening view: 0.4827 mean displayed
    // luminance here, 0.4792 true, 0.4629 on the splat path. Needs the progressive band
    // renderer in src/62_direct.js, without which directCap holds the orbit near a
    // thousand iterations against the ninety five thousand this view needs.
    blurb: 'The same escape time field as the splat version, evaluated once per pixel in a fragment ' +
           'shader instead of once per quadtree cell on the CPU. Two million independent orbits, no ' +
           'refinement and no budget: exact at the sampling rate of the screen. The orbit runs in ' +
           'double single arithmetic, a pair of floats carrying 48 bits, because WebGL 2 has no fp64 ' +
           'and a single float hits its wall at ten to the three. That puts this path a few decades ' +
           'shallower than the CPU one and some hundreds of times faster.',
    // No splat budget and no split threshold on this path. The base iteration count is
    // shared with the splat version so both are asked the same question; the stationary
    // boost is applied in directCap rather than planeRoot.
    view: Object.assign({}, PLANE[base].view, { budget: 1000, splitPx: 1.4 }),
  });
}

// The same view drawn the other way, both directions, so one button is a toggle.
const PIXEL_PAIR = {
  mandel2d: 'mandelgpu', mandelgpu: 'mandel2d',
  julia2d: 'juliagpu', juliagpu: 'julia2d',
};

// Depends on IFS from 20_ifs.js; only the filename prefixes enforce that order.
const PRESETS = Object.assign({}, PLANE, TERRAIN, IFS);
for (const k in IFS) prepareMaps(IFS[k].maps);

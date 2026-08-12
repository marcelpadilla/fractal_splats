/* ===================== the escape time field, in the plane ================ */
// Escape count as colour, drawn flat: 30_mandel.js reads the same orbit as a distance estimate
// and lights it as a surface instead. The smooth escape count is continuous in the point and
// unbounded at the boundary, so neighbouring splats get neighbouring colours and a periodic
// palette keeps cycling as the camera descends. Every function here writes into `pfl`.
//
//   z -> z^p + c        Mandelbrot: z0 = 0, c = the point.   dz/dc starts at 0
//                       Julia:      z0 = the point, c fixed. dz/dz0 starts at 1

const PB = 1e6, PB2 = PB * PB;              // bailout radius and its square
const pfl = {
  nu: 0,        // smooth escape count
  de: 0,        // distance estimate to the set
  gnu: 0,       // |grad nu|, per unit length in the plane
  ux: 0, uz: 0, // unit gradient of log|z|, pointing away from the set
  inside: 0,    // no escape: proven by a cycle, or assumed after maxIter
  iters: 0,
  // Why the orbit stopped: 0 analytic interior, 1 condemned by the derivative, 2 attracting
  // cycle, 3 iterations exhausted, 4 escaped, 5 the arithmetic gave up.
  why: 0,
};

// `rTol` is the cell's own half size and drives the derivative bailout: once |dz| passes
// 2 B log B / rTol the distance estimate is below the cell size, so the cell straddles the
// boundary whatever follows. It fires near iteration 100 instead of 3000.
function planeField(px, py, p, c0x, c0y, julia, maxIter, rTol) {
  // Analytic interior test for the cardioid and the period two bulb, a correctness fix rather
  // than an optimization: near a component's boundary the multiplier is close to one, so at a
  // 228 iteration cap the cycle test proved only 0.2 percent of drawn cells interior and left
  // 51.6 percent unresolved, drawn as the dark edge colour.
  if (!julia && p === 2 && inMainBody(px, py)) {
    pfl.iters = 0; pfl.inside = 1; pfl.nu = 0; pfl.de = 0; pfl.gnu = 0; pfl.why = 0;
    return;
  }
  let zx, zz, cx, cz, dx, dz;
  if (julia) { zx = px; zz = py; cx = c0x; cz = c0y; dx = 1; dz = 0; }
  else { zx = 0; zz = 0; cx = px; cz = py; dx = 0; dz = 0; }
  // rTol^2, because the condemnation test below is squared to avoid a square root.
  const rt2 = rTol > 0 ? rTol * rTol : 0;
  const lnp = p === 2 ? Math.LN2 : Math.log(p);
  const LPB = Math.log(PB);
  let rx = 0, rz = 0, per = 8, chk = 8;
  let n = 0, esc = false, m = 0, cyc = false, tPrev = 0;
  // Two loops. The general-p loop runs 16 ns an iteration and the p = 2 specialization 5, and a
  // near boundary cell at depth is 90 percent of a build. The specialization folds away z^(p-1)
  // and keeps the derivative and cycle tests inside the one-in-eight branch.
  //
  //   dz' = p z^(p-1) dz + (1 for the Mandelbrot, 0 for a Julia),  then z' = z^p + c
  if (p === 2) {
    const one = julia ? 0 : 1;
    for (; n < maxIter; n++) {
      const ndx = 2 * (zx * dx - zz * dz) + one;
      const ndz = 2 * (zx * dz + zz * dx);
      dx = ndx; dz = ndz;
      const nzx = zx * zx - zz * zz + cx;
      zz = 2 * zx * zz + cz; zx = nzx;
      m = zx * zx + zz * zz;
      if (m > PB2) { n++; esc = true; break; }
      if ((n & 7) === 7) {
        // Condemned: the distance estimate has dropped below the cell's own half size, so the
        // cell straddles the boundary and no further iteration changes that. de = |z| log(m) /
        // |dz| with m = |z|^2, so de < rTol squares to m log(m)^2 < rTol^2 |dz|^2, one log per
        // eight iterations and no square root. |z| cannot be a constant: the test fires mid
        // orbit, where |z| has passed 2 and moves by six orders of magnitude on its way to the
        // bailout radius. A fixed 48/rTol condemns cells whose true distance is 19 of their own
        // radii, and using the bailout radius 1e6 makes the test ten million times too lax.
        if (m > 1 && rt2 > 0) {
          const lm = Math.log(m);
          // t is the test as a ratio, so t > 1 is de < rTol. Kept as a ratio because its value
          // at the two checks either side of the crossing drives the interpolation below.
          const t = (dx * dx + dz * dz) * rt2 / (m * lm * lm);
          if (t > 1) {
            // nu for a condemned cell, interpolated across the crossing of t = 1 so it is
            // continuous in the point and drifts by about one iteration per level of refinement.
            // t grows geometrically along the orbit, so the crossing is linear in log t. Against
            // a 3x3 supersampled reference at 1377x1592: nu = 0 gives rms 0.224 and reads 72
            // percent set against a true 60, the cell mean of the ramp 0.118 and a grey halo,
            // nu = n 0.063 but in jumps of 8, a quarter turn of the ramp, because the test runs
            // one iteration in eight. See FILT_MAX in src/52_cut_plane.js.
            let f = 1;
            if (tPrev > 0) {
              const b = Math.log(t / tPrev);
              f = b > 0 ? -Math.log(tPrev) / b : 1;
              if (!(f >= 0)) f = 0; else if (f > 1) f = 1;
            }
            pfl.iters = n; pfl.inside = 0;
            pfl.nu = n - 8 + 8 * f;
            pfl.de = 0; pfl.gnu = 1e30; pfl.why = 1;
            return;
          }
          tPrev = t;
        }
        // Brent cycle detection. 1e-10 rather than 1e-16 because a multiplier near one needs
        // thousands of iterations to converge to the last bit; the cost is that a near boundary
        // point whose orbit nearly closes is painted as set. One check in eight delays detection
        // by at most eight and keeps it off the critical path.
        if (n > 24) {
          if (Math.abs(zx - rx) + Math.abs(zz - rz) < 1e-10) { cyc = true; break; }
          if ((chk -= 8) <= 0) { rx = zx; rz = zz; per += per; chk = per; }
        }
      }
    }
  } else {
    const one = julia ? 0 : 1;
    for (; n < maxIter; n++) {
      let wx = 1, wz = 0;                 // z^(p-1) by repeated multiplication
      for (let k = 1; k < p; k++) { const t = wx * zx - wz * zz; wz = wx * zz + wz * zx; wx = t; }
      const mx = p * wx, mz = p * wz;
      const ndx = mx * dx - mz * dz + one;
      const ndz = mx * dz + mz * dx;
      dx = ndx; dz = ndz;
      const nzx = wx * zx - wz * zz + cx;
      zz = wx * zz + wz * zx + cz; zx = nzx;
      m = zx * zx + zz * zz;
      if (m > PB2) { n++; esc = true; break; }
      if ((n & 7) === 7) {
        if (m > 1 && rt2 > 0) {
          const lm = Math.log(m);
          const t = (dx * dx + dz * dz) * rt2 / (m * lm * lm);
          if (t > 1) {
            // See the p = 2 loop above.
            let f = 1;
            if (tPrev > 0) {
              const b = Math.log(t / tPrev);
              f = b > 0 ? -Math.log(tPrev) / b : 1;
              if (!(f >= 0)) f = 0; else if (f > 1) f = 1;
            }
            pfl.iters = n; pfl.inside = 0;
            pfl.nu = n - 8 + 8 * f;
            pfl.de = 0; pfl.gnu = 1e30; pfl.why = 1;
            return;
          }
          tPrev = t;
        }
        if (n > 24) {
          if (Math.abs(zx - rx) + Math.abs(zz - rz) < 1e-10) { cyc = true; break; }
          if ((chk -= 8) <= 0) { rx = zx; rz = zz; per += per; chk = per; }
        }
      }
    }
  }
  pfl.iters = n;
  if (!esc) {
    // A cycle proves interior. Running out of iterations proves nothing, so an exhausted cell is
    // unresolved and very close to the set, not interior.
    if (cyc || n < maxIter) { pfl.inside = 1; pfl.nu = 0; pfl.de = 0; pfl.gnu = 0; pfl.why = 2; return; }
    // Exhausted, and no local test resolves it. What an exhausted cell really is depends on
    // where in the plane the frame sits, not on depth: against a sixty times cap, 68 percent
    // were interior at zoom 10^0.4, 82 at 10^3, 17 at 10^6.2, none at 10^10. Nor does a distance
    // estimate separate them: `de > 64 rTol means interior` called 1339 exterior cells interior
    // against 318 correct at 10^6 and took the frame to 73 percent set, because a deep exterior
    // point can sit hundreds of cell radii out and still exceed the cap, the potential near a
    // Misiurewicz point being minute while the distance is not. Only more iterations resolve it,
    // which is what the refinement ramp in planeRoot supplies.
    pfl.inside = 0; pfl.nu = n; pfl.de = 0; pfl.gnu = 1e30; pfl.why = 3;
    return;
  }
  const dm = Math.sqrt(dx * dx + dz * dz);
  const lr = 0.5 * Math.log(m);                        // log|z|
  if (!(dm < 1e300) || !(m < 1e300) || !(dm > 1e-300)) {
    pfl.inside = 0; pfl.nu = 0; pfl.de = 0; pfl.gnu = 1e30; pfl.why = 5;
    return;
  }
  const az = Math.sqrt(m);
  // Smooth escape count: n minus log_p of log|z| over log B, which removes the integer banding
  // and makes nu a continuous function of the point.
  pfl.nu = n - Math.log(lr / LPB) / lnp;
  pfl.de = 2 * az * lr / dm;
  // |grad nu| = |dz| / (|z| log|z| ln p). Refinement is driven by it: |grad nu| times the cell
  // size is how much of the colour ramp the cell spans, and under a fraction of a step is done.
  pfl.gnu = dm / (az * (lr > 1e-12 ? lr : 1e-12) * lnp);
  // grad log|z| = (Re(dz/z), -Im(dz/z)), pointing away from the set. With `de` this gives a
  // Newton step onto the boundary, which names an aim point exactly at any depth.
  const wr = (dx * zx + dz * zz) / m, wi = (dz * zx - dx * zz) / m;
  const gm = Math.sqrt(wr * wr + wi * wi);
  if (gm > 1e-300) { pfl.ux = wr / gm; pfl.uz = -wi / gm; } else { pfl.ux = 0; pfl.uz = 0; }
  pfl.inside = 0; pfl.why = 4;
}

// A point exactly on the Julia set, by the chaos game run backwards. The set is the closure of
// the backward orbit of a repelling fixed point, so solve z^p + c = z for a root with
// |p z^(p-1)| > 1 and apply inverse branches z <- (z - c)^(1/p), branch chosen by index; the
// sequence equidistributes, so `seed` picks a different place on the same set. Constructed
// rather than named because c is live: a coordinate on the set for one c lies inside the filled
// set for another, and the filled set is one flat colour.
function juliaPoint(cx, cy, p, seed) {
  // Newton on f(z) = z^p + c - z, started off the real axis so the iteration does not sit on a
  // symmetry line. An attracting root is rejected by trying the next start.
  let zx = 0.6, zy = 0.35;
  for (let attempt = 0; attempt < 6; attempt++) {
    for (let k = 0; k < 40; k++) {
      // w = z^(p-1), f = w z + c - z, f' = p w - 1
      let wx = 1, wy = 0;
      for (let j = 1; j < p; j++) { const t = wx * zx - wy * zy; wy = wx * zy + wy * zx; wx = t; }
      const fx = wx * zx - wy * zy + cx - zx, fy = wx * zy + wy * zx + cy - zy;
      const dx = p * wx - 1, dy = p * wy;
      const q = dx * dx + dy * dy;
      if (!(q > 1e-300)) break;
      zx -= (fx * dx + fy * dy) / q;
      zy -= (fy * dx - fx * dy) / q;
    }
    let wx = 1, wy = 0;
    for (let j = 1; j < p; j++) { const t = wx * zx - wy * zy; wy = wx * zy + wy * zx; wx = t; }
    const mult = p * Math.hypot(wx, wy);
    if (mult > 1.0000001 && isFinite(zx + zy)) break;
    zx = -0.7 + 0.31 * attempt; zy = 0.62 - 0.19 * attempt;
  }
  if (!isFinite(zx + zy)) return [0, 0];
  // Backward orbit. Each step contracts by 1/|f'|, so 48 steps put the point on the set to full
  // double precision.
  let st = (seed | 0) * 2654435761 + 1013904223;
  for (let k = 0; k < 48; k++) {
    st = (st * 1664525 + 1013904223) | 0;
    const br = ((st >>> 8) % p);
    const ax = zx - cx, ay = zy - cy;
    const rr = Math.hypot(ax, ay);
    if (!(rr > 1e-300)) return [zx, zy];
    const th = (Math.atan2(ay, ax) + 2 * Math.PI * br) / p;
    const rp = Math.pow(rr, 1 / p);
    zx = rp * Math.cos(th); zy = rp * Math.sin(th);
  }
  return [zx, zy];
}

// A boundary point of the exponent p Mandelbrot set, by bisection along a ray. A named
// coordinate belongs to one exponent: the seahorse valley point is on the quadratic boundary to
// eighteen digits and well inside the cubic set, so raising the exponent aims into solid
// interior. Bisection needs no names, because c = 0 is in the set for every p, z = 0 being a
// fixed point of z -> z^p, and a large enough |c| escapes for every p. The crossing is an
// ordinary boundary point, not a distinguished one such as a Misiurewicz point.
function mandelBoundaryRay(theta, p, maxIter) {
  const ux = Math.cos(theta), uy = Math.sin(theta);
  const escapes = (r) => {
    planeField(ux * r, uy * r, p, 0, 0, false, maxIter, 0);
    return pfl.why === 4 || pfl.why === 5;
  };
  let lo = 0, hi = 2.5;
  for (let k = 0; k < 40 && !escapes(hi); k++) hi *= 1.6;   // out until it certainly leaves
  if (!escapes(hi)) return [0, 0];
  for (let k = 0; k < 200; k++) {
    const mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi) break;                      // the doubles have run out
    if (escapes(mid)) hi = mid; else lo = mid;
  }
  const r = 0.5 * (lo + hi);
  return [ux * r, uy * r];
}

// Push a point onto the boundary of whichever set this is, by Newton steps along the distance
// estimate. The step projectToBoundary takes for the Mandelbrot, but valid for a Julia set too:
// an unprojected Julia aim sits in uniform escape count, and 31 of 90 frames of a Julia descent
// came back dark before this existed.
function projectToPlane(px, py, p, c0x, c0y, julia, maxIter) {
  for (let k = 0; k < 8; k++) {
    planeField(px, py, p, c0x, c0y, julia, maxIter, 0);
    if (pfl.inside || !(pfl.de > 0)) break;
    const step = pfl.de;
    if (!isFinite(step) || step > 1) break;
    px -= step * pfl.ux;
    py -= step * pfl.uz;
  }
  return [px, py];
}

/* ------------------------- the field cache, for the plane ------------------ */
// A cell's field value does not depend on the camera, and the quadtree's centres are exact
// binary halvings of a fixed root, so the key is exact and a hit cannot be wrong. It pays
// because a cut lands every eleven frames while the camera moves 0.07 e-folds, so most cells of
// the next cut were just evaluated.
//
// The key is (cx, cz, r), three doubles, and r has to be in it: a cell corner is also the centre
// of a cell four levels down and rTol sets the derivative bailout, so the two are different
// questions about the same point. Four way linear probing, because at load 0.15 the table is
// collision limited rather than capacity limited and four probes cost four integer compares.
const PFC_BITS = 20, PFC_N = 1 << PFC_BITS, PFC_MASK = PFC_N - 1, PFC_WAYS = 4;
const pfcX = new Float64Array(PFC_N), pfcZ = new Float64Array(PFC_N), pfcR = new Float64Array(PFC_N);
const pfcNu = new Float64Array(PFC_N);
const pfcV = new Float32Array(PFC_N * 3);     // de, gnu, iters
const pfcWhy = new Int8Array(PFC_N);
const pfcGen = new Int32Array(PFC_N);
let pfcCur = 0, pfcHits = 0, pfcMiss = 0, pfcSaved = 0;
const pfcBuf = new Float64Array(3);
const pfcInt = new Int32Array(pfcBuf.buffer);

function pfcHash(cx, cz, r) {
  pfcBuf[0] = cx; pfcBuf[1] = cz; pfcBuf[2] = r;
  let h = Math.imul(pfcInt[0], 0x9e3779b1) ^ Math.imul(pfcInt[1], 0x85ebca6b) ^
          Math.imul(pfcInt[2], 0xc2b2ae35) ^ Math.imul(pfcInt[3], 0x27d4eb2f) ^
          Math.imul(pfcInt[4], 0x165667b1) ^ Math.imul(pfcInt[5], 0xd3a2646c);
  h ^= h >>> 15; h = Math.imul(h, 0x2545f491); h ^= h >>> 13;
  return h & PFC_MASK;
}

// Everything the answer depends on besides the cell: the quantized iteration cap, the exponent,
// the family parameter, and which set this is. Folded into one non zero generation stamp so a
// change invalidates the whole table without touching 26 MB of it.
function planeStamp(maxIter, p, c0x, c0y, julia) {
  let g = Math.imul(maxIter + 1, 1000003) ^ Math.imul(p, 0x5bd1e995) ^ (julia ? 0x1b873593 : 0);
  pfcBuf[0] = c0x; g = Math.imul(g ^ pfcInt[0], 0x85ebca6b) ^ pfcInt[1];
  pfcBuf[0] = c0y; g = Math.imul(g ^ pfcInt[0], 0xc2b2ae35) ^ pfcInt[1];
  return g | 1;
}

// Same contract as planeField, including writing `pfl`. Only the quadtree uses it;
// projectToPlane, juliaPoint and mandelBoundaryRay call planeField directly, because they
// evaluate with rTol = 0 and that is a different question about the same point.
function planeFieldCached(px, py, p, c0x, c0y, julia, maxIter, rTol) {
  // A lookup is three float64 stores through a shared buffer, six imuls and a four way probe, so
  // at 17 iterations a cell it costs more than the field it avoids: on the opening view this
  // wrapper measured 29.8 percent of the cut against 24.3 for planeField. Deep down a cell runs
  // hundreds of iterations and the cache is worth a factor of two, hence the gate on cost.
  if (!job.pfcOn) { planeField(px, py, p, c0x, c0y, julia, maxIter, rTol); return; }
  const h0 = pfcHash(px, py, rTol);
  for (let q = 0; q < PFC_WAYS; q++) {
    const s = (h0 + q) & PFC_MASK;
    if (pfcGen[s] === pfcCur && pfcX[s] === px && pfcZ[s] === py && pfcR[s] === rTol) {
      const v3 = s * 3;
      pfl.nu = pfcNu[s]; pfl.de = pfcV[v3]; pfl.gnu = pfcV[v3 + 1];
      pfl.iters = pfcV[v3 + 2];
      const w = pfcWhy[s];
      pfl.why = w; pfl.inside = (w === 0 || w === 2) ? 1 : 0;
      pfcHits++; pfcSaved += pfl.iters;
      return;
    }
  }
  planeField(px, py, p, c0x, c0y, julia, maxIter, rTol);
  pfcMiss++;
  let s = h0;
  for (let q = 0; q < PFC_WAYS; q++) {
    const t = (h0 + q) & PFC_MASK;
    if (pfcGen[t] !== pfcCur) { s = t; break; }
  }
  const v3 = s * 3;
  pfcX[s] = px; pfcZ[s] = py; pfcR[s] = rTol; pfcGen[s] = pfcCur;
  pfcNu[s] = pfl.nu; pfcV[v3] = pfl.de; pfcV[v3 + 1] = pfl.gnu; pfcV[v3 + 2] = pfl.iters;
  pfcWhy[s] = pfl.why;
}

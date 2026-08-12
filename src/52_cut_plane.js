/* ================= refinement: the escape time field, in plane ============ */
// A quadtree over the plane, one flat Gaussian per cell, coloured by the smooth escape count.
// Split test: span = |grad nu| * 2r / (nuCycle * nu^(2/3)), the fraction of one turn of the
// colour ramp a cell of half size r covers; below spanTol splitting cannot change a pixel, and
// emitPlane draws the cell as the exact box filtered mean of the ramp over its own span, so being
// unresolved costs saturation and never correctness. Every splat sits in the plane y = 0 and
// nothing is lit, so the accumulation is a Gaussian weighted reconstruction of the field: no
// prepass, no sort.
const PCHUNK = 96;
const whyN = new Int32Array(6), whyI = new Float64Array(6);
// The largest a cell may be on screen whatever its ramp span, as a fraction of the frame height
// rather than an absolute pixel count, so the cell count across the frame is fixed: 1/48 is about
// 48 by 55 either way, 33 px at a height of 1592 and 13 px at 621. It stops the quadtree showing
// as soft banding in the far field. An absolute 6 px cap asks for 61 000 cells on a 1377x1592
// window and never refines the boundary at all.
const PLANE_MAX_FRAC = 1 / 48;
// Milliseconds of CPU a plane cut may spend before it stops splitting and drains. An IFS cut
// costs 19 to 22 ms, so 90 lands within about eleven frames of a 5 ms slice, which is 0.07
// e-folds of camera motion at the default descent rate.
let PLANE_MS = 90;

function planeRoot(vw, vh) {
  const P = PRESETS[cfg.preset];
  job.nmaps = 4;
  job.usePrio = true;
  job.julia = P.julia ? 1 : 0;
  job.power = Math.max(2, Math.min(8, Math.round(cfg.power)));
  job.c0x = cfg.cx; job.c0y = cfg.cy;
  // The ramp coordinate is 3*cbrt(nu), not nu, so nuCycle is cube roots of escape count per turn
  // and one setting holds the band count over the whole descent; linear in nu, a cycle that gives
  // the opening view a couple of bands gives a deep view two hundred. It also makes the object
  // cheaper with depth: |grad u| = |grad nu| / (nuCycle * nu^(2/3)).
  job.nuCyc = P.nuCycle || 4.5;
  job.nuInv = 1 / job.nuCyc;
  // Peak optical depth per cell, and only the floor kernPeak hands back: the bandwidth is a per
  // cell decision and the peak has to follow it, or a narrow kernel leaves the background showing
  // through at the worst point of the lattice. See kernPeak.
  job.wSurf = T_MIN;
  // The coverage cap, which in the Gaussian view is also the size of the primitive, so no cell
  // may sit at 70 px because the field under it is flat. See GAUSS_PX_PLANE in src/40_state.js.
  job.maxPx = cfg.gauss ? GAUSS_PX_PLANE : Math.max(6, vh * PLANE_MAX_FRAC);
  // The ceiling bounds latency only, so it lifts when the camera stops: `stillCuts` counts
  // consecutive cuts finished stationary and the ceiling doubles each time, to eight doublings,
  // 90 ms moving and 23 s standing still. At 10^6 on a 1377x1592 window a cut under 90 ms is
  // 19 000 splats against 242 000 converged. Converging costs 0.44 s at the opening view, 2.2 s
  // at 10^2 and 8.0 s at 10^6, where the iteration cap is 95 403.
  const boost = 1 << (stillCuts < 8 ? stillCuts : 8);
  job.msCap = PLANE_MS * boost;
  // The iteration cap steps once rather than ramping with the ceiling, because the field cache
  // keys on the quantized cap and every change of it invalidates every stored cell: ramping both
  // made each step pay cold, five cuts over 2.9 s of CPU ending at 43 000 splats with no cell
  // under 3 px against 275 000 for one converged cut.
  const itBoost = stillCuts > 0 ? 16 : 1;
  job.fadeK = 1 / (FADE_BAND - 1); job.budK = 1 / (BUD_BAND - 1);
  // Iterations grow with depth, since the escape count at distance d from the set grows like
  // log(1/d), and are quantized to powers of 1.3 above 80 so the field cache survives a descent.
  // The stillness boost applies here too: at 10^6 on 1377x1592 under the 90 ms ceiling, raising
  // the base cap from 500 to 4000 took the cut from 45 000 cells to 17 000 and the unresolved
  // width from 4.5 px to 9.3, while converged it took the exhausted region from 63.3 percent of
  // the frame to 13.0.
  const lz = Math.max(0, logZoom());
  const want = Math.max(80, Math.min(200000, cfg.iters * (1 + 0.45 * lz) * itBoost));
  const mstep = Math.max(0, Math.round(Math.log(want / 80) / Math.log(1.3)));
  job.maxIter = Math.round(Math.min(200000, 80 * Math.pow(1.3, mstep)));
  stats.maxIter = job.maxIter;
  whyN.fill(0); whyI.fill(0);
  // The reference point every batched query is an offset from. It has to be inside the frame,
  // because the offset crosses to the GPU as one float; planeFlush guards a level whose cells are
  // too far off it.
  job.gcx = job.pos[0]; job.gcz = job.pos[2];
  pendN = 0;
  gfVerdict();
  gfBatches = 0; gfQueries = 0; gfMs = 0; gfIters = 0;
  // Arm the field cache. A change of exponent, family parameter, set or quantized iteration cap
  // makes every stored answer a different question, so the generation is bumped, not the table
  // cleared.
  const st = planeStamp(job.maxIter, job.power, job.c0x, job.c0y, job.julia);
  if (st !== pfcCur) { pfcCur = st; }
  pfcHits = 0; pfcMiss = 0; pfcSaved = 0;
  // See planeFieldCached: under PFC_MIN_ITERS a cell is cheaper to recompute than to look up. The
  // average is the cut's own executed cost, smoothed, so the cache arms itself as cells get dear.
  job.pfcOn = planeItEma > PFC_MIN_ITERS;
  // The root cell is 2.5 times the framing radius: the camera frames a sphere of radius P.radius,
  // so a square of side 2*P.radius fills the height and leaves black beyond the frame on wide
  // aspects. It costs one level of depth, and everything outside the frustum is culled by `place`
  // before it is refined.
  if (!planeCell(0, P.center[0], P.center[1], P.radius * 2.5, 0)) return;
  frontier[0] = 0;
  job.nf = 1;
}

// One cell, one field evaluation. `r` is the half size. The splat's variance is r^2/3, the cell's
// own second moment, scaled by the preset's sigma: at sigma sqrt(3) one standard deviation is the
// cell's half size, so cells 2r apart overlap at one sigma and the profiles sum flat to under a
// percent, which makes the accumulation a reconstruction rather than a mosaic. A thin term across
// the plane keeps the covariance invertible. The field is sampled at the centre, and a verdict
// about the centre is not a verdict about the cell: the Mandelbrot root cell is centred inside
// the main cardioid, so a centre only test called it uniform interior.
const RT2 = Math.SQRT2;
// Is the set inside this cell. One test, read by the split decision, the priority, the kernel
// bandwidth and the anti-aliasing. `de` is a lower bound on the distance from the centre to the
// set, so an escaped cell with de >= r sqrt(2) cannot contain any of it; a condemned or exhausted
// cell is one the orbit could not settle. A proven interior cell is all set and nothing about it
// is unresolved.
function planeTangle(kind, de, r) {
  return kind === 2 || kind === 3 || (kind === 0 && de < RT2 * r);
}
// Below this the escape count carries no colour: a point that leaves on the first iteration is
// out in the far field. Floors nu in the ramp coordinate and in the span.
const NU_MIN = 1.5;
function clamp01(x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
// The cross fade weight the next cell created inherits. A global rather than an argument because
// planeCell is called from three places and only one of them is a split.
let cellBl = 1;
// The geometry half of a cell: everything that does not need the field. Split out because a GPU
// batch needs the geometry of every cell in it before any of them has an answer, and run first so
// a cell outside the frustum is culled before its orbit runs, which is 14 percent of the cells
// created at the opening view. See planeFlush.
function planeCellGeom(n, cx, cz, r, depth) {
  const o6 = n * 6, o9 = n * 9;
  const v = r * r / 3, th = 0.06 * r, t2 = th * th;
  pool.cov[o6] = v;      pool.cov[o6 + 1] = 0; pool.cov[o6 + 2] = 0;
  pool.cov[o6 + 3] = t2; pool.cov[o6 + 4] = 0;
  pool.cov[o6 + 5] = v;
  const o3 = n * 3;
  pool.mu[o3] = cx; pool.mu[o3 + 1] = 0; pool.mu[o3 + 2] = cz;
  pool.M[o9] = cx; pool.M[o9 + 1] = cz; pool.M[o9 + 2] = r;
  pool.M[o9 + 6] = -1;             // no averaged colour yet; see emitPlane and planeFlush
  pool.w[n] = 1;
  pool.depth[n] = depth;
  pool.ls[n] = Math.log10(r);
  pool.bl[n] = cellBl;
  const rad = r * job.invD * job.sig * 1.1547;      // 2/sqrt(3): the cell's own half width
  return place(n, (cx - job.pos[0]) * job.invD, (0 - job.pos[1]) * job.invD,
               (cz - job.pos[2]) * job.invD, rad);
}

// The field half: one verdict about the cell's centre, plus whether the corner guard agreed,
// turned into a ramp span and a priority. Callers are the CPU path below and the GPU path in
// planeFlush.
function planeFinish(n, nu, de, gnu, why, agree) {
  const o9 = n * 9, r = pool.M[o9 + 2];
  const inside = (why === 0 || why === 2);
  // The span is the ramp's filter width and nothing else. It used to carry the split verdict too,
  // as a 1e6 sentinel wherever the distance estimate put the set inside the cell, so emitPlane
  // averaged the ramp over the whole FILT_MAX cap of 0.30 of a turn on every cell touching the
  // set, and the four pixel band there came back at 0.71 of a per pixel reference's saturation.
  // The split is decided from the kind and the distance estimate now; see `tangle` in stepPlane.
  let span;
  if (inside || why === 3) span = agree ? 0 : 1e6;
  else if (why === 1) span = 1e6;                  // condemned: nothing is known about the ramp
  else {
    const nc = nu > NU_MIN ? nu : NU_MIN;
    span = gnu * 2 * r * job.nuInv / Math.pow(nc, 2 / 3);
    if (!(span < 1e6)) span = 1e6;
  }
  pool.M[o9 + 3] = nu; pool.M[o9 + 4] = de;
  // Four kinds. 0 escaped, so nu and de are both real numbers about this point. 1 proven
  // interior, by a cycle or by the analytic body test. 2 condemned: the boundary is inside the
  // cell, nu is the interpolated iteration the orbit was abandoned at and de is not a distance.
  // 3 exhausted, neither escaped nor closed, which is stronger evidence of being in the set than
  // 2 is. Collapsing 2 and 3 into 1 paints unrefined exterior as set.
  pool.M[o9 + 5] = inside ? 1 : (why === 4 ? 0 : (why === 1 ? 2 : 3));
  pool.cc[n] = span;
  // Priority under budget pressure: footprint times ramp span, so the budget goes where the
  // colour is both large on screen and changing. A cell the set runs through takes the maximum
  // whatever its span says, because there the unresolved thing is the geometry.
  const pspan = planeTangle(pool.M[o9 + 5], de, r) ? 4 : span;
  pool.prio[n] = pool.fp[n] * (pspan > 4 ? 4 : pspan);
}

function planeCell(n, cx, cz, r, depth) {
  if (!planeCellGeom(n, cx, cz, r, depth)) return false;
  planeCellField(n, cx, cz, r);
  return true;
}

// The field, the corner guard and the finish, on the CPU, for a cell whose geometry is written.
// Called by planeCell and by planeFlushCPU, the fallback when a batch is too small or too far off
// centre for the GPU.
function planeCellField(n, cx, cz, r) {
  planeFieldCached(cx, cz, job.power, job.c0x, job.c0y, job.julia, job.maxIter, r);
  job.evals++; job.iters += pfl.iters;
  whyN[pfl.why]++; whyI[pfl.why] += pfl.iters;
  const inside = pfl.inside, nu = pfl.nu, de = pfl.de, gnu = pfl.gnu, why = pfl.why;
  let agree = true;
  if (inside) {
    // Guard one, interior. There is no exterior distance estimate to lean on, so test the four
    // corners too and call the cell uniform only if they agree; without it every interior cell
    // refines to the pixel. The analytic test runs first, the cardioid and the period two bulb
    // being two lines of algebra: at the opening view the cut ran 466 129 evaluations for 155 465
    // cells, of which 310 664 were corner tests, and 98 percent of interior centres were proven
    // by `inMainBody` rather than by an orbit.
    const anal = why === 0;
    let allIn = true;
    for (let k = 0; k < 4 && allIn; k++) {
      const ox = (k & 1) ? r : -r, oz = (k & 2) ? r : -r;
      if (anal && inMainBody(cx + ox, cz + oz)) continue;
      planeFieldCached(cx + ox, cz + oz, job.power, job.c0x, job.c0y, job.julia, job.maxIter, r);
      job.evals++; job.iters += pfl.iters;
      if (!pfl.inside) allIn = false;
    }
    agree = allIn;
  } else if (why === 3) {
    // Guard three, exhausted. An orbit that runs out of iterations has no escape count and no
    // distance, and giving it |grad nu| = 1e30 makes it demand refinement however large a uniform
    // region is: at 10^6 on 1377x1592, 237 358 of 333 000 cells came back exhausted, so the
    // budget tiled the flat deep boundary layer instead of the boundary. So test the corners as
    // well. They are shared with the four neighbours and the cache keys on exact coordinates, so
    // this costs about one extra evaluation per cell.
    let allEx = true;
    for (let k = 0; k < 4 && allEx; k++) {
      const ox = (k & 1) ? r : -r, oz = (k & 2) ? r : -r;
      planeFieldCached(cx + ox, cz + oz, job.power, job.c0x, job.c0y, job.julia, job.maxIter, r);
      job.evals++; job.iters += pfl.iters;
      if (!(pfl.why === 3 || pfl.inside)) allEx = false;
    }
    agree = allEx;
  }
  // Guard two, exterior, lives in planeFinish with the span arithmetic. A cell whose half
  // diagonal fits inside `de` contains no boundary: out there the field is analytic and its
  // linear estimate can be trusted. A cell that fails that test may straddle anything, so it
  // splits whatever its centre reports.
  planeFinish(n, nu, de, gnu, why, agree);
}

/* ---------------- the same field, in batches, on the GPU ------------------ */
// The quadtree stays on the CPU and only the arithmetic moves. Timed on an RTX 4090 at 1100x1280
// on a converged cut's real query set, synchronous readback included, one batch against the same
// points on one CPU thread, in milliseconds: 13.8 against 166 at 10^0 for 213 666 cells at cap
// 25 695, 13.7 against 637 at 10^2.2, 14.0 against 137 at 10^4, and 19.6 against 2447 at 10^6 for
// 320 484 cells at cap 95 403. See ptime.html and knowledge/plane_lod.md. stepPlane is level
// synchronous, so a level's children are one batch: decide the splits, write and cull the
// children's geometry, then one draw and one readback answers all of them. Centres that come back
// interior or exhausted need the corner guard, which is a second batch. Two round trips a level,
// above GF_MIN only. The readback is synchronous of necessity; see the header of src/63_gfield.js.
let pend = new Int32Array(1 << 16);
let pendN = 0;
// The centre verdicts of a batch, held while the corner batch runs over the same arrays, and the
// map from a corner query back to the cell that asked for it.
let bNu = new Float32Array(0), bDe = bNu, bGnu = bNu;
let bKind = new Int32Array(0), bAgree = new Uint8Array(0);
let cornOwner = new Int32Array(0);
// The summed set fill over a tangle cell's quarter points, accumulated as the corner batch comes
// back, and the count of how many arrived. Not a colour: see planeNeedsAA.
let bAcc = new Float32Array(0), bAccN = new Int32Array(0);
// How deep the double single arithmetic may be trusted for a cut. The per pixel object runs the
// same kernel near 10^9; a cut is stricter because the quadtree branches on the verdict, so a
// disagreement is a different tree rather than a wrong pixel. dscheck.html against the CPU field,
// per cell kind: 99.9 percent agreement at the opening view, 98.9 at 10^2, 99.5 at 10^4, 98.0 at
// 10^6 and 89.2 at 10^8, every disagreement being escaped against condemned and never interior
// confusion.
let GF_MAX_DEC = 7;
// Self test, since the GPU path runs on machines nobody here has. A batch reports the iterations
// it executed and how long it took, so the achieved rate is known; one CPU thread does 3.1e5
// iterations a millisecond, measured, and two consecutive cuts under half that switch the path
// off for the session. An RTX 4090 reads 1.1 to 4.5 million a millisecond.
const GF_CPU_RATE = 1.55e5;         // half of one CPU thread's measured 3.1e5 iters/ms
let gfSlow = 0, gfIters = 0;
function gfVerdict() {
  if (!gfOn || gfMs <= 0 || gfIters < 1e6) return;
  if (gfIters / gfMs >= GF_CPU_RATE) { gfSlow = 0; return; }
  if (++gfSlow >= 2) { gfOn = false; gfWhy = 'slower than this CPU'; }
}
let gfWhy = '';
function planeGpuField() {
  return gfOn && job.plane && progGF !== null && typeof gl !== 'undefined' && gl &&
         fboInternal !== 0 &&
         logZoom() < GF_MAX_DEC;   // logZoom(), not stats.logZoom: that one is only
                                   // refreshed while the autopilot is descending
}
// The progGF !== null test above is a schedule, not a null check: linking the batch program
// compiles PERT_GLSL through the HLSL compiler at about 1000 ms on this driver, the first batch
// of the first cut coming back at 1076 ms of which 79 was work. So the first plane frame is built
// on the CPU and gfWarm starts the link only after that frame has been presented, then polls it
// without blocking.
function gfWarm() { if (gfOn && !progGF && typeof gl !== 'undefined' && gl) initGField(); }
function pendPush(n) {
  if (pendN >= pend.length) { const t = new Int32Array(pend.length * 2); t.set(pend); pend = t; }
  pend[pendN++] = n;
}
function bReserve(n) {
  if (bNu.length >= n) return;
  const m = 1 << Math.ceil(Math.log2(Math.max(n, 4096)));
  bNu = new Float32Array(m); bDe = new Float32Array(m); bGnu = new Float32Array(m);
  bKind = new Int32Array(m); bAgree = new Uint8Array(m);
  bAcc = new Float32Array(m); bAccN = new Int32Array(m);   // the summed set fill, one per cell
  if (cornOwner.length < GF_CAP) cornOwner = new Int32Array(GF_CAP);
}
// kind as gfEvaluate packs it, back to the `why` numbering the CPU stats use.
const KIND_WHY = [4, 2, 1, 3];
// gfA carries (nu, de, gnu, kind + 8*iters) per slot; see gfEvaluate. The fourth channel is exact
// in fp32 up to two million iterations, so the split is a truncation and not a rounding.
function packKind(p) { return p - 8 * Math.floor(p / 8); }

// Run `n` queries already staged in gfQ, in chunks of GF_CAP, calling `take(absolute, slot)` for
// each answer. Chunking moves the next chunk to the front of gfQ, safe because what it overwrites
// is consumed and a chunk is never larger than GF_CAP.
function gfRun(n, cap, take) {
  let done = 0;
  while (done < n) {
    const m = Math.min(GF_CAP, n - done);
    if (done > 0) gfQ.copyWithin(0, done * 4, (done + m) * 4);
    if (!gfEvaluate(m, cap, job.gcx, job.gcz)) return false;
    for (let k = 0; k < m; k++) take(done + k, k);
    done += m;
  }
  return true;
}

// Everything pending, finished. Falls back to the CPU when the batch is too small to pay for
// itself, the GPU is unavailable, or the offsets are too large for an fp32 query to name the
// cell: dc is exact as given but is computed as a difference of doubles and rounded, so a cell a
// million frame widths off centre is named only to a millionth of its own offset. That only
// happens high in the tree, where levels go to the CPU anyway.
function planeFlush() {
  const n = pendN;
  pendN = 0;
  if (n === 0) return;
  if (n < GF_MIN || !planeGpuField()) { planeFlushCPU(n); return; }
  let far = 0, rmin = Infinity;
  for (let k = 0; k < n; k++) {
    const o9 = pend[k] * 9;
    const a = Math.abs(pool.M[o9] - job.gcx), b = Math.abs(pool.M[o9 + 1] - job.gcz);
    if (a > far) far = a;
    if (b > far) far = b;
    if (pool.M[o9 + 2] < rmin) rmin = pool.M[o9 + 2];
  }
  if (far * 1.2e-7 > 0.02 * rmin) { planeFlushCPU(n); return; }
  bReserve(n);
  const cap = job.maxIter;
  gfReserve(n);
  for (let k = 0; k < n; k++) {
    const o9 = pend[k] * 9, q = k * 4;
    gfQ[q] = pool.M[o9] - job.gcx;
    gfQ[q + 1] = pool.M[o9 + 1] - job.gcz;
    gfQ[q + 2] = pool.M[o9 + 2];
    gfQ[q + 3] = 0;
  }
  if (!gfRun(n, cap, (k, slot) => {
    const o = slot * 4;
    bNu[k] = gfA[o]; bDe[k] = gfA[o + 1]; bGnu[k] = gfA[o + 2];
    const p = gfA[o + 3], kind = packKind(p);
    bKind[k] = kind; bAgree[k] = 1;
    job.evals++; job.iters += (p - kind) / 8; gfIters += (p - kind) / 8;
    whyN[KIND_WHY[kind]]++;
  })) { planeFlushCPU(n); return; }
  // The corner guard, as a second batch: a cell whose centre is interior or exhausted is uniform
  // only if its four corners agree, and the guards in planeCellField say why. The analytic body
  // test proves a corner interior for nothing, so those are never sent. The same batch does the
  // anti-aliasing, being the same four points; a tangle cell needs no guard and its corners are
  // shaded rather than tested. See planeNeedsAA.
  const anal = !job.julia && job.power === 2;
  gfReserve(GF_CAP);
  for (let k = 0; k < n; k++) bAccN[k] = 0;
  const cornTake = (abs, slot) => {
    const k = cornOwner[abs], o = slot * 4;
    const p = gfA[o + 3], kind = packKind(p);
    job.evals++; job.iters += (p - kind) / 8; gfIters += (p - kind) / 8;
    if (bKind[k] === 1 || bKind[k] === 3) {
      // Interior wants every corner interior; exhausted wants every corner exhausted or interior.
      if (!(bKind[k] === 1 ? kind === 1 : (kind === 3 || kind === 1))) bAgree[k] = 0;
      return;
    }
    if (bAccN[k] === 0) bAcc[k] = 0;
    bAcc[k] += planeFillOf(gfA[o], gfA[o + 1], kind, pool.M[pend[k] * 9 + 2] * 0.5);
    bAccN[k]++;
  };
  // In passes, because four corners a cell can be several times a batch and a query that does not
  // fit is a query that is silently never asked.
  for (let k0 = 0; k0 < n;) {
    let nc = 0, k = k0;
    for (; k < n && nc + 4 <= GF_CAP; k++) {
      const kind = bKind[k];
      const o9 = pend[k] * 9, cx = pool.M[o9], cz = pool.M[o9 + 1], r = pool.M[o9 + 2];
      const guard = kind === 1 || kind === 3;
      if (!guard && !planeNeedsAA(kind, bDe[k], r)) continue;
      // The guard samples the corners and the anti-aliasing samples the quarter points. A corner
      // is shared with the four neighbouring cells, right for a guard about the cell's own
      // boundary and wrong for an average: it puts a second low pass two cells wide over the
      // result, and the tangle's gradient energy fell to 0.026 against a supersampled truth's
      // 0.048. The quarter points carry half the rTol, so the derivative bailout resolves the sub
      // cell.
      const d = guard ? r : r * 0.5;
      for (let j = 0; j < 4; j++) {
        const ox = (j & 1) ? d : -d, oz = (j & 2) ? d : -d;
        if (kind === 1 && anal && inMainBody(cx + ox, cz + oz)) continue;
        const q = nc * 4;
        gfQ[q] = cx + ox - job.gcx; gfQ[q + 1] = cz + oz - job.gcz;
        gfQ[q + 2] = d; gfQ[q + 3] = 0;
        cornOwner[nc] = k; nc++;
      }
    }
    if (nc > 0) gfRun(nc, cap, cornTake);
    k0 = k;
  }
  for (let k = 0; k < n; k++) {
    const i = pend[k];
    planeFinish(i, bNu[k], bDe[k], bGnu[k], KIND_WHY[bKind[k]], !!bAgree[k]);
    if (bAccN[k] === 4) {
      // Centre a third, each quarter point a sixth. See planeNeedsAA for the weights and for why
      // it is the fill that is averaged rather than the shaded colour.
      const o9 = i * 9, r = pool.M[o9 + 2];
      const kbar = planeFillOf(bNu[k], bDe[k], bKind[k], r) / 3 + bAcc[k] / 6;
      planeSample(bNu[k], bDe[k], bKind[k], r, FILT_MAX, kbar);
      pool.M[o9 + 6] = sc[0]; pool.M[o9 + 7] = sc[1]; pool.M[o9 + 8] = sc[2];
    }
  }
}

function planeFlushCPU(n) {
  for (let k = 0; k < n; k++) {
    const i = pend[k], o9 = i * 9;
    planeCellField(i, pool.M[o9], pool.M[o9 + 1], pool.M[o9 + 2]);
  }
}

function stepPlane(budgetMs) {
  const t0 = performance.now();
  const tEnd = t0 + budgetMs;
  let chunk = PCHUNK;
  const cap = job.cap, splitPx = job.splitPx, tol = job.spanTol;
  let np = job.np, nEmit = job.nEmit, cutSize = job.cutSize;
  let dmin = job.dmin, dmax = job.dmax, visited = job.visited, culled = job.culled;
  let bLow = job.bLow;
  const msCap = job.msCap, fadeK = job.fadeK, budK = job.budK, maxPx = job.maxPx;
  let over = job.over;
  let done = false;

  outer:
  for (;;) {
    if (job.nf === 0) { done = true; break; }
    if (!job.levelReady) { prepareLevel(); bLow = job.bLow; }
    let q = job.q, nn = job.nn;
    while (q < job.nf) {
      if (--chunk <= 0) {
        chunk = PCHUNK;
        const now = performance.now();
        // The latency ceiling. Past it nothing else splits and the walk drains the frontier to
        // leaves, which finishes the cut in one more pass over what is already there.
        if (!over && job.ms + (now - t0) > msCap) over = true;
        if (now >= tEnd) { job.q = q; job.nn = nn; job.over = over; break outer; }
      }
      visited++;
      const i = frontier[q++];
      // Split on size and on ramp span, and both have to want it: out in the smooth exterior the
      // field is nearly linear and one large splat is already the exact answer, so only the size
      // cap keeps the quadtree from banding. Past the latency ceiling only the size cap still
      // fires, because suppressing refinement outright leaves holes in a breadth first walk: a
      // cut at 10^13 that ran out of clock covered 27 percent of the frame against a stale cut's
      // 100. At 1/48 of the frame height that clause asks for about 2 600 cells, which the drain
      // can honour; at an absolute 6 px cap it asked for 100 000.
      const o9i = i * 9;
      // In the Gaussian view the criterion is size and nothing else, so this object shows the
      // same lattice of primitives an IFS object does. See GAUSS_PX_PLANE.
      const want = pool.fp[i] > maxPx ||
        (!over && (pool.cc[i] > tol || planeTangle(pool.M[o9i + 5], pool.M[o9i + 4], pool.M[o9i + 2])));
      let split = want && pool.fp[i] > splitPx && pool.depth[i] < 200 && cutSize + 5 <= cap;
      let tbud = 1;
      if (split && bLow !== -Infinity) {
        tbud = (pool.prio[i] / bLow - 1) * budK;
        if (!(tbud > 0)) { split = false; tbud = 0; } else if (tbud > 1) tbud = 1;
      }
      let made = 0, full = false, fadeTf = 1, fadeBl = 1;
      if (split) {
        const o9 = i * 9;
        const cx = pool.M[o9], cz = pool.M[o9 + 1], r = pool.M[o9 + 2] * 0.5;
        const d = pool.depth[i] + 1;
        // A cell splits into four across a dissolve, as an IFS node splits into nmaps across one.
        // The colour is a density weighted mean, so drawing the parent at 1-t alongside four
        // children at t interpolates between the two, and opacity is conserved because the four
        // children tile the parent.
        let tf = (pool.fp[i] / splitPx - 1) * fadeK;
        if (!(tf > 0)) tf = 0; else if (tf > 1) tf = 1;
        if (tbud < tf) tf = tbud;
        fadeBl = pool.bl[i]; fadeTf = tf;
        cellBl = fadeBl * tf;
        for (let k = 0; k < 4; k++) {
          if (np >= MAXCAP) { full = true; break; }
          const ox = (k & 1) ? r : -r, oz = (k & 2) ? r : -r;
          // Geometry now, field later: the child is placed and culled here, which needs nothing
          // but its position, and its orbit is deferred to the level's batch. See planeFlush.
          if (planeCellGeom(np, cx + ox, cz + oz, r, d)) {
            pendPush(np); nextF[nn++] = np; np++; made++;
          } else culled++;
        }
        cellBl = 1;
        // How long a child may wait for its field. The ceiling and the frame slice are checked
        // every PCHUNK frontier cells against the wall clock, so deferred work is work the
        // ceiling cannot see. A GPU level costs about five milliseconds, inside the noise;
        // deferring a level on the CPU path let cuts overshoot, and the descent's own wall test
        // came back with 8 to 19 dark frames on the climb.
        if (pendN >= (planeGpuField() ? GF_CAP : 1)) planeFlush();
      }
      let emitI = -1, emitBl = 0;
      if (made > 0) {
        cutSize += made - 1;
        if (fadeTf < 1) { emitI = i; emitBl = fadeBl * (1 - fadeTf); cutSize++; }
      } else if (split && !full) { cutSize--; culled++; }
      else { emitI = i; emitBl = pool.bl[i]; }
      if (emitI >= 0 && nEmit < cap) {
        emitPlane(emitI, nEmit++, emitBl);
        if (pool.depth[emitI] < dmin) dmin = pool.depth[emitI];
        if (pool.depth[emitI] > dmax) dmax = pool.depth[emitI];
      }
    }
    // Every child of this level gets its field here, in one batch, before the level becomes the
    // frontier: the next level's split decisions read exactly these answers.
    planeFlush();
    const t = frontier; frontier = nextF; nextF = t;
    job.nf = nn; job.levelReady = false; job.q = 0; job.nn = 0;
  }

  job.np = np; job.nEmit = nEmit; job.cutSize = cutSize;
  job.dmin = dmin; job.dmax = dmax; job.visited = visited; job.culled = culled;
  job.over = over;
  job.ms += performance.now() - t0;
  if (done) job.active = false;
  return done;
}

// Colour one cell. Two terms, both averages over the cell rather than samples at its centre: the
// ramp, box filtered over the cell's own span exactly, and how much of the cell is set rather
// than exterior, from the distance estimate. The second term replaces a fade toward the ramp's
// mean, which is the right average of term one but ignores that the cell is largely filled by the
// set, and the set is not a mid tone: averaging only the exterior painted the near boundary
// region one flat mid grey.
//
// The set's share of a cell known to straddle the boundary and nothing more. Half is the unbiased
// guess for a box cut by a curve, nudged up because the escaped side of such a cell is where the
// ramp oscillates fastest, which the box filter has already averaged to the ramp mean.
const TANGLE_SET = 0.45;
// The same guess for a cell whose orbit ran out of iterations. Higher, because not escaping in
// maxIter steps is stronger evidence of being in the set than the derivative test's "the boundary
// is somewhere in this cell". Not 1: calling an exhausted cell interior outright fills a deep
// frame with flat colour.
const DEEP_SET = 0.60;
// How much of one turn of the ramp the box filter may average over. The exact antialiased colour
// of a cell is the average of the ramp over its own span, but near the boundary |grad nu|
// diverges, so the span passes one turn at any affordable cell size and the exact answer is then
// the ramp's mean, a neutral grey: against a supersampled reference the near boundary region
// carried an eighth of the truth's gradient energy and the halo read as fog. Capped, the cell
// keeps the hue at its own centre for the rest. That point sampling is safe here because the cell
// grid is a fixed set of binary halvings of the plane, so the colour is a property of the place
// rather than of the camera, and a splitting cell is cross faded onto its children.
const FILT_MAX = 0.30;
// The kernel bandwidth is a per cell decision. A splat cut is a kernel regression of the field,
// the tone map dividing accumulated colour by accumulated weight, so the bandwidth of the
// regression is the splat's own sigma. A wide kernel interpolates the smooth exterior between
// sample points; the same kernel on the boundary is blur, since the field is not smooth at any
// scale the budget reaches and a 1 px cell with sigma 1 px smears its colour over nine pixels. So
// kappa runs from 1 at spanTol down to KERN_MIN over KERN_OCT octaves of span, smoothstepped.
// Proven interior takes KERN_MIN outright: it is one flat colour inside, and at sigma equal to a
// maxPx half size its tail carried the set's colour thirty pixels out over the boundary.
// Narrowing stops the profiles summing flat, by 2 exp(-4.93 kappa^2) per axis, 1.4 percent at
// kappa 1 and 74 at 0.45; the ripple cancels in the colour, a ratio carrying it in both halves,
// and survives in the opacity, which is why kernPeak exists. Against a three by three
// supersampled reference at 1377x1592, gradient energy per pixel is 0.203 of the reference at
// kappa 1 and 0.413 at 0.45.
let KERN_MIN = 0.24;
const KERN_OCT = 3.5;
// How many standard deviations of quad a flat splat gets, as a multiple of kern.extent. It exists
// to let the bandwidth go under what the support would otherwise allow: the fragment shader
// discards past exp(-6.91), so once kappa drops below sqrt(2)/extent, 0.447 at beta 3, a cell
// corner lies outside every neighbouring profile and is empty rather than dim. Measured at 10^3
// against a per pixel render, an ideal constant per cell reconstruction at the cut's own 1.4 px
// cell size scores rms 0.040 and the shipped render 0.069. A larger quad costs fill rate only,
// one to two percent of a plane frame.
let FLAT_EXT = 2.4;
// The peak optical depth that goes with a bandwidth, and it is not 1/kappa^2. Cells of half size
// r sit on a lattice of pitch 2r, so the sum of their profiles is periodic: for a Gaussian of
// standard deviation kappa*r the mean of the sum is 2 pi kappa^2 / 4 per axis pair and the
// relative ripple is 2 exp(-2 pi^2 kappa^2) per axis, so the worst point carries
//
//     mean * (1 - 2 exp(-4.9348 kappa^2))^2
//
// of the nominal weight. Scaling by 1/kappa^2 holds the mean and lets the worst point fall, and
// the worst point is where the background shows through as a dot screen: under that law the
// darkest pixel of a flat patch whose true value is 0.0510 read 0.0471 at kappa 0.55 and 0.0157
// at 0.35. T_MIN = 8 because the opacity is 1 - exp(-T), leaving the background three parts in
// ten thousand there.
const T_MIN = 8;
// The Fourier argument above is for a Gaussian and the profile is a super Gaussian at beta =
// cfg.kernel, flat topped with a shorter shoulder, so the closed form left the lattice corner at
// 1.6 percent of its true brightness at kappa 0.35. kernTable sums the actual profile over the
// actual lattice at its worst point instead, the cell corner for a square lattice of pitch 2r, in
// 64 entries over kappa. The peak is T_MIN divided by that sum. uFilter's 0.5 px of prefilter is
// left out, which errs high, and excess optical depth is free because the colour is a weighted
// mean.
const KP_N = 64, KP_LO = 0.2, KP_HI = 1.0;
let kpTab = null, kpFor = -1;
function kernTable(beta, shape) {
  const t = new Float64Array(KP_N + 1);
  const b2 = beta * 0.5;
  for (let k = 0; k <= KP_N; k++) {
    const kap = KP_LO + (KP_HI - KP_LO) * k / KP_N;
    let s = 0;
    for (let i = -6; i <= 6; i++) for (let j = -6; j <= 6; j++) {
      // The corner sits at (r, r) from the centre at the origin, so a centre at (2i, 2j) r is
      // (2i - 1, 2j - 1) r away, in units of r.
      const dx = 2 * i - 1, dy = 2 * j - 1;
      const q = (dx * dx + dy * dy) / (kap * kap);
      const e = 0.5 * Math.pow(shape * q, b2);
      if (e < 6.91) s += Math.exp(-e);          // the shader discards past this
    }
    t[k] = s;
  }
  return t;
}
function kernPeak(kap) {
  if (kpFor !== kern.beta) { kpTab = kernTable(kern.beta, kern.shape); kpFor = kern.beta; }
  let u = (kap - KP_LO) / (KP_HI - KP_LO) * KP_N;
  if (u < 0) u = 0; else if (u > KP_N - 1e-6) u = KP_N - 1e-6;
  const j = u | 0, f = u - j;
  const s = kpTab[j] + (kpTab[j + 1] - kpTab[j]) * f;
  const w = T_MIN / (s > 1e-6 ? s : 1e-6);
  // Bounded above so the accumulation stays in half float range: 2000 with a few profiles
  // overlapping is 1e4 against the format's 65504, and the colour is a ratio so the magnitude
  // carries nothing.
  return w < T_MIN ? T_MIN : (w > 2000 ? 2000 : w);
}
// One sample of the colour law, in linear light, into `sc`. Lifted out of emitPlane so that a
// cell can be shaded from several samples of the field rather than from one.
const sc = [0, 0, 0];
// How much of a sample the set fills, on its own, in [0,1], smoothstepped. Lifted out of
// planeSample so the anti-aliasing can average this and nothing else; see planeNeedsAA. The three
// verdicts that are not "escaped" have no distance to work with: proven interior is all set,
// condemned is TANGLE_SET, out of iterations is DEEP_SET.
function planeFillOf(nu, de, kind, r) {
  if (kind === 1) return 1;
  if (kind === 3) return DEEP_SET;
  let k = kind === 0 ? 1 - de / (RT2 * r) : TANGLE_SET;
  if (!(k > 0)) return 0;
  if (k > 1) k = 1;
  return k * k * (3 - 2 * k);
}
// `fill` overrides the set share when it is >= 0, and is then taken as already smoothstepped.
function planeSample(nu, de, kind, r, span, fill) {
  let cr, cg, cb;
  if (kind === 1) {
    // Proven interior, by a cycle or by the analytic body test. Deliberately not black: a view
    // dominated by the set is a legitimate place to be and should look like somewhere.
    cr = lakeCol[0]; cg = lakeCol[1]; cb = lakeCol[2];
  } else if (kind === 3) {
    // Out of iterations, and the colour must not depend on the iteration cap. nu = maxIter is
    // consistent within one cut, but the cap grows in quantized steps of 1.3 and the ramp
    // coordinate is 3 cbrt(nu) / nuCycle, so one step moves it 1.35 turns: cap 1864 gives hue 277
    // degrees, 2423 gives 42, 3150 gives 213, over the 63 percent of the frame that region covers
    // at 10^6. The jump is not between cells, so no cross fade can help; the colour is fixed at
    // the ramp's mean pulled toward the set by DEEP_SET.
    cr = palMean[0] + (lakeCol[0] - palMean[0]) * DEEP_SET;
    cg = palMean[1] + (lakeCol[1] - palMean[1]) * DEEP_SET;
    cb = palMean[2] + (lakeCol[2] - palMean[2]) * DEEP_SET;
  } else {
    if (span > FILT_MAX) span = FILT_MAX;
    const c = rampBox(3 * Math.cbrt(nu > NU_MIN ? nu : NU_MIN) * job.nuInv, span);
    cr = c[0]; cg = c[1]; cb = c[2];
    // How much of the cell the set fills. Escaped, kind 0: `de` is a lower bound on the distance
    // from the centre to the set, so a cell with de >= r sqrt(2) contains none of it and is
    // untouched. That is the same threshold that forces a split, so this term only fires on a
    // cell the budget or the ceiling refused to refine. Condemned, kind 2: there is no de, only
    // the fact that the boundary crosses the cell, so TANGLE_SET is the guess. Painting such a
    // cell the set's colour outright measured 72.0 percent set against a three by three
    // supersampled truth's 60.2.
    let k = fill >= 0 ? fill : planeFillOf(nu, de, kind, r);
    if (k > 0) {
      cr += (lakeCol[0] - cr) * k;
      cg += (lakeCol[1] - cg) * k;
      cb += (lakeCol[2] - cb) * k;
    }
    // Relief, and only relief. de/r is the distance to the set in units of the cell's own size,
    // so it is scale relative by construction and its variation is the same at every depth; six
    // octaves of it, centred on one so it cannot darken the frame overall. No brightness term can
    // manufacture hue the field does not have: at 10^12 the tenth to ninetieth percentile of nu
    // spans under 0.4 turns of the ramp under any function of nu.
    const rel = de > 0 ? clamp01(Math.log2(de / r) * (1 / 6)) : 0;
    const sh = 0.88 + 0.22 * rel;
    cr *= sh; cg *= sh; cb *= sh;
  }
  sc[0] = cr; sc[1] = cg; sc[2] = cb;
}

// Anti-aliasing for the splat path, the same idea as shadePixel in src/62_direct.js. A resolved
// cell is one sample's worth of information; a tangle cell's centre sample is one draw from a
// distribution, and at 10^2 on a 1100x1280 frame adjacent 1.4 px cells there differ by up to the
// full range of the palette, which reads as coloured static. So a tangle cell is answered from
// five samples, its centre at weight 1/3 and its four quarter points at 1/6 each. What is
// averaged is the set fill and not the colour, because a low pass over a cyclic ramp walks toward
// its grey mean: with the colours averaged the four pixel band around the set came back at 0.83
// of a per pixel reference's saturation against 0.94 without, and the frame's gradient energy
// fell from 0.221 of the reference to 0.108. The escape count needs no samples, being already box
// filtered over the exact span the cell covers. Four extra orbits a tangle cell is one and a half
// to two times the cut's field cost, so this rides with planeGpuField and turns off with it;
// `sa=0` disables it. Proven interior and out of iterations are excluded, both being one flat
// colour.
let SPLAT_AA = 1;
function planeNeedsAA(kind, de, r) {
  return SPLAT_AA > 0 && kind !== 1 && kind !== 3 && planeTangle(kind, de, r);
}

function emitPlane(i, slot, bl) {
  const o9 = i * 9, o3 = i * 3;
  const r = pool.M[o9 + 2], de = pool.M[o9 + 4], nu = pool.M[o9 + 3], kind = pool.M[o9 + 5];
  let cr, cg, cb;
  // A cell shaded from several samples carries the answer already; see planeFlush. M[6] is set to
  // -1 by planeCellGeom for every other cell, and no colour channel is ever negative.
  if (pool.M[o9 + 6] >= 0) {
    cr = pool.M[o9 + 6]; cg = pool.M[o9 + 7]; cb = pool.M[o9 + 8];
  } else {
    planeSample(nu, de, kind, r, pool.cc[i], -1);
    cr = sc[0]; cg = sc[1]; cb = sc[2];
  }
  // The kernel bandwidth for this cell, as a fraction of its half size. See KERN_MIN.
  let kap;
  // An escaped cell clear of the set interpolates the field, and how wide a kernel it wants
  // follows from how badly it resolves the ramp; everything else is boundary layer and takes the
  // narrow one. The test is planeTangle rather than `kind === 0`, because an escaped cell with
  // the set running through it has a small span and would otherwise get the wide kernel.
  if (!planeTangle(kind, de, r)) {
    const sp = pool.cc[i];
    if (!(sp > job.spanTol)) kap = 1;
    else {
      let u = Math.log2(sp / job.spanTol) * (1 / KERN_OCT);
      if (u > 1) u = 1;
      u = u * u * (3 - 2 * u);
      kap = 1 + (KERN_MIN - 1) * u;
    }
  } else {
    // Interior, condemned and exhausted are all boundary layer: nothing to interpolate.
    kap = KERN_MIN;
  }
  // The bandwidth cannot go below what the kernel's own support can cover. The fragment shader
  // discards past exp(-6.91) and the quad is kern.extent standard deviations across, so a cell
  // corner, sqrt(2) half sizes from the centre, falls outside every neighbouring profile once
  // kappa drops under sqrt(2)/extent, 0.426 at beta 3: there it is empty rather than dim and no
  // peak can fill it. Measured, kappa 0.40 left the darkest pixel of a flat region at 0.0157 of
  // full range against a true 0.0510 whatever the peak was. Derived rather than written down,
  // because it moves with the kernel exponent. The shader's 0.5 px prefilter counts toward it in
  // quadrature, so what has to reach the corner is
  //
  //     extent^2 ((kappa r)^2 + 1/4)  >=  (1.05 sqrt2 r)^2,   r in pixels
  //
  // which any kappa satisfies for a cell near the split threshold and which is essentially the
  // old bound for a large one.
  const rpx = 0.5 * pool.fp[i];
  const k0 = 1.05 * Math.SQRT2 / (kern.extent * FLAT_EXT);
  const kFloor = rpx > 1e-6 ? Math.sqrt(Math.max(0, k0 * k0 - 0.25 / (rpx * rpx))) : k0;
  if (kap < kFloor) kap = kFloor;
  // The Gaussian view wants one bandwidth and the round number, a splat at one standard deviation
  // to its own half size having its one sigma contour inscribed in its cell. The per cell
  // narrowing would only make the primitives harder to see.
  if (cfg.gauss) kap = 1;
  // The compact flat instance: seven floats rather than thirteen, and see FLAT_FLOATS in
  // src/50_cut.js for why the other six carry nothing for a plane object. The vertex shader
  // reconstructs the y from a uniform and the whole covariance from `rad`, the cell's half width
  // in the cut's normalized frame, because pool.cov holds diag(r^2/3, (0.06 r)^2, r^2/3) and
  // every entry is a function of r alone. Scaling `rad` by kappa is therefore the whole of the
  // variable bandwidth, and `place` has already decided the split from the cell's true size.
  const o = slot * FLAT_FLOATS;
  emitted[slot] = i;
  instances[o] = pool.rel[o3]; instances[o + 1] = pool.rel[o3 + 2];
  instances[o + 2] = r * job.invD * kap;
  instances[o + 3] = cr; instances[o + 4] = cg; instances[o + 5] = cb;
  instances[o + 6] = kernPeak(kap) * bl;
  job.sumW += 1; job.sumWvis += 1;
}

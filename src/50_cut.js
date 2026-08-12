/* ========================= refinement: the cut ========================== */
// Resumable refinement. A rebuild is a job spread over several frames against a per frame work
// budget, so the frame rate belongs to the renderer and not to the tree walk. The last finished
// cut keeps being drawn meanwhile, corrected exactly for the camera drift since, so staleness
// costs accuracy in the choice of splats and never in where they are drawn.
// Order: startJob, then stepJob until it returns true, then finishJob, which publishes `built`.

const FLOATS = 13;                    // pos3 cov6 col3 w1
// A plane cut needs seven floats, not thirteen: every cell sits at y = 0 and is a disc, so pos.y is
// the shared job.planeY and cov is diag(r^2/3, (0.06 r)^2, r^2/3), a function of r alone, rebuilt
// in the vertex shader; the 0.06 must match 52_cut_plane.js:202 and 60_gl.js:186. 28 bytes an
// instance against 52. Colour, weight and position all stay fp32: JS has no cheap half conversion,
// two million per cut, and a half's three decimal digits cannot place a splat inside a pixel.
const FLAT_FLOATS = 7;                // pos2 rad1 col3 w1
let instances = new Float32Array(65536 * FLOATS);
// Floats per splat in the current cut, in one place because the emit kernels, the allocation and
// the upload must agree. Reads the cut on screen, so it is valid only after finishJob.
function cutFloats() { return built.flat ? FLAT_FLOATS : FLOATS; }
// The cut on screen, in the frame it was built in. `Rq` and `sSince` relate that frame to the
// camera's current one: a rebase re-expresses the scene in a child piece's coordinates and an
// older cut knows nothing about it. Without them the rebase frame drew the old cut at the wrong
// scale and exposure, a 63 percent frame flash every 0.35 decades of descent. See `reprojection`.
const built = {
  pos: new Float64Array(3), R: new Float64Array(9), dist: 1,
  Rq: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
  sSince: 1, norm: 1,
  valid: false, count: 0, surface: false, prepass: false,
  // A flat cut: seven floats an instance, its own vertex shader, `planeY` the y they all share.
  flat: false, planeY: 0,
};
const HB = 192;                       // histogram buckets, 4 per octave
const hist = new Int32Array(HB);
const hHist = new Int32Array(256);      // log2 histogram of terrain height
const LX = -0.4243, LY = 0.7071, LZ = 0.5657;    // terrain light, world fixed

const job = {
  active: false, terrain: false, vw: 0, vh: 0, ms: 0,
  R: new Float64Array(9), pos: new Float64Array(3), dist: 1, invD: 1, focal: 1,
  tanX: 0, tanY: 0, nrmX: 0, nrmY: 0, K: 4, zFar: 15, fogK: 0.5, f32: false,
  cap: 0, nmaps: 0, splitPx: 2, sig: 1, norm: 1, attFloor: 0,
  huBase: 0, fadeK: 0, budK: 0,
  nf: 0, nn: 0, q: 0, cutSize: 0, np: 0, nEmit: 0,
  dmin: 0, dmax: 0, visited: 0, culled: 0, sumW: 0, sumWvis: 0,
  bLow: -Infinity, bAllow: 0, levelReady: false, usePrio: false, bFloor: 0, bRaw: 0,
  // The frame record handed to `built` at finish. Transported by every rebase that lands mid job,
  // unlike the job's own camera fields, because the kernel must project every node with one camera
  // or the cut is internally inconsistent. See `rebaseRecord`.
  out: {
    pos: new Float64Array(3), R: new Float64Array(9), dist: 1,
    Rq: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), sSince: 1,
  },
  // terrain kernel:
  maxIter: 400, relief: 1, hBase: 0, hSpan: 1, contour: 0, wSurf: 6, hmode: 0, hc: 1,
  eq: false, hLo: 0, hInv: 1, fcOn: false,
  // plane kernel:
  plane: false, julia: 0, power: 2, c0x: 0, c0y: 0, nuCyc: 24, nuInv: 1, spanTol: 0.18,
  flat: false, planeY: 0,     // see FLAT_FLOATS: a plane cut's compact instance format
  maxPx: 24,        // plane: the coverage cap, a fraction of the frame height, see planeRoot
  pfcOn: false,     // plane: is the field cache worth its own cost at this depth
  // Ceiling on the CPU one cut may spend, in ms: a latency budget, distinct from the splat budget.
  // Past it the walk stops splitting and drains to leaves, so the cut arrives coarser but on time.
  // The slowest five percent of Mandelbrot cuts took 104 frames, 1.4 e-folds of camera motion.
  msCap: 1e9, over: false,
  hCount: 0,
  evals: 0, iters: 0,
};

function startJob(vw, vh) {
  updateBasis();
  const terrain = isTerrain(), plane = isPlane();
  job.active = true; job.terrain = terrain; job.plane = plane; job.ms = 0;
  job.flat = plane;
  // How much of one turn of the colour ramp a plane cell may span before it is worth splitting.
  // Under this, splitting cannot change a pixel's colour.
  job.spanTol = 0.05;
  job.msCap = 1e9;
  job.wSurf = 6.0;
  job.vw = vw; job.vh = vh;
  job.R.set(basis.R);
  // Build for where the camera will be, `buildLead` seconds ahead, since a cut takes that long and
  // the camera does not wait. Downward the drift is 0.08 e-folds and invisible; outward a stale cut
  // under covers the frame: at 2.5 times the descent rate the frame grew 7x during one build, the
  // cut shrank to two percent of the frame area and 126 of 900 frames came back dark. job.out holds
  // the prediction, so `reprojection` corrects to the true camera and only refinement looks ahead.
  const dP = cam.dist * Math.exp(zoomRate * buildLead);
  const kP = dP / cam.dist;
  job.dist = dP; job.invD = 1 / dP;
  job.pos[0] = cam.target[0] + (basis.pos[0] - cam.target[0]) * kP;
  job.pos[1] = cam.target[1] + (basis.pos[1] - cam.target[1]) * kP;
  job.pos[2] = cam.target[2] + (basis.pos[2] - cam.target[2]) * kP;
  job.focal = 0.5 * vh / Math.tan(0.5 * cam.fov);
  lastFocal = job.focal;
  // Cull against a frustum 1.12 times wider than the drawn one, so a cut that is a few frames old
  // still covers the edges of the frame.
  job.tanY = Math.tan(0.5 * cam.fov) * 1.12;
  job.tanX = job.tanY * vw / vh;
  job.nrmX = 1 / Math.sqrt(1 + job.tanX * job.tanX);
  job.nrmY = 1 / Math.sqrt(1 + job.tanY * job.tanY);
  job.K = suppR;
  // Depth of field in units of camera distance: past it the shader's fog has attenuated a splat
  // below a thousandth, so refining is wasted. The 8 is the ATT table's argument range in
  // src/40_state.js. Without it the coarse structure around the camera eats the whole budget.
  job.zFar = 1 + ZF_MUL * 8 / Math.max(cfg.fog, 0.02);
  job.fogK = cfg.fog;            // the shader's own attenuation rate, see `place`
  job.f32 = cfg.precision === 'f32';
  // The tier's share of the preset's budget: the preset says what the object wants, perfBudget what
  // the machine holds, and separating them lets the slider keep reading the object's own number.
  job.cap = Math.min(perfBudget(), MAXBUDGET);
  // Halve the split threshold over the stillness ramp, a factor of four in cell count. A finer
  // threshold is a longer rebuild, so during a descent the cut on screen is always the stale one;
  // stillCuts is zero the moment anything moves. Plane objects ramp their latency ceiling instead.
  const stillPx = (isPlane() || stillCuts <= 0) ? 1
    : 1 - 0.5 * Math.min(1, stillCuts / 4);
  // The Gaussian view coarsens outright and overrides the sharpening; the plane objects get their
  // own absolute size, and their split test becomes a size test. See GAUSS_PX and GAUSS_PX_PLANE.
  job.splitPx = cfg.gauss ? (plane ? GAUSS_PX_PLANE : GAUSS_PX)
    : cfg.splitPx * stillPx;
  job.sig = cfg.sigma;
  // No splat may stay large on screen however little light reaches it: splitting on visible error
  // alone left eleven splats owning 44 percent of the drawn screen area on a descended folded
  // dragon while carrying 0.1 percent of the light, the largest 167 px across. The floor
  // splitPx/SIZE_CAP makes priority pass splitPx exactly when fp passes SIZE_CAP, so an oversized
  // node always splits, the order among visible nodes is unchanged, and it costs about
  // SIZE_CAP^2/splitPx^2 extra splats once.
  job.attFloor = job.splitPx / (cfg.gauss ? GAUSS_CAP : SIZE_CAP);
  // The cut wide priority floor, keyed on everything that changes what a full budget means, so a
  // new object, window or tier restarts the controller rather than inheriting a stale number.
  bFloorReset(cfg.preset + '|' + job.cap + '|' + vw + 'x' + vh + '|' + (cfg.gauss ? 'g' : '-'));
  job.bFloor = bFloor; job.bRaw = 0;
  // The exposure baked into the weights. A branch weight at depth 45 of an eight map IFS is 8^-45,
  // under the smallest normal Float32, so the buffer cannot carry a raw weight and a separate gain;
  // `reprojection` then corrects only the ratio to the current camera, which is near one.
  job.norm = measureNormAt(dP);
  job.nf = 0; job.nn = 0; job.q = 0; job.cutSize = 1; job.np = 1; job.nEmit = 0;
  job.dmin = 1 << 30; job.dmax = 0; job.visited = 0; job.culled = 0;
  job.sumW = 0; job.sumWvis = 0; job.evals = 0; job.iters = 0;
  job.hCount = 0; job.over = false;
  // Every plane splat is at y = 0, so this is the y they all share once the position is camera
  // relative and scaled. Computed here once instead of stored 500 000 times.
  job.planeY = (0 - job.pos[1]) * job.invD;
  job.out.pos.set(job.pos); job.out.R.set(basis.R); job.out.dist = dP;
  job.out.Rq.set([1, 0, 0, 0, 1, 0, 0, 0, 1]); job.out.sSince = 1;
  if (terrain) hHist.fill(0);
  job.levelReady = false;
  if (instances.length < job.cap * FLOATS) instances = new Float32Array(job.cap * FLOATS);
  if (plane) planeRoot(vw, vh); else if (terrain) terrRoot(vw, vh); else ifsRoot(vw, vh);
}

/* --------------------------- shared projection --------------------------- */
// Project a node centre that is already camera relative and scaled, cull it against the frustum and
// the depth of field, and write pool.sx, pool.sy, pool.fp, pool.prio and pool.rel[3n..3n+2]. False
// when the node is out of frame. The IFS kernel inlines its own copy in the inner loop.
function place(n, dx, dy, dz, rad) {
  const R = job.R;
  const ex = R[0] * dx + R[1] * dy + R[2] * dz;
  const ey = R[3] * dx + R[4] * dy + R[5] * dz;
  const ez = -(R[6] * dx + R[7] * dy + R[8] * dz);
  const rr = job.K * rad;
  if (ez + rr < 1e-4 || ez - rr > job.zFar ||
      (Math.abs(ex) - ez * job.tanX) * job.nrmX > rr ||
      (Math.abs(ey) - ez * job.tanY) * job.nrmY > rr) return false;
  const z = ez > 1e-4 ? ez : 1e-4;
  const iz = 1 / z;                     // one reciprocal, not three divisions
  pool.sx[n] = job.vw * 0.5 + job.focal * ex * iz;
  pool.sy[n] = job.vh * 0.5 - job.focal * ey * iz;
  const fp = job.focal * rad * iz;
  pool.fp[n] = fp;
  // Visible error, footprint times the light that reaches the eye, evaluated at the nearest point
  // of the node's support, ez - rr, not at its centre: a walk root's centre can sit 60 camera
  // distances away while its near face is at the camera, and attenuating by the centre made the
  // root fail its own split test, collapsing the tree to one splat and 96 black frames in 480. The
  // floor is the screen size cap, so the test fires on size alone past sizeCap. See `job.attFloor`.
  const zn = ez - rr;
  const att = attOf(job.fogK * (zn - 1));
  pool.prio[n] = fp * (att > job.attFloor ? att : job.attFloor);
  const o3 = n * 3;
  pool.rel[o3] = dx; pool.rel[o3 + 1] = dy; pool.rel[o3 + 2] = dz;
  return true;
}

function ifsRoot(vw, vh) {
  const P = PRESETS[cfg.preset];
  job.nmaps = maps.length;
  // The hue the anchor's swallowed letters are worth: everything below this walk root sums its own
  // local hue from zero, so the drawn hue is frac(huBase + local). See `anchor.hue` and buildMapHue
  // in src/40_state.js. Captured here so a rebase during a rebuild cannot move the colours of the
  // cut being built. HUE_OFF is zero in every shipped frame.
  job.huBase = anchor.hue + HUE_OFF - Math.floor(anchor.hue + HUE_OFF);
  // Width of the cross fade band, as a ratio of priority above the split threshold.
  job.fadeK = 1 / (FADE_BAND - 1);
  job.budK = 1 / (BUD_BAND - 1);
  job.usePrio = false;
  const PM = pool.M, Pmu = pool.mu, Pcov = pool.cov;
  const rmu = root.mu, rcov = root.cov;
  PM[0] = 1; PM[1] = 0; PM[2] = 0; PM[3] = 0; PM[4] = 1; PM[5] = 0; PM[6] = 0; PM[7] = 0; PM[8] = 1;
  Pmu[0] = rmu[0]; Pmu[1] = rmu[1]; Pmu[2] = rmu[2];
  for (let i = 0; i < 6; i++) Pcov[i] = rcov[i];
  pool.w[0] = 1; pool.depth[0] = 0; pool.cc[0] = 0;
  pool.ls[0] = 0; pool.bl[0] = 1;
  pool.par[0] = -1; pool.mi[0] = -1;
  let mx = rmu[0], my = rmu[1], mz = rmu[2];
  if (job.f32) { mx = Math.fround(mx); my = Math.fround(my); mz = Math.fround(mz); }
  const rad = Math.sqrt(rcov[0] + rcov[3] + rcov[5]) * job.invD * job.sig;
  if (!place(0, (mx - job.pos[0]) * job.invD, (my - job.pos[1]) * job.invD,
             (mz - job.pos[2]) * job.invD, rad)) return;
  frontier[0] = 0;
  job.nf = 1;
}

/* ------------------------------- terrain -------------------------------- */
function terrRoot(vw, vh) {
  const P = PRESETS[cfg.preset];
  job.nmaps = 4;
  job.usePrio = true;
  job.relief = cfg.relief;
  job.hmode = cfg.height;
  job.contour = cfg.contour;
  // Peak optical depth of one surfel, high enough that the absorption model saturates across the
  // footprint; lower and the kernel's own falloff shows at the cell spacing as a dot screen.
  job.wSurf = 6.0;
  job.hc = PRESETS[cfg.preset].radius * 0.9;
  // The escape count of a point at distance d from the set grows like log(1/d), so the iteration
  // cap grows logarithmically with depth, which is the only reason a deep zoom is affordable.
  const lz = Math.max(0, logZoom());
  // Quantized to steps of 1.3 so the field cache survives a descent instead of being invalidated
  // every frame by a cap that creeps. See mandelFieldCached.
  const want = Math.max(80, Math.min(60000, cfg.iters * (1 + 0.42 * lz)));
  const mstep = Math.max(0, Math.round(Math.log(want / 80) / Math.log(1.3)));
  job.maxIter = Math.round(Math.min(60000, 80 * Math.pow(1.3, mstep)));
  stats.maxIter = job.maxIter;
  // The hypsometric span, a fallback: the first rebuild of an object, before the height histogram
  // exists, and the potential mode, which has its own analytic span. The distance mode equalizes in
  // log height instead, see terrLo in src/40_state.js. Relief scales with camera distance once the
  // camera is among it and is bounded by the measured height when the object is in frame.
  if (job.hmode === 0) {
    job.hBase = 0;
    job.hSpan = Math.min(terrHMax,
      (terrSpanRel > 0 ? terrSpanRel : 1.4 * cfg.relief) * job.dist);
  }
  else {
    const nuT = 6 + 3.3219 * lz;
    job.hBase = 0.11 * cfg.relief * nuT;
    job.hSpan = 0.11 * cfg.relief * 14;
  }
  if (job.hSpan <= 0) job.hSpan = 1e-30;
  job.eq = (job.hmode === 0) && terrEq;
  job.hLo = terrLo;
  job.hInv = 1 / Math.max(terrHi - terrLo, 1e-6);
  job.fcOn = terrItEma > FC_MIN_ITERS;
  fcCur = fcStamp(mstep, job.relief, job.hmode, job.hc);
  fcHits = 0; fcMiss = 0; fcSaved = 0;
  if (!terrCell(0, P.center[0], P.center[1], P.radius, 0)) return;
  frontier[0] = 0;
  job.nf = 1;
}

// Build one surfel from one field evaluation. The cell is a square of half size r in the c plane;
// the surfel is the moment matched Gaussian of the tangent patch over it, so covariance
// (1/3)(u u^T + v v^T) with u and v the tangent vectors of the height field across the cell, plus a
// thin term along the normal so the ellipsoid is not exactly degenerate.
function terrCell(n, cx, cz, r, depth) {
  if (job.fcOn) mandelFieldCached(cx, cz, job.maxIter, r * 0.5, job.relief, job.hmode, job.hc);
  else mandelField(cx, cz, job.maxIter, r * 0.5, job.relief, job.hmode, job.hc);
  job.evals++; job.iters += fld.iters;
  const h = fld.h, gx = fld.gx, gz = fld.gz;
  const nx = fld.nx, ny = fld.ny, nz = fld.nz;
  const r2 = r * r / 3, th = 0.2 * r, t2 = th * th;
  const o6 = n * 6, o9 = n * 9, o3 = n * 3;
  pool.cov[o6]     = r2 + t2 * nx * nx;
  pool.cov[o6 + 1] = r2 * gx + t2 * nx * ny;
  pool.cov[o6 + 2] = t2 * nx * nz;
  pool.cov[o6 + 3] = r2 * (gx * gx + gz * gz) + t2 * ny * ny;
  pool.cov[o6 + 4] = r2 * gz + t2 * ny * nz;
  pool.cov[o6 + 5] = r2 + t2 * nz * nz;
  pool.mu[o3] = cx; pool.mu[o3 + 1] = h; pool.mu[o3 + 2] = cz;
  pool.M[o9] = cx; pool.M[o9 + 1] = cz; pool.M[o9 + 2] = r;
  pool.M[o9 + 3] = fld.de; pool.M[o9 + 4] = fld.nu;
  pool.M[o9 + 5] = fld.inside ? 1 : (fld.edge ? 2 : 0);
  pool.M[o9 + 6] = nx; pool.M[o9 + 7] = ny; pool.M[o9 + 8] = nz;
  pool.w[n] = job.wSurf;
  pool.depth[n] = depth;
  let px = cx, py = h, pz = cz;
  if (job.f32) { px = Math.fround(px); py = Math.fround(py); pz = Math.fround(pz); }
  const rad = Math.sqrt(pool.cov[o6] + pool.cov[o6 + 3] + pool.cov[o6 + 5]) * job.invD * job.sig;
  if (!place(n, (px - job.pos[0]) * job.invD, (py - job.pos[1]) * job.invD,
             (pz - job.pos[2]) * job.invD, rad)) return false;
  // `place` has set prio to the visible error, which the split test wants, so keep a copy in
  // pool.cc: the unboosted priority for the terrain, the colour coordinate for the IFS kernel. The
  // boost raises cells close to the boundary relative to their own size, where the height field is
  // not yet resolved. It sets order under budget pressure only and must not enter the test: a boost
  // of four would push those cells five times past it and cost twenty five times the cells.
  pool.cc[n] = pool.prio[n];
  const de = fld.de;
  const boost = fld.inside ? 0 : Math.min(4, 0.5 * r / Math.max(de, r * 1e-6));
  pool.prio[n] *= 1 + boost;
  return true;
}

// Same time budget as the IFS kernel, with a smaller chunk between clock reads: a terrain cell is
// two orders of magnitude dearer than an IFS node.
const TCHUNK = 64;
function stepTerr(budgetMs) {
  const t0 = performance.now();
  const tEnd = t0 + budgetMs;
  let chunk = TCHUNK;
  const cap = job.cap, splitPx = job.splitPx;
  let np = job.np, nEmit = job.nEmit, cutSize = job.cutSize;
  let dmin = job.dmin, dmax = job.dmax, visited = job.visited, culled = job.culled;
  let bAllow = job.bAllow;
  let bLow = job.bLow, bHi = bLow * 1.19;
  let done = false;

  outer:
  for (;;) {
    if (job.nf === 0) { done = true; break; }
    if (!job.levelReady) {
      prepareLevel();
      bAllow = job.bAllow; bLow = job.bLow; bHi = bLow * 1.19;
    }
    let q = job.q, nn = job.nn;
    while (q < job.nf) {
      if (--chunk <= 0) {
        chunk = TCHUNK;
        if (performance.now() >= tEnd) { job.q = q; job.nn = nn; break outer; }
      }
      visited++;
      const i = frontier[q++];
      // Split on the visible error without the boundary boost, see terrCell. 200 is the depth
      // ceiling and 4 the child arity.
      let split = pool.cc[i] > splitPx && pool.depth[i] < 200 && cutSize + 4 <= cap;
      if (split && bLow !== -Infinity && pool.prio[i] < bLow) split = false;
      let made = 0, full = false;
      if (split) {
        const o9 = i * 9;
        const cx = pool.M[o9], cz = pool.M[o9 + 1], r = pool.M[o9 + 2] * 0.5;
        const d = pool.depth[i] + 1;
        for (let k = 0; k < 4; k++) {
          if (np >= MAXCAP) { full = true; break; }
          const ox = (k & 1) ? r : -r, oz = (k & 2) ? r : -r;
          // A culled child leaves its pool slot free for the next one.
          if (terrCell(np, cx + ox, cz + oz, r, d)) { nextF[nn++] = np; np++; made++; }
          else culled++;
        }
      }
      if (made > 0) cutSize += made - 1;
      else if (split && !full) { cutSize--; culled++; }
      else if (nEmit < cap) { emitTerr(i, nEmit++); if (pool.depth[i] < dmin) dmin = pool.depth[i]; if (pool.depth[i] > dmax) dmax = pool.depth[i]; }
    }
    const t = frontier; frontier = nextF; nextF = t;
    job.nf = nn; job.levelReady = false; job.q = 0; job.nn = 0;
  }

  job.np = np; job.nEmit = nEmit; job.cutSize = cutSize;
  job.dmin = dmin; job.dmax = dmax; job.visited = visited; job.culled = culled;
  job.bAllow = bAllow;
  job.ms += performance.now() - t0;
  if (done) job.active = false;
  return done;
}

// Shade one surfel. The normal is known analytically, so a lambert term plus a hemispheric sky term
// costs one dot product. Colour is hypsometric in height relative to the current view, invariant
// under zoom; the darkened bands are the true equipotentials, faded out near the boundary.
function emitTerr(i, slot) {
  const o9 = i * 9, o6 = i * 6, o3 = i * 3;
  const kind = pool.M[o9 + 5];
  const nx = pool.M[o9 + 6], ny = pool.M[o9 + 7], nz = pool.M[o9 + 8];
  let cr, cg, cb;
  if (kind === 1) {
    // The interior has zero distance estimate and zero potential, so nothing can colour it: flat
    // and unlit it measured 0.001 in linear light against a 0.0002 background, and the Feigenbaum
    // point is 100 percent interior past 1e7. It is a flat level surface, so shade it as water:
    // Schlick against the level normal, F0 = 0.02, so grazing lake reflects the sky.
    const vx = pool.rel[o3], vy = pool.rel[o3 + 1], vz = pool.rel[o3 + 2];
    const vl = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1;
    const ct = Math.abs(vy) / vl;                       // |cos| against world up
    const om = 1 - ct;
    const F = 0.02 + 0.98 * om * om * om * om * om;
    // Sun on water: mirror the view about the level surface and aim at the light. Exponent 260
    // sets the highlight width, 1.7 its strength.
    const gr = (vx / vl) * LX - (vy / vl) * LY + (vz / vl) * LZ;
    const g = gr > 0 ? Math.pow(gr, 260) * 1.7 : 0;
    cr = lakeCol[0] + (sheenCol[0] - lakeCol[0]) * F + g;
    cg = lakeCol[1] + (sheenCol[1] - lakeCol[1]) * F + g * 0.97;
    cb = lakeCol[2] + (sheenCol[2] - lakeCol[2]) * F + g * 0.88;
  }
  else {
    // Hypsometric, equalized in log height between two measured percentiles of the cut's own height
    // distribution. The square root is the fallback: deep in a valley most visible area is shore, so
    // a linear ramp puts most of the frame in the bottom eighth of the palette, and even the square
    // root over a span anchored at zero left 16 to 43 percent of the land in the darkest tenth.
    let t;
    if (job.eq) {
      const hr = pool.mu[o3 + 1] * job.invD;
      t = hr > 0 ? (Math.log2(hr) - job.hLo) * job.hInv : 0;
    } else {
      t = (pool.mu[o3 + 1] - job.hBase) / job.hSpan;
      t = t > 0 ? Math.sqrt(t) : 0;
    }
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    let pi = (t * (PAL - 1)) | 0;
    if (pi < 0) pi = 0; else if (pi > PAL - 1) pi = PAL - 1;
    const p3 = pi * 3;
    let sh = 0.36 * (0.5 + 0.5 * ny) + 0.95 * Math.max(0, nx * LX + ny * LY + nz * LZ);
    if (job.contour > 0 && kind === 0) {
      const de = pool.M[o9 + 3], r = pool.M[o9 + 2];
      const st = 1 - 2.5 * r / (de > 0 ? de : 1e-300);
      if (st > 0) {
        const nu = pool.M[o9 + 4];
        let f = nu - Math.floor(nu);
        if (f > 0.5) f = 1 - f;
        const line = 1 - 14 * f;
        if (line > 0) sh *= 1 - 0.5 * job.contour * st * (line > 1 ? 1 : line);
      }
    }
    cr = palette[p3] * sh; cg = palette[p3 + 1] * sh; cb = palette[p3 + 2] * sh;
  }
  const o = slot * FLOATS, d2 = job.invD * job.invD;
  emitted[slot] = i;
  instances[o] = pool.rel[o3]; instances[o + 1] = pool.rel[o3 + 1]; instances[o + 2] = pool.rel[o3 + 2];
  instances[o + 3] = pool.cov[o6] * d2; instances[o + 4] = pool.cov[o6 + 1] * d2;
  instances[o + 5] = pool.cov[o6 + 2] * d2; instances[o + 6] = pool.cov[o6 + 3] * d2;
  instances[o + 7] = pool.cov[o6 + 4] * d2; instances[o + 8] = pool.cov[o6 + 5] * d2;
  instances[o + 9] = cr; instances[o + 10] = cg; instances[o + 11] = cb;
  instances[o + 12] = pool.w[i];
  job.sumW += 1; job.sumWvis += 1;
  if (kind !== 1) {
    // Histogram of log2(h / dist), for a percentile. The distribution is heavy
    // tailed, so a mean is dragged around by a handful of far cells.
    const hr = pool.mu[o3 + 1] * job.invD;
    if (hr > 0) {
      let b = (Math.log2(hr) * 4 + 128) | 0;
      if (b < 0) b = 0; else if (b > 255) b = 255;
      hHist[b]++; job.hCount++;
    }
  }
}

/* ------------------------- level priority threshold --------------------- */
// Which nodes of the current level may split when the budget cannot take them all. A histogram over
// log2(priority) finds the threshold in one pass; sorting the level costs more than the refinement.
// Priorities are already written where the node is created, by `place` and by the IFS kernel.
let lastFocal = 1;
// The precision cost of the position subtraction in the kernel, in pixels, for the camera right
// now; with rebasing on every coordinate is order one however deep the zoom, so it stops growing.
// It must be live and not a by-product of a finished rebuild: a terrain rebuild past 1e6 takes tens
// of frames, the descent's park reads it, and a stale one let the camera fall from 1e14 to 1e29
// with the frame frozen at 1e14. It over-reports the true spacing 4.5x, so the park fires early.
function errPxNow() {
  // The per pixel GPU path has a different wall. WebGL 2 has no fp64, so its orbit runs in double
  // single, and what that delivers is a property of the driver: dsgl.html measures 38 to 43 bits
  // over the first thousand iterations against the same orbit in doubles. Take 40, eps 9.1e-13
  // against a double's 1.1e-16, so the descent turns a little past 1e9 where the CPU path reaches
  // 1e13. That is the arithmetic, not the representation, which dsprec.mjs puts at 2^-49.6.
  const eps = isDirect() ? 9.1e-13 : (cfg.precision === 'f32' ? 6e-8 : 1.1e-16);
  // The magnitude of world coordinate the rounding applies to: for an IFS the anchor holds
  // everything at order one, for a flat field it is the plane coordinate itself, since cells near
  // the aim carry the aim's magnitude. Using the framing distance there overstated the error 15x
  // and cost 1.2 decades of descent.
  const worldScale = isField()
    ? Math.max(Math.hypot(cam.target[0], cam.target[2]), 1e-4)
    : Math.max(Math.hypot(cam.target[0], cam.target[1], cam.target[2]) + cam.startDist, 1e-30);
  return eps * worldScale / Math.max(cam.dist, 1e-300) * lastFocal;
}

const SUBH = 32;                       // sub-buckets inside the edge bucket
const subHist = new Int32Array(SUBH);
function prepareLevel() {
  const nf = job.nf, cap = job.cap, nmaps = job.nmaps;
  const Ppr = pool.prio;
  job.bLow = -Infinity;
  // A cut wide floor under the per level threshold, the fix for shimmer under a smoothly moving
  // camera. The threshold derived below is a property of one level: it rations the frontier against
  // whatever the levels above left, which compounds. The Pythagoras tree's ratios are 0.809 and
  // 0.588, so a level synchronous frontier passes twenty five levels holding pieces whose sizes
  // differ by 1.38^25, and a small change at level nine reshuffles the bottom. No cross fade band
  // helps: they dissolve a node crossing its own threshold, and here the threshold itself jumps.
  if (job.bFloor > 0) job.bLow = job.bFloor;
  // A split costs nmaps + 1 slots: inside either cross fade band the parent is drawn alongside its
  // children. Reserving that slot keeps the per node backstop `cutSize + nmaps <= cap` from firing;
  // that backstop has no dissolve and is frontier order dependent, so it emits whole whatever the
  // walk reaches once the cut is full, 3 821 splats over 3 px at full weight on the icosahedral
  // flake, and flips an arbitrary subset of a level every rebuild.
  //
  // The node pool is the second ceiling: `cap` bounds the splats a cut may draw, MAXCAP the nodes
  // it may allocate, and a level synchronous walk that reclaims nothing allocates the sum of its
  // frontier over every level. The Pythagoras tree's long branch contracts by 0.848, so a pixel
  // takes about forty levels against the folded dragon's twenty four. `np >= MAXCAP` in the kernel
  // breaks the child loop mid node, so weight is not conserved; rationing by (MAXCAP - np)/nmaps
  // makes that unreachable and puts both ceilings through one level set of priority.
  const maxSplits = Math.min(Math.floor((cap - job.cutSize) / (nmaps + 1)),
                             Math.floor((MAXCAP - job.np) / nmaps));
  if (maxSplits < nf) {
    hist.fill(0);
    for (let q = 0; q < nf; q++) {
      const pr = Ppr[frontier[q]];
      let b = pr > 0 ? (Math.log2(pr) * 4 + 96) | 0 : 0;
      if (b < 0) b = 0; else if (b >= HB) b = HB - 1;
      hist[b]++;
    }
    let rem = maxSplits > 0 ? maxSplits : 0, edge = 0;
    for (let b = HB - 1; b >= 0; b--) {
      if (hist[b] <= rem) { rem -= hist[b]; edge = b; continue; }
      edge = b; break;
    }
    // Refine within the edge bucket, a factor 2^(1/4) wide, split 32 ways, so the threshold is a
    // number every node is compared against rather than a count handed out in walk order: the
    // boundary is then a level set of priority that sweeps smoothly instead of dissolving.
    const lo = Math.pow(2, (edge - 96) / 4);
    const sc = SUBH / (Math.LN2 * 0.25);
    subHist.fill(0);
    for (let q = 0; q < nf; q++) {
      const pr = Ppr[frontier[q]];
      if (!(pr >= lo)) continue;
      let sb = (Math.log(pr / lo) * sc) | 0;
      if (sb < 0) sb = 0; else if (sb >= SUBH) continue;   // above the edge bucket entirely
      subHist[sb]++;
    }
    let sub = 0, r2 = rem;
    for (let b = SUBH - 1; b >= 0; b--) {
      if (subHist[b] <= r2) { r2 -= subHist[b]; sub = b; continue; }
      sub = b + 1; break;
    }
    const lvl = lo * Math.exp(sub / sc);
    // The level's own threshold before the floor, and the tightest any level of this cut asked for.
    // The controller must track this and not the floored `bLow`, which is at least `bFloor` by
    // construction, so a floor smoothed toward it could only rise: feedback that runs away.
    if (lvl > job.bRaw) job.bRaw = lvl;
    if (lvl > job.bLow) job.bLow = lvl;         // whichever ceiling is tighter decides
  }
  job.bAllow = 0x7fffffff;
  job.q = 0; job.nn = 0; job.levelReady = true;
}

/* --------------------- the cut wide priority floor ---------------------- */
// One number carried between cuts: how small a splat this object may make at this budget. See
// prepareLevel for why a per level threshold alone shimmers.
//
// A first order filter, not a relay. Steering the fill fraction instead, up five percent over 94
// percent fill and down four under 80, is a relay in front of a plant with dead time, since a cut
// takes several frames to arrive and reports a floor several frames old: on the Pythagoras tree it
// hunted 89k, 83k, 110k, 103k, 136k, 127k, 168k, 157k, 208k, 194k splats a cut. `job.bRaw`, the
// tightest threshold any level of the last cut derived on its own, has the natural threshold as its
// fixed point. It must be the unfloored value, since `bLow` is at least `bFloor` by construction
// and smoothing toward it could only raise the floor. No raw threshold means the floor decays.
let bFloor = 0, bFloorKey = '';
const BF_LERP = 0.15;                  // how much of the gap the floor closes per cut
const BF_RELEASE = 0.85;               // and how fast it lets go when nothing needs rationing
function bFloorReset(key) {
  if (key === bFloorKey) return;
  bFloorKey = key; bFloor = 0;
}
function bFloorUpdate(raw) {
  if (raw > 0 && isFinite(raw)) {
    bFloor = bFloor > 0 ? bFloor + (raw - bFloor) * BF_LERP : raw;
  } else if (bFloor > 0) {
    bFloor *= BF_RELEASE;
    if (bFloor < 1e-7) bFloor = 0;
  }
  if (!(bFloor < 1e6)) bFloor = 1e6;
}

// Advance the current job by at most `budgetMs` of real work. Returns true when
// the cut is finished.
function stepJob(budgetMs) {
  if (job.plane) return stepPlane(budgetMs);
  return job.terrain ? stepTerr(budgetMs) : stepIFS(budgetMs);
}

function finishJob() {
  nEmit = job.nEmit;
  pool.n = job.np;
  // Track the threshold this cut actually needed. See bFloorUpdate.
  bFloorUpdate(job.bRaw);
  // Top of the hypsometric ramp: the 88th percentile of the height histogram times 1.15, which for
  // a roughly exponential distribution lands near the ninetieth, about 2.4 times the mean. Smoothed
  // at 0.4 per cut, in units of camera distance, so it holds still as the camera turns and closes.
  if (job.terrain && job.hCount > 256) {
    let want = Math.floor(job.hCount * 0.88), acc = 0, b = 0;
    for (; b < 256; b++) { acc += hHist[b]; if (acc >= want) break; }
    const rel = Math.pow(2, (b - 128) / 4) * 1.15;
    if (isFinite(rel) && rel > 0) {
      terrSpanRel = terrSpanRel > 0 ? terrSpanRel + (rel - terrSpanRel) * 0.4 : rel;
    }
    // The ends of the ramp, in log2 of height over camera distance, at the 6th and 97th percentiles
    // and smoothed at 0.35 per cut. Equalizing between percentiles rather than scaling from zero
    // keeps the whole ramp in use at every depth, see terrLo in src/40_state.js. The 1.2 octave
    // floor on the width stops a location with almost no relief from turning the ramp into a step.
    let lo = 0, hi = 0, a2 = 0;
    const wLo = job.hCount * 0.06, wHi = job.hCount * 0.97;
    for (let k = 0; k < 256; k++) {
      const prev = a2; a2 += hHist[k];
      if (prev < wLo && a2 >= wLo) lo = (k - 128) / 4;
      if (prev < wHi && a2 >= wHi) { hi = (k - 128) / 4; break; }
    }
    if (hi - lo < 1.2) hi = lo + 1.2;
    if (isFinite(lo) && isFinite(hi)) {
      if (!terrEq) { terrLo = lo; terrHi = hi; terrEq = true; }
      else { terrLo += (lo - terrLo) * 0.35; terrHi += (hi - terrHi) * 0.35; }
    }
  }
  stats.errPx = errPxNow();
  stats.splats = job.nEmit;
  stats.visited = job.visited;
  stats.culled = job.culled;
  stats.evals = job.evals;
  // Nominal is what the cells are worth, executed what ran, nominal minus what the field cache
  // served. Nominal alone beside a rebuild time would imply 1.3 G iterations a second at 1e7.
  stats.iters = job.iters;
  stats.itersReal = Math.max(0, job.iters - fcSaved - pfcSaved);
  if (job.plane) {
    stats.fieldHit = pfcHits / Math.max(pfcHits + pfcMiss, 1);
    // How much of this cut's field came off the GPU, and what it cost. See planeFlush.
    stats.gfQueries = gfQueries; stats.gfBatches = gfBatches; stats.gfMs = gfMs;
    // This gate reads the nominal cost per cell, not the executed one: with the cache off the two
    // are equal so the gate switches on, with it on the executed cost collapses and the gate
    // switches off, which oscillated every cut. A gate must not read a quantity it controls.
    if (job.evals > 1000) {
      const ipc = job.iters / job.evals;
      planeItEma = planeItEma > 0 ? planeItEma + (ipc - planeItEma) * 0.5 : ipc;
    }
  }
  if (job.terrain && job.evals > 1000) {
    // This gate reads the executed cost per cell, not the nominal one, because that is what
    // decides whether hashing is cheaper than iterating.
    const ipc = stats.itersReal / job.evals;
    terrItEma = terrItEma > 0 ? terrItEma + (ipc - terrItEma) * 0.5 : ipc;
    stats.fieldHit = fcHits / Math.max(fcHits + fcMiss, 1);
  }
  stats.dmin = (job.dmin === (1 << 30) ? 0 : job.dmin) + anchor.depth;
  stats.dmax = job.dmax + anchor.depth;
  stats.logZoom = logZoom();
  stats.buildMs = job.ms;
  stats.rebuilds++;
  // From the job's transported record, not the camera the kernel walked with: a rebase landing mid
  // rebuild made k wrong by exactly s and the gain by s squared, a 53 percent flash over 8 frames.
  built.pos.set(job.out.pos);
  built.R.set(job.out.R);
  built.dist = job.out.dist;
  built.Rq.set(job.out.Rq);
  built.sSince = job.out.sSince;
  built.count = job.nEmit;
  // `surface` selects the opacity model, constant peak rather than conserved energy, which both a
  // lit terrain and a flat field want. `prepass` is separate: it keeps the front layer only, which
  // a height field wants and a single layer plane does not.
  built.surface = job.terrain || job.plane;
  built.prepass = job.terrain;
  built.norm = job.norm;
  // Published together with the count, because `draw` reads both and a cut drawn with the wrong
  // instance stride is a screen of noise rather than a subtle error.
  built.flat = job.flat;
  built.planeY = job.planeY;
  built.valid = true;
  return job.nEmit;
}

// Blocking full rebuild, for build.mjs and the headless harnesses. The UI builds a new object's
// first cut as an ordinary stepped job.
function buildCut(vw, vh) {
  startJob(vw, vh);
  while (!stepJob(Infinity)) { /* unbounded time budget: one pass */ }
  return finishJob();
}

// How far the camera has drifted from the cut being drawn, and the exact correction for it.
//
// The cut stores each splat as (mu - pos_C) / dist_C, in the world coordinates of the frame C it
// was built in. The eye position of that splat now is
//
//     eye = R_F Q^-1 (mu_C - campos_C) / dist_F
//         = (R_F Rq^T) k rel_C  -  R_F o
//
// with Q = S Rq the composed map from C to the current frame F, k = dist_C / (S dist_F) and o the
// camera offset expressed in F. The splat rotates by the extra Rq^T and the offset does not, which
// is why the two cannot share one matrix. With Rq = I this collapses to the single matrix form,
// the usual case: Rq is identity unless a rebase landed while this cut was on screen.
//
// Same for the gain. Exposure is dist^-D and the cut's weights are branch weights relative to its
// own walk root, so the distance that belongs with them is the camera distance measured in C,
// which is dist_F * sSince.
const repro = {
  k: 1, o: new Float32Array(3),
  Rc: new Float32Array(9),            // column major, cut frame to eye
  gain: 1,
};
const rcT = new Float64Array(9);
function reprojection() {
  if (!built.valid) {
    repro.k = 1; repro.o[0] = repro.o[1] = repro.o[2] = 0;
    repro.gain = 1;
    const G = basis.Rgl; for (let i = 0; i < 9; i++) repro.Rc[i] = G[i];
    return 0;
  }
  const invD = 1 / cam.dist;
  repro.k = built.dist * invD;
  repro.o[0] = (basis.pos[0] - built.pos[0]) * invD;
  repro.o[1] = (basis.pos[1] - built.pos[1]) * invD;
  repro.o[2] = (basis.pos[2] - built.pos[2]) * invD;
  mat3MulT(cam.R, built.Rq, rcT);
  repro.Rc[0] = rcT[0]; repro.Rc[1] = rcT[3]; repro.Rc[2] = rcT[6];
  repro.Rc[3] = rcT[1]; repro.Rc[4] = rcT[4]; repro.Rc[5] = rcT[7];
  repro.Rc[6] = rcT[2]; repro.Rc[7] = rcT[5]; repro.Rc[8] = rcT[8];
  repro.gain = measureNormAt(cam.dist * built.sSince) / built.norm;
  let rot = 0;
  for (let i = 0; i < 9; i++) { const d = cam.R[i] - built.R[i]; rot += d * d; }
  return Math.max(Math.abs(Math.log(repro.k)),
    Math.hypot(repro.o[0], repro.o[1], repro.o[2]) * 3, Math.sqrt(rot) * 1.4);
}

/* --------------------------- the IFS hot kernel -------------------------- */
// stepIFS advances the job by at most `budgetMs` of wall clock, read every CHUNK
// node expansions. Flat typed arrays, loop invariants hoisted. The pushforward is
// exact:
//
//     M'     = M A_i                  (27 multiplies, or 3 when A_i = s I)
//     mu'    = mu + M d_i             (9)   d_i = w_i(mu_root) - mu_root
//     Sigma' = M' Sigma_root M'^T     (45,  or 6)
//
// Wall clock and not a node count fed back from the last rebuild's rate: one cold
// slice read 390 expansions/ms against a true 25 000 and starved every later
// rebuild to about 2 000 nodes a frame.
const CHUNK = 512;              // expansions per clock read, about 20 us at 25 000/ms
function stepIFS(budgetMs) {
  const t0 = performance.now();
  const tEnd = t0 + budgetMs;
  let chunk = CHUNK;
  const R = job.R;
  const r0 = R[0], r1 = R[1], r2 = R[2], r3 = R[3], r4 = R[4], r5 = R[5];
  const r6 = R[6], r7 = R[7], r8 = R[8];
  const invD = job.invD, d2 = invD * invD, focal = job.focal, sig = job.sig;
  const vw = job.vw, vh = job.vh, f32 = job.f32;
  const cpx = f32 ? Math.fround(job.pos[0]) : job.pos[0];
  const cpy = f32 ? Math.fround(job.pos[1]) : job.pos[1];
  const cpz = f32 ? Math.fround(job.pos[2]) : job.pos[2];
  const tanX = job.tanX, tanY = job.tanY, nrmX = job.nrmX, nrmY = job.nrmY;
  const NEAR = 1e-4, K = job.K, cap = job.cap, nmaps = job.nmaps, zFar = job.zFar;
  const splitPx = job.splitPx, fogK = job.fogK;
  const attFloor = job.attFloor;
  const huBase = job.huBase, fadeK = job.fadeK, budK = job.budK;
  const norm = job.norm;
  const PM = pool.M, Pmu = pool.mu, Pcov = pool.cov, Prel = pool.rel;
  const Pw = pool.w, Pcc = pool.cc, Pdep = pool.depth, Ppr = pool.prio;
  const Pfp = pool.fp, Psx = pool.sx, Psy = pool.sy;
  const Pls = pool.ls, Pbl = pool.bl;
  const Ppar = pool.par, Pmi = pool.mi;
  const rc0 = root.cov[0], rc1 = root.cov[1], rc2 = root.cov[2];
  const rc3 = root.cov[3], rc4 = root.cov[4], rc5 = root.cov[5];
  let np = job.np, nEmit = job.nEmit, cutSize = job.cutSize;
  let dmin = job.dmin, dmax = job.dmax, visited = job.visited, culled = job.culled;
  let sumW = job.sumW, sumWvis = job.sumWvis, bAllow = job.bAllow;
  const padX = vw * 0.1, padY = vh * 0.1;      // off screen margin of the sumWvis test
  // Per level, refreshed by prepareLevel below; a stale threshold stops refinement.
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
        chunk = CHUNK;
        if (performance.now() >= tEnd) { job.q = q; job.nn = nn; break outer; }
      }
      visited++;
      const i = frontier[q++];
      // Split on visible error, footprint times the attenuation the shader applies,
      // att = exp(-fog (z-1)), not on footprint alone: footprint alone put 66 percent
      // of the budget where att < 0.05 and left the near field at 6.2 px on the folded
      // dragon three decades down. Reserves nmaps and not nmaps - 1, since a node in
      // the fade band is drawn as well as its children; that keeps cutSize <= cap exact.
      let split = Ppr[i] > splitPx && Pdep[i] < 240 && cutSize + nmaps <= cap;
      // Same dissolve as the size threshold, or a node crossing bLow swaps itself for
      // its children at full weight in one frame. Band width is BUD_BAND, 40_state.js.
      let tbud = 1;
      if (split && bLow !== -Infinity) {
        tbud = (Ppr[i] / bLow - 1) * budK;
        if (!(tbud > 0)) { split = false; tbud = 0; } else if (tbud > 1) tbud = 1;
      }
      let made = 0, full = false, fadeTf = 1, fadeBl = 1;
      if (split) {
        const po3 = i * 3, po6 = i * 6, po9 = i * 9;
        const p0 = PM[po9], p1 = PM[po9 + 1], p2 = PM[po9 + 2];
        const p3 = PM[po9 + 3], p4 = PM[po9 + 4], p5 = PM[po9 + 5];
        const p6 = PM[po9 + 6], p7 = PM[po9 + 7], p8 = PM[po9 + 8];
        const pmx = Pmu[po3], pmy = Pmu[po3 + 1], pmz = Pmu[po3 + 2];
        const pw = Pw[i], pd = Pdep[i] + 1;
        const pcc = Pcc[i], pls = Pls[i];
        // Position in the cross fade band: 0 at the split threshold, 1 at FADE_BAND
        // times it. Below 1 the parent is drawn too, at the complementary weight, so a
        // split dissolves rather than substitutes. Weight is conserved exactly,
        // w(1-t) + t*sum(w p_i) = w, because the p_i sum to one.
        let tf = (Ppr[i] / splitPx - 1) * fadeK;
        if (!(tf > 0)) tf = 0; else if (tf > 1) tf = 1;
        if (tbud < tf) tf = tbud;              // whichever threshold is nearer decides
        const pbl = Pbl[i], cbl = pbl * tf;
        fadeTf = tf; fadeBl = pbl;
        const ps0 = Pcov[po6], ps1 = Pcov[po6 + 1], ps2 = Pcov[po6 + 2];
        const ps3 = Pcov[po6 + 3], ps4 = Pcov[po6 + 4], ps5 = Pcov[po6 + 5];
        for (let mi = 0; mi < nmaps; mi++) {
          if (np >= MAXCAP) { full = true; break; }
          const n = np, no3 = n * 3, no6 = n * 6, no9 = n * 9;
          const ao = mi * 9, doff = mi * 3;
          let q0, q1, q2, q3, q4, q5, q6, q7, q8;
          if (mapSc[mi]) {
            const s = mapS[mi], g = s * s;
            q0 = p0 * s; q1 = p1 * s; q2 = p2 * s;
            q3 = p3 * s; q4 = p4 * s; q5 = p5 * s;
            q6 = p6 * s; q7 = p7 * s; q8 = p8 * s;
            Pcov[no6] = ps0 * g; Pcov[no6 + 1] = ps1 * g; Pcov[no6 + 2] = ps2 * g;
            Pcov[no6 + 3] = ps3 * g; Pcov[no6 + 4] = ps4 * g; Pcov[no6 + 5] = ps5 * g;
          } else {
            const a0 = mapA[ao], a1 = mapA[ao + 1], a2 = mapA[ao + 2];
            const a3 = mapA[ao + 3], a4 = mapA[ao + 4], a5 = mapA[ao + 5];
            const a6 = mapA[ao + 6], a7 = mapA[ao + 7], a8 = mapA[ao + 8];
            q0 = p0 * a0 + p1 * a3 + p2 * a6;      // M' = M A_i
            q1 = p0 * a1 + p1 * a4 + p2 * a7;
            q2 = p0 * a2 + p1 * a5 + p2 * a8;
            q3 = p3 * a0 + p4 * a3 + p5 * a6;
            q4 = p3 * a1 + p4 * a4 + p5 * a7;
            q5 = p3 * a2 + p4 * a5 + p5 * a8;
            q6 = p6 * a0 + p7 * a3 + p8 * a6;
            q7 = p6 * a1 + p7 * a4 + p8 * a7;
            q8 = p6 * a2 + p7 * a5 + p8 * a8;
            const t00 = q0 * rc0 + q1 * rc1 + q2 * rc2;   // Sigma' = M' Sigma M'^T
            const t01 = q0 * rc1 + q1 * rc3 + q2 * rc4;
            const t02 = q0 * rc2 + q1 * rc4 + q2 * rc5;
            const t10 = q3 * rc0 + q4 * rc1 + q5 * rc2;
            const t11 = q3 * rc1 + q4 * rc3 + q5 * rc4;
            const t12 = q3 * rc2 + q4 * rc4 + q5 * rc5;
            const t20 = q6 * rc0 + q7 * rc1 + q8 * rc2;
            const t21 = q6 * rc1 + q7 * rc3 + q8 * rc4;
            const t22 = q6 * rc2 + q7 * rc4 + q8 * rc5;
            Pcov[no6]     = t00 * q0 + t01 * q1 + t02 * q2;
            Pcov[no6 + 1] = t00 * q3 + t01 * q4 + t02 * q5;
            Pcov[no6 + 2] = t00 * q6 + t01 * q7 + t02 * q8;
            Pcov[no6 + 3] = t10 * q3 + t11 * q4 + t12 * q5;
            Pcov[no6 + 4] = t10 * q6 + t11 * q7 + t12 * q8;
            Pcov[no6 + 5] = t20 * q6 + t21 * q7 + t22 * q8;
          }
          const dd0 = mapD[doff], dd1 = mapD[doff + 1], dd2 = mapD[doff + 2];
          let mx = pmx + p0 * dd0 + p1 * dd1 + p2 * dd2;      // mu' = mu + M d_i
          let my = pmy + p3 * dd0 + p4 * dd1 + p5 * dd2;
          let mz = pmz + p6 * dd0 + p7 * dd1 + p8 * dd2;
          // Pre cull write; a culled child leaves slot n to the next map, so only PM waits.
          Pmu[no3] = mx; Pmu[no3 + 1] = my; Pmu[no3 + 2] = mz;

          if (f32) { mx = Math.fround(mx); my = Math.fround(my); mz = Math.fround(mz); }
          const dx = (mx - cpx) * invD, dy = (my - cpy) * invD, dz = (mz - cpz) * invD;
          const rad = Math.sqrt(Pcov[no6] + Pcov[no6 + 3] + Pcov[no6 + 5]) * invD * sig;
          const ex = r0 * dx + r1 * dy + r2 * dz;
          const ey = r3 * dx + r4 * dy + r5 * dz;
          const ez = -(r6 * dx + r7 * dy + r8 * dz);
          const rr = K * rad;
          if (ez + rr < NEAR || ez - rr > zFar ||
              (Math.abs(ex) - ez * tanX) * nrmX > rr ||
              (Math.abs(ey) - ez * tanY) * nrmY > rr) { culled++; continue; }

          PM[no9] = q0; PM[no9 + 1] = q1; PM[no9 + 2] = q2;
          PM[no9 + 3] = q3; PM[no9 + 4] = q4; PM[no9 + 5] = q5;
          PM[no9 + 6] = q6; PM[no9 + 7] = q7; PM[no9 + 8] = q8;
          Prel[no3] = dx; Prel[no3 + 1] = dy; Prel[no3 + 2] = dz;
          const z = ez > NEAR ? ez : NEAR;
          const iz = 1 / z;                  // one reciprocal, not three divisions
          Psx[n] = vw * 0.5 + focal * ex * iz;
          Psy[n] = vh * 0.5 - focal * ey * iz;
          const fpn = focal * rad * iz;
          Pfp[n] = fpn;                      // debug field, unread on the IFS path
          // Attenuation at the near face of the support, ez - rr, not at the centre,
          // and floored by attFloor so footprint alone splits an oversized node: see
          // `place` in src/50_cut.js for both. Table lookup inlined from attOf
          // (src/40_state.js) because Math.exp here was 21 percent of a dragon rebuild.
          const zn = ez - rr;
          const xa = fogK * (zn - 1);
          let att;
          if (!(xa > 0)) att = 1;
          else {
            const u = xa * ATT_SC;
            if (u >= ATT_N) att = 0;
            else { const j = u | 0, ff = u - j; att = ATT[j] + (ATT[j + 1] - ATT[j]) * ff; }
          }
          Ppr[n] = fpn * (att > attFloor ? att : attFloor);
          Pw[n] = pw * mapP[mi];
          // Hue is one increment per letter, summed down the address and wrapped into
          // [0,1) as it goes, so a Float32 carries it at a uniform 1e-7 and a 240 deep
          // address is no worse than a 2 deep one. Only frac is ever read.
          let hu = pcc + mapHue[mi];
          hu -= Math.floor(hu);
          Pcc[n] = hu;
          // log10 of the piece's own diameter, integrated down the branch. Anchor
          // relative and exactly compensated by anchor.logScale, so it names an
          // absolute size at any depth. Read by the headless tests.
          Pls[n] = pls + mapLS[mi];
          Pbl[n] = cbl;
          Pdep[n] = pd;
          Ppar[n] = i; Pmi[n] = mi;      // so a clicked node can name its word
          np++;
          nextF[nn++] = n;
          made++;
        }
      }
      // Which node is drawn for this visit and at what fraction of its weight: a leaf
      // whole, a node inside the fade band at 1-t alongside its children.
      let emitI = -1, emitBl = 0;
      if (made > 0) {
        cutSize += made - 1;
        if (fadeTf < 1) { emitI = i; emitBl = fadeBl * (1 - fadeTf); cutSize++; }
      } else if (split && !full) {
        // Every child was culled. The children partition the parent's support and each
        // child's bound contains its own, so the parent has nothing visible either.
        cutSize--;
        culled++;
      } else {
        emitI = i; emitBl = Pbl[i];
      }
      if (emitI >= 0 && nEmit < cap) {
        emitted[nEmit] = i;
        const o = nEmit * FLOATS, o3 = i * 3, o6 = i * 6;
        instances[o]      = Prel[o3];
        instances[o + 1]  = Prel[o3 + 1];
        instances[o + 2]  = Prel[o3 + 2];
        instances[o + 3]  = Pcov[o6] * d2;
        instances[o + 4]  = Pcov[o6 + 1] * d2;
        instances[o + 5]  = Pcov[o6 + 2] * d2;
        instances[o + 6]  = Pcov[o6 + 3] * d2;
        instances[o + 7]  = Pcov[o6 + 4] * d2;
        instances[o + 8]  = Pcov[o6 + 5] * d2;
        // Colour is the address sum. The per map hue increments have zero measure
        // weighted mean (buildMapHue, src/40_state.js), so a region's mean hue holds
        // exactly at every level while a spread opens inside it. Hue keyed on a piece's
        // size cannot: refining replaces its pieces with smaller ones, a new colour.
        let u = huBase + Pcc[i];
        u -= Math.floor(u);
        let pi = (u * PAL) | 0;
        if (pi > PAL - 1) pi = PAL - 1; else if (pi < 0) pi = 0;
        const p3 = pi * 3;
        instances[o + 9]  = palette[p3];
        instances[o + 10] = palette[p3 + 1];
        instances[o + 11] = palette[p3 + 2];
        const wi = Pw[i] * emitBl;
        instances[o + 12] = wi * norm;
        sumW += wi;
        if (Psx[i] > -padX && Psx[i] < vw + padX && Psy[i] > -padY && Psy[i] < vh + padY) {
          sumWvis += wi;
        }
        nEmit++;
        const d = Pdep[i];
        if (d < dmin) dmin = d;
        if (d > dmax) dmax = d;
      }
    }
    const t = frontier; frontier = nextF; nextF = t;
    job.nf = nn; job.levelReady = false; job.q = 0; job.nn = 0;
  }

  job.np = np; job.nEmit = nEmit; job.cutSize = cutSize;
  job.dmin = dmin; job.dmax = dmax; job.visited = visited; job.culled = culled;
  job.sumW = sumW; job.sumWvis = sumWvis; job.bAllow = bAllow;
  job.ms += performance.now() - t0;
  if (done) job.active = false;
  return done;
}

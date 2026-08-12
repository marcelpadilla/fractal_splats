// Continuous frame diagnostics.
//
// Every visual claim in this project so far has been a still frame or a number,
// and the complaints that came back were all about MOTION: flicker, popping, a
// frame going black. A still cannot show any of those. So this rasterizes the
// splat buffer on the CPU with the same arithmetic the shaders use, drives the
// real frame loop for hundreds of frames, and reports what changes between one
// frame and the next.
//
// Driven by diag.mjs. See BW/BH/DS below for why the cut is built large and the
// raster is small.

import { api, get, set } from './harness.mjs';

const { cfg, cam, stats, job, built, repro, basis, palette } = api;
const FLOATS = api.FLOATS;
// A plane cut uploads seven floats an instance instead of thirteen; see FLAT_FLOATS in
// src/50_cut.js. This model has to decode BOTH, or every check on a plane object reads three
// quarters garbage and reports a frame that never existed.
const FLAT_FLOATS = api.FLAT_FLOATS;

// The cut is built at the resolution a person actually runs, because every level
// of detail decision is a screen footprint in pixels and building small would
// measure a different renderer. The raster is a clean integer downscale of that
// same frame: the focal length and the antialiasing filter scale together, so a
// splat lands in exactly the right place at 1/DS the size. Whole frame
// brightness, which is what flicker is, survives a downscale exactly.
export const BW = 1600, BH = 900, DS = 5;
export const VW = BW / DS, VH = BH / DS;

/* ------------------------------- rasterizer ------------------------------ */
// A transcription of VS + FS + TONE_FS from src/60_gl.js. If the two ever
// disagree the numbers here are worthless, so the shared constants are read out
// of the app rather than repeated.
const accum = new Float32Array(VW * VH * 4);
const rgb = new Float32Array(VW * VH * 3);
// Depth buffer for the surface prepass. Not optional: leaving it out made this
// instrument blind to an entire pass of the real renderer, and the terrain going
// completely black with 250 000 splats in the buffer lived exactly there. An
// instrument that models only some of the passes will confidently report that a broken
// frame is fine.
const depth = new Float32Array(VW * VH);

// Stand in for the GPU's copy of the instance buffer. This matters: a resumable
// job writes into `instances` as it goes, so at any moment mid rebuild that array
// holds the new cut's first nEmit splats followed by the tail of the old one.
// The browser never sees that, because `upload` only copies on finishJob. Reading
// `instances` directly reports a torn frame that does not exist on screen, and it
// cost me an hour of chasing a flash that was the harness.
let vram = new Float32Array(1);
export function upload() {
  const inst = get('instances');
  const n = built.count * (built.flat ? FLAT_FLOATS : FLOATS);
  if (vram.length < n) vram = new Float32Array(inst.length);
  vram.set(inst.subarray(0, n));
}

export function rasterize(vw = VW, vh = VH) {
  const inst = vram;
  const kern = get('kern');
  const count = built.count;
  const surface = built.surface;
  const focal = 0.5 * vh / Math.tan(0.5 * cam.fov);
  const gain = surface ? 1 : vw * vh * repro.gain;
  const R = basis.R;                       // world to eye, for the offset
  const C = api.mat3MulT(cam.R, built.Rq); // the cut's frame to eye, for the splat
  const k = repro.k, o = repro.o;
  // The camera offset, rotated once here instead of per splat.
  const ox = R[0] * o[0] + R[1] * o[1] + R[2] * o[2];
  const oy = R[3] * o[0] + R[4] * o[1] + R[5] * o[2];
  const oz = R[6] * o[0] + R[7] * o[1] + R[8] * o[2];
  const sig2 = cfg.sigma * cfg.sigma, k2 = k * k * sig2;
  const shape = kern.shape, beta2 = kern.beta * 0.5;
  const knorm = kern.knorm, extent = kern.extent;
  const fog = cfg.fog, filt = 0.5 / DS, fr = filt * filt;
  const fogc = get('bg1'), fg0 = fogc[0], fg1 = fogc[1], fg2 = fogc[2];
  accum.fill(0);
  // Log depth, exactly as src/60_gl.js sets uDepth, and the same near and far planes.
  const zf = Math.max(2, 1 + 8 / Math.max(cfg.fog, 0.02)), zn = 1e-4;
  const dScale = 1 / Math.log(zf / zn), dLogZn = Math.log(zn);
  // Quantized to 24 bits, like the DEPTH_COMPONENT24 renderbuffer. Not a detail: the
  // whole point of the prepass is that the front surface is pushed back by a small
  // amount, and if that amount rounds to zero the surfel fails its own depth test. A
  // float64 depth buffer cannot express that failure, so an unquantized model of this
  // pass reports a completely black terrain frame as perfectly healthy.
  const ndcOf = z => {
    const v = (Math.log(Math.max(z, 1e-7)) - dLogZn) * dScale * 2 - 1;
    const c = v < -1 ? -1 : (v > 1 ? 1 : v);
    return Math.round((c * 0.5 + 0.5) * 16777215) / 16777215 * 2 - 1;
  };
  if (surface) depth.fill(1);

  let drawn = 0, frags = 0, occluded = 0;
  // The flat instance layout, unpacked into the same quantities the general one gives, so that
  // everything below this is one code path. pos2 rad1 col3 w1 against pos3 cov6 col3 w1.
  const FLAT = !!built.flat, FL = FLAT ? FLAT_FLOATS : FLOATS;
  const planeY = built.planeY;
  for (let pass = surface ? 0 : 1; pass < 2; pass++)
  for (let q = 0; q < count; q++) {
    const b = q * FL;
    const px = inst[b] * k;
    const py = (FLAT ? planeY : inst[b + 1]) * k;
    const pz = (FLAT ? inst[b + 1] : inst[b + 2]) * k;
    // pc = uRc * (aPos * uK) - uR * uO
    const cx = C[0] * px + C[1] * py + C[2] * pz - ox;
    const cy = C[3] * px + C[4] * py + C[5] * pz - oy;
    const cz = C[6] * px + C[7] * py + C[8] * pz - oz;
    const z = -cz;
    if (!(z > 1e-6)) continue;
    // Sc = C S C^T with S the covariance scaled by k2.
    // For a flat cut the whole covariance is diag(r^2/3, (0.06 r)^2, r^2/3), reconstructed from
    // the one stored scalar exactly as FLAT_VS does.
    const rad = FLAT ? inst[b + 2] : 0;
    const fv = rad * rad * (1 / 3), ft2 = 0.0036 * rad * rad;
    const s0 = (FLAT ? fv : inst[b + 3]) * k2, s1 = FLAT ? 0 : inst[b + 4] * k2;
    const s2 = FLAT ? 0 : inst[b + 5] * k2;
    const s3 = (FLAT ? ft2 : inst[b + 6]) * k2, s4 = FLAT ? 0 : inst[b + 7] * k2;
    const s5 = (FLAT ? fv : inst[b + 8]) * k2;
    const cOff = FLAT ? b + 3 : b + 9;
    const iz = 1 / z, iz2 = iz * iz;
    const j00 = focal * iz, j02 = focal * cx * iz2;
    const j11 = focal * iz, j12 = focal * cy * iz2;
    // Sc j = C S C^T j, done as (C^T j) then S then dot.
    const a0 = C[0] * j00 + C[6] * j02, a1 = C[1] * j00 + C[7] * j02, a2 = C[2] * j00 + C[8] * j02;
    const b0 = C[3] * j11 + C[6] * j12, b1 = C[4] * j11 + C[7] * j12, b2 = C[5] * j11 + C[8] * j12;
    const Sa0 = s0 * a0 + s1 * a1 + s2 * a2;
    const Sa1 = s1 * a0 + s3 * a1 + s4 * a2;
    const Sa2 = s2 * a0 + s4 * a1 + s5 * a2;
    const Sb0 = s0 * b0 + s1 * b1 + s2 * b2;
    const Sb1 = s1 * b0 + s3 * b1 + s4 * b2;
    const Sb2 = s2 * b0 + s4 * b1 + s5 * b2;
    let c00 = a0 * Sa0 + a1 * Sa1 + a2 * Sa2;
    let c01 = a0 * Sb0 + a1 * Sb1 + a2 * Sb2;
    let c11 = b0 * Sb0 + b1 * Sb1 + b2 * Sb2;
    c00 += fr; c11 += fr;
    // Sc[2][2] = e3^T C S C^T e3, with e3 the eye z axis, i.e. row 3 of C.
    const g0 = C[6], g1 = C[7], g2 = C[8];
    const Sg0 = s0 * g0 + s1 * g1 + s2 * g2;
    const Sg1 = s1 * g0 + s3 * g1 + s4 * g2;
    const Sg2 = s2 * g0 + s4 * g1 + s5 * g2;
    const scz = g0 * Sg0 + g1 * Sg1 + g2 * Sg2;
    const det = Math.max(c00 * c11 - c01 * c01, 1e-14);
    const att = Math.exp(-fog * Math.max(0, z - 1));
    let peak, cr, cg, cb;
    const w = inst[FLAT ? b + 6 : b + 12];
    if (!surface) {
      peak = w * gain * att * knorm / Math.sqrt(det);
      cr = inst[cOff]; cg = inst[cOff + 1]; cb = inst[cOff + 2];
    } else {
      peak = w * gain;
      cr = fg0 + (inst[cOff] - fg0) * att;
      cg = fg1 + (inst[cOff + 1] - fg1) * att;
      cb = fg2 + (inst[cOff + 2] - fg2) * att;
    }
    if (!(peak > 1e-7)) continue;
    const i00 = c11 / det, i01 = -c01 / det, i11 = c00 / det;
    const rx = Math.min(extent * Math.sqrt(c00), vw);
    const ry = Math.min(extent * Math.sqrt(c11), vh);
    const ux = focal * cx * iz + vw * 0.5;
    const uy = -(focal * cy * iz) + vh * 0.5;      // GL y up, buffer y down
    let x0 = Math.floor(ux - rx), x1 = Math.ceil(ux + rx);
    let y0 = Math.floor(uy - ry), y1 = Math.ceil(uy + ry);
    if (x1 < 0 || y1 < 0 || x0 >= vw || y0 >= vh) continue;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 > vw - 1) x1 = vw - 1; if (y1 > vh - 1) y1 = vh - 1;
    if (pass === 1) drawn++;
    // Prepass writes the front surface pushed back uSlab of its own sigma_z; the colour
    // pass tests the unpushed depth against it with LESS and depth writes off.
    const nd = ndcOf(pass === 0 ? z + 4.0 * Math.sqrt(Math.max(scz, 0)) : z);
    const core = pass === 0 ? 0.22 : 0.0;
    for (let y = y0; y <= y1; y++) {
      const dy = -(y + 0.5 - vh * 0.5) - (-(uy - vh * 0.5));   // offset in GL space
      const row = y * vw;
      for (let x = x0; x <= x1; x++) {
        const dx = (x + 0.5) - ux;
        const qf = i00 * dx * dx + 2 * i01 * dx * dy + i11 * dy * dy;
        const e = 0.5 * Math.pow(Math.max(shape * qf, 1e-9), beta2);
        if (e > 6.91) continue;
        const kv = Math.exp(-e);
        if (kv < core) continue;
        if (pass === 0) { if (nd < depth[row + x]) depth[row + x] = nd; continue; }
        if (surface && !(nd < depth[row + x])) { occluded++; continue; }
        const a = peak * kv;
        const p = (row + x) * 4;
        accum[p] += cr * a; accum[p + 1] += cg * a; accum[p + 2] += cb * a; accum[p + 3] += a;
        frags++;
      }
    }
  }

  // TONE_FS.
  const bg0 = get('bg0'), bg1 = get('bg1');
  const optical = cfg.density, glow = cfg.glow, white = 12;
  const lw = Math.log(1 + white);
  for (let y = 0; y < vh; y++) {
    const v = 1 - (y + 0.5) / vh;                 // vUv.y, GL origin at bottom
    for (let x = 0; x < vw; x++) {
      const p = (y * vw + x) * 4, o3 = (y * vw + x) * 3;
      const A = Math.max(accum[p + 3], 0);
      const T = A * optical;
      const inv = 1 / Math.max(A, 1e-12);
      const a = 1 - Math.exp(-T);
      const g = Math.min(Math.log(1 + T) / lw, 6);
      const bl = a + (g - a) * glow;
      let c0 = accum[p] * inv * bl, c1 = accum[p + 1] * inv * bl, c2 = accum[p + 2] * inv * bl;
      const ca = Math.min(Math.max(a, 0), 1);
      c0 += (bg0[0] + (bg1[0] - bg0[0]) * v) * (1 - ca);
      c1 += (bg0[1] + (bg1[1] - bg0[1]) * v) * (1 - ca);
      c2 += (bg0[2] + (bg1[2] - bg0[2]) * v) * (1 - ca);
      const l = 0.2126 * c0 + 0.7152 * c1 + 0.0722 * c2;
      rgb[o3] = Math.max(l + (c0 - l) * 1.14, 0);
      rgb[o3 + 1] = Math.max(l + (c1 - l) * 1.14, 0);
      rgb[o3 + 2] = Math.max(l + (c2 - l) * 1.14, 0);
    }
  }
  return { drawn, frags, occluded };
}

// Perceptual summary of one frame: mean of the gamma encoded luminance, which is
// what "the frame went black" and "the frame flashed" are statements about.
export function frameStats() {
  let sum = 0, lit = 0, litSum = 0;
  let o1 = 0, o2 = 0, o1s = 0, o2s = 0;
  const n = VW * VH;
  const lum = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const l = 0.2126 * rgb[i * 3] + 0.7152 * rgb[i * 3 + 1] + 0.0722 * rgb[i * 3 + 2];
    const e = Math.pow(Math.max(l, 0), 1 / 2.2);
    lum[i] = e; sum += e;
    if (e > 0.06) {
      lit++; litSum += e;
      // Opponent chroma of the displayed pixel, normalized by its own brightness so
      // this measures HUE spread and not brightness spread. Its standard deviation
      // over the covered part of the frame is how much of the palette a viewer can
      // actually see, as opposed to how much of it the colour coordinate spans.
      const s0 = rgb[i * 3], s1 = rgb[i * 3 + 1], s2 = rgb[i * 3 + 2];
      const t = Math.max(s0 + s1 + s2, 1e-9);
      const u = (s0 - s1) / t, v = (s1 - s2) / t;
      o1 += u; o2 += v; o1s += u * u; o2s += v * v;
    }
  }
  // `mean` is the whole frame and is what flicker is measured on. `onObject` divides
  // out the coverage, which is the honest way to ask whether the EXPOSURE is flat:
  // an object that surrounds the camera at depth legitimately fills more of the
  // frame than one seen whole from outside, and that is not an exposure error.
  const L = Math.max(lit, 1);
  const hue = Math.sqrt(Math.max(o1s / L - (o1 / L) ** 2, 0) + Math.max(o2s / L - (o2 / L) ** 2, 0));
  return { mean: sum / n, cover: lit / n, onObject: litSum / Math.max(lit, 1), hue, lum };
}

/* ------------------------------ frame driver ----------------------------- */
// The real loop from src/70_ui.js, minus the GL calls. Same order, same
// resumable refinement, same work budget adaptation, so a bug in the loop shows
// up here.
export function runFrames(preset, nf, opts = {}) {
  const dt = opts.dt || 1 / 60;
  api.loadPreset(preset, false);
  if (opts.cfg) for (const k in opts.cfg) cfg[k] = opts.cfg[k];
  if (opts.cfg && opts.cfg.kernel !== undefined) set('kern', api.kernelConst(cfg.kernel));
  cfg.autopilot = opts.autopilot !== undefined ? opts.autopilot : true;
  api.buildCut(BW, BH);
  upload();
  repro.k = 1; repro.o[0] = repro.o[1] = repro.o[2] = 0;

  let work = api.frameWork ? api.frameWork() : 12000, acc = 0;
  const out = [];
  let prev = null;
  for (let f = 0; f < nf; f++) {
    if (opts.turn) api.turnCamera(opts.turn * dt, 0);
    api.advanceCamera(dt);
    api.updateBasis();
    const drift = reproject();
    acc += dt;
    if (job.active) {
      if (api.stepJob(work)) { api.finishJob(); upload(); acc = 0; }
    } else if (get('dirty') || !built.valid || drift > 0.04 || acc > 0.5) {
      api.startJob(BW, BH);
      set('dirty', false);
      if (api.stepJob(work)) { api.finishJob(); upload(); acc = 0; }
    }
    if (api.frameWork) work = api.frameWork();
    else if (job.ms > 0 && job.visited > 0) {
      work = Math.max(2000, Math.min(400000, Math.round(job.visited / Math.max(job.ms, 0.05) * 6)));
    }
    reproject();
    const r = rasterize();
    const s = frameStats();
    let dLum = 0;
    if (prev) {
      for (let i = 0; i < s.lum.length; i++) dLum += Math.abs(s.lum[i] - prev[i]);
      dLum /= s.lum.length;
    }
    prev = s.lum;
    out.push({
      f, mean: s.mean, cover: s.cover, onObject: s.onObject, dLum, splats: built.count, drawn: r.drawn,
      zoom: stats.logZoom, dist: cam.dist, err: stats.errPx,
      rebases: stats.rebases, dmax: stats.dmax,
    });
  }
  return out;
}

const reprojection = get('reprojection');
function reproject() { return reprojection(); }

/* ================== the escape time field, per pixel, on the GPU ========== */
// The same field as src/32_plane.js, evaluated once per pixel in a fragment shader rather than once
// per quadtree cell on the CPU: no splats, no refinement, no budget. It is the comparison object
// for the splat path, so everything that can be shared stays shared. Per frame it does 7 to 11
// times more arithmetic and is still faster on wall clock: 310 M iterations a second on one CPU
// thread against 87 to 95 G here (knowledge/mandelbrot_speed.md). Precision is double single, from
// src/61_pert.js: the wall is near 1e9 zoom against the splat path's 1e13, and errPxNow turns the
// descent around at it.
const DIRECT_VS = `#version 300 es
precision highp float;
void main() {
  vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}`;

const DIRECT_FS = `#version 300 es
precision highp float;
precision highp int;
` + PERT_GLSL + `
// ---- geometry --------------------------------------------------------------
uniform vec2 uVp;            // viewport, pixels
uniform float uFocal;
uniform vec2 uA, uB;         // the homography's two numerators, see directMap
uniform vec3 uWrow;          // and its denominator
uniform float uPosY;         // camera height above the plane
uniform float uPix;          // one pixel, in plane units, at the centre of the frame
// ---- appearance ------------------------------------------------------------
uniform float uNuInv;        // 1 / nuCycle
uniform sampler2D uPalTex;   // the colour ramp: PAL x 1, linear light, a closed loop
uniform sampler2D uPalIntTex;// and its integral, (PAL+1) x 1. See rampBox
uniform vec3 uPalMean;
uniform vec3 uLake;
uniform int uSS;             // supersampling grid, uSS by uSS, and 1 is off. See shadePixel
uniform int uDbg;            // 1: draw the recovered pixel coordinate instead of the fractal

out vec4 outColor;

const float TANGLE_SET = 0.45;           // see emitPlane in src/52_cut_plane.js
const float DEEP_SET = 0.60;
const float FILT_MAX = 0.30;
const int PAL_I = ` + PAL + `;           // the palette table's length, shared with src/40_state.js
const float PAL_F = float(PAL_I);

// THE SAME TABLE THE SPLATS USE, THROUGH THE SAME LAW, and both halves of that sentence were
// false until this was written.
//
// This path used to carry the preset's NINE STOPS as a uniform array and interpolate between them
// in linear light. The splat path builds a 256 entry table by interpolating the same stops in
// DISPLAY space and linearizing afterwards. Those are different curves: a lerp between two stops
// lands at the same triple either way, but read as display that triple is darker than read as
// linear light, so the two renderers disagreed about every colour that was not a stop.
//
// MEASURED, over the whole of the old nine stop ramp, and the number is worth writing down
// because the guess was wrong. This path came out 1.4 percent lighter in Oklab L and 2.7 percent
// LESS colourful, with a worst case gap of 0.021 in L, about two percent of the ramp's range. So
// the disagreement was real and it was small, and it is not what the two objects looked different
// by. The answer to that was the anti-aliasing, in planeNeedsAA over in src/52_cut_plane.js, which
// was averaging shaded colours and cost eleven points of saturation and half the detail.
//
// It is fixed anyway, and the reason is not the size of the error. The whole point of having both
// objects in the menu is that the button between them changes the REPRESENTATION and nothing else;
// a difference of any size that comes from the palette rather than from the representation makes
// the comparison say something it does not mean.
//
// So the table crosses as a texture and there are no stops here at all. texelFetch and an explicit
// mix rather than hardware LINEAR filtering, because the integral below needs fp32 and filtering a
// float texture needs an extension this does not otherwise want. No back quotes anywhere in this
// comment, either: it lives inside a JavaScript template literal, and one closed the shader.
vec3 palAt(int i) { return texelFetch(uPalTex, ivec2(i, 0), 0).rgb; }
vec3 ramp(float t) {
  float x = fract(t) * PAL_F;
  int i = int(x);
  return mix(palAt(i), palAt(i + 1 == PAL_I ? 0 : i + 1), x - float(i));
}
// F(v), the integral of the ramp from 0 to v, extended past one period by adding whole period
// integrals: that is exactly palMean per turn. b < 2 always here, so one wrap is enough.
vec3 palIntAt(float t) {
  vec3 add = vec3(0.0);
  if (t >= 1.0) { t -= 1.0; add = uPalMean; }
  float x = t * PAL_F;
  int i = int(x);
  if (i > PAL_I) i = PAL_I;
  vec3 a = texelFetch(uPalIntTex, ivec2(i, 0), 0).rgb;
  vec3 b = texelFetch(uPalIntTex, ivec2(i + 1, 0), 0).rgb;
  return mix(a, b, x - float(i)) + add;
}
// The exact average of the periodic ramp over an interval of width s turns centred on t. Two
// lookups and a subtraction, and it is the SAME arithmetic as rampBox in src/40_state.js, line for
// line. What was here before was twelve stratified samples of the nine stop ramp, which was an
// approximation to a different function.
vec3 rampBox(float t, float s) {
  if (s < 1e-4) return ramp(t);
  if (s >= 1.0) return uPalMean;
  float a = t - 0.5 * s;
  a -= floor(a);
  return (palIntAt(a + s) - palIntAt(a)) * (1.0 / s);
}

// ONE SAMPLE, in linear light, and the whole colour law lives here now.
//
// It used to be inline in main() and it had to move for anti-aliasing, but the move fixed two
// things on its own. The ramp's box filter width came from fwidth(t), the screen space derivative
// of the ramp coordinate, and that is the one place the two paths could not agree because the
// splat path computes the same width analytically from |grad nu|. Worse, fwidth is a difference
// against the NEIGHBOURING FRAGMENT, so at the boundary, where the neighbour is interior and its
// t is a constant standing in for nothing, the width was the difference between a real ramp
// coordinate and a placeholder: garbage exactly where the filter matters. It is now
//
//     span = |grad nu| * (one sample's width) * d t / d nu
//
// which is a property of the point, is what the quadtree already computes per cell, and needs no
// neighbour. And a sample whose distance estimate says the boundary is inside its own footprint
// gets the full width, the same rule and the same cap planeCell uses.
vec3 shadeAt(vec2 dc, float rTol) {
  Fld f = pertField(dc, rTol);
  if (f.why == 2) return uLake;                                   // proven interior
  if (f.why == 3) return mix(uPalMean, uLake, DEEP_SET);          // out of iterations
  float nc = max(f.nu, NU_MIN);
  float t = 3.0 * pow(nc, 1.0 / 3.0) * uNuInv;
  float span = FILT_MAX;
  if (f.why == 4 && f.de >= 1.41421356 * rTol) {
    span = min(f.gnu * 2.0 * rTol * uNuInv / pow(nc, 2.0 / 3.0), FILT_MAX);
  }
  vec3 col = rampBox(t, span);
  // How much of the sample the set fills. de is a lower bound on the distance to the set, so a
  // sample with de >= rTol sqrt(2) cannot contain any of it and gets no darkening at all.
  float k = (f.why == 4) ? 1.0 - f.de / (1.41421356 * rTol) : TANGLE_SET;
  if (k > 0.0) {
    k = min(k, 1.0);
    k = k * k * (3.0 - 2.0 * k);
    col = mix(col, uLake, k);
  }
  float rel = f.de > 0.0 ? clamp(log2(f.de / rTol) * (1.0 / 6.0), 0.0, 1.0) : 0.0;
  return col * (0.88 + 0.22 * rel);
}

// ADAPTIVE SUPERSAMPLING, and adaptive is what makes it affordable.
//
// The escape time field is analytic in the exterior and the ramp is box filtered over the exact
// span the sample covers, so out there one sample is not an approximation, it is the answer: extra
// samples of a linear field average to the same number. Every jagged edge in this render is within
// one distance estimate of the set, and the distance estimate is already in hand from the first
// sample. So: shade the centre, and if it escaped with the boundary more than its own footprint
// away, stop. Otherwise lay down an uSS by uSS grid over the pixel, each sample carrying the
// SUB pixel footprint as its own rTol, and average in linear light.
//
// That last point is what makes it a real supersample rather than a blur: rTol drives the
// derivative bailout and the set fill, so a sub sample is asked the question at its own scale and
// resolves structure the whole pixel cannot. Cost is measured, not assumed; see the note in
// knowledge/plane_lod.md. Only the boundary pixels pay, but the boundary pixels are also the
// expensive ones, so this is switched off entirely while the camera moves.
vec3 shadePixel(vec2 dc, float pix) {
  float rTol = pix * 0.5;
  Fld f = pertField(dc, rTol);
  if (uSS < 2 || (f.why == 4 && f.de >= 2.0 * rTol)) return shadeAt(dc, rTol);
  int n = uSS;
  float inv = 1.0 / float(n);
  float sub = rTol * inv;
  vec3 acc = vec3(0.0);
  for (int j = 0; j < 4; j++) {
    if (j >= n) break;
    for (int i = 0; i < 4; i++) {
      if (i >= n) break;
      vec2 o = pix * vec2((float(i) + 0.5) * inv - 0.5, (float(j) + 0.5) * inv - 0.5);
      acc += shadeAt(dc + o, sub);
    }
  }
  return acc / float(n * n);
}

void main() {
  // Which point of the plane is this pixel: exactly the inverse of place() in src/50_cut.js, as a
  // homography, expressed as an OFFSET from the centre of the frame. The offset is of order the
  // frame width, so fp32 carries it to a ten thousandth of a pixel, and it is ALSO exactly the dc
  // that perturbation wants. That coincidence is the whole reason this path needs no extended
  // precision anywhere: the only quantity that ever needs fourteen digits is the reference point,
  // and that one lives on the CPU.
  float u = (gl_FragCoord.x - 0.5 * uVp.x) / uFocal;
  float v = (gl_FragCoord.y - 0.5 * uVp.y) / uFocal;
  float W = uWrow.x * u + uWrow.y * v + uWrow.z;
  vec2 dc = uPosY * (uA * u + uB * v) / W;
  vec2 off = dc;

  // A diagnostic that earned its place. When the deep frame came back quantized into blocks exactly
  // one fp32 ulp of the frame centre across, there were two candidates, the coordinate and the
  // orbit, and no way to tell them apart from a picture of a fractal. This draws the coordinate
  // alone, as a four pixel sawtooth. Reached with dbg=1, and free when off.
  if (uDbg == 1) {
    float qx = off.x / uPix, qy = off.y / uPix;
    outColor = vec4(fract(qx * 0.25), fract(qy * 0.25), 0.5 * fract(qx * 0.03125), 1.0);
    return;
  }

  // Evaluation only: draw the VERDICT rather than the colour, at one sample. Two of the four
  // verdicts are dark and two are mid, so a frame cannot be read for which one it is, and the one
  // bug this path has had twice is a verdict that is right on the CPU and wrong here. dbg=2.
  //   red escaped, blue proven interior, green condemned, white out of iterations.
  if (uDbg == 2) {
    Fld f = pertField(dc, uPix * 0.5);
    vec3 c2 = vec3(1.0);
    if (f.why == 2) c2 = vec3(0.1, 0.3, 1.0);
    else if (f.why == 4) c2 = vec3(1.0, 0.2, 0.1);
    else if (f.why == 1) c2 = vec3(0.1, 0.8, 0.2);
    outColor = vec4(c2, 1.0);
    return;
  }
  // ---- colour, the same law the splats use ---------------------------------
  // Same ramp coordinate, same box filter, same set fill guesses, same proximity shading, and now
  // the same ANALYTIC filter width as well. The point of this object is a comparison, so anything
  // different here would make it useless. See shadeAt and shadePixel.
  vec3 col = shadePixel(dc, uPix);
  // The same tail as TONE_FS in src/60_gl.js: hold saturation, then encode.
  float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
  col = max(mix(vec3(l), col, 1.14), vec3(0.0));
  outColor = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}`;

// Present an already shaded RGBA8 texture. A still pass fills one a band at a time and shows it.
const BLIT_FS = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uVp;
out vec4 outColor;
void main() { outColor = texelFetch(uTex, ivec2(gl_FragCoord.xy), 0); }`;

const DIRECT_UNI = ['uVp', 'uFocal', 'uA', 'uB', 'uWrow', 'uPosY', 'uPix', 'uNuInv',
  'uPalTex', 'uPalIntTex', 'uPalMean', 'uLake', 'uSS', 'uDbg',
  'uCtrX', 'uCtrY', 'uCX', 'uCY', 'uJulia', 'uPower', 'uMaxIter'];
let progDirect = null, progBlit = null, blitUni = {};
const dirUni = {};
// The progressive target and where the pass has got to. See drawDirect.
let dirFbo = null, dirTex = null, dirTW = 0, dirTH = 0;
let dirBand = 0, dirBands = 1, dirPassCap = 0, dirHave = false, dirDone = false;
// The cap the shader runs at and the one it climbs towards, as in the splat path's stationary ramp.
let dirCap = 800, dirWant = 800, dirIters = 0, dirMs = 0, dirSS = 1;
// Strips per frame. A single draw call running a second or more trips the Windows display driver
// reset, which kills the context; eight strips bound one command without changing the frame.
const DIR_STRIPS = 8;
// The ceiling on pixels times iterations in one frame. dsrate.html times this kernel on an off
// screen 1377x1592 target with a pipeline flush around it, RTX 4090 through ANGLE and D3D11: 87 to
// 95 G iterations a second in double single, against 1900 to 2190 G for plain fp32 and 310 M for
// one CPU thread. Every constant below follows from the 95 G figure. Two ceilings, as planeRoot has
// two: a moving camera is paying latency and a still one is not, so moving gets 2e9 pixel
// iterations, a 21 ms frame.
let DIR_WORK_MOVE = 2.0e9;
// The still ceiling, twenty times the moving one, because a still frame is a pass rather than a
// frame. At zoom 1e6 the view needs a cap near 95 000; a ceiling of 2e10 over a two megapixel frame
// allowed 9 000, leaving 40 percent of the frame at the flat out-of-iterations colour, mean
// displayed luminance 0.33 against the splat path's 0.26. But 95 000 over two megapixels is 2.1e11
// iterations, 2.2 s at 95 G, too long for one draw, so the pass is split into bands of
// DIR_BAND_WORK, one per frame: a shallow view converges in one band and a deep one takes as many
// as it needs.
let DIR_WORK = 4.0e11;
// About 65 ms of GPU at the measured double single rate. Evaluation only, settable by `bw=`: a
// headless page renders about four frames before its time budget expires, so a banded pass never
// finishes there and a capture needs `bw=1e12`, which puts the whole pass in one band.
let DIR_BAND_WORK = 6.0e9;
const DIR_BAND_MAX = 96;
// Evaluation only: draw the recovered pixel coordinate instead of the fractal. See uDbg.
let DIR_DBG = 0;
// The supersampling grid for a still frame, uSS by uSS inside each pixel of the boundary layer; 1
// is off. Two rather than three by default, from knowledge/plane_lod.md: the third sample changes
// the frame by under a percent of full range for another factor of two in cost. A moving frame
// never supersamples, because under DIR_WORK_MOVE that factor would come out of the iteration cap
// instead.
let DIR_SS = 2;

function isDirect() { return !!PRESETS[cfg.preset].direct; }

// Linked asynchronously: the double single orbit takes about a second through the HLSL compiler.
let dirLink = null;
function initDirect() {
  if (!dirLink) dirLink = linkAsync(DIRECT_VS, DIRECT_FS);
  if (!linkReady(dirLink)) return false;
  progDirect = dirLink.p;
  for (const k of DIRECT_UNI) dirUni[k] = gl.getUniformLocation(progDirect, k);
  progBlit = link(DIRECT_VS, BLIT_FS);
  blitUni.uTex = gl.getUniformLocation(progBlit, 'uTex');
  dirTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, dirTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  dirFbo = gl.createFramebuffer();
  return true;
}

// The palette and its integral, as two one row RGBA32F textures. fp32 and not fp16: rampBox reads
// the integral as a difference of two nearby values, so over a span of a thousandth of a turn a
// half float's eleven bit mantissa leaves about one bit of answer. The 257 texel table is three
// kilobytes.
let palTex = null, palIntTex = null, palTexGen = -1;
const palBuf = new Float32Array(PAL * 4), palIntBuf = new Float32Array((PAL + 1) * 4);
function palUpload() {
  if (palTexGen === palGen) return;
  palTexGen = palGen;
  if (!palTex) {
    palTex = gl.createTexture(); palIntTex = gl.createTexture();
    for (const t of [palTex, palIntTex]) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      // NEAREST throughout: every lookup in the shader is a texelFetch with the interpolation
      // written out, so no sampler state is consulted and no float filtering extension is needed.
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
  }
  for (let i = 0; i < PAL; i++) {
    palBuf[i * 4] = palette[i * 3]; palBuf[i * 4 + 1] = palette[i * 3 + 1];
    palBuf[i * 4 + 2] = palette[i * 3 + 2]; palBuf[i * 4 + 3] = 1;
  }
  for (let i = 0; i <= PAL; i++) {
    palIntBuf[i * 4] = palInt[i * 3]; palIntBuf[i * 4 + 1] = palInt[i * 3 + 1];
    palIntBuf[i * 4 + 2] = palInt[i * 3 + 2]; palIntBuf[i * 4 + 3] = 1;
  }
  gl.bindTexture(gl.TEXTURE_2D, palTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, PAL, 1, 0, gl.RGBA, gl.FLOAT, palBuf);
  gl.bindTexture(gl.TEXTURE_2D, palIntTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, PAL + 1, 1, 0, gl.RGBA, gl.FLOAT, palIntBuf);
}

// The progressive target, at the size of the frame. RGBA8 suffices: the shader writes the finished
// pixel, gamma encoded, so there is nothing to accumulate and nothing that needs range.
function dirResize(vw, vh) {
  if (vw === dirTW && vh === dirTH) return true;
  gl.bindTexture(gl.TEXTURE_2D, dirTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, vw, vh, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, dirFbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, dirTex, 0);
  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  dirTW = ok ? vw : 0; dirTH = ok ? vh : 0;
  dirHave = false; dirBand = 0;
  return ok;
}

// The image to plane map, in closed form. A pixel's ray is d = u*right + v*up + fwd and the plane
// is y = 0, so the hit is pos - pos_y * d / d_y, a homography. As an offset from the frame centre
// it loses the large common part exactly:
//
//   hit_x - ctr_x = pos_y (u ax + v bx) / W,   ax = fwd_x r_y / fwd_y - r_x
//   W = u r_y + v up_y + fwd_y
//
// so everything the shader touches per pixel is of order one except pos_y, and the centre, the only
// quantity needing more than fp32, is the perturbation reference point and stays on the CPU. Kept a
// separate function so the headless suite can compose it with place() and demand the identity: a
// slightly wrong hand derived inverse renders a correct looking image of the wrong region.
function directMap(vw, vh) {
  updateBasis();
  const focal = 0.5 * vh / Math.tan(0.5 * cam.fov);
  const r = basis.right, up = basis.up, f = basis.fwd, pos = basis.pos;
  const fy = f[1];
  if (!(Math.abs(fy) > 1e-9)) return null;       // looking along the plane, nothing to draw
  const t0 = -pos[1] / fy;
  const m = {
    focal, vw, vh, posY: pos[1],
    ctrX: pos[0] + t0 * f[0],
    ctrZ: pos[2] + t0 * f[2],
    ax: f[0] * r[1] / fy - r[0], bx: f[0] * up[1] / fy - up[0],
    az: f[2] * r[1] / fy - r[2], bz: f[2] * up[1] / fy - up[2],
    wr: r[1], wu: up[1], wf: fy,
  };
  // One pixel, in plane units, at the centre of the frame. This is the rTol every bailout in the
  // shader is measured against, exactly as a cell's half size is on the CPU.
  m.pix = Math.abs(m.posY) * Math.hypot(m.ax, m.az) / (focal * Math.abs(fy));
  return m;
}

// The same map the shader applies, in doubles, for a pixel in the renderer's own screen
// coordinates: x right, y down from the top, which is what place() writes into pool.sx and pool.sy.
// The shader flips y because gl_FragCoord counts from the bottom.
function directPlaneAt(m, sx, sy) {
  const u = (sx - 0.5 * m.vw) / m.focal, v = (0.5 * m.vh - sy) / m.focal;
  const W = m.wr * u + m.wu * v + m.wf;
  return [m.ctrX + m.posY * (m.ax * u + m.bx * v) / W,
          m.ctrZ + m.posY * (m.az * u + m.bz * v) / W];
}

// The analytic interior test is per pixel, in double single, inside pertField; see dsInMainBody
// in src/61_pert.js. It needs the plane coordinate to full precision, which a per frame CPU verdict
// over the frame corners cannot supply.

// The iteration cap for the next frame, in the same two regimes as planeRoot. Moving, the cap is
// hill climbed against measured frame time, because EXT_disjoint_timer_query is off in Chrome by
// default; under vsync the frame time is pinned at 16.7 ms and says nothing, so the climb is one
// sided: raise until the frame time comes off the vsync floor, then back off. Still, it goes to the
// target.
function directCap(moving, lastMs, ss) {
  const lz = Math.max(0, logZoom());
  const want = Math.max(80, Math.min(200000, cfg.iters * (1 + 0.45 * lz) * 16));
  // The work ceiling bounds orbits, and a supersampled pixel runs ss*ss of them, so the ceiling on
  // the iteration cap comes down by the same factor.
  const ceil = (moving ? DIR_WORK_MOVE : DIR_WORK)
    / Math.max(fboW * fboH * ss * ss, 1);
  dirWant = Math.min(want, ceil);
  if (moving) {
    if (lastMs > 26) dirCap = Math.max(200, dirCap * 0.72);
    else if (dirCap < dirWant) dirCap = Math.min(dirWant, dirCap * 1.3);
    else dirCap = dirWant;
  } else {
    // Taken outright rather than climbed to: a still frame is a pass of as many bands as the work
    // needs, so the frame rate is bounded by the band and not by the cap.
    dirCap = dirWant;
  }
  return Math.round(dirCap);
}

function drawDirect(vw, vh, moving, lastMs) {
  // Not linked yet: clear to the background and let the frame loop keep running.
  if (!progDirect && !initDirect()) { dirClear(vw, vh); return; }
  const P = PRESETS[cfg.preset];
  const m = directMap(vw, vh);
  if (!m) return;
  const off = dirResize(vw, vh);
  // The pass. Moving, the frame is one draw at whatever cap fits inside a vsync. Still, it is a
  // pass at the converged cap, one horizontal band per frame into the off screen target, which is
  // presented every frame; a new pass starts when the camera moves or the target cap changes. Bands
  // go into the texture the screen is reading, so a frame in flight is part old pass and part new
  // and the seam is a step in how far the boundary layer has resolved. Supersampling is a property
  // of the pass, so it is fixed here and used for the cap, the band count and the shader alike.
  const ss = moving ? 1 : Math.max(1, Math.min(4, Math.round(DIR_SS)));
  dirSS = ss;
  const cap = directCap(moving, lastMs, ss);
  // The cap a whole frame in one draw can afford. Used while moving, and for the first still frame
  // after a move so the screen is not empty while the pass runs: at zoom 1e6 the pass cap of 88 800
  // as one draw is over two seconds, which is a display driver reset.
  const cheap = Math.max(200, Math.round(DIR_WORK_MOVE / Math.max(vw * vh, 1)));
  // Orbits per frame, for the band arithmetic. ss*ss is the worst case rather than the mean, since
  // only the boundary layer supersamples: too many bands costs frames, too few costs a driver
  // reset.
  const work = vw * vh * ss * ss;
  let whole, passCap;
  if (moving || !off) {
    whole = true; passCap = moving ? cap : cheap;
    dirBand = 0; dirDone = false; dirPassCap = 0;
  } else if (!dirHave) {
    // Nothing on the target yet: lay down a whole frame before banding, or the screen shows one
    // band of an undefined texture. At the cheap cap, unless the whole pass already fits in one
    // band.
    whole = true;
    dirPassCap = cap;
    dirBands = Math.max(1, Math.min(DIR_BAND_MAX, Math.ceil(work * cap / DIR_BAND_WORK)));
    passCap = dirBands === 1 ? cap : cheap;
    dirDone = dirBands === 1;
  } else {
    whole = false;
    if (dirBand === 0 && (!dirDone || cap !== dirPassCap)) {
      dirPassCap = cap;
      dirBands = Math.max(1, Math.min(DIR_BAND_MAX,
        Math.ceil(work * dirPassCap / DIR_BAND_WORK)));
      dirDone = false;
    }
    passCap = dirPassCap;
    // Converged and unchanged: present what is there and do no work.
    if (dirDone) { dirIters = passCap; dirPresent(vw, vh); return; }
  }
  dirIters = passCap;
  const pw = Math.max(2, Math.min(8, Math.round(cfg.power)));
  const julia = P.julia ? 1 : 0;
  const cX = splitDS(m.ctrX), cZ = splitDS(m.ctrZ);
  const jx = splitDS(cfg.cx), jy = splitDS(cfg.cy);

  // Always into the off screen target when there is one, and present from it: a read back from the
  // default framebuffer is the one blit whose source format this file does not control.
  gl.bindFramebuffer(gl.FRAMEBUFFER, off ? dirFbo : null);
  gl.disable(gl.BLEND);
  gl.disable(gl.DEPTH_TEST);
  gl.useProgram(progDirect);
  gl.uniform2f(dirUni.uCtrX, cX[0], cX[1]);
  gl.uniform2f(dirUni.uCtrY, cZ[0], cZ[1]);
  gl.uniform2f(dirUni.uCX, jx[0], jx[1]);
  gl.uniform2f(dirUni.uCY, jy[0], jy[1]);
  gl.uniform2f(dirUni.uVp, vw, vh);
  gl.uniform1f(dirUni.uFocal, m.focal);
  gl.uniform2f(dirUni.uA, m.ax, m.az);
  gl.uniform2f(dirUni.uB, m.bx, m.bz);
  gl.uniform3f(dirUni.uWrow, m.wr, m.wu, m.wf);
  gl.uniform1f(dirUni.uPosY, m.posY);
  gl.uniform1f(dirUni.uPix, m.pix);
  gl.uniform1i(dirUni.uJulia, julia);
  gl.uniform1i(dirUni.uPower, pw);
  gl.uniform1i(dirUni.uMaxIter, passCap);
  gl.uniform1i(dirUni.uSS, ss);
  gl.uniform1i(dirUni.uDbg, DIR_DBG);
  gl.uniform1f(dirUni.uNuInv, 1 / (P.nuCycle || 4.5));
  palUpload();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, palTex);
  gl.uniform1i(dirUni.uPalTex, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, palIntTex);
  gl.uniform1i(dirUni.uPalIntTex, 1);
  gl.activeTexture(gl.TEXTURE0);
  gl.uniform3f(dirUni.uPalMean, palMean[0], palMean[1], palMean[2]);
  gl.uniform3f(dirUni.uLake, lakeCol[0], lakeCol[1], lakeCol[2]);
  // Rows this call owns: the whole frame while moving or before the target is filled, else a band.
  const b0 = whole ? 0 : Math.floor(vh * dirBand / dirBands);
  const b1 = whole ? vh : Math.floor(vh * (dirBand + 1) / dirBands);
  gl.enable(gl.SCISSOR_TEST);
  const t1 = performance.now();
  for (let s = 0; s < DIR_STRIPS; s++) {
    const y0 = b0 + Math.floor((b1 - b0) * s / DIR_STRIPS);
    const y1 = b0 + Math.floor((b1 - b0) * (s + 1) / DIR_STRIPS);
    if (y1 <= y0) continue;
    gl.viewport(0, 0, vw, vh);
    gl.scissor(0, y0, vw, y1 - y0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.flush();
  }
  gl.disable(gl.SCISSOR_TEST);
  // Wall time of the submission, not of the work: the only signal available without a timer query,
  // and a submission that blocks has filled the driver's queue.
  dirMs = performance.now() - t1;
  if (!off) return;                      // no target: that draw went straight to the screen
  if (whole) { dirHave = true; dirBand = 0; }
  else if (dirBand + 1 < dirBands) dirBand++;
  else { dirBand = 0; dirDone = true; }
  dirPresent(vw, vh);
}

// The background alone, for the frames before the orbit shader has finished linking.
function dirClear(vw, vh) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, vw, vh);
  gl.disable(gl.SCISSOR_TEST);
  gl.clearColor(Math.pow(bg0[0], 1 / 2.2), Math.pow(bg0[1], 1 / 2.2), Math.pow(bg0[2], 1 / 2.2), 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

// The only path to the default framebuffer while the camera holds still.
function dirPresent(vw, vh) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, vw, vh);
  gl.disable(gl.BLEND);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.SCISSOR_TEST);
  gl.useProgram(progBlit);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, dirTex);
  gl.uniform1i(blitUni.uTex, 0);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

// Click to dive is gone from every path; see the pointerup handler in src/70_ui.js.

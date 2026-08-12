/* ================================ WebGL ================================= */
// The rasterizer. Two vertex shaders over one instance buffer, general (13 floats an instance) and
// flat (7), a shared splat fragment shader, a tone map over the accumulation buffer, and the
// magnifier squares, which redraw the same splats at a larger focal length. Instances and the
// reprojection terms uK, uO, uRc come from the cut in src/50_cut.js. Pixel coordinates have their
// origin at the bottom left. The shaders are template literals, so no backtick may appear in one.
const VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aCorner;
layout(location=1) in vec3 aPos;
layout(location=2) in vec3 aCovA;   // xx xy xz
layout(location=3) in vec3 aCovB;   // yy yz zz
layout(location=4) in vec3 aCol;
layout(location=5) in float aW;
uniform mat3 uR;           // world to eye, current camera
uniform mat3 uRc;          // the CUT's own frame to eye. Equal to uR except in
                           // the two or three frames after a rebase, when the
                           // scene has been re-expressed in a child piece's
                           // coordinates and the cut on screen predates it.
uniform vec2 uFocal;
uniform vec2 uVp;
uniform float uGain;
uniform float uFilter;
uniform float uFog;
uniform vec3 uFogCol;
uniform float uK;          // rescale from the cut's camera to the current one
uniform vec3 uO;           // and offset
uniform float uSig2;       // splat size, squared
uniform float uKnorm;      // super Gaussian area normalization
uniform float uExtent;     // support radius, in sigma
uniform float uEnergy;     // 1 = emissive measure, 0 = opaque surface
uniform float uSlab;       // depth prepass: push depth back this many sigma_z
uniform float uFadePx;     // fade a measure's splat out above this screen sigma
uniform float uZFar;       // the depth of field limit, where the cut culls
uniform vec2 uDepth;       // 1/log(zf/zn), log(zn)
// A CONSTANT SCREEN OFFSET, in pixels, added after projection. Zero for every normal frame. The
// magnifier that once lived in the Gaussian view was one of these; it is kept because the flat
// and general paths must agree about the projection and a zero here costs nothing.
uniform vec2 uShift;
out vec3 vConic;
out vec2 vOff;
out vec3 vCol;
out float vPeak;
void main() {
  // The splat carries the cut's rotation, the camera offset carries the current
  // one. See reprojection() in src/50_cut.js for the derivation.
  vec3 pc = uRc * (aPos * uK) - uR * uO;
  float z = -pc.z;
  if (z < 1e-6) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vConic = vec3(1.0, 0.0, 1.0); vOff = vec2(1e4); vCol = vec3(0.0); vPeak = 0.0;
    return;
  }
  float k2 = uK * uK * uSig2;
  mat3 S = mat3(aCovA.x, aCovA.y, aCovA.z,
                aCovA.y, aCovB.x, aCovB.y,
                aCovA.z, aCovB.y, aCovB.z) * k2;
  mat3 Sc = uRc * S * transpose(uRc);         // covariance in eye space
  float iz = 1.0 / z;
  vec3 j0 = vec3(uFocal.x * iz, 0.0, uFocal.x * pc.x * iz * iz);   // d(pixel u)/d(pc)
  vec3 j1 = vec3(0.0, uFocal.y * iz, uFocal.y * pc.y * iz * iz);   // d(pixel v)/d(pc)
  float c00 = dot(j0, Sc * j0);
  float c01 = dot(j0, Sc * j1);
  float c11 = dot(j1, Sc * j1);
  // Convolve with the pixel footprint. That is the whole anti-aliasing story
  // here: the sampling rate is known exactly, so clamp straight to it.
  float fr = uFilter * uFilter;
  c00 += fr; c11 += fr;
  float det = max(c00 * c11 - c01 * c01, 1e-14);
  float att = exp(-uFog * max(0.0, z - 1.0));
  // A measure is emissive, so distance dims it and the peak is set so the
  // integral over the image equals the branch weight, which makes brightness
  // independent of dilation and therefore of zoom. A surface is not: distance
  // must fade it toward the far colour without making it transparent, and its
  // peak is an opacity, independent of how large it lands on screen.
  if (uEnergy > 0.5) {
    // Refinement caps the footprint, but the budget and the depth limit can always
    // win, so whatever is left over has to LEAVE smoothly rather than pop. Fade over
    // one octave above the cap. This is the only place a splat's light is not
    // conserved, and it is deliberate: an oversized splat is a coarse stand-in for
    // structure that is not being drawn, and a faint wash covering half the screen is
    // worse than nothing there. Surfaces are exempt, see uEnergy.
    float sp = sqrt(max(c00, c11));
    float fade = 1.0 - smoothstep(uFadePx, uFadePx * 2.0, sp);
    // AND a taper to zero at the depth of field limit, which the surface path below has
    // always had and this one did not. The fog term alone is exp(-fog (z-1)), so at the
    // limit it is exp(-8) = 3.4e-4 of the near field and the cut then culls the splat
    // outright. That reads as a fade that goes soft and then stops, for a reason worth
    // writing down: the frame is shown through a 1/2.2 encode, so a factor of 3.4e-4 in
    // linear light is still 3.6 percent of full displayed brightness, and a step from 3.6
    // percent to nothing against a background that is itself only 2 percent is visible.
    //
    // Measured, both directions. Moving the cull twice and four times further out changes
    // the frame by 0.0000 of full range, so the light past the limit really is nothing and
    // the cull is not the problem. The light in the last half of the depth of field is NOT
    // nothing, and this is what lands it on zero smoothly, with zero slope, instead of
    // chopping it.
    fade *= 1.0 - smoothstep(uZFar * 0.55, uZFar, z);
    vPeak = aW * uGain * att * uKnorm * inversesqrt(det) * fade;
    vCol = aCol;
  } else {
    // A surface is culled at the depth of field limit, and an opaque surface that
    // simply STOPS leaves a hard edge across the frame wherever the fog colour does
    // not happen to equal the background gradient at that height. Fading the opacity
    // out over the last quarter of the range dissolves it into the background instead,
    // which is the same "leave smoothly rather than pop" the measure path gets above.
    vPeak = aW * uGain * (1.0 - smoothstep(uZFar * 0.72, uZFar, z));
    vCol = mix(uFogCol, aCol, att);
  }
  vConic = vec3(c11, -c01, c00) / det;
  float rx = min(uExtent * sqrt(c00), uVp.x);
  float ry = min(uExtent * sqrt(c11), uVp.y);
  vOff = aCorner * vec2(rx, ry);
  // The prepass writes the front surface pushed BACK, and the colour pass then keeps
  // everything in front of that. The push has to be at least a resolvable amount of
  // depth, or a surfel whose eye-space z variance is small rejects ITSELF: it writes
  // its own depth and then fails its own LESS test. A flat lake seen from above is
  // exactly that case, and it took the terrain to a completely black frame with 250 000
  // splats sitting in the buffer. The relative term guarantees the margin whatever the
  // covariance does; the slab is what actually sets the thickness.
  float zd = z * 1.002 + uSlab * sqrt(max(Sc[2][2], 0.0));
  float ndc = (log(max(zd, 1e-7)) - uDepth.y) * uDepth.x * 2.0 - 1.0;
  vec2 ctr = vec2(uFocal.x * pc.x * iz, uFocal.y * pc.y * iz) + uShift;
  gl_Position = vec4((ctr + vOff) / (0.5 * uVp), clamp(ndc, -1.0, 1.0), 1.0);
}`;

// The same projection for a flat cut, from seven floats an instance instead of thirteen. A plane
// cell is a disc at y = uPlaneY, so its covariance is diag(r^2/3, (0.06 r)^2, r^2/3) and the single
// scalar `aRad` determines it. The thin term across the plane must stay nonzero: a disc has zero
// variance normal to itself, and a degenerate covariance has no conic inverse. See FLAT_FLOATS.
const FLAT_VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aCorner;
layout(location=1) in vec2 aPos2;    // x and z in the cut's own normalized frame
layout(location=2) in float aRad;    // the cell's half width, same frame
layout(location=3) in vec3 aCol;
layout(location=4) in float aW;
uniform mat3 uR;
uniform mat3 uRc;
uniform vec2 uFocal;
uniform vec2 uVp;
uniform float uFilter;
uniform float uFog;
uniform vec3 uFogCol;
uniform float uK;
uniform vec3 uO;
uniform float uSig2;
uniform float uExtent;
uniform float uZFar;
uniform vec2 uDepth;
uniform vec2 uShift;                 // see the note on uShift in VS
uniform float uPlaneY;               // the y every splat in this cut shares
out vec3 vConic;
out vec2 vOff;
out vec3 vCol;
out float vPeak;
void main() {
  vec3 aPos = vec3(aPos2.x, uPlaneY, aPos2.y);
  vec3 pc = uRc * (aPos * uK) - uR * uO;
  float z = -pc.z;
  if (z < 1e-6) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vConic = vec3(1.0, 0.0, 1.0); vOff = vec2(1e4); vCol = vec3(0.0); vPeak = 0.0;
    return;
  }
  // THE COVARIANCE IS RANK ONE PLUS A MULTIPLE OF THE IDENTITY, and writing it that way removes
  // two 3x3 matrix products a vertex. This is the honest answer to "is this a 3D Gaussian with a
  // wasted third component": the STORAGE has been two dimensional for a while, seven floats an
  // instance, but the projection was still building a 3x3, rotating it into eye space with
  // uRc S uRc^T and contracting it twice, which is fifty four multiplies to express something
  // with two degrees of freedom.
  //
  //   S = r^2 (v I + (t2 - v) e_y e_y^T),  so  Sc = uRc S uRc^T = r^2 (v I + (t2 - v) n n^T)
  //
  // with n = uRc e_y, the plane's normal in eye space, which is the second COLUMN of uRc and is
  // therefore free. The identity part contracts to plain dot products of the Jacobian rows with
  // themselves, and those have structure too: with j0 = f/z (1, 0, sx) and j1 = f/z (0, 1, sy),
  // where sx and sy are the projected coordinates, every entry is two multiplies.
  float iz = 1.0 / z;
  float ex = uFocal.x * iz, ey = uFocal.y * iz;
  float sx = pc.x * iz, sy = pc.y * iz;
  vec3 nrm = uRc[1];                            // uRc * (0,1,0): the plane's normal in eye space
  float A = uK * uK * uSig2 * aRad * aRad;
  const float V = 1.0 / 3.0;                    // the cell's own second moment, over r^2
  const float DT = 0.0036 - V;                  // (0.06 r)^2 minus it: the thin axis across the plane
  float n0 = nrm.x + nrm.z * sx;
  float n1 = nrm.y + nrm.z * sy;
  float c00 = A * ex * ex * (V * (1.0 + sx * sx) + DT * n0 * n0);
  float c01 = A * ex * ey * (V * (sx * sy)       + DT * n0 * n1);
  float c11 = A * ey * ey * (V * (1.0 + sy * sy) + DT * n1 * n1);
  float fr = uFilter * uFilter;
  c00 += fr; c11 += fr;
  float det = max(c00 * c11 - c01 * c01, 1e-14);
  float att = exp(-uFog * max(0.0, z - 1.0));
  // A plane cut is drawn by the SURFACE model, not the emissive one, and getting this wrong is
  // invisible in code and glaring on screen. built.surface is true for a plane object, so the
  // general shader takes its uEnergy = 0 branch: the peak is an OPACITY, independent of how large
  // the splat lands, the gain is 1 rather than vw*vh*gain, the colour is mixed toward the far
  // colour by distance, and the screen size fade is deliberately not applied because a surface
  // that thins out with size stops covering. Only that branch is kept here; there is no uEnergy
  // because a plane cut is never the other kind.
  vPeak = aW * (1.0 - smoothstep(uZFar * 0.72, uZFar, z));
  vCol = mix(uFogCol, aCol, att);
  vConic = vec3(c11, -c01, c00) / det;
  float rx = min(uExtent * sqrt(c00), uVp.x);
  float ry = min(uExtent * sqrt(c11), uVp.y);
  vOff = aCorner * vec2(rx, ry);
  float ndc = (log(max(z * 1.002, 1e-7)) - uDepth.y) * uDepth.x * 2.0 - 1.0;
  vec2 ctr = vec2(uFocal.x * pc.x * iz, uFocal.y * pc.y * iz) + uShift;
  gl_Position = vec4((ctr + vOff) / (0.5 * uVp), clamp(ndc, -1.0, 1.0), 1.0);
}`;

// Footprint: a super Gaussian exp(-0.5 (s q)^(beta/2)) with q the Mahalanobis form. beta = 2 is the
// ordinary Gaussian, higher beta flattens the top and steepens the shoulder, and s holds the second
// moment equal to the covariance so the shape sharpens without the splat shrinking.
const FS = `#version 300 es
precision highp float;
in vec3 vConic;
in vec2 vOff;
in vec3 vCol;
in float vPeak;
uniform float uShape;
uniform float uBeta2;
uniform float uCore;
// SHOW THE GAUSSIANS. 0 is off and costs one compare. Above zero it is the half width of the
// contour line IN PIXELS, and pixels is the whole of the fix recorded below.
uniform float uRing;
uniform vec3 uRingCol;
out vec4 outColor;
void main() {
  float q = vConic.x * vOff.x * vOff.x + 2.0 * vConic.y * vOff.x * vOff.y
          + vConic.z * vOff.y * vOff.y;
  float e = 0.5 * pow(max(uShape * q, 1e-9), uBeta2);
  float k = exp(-e);
  // HOW FAST THE KERNEL CHANGES ACROSS ONE FRAGMENT. Taken here, at the top, and not down in the
  // branch that uses it: a derivative asked for after a discard is undefined by the spec, and this
  // shader discards twice below. The test is on a UNIFORM, which is the one kind of control flow a
  // derivative may be taken inside, so the ordinary render still pays nothing for it.
  float g = uRing > 0.0 ? max(fwidth(k), 1e-7) : 1.0;
  if (e > 6.91) discard;
  if (k < uCore) discard;
  float a = vPeak * k;
  vec3 c = vCol;
  if (uRing > 0.0) {
    // SHOW THE GAUSSIANS: each splat drawn as its own one sigma ellipse and nothing else.
    //
    // The contour is a level set of the KERNEL rather than of the Mahalanobis form, which is the
    // same ellipse and is the right way to say it: for a Gaussian the one sigma ellipse is
    // k = exp(-1/2) = 0.6065 whatever the covariance is, and that stays true for the super Gaussian
    // at any exponent, so this needs to know nothing about uShape or uBeta2.
    //
    // The OUTLINE is neutral and the INTERIOR carries the splat's own colour. The outline has to
    // be one colour everywhere or a splat is legible only where it happens to differ in hue from
    // whatever it lies on, and the outlines are what carry the geometry: how large each ellipse
    // is, which way it points, and how they overlap. The interior has no such constraint. It is
    // already being drawn so that overlaps read at all, and drawing it in the object's colour
    // costs nothing and says which piece each ellipse belongs to.
    //
    // The two are mixed by ALPHA rather than added, because the tone map divides accumulated
    // colour by accumulated alpha: emitting vCol*aFill + uRingCol*aRing against a total alpha of
    // aFill + aRing is what makes the ring read as white over a tinted body instead of the whole
    // splat washing to the average of the two.
    //
    // No backticks in this comment, and that is not a style rule. This whole shader is a JS
    // template literal, so one backtick here ends the string and the file stops parsing.
    //
    // THE LINE IS A WIDTH IN PIXELS, and it used to be a width in units of the kernel's own value.
    // That difference is the whole of the aliasing this mode had.
    //
    // A contour width stated in kernel units is a width that SCALES WITH THE SPLAT. Near the one
    // sigma contour the kernel falls by 0.6065 per sigma, so a line of w in kernel units is
    // w * R / 0.6065 pixels wide on a splat whose one sigma radius is R pixels. That was fine while
    // this mode drew ellipses twenty eight pixels across: at 1/26 kernel units the line came out
    // 0.9 px and looked like a line. The same constant on the six pixel ellipses this mode draws
    // now is 0.19 px, which is a line thinner than the grid it is being sampled on, so it turns
    // into a dotted crawl along every boundary. That is the pixelation, and no amount of tuning the
    // constant fixes it, because there is no one constant: the splats are not all the same size.
    //
    // fwidth(k) is how much the kernel changes between one fragment and the next, so (k - contour)
    // divided by it is the signed distance to the contour measured IN FRAGMENTS. A line of a fixed
    // number of those is the same width everywhere, on every splat, at every zoom, and inside the
    // magnified corner as well, which is why the caller no longer scales this by the magnification.
    // The falloff being smooth over that distance is what antialiases it: a fragment half a pixel
    // off the contour gets half the weight, which is exactly what a good line does.
    //
    // The INTERIOR fades out across one pixel at the contour for the same reason. It used to end at
    // a hard discard, and a hard edge on a tinted disc is a jagged silhouette however thin the tail
    // behind it is. The tail itself still has to go, or a few hundred overlapping Gaussians pile up
    // into a white fog with the ellipses buried in it; that is what the discard below does, three
    // line widths outside the contour, where there is nothing left to draw.
    //
    // AND THE LINE IS NEVER DRAWN NARROWER THAN THE GRID, it is drawn DIMMER instead. Below about
    // half a pixel a Gaussian line is narrower than the spacing of the samples taken from it, so
    // how bright it comes out depends on where the pixel centres happen to fall across it, which
    // is the speckle all over again in a subtler form. Widening it to the floor and scaling the
    // amplitude by the same ratio keeps the LIGHT the line carries the same, which is what the eye
    // integrates anyway, and makes every value of the width setting mean something.
    float dp = (k - 0.60653) / g;
    float w = max(uRing, 0.55);
    if (dp < -3.0 * w) discard;
    float t = dp / w;
    float ring = (uRing / w) * exp(-t * t);
    float fill = clamp(dp + 0.5, 0.0, 1.0);
    float aFill = vPeak * ${GAUSS_FILL} * k * fill;
    float aRing = vPeak * 2.4 * ring;
    a = aFill + aRing;
    c = (vCol * aFill + uRingCol * aRing) / max(a, 1e-6);
  }
  outColor = vec4(c * a, a);
}`;

const TONE_VS = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
  vUv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

// Two opacity models over the same accumulation buffer, both order independent, so there is no
// sort anywhere.
//
//   solid  a = 1 - exp(-T)         emission absorption at a uniform emission to extinction ratio,
//                                  for which the density weighted mean colour is exact. Saturating
//                                  past T of about three is what stops a two dimensional measure,
//                                  which projects to twice the density down its symmetry axes,
//                                  changing brightness with the viewing direction.
//
//   glow   b = log(1+T)/log(1+W)   Draves log density, for a measure whose density is unbounded.
const TONE_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform float uOptical;
uniform float uGlow;
uniform float uWhite;
uniform vec3 uBg0;
uniform vec3 uBg1;
out vec4 outColor;
void main() {
  vec4 acc = texture(uTex, vUv);
  float T = max(acc.a, 0.0) * uOptical;
  vec3 mean = acc.rgb / max(acc.a, 1e-12);
  float a = 1.0 - exp(-T);
  float g = min(log(1.0 + T) / log(1.0 + uWhite), 6.0);
  float b = mix(a, g, uGlow);
  vec3 c = mean * b;
  c += mix(uBg0, uBg1, vUv.y) * (1.0 - clamp(a, 0.0, 1.0));
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = max(mix(vec3(l), c, 1.14), vec3(0.0));      // hold saturation in the cores
  outColor = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
}`;

let gl, canvas, progSplat, progTone, progFlat, progRect, vao, vaoFlat, instBuf, fbo, fboTex, fboDepth;
let fboW = 0, fboH = 0, fboInternal = 0;
const uni = {};

function compile(src, type) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function link(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(vs, gl.VERTEX_SHADER));
  gl.attachShader(p, compile(fs, gl.FRAGMENT_SHADER));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

/* ------------------------- linking without stalling ---------------------- */
// The batched escape time field program and the per pixel one carry the whole double single orbit
// and take about a second each to link, on the main thread: `linkProgram` returns at once but the
// first query of its result blocks. KHR_parallel_shader_compile replaces that query with a pollable
// flag, and until it is true nothing may touch the program, not LINK_STATUS, not
// getUniformLocation, not useProgram, or the extension has bought nothing. Without the extension
// this degrades to a blocking link.
let parExt;
function linkAsync(vs, fs) {
  if (parExt === undefined) parExt = gl.getExtension('KHR_parallel_shader_compile') || null;
  const p = gl.createProgram();
  gl.attachShader(p, compile(vs, gl.VERTEX_SHADER));
  gl.attachShader(p, compile(fs, gl.FRAGMENT_SHADER));
  gl.linkProgram(p);
  return { p, ext: parExt };
}
// True once the program may be used. Costs one non blocking query a frame while it is false.
function linkReady(h) {
  if (!h) return false;
  if (h.done) return true;
  if (h.ext && !gl.getProgramParameter(h.p, h.ext.COMPLETION_STATUS_KHR)) return false;
  if (!gl.getProgramParameter(h.p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(h.p));
  h.done = true;
  return true;
}

// Solid rectangles in pixel coordinates, for the frames around the magnifier squares and around
// the regions they show. Four vertices from gl_VertexID, no buffer. Origin bottom left.
const RECT_VS = `#version 300 es
precision highp float;
uniform vec4 uRect;                  // x0, y0, x1, y1 in pixels
uniform vec2 uVp;
void main() {
  vec2 c = vec2((gl_VertexID == 1 || gl_VertexID == 3) ? uRect.z : uRect.x,
                (gl_VertexID >= 2) ? uRect.w : uRect.y);
  gl_Position = vec4(c / uVp * 2.0 - 1.0, 0.0, 1.0);
}`;

const RECT_FS = `#version 300 es
precision highp float;
uniform vec4 uCol;
out vec4 outColor;
void main() { outColor = uCol; }`;

const SPLAT_UNI = ['uR', 'uRc', 'uFocal', 'uVp', 'uGain', 'uFilter', 'uFog', 'uFogCol', 'uK', 'uO',
  'uSig2', 'uKnorm', 'uExtent', 'uEnergy', 'uSlab', 'uDepth', 'uShape', 'uBeta2', 'uCore', 'uFadePx',
  'uZFar', 'uRing', 'uRingCol', 'uShift'];
// The flat program's own set. No uGain, uKnorm, uEnergy or uSlab: a plane cut always takes the
// surface model with no prepass, so those are constants in FLAT_VS. Fragment uniforms added below.
const FLAT_UNI = ['uR', 'uRc', 'uFocal', 'uVp', 'uFilter', 'uFog', 'uFogCol', 'uK', 'uO', 'uSig2',
  'uExtent', 'uDepth', 'uZFar', 'uPlaneY', 'uShift'];
const flatUni = {};
const rectUni = {};

function initGL(cv) {
  canvas = cv;
  gl = cv.getContext('webgl2', { antialias: false, alpha: false, depth: false, powerPreference: 'high-performance' });
  if (!gl) return false;
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  stats.gpu = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : 'not reported';
  progSplat = link(VS, FS);
  progFlat = link(FLAT_VS, FS);            // same fragment shader: the footprint is unchanged
  progTone = link(TONE_VS, TONE_FS);
  progRect = link(RECT_VS, RECT_FS);
  for (const k of ['uRect', 'uVp', 'uCol']) rectUni[k] = gl.getUniformLocation(progRect, k);
  for (const k of SPLAT_UNI) uni[k] = gl.getUniformLocation(progSplat, k);
  for (const k of FLAT_UNI) flatUni[k] = gl.getUniformLocation(progFlat, k);
  for (const k of ['uShape', 'uBeta2', 'uCore', 'uRing', 'uRingCol']) {
    flatUni[k] = gl.getUniformLocation(progFlat, k);
  }
  for (const k of ['uTex', 'uOptical', 'uGlow', 'uWhite', 'uBg0', 'uBg1']) {
    uni[k] = gl.getUniformLocation(progTone, k);
  }

  vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  instBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
  gl.bufferData(gl.ARRAY_BUFFER, MAXBUDGET * FLOATS * 4, gl.DYNAMIC_DRAW);
  const stride = FLOATS * 4;
  for (const [loc, size, off] of [[1, 3, 0], [2, 3, 12], [3, 3, 24], [4, 3, 36], [5, 1, 48]]) {
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, off);
    gl.vertexAttribDivisor(loc, 1);
  }
  gl.bindVertexArray(null);

  // A second VAO over the same instance buffer with the compact layout. The buffer is sized for the
  // 13 float format, so a flat cut uses less of it and nothing is reallocated on an object change.
  vaoFlat = gl.createVertexArray();
  gl.bindVertexArray(vaoFlat);
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
  {
    const fs = FLAT_FLOATS * 4;
    for (const [loc, size, off] of [[1, 2, 0], [2, 1, 8], [3, 3, 12], [4, 1, 24]]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, fs, off);
      gl.vertexAttribDivisor(loc, 1);
    }
  }
  gl.bindVertexArray(null);

  // Accumulation needs range far past 1.0, so a float target if one is available.
  if (gl.getExtension('EXT_color_buffer_float')) { fboInternal = gl.RGBA16F; stats.format = 'RGBA16F'; }
  else if (gl.getExtension('EXT_color_buffer_half_float')) { fboInternal = gl.RGBA16F; stats.format = 'RGBA16F'; }
  else { fboInternal = gl.RGBA8; stats.format = 'RGBA8, clamped'; }
  fbo = gl.createFramebuffer();
  fboTex = gl.createTexture();
  fboDepth = gl.createRenderbuffer();
  return true;
}

function resizeTargets(w, h) {
  if (w === fboW && h === fboH) return;
  fboW = w; fboH = h;
  for (let attempt = 0; attempt < 2; attempt++) {
    gl.bindTexture(gl.TEXTURE_2D, fboTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, fboInternal, w, h, 0, gl.RGBA,
      fboInternal === gl.RGBA8 ? gl.UNSIGNED_BYTE : gl.HALF_FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindRenderbuffer(gl.RENDERBUFFER, fboDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fboTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, fboDepth);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (ok) return;
    fboInternal = gl.RGBA8;                    // fall back and try once more
    stats.format = 'RGBA8, clamped';
  }
}

// One splat pass into the accumulation buffer. A surface needs the front layer only, or a hill
// behind a ridge blends into it: the prepass writes the front surface pushed back by its own
// thickness and the colour pass accepts what is inside that slab. Order independent either way, no
// sort, and the accumulation stays additive.
//
// `mag` is null for a normal frame. For a magnifier square it is {m, x, y}: scale the focal length
// by m, which magnifies about the eye axis and carries the projected covariance with it, then shift
// by (x, y) pixels to centre the chosen point. The square redraws the splats, it does not resample.
function splatPass(vw, vh, count, mag) {
  const m = mag ? mag.m : 1, shx = mag ? mag.x : 0, shy = mag ? mag.y : 0;
  const focal = m * 0.5 * vh / Math.tan(0.5 * cam.fov);
  // A width in pixels, which the shader measures with a screen space derivative, so a magnifier
  // square gets the same line as the frame without being scaled by m.
  const ring = ringWidth();
  const surface = built.surface, prepass = built.prepass;
  const zf = Math.max(2, 1 + ZF_MUL * 8 / Math.max(cfg.fog, 0.02));
  const zn = 1e-4;
  if (count > 0 && built.flat) {
    // The flat path: seven floats an instance, no prepass, surface model. Kept separate from the
    // general path below because the two disagree about the vertex layout.
    gl.useProgram(progFlat);
    gl.uniformMatrix3fv(flatUni.uR, false, basis.Rgl);
    gl.uniformMatrix3fv(flatUni.uRc, false, repro.Rc);
    gl.uniform2f(flatUni.uFocal, focal, focal);
    gl.uniform2f(flatUni.uVp, vw, vh);
    gl.uniform1f(flatUni.uFilter, 0.5);
    gl.uniform1f(flatUni.uFog, cfg.fog);
    gl.uniform3f(flatUni.uFogCol, bg1[0], bg1[1], bg1[2]);
    gl.uniform1f(flatUni.uK, repro.k);
    gl.uniform3fv(flatUni.uO, repro.o);
    gl.uniform1f(flatUni.uSig2, cfg.sigma * cfg.sigma);
    gl.uniform1f(flatUni.uExtent, kern.extent * FLAT_EXT);
    gl.uniform1f(flatUni.uZFar, zf);
    gl.uniform2f(flatUni.uDepth, 1 / Math.log(zf / zn), Math.log(zn));
    gl.uniform1f(flatUni.uPlaneY, built.planeY);
    gl.uniform1f(flatUni.uShape, kern.shape);
    gl.uniform1f(flatUni.uBeta2, kern.beta * 0.5);
    gl.uniform1f(flatUni.uCore, 0.0);
    gl.uniform1f(flatUni.uRing, ring);
    gl.uniform3f(flatUni.uRingCol, RING_COL[0], RING_COL[1], RING_COL[2]);
    gl.uniform2f(flatUni.uShift, shx, shy);
    gl.bindVertexArray(vaoFlat);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  } else if (count > 0) {
    gl.useProgram(progSplat);
    gl.uniformMatrix3fv(uni.uR, false, basis.Rgl);
    gl.uniformMatrix3fv(uni.uRc, false, repro.Rc);
    gl.uniform2f(uni.uFocal, focal, focal);
    gl.uniform2f(uni.uVp, vw, vh);
    // A magnified measure spreads its weight over m^2 more pixels and the vertex shader divides by
    // the root of the projected determinant, so its peak falls by m^2. Put it back: the square is a
    // diagram of the primitives and has to be as legible as the frame.
    gl.uniform1f(uni.uGain, surface ? 1 : vw * vh * repro.gain * m * m);
    gl.uniform1f(uni.uFilter, 0.5);
    gl.uniform1f(uni.uFog, cfg.fog);
    gl.uniform3f(uni.uFogCol, bg1[0], bg1[1], bg1[2]);
    gl.uniform1f(uni.uK, repro.k);
    gl.uniform3fv(uni.uO, repro.o);
    gl.uniform1f(uni.uSig2, cfg.sigma * cfg.sigma);
    gl.uniform1f(uni.uKnorm, kern.knorm);
    gl.uniform1f(uni.uExtent, kern.extent);
    gl.uniform1f(uni.uEnergy, surface ? 0 : 1);
    // A measure's splat fades out above this screen size, an oversized splat standing in for
    // structure that is not drawn. Every splat in the Gaussian view is oversized on purpose, so the
    // cap rises with the split threshold.
    gl.uniform1f(uni.uFadePx, (cfg.gauss ? GAUSS_CAP : SIZE_CAP) * m);
    gl.uniform1f(uni.uZFar, zf);
    gl.uniform1f(uni.uShape, kern.shape);
    gl.uniform1f(uni.uBeta2, kern.beta * 0.5);
    gl.uniform1f(uni.uRing, ring);
    gl.uniform3f(uni.uRingCol, RING_COL[0], RING_COL[1], RING_COL[2]);
    gl.uniform2f(uni.uShift, shx, shy);
    gl.uniform2f(uni.uDepth, 1 / Math.log(zf / zn), Math.log(zn));
    gl.bindVertexArray(vao);
    if (prepass) {
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.depthMask(true);
      gl.colorMask(false, false, false, false);
      gl.uniform1f(uni.uSlab, 4.0);
      gl.uniform1f(uni.uCore, 0.22);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
      gl.colorMask(true, true, true, true);
      gl.depthMask(false);
      gl.uniform1f(uni.uSlab, 0.0);
      gl.uniform1f(uni.uCore, 0.0);
    } else {
      gl.disable(gl.DEPTH_TEST);
      gl.depthMask(false);
      gl.uniform1f(uni.uSlab, 0.0);
      gl.uniform1f(uni.uCore, 0.0);
    }
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(null);
  }
}

// The tone map, from the accumulation buffer to the screen.
function tonePass(vw, vh) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, vw, vh);
  gl.useProgram(progTone);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, fboTex);
  gl.uniform1i(uni.uTex, 0);
  gl.uniform1f(uni.uOptical, cfg.density);
  gl.uniform1f(uni.uGlow, cfg.glow);
  gl.uniform1f(uni.uWhite, 12);
  gl.uniform3f(uni.uBg0, bg0[0], bg0[1], bg0[2]);
  gl.uniform3f(uni.uBg1, bg1[0], bg1[1], bg1[2]);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

// The i-th magnifier square, top down the right hand edge, in device pixels, origin bottom left.
// Null when there is no room; the frame then shows fewer squares rather than cramped ones, and null
// on the second is the ordinary case on a short window.
function loupeRect(vw, vh, i) {
  const short = Math.min(vw, vh);
  const s = Math.round(Math.max(LOUPE_MIN, Math.min(LOUPE_MAX, LOUPE_FRAC * short)));
  const pad = Math.round(LOUPE_PAD * short);
  const gap = Math.round(LOUPE_GAP * short);
  // Enough ellipses across to read as a field rather than as two blobs. See LOUPE_ROOM.
  if (s < LOUPE_ROOM * LOUPE_MAG[i] * GAUSS_PX) return null;
  // LOUPE_MIN is a floor in absolute pixels, so past this the square is refused rather than shrunk
  // below it. The smallest frame that gets one is about 263 px; the project page never gives the
  // iframe less than 296.
  if (s > short * 0.38) return null;
  // Clear of the panel, tested in CSS pixels because the panel is laid out by the stylesheet. The
  // capture harness has no layout, so a missing clientWidth means the frame's own width.
  if (((canvas && canvas.clientWidth) || vw) < LOUPE_CLEAR) return null;
  // Room for the whole column, and never more than 0.72 of the frame's height in squares.
  if ((i + 1) * s + i * gap + 2 * pad > vh * 0.72) return null;
  return { x: vw - pad - s, y: vh - pad - s - i * (s + gap), s };
}

// One solid rectangle in pixel coordinates, colour premultiplied, so the blend is
// ONE, ONE_MINUS_SRC_ALPHA. `ring` is a rectangular outline of thickness t just inside the given
// box: four rects, the left and right shortened so the corners are not drawn twice, since at alpha
// below one a doubled corner is a visibly brighter dot.
function rectCol(x0, y0, x1, y1, c, a) {
  gl.uniform4f(rectUni.uRect, x0, y0, x1, y1);
  gl.uniform4f(rectUni.uCol, c[0] * a, c[1] * a, c[2] * a, a);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}
function ring(x0, y0, x1, y1, t, c, a) {
  rectCol(x0, y0, x1, y0 + t, c, a);
  rectCol(x0, y1 - t, x1, y1, c, a);
  rectCol(x0, y0 + t, x0 + t, y1 - t, c, a);
  rectCol(x1 - t, y0 + t, x1, y1 - t, c, a);
}

// The aim point projected to a pixel offset from the middle of the frame, the vertex shader's
// projection on one point in world coordinates rather than the cut's: `cam.goal` and `basis.pos`
// are both in the current frame, so no reprojection is involved. Clamped to keep the marked box
// inside the frame, `pad` being its half side plus a margin, because early in a descent the aim can
// be well outside.
function loupeAim(vw, vh, pad) {
  const out = [0, 0];
  const R = basis.R, p = basis.pos, g = cam.goal;
  const d0 = g[0] - p[0], d1 = g[1] - p[1], d2 = g[2] - p[2];
  const z = -(R[6] * d0 + R[7] * d1 + R[8] * d2);          // see updateBasis: row 2 is -forward
  if (!(z > 1e-9)) return out;
  const focal = 0.5 * vh / Math.tan(0.5 * cam.fov);
  const x = focal * (R[0] * d0 + R[1] * d1 + R[2] * d2) / z;
  const y = focal * (R[3] * d0 + R[4] * d1 + R[5] * d2) / z;
  if (!isFinite(x) || !isFinite(y)) return out;
  const lx = Math.max(0, vw * 0.5 - pad), ly = Math.max(0, vh * 0.5 - pad);
  out[0] = Math.max(-lx, Math.min(lx, x));
  out[1] = Math.max(-ly, Math.min(ly, y));
  return out;
}

function drawInset(vw, vh, count) {
  const rects = [];
  for (let i = 0; i < LOUPE_MAG.length; i++) {
    const r = loupeRect(vw, vh, i);
    if (!r) break;
    rects.push(r);
  }
  if (!rects.length) return;
  // The magnified point, as a pixel offset from the middle of the frame. Both squares show the same
  // point; the clamp is against the largest marked region, the weakest magnification's.
  const aim = loupeAim(vw, vh, rects[0].s / (2 * LOUPE_MAG[0]) + 8);
  const px = aim[0], py = aim[1];

  gl.enable(gl.SCISSOR_TEST);
  for (let i = 0; i < rects.length; i++) {
    const r = rects[i], m = LOUPE_MAG[i];
    // The projection scales about the eye axis, so the aim lands at m*(px, py) and the shift is
    // whatever moves that to the middle of the square.
    const cx = r.x + r.s * 0.5 - vw * 0.5, cy = r.y + r.s * 0.5 - vh * 0.5;
    const mag = { m, x: cx - m * px, y: cy - m * py };
    gl.scissor(r.x, r.y, r.s, r.s);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, vw, vh);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    splatPass(vw, vh, count, mag);
    tonePass(vw, vh);                 // still scissored, so it writes the square and nothing else
  }
  gl.disable(gl.SCISSOR_TEST);

  // The marks: a frame around each square, then the region it came from, both a light line between
  // two dark ones, since a single translucent mark at 0.42 alpha vanished on a saturated escape
  // time field. All three rings sit outside the square, clear of the splats it exists to show.
  gl.useProgram(progRect);
  gl.uniform2f(rectUni.uVp, vw, vh);
  gl.enable(gl.BLEND);
  gl.blendEquation(gl.FUNC_ADD);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  const w = Math.max(1, Math.round(vh / 720));
  for (const r of rects) {
    const x1 = r.x + r.s, y1 = r.y + r.s;
    ring(r.x - 3 * w, r.y - 3 * w, x1 + 3 * w, y1 + 3 * w, w, BLACK_COL, 0.85);
    ring(r.x - 2 * w, r.y - 2 * w, x1 + 2 * w, y1 + 2 * w, w, MARK_COL, 1.0);
    ring(r.x - w, r.y - w, x1 + w, y1 + w, w, BLACK_COL, 0.85);
  }
  // One region box per square, drawn the same way. The second magnification is the square of the
  // first, so the boxes nest in that proportion and the three views read as one continued zoom.
  const mx = vw * 0.5 + px, my = vh * 0.5 + py;
  for (let i = 0; i < rects.length; i++) {
    const h = rects[i].s / (2 * LOUPE_MAG[i]);   // half the side of the region that square shows
    ring(mx - h - 2 * w, my - h - 2 * w, mx + h + 2 * w, my + h + 2 * w, w, BLACK_COL, 0.85);
    ring(mx - h - w, my - h - w, mx + h + w, my + h + w, w, MARK_COL, 1.0);
    ring(mx - h, my - h, mx + h, my + h, w, BLACK_COL, 0.85);
  }
  gl.disable(gl.BLEND);
}

// The captions are HTML, positioned over the canvas from the rectangles the squares are drawn in.
// Those are device pixels and a CSS position is in CSS pixels, differing by the resource tier's
// pixel ratio, so both are scaled by clientWidth/vw and clientHeight/vh. The sentence says the mode
// coarsens the cut on purpose, which is why its splat count is an order of magnitude below normal.
let noteKey = '';
function loupeNote(vw, vh, count) {
  const box = el('loupe-note');
  if (!box) return;                                  // canvas only page, e.g. the capture harness
  const on = !!cfg.gauss && !isDirect() && count > 0 && !!loupeRect(vw, vh, 0);
  const key = on ? vw + 'x' + vh + '/' + canvas.clientWidth + 'x' + canvas.clientHeight : '';
  if (key === noteKey) return;                       // only on a resize, not every frame
  noteKey = key;
  box.hidden = !on;
  if (!on) return;
  const sx = canvas.clientWidth / Math.max(vw, 1);
  const sy = canvas.clientHeight / Math.max(vh, 1);
  let last = null;
  for (let i = 0; i < LOUPE_MAG.length; i++) {
    const lab = el('lz' + i);
    if (!lab) continue;
    const r = loupeRect(vw, vh, i);
    lab.hidden = !r;
    if (!r) continue;
    const m = LOUPE_MAG[i];
    lab.textContent = MULT + (m < 10 ? m.toFixed(1) : String(Math.round(m)));
    lab.style.left = (r.x * sx + 4) + 'px';
    lab.style.top = ((vh - r.y - r.s) * sy + 4) + 'px';
    last = r;
  }
  // Set to the square's own width and dropped below 150 CSS pixels, where at 120 it wraps to seven
  // lines of three words. The squares still carry their magnification.
  const cap = el('lcap');
  if (cap) {
    const wide = last && last.s * sx >= 150;
    cap.hidden = !wide;
    if (wide) {
      cap.style.left = (last.x * sx) + 'px';
      cap.style.top = ((vh - last.y) * sy + 5) + 'px';
      cap.style.width = (last.s * sx) + 'px';
    }
  }
}

function draw(vw, vh, count) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.viewport(0, 0, vw, vh);
  gl.clearColor(0, 0, 0, 0);
  gl.clearDepth(1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  splatPass(vw, vh, count);
  tonePass(vw, vh);
  if (cfg.gauss && count > 0) drawInset(vw, vh, count);
  loupeNote(vw, vh, count);
}

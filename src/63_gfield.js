/* ============ the escape time field, in batches, on the GPU =============== */
// Batched evaluation of the escape time field. Only the arithmetic moves to the GPU: the walk that
// decides which cells exist costs under 1 us of the 25 us a cell takes. stepPlane is level
// synchronous and cells within a level are independent, so one level of children is one batch; see
// planeFlush in 52_cut_plane.js. Levels grow about 4x, so the last four hold 99 percent of the
// cells and the top ten are cheaper on the CPU than a batch costs. The readback is synchronous by
// necessity: level n+1's queries need level n's answers, so a fence has nothing to overlap; cost
// per round trip is in knowledge/mandelbrot_speed.md.

// Queries are (dcx, dcy, rTol) and answers are (nu, de, gnu, kind + 8*iters), both RGBA32F, both
// indexed by a linear slot. GF_W is a power of two so the shader forms the slot by shift and or.
const GF_W = 1024, GF_WSHIFT = 10, GF_WMASK = GF_W - 1;
// Largest batch: 512k covers one whole level of a 500 000 splat cut. Anything larger is chunked.
const GF_CAP = 1 << 19;
// Below this the fixed cost of a batch, upload plus draw setup plus flush, exceeds what it saves.
let GF_MIN = 4096;
let gfOn = true;                    // master switch, so `gf=0` measures without it

const GF_VS = `#version 300 es
precision highp float;
void main() {
  vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}`;

const GF_FS = `#version 300 es
precision highp float;
precision highp int;
` + PERT_GLSL + `
uniform sampler2D uQ;        // the queries: dcx, dcy, rTol
uniform int uCount;
out vec4 outColor;
void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  int slot = (t.y << ${GF_WSHIFT}) | t.x;
  if (slot >= uCount) { outColor = vec4(0.0); return; }
  vec3 q = texelFetch(uQ, t, 0).xyz;
  Fld f = pertField(q.xy, q.z);
  // kind, packed as a float, with the same numbering planeCell uses so the CPU side needs no
  // translation: 0 escaped, 1 proven interior, 2 condemned, 3 exhausted. See planeCell.
  float kind = f.why == 2 ? 1.0 : (f.why == 4 ? 0.0 : (f.why == 1 ? 2.0 : 3.0));
  outColor = vec4(f.nu, f.de, f.gnu, kind + 8.0 * float(f.iters));
}`;

let progGF = null, gfQTex = null, gfFbo = null, gfOutTex = null, gfH = 0;
const gfUni = {};
// Query and answer staging, one slot per query, four floats each.
let gfQ = new Float32Array(GF_W * 4 * 4);
let gfA = new Float32Array(GF_W * 4 * 4);
let gfBatches = 0, gfQueries = 0, gfMs = 0;

// Room for `n` queries, the same rectangle gfEvaluate uploads, so the caller can fill gfQ first.
// Grown on demand and never shrunk: a cut asks for the same order of magnitude every frame.
function gfReserve(n) {
  const need = GF_W * Math.max(1, Math.ceil(Math.min(n, GF_CAP) / GF_W)) * 4;
  if (gfQ.length < need) gfQ = new Float32Array(need);
}

// Started, not finished: reading the program before it reports complete is what blocks, so
// initGField polls linkReady and leaves progGF null until then.
let gfLink = null;
function initGField() {
  if (!gfLink) gfLink = linkAsync(GF_VS, GF_FS);
  if (!linkReady(gfLink)) return;
  progGF = gfLink.p;
  for (const k of ['uQ', 'uCount', 'uCtrX', 'uCtrY', 'uCX', 'uCY', 'uJulia', 'uPower',
                   'uMaxIter']) {
    gfUni[k] = gl.getUniformLocation(progGF, k);
  }
  gfQTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, gfQTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gfOutTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, gfOutTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gfFbo = gl.createFramebuffer();
}

// Grow only, to a power of two: reallocating the 1024 by h RGBA32F pair on every call cost over a
// second of the opening build, because respecifying a texture the previous draw still references
// flushes the driver pipeline. Over allocating is free, the draw and the readback both use `h`.
function gfResize(h) {
  if (h <= gfH) return true;
  h = 1 << Math.ceil(Math.log2(h));
  gl.bindTexture(gl.TEXTURE_2D, gfQTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, GF_W, h, 0, gl.RGBA, gl.FLOAT, null);
  gl.bindTexture(gl.TEXTURE_2D, gfOutTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, GF_W, h, 0, gl.RGBA, gl.FLOAT, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, gfFbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, gfOutTex, 0);
  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gfH = ok ? h : 0;
  return ok;
}

// True when the batch path is enabled and GL is up. fboInternal is nonzero whatever format initGL
// picked, so this does not prove a float render target; gfResize is what proves that.
function gfReady() {
  return gfOn && typeof gl !== 'undefined' && gl && fboInternal !== 0;
}

// Evaluate `n` queries. gfQ holds (dcx, dcy, rTol, unused) per slot, gfA receives
// (nu, de, gnu, kind + 8*iters). One upload, one draw, one synchronous readback. iters rides in the
// fourth channel rather than a fifth: kind is 0 to 3 and iters an integer, so kind + 8*iters is
// exact in fp32 to 2 097 152 iterations, ten times job.maxIter's 200 000 ceiling.
function gfEvaluate(n, maxIter, ctrX, ctrZ) {
  if (!progGF) initGField();
  if (n > GF_CAP) return false;
  const h = Math.max(1, Math.ceil(n / GF_W));
  if (!gfResize(h)) return false;
  if (gfA.length < GF_W * h * 4) gfA = new Float32Array(GF_W * h * 4);
  if (gfQ.length < GF_W * h * 4) return false;      // caller skipped gfReserve
  const t0 = performance.now();
  gl.bindTexture(gl.TEXTURE_2D, gfQTex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, GF_W, h, gl.RGBA, gl.FLOAT,
                   gfQ.subarray(0, GF_W * h * 4));
  gl.bindFramebuffer(gl.FRAMEBUFFER, gfFbo);
  gl.viewport(0, 0, GF_W, h);
  gl.disable(gl.BLEND);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.SCISSOR_TEST);
  gl.useProgram(progGF);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, gfQTex);
  gl.uniform1i(gfUni.uQ, 0);
  // Origin the query offsets are measured from, as a double single pair. The caller supplies it
  // with the offsets so the two cannot disagree; ctrZ is the world axis that is the plane's y.
  const cX = splitDS(ctrX), cZ = splitDS(ctrZ);
  const jx = splitDS(job.c0x), jy = splitDS(job.c0y);
  gl.uniform2f(gfUni.uCtrX, cX[0], cX[1]);
  gl.uniform2f(gfUni.uCtrY, cZ[0], cZ[1]);
  gl.uniform2f(gfUni.uCX, jx[0], jx[1]);
  gl.uniform2f(gfUni.uCY, jy[0], jy[1]);
  gl.uniform1i(gfUni.uJulia, job.julia);
  gl.uniform1i(gfUni.uPower, job.power);
  gl.uniform1i(gfUni.uMaxIter, maxIter);
  gl.uniform1i(gfUni.uCount, n);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.readPixels(0, 0, GF_W, h, gl.RGBA, gl.FLOAT, gfA);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.activeTexture(gl.TEXTURE0);
  gfMs += performance.now() - t0;
  gfBatches++; gfQueries += n;
  return true;
}

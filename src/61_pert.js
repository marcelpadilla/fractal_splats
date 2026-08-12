/* =========== the escape time field on the GPU, in double single =========== */
// WebGL 2 has no fp64 and a deep zoom needs fourteen digits of the plane coordinate, so a real is a
// pair of floats: about 43 bits on this driver against a double's 53, reaching zoom 1e9 against the
// CPU path's 1e13, at 87 to 95 G iterations a second against 310 M on one CPU thread. Used by
// src/62_direct.js and src/63_gfield.js; the fields returned match planeField in src/32_plane.js,
// so a GPU verdict and a CPU verdict are interchangeable. Perturbation is rejected on measurement:
// the escape count comes out wrong by a mean of 397 iterations and log2|dz| drifts 1.2 to 1.8 high
// whatever the storage, because these frames are mostly far field and a far field point rebases
// within a few dozen iterations. See knowledge/mandelbrot_speed.md. Read exactAdd first.

const PERT_GLSL = `
uniform vec2 uCtrX, uCtrY;   // the plane coordinate the query offsets are measured from, ds
uniform vec2 uCX, uCY;       // the family parameter, ds
uniform int uJulia;
uniform int uPower;
uniform int uMaxIter;

const float PB2 = 1e12;
const float LPB = 13.8155106;            // log(1e6)
const float NU_MIN = 1.5;
// |dz| passes ten to the twelve at a deep bailout and its square overflows fp32, so the derivative is
// carried as a mantissa and a separate binary exponent. Rescaling by a power of two is exact, and the
// bailout test then runs in log space, where it is also cheaper.
//
// This replaced a CAP on |dz|^2 at 1e24, which was written as an overflow guard and used as a bailout.
// The real condemnation fires at |dz| = 2|z|log|z|/rTol, which at ten to the six and a half is 2.2e12,
// so |dz|^2 is 4.8e24 and the guard fired FIRST, on most of the frame, in a branch that never assigned
// nu. Two thirds of a deep frame came out one flat colour.
const float DM_SCALE = 1e18, DM_DOWN = 2.3283064e-10, DM_DOWN2 = 5.4210109e-20;

// ---- double single ---------------------------------------------------------
// A value is a vec2 (hi, lo) standing for hi + lo.
//
// EVERY TEXTBOOK EXACT ADDITION FAILS ON THIS DRIVER, and it fails silently. The classical two sum
// recovers the rounding error after the fact:
//
//     s = a + b;  v = s - a;  e = (b - v) + (a - (s - v));
//
// which is exact in IEEE arithmetic and is ALGEBRAICALLY ZERO. A compiler that assumes addition is
// associative folds it to nothing, and then the shader keeps running and quietly delivers fp32: a
// correct looking image that has gone blocky, with no error anywhere. Measured on ANGLE over D3D11,
// with every operand arriving through a uniform so nothing could be constant folded, three
// formulations return exactly zero: the plain two sum, Dekker's magnitude ordered version, and the
// same with every intermediate forced through uintBitsToFloat(floatBitsToUint(x)), which the HLSL
// compiler sees straight through.
//
// What survives is not recovering the error at all. Split the addends by MASKING mantissa bits so the
// high halves add with nothing to lose and the remainder is exactly the masked off tail: an exponent
// compare, a shift, a mask, and subtractions of exactly representable numbers. No cancelling
// expression anywhere in it, so nothing to reassociate.
float dsSplitHi(float a) { return uintBitsToFloat(floatBitsToUint(a) & 0xfffff000u); }

// s + e = a + b, exactly. Both addends are truncated so their sum lands on a grid coarse enough that a
// carry into the next exponent cannot round it: the larger loses its lowest mantissa bit and the
// smaller is masked down to a multiple of twice the larger's ulp.
//
// The guard is 24 and NOT 25, and that off by one was worth eighteen bits. An fp32 mantissa is bits 0
// to 22, so the mask only touches mantissa bits while sh is at most 23; at sh = 24 it clears the low
// bit of the EXPONENT, which does not lose precision, it changes the value by a factor of two. sh = 24
// is the case where the small addend sits exactly at the ulp of the large one, which happens
// constantly. Measured on the GPU, the orbit carried 22 bits at sixteen iterations and nothing at all
// by a thousand; with the guard at 24, 43.2 bits at sixteen and 33.9 at a thousand.
void exactAdd(float a, float b, out float s, out float e) {
  bool aBig = abs(a) >= abs(b);
  float x = aBig ? a : b;
  float y = aBig ? b : a;
  int ex = (floatBitsToInt(x) >> 23) & 0xff;
  int ey = (floatBitsToInt(y) >> 23) & 0xff;
  int sh = ex - ey + 1;
  if (sh >= 24 || ey == 0 || ex == 0) { s = x; e = y; return; }
  float yh = uintBitsToFloat(floatBitsToUint(y) & (0xffffffffu << uint(sh)));
  float xh = uintBitsToFloat(floatBitsToUint(x) & 0xfffffffeu);
  s = xh + yh;
  e = (x - xh) + (y - yh);
}
vec2 dsAdd(vec2 a, vec2 b) {
  float s, e;
  exactAdd(a.x, b.x, s, e);
  // e, a.y and b.y all sit at or below the ulp of s, so an ordinary sum of them rounds at the 2^-48
  // level, which is this representation's own floor. One more exact addition then moves whatever
  // belongs in the high word up into it.
  float lo = e + (a.y + b.y);
  float s2, e2;
  exactAdd(s, lo, s2, e2);
  return vec2(s2, e2);
}
vec2 dsSub(vec2 a, vec2 b) { return dsAdd(a, vec2(-b.x, -b.y)); }
vec2 dsMul(vec2 a, vec2 b) {
  float ah = dsSplitHi(a.x), al = a.x - ah;      // exact, both halves
  float bh = dsSplitHi(b.x), bl = b.x - bh;
  float p = a.x * b.x;                            // rounded product
  // ... and its exact error, from the four cross terms. ah*bh, ah*bl, al*bh, al*bl each fit in 24
  // bits, so every one is exact, and ah*bh - p is exact by Sterbenz. This one DOES survive: the
  // compiler cannot prove ah*bh equals a.x*b.x, because ah came out of a bit mask, so there is no
  // algebraic identity for it to cancel.
  float e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  e += a.x * b.y + a.y * b.x;
  float s, r;
  exactAdd(p, e, s, r);
  return vec2(s, r);
}
// Multiplying by two is exact in binary, so it needs no renormalization at all.
vec2 dsTwice(vec2 a) { return vec2(a.x + a.x, a.y + a.y); }

// The result. why uses the same numbering as pfl in src/32_plane.js: 4 escaped, 2 an attracting cycle,
// 1 condemned by the derivative, 3 out of iterations.
struct Fld { float nu; float de; float gnu; int why; int iters; };

// THE ANALYTIC INTERIOR TEST, IN DOUBLE SINGLE, and it used to be a uniform computed on the CPU at
// the frame's centre and four corners because "it needs the coordinate to full precision, and the
// shader has only an fp32 offset". That was true before this file existed. The shader has the
// coordinate to full precision now: px and py below ARE the point, as a pair, to about 43 bits, so
// the same two lines of algebra run here and answer per pixel.
//
// Not an optimization. Measured at ten to the two and a bit on a 1100x1280 frame of the seahorse
// valley, with the frame straddling the cardioid's boundary so the five point CPU test returned
// "mixed" and switched itself off: a band 25 px wide along the inside of the cardioid, and a ring
// round every bulb in the frame, came back OUT OF ITERATIONS and was painted the flat light grey
// that verdict gets. That band is the "weird thick gray bands especially around the black blobs".
// It is not a colour bug and it is not a cap that is too low: an interior point is proven interior
// only when its orbit closes, the multiplier of a component goes to one at that component's own
// boundary, so the number of iterations needed goes to infinity there no matter what the cap is.
// The cardioid and the period two bulb are the two cases where no orbit is needed at all.
//
//   cardioid:  q = (x - 1/4)^2 + y^2,  inside iff q (q + x - 1/4) < y^2 / 4
//   period 2:  (x + 1)^2 + y^2 < 1/16
//
// Both comparisons are made by subtracting in double single and testing the sign of the HIGH word,
// which is the whole point: near the boundary the two sides cancel to many digits, and in fp32 the
// difference would be noise.
bool dsInMainBody(vec2 x, vec2 y) {
  vec2 y2 = dsMul(y, y);
  vec2 xm = dsSub(x, vec2(0.25, 0.0));
  vec2 q = dsAdd(dsMul(xm, xm), y2);
  // 0.25 * y2 is exact: a power of two scales both words with no rounding.
  if (dsSub(dsMul(q, dsAdd(q, xm)), vec2(0.25 * y2.x, 0.25 * y2.y)).x < 0.0) return true;
  vec2 xp = dsAdd(x, vec2(1.0, 0.0));
  return dsSub(dsAdd(dsMul(xp, xp), y2), vec2(0.0625, 0.0)).x < 0.0;
}

// dc is the offset of this point from uCtr, in plane units, added in double single so the sum keeps
// its digits. rTol is the half size of the cell or half a pixel, whichever the caller means; it drives
// the derivative bailout.
Fld pertField(vec2 dc, float rTol) {
  Fld o;
  o.nu = 0.0; o.de = 0.0; o.gnu = 1e30; o.why = 3; o.iters = 0;
  vec2 px = dsAdd(uCtrX, vec2(dc.x, 0.0));
  vec2 py = dsAdd(uCtrY, vec2(dc.y, 0.0));
  // Proven interior, exactly, in about sixty flops instead of an orbit. See dsInMainBody.
  if (uJulia == 0 && uPower == 2 && dsInMainBody(px, py)) { o.why = 2; return o; }
  vec2 zx, zy, cx, cy;
  float dzx, dzy;
  if (uJulia == 1) { zx = px; zy = py; cx = uCX; cy = uCY; dzx = 1.0; dzy = 0.0; }
  else { zx = vec2(0.0); zy = vec2(0.0); cx = px; cy = py; dzx = 0.0; dzy = 0.0; }

  float lrt2 = 2.0 * log2(max(rTol, 1e-38));
  float one = (uJulia == 1) ? 0.0 : 1.0;
  float fp = float(uPower);
  float lnp = log(fp);
  float m2 = 0.0, dlog2 = 0.0, ltPrev = -1e30;
  int n = 0;
  int per = 8, chk = 8;
  // Brent's reference point, kept as the FULL double single pair rather than as the high words.
  // See the cycle test below; this is the difference between an interior that is proven and an
  // interior that comes back as the out-of-iterations colour over half a deep frame.
  vec2 rrx = vec2(0.0), rry = vec2(0.0);

  for (int it = 0; it < 1000000; it++) {
    if (it >= uMaxIter) break;
    // z^(p-1), with a separate branch for the square. The same split the CPU kernel makes and for the
    // same reason: this loop IS the object. The general form costs four double single multiplies on
    // the p = 2 case that the square needs three of in total. uPower is a uniform, so the branch is
    // coherent across the whole draw and costs nothing in divergence.
    vec2 wx, wy;
    if (uPower == 2) { wx = zx; wy = zy; }
    else {
      wx = vec2(1.0, 0.0); wy = vec2(0.0);
      for (int k = 1; k < 8; k++) {
        if (k >= uPower) break;
        vec2 t = dsSub(dsMul(wx, zx), dsMul(wy, zy));
        wy = dsAdd(dsMul(wx, zy), dsMul(wy, zx));
        wx = t;
      }
    }
    // dz' = p z^(p-1) dz + 1. The derivative is fp32: it only ever feeds a distance estimate and a
    // bailout test, both of which want three digits, not fourteen.
    float mxr = fp * wx.x, myr = fp * wy.x;
    float ndzx = mxr * dzx - myr * dzy + one;
    float ndzy = mxr * dzy + myr * dzx;
    dzx = ndzx; dzy = ndzy;
    vec2 nzx, nzy;
    if (uPower == 2) {
      nzx = dsAdd(dsSub(dsMul(zx, zx), dsMul(zy, zy)), cx);
      nzy = dsAdd(dsTwice(dsMul(zx, zy)), cy);
    } else {
      nzx = dsAdd(dsSub(dsMul(wx, zx), dsMul(wy, zy)), cx);
      nzy = dsAdd(dsAdd(dsMul(wx, zy), dsMul(wy, zx)), cy);
    }
    zx = nzx; zy = nzy;
    n = it + 1;
    m2 = zx.x * zx.x + zy.x * zy.x;
    if (m2 > PB2) { o.why = 4; break; }
    float dm = dzx * dzx + dzy * dzy;
    if (dm > DM_SCALE) { dzx *= DM_DOWN; dzy *= DM_DOWN; dm *= DM_DOWN2; dlog2 += 32.0; }
    if ((it & 7) == 7) {
      // Condemned: the running distance estimate has dropped below rTol, so the boundary is inside
      // this cell or pixel and no further iteration can change the verdict. Written exactly, as
      // m log(m)^2 < rTol^2 |dz|^2, in log2 because the ratio itself passes ten to the twenty five.
      if (m2 > 1.0) {
        float lm = log(m2);
        float lt = log2(dm / (m2 * lm * lm)) + 2.0 * dlog2 + lrt2;
        if (lt > 0.0) {
          // Interpolate the crossing between the last two checks, so the escape count is continuous
          // in the point instead of jumping by eight. In log space that is a division, not two logs.
          float fr = 1.0;
          if (ltPrev > -1e29 && lt > ltPrev) fr = clamp(-ltPrev / (lt - ltPrev), 0.0, 1.0);
          o.nu = float(n) - 8.0 + 8.0 * fr;
          o.why = 1; break;
        }
        ltPrev = lt;
      }
      // BRENT'S CYCLE TEST, ON THE PAIR AND NOT ON THE HIGH WORD, and this one line was most of
      // what looked like a tone bug in this path.
      //
      // An interior point's orbit converges to an attracting cycle, and that convergence is the
      // only thing that ever PROVES interior; running out of iterations proves nothing and is drawn
      // as its own flat colour, which is light. The test used to compare high words with a
      // tolerance of 1e-10, and the note beside it argued that this was the conservative direction
      // because at |z| of order one an fp32 ulp is 1.2e-7, so only bitwise equality could meet it.
      // That is true and it is exactly the problem: a cycle whose multiplier is near one converges
      // geometrically and reaches 1.2e-7 in a few hundred iterations and bitwise equality never,
      // because the low word keeps moving. So at depth NOTHING was ever proven interior. Measured
      // at ten to the six on a 1377x1592 frame, against the CPU path which proves it in doubles:
      // whole interior regions came back as the out-of-iterations grey, about 40 percent of the
      // frame, and the mean displayed luminance was 0.33 against the splat path's 0.26.
      //
      // The pair carries about 43 bits, so the same 1e-10 IS reachable against the pair. The
      // difference of the high words is exact by Sterbenz once they are within a factor of two of
      // each other, which they are long before this fires, so adding the difference of the low
      // words recovers the full precision residual. Same tolerance, same conservatism, and it now
      // fires when the orbit really has closed.
      if (n > 24) {
        float ex = (zx.x - rrx.x) + (zx.y - rrx.y);
        float ey = (zy.x - rry.x) + (zy.y - rry.y);
        if (abs(ex) + abs(ey) < 1e-10) { o.why = 2; break; }
        chk -= 8;
        if (chk <= 0) { rrx = zx; rry = zy; per += per; chk = per; }
      }
    }
  }
  o.iters = n;
  if (o.why == 4) {
    float az = sqrt(m2);
    float lr = 0.5 * log(m2);
    // Smooth escape count: n minus log_p of log|z| over log B, which removes the integer banding and
    // makes nu a continuous function of the point. That continuity is the whole reason the colour can
    // be gradual.
    o.nu = float(n) - log(lr / LPB) / lnp;
    // de = 2|z|log|z| / |dz|, with |dz| carried as a mantissa and an exponent, so recovered through
    // exp2. A point whose derivative has run away underflows to zero, which is the right answer for a
    // point that is on the boundary as far as this cell can tell.
    float ldz = 0.5 * log2(dzx * dzx + dzy * dzy) + dlog2;
    float num = 2.0 * az * lr;
    o.de = num > 0.0 ? exp2(log2(num) - ldz) : 0.0;
    // |grad nu| = |dz| / (|z| log|z| ln p), which the quadtree's split criterion needs.
    o.gnu = exp2(min(ldz - log2(az * max(lr, 1e-12) * lnp), 100.0));
  }
  return o;
}
`;

// A double as the pair of floats the uniforms above take: the nearest fp32 and the residual, itself
// rounded to fp32, so the pair carries about 48 bits.
function splitDS(x) {
  const hi = Math.fround(x);
  return [hi, Math.fround(x - hi)];
}

/* ============================== IFS objects ============================== */
// The IFS catalogue and the moment machinery that gives each attractor its root splat.
// An object is a list of affine maps w_i(x) = A_i x + b_i with weights p_i, plus the view,
// palette and hue step it is drawn with. Objects marked `flat` lie in the x-z plane, which is
// what `rebase` requires. Nothing here draws: src/55_cut_ifs.js walks the tree these maps define.
//
// A map is {A, b, p, cc}, cc being its hue coordinate. `spec` builds one from a description:
// scale a number or [sx,sy,sz], rots [[ax,ay,az,deg], ...], t [x,y,z]. Rotations listed later
// apply further out, so `rots: [[z,45],[x,20]]` means rotX(20) * rotZ(45) * diag(scale).

function spec(o) {
  let A = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (const r of (o.rots || [])) A = mat3Mul(rotAxis(r[0], r[1], r[2], r[3]), A);
  const s = o.scale;
  const sv = typeof s === 'number' ? [s, s, s] : s;
  const B = new Float64Array(9);              // A diag(sv): scale, then rotate
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) B[r * 3 + c] = A[r * 3 + c] * sv[c];
  return { A: B, b: new Float64Array(o.t), p: o.p === undefined ? 1 : o.p, cc: o.cc };
}

const IFS = {
  sierpinski: {
    // A single map's fixed point satisfies x = 0.5x + 0.5q, so the aim is the vertex q, and the
    // structure hangs off one side of the frame rather than surrounding the camera.
    aim: '0',
    name: 'Sierpinski tetrahedron',
    kind: 'ifs',
    blurb: 'Four maps, contraction one half, no rotation. Ground truth: the attractor and its invariant ' +
           'measure are known in closed form and the pushforward is exact, so anything wrong on screen is ' +
           'the renderer and not the mathematics. Similarity dimension exactly 2.',
    // Four maps at equal measure, so the four increments are symmetric about zero and
    // the tetrahedron's mean hue is exactly fixed. See `hueStep` in src/40_state.js.
    hueStep: 0.080,
    palette: [[0.24, 0.75, 1.00], [0.30, 0.24, 1.00], [0.87, 0.24, 1.00],
              [1.00, 0.24, 0.56], [1.00, 0.49, 0.24], [0.93, 1.00, 0.24],
              [0.37, 1.00, 0.24], [0.24, 1.00, 0.69], [0.24, 0.75, 1.00]],
    bg: [[0.012, 0.016, 0.032], [0.026, 0.034, 0.062]],
    // From a sweep: at 0.70 px the cut converges at 300k splats with the 99th percentile of
    // visible error at 0.69 px, and sharpening further binds the budget and lets that tail back
    // up to 1.0 px. See ../sweep.mjs.
    view: { splitPx: 0.70, budget: 320000, density: 1.0, glow: 0.18, kernel: 3.4, sigma: 1.0,
            fog: 0.55, yaw: 0.60, pitch: -0.24 },
    maps: (() => {
      const v = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]];
      return v.map((q, i) => spec({
        scale: 0.5, t: [q[0] * 0.5, q[1] * 0.5, q[2] * 0.5], p: 0.25, cc: [0.03, 0.34, 0.66, 0.97][i],
      }));
    })(),
  },

  gasket: {
    aim: '0.1',
    name: 'Sierpinski triangle',
    kind: 'ifs',
    flat: true,
    // Circumradius 1. Invariant covariance exactly I/6 in the plane, so the root splat is a disc.
    blurb: 'Three maps, contraction one half, onto the vertices of an equilateral triangle. ' +
           'Similarity dimension log3/log2 = 1.585. No map carries a rotation, so the descent is a ' +
           'straight magnification and every splat is the exact pushforward of its parent.',
    hueStep: 0.085,
    palette: [[0.98, 0.82, 0.30], [0.99, 0.55, 0.26], [0.95, 0.31, 0.42],
              [0.72, 0.28, 0.85], [0.34, 0.42, 0.98], [0.22, 0.72, 0.96],
              [0.24, 0.92, 0.76], [0.60, 0.97, 0.40], [0.98, 0.82, 0.30]],
    bg: [[0.013, 0.014, 0.022], [0.028, 0.030, 0.048]],
    view: { splitPx: 0.60, budget: 300000, density: 1.3, glow: 0.22, kernel: 3.4, sigma: 1.2,
            fog: 0.55, yaw: 0, pitch: -1.5708 },
    maps: (() => {
      const h = Math.sqrt(3) / 2;
      // At pitch -pi/2 screen up is +z and screen right is -x, so this list is apex up with
      // the base below it.
      const v = [[0, 1], [h, -0.5], [-h, -0.5]];
      return v.map((q, i) => spec({
        scale: 0.5, t: [q[0] * 0.5, 0, q[1] * 0.5], p: 1 / 3, cc: (i + 0.5) / 3,
      }));
    })(),
  },

  dragon: {
    name: 'Folded dragon',
    kind: 'ifs',
    // The Heighway curve tiles the plane, so its similarity dimension is exactly 2, and tilting
    // the maps keeps every map a similarity and so keeps the dimension.
    blurb: 'Two similarities of ratio 1/sqrt(2), the Heighway dragon, with both maps tilted 20 degrees out ' +
           'of the plane. The plane filling curve becomes a folded sheet. The root covariance is anisotropic, ' +
           'so every splat is an oriented flake lying along the local fold, inherited through M Sigma M^T.',
    // Closed loop: the last stop is the first, so the ramp has no seam, and the hue coordinate is
    // a sum taken mod one that crosses the wrap constantly. Luminance is held even around the
    // loop or the bands read as brightness steps. Two maps, so touching pieces differ by 2h.
    hueStep: 0.060,
    palette: [[1.00, 0.74, 0.24], [0.98, 0.40, 0.26], [0.90, 0.22, 0.52],
              [0.60, 0.26, 0.84], [0.26, 0.40, 0.94], [0.16, 0.74, 0.90],
              [0.26, 0.88, 0.62], [0.66, 0.94, 0.34], [1.00, 0.74, 0.24]],
    bg: [[0.014, 0.012, 0.020], [0.030, 0.026, 0.044]],
    // A folded sheet surrounds the camera at depth and covers half the frame, so this object
    // stays budget limited. fog 1.10 rather than 0.5 shortens the depth of field, which at the
    // same budget takes the visible error tail from 1.31 px to 0.94 px.
    view: { splitPx: 0.80, budget: 340000, density: 1.3, glow: 0.15, kernel: 3.6, sigma: 1.05,
            fog: 1.10, yaw: 0.58, pitch: -0.20 },
    maps: [
      spec({ scale: Math.SQRT1_2, rots: [[0, 0, 1, 45], [1, 0, 0, 20]], t: [0, 0, 0], p: 0.5, cc: 0.06 }),
      spec({ scale: Math.SQRT1_2, rots: [[0, 0, 1, 135], [1, 0, 0, -20]], t: [1, 0, 0], p: 0.5, cc: 0.94 }),
    ],
  },

  cantor: {
    aim: '0.1',
    name: 'Cantor cube',
    kind: 'ifs',
    blurb: 'The middle thirds Cantor set cubed: eight corner copies at ratio one third. Dimension ' +
           'log8/log3 = 1.893, thin enough that it never closes into fog no matter how deep the cut goes. ' +
           'Invariant covariance exactly I/2, so the splats are spheres.',
    hueStep: 0.090,
    palette: [[0.24, 0.56, 1.00], [0.49, 0.24, 1.00], [1.00, 0.24, 0.93],
              [1.00, 0.24, 0.37], [1.00, 0.69, 0.24], [0.75, 1.00, 0.24],
              [0.24, 1.00, 0.30], [0.24, 1.00, 0.87], [0.24, 0.56, 1.00]],
    bg: [[0.010, 0.013, 0.026], [0.022, 0.030, 0.056]],
    view: { splitPx: 0.60, budget: 320000, density: 1.5, glow: 0.10, kernel: 4.2, sigma: 1.0,
            fog: 0.80, yaw: 0.42, pitch: -0.52 },
    maps: (() => {
      const out = [];
      for (let i = 0; i < 8; i++) {
        const x = (i & 1) ? 1 : -1, y = (i & 2) ? 1 : -1, z = (i & 4) ? 1 : -1;
        out.push(spec({
          scale: 1 / 3, t: [x * 2 / 3, y * 2 / 3, z * 2 / 3], p: 1 / 8,
          cc: (i + 0.5) / 8,          // colour by which corner of the parent cube
        }));
      }
      return out;
    })(),
  },

  koch: {
    aim: '0.1',
    name: 'Koch curve',
    kind: 'ifs',
    // Planar, so it sits in the 2D group of the picker and the autopilot must not turn it: a
    // yaw about world up takes a flat curve edge on once a quarter turn.
    flat: true,
    // The middle two maps stay in the base plane: lifting them 24 degrees folds the sub-copies in
    // every direction and a fold stops being distinguishable from a corner.
    blurb: 'Four similarities of ratio one third, the von Koch curve, planar. Dimension ' +
           'log4/log3 = 1.262, the thinnest object here: nearly a curve, and it reads as a ' +
           'crystalline thread rather than a cloud.',
    // Five splats a pixel here, 11 100 over about 2000 px of thread, so neighbouring pieces
    // average into the same pixel. An increment summed down the address keeps them one increment
    // apart, since they share a long prefix; a coordinate mixed by recency reads as flat colour.
    hueStep: 0.084,
    palette: [[0.39, 0.24, 1.00], [0.96, 0.24, 1.00], [1.00, 0.24, 0.47],
              [1.00, 0.58, 0.24], [0.85, 1.00, 0.24], [0.28, 1.00, 0.24],
              [0.24, 1.00, 0.77], [0.24, 0.66, 1.00], [0.39, 0.24, 1.00]],
    bg: [[0.010, 0.014, 0.030], [0.024, 0.032, 0.060]],
    // sigma 1.7 thickens every splat along its own covariance, which for a curve piece is across
    // the thread as well as along it, and the glow term then lets the thin core saturate. The
    // cheapest object here by an order of magnitude, so splitPx 0.34 fits.
    view: { splitPx: 0.34, budget: 220000, density: 1.6, glow: 0.42, kernel: 3.0, sigma: 1.7,
            fog: 0.60, yaw: 0, pitch: -1.5708 },
    maps: (() => {
      // The apex is computed rather than written down: the third map has to start exactly where
      // the second ends or the curve comes apart. Rotations are about y, so everything stays in
      // the x-z plane. The base runs (0,0,0) to (1,0,0), the second map turns 60 degrees up into
      // +z and ends at the apex, the third turns 60 back down and lands on (2/3, 0, 0).
      const H = Math.sqrt(3) / 6;
      const apex = [0.5, 0, H];
      return [
        spec({ scale: 1 / 3, t: [0, 0, 0], p: 0.25, cc: 0.05 }),
        spec({ scale: 1 / 3, rots: [[0, 1, 0, -60]], t: [1 / 3, 0, 0], p: 0.25, cc: 0.38 }),
        spec({ scale: 1 / 3, rots: [[0, 1, 0, 60]], t: apex, p: 0.25, cc: 0.72 }),
        spec({ scale: 1 / 3, t: [2 / 3, 0, 0], p: 0.25, cc: 0.97 }),
      ];
    })(),
  },

  vicsek: {
    aim: '0.1',
    name: 'Vicsek cross',
    kind: 'ifs',
    blurb: 'The centre cell of a three by three by three block plus its six face neighbours, ratio one ' +
           'third. Dimension log7/log3 = 1.771. No rotations at all, so every pushforward takes the scalar ' +
           'fast path and the splats stay spheres: the cheapest object here per splat, and the most ' +
           'architectural, a cross repeating on every axis at every scale.',
    hueStep: 0.088,
    palette: [[0.24, 1.00, 0.24], [0.24, 1.00, 0.81], [0.24, 0.62, 1.00],
              [0.43, 0.24, 1.00], [1.00, 0.24, 1.00], [1.00, 0.24, 0.43],
              [1.00, 0.62, 0.24], [0.81, 1.00, 0.24], [0.24, 1.00, 0.24]],
    bg: [[0.010, 0.016, 0.014], [0.024, 0.036, 0.032]],
    view: { splitPx: 0.62, budget: 300000, density: 1.35, glow: 0.14, kernel: 3.8, sigma: 1.0,
            fog: 0.70, yaw: 0.52, pitch: -0.34 },
    maps: (() => {
      const out = [spec({ scale: 1 / 3, t: [0, 0, 0], p: 1 / 7, cc: 0.5 })];
      const ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      for (let a = 0; a < 3; a++) {
        for (const sg of [1, -1]) {
          out.push(spec({
            scale: 1 / 3, p: 1 / 7,
            t: [ax[a][0] * sg * 2 / 3, ax[a][1] * sg * 2 / 3, ax[a][2] * sg * 2 / 3],
            cc: 0.5 + sg * (a + 1) / 7,        // colour by which arm of the cross
          }));
        }
      }
      return out;
    })(),
  },

  // No icosahedral flake. Twelve similarities of ratio 1/3, one per icosahedron vertex,
  // dimension log12/log3 = 2.262: at the mid descent view its splats covered 4.3 times the frame
  // area, with a 99th percentile footprint of 7 px against the tetrahedron's 1.2, and no splitPx
  // made the cut converge under a 340 000 budget. Shrinking sigma to 0.50 took the area ratio to
  // 0.20 and it still read as fog. A measure that fills more than a surface cannot be drawn by a
  // fixed number of primitives at a fixed screen size, so nothing here has dimension over 2.

  terdragon: {
    // The middle piece: its fixed point is interior to the sheet rather than at an end of the
    // curve, so the structure surrounds the aim from the first frame.
    aim: '1',
    name: 'Folded terdragon',
    kind: 'ifs',
    // Three similarities of ratio 1/sqrt(3), hexagonal where the Heighway dragon is square. The
    // multipliers are forced: three similar pieces spanning 0 to 1 need a1 + a2 + a3 = 1, and the
    // turn sequence of plus and minus 120 degrees gives a1 = a3 = e^(i30)/sqrt(3) and
    // a2 = e^(-i90)/sqrt(3), which sums to 1. Similarity dimension log3/log sqrt3 = 2, kept by the
    // tilt since every map stays a similarity.
    //
    // All three are tilted, the middle one hardest at 70 degrees against the outer two at 36.
    // Tilting only the outer two leaves the middle piece flat and measures as a thin band:
    // invariant covariance with a 0.25 ratio between its thinnest and widest standard deviations,
    // 31 percent of the frame height covered, 6 percent of a 96 by 48 grid holding any splat. At
    // MID 70 those are 0.74, 61 percent and 11.9 percent, alongside the folded dragon on all three.
    blurb: 'Three similarities of ratio one over root three, the terdragon, with the outer two tilted 24 ' +
           'degrees out of the plane. Similarity dimension exactly 2, so the cut converges rather than ' +
           'saturating. Where the Heighway dragon folds a square lattice this folds a hexagonal one: the ' +
           'sheet carries a three fold screw and the oriented flakes lie along it.',
    // Three maps, so the increments sum to zero and the mean hue is fixed. Between the dragon's
    // 0.060 for two maps and the tetrahedron's 0.080 for four.
    hueStep: 0.070,
    // Cool through the middle with the warm end held to a third of the loop, so the near face of
    // a fold picks up the warm stops and the far one falls away.
    palette: [[0.20, 0.86, 0.92], [0.24, 0.60, 0.98], [0.42, 0.34, 0.94],
              [0.72, 0.30, 0.86], [0.96, 0.38, 0.62], [1.00, 0.62, 0.36],
              [0.94, 0.86, 0.42], [0.54, 0.90, 0.62], [0.20, 0.86, 0.92]],
    bg: [[0.010, 0.014, 0.022], [0.024, 0.032, 0.048]],
    // The folded dragon's settings: same dimension, same density of sheet around the camera at
    // depth. fog 1.10 keeps the depth of field short.
    view: { splitPx: 0.80, budget: 340000, density: 1.3, glow: 0.15, kernel: 3.6, sigma: 1.05,
            fog: 1.10, yaw: 0.44, pitch: -0.26 },
    maps: (() => {
      // x is the real axis and z the imaginary one, so multiplying by e^(i phi) is a rotation
      // about y by minus phi: rotAxis(y, a) takes x to (cos a, 0, -sin a). The von Koch apex
      // above has to land exactly on (2/3, 0, 0), which pins the sign.
      const s = 1 / Math.sqrt(3);
      const TILT = 36, MID = 70;
      // The two joins, computed: t2 is where piece one ends and t3 where piece two ends, both in
      // the untilted curve, so the three pieces partition one curve before the fold is applied.
      const t2 = [s * Math.cos(Math.PI / 6), 0, s * Math.sin(Math.PI / 6)];
      const t3 = [t2[0] + s * Math.cos(-Math.PI / 2), 0, t2[2] + s * Math.sin(-Math.PI / 2)];
      return [
        spec({ scale: s, rots: [[0, 1, 0, -30], [1, 0, 0, TILT]], t: [0, 0, 0], p: 1 / 3, cc: 0.06 }),
        spec({ scale: s, rots: [[0, 1, 0, 90], [1, 0, 0, MID]], t: t2, p: 1 / 3, cc: 0.50 }),
        spec({ scale: s, rots: [[0, 1, 0, -30], [1, 0, 0, -TILT]], t: t3, p: 1 / 3, cc: 0.94 }),
      ];
    })(),
  },

  /* ---------------- the flat self similar objects ------------------------- */
  // Similarity IFS in the x-z plane, which is the condition `rebase` needs. An escape time set
  // costs hundreds of milliseconds to seconds a cut and walls at about 1e13, while an IFS is its
  // own level of detail hierarchy: a cut is twenty milliseconds and rebasing makes the descent
  // unbounded.
  //
  // Descending at the fixed point of a word multiplies the picture by that word's complex
  // multiplier each time, so a multiplier carrying a rotation makes the zoom a logarithmic spiral.
  // Levy at word 0 is e^(i45)/sqrt2, 0.83 turns a decade; Pythagoras at word 0 is 0.809 e^(i36),
  // 1.09 turns a decade. The gasket, pentaflake and carpet have no rotation in any map.

  levy: {
    // A one letter word, so the aim sits at an end of the curve: the two letter word's multiplier
    // is (e^(i45)/sqrt2)(e^(-i45)/sqrt2) = 1/2 exactly, and this object is here for the spiral.
    aim: '0',
    name: 'Levy C curve',
    kind: 'ifs',
    flat: true,
    // Cesaro 1906, Levy 1938. The curve crosses itself endlessly.
    blurb: 'Two similarities of ratio one over root two turning through plus and minus 45 degrees. The ' +
           'von Koch construction with the apex angle opened from 60 to 90, which changes the dimension ' +
           'from 1.26 to exactly 2 and the curve from a thread into a cloud. Its zoom map carries a 45 ' +
           'degree rotation, so descending into it turns twice for every decade: a logarithmic spiral.',
    hueStep: 0.060,
    palette: [[0.24, 0.98, 0.90], [0.24, 0.66, 1.00], [0.42, 0.36, 1.00],
              [0.80, 0.30, 0.98], [1.00, 0.34, 0.66], [1.00, 0.58, 0.32],
              [0.96, 0.90, 0.38], [0.52, 0.98, 0.56], [0.24, 0.98, 0.90]],
    bg: [[0.010, 0.014, 0.026], [0.022, 0.030, 0.052]],
    // Dimension 2 in a plane, so it is dense and wants the dragon's settings rather than the
    // Koch curve's: less glow, less sigma, a coarser threshold. Straight down, no yaw.
    view: { splitPx: 0.70, budget: 300000, density: 1.25, glow: 0.22, kernel: 3.2, sigma: 1.15,
            fog: 0.60, yaw: 0, pitch: -1.5708 },
    maps: (() => {
      // From 0 to 1 through the apex 0.5 + 0.5i. Rotation about y by minus phi is multiplication
      // by e^(i phi); see the terdragon for the sign.
      const s = Math.SQRT1_2;
      return [
        spec({ scale: s, rots: [[0, 1, 0, -45]], t: [0, 0, 0], p: 0.5, cc: 0.06 }),
        spec({ scale: s, rots: [[0, 1, 0, 45]], t: [0.5, 0, 0.5], p: 0.5, cc: 0.94 }),
      ];
    })(),
  },

  pythagoras: {
    aim: '0',
    name: 'Pythagoras tree',
    kind: 'ifs',
    flat: true,
    // The canopy of the Pythagoras tree: two similarities of ratio cos(theta) and sin(theta),
    // turning by +theta and -(90 - theta). The similarity dimension solves cos^d + sin^d = 1,
    // which is 2 at every lean by the Pythagorean identity.
    //
    // Lean 36 degrees, not 45, where the tree is symmetric and its two maps are the Levy curve's
    // up to a translation. The lean sets how unbalanced the two ratios are, and the walk is level
    // synchronous, so at level n the frontier spans (cos/sin)^n in size and a frontier is
    // allocated at every level in between; see prepareLevel. Measured at 876x601 two decades
    // down: 32 degrees spans depth 8 to 32 at an 18.3 ms rebuild, 36 spans 9 to 25 at 20.5 ms,
    // 40 spans 11 to 20 at 59.5 ms and flattens the lean into a symmetric arch. The zoom map at
    // the long branch's fixed point is 0.809 e^(i36), 1.09 turns a decade.
    blurb: 'The canopy of the Pythagoras tree, leaning 36 degrees: two similarities of ratio cos and sin ' +
           'of the lean, turning by plus 36 and minus 54 degrees. Its similarity dimension is exactly 2 ' +
           'at any lean, because that is the Pythagorean identity. Descending the long branch turns the ' +
           'picture 36 degrees for every factor of 0.81, which is one and a tenth turns a decade.',
    hueStep: 0.060,
    palette: [[0.98, 0.86, 0.30], [0.62, 0.94, 0.34], [0.26, 0.90, 0.52],
              [0.20, 0.80, 0.86], [0.28, 0.52, 0.98], [0.62, 0.36, 0.96],
              [0.94, 0.36, 0.74], [1.00, 0.56, 0.30], [0.98, 0.86, 0.30]],
    bg: [[0.012, 0.016, 0.020], [0.026, 0.034, 0.044]],
    // splitPx 1.55, coarse against the 0.70 the other flat objects use, because this is the one
    // object whose demand has no bound: its cover grows from 0.10 to 0.45 of the frame during a
    // descent, so at a finer threshold the cut pins to its cap. At the cap refinement stops at the
    // per node backstop `cutSize + nmaps <= cap`, which has no cross fade behind it, and the
    // surviving nodes are whichever the walk reached first, an order that shifts with the camera,
    // so a saturated cut shimmers near the centre of the spiral.
    //
    // 1.55 measured at 1e0.48 against a 0.85 px reference cut of the same view: gradient energy
    // 0.955 of the reference and 0 px equivalent Gaussian blur, against 0.888 and 0.4 px at 1.90
    // and 0.999 and 0 px at 1.25. Over a full rebase cycle to 1e0.87, 300 frames, the cut peaks at
    // 191 037 splats against 128 276 at 1.90 and 231 605 at 1.40, none reaching cap, so budget
    // 460 000 is 2.4 times the measured peak.
    view: { splitPx: 1.55, budget: 460000, density: 1.2, glow: 0.26, kernel: 3.2, sigma: 1.15,
            fog: 0.60, yaw: 0, pitch: -1.5708 },
    maps: (() => {
      const TH = 36 * Math.PI / 180;
      const c = Math.cos(TH), sn = Math.sin(TH);
      // The square's top edge runs from i to 1 + i; the right triangle on it has its apex at
      // i + c e^(i theta). The left child sits on i..apex, the right child on apex..1+i.
      const apex = [c * c, 0, 1 + c * sn];
      return [
        spec({ scale: c, rots: [[0, 1, 0, -36]], t: [0, 0, 1], p: c * c, cc: 0.08 }),
        spec({ scale: sn, rots: [[0, 1, 0, 54]], t: apex, p: sn * sn, cc: 0.92 }),
      ];
    })(),
  },

  pentaflake: {
    aim: '0.1',
    name: 'Pentaflake',
    kind: 'ifs',
    flat: true,
    // Five and not six: adding the central inverted copy takes the dimension from 1.672 to 1.86
    // and fills the holes that are the look of the object.
    blurb: 'Five copies of a pentagon at ratio one over phi squared, one at each vertex, which is the ' +
           'packing Durer drew in 1525. Similarity dimension log5 / log(phi squared) = 1.672: thin ' +
           'enough that the holes stay open at every depth, and fivefold, so it is the one object here ' +
           'with a symmetry that no lattice has.',
    hueStep: 0.085,
    palette: [[0.30, 0.42, 1.00], [0.66, 0.32, 1.00], [1.00, 0.32, 0.86],
              [1.00, 0.36, 0.42], [1.00, 0.66, 0.28], [0.88, 0.96, 0.32],
              [0.40, 0.98, 0.44], [0.28, 0.92, 0.92], [0.30, 0.42, 1.00]],
    bg: [[0.010, 0.012, 0.028], [0.024, 0.028, 0.056]],
    view: { splitPx: 0.50, budget: 260000, density: 1.4, glow: 0.30, kernel: 3.2, sigma: 1.3,
            fog: 0.60, yaw: 0, pitch: -1.5708 },
    maps: (() => {
      const PHI = (1 + Math.sqrt(5)) / 2;
      const r = 1 / (PHI * PHI);              // 0.381966
      const d = 1 - r;                        // child centres, in parent circumradius units
      const out = [];
      for (let k = 0; k < 5; k++) {
        const a = Math.PI / 2 + k * 2 * Math.PI / 5;
        out.push(spec({
          scale: r, t: [d * Math.cos(a), 0, d * Math.sin(a)], p: 0.2, cc: (k + 0.5) / 5,
        }));
      }
      return out;
    })(),
  },

  carpet: {
    aim: '0.1',
    name: 'Sierpinski carpet',
    kind: 'ifs',
    flat: true,
    blurb: 'Eight copies at ratio one third, the three by three block with its middle removed. ' +
           'Dimension log8/log3 = 1.893. Every map is a pure contraction with no rotation in it, so ' +
           'the zoom is a straight magnification and the object is a lattice of holes inside holes.',
    hueStep: 0.090,
    palette: [[1.00, 0.62, 0.24], [1.00, 0.34, 0.36], [0.92, 0.28, 0.72],
              [0.58, 0.34, 0.98], [0.26, 0.54, 1.00], [0.22, 0.84, 0.94],
              [0.36, 0.96, 0.58], [0.86, 0.98, 0.32], [1.00, 0.62, 0.24]],
    bg: [[0.014, 0.012, 0.018], [0.030, 0.026, 0.040]],
    view: { splitPx: 0.55, budget: 280000, density: 1.4, glow: 0.24, kernel: 3.4, sigma: 1.25,
            fog: 0.60, yaw: 0, pitch: -1.5708 },
    maps: (() => {
      const out = [];
      let k = 0;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
        if (i === 0 && j === 0) continue;     // the hole
        out.push(spec({
          scale: 1 / 3, t: [i * 2 / 3, 0, j * 2 / 3], p: 1 / 8, cc: (k + 0.5) / 8,
        }));
        k++;
      }
      return out;
    })(),
  },

};

/* -------------------- per map facts the kernel relies on ----------------- */
// scalar:    A = s I, so the pushforward takes a 24 multiply path instead of 81.
// conformal: A = s R with R orthogonal. `rebase` is exact only when every map is conformal,
//            because a non conformal map shears the camera as well as the object.
function prepareMaps(ms) {
  let sum = 0; for (const m of ms) sum += m.p;
  for (const m of ms) {
    m.p /= sum;
    const A = m.A, s = A[0];
    m.scalar = (Math.abs(A[4] - s) < 1e-15 && Math.abs(A[8] - s) < 1e-15 &&
      Math.abs(A[1]) + Math.abs(A[2]) + Math.abs(A[3]) +
      Math.abs(A[5]) + Math.abs(A[6]) + Math.abs(A[7]) < 1e-15);
    m.s = s;
    // |det A|^(1/3) is the uniform scale factor of the map by volume.
    m.scale = Math.pow(Math.abs(det3(A)), 1 / 3);
    // A A^T = scale^2 I to within a tolerance?
    let off = 0, dev = 0;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      let v = 0;
      for (let k = 0; k < 3; k++) v += A[r * 3 + k] * A[c * 3 + k];
      if (r === c) dev = Math.max(dev, Math.abs(v - m.scale * m.scale));
      else off = Math.max(off, Math.abs(v));
    }
    m.conformal = (off < 1e-12 && dev < 1e-12 * Math.max(1, m.scale * m.scale));
    m.inv = mat3Inv(m.A);
  }
}

function mat3Inv(a) {
  const c = new Float64Array(9);
  c[0] = a[4] * a[8] - a[5] * a[7];
  c[1] = a[2] * a[7] - a[1] * a[8];
  c[2] = a[1] * a[5] - a[2] * a[4];
  c[3] = a[5] * a[6] - a[3] * a[8];
  c[4] = a[0] * a[8] - a[2] * a[6];
  c[5] = a[2] * a[3] - a[0] * a[5];
  c[6] = a[3] * a[7] - a[4] * a[6];
  c[7] = a[1] * a[6] - a[0] * a[7];
  c[8] = a[0] * a[4] - a[1] * a[3];
  const d = a[0] * c[0] + a[1] * c[3] + a[2] * c[6];
  if (!isFinite(d) || Math.abs(d) < 1e-300) return null;
  for (let i = 0; i < 9; i++) c[i] /= d;
  return c;
}

/* --------------------------- words and their maps ------------------------ */
// The composed map of a branch word u = (i1 ... in), with each new map on the
// inside: F_u = w_i1 o w_i2 o ... o w_in, so M = A_i1 A_i2 ... A_in and
// t = b_i1 + A_i1 b_i2 + A_i1 A_i2 b_i3 + ...
function wordMap(ms, word) {
  const M = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const t = new Float64Array(3);
  const q = new Float64Array(3);
  for (const i of word) {
    const m = ms[i];
    mat3Vec(M, m.b, q);
    t[0] += q[0]; t[1] += q[1]; t[2] += q[2];
    M.set(mat3Mul(M, m.A));
  }
  return { M, t };
}

// The fixed point of F_u is exactly on the attractor, at the address u repeated forever.
// Recomputed from the word rather than transported: rebasing applies the inverse of a map, which
// expands, so a carried point's error grows by 1/s per rebase and at s = 1/3 an initial 1e-16
// reaches order one after 34. A 3x3 solve is exact at any depth.
function fixWord(ms, word) {
  const wm = wordMap(ms, word);
  return solveFixedPoint(wm.M, wm.t);
}

// Consuming the first letter of a periodic word leaves the word cyclically rotated, so the aim
// point in the new frame is the fixed point of that rotation.
function rotateWord(word, k) {
  const n = word.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = word[(i + k) % n];
  return out;
}

/* ===================== invariant measure of the IFS ====================== */
// The root splat is the Gaussian matching the mean and covariance of the
// invariant measure. Both follow from fixed point iteration on the moments:
//   mu = sum_i p_i (A_i mu + b_i)
//   S  = sum_i p_i (A_i S A_i^T + (A_i mu) b_i^T + b_i (A_i mu)^T + b_i b_i^T)
function rootMoments(maps) {
  let mu = new Float64Array(3);
  let S = new Float64Array(9);
  const Am = new Float64Array(3);
  for (let it = 0; it < 400; it++) {
    const mu2 = new Float64Array(3);
    const S2 = new Float64Array(9);
    for (const m of maps) {
      const A = m.A, b = m.b, p = m.p;
      mat3Vec(A, mu, Am);
      mu2[0] += p * (Am[0] + b[0]);
      mu2[1] += p * (Am[1] + b[1]);
      mu2[2] += p * (Am[2] + b[2]);
      const AS = mat3Mul(A, S);
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let k = 0; k < 3; k++) v += AS[r * 3 + k] * A[c * 3 + k];   // (A S A^T)_rc
        v += Am[r] * b[c] + b[r] * Am[c] + b[r] * b[c];
        S2[r * 3 + c] += p * v;
      }
    }
    mu = mu2; S = S2;
  }
  const cov = new Float64Array(6);
  cov[0] = S[0] - mu[0] * mu[0];
  cov[1] = S[1] - mu[0] * mu[1];
  cov[2] = S[2] - mu[0] * mu[2];
  cov[3] = S[4] - mu[1] * mu[1];
  cov[4] = S[5] - mu[1] * mu[2];
  cov[5] = S[8] - mu[2] * mu[2];
  // 1e-7 isotropic floor so a planar attractor cannot produce a zero volume covariance. It rides
  // through A S A^T like the rest, so a child is still the exact pushforward of its parent.
  const tr = (cov[0] + cov[3] + cov[5]) / 3;
  cov[0] += 1e-7 * tr; cov[3] += 1e-7 * tr; cov[5] += 1e-7 * tr;
  return { mu, cov };
}

// Information dimension of the invariant measure,
//     D = sum p_i log p_i / sum p_i log s_i
// which is the exponent in "measure inside a ball of radius r scales as r^D".
// That is exactly what the exposure law needs: see `measureNorm`.
function measureDimension(maps) {
  let num = 0, den = 0;
  for (const m of maps) { num += m.p * Math.log(m.p); den += m.p * Math.log(m.scale); }
  const D = den < -1e-12 ? num / den : 2;
  return Math.max(0.2, Math.min(3, D));
}

// How far the attractor reaches, in Mahalanobis units of the root Gaussian. Mahalanobis distance
// is invariant under affine maps, so one number measured at the root is exact at every level of
// the tree. A fixed number of sigmas is wrong: an elongated attractor reaches past five.
function supportRadius(maps, root) {
  const inv = sym3Inv(root.cov);
  if (!inv) return { maha: 12, euclid: Math.sqrt(root.cov[0] + root.cov[3] + root.cov[5]) * 3 };
  const cum = [];
  let acc = 0;
  for (const m of maps) { acc += m.p; cum.push(acc); }
  let seed = 0x2545f491 | 0;
  const rnd = () => {
    seed ^= seed << 13; seed |= 0;
    seed ^= seed >>> 17;
    seed ^= seed << 5; seed |= 0;
    return (seed >>> 0) / 4294967296;
  };
  let x = root.mu[0], y = root.mu[1], z = root.mu[2], worst = 0, far = 0;
  const N = 40000;
  for (let it = 0; it < N; it++) {
    const u = rnd();
    let j = 0;
    while (j < cum.length - 1 && u > cum[j]) j++;
    const A = maps[j].A, b = maps[j].b;
    const nx = A[0] * x + A[1] * y + A[2] * z + b[0];
    const ny = A[3] * x + A[4] * y + A[5] * z + b[1];
    const nz = A[6] * x + A[7] * y + A[8] * z + b[2];
    x = nx; y = ny; z = nz;
    if (it < 60) continue;                    // let the chaos game settle
    const dx = x - root.mu[0], dy = y - root.mu[1], dz = z - root.mu[2];
    const q = dx * (inv[0] * dx + inv[1] * dy + inv[2] * dz) +
              dy * (inv[1] * dx + inv[3] * dy + inv[4] * dz) +
              dz * (inv[2] * dx + inv[4] * dy + inv[5] * dz);
    if (q > worst) worst = q;
    const e = dx * dx + dy * dy + dz * dz;
    if (e > far) far = e;
  }
  return {
    maha: Math.min(Math.max(Math.sqrt(worst) * 1.15, 3), 30),   // for culling
    euclid: Math.sqrt(far) * 1.04,                              // for framing
  };
}

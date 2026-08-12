/* ========================= the resource governor ========================= */
// Scales the renderer's desktop budgets to the device. Each tier sets dpr (canvas pixel ratio,
// capped because a phone reports 3 and a 400 by 800 CSS canvas is then 2.9 Mpx of fragments),
// px (ceiling on width times height, applied after the dpr cap so a maximized 4K window is still
// bounded), budget (multiplier on the preset's splat budget) and work (refinement milliseconds a
// frame). Tier names denote target devices, not quality levels: low is a phone, high is a desktop
// with a discrete GPU, medium is everything else.
const PERF_TIERS = [
  { name: 'low', dpr: 1.0, px: 640000, budget: 0.22, work: 2.5 },
  { name: 'medium', dpr: 1.25, px: 1500000, budget: 0.5, work: 5.0 },
  { name: 'high', dpr: 1.5, px: 2800000, budget: 1.0, work: 9.0 },
];
// Current tier, and whether the governor may move it. Anything but auto is the viewer's choice and
// is never overridden. The tier opens on the guess below, before any frame has been timed.
let perfAuto = true;
let perfTier = perfGuess();

// A coarse pointer with no hover and a short screen edge below 820 CSS px means a phone;
// deviceMemory is the only hardware number a browser exposes.
function perfGuess() {
  if (typeof window === 'undefined' || typeof matchMedia !== 'function') return 2;
  const coarse = matchMedia('(pointer: coarse)').matches && !matchMedia('(any-hover: hover)').matches;
  const small = Math.min(screen.width || 9999, screen.height || 9999) < 820;
  const mem = navigator.deviceMemory || 8;
  if (coarse && small) return 0;
  if (coarse || mem <= 4) return 1;
  return 2;
}

// An unknown name means auto. Resets the window counters either way: perfFrame stops counting
// while perfAuto is false, so frames from before the change would close the first window after it.
function perfSet(name) {
  let tier = -1;
  for (let i = 0; i < PERF_TIERS.length; i++) if (PERF_TIERS[i].name === name) tier = i;
  perfAuto = (tier < 0);
  perfTier = perfAuto ? perfGuess() : tier;
  perfN = 0; perfBad = 0; perfGood = 0; perfWarm = 30;
}
function perfNow() { return PERF_TIERS[perfTier]; }
function perfName() { return PERF_TIERS[perfTier].name + (perfAuto ? ' (auto)' : ''); }

// Called once a frame with the frame duration in seconds. Counts how many of the last PERF_WIN
// frames were slower than PERF_BAD instead of averaging, so a single stall cannot move the tier.
// Steps down on a majority of bad frames and never back above a tier that has already failed this
// session, which is what stops the oscillation an unqualified hysteresis produces.
const PERF_WIN = 45;             // about three quarters of a second at 60 Hz
const PERF_BAD = 1 / 26;         // slower than 26 fps
const PERF_GOOD = 1 / 54;        // faster than 54 fps
// perfCeil is a ceiling and not a floor: the highest tier this session may still climb to.
let perfN = 0, perfBad = 0, perfGood = 0, perfCeil = PERF_TIERS.length - 1, perfWarm = 30;
function perfFrame(dt) {
  if (!perfAuto) return;
  // Shader compilation and the first cut land in the opening frames and would demote every machine.
  if (perfWarm > 0) { perfWarm--; return; }
  perfN++;
  if (dt > PERF_BAD) perfBad++;
  else if (dt < PERF_GOOD) perfGood++;
  if (perfN < PERF_WIN) return;
  const bad = perfBad / perfN, good = perfGood / perfN;
  perfN = 0; perfBad = 0; perfGood = 0;
  if (bad > 0.5 && perfTier > 0) {
    perfTier--;
    perfCeil = perfTier;         // never above the tier that failed
    dirty = true;
  } else if (good > 0.9 && perfTier < perfCeil) {
    perfTier++;
    dirty = true;
  }
}

// Drawing buffer size for a CSS size. Ratio cap first, then the absolute pixel ceiling, which only
// fires on a large window.
function perfViewport(cssW, cssH) {
  const t = perfNow();
  const dpr = Math.min(window.devicePixelRatio || 1, t.dpr);
  let w = Math.max(64, Math.round(cssW * dpr));
  let h = Math.max(64, Math.round(cssH * dpr));
  const over = (w * h) / t.px;
  if (over > 1) {
    const k = 1 / Math.sqrt(over);
    w = Math.max(64, Math.round(w * k));
    h = Math.max(64, Math.round(h * k));
  }
  return [w, h];
}

// Node arena size, fixed at load: the pool is one set of typed arrays allocated once and no tier
// change can resize it, so the opening guess is the only instrument available. At 204 bytes a node
// these are 41 MB, 109 MB and 273 MB. Read once, in src/40_state.js.
const PERF_ARENA = [150000, 400000, 1000000];
function perfArenaBudget() { return PERF_ARENA[perfGuess()]; }

// How much further back a portrait window sits to frame the same object: 1 when wider than tall,
// up to 2.6 upright. From the canvas rather than the window, because the demo runs in an iframe.
function frameAspect() {
  let w = 0, h = 0;
  if (typeof document !== 'undefined') {
    const cv = document.getElementById('gl');
    if (cv) { w = cv.clientWidth; h = cv.clientHeight; }
  }
  // Some paths run before layout, and the capture harness gives the canvas none at all.
  if (!(w > 0 && h > 0) && typeof window !== 'undefined') {
    w = window.innerWidth || 0; h = window.innerHeight || 0;
  }
  if (!(w > 0 && h > 0)) return 1;
  return Math.max(1, Math.min(2.6, h / w));
}

// Applied at startJob rather than written into cfg.budget, so a tier change leaves the preset's
// own number and the slider that reads it alone.
function perfBudget() {
  return Math.max(2000, Math.round(cfg.budget * perfNow().budget));
}

// The escape time sets are not offered at the bottom tier: their cut costs hundreds of
// milliseconds to seconds against about twenty for a tree walk, and no setting makes it cheap.
function perfHeavyOK() { return perfTier > 0; }

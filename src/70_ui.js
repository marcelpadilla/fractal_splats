// 70_ui.js is the front end: pointer and keyboard input, the control panel, the deep link query
// parser, the HUD, and the requestAnimationFrame loop that advances the camera and steps the cut.
// Nothing here computes geometry; it drives the pieces in 40_state.js, 50_cut.js and 60_gl.js.
// The camera state lives in the current anchor's frame rather than in world coordinates, so
// anything that writes cam.target or cam.goal runs after the anchor word has been replayed.
// Query parameters are parsed in applyURL; the ones marked evaluation only exist for the capture
// harnesses in workbench and are not part of the demo.

/* ============================== interaction ============================= */
let drag = null, moved = 0;

// Drag rotates a 3D object and is ignored for a 2D one. There is no pan and no click to re-aim:
// the aim point is the only thing holding the camera near the object. Pinch replaces the wheel on
// touch, and the pointer map is kept rather than a start span so a paused pinch does not jump.
const touches = new Map();
let pinch = 0;
function pinchSpan() {
  const p = [...touches.values()];
  if (p.length < 2) return 0;
  return Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
}

function attachInput(cv) {
  cv.addEventListener('pointerdown', e => {
    cv.setPointerCapture(e.pointerId);
    touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (touches.size === 2) { pinch = pinchSpan(); drag = null; return; }
    drag = { x: e.clientX, y: e.clientY };
    moved = 0;
  });
  cv.addEventListener('pointermove', e => {
    if (touches.has(e.pointerId)) touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (touches.size >= 2) {
      const s = pinchSpan();
      if (pinch > 4 && s > 4) {
        // Spreading the fingers means closer, so the camera distance falls by the span ratio.
        cam.dist = Math.min(cam.dist * (pinch / s), cam.startDist * 40);
        rebaseAll();
        dirty = true;
      }
      pinch = s;
      return;
    }
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    moved += Math.abs(dx) + Math.abs(dy);
    drag.x = e.clientX; drag.y = e.clientY;
    if (is2D(PRESETS[cfg.preset])) return;
    // Rotation about the eye's own axes; pi radians across the canvas width.
    const k = Math.PI / Math.max(cv.clientWidth || 1, 1);
    turnCamera(-dx * k, -dy * k);
    dirty = true;
  });
  cv.addEventListener('pointerup', e => {
    const wasPinch = touches.size >= 2;
    touches.delete(e.pointerId);
    pinch = pinchSpan();
    const wasDrag = moved > 6;
    drag = null;
    // No tap to re-aim. Wheel and pinch move along the existing aim, which cannot leave the object.
  });
  cv.addEventListener('pointercancel', e => {
    touches.delete(e.pointerId); pinch = pinchSpan(); drag = null;
  });
  cv.addEventListener('wheel', e => {
    e.preventDefault();
    const step = Math.sign(e.deltaY) * Math.min(1, Math.abs(e.deltaY) / 320) * 0.3;
    cam.dist = Math.min(cam.dist * Math.exp(step), cam.startDist * 40);
    rebaseAll();
    dirty = true;
  }, { passive: false });
  window.addEventListener('keydown', e => {
    const tag = e.target && e.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.target && e.target.isContentEditable) return;   // the typed speed, see #v-rate
    if (e.key === ' ') { cfg.autopilot = !cfg.autopilot; syncUI(); e.preventDefault(); }
    else if (e.key === 'x' || e.key === 'X') swapRepresentation();
    else if (e.key === 'g' || e.key === 'G') toggleGauss();
    else if (e.key === 'r' || e.key === 'R') resetView();
    // Only when there is something to restore, so Escape never HIDES anything and never fights
    // the browser's own use of it for leaving full screen.
    else if (e.key === 'Escape' && uiHidden) setUIHidden(false);
  });
}

/* ================================= UI =================================== */
const el = id => (typeof document === 'undefined' ? null : document.getElementById(id));

function fmt(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'G';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(Math.round(n));
}

// Slider position in [0, 1] to speed and back. Linear on [0, SPD_KNEE_X] from SPD_MIN to
// SPD_KNEE, geometric above it up to SPD_MAX, so the two pieces agree at the knee. spdToPos
// saturates at 1, so a typed speed above SPD_MAX parks the handle at the right hand stop.
function spdFromPos(x) {
  x = Math.max(0, Math.min(1, x));
  if (x <= SPD_KNEE_X) return SPD_MIN + (SPD_KNEE - SPD_MIN) * (x / SPD_KNEE_X);
  return SPD_KNEE * Math.pow(SPD_MAX / SPD_KNEE, (x - SPD_KNEE_X) / (1 - SPD_KNEE_X));
}
function spdToPos(v) {
  if (!(v > SPD_MIN)) return 0;
  if (v <= SPD_KNEE) return SPD_KNEE_X * (v - SPD_MIN) / (SPD_KNEE - SPD_MIN);
  return Math.min(1, SPD_KNEE_X + (1 - SPD_KNEE_X) *
    Math.log(v / SPD_KNEE) / Math.log(SPD_MAX / SPD_KNEE));
}
// Three significant figures at every magnitude in a fixed width, so the panel does not reflow.
function fmtSpeed(v) { return v < 10 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : v.toFixed(0); }

const SLIDERS = [
  // Speed is shown as a multiple of RATE1, not in e-folds a second.
  ['s-rate', 'rate', v => spdFromPos(v) * RATE1, () => fmtSpeed(cfg.rate / RATE1)],
  ['s-cx', 'cx', v => v, () => cfg.cx.toFixed(3)],
  ['s-cy', 'cy', v => v, () => cfg.cy.toFixed(3)],
  ['s-power', 'power', v => Math.round(v), () => 'z^' + Math.round(cfg.power)],
];
const SLIDER_INV = {};
// State back to slider units, so a rate from the query string or a preset positions the handle.
SLIDER_INV.rate = r => spdToPos(r / RATE1);

function syncUI() {
  if (!el('s-rate')) return;
  const set = (id, v) => { const s = el(id); if (s && document.activeElement !== s) s.value = v; };
  for (const [id, key, , show] of SLIDERS) {
    const inv = SLIDER_INV[key];
    set(id, inv ? inv(cfg[key]) : cfg[key]);
    const v = el('v-' + id.slice(2));
    // Never while focused: #v-rate is contenteditable, and rewriting its textContent under a
    // caret moves the caret to the start.
    if (v && document.activeElement !== v) v.textContent = show();
  }
  const sel = el('preset');
  if (sel && sel.value !== menuValue()) sel.value = menuValue();   // a URL may override the default
  // The run control shows the action rather than the state, the media control convention.
  const b = el('btn-run');
  if (b) {
    b.innerHTML = cfg.autopilot ? '&#9208;' : '&#9654;';        // pause bars, play triangle
    b.dataset.on = cfg.autopilot ? '1' : '0';
    b.setAttribute('aria-label', cfg.autopilot ? 'pause' : 'play');
    b.setAttribute('title', cfg.autopilot ? 'Pause the descent' : 'Resume the descent');
  }
  // Both are sliding switches with fixed labels. The state is entirely in `data-on`, which drives
  // the knob, the track colour and which label is dimmed, in CSS.
  const rb = el('btn-rep');
  if (rb) {
    const on = isDirect();
    rb.dataset.on = on ? '1' : '0';
    rb.setAttribute('aria-checked', on ? 'true' : 'false');
    rb.disabled = !PIXEL_PAIR[cfg.preset];
  }
  const gb = el('btn-gauss');
  if (gb) {
    gb.dataset.on = cfg.gauss ? '1' : '0';
    gb.setAttribute('aria-checked', cfg.gauss ? 'true' : 'false');
    // The one label that changes: this switch is off/on rather than two named states.
    const lb = el('lb-gauss');
    if (lb) lb.textContent = cfg.gauss ? 'on' : 'off';
    // The per pixel object has no splats, so the switch is disabled rather than silently inert.
    gb.disabled = isDirect();
  }
  // The tool bar. The period is written like every other value except while it is being typed
  // into, since rewriting the textContent under a caret moves the caret to the start.
  const ab = el('btn-auto');
  if (ab) {
    ab.dataset.on = autoOn ? '1' : '0';
    ab.setAttribute('aria-checked', autoOn ? 'true' : 'false');
  }
  const va = el('v-auto');
  if (va && document.activeElement !== va) va.textContent = String(autoPeriod);
  const fb = el('btn-full');
  if (fb) {
    const on = !!fsElement();
    fb.dataset.fs = on ? '1' : '0';
    fb.setAttribute('aria-label', on ? 'leave full screen' : 'full screen');
    fb.setAttribute('title', on ? 'Leave full screen' : 'Full screen');
  }
  const hb = el('btn-hide');
  if (hb) hb.setAttribute('aria-pressed', uiHidden ? 'true' : 'false');
  const am = el('btn-aim');
  if (am && aimList.length > 1) {
    const at = Math.max(0, aimList.indexOf(aimIdx));
    const nx = (at + 1) % aimList.length;
    am.setAttribute('title', 'Aiming at the ' + (AIM_KIND[at] || 'target ' + at) +
      '. Next: the ' + (AIM_KIND[nx] || 'target ' + nx));
  }
  const wrap = el('controls');
  if (wrap) {
    wrap.dataset.kind = PRESETS[cfg.preset].kind;
    wrap.dataset.julia = PRESETS[cfg.preset].julia ? '1' : '0';
    // Only where there is more than one place worth falling into.
    wrap.dataset.aims = aimList.length > 1 ? '1' : '0';
  }
  if (el('h-object')) el('h-object').textContent = PRESETS[cfg.preset].name;
}

function is2D(P) { return P.kind === 'plane' || !!P.flat; }

// What the picker shows. The per pixel objects are not menu entries but the same scenes drawn the
// other way, so while one is loaded the picker names its splat twin instead of going blank.
function menuValue() { return (PIXEL_PAIR[cfg.preset] && PRESETS[cfg.preset].direct) ? PIXEL_PAIR[cfg.preset] : cfg.preset; }

// Three groups. The first two split by ambient dimension. The third holds the escape time sets,
// which are only quasi self similar: every level down is new structure, the cost per point grows
// with depth, and the zoom ends at a precision wall. A plane cut is 0.4 to 9 s against 1 to 140 ms
// for a similarity IFS answering the same question, see knowledge/flat_self_similar.md. Order
// inside a group is alphabetical by the name on screen.
const MENU_GROUPS = [
  ['3D self similar',     P => !is2D(P)],
  ['2D self similar',     P => is2D(P) && P.kind !== 'plane'],
  ['2D not self similar', P => P.kind === 'plane'],
];

function fillPresetMenu(sel) {
  const keys = Object.keys(PRESETS);
  for (const [label, want] of MENU_GROUPS) {
    const g = document.createElement('optgroup');
    g.label = label;
    const mine = keys.filter(k => {
      const P = PRESETS[k];
      if (P.hidden) return false;
      if (!want(P)) return false;
      // Escape time cuts are hundreds of ms to seconds against about 20 ms for everything else,
      // so the bottom tier does not offer them. See perfHeavyOK.
      if (P.kind === 'plane' && !perfHeavyOK()) return false;
      return true;
    }).sort((a, b) => PRESETS[a].name.localeCompare(PRESETS[b].name));
    for (const k of mine) {
      const o = document.createElement('option');
      o.value = k; o.textContent = PRESETS[k].name;
      g.appendChild(o);
    }
    if (g.children.length) sel.appendChild(g);
  }
}

function bindUI() {
  const sel = el('preset');
  if (!sel) return;                       // canvas only page, e.g. the capture harness
  // `is2D` has to return a boolean, hence `!!P.flat`: the group predicates negate it, and an
  // undefined result puts every object with neither field on the wrong side of the split.
  fillPresetMenu(sel);
  sel.value = menuValue();
  // Picking a scene keeps the representation: with the per pixel toggle on, the twin is loaded.
  sel.addEventListener('change', () => {
    const want = (isDirect() && PIXEL_PAIR[sel.value]) ? PIXEL_PAIR[sel.value] : sel.value;
    loadPreset(want, false);
    // A hand on the menu outranks the clock: the object just chosen gets a full period, and the
    // walk carries on from where that object sits in the order rather than from where the timer
    // had got to. An object outside the order gives -1, which steps to its head.
    if (autoOn) { autoAt = autoOrder.indexOf(cfg.preset); autoT = 0; }
  });

  for (const [id, key, xf] of SLIDERS) {
    const s = el(id);
    if (!s) continue;
    const inv = SLIDER_INV[key];
    s.value = inv ? inv(cfg[key]) : cfg[key];
    s.addEventListener('input', () => {
      cfg[key] = xf(+s.value);
      if (key === 'kernel') kern = kernelConst(cfg.kernel);
      // c and the exponent move the set, so the previous aim is no longer on it. Rebuild the
      // targets and re-aim; projDist resets so the next halving of cam.dist re-projects.
      if (key === 'cx' || key === 'cy' || key === 'power') {
        const P = PRESETS[cfg.preset];
        if (P.kind === 'plane') {
          planeTargets(P);
          setGoal(targetIdx);
          projDist = 1e300;
        }
      }
      dirty = true; syncUI();
    });
  }

  // Both speed marks come from the mapping above rather than the stylesheet. Custom properties
  // inherit, so setting them on the panel reaches the track pseudo element and the tick alike.
  const panel = el('controls');
  if (panel) {
    panel.style.setProperty('--spd-def', spdToPos(1).toFixed(4));
    panel.style.setProperty('--spd-knee', (SPD_KNEE_X * 100).toFixed(1) + '%');
  }

  // Typed speed. The field accepts up to SPD_TYPED_MAX, past the slider's SPD_MAX, because the
  // renderer holds there; see speed_probe.mjs. Enter or blur commits, Escape restores, and what
  // is committed is clamped and written back, so the field shows the speed in force.
  const vr = el('v-rate');
  if (vr) {
    let cancelled = false;
    const commit = () => {
      if (!cancelled) {
        // textContent and not innerHTML: this is a contenteditable and a paste can carry markup.
        const n = parseFloat((vr.textContent || '').replace(',', '.').replace(/[^0-9.eE+-]/g, ''));
        if (isFinite(n)) cfg.rate = Math.max(0, Math.min(SPD_TYPED_MAX, n)) * RATE1;
      }
      cancelled = false;
      dirty = true;
      syncUI();                     // which rewrites the field, now that it is no longer focused
    };
    vr.addEventListener('focus', () => {
      // Select the whole number, so typing replaces it rather than appending.
      const r = document.createRange(); r.selectNodeContents(vr);
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
    });
    vr.addEventListener('blur', commit);
    vr.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); vr.blur(); }
      else if (e.key === 'Escape') { e.preventDefault(); cancelled = true; vr.blur(); }
    });
  }

  const rn = el('btn-run');
  if (rn) rn.addEventListener('click', () => { cfg.autopilot = !cfg.autopilot; syncUI(); });
  const rs = el('btn-reset');
  if (rs) rs.addEventListener('click', resetView);
  const am = el('btn-aim');
  if (am) am.addEventListener('click', nextAim);
  // The readout's second half: zoom, depth range, precision, buffer format, cut cost and hardware.
  // The state lives on the panel, so the hiding is CSS.
  const ib = el('btn-info');
  if (ib) ib.addEventListener('click', () => {
    const w = el('controls');
    if (!w) return;
    const on = w.dataset.info === '1' ? '0' : '1';
    w.dataset.info = on;
    ib.dataset.on = on;
    ib.setAttribute('title', on === '1' ? 'Fewer numbers' : 'More numbers');
  });
  const rb = el('btn-rep');
  if (rb) rb.addEventListener('click', swapRepresentation);
  const gb = el('btn-gauss');
  if (gb) gb.addEventListener('click', toggleGauss);
  // Auto switch. Turning it on draws the order; turning it off leaves the object that is on
  // screen where it is, because stopping a slideshow should not also move it.
  const ab = el('btn-auto');
  if (ab) ab.addEventListener('click', () => {
    autoOn = !autoOn;
    if (autoOn) autoShuffle(); else autoT = 0;
    syncUI();
  });
  // The period, typed. Same protocol as the speed field above: Enter or blur commits, Escape
  // restores, and what is committed is clamped and written back so the field shows the period in
  // force. Integer seconds, because nobody wants to specify a scene change to the half second,
  // and the count restarts on commit so a new period means what it says from now.
  const va = el('v-auto');
  if (va) {
    let cancelled = false;
    const commit = () => {
      if (!cancelled) {
        const n = parseFloat((va.textContent || '').replace(',', '.').replace(/[^0-9.eE+-]/g, ''));
        if (isFinite(n)) {
          const v = Math.max(AUTO_MIN, Math.min(AUTO_MAX, Math.round(n)));
          // Only when it actually moved. Leaving the field is a commit whether or not anything was
          // typed, so restarting the count unconditionally would mean that tabbing through the bar
          // pushes the next switch a full period away every time.
          if (v !== autoPeriod) { autoPeriod = v; autoT = 0; }
        }
      }
      cancelled = false;
      syncUI();
    };
    va.addEventListener('focus', () => {
      const r = document.createRange(); r.selectNodeContents(va);
      const sl = window.getSelection(); sl.removeAllRanges(); sl.addRange(r);
    });
    va.addEventListener('blur', commit);
    va.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); va.blur(); }
      else if (e.key === 'Escape') { e.preventDefault(); cancelled = true; va.blur(); }
    });
  }
  // Full screen, and the state read back from the browser rather than remembered here: Escape
  // and F11 leave it without going through this button, so the icon has to follow the document.
  const fb = el('btn-full');
  if (fb) {
    if (!fsAvailable()) fb.hidden = true;
    else {
      fb.addEventListener('click', fsToggle);
      document.addEventListener('fullscreenchange', syncUI);
      document.addEventListener('webkitfullscreenchange', syncUI);
    }
  }
  const hb = el('btn-hide');
  if (hb) hb.addEventListener('click', () => setUIHidden(!uiHidden));
  // Anything but Auto turns the governor off. The menu is rebuilt because the bottom tier does
  // not offer the escape time objects; one already on screen stays there and only leaves the list.
  const qs = el('quality');
  if (qs) {
    qs.addEventListener('change', () => {
      perfSet(qs.value);
      rebuildPresetMenu();
      dirty = true; syncUI();
    });
  }
}

// The object menu, rebuilt in place. Called when the resource tier changes.
function rebuildPresetMenu() {
  const sel = el('preset');
  if (!sel) return;
  sel.innerHTML = '';
  fillPresetMenu(sel);
  sel.value = menuValue();
}

// What the reset control returns to. Captured at load, before applyURL runs, so a deep link cannot
// make its own settings the thing reset goes back to. Only the three that no preset names: the
// camera and every rendering parameter come from the preset itself.
const CFG0 = { rate: cfg.rate, autopilot: cfg.autopilot, gauss: cfg.gauss };

// Put the current object back to how it opened. loadPreset with keepView false restores the camera,
// the anchor and the preset's own view block; the speed, the run state and the Gaussian view are
// not in a preset, so they are named here. A per pixel view resets to its splat partner, because
// the representation is a setting rather than an object: menuValue already names that partner.
function resetView() {
  cfg.rate = CFG0.rate;
  cfg.autopilot = CFG0.autopilot;
  cfg.gauss = CFG0.gauss;
  loadPreset(menuValue(), false);
  paramKey = '';
  stillT = 0; stillCuts = 0;
  dirty = true;
  syncUI();
}

// The three kinds a preset names, in the order it names them. Position is the meaning: an object
// lists its aims interior first and walks out to the tip.
const AIM_KIND = ['interior', 'boundary', 'tip'];

// Aim the descent somewhere else on the same object. It snaps back out to the opening view first,
// because a re-aim at depth is not a re-aim: the new target is astronomically far away in units of
// the current camera distance, so the aim invariant would drag the camera the whole way in one
// frame and land it on nothing. Starting again from the top is what "zoom into a different place"
// means. The speed and the run state are the viewer's and survive, as they do across an auto
// switch; everything else about the view is the object's and is reset.
function nextAim() {
  if (aimList.length < 2) return;
  const at = aimList.indexOf(aimIdx);
  const want = aimList[((at < 0 ? 0 : at) + 1) % aimList.length];
  const rate = cfg.rate, run = cfg.autopilot;
  loadPreset(cfg.preset, false);       // which resets aimIdx to the head of the list
  aimIdx = want;
  setGoal(want);
  cfg.rate = rate;
  cfg.autopilot = run;
  climbing = false;
  projDist = 1e300;
  paramKey = '';
  stillT = 0; stillCuts = 0;
  dirty = true;
  syncUI();
}

// Swap the splat render for the per pixel one without moving the camera: the two are the same
// field, palette, colour law and targets, so loadPreset keeps the view. `paramKey` is cleared so
// the next frame rebuilds rather than drawing the other object's cut through this object's shader.
function swapRepresentation() {
  const other = PIXEL_PAIR[cfg.preset];
  if (!other) return;
  loadPreset(other, true);
  paramKey = '';
  stillT = 0; stillCuts = 0;
  dirty = true;
  syncUI();
}

// The Gaussian view is remembered across reloads and across a change of object. It moves the split
// threshold, a structural property of the cut, so toggleGauss clears `paramKey` and forces a
// rebuild; the two thresholds disagree about splat size by a factor of nine. localStorage throws
// in a sandboxed iframe or with cookies blocked, so both sides are wrapped.
const GAUSS_KEY = 'fractalsplats.gauss';
function gaussRemember(v) {
  try { localStorage.setItem(GAUSS_KEY, v ? '1' : '0'); } catch (e) { /* private mode, no matter */ }
}
function gaussRecall() {
  try { return localStorage.getItem(GAUSS_KEY) === '1' ? 1 : 0; } catch (e) { return 0; }
}

function toggleGauss() {
  if (isDirect()) return;
  cfg.gauss = cfg.gauss ? 0 : 1;
  gaussRemember(cfg.gauss);
  paramKey = '';
  syncUI();
}

/* ============================ the room ================================== */
// Three controls that are about the window rather than about the object: whether the viewer
// moves on by itself, what fills the screen, and whether there is anything on screen but the
// render. None of them changes what is drawn, which is why they are at the opposite corner
// from the panel that does. See #tools in page.part.html.

/* ------------------------------ auto switch ------------------------------- */
// A random order of the objects, walked and then looped, so a viewer left running shows the
// whole set rather than the same object twice in a row. The order is drawn once when the
// switch is turned on and kept, so what plays is a shuffled playlist and not an independent
// coin toss each time, which is the difference between seeing everything and seeing four of
// them repeatedly.
//
// THE ESCAPE TIME FIELDS ARE NOT IN IT. A plane cut is 0.4 to 9 seconds of one CPU thread
// against 8 to 14 ms for a similarity IFS, and it goes on sharpening for twenty five seconds
// after that, so a timed switch into the Mandelbrot or the Julia set lands on a coarse frame
// that is still improving when the timer fires again. The Mandelbrot terrain is a field too
// and is `hidden`, so the menu rule already excludes it.
const AUTO_MIN = 3, AUTO_MAX = 3600;
let autoOn = false, autoPeriod = 60, autoT = 0, autoOrder = [], autoAt = -1;

// Named POSITIVELY, `kind === 'ifs'`, and not as everything that is not a plane: the second
// form admits the Mandelbrot terrain, which is out today only because it is `hidden`, and whose
// load alone is a 28x28 grid of field probes plus a boundary projection for each of six targets,
// synchronously, inside the frame callback. `hidden` and `direct` are kept as well so this pool
// can never contain something the object menu does not.
function autoPool() {
  return Object.keys(PRESETS).filter(k => {
    const P = PRESETS[k];
    return !P.hidden && !P.direct && P.kind === 'ifs';
  });
}

// Fisher-Yates, and the walk starts at whatever is on screen: `indexOf` returns -1 for an
// object outside the order, which is the Mandelbrot or the Julia set, and -1 steps to 0, so
// the first switch is to the head of the order rather than past it.
function autoShuffle() {
  const pool = autoPool();
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = pool[i]; pool[i] = pool[j]; pool[j] = t;
  }
  autoOrder = pool;
  autoAt = pool.indexOf(cfg.preset);
  autoT = 0;
}

// The speed and the run state are the VIEWER's and not the object's, so both are carried
// across by hand: loadPreset writes the new object's own view block over cfg and sets the
// autopilot from its kind, which would silently return a projector to speed 1 and restart a
// descent the viewer had paused. Everything else about the object is meant to change; that is
// what the switch is for.
function autoNext() {
  if (!autoOrder.length) return;
  autoAt = (autoAt + 1) % autoOrder.length;
  const rate = cfg.rate, run = cfg.autopilot;
  loadPreset(autoOrder[autoAt], false);
  cfg.rate = rate;
  cfg.autopilot = run;
  // The flight state is the previous object's and loadPreset does not own it. Switching away
  // from an escape time field that had turned around at its precision wall leaves `climbing`
  // set, and the new object then opens at cam.startDist with the climb branch live: it passes
  // the turnaround test on its first frame and calls setGoal(targetIdx + 1), throwing away the
  // aim word the preset names on purpose.
  climbing = false;
  projDist = 1e300;
  dirty = true;
  syncUI();
}

// On the frame clock and not on an interval. requestAnimationFrame does not fire in a
// background tab, so a timer would come back to a viewer that had switched thirty times with
// nothing drawn and a cut queued for each; counting frames' worth of dt makes the period mean
// seconds of RENDERED time, which is the only kind this viewer has.
function autoTick(dt) {
  if (!autoOn || !(autoPeriod > 0)) return;
  autoT += dt;
  if (autoT < autoPeriod) return;
  autoT = 0;
  autoNext();
}

/* ------------------------------ full screen ------------------------------- */
// The browser's own, which is not the same thing as the project page's "Open in full screen":
// that one promotes the iframe to a fixed overlay INSIDE the page, so the page's own chrome is
// still around it. This one takes the display, which is the one that matters on a projector.
// Inside an iframe it needs allowfullscreen, which the project page's iframe carries; where the
// API is missing or the frame was embedded without it, the button is not offered at all rather
// than offered and inert.
function fsElement() {
  if (typeof document === 'undefined') return null;
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}
function fsAvailable() {
  if (typeof document === 'undefined') return false;
  const d = document.documentElement;
  if (!d || !(d.requestFullscreen || d.webkitRequestFullscreen)) return false;
  const en = document.fullscreenEnabled !== undefined
    ? document.fullscreenEnabled : document.webkitFullscreenEnabled;
  return en !== false;
}
function fsToggle() {
  const d = document.documentElement;
  if (!d) return;
  // Both calls reject rather than throw when the gesture is not trusted or the frame is not
  // permitted, and an unhandled rejection in a demo is a console error nobody can act on.
  if (fsElement()) {
    const x = document.exitFullscreen || document.webkitExitFullscreen;
    if (x) { const r = x.call(document); if (r && r.catch) r.catch(() => {}); }
  } else {
    const q = d.requestFullscreen || d.webkitRequestFullscreen;
    if (q) { const r = q.call(d); if (r && r.catch) r.catch(() => {}); }
  }
}

/* --------------------------- the hidden interface ------------------------- */
// The state is one attribute on the root element and the hiding itself is CSS, the same way
// the readout's second half is hidden behind data-info. Escape restores; so does the control
// itself, which is the only thing left on screen.
let uiHidden = false;
function setUIHidden(v) {
  uiHidden = !!v;
  if (typeof document !== 'undefined' && document.documentElement) {
    document.documentElement.setAttribute('data-ui', uiHidden ? '0' : '1');
  }
  syncUI();
}

// A view is named by the original frame plus the anchor word, so replay is exact: walk the same
// sequence of maps, then write the camera, which the query already gives in the anchor's frame.
function applyURL() {
  const q = new URLSearchParams(location.search);
  // The tier is read before the preset: it decides which objects are in the menu and how large
  // the first cut may be. `q=auto` is the default.
  if (q.has('q')) perfSet(q.get('q'));
  // Auto switch, before the object is loaded and before the early return below, so a link can
  // ask for it with or without naming an object. The order is drawn in boot(), once whatever
  // object this query asks for is the one on screen.
  if (q.has('ap') && isFinite(+q.get('ap'))) {
    autoPeriod = Math.max(AUTO_MIN, Math.min(AUTO_MAX, Math.round(+q.get('ap'))));
  }
  if (q.has('as')) autoOn = q.get('as') === '1';
  if (!q.has('p') || !PRESETS[q.get('p')]) return false;
  const num = (k, d) => (q.has(k) && isFinite(+q.get(k))) ? +q.get(k) : d;
  loadPreset(q.get('p'), false);
  cfg.budget = Math.max(1000, Math.min(MAXBUDGET, num('b', cfg.budget)));
  cfg.splitPx = num('sp', cfg.splitPx);
  // Show the individual Gaussians; in the query so a capture of the mode is reproducible.
  cfg.gauss = Math.max(0, Math.min(1, Math.round(num('gs', cfg.gauss))));
  cfg.density = num('dn', cfg.density);
  cfg.glow = num('gw', cfg.glow);
  cfg.fog = num('fg', cfg.fog);
  cfg.kernel = num('kn', cfg.kernel);
  cfg.sigma = num('sg', cfg.sigma);
  cfg.rate = Math.max(0, Math.min(SPD_TYPED_MAX * RATE1, num('rt', cfg.rate)));
  cfg.relief = num('rl', cfg.relief);
  cfg.iters = Math.round(num('it', cfg.iters));
  // The family parameters, applied before the targets are built, hence the second loadPreset.
  const pw0 = cfg.power, cx0 = cfg.cx, cy0 = cfg.cy;
  cfg.power = Math.round(num('pw', cfg.power));
  cfg.cx = num('cx', cfg.cx); cfg.cy = num('cy', cfg.cy);
  if (cfg.power !== pw0 || cfg.cx !== cx0 || cfg.cy !== cy0) loadPreset(q.get('p'), false);
  cfg.contour = num('cn', cfg.contour);
  cfg.height = num('hm', cfg.height) ? 1 : 0;
  kern = kernelConst(cfg.kernel);
  if (q.get('pr') === 'f32' || q.get('pr') === 'f64') cfg.precision = q.get('pr');
  fixedDt = Math.max(0, Math.min(0.5, num('dt', 0)));
  // Evaluation only: the screen space size cap, for the popping harness.
  SIZE_CAP = Math.max(0.5, Math.min(400, num('sc', SIZE_CAP)));
  // Evaluation only: the plane's latency ceiling, in ms of wall clock, so a loaded machine drains
  // earlier and cuts differently. Two captures of one static view differed twofold in rms.
  PLANE_MS = Math.max(5, Math.min(60000, num('mc', PLANE_MS)));
  // Evaluation only: how far past exp(-8) of attenuation the depth of field reaches. See ZF_MUL.
  ZF_MUL = Math.max(1, Math.min(8, num('zk', ZF_MUL)));
  // Evaluation only: the narrow end of the plane's per cell kernel bandwidth. `kb=1` is the A, a
  // single global bandwidth. See KERN_MIN in src/52_cut_plane.js.
  KERN_MIN = Math.max(0.1, Math.min(1, num('kb', KERN_MIN)));
  // Evaluation only: the flat splat's quad size, in multiples of the kernel's support. It is what
  // lets `kb` go below 0.447. See FLAT_EXT in src/52_cut_plane.js.
  FLAT_EXT = Math.max(1, Math.min(4, num('fx', FLAT_EXT)));
  // Evaluation only: the per pixel path's coordinate diagnostic, and its work ceiling.
  DIR_DBG = Math.max(0, Math.min(2, Math.round(num('dbg', 0))));
  DIR_WORK = Math.max(1e8, Math.min(1e13, num('dw', DIR_WORK)));
  // Evaluation only: the work one band of a still per pixel pass may do. `bw=1e12` collapses the
  // pass into one band, which capture needs: a headless page renders about four frames in all.
  DIR_BAND_WORK = Math.max(1e8, Math.min(1e13, num('bw', DIR_BAND_WORK)));
  // Evaluation only: the per pixel path's still frame supersampling grid. `ss=1` is the A.
  DIR_SS = Math.max(1, Math.min(4, Math.round(num('ss', DIR_SS))));
  // Evaluation only: the batched GPU field behind the quadtree. `gf=0` is the A, a cut whose every
  // number comes from the CPU arithmetic. See planeFlush in src/52_cut_plane.js.
  if (q.has('gf')) gfOn = q.get('gf') === '1';
  GF_MAX_DEC = Math.max(0, Math.min(14, num('gd', GF_MAX_DEC)));
  // Evaluation only: the Gaussian view's shape numbers. `rw` is the outline half width in pixels,
  // `gp` the split threshold that sets how large the ellipses are.
  GAUSS_RING = Math.max(0.05, Math.min(4, num('rw', GAUSS_RING)));
  GAUSS_PX = Math.max(1, Math.min(40, num('gp', GAUSS_PX)));
  // The plane objects have their own size in this mode. See GAUSS_PX_PLANE in src/40_state.js.
  GAUSS_PX_PLANE = Math.max(1, Math.min(60, num('gpp', GAUSS_PX_PLANE)));
  // Evaluation only: a constant added to every splat's hue, in turns. Zero everywhere except in
  // the frames of a looping zoom video. See HUE_OFF in src/40_state.js.
  HUE_OFF = num('hu', 0);
  // Evaluation only: the splat path's anti-aliasing, five samples a tangle cell. `sa=0` is the A.
  SPLAT_AA = Math.max(0, Math.min(1, Math.round(num('sa', SPLAT_AA))));
  // Evaluation only: pin the stationary refinement level, so a capture shows the converged frame
  // without waiting for the ramp or depending on how fast the page's clock runs.
  if (q.has('st')) { stillPin = Math.max(0, Math.min(STILL_MAX, Math.round(num('st', 0)))); stillCuts = stillPin; }
  if (q.has('au')) cfg.autopilot = q.get('au') === '1';
  if (q.has('rb')) cfg.rebase = q.get('rb') === '1';
  // Ordering requirement: the query's camera is already in the anchor's coordinates, so the
  // anchor word is replayed here and the camera written afterwards. Replaying it second sends
  // the camera through the inverse maps twice, which at 93 rebases lands it about 1e28 away.
  const w = q.get('aw');
  if (w && maps) for (const part of w.split('.')) {
    const i = +part;
    if (i >= 0 && i < maps.length) rebaseInto(i);
  }
  cam.startDist = num('d0', cam.startDist);
  cam.dist = num('dz', cam.dist);
  if (q.has('tx')) cam.target.set([num('tx', 0), num('ty', 0), num('tz', 0)]);
  // Only a saved view names the aim. Without `gx`, `cam.target` at load is the object's centre of
  // mass, which for the Pythagoras canopy lies in the empty region inside the arc; the preset's
  // own aim is a fixed point of a branch word and lies on the attractor.
  if (q.has('gx')) cam.goal.set([num('gx', 0), num('gy', 0), num('gz', 0)]);
  else if (q.has('tx')) cam.goal.set(cam.target);
  if (q.has('cr')) {
    const v = q.get('cr').split(',').map(Number);
    if (v.length === 9 && v.every(isFinite)) { cam.R.set(v); orthoBasis(); }
  }
  // The aim's word and phase, so that a replayed view can keep descending.
  const wd = q.get('wd');
  if (wd && maps) {
    const lw = wd.split('.').map(Number).filter(i => i >= 0 && i < maps.length);
    if (lw.length) { cam.goalWord = lw; cam.goalPhase = Math.round(num('wp', 0)) % lw.length; }
  }
  projDist = cam.dist;
  return true;
}

/* ================================ loop ================================== */
// One step of camera motion: the target ease, the aim invariant, the autopilot descent and the
// rebase. Factored out so the headless tests drive the real descent.
let projDist = 1e300;
let parked = false;
let climbing = false;   // the flight has turned around at the precision wall
// A stale cut under covers the frame while climbing and over covers it while descending, and
// under covering shows the background where over covering only blurs, so the climb is the
// direction that has to be conservative. At 2.5 the frame grew sevenfold during one cut.
const CLIMB_RATE = 1.5;
let fixedDt = 0;             // set by the dt= query parameter, for headless capture

function advanceCamera(dt) {
  // Ease onto the descent target, then enforce the invariant |goal - target| <= 0.4 * cam.dist,
  // with goal a point on the object. The ease alone does not hold it: cam.dist shrinks at
  // cfg.rate while the residual decays at the ease rate, so a descent faster than the ease lets
  // the aim drift off in relative terms.
  const k = 1 - Math.exp(-Math.max(2, 3 * cfg.rate) * dt);
  let mv = 0;
  for (let i = 0; i < 3; i++) {
    const d = (cam.goal[i] - cam.target[i]) * k;
    cam.target[i] += d;
    mv += Math.abs(d);
  }
  const rx = cam.goal[0] - cam.target[0];
  const ry = cam.goal[1] - cam.target[1];
  const rz = cam.goal[2] - cam.target[2];
  const rem = Math.sqrt(rx * rx + ry * ry + rz * rz);
  const maxRem = cam.dist * 0.4;
  if (rem > maxRem) {
    const pull = 1 - maxRem / rem;
    cam.target[0] += rx * pull;
    cam.target[1] += ry * pull;
    cam.target[2] += rz * pull;
    mv += rem * pull;
  }
  if (mv > cam.dist * 1e-4) dirty = true;

  // The terrain aim rides at a fixed fraction of cam.dist, so it is refreshed as the distance
  // changes, autopilot or not.
  if (isTerrain()) cam.goal[1] = terrainAimHeight();
  else if (isPlane()) cam.goal[1] = 0;
  if (!cfg.autopilot) { zoomRate = 0; rebaseAll(); return; }
  // An IFS has no precision wall: a rebase resets the coordinates exactly. A flat field has no
  // self similarity to rebase into, so its wall is real: the plane coordinate is a double, and
  // once a screen pixel is narrower than the spacing of doubles at the aim there is nothing left
  // to resolve, a little over thirteen decades from the opening view. There the flight climbs back
  // out and re-aims at the top. Both readouts are recomputed every frame, not at the end of a
  // rebuild that can take tens of frames.
  stats.errPx = errPxNow();
  stats.logZoom = logZoom();
  const wall = cfg.precision === 'f32' ? 2.5 : 0.35;
  if (!climbing && stats.errPx > wall) climbing = true;
  // Where the camera will be when the cut about to start lands: `buildMs` CPU ms at one frameWork()
  // slice a frame is buildMs/frameWork() frames from now. Set here rather than in the frame loop,
  // so the headless harnesses that drive advanceCamera directly see the same prediction.
  //
  // Smoothed at gain 0.2. At gain one the lead is a two cycle oscillator: a large cut takes longer
  // to build, so the lead is longer, so the next cut is built further down the descent and comes
  // back coarser and smaller. Measured on the Pythagoras tree, splats per cut over a descent: 89k,
  // 83k, 110k, 103k, 136k, 127k, 168k, 157k, 208k, 194k, six to ten percent on top of the growth.
  const leadWant = Math.min(0.8, stats.buildMs / frameWork() * Math.min(Math.max(dt, 1 / 240), 0.05));
  buildLead += (leadWant - buildLead) * 0.2;
  // A cut covers the frame it was built for, so once the climbing frame is wider than the cut the
  // background shows through. Cap the growth at GROW e-folds per cut: at 2.4 e-folds a second
  // against a 0.55 s cut the frame grew 3.7 times and coverage fell to 2 percent, 126 of 900 dark.
  const GROW = 0.25;
  let cRate = CLIMB_RATE * cfg.rate;
  if (buildLead > 1e-3) cRate = Math.min(cRate, GROW / buildLead);
  zoomRate = climbing ? cRate : -cfg.rate;
  if (climbing) {
    cam.dist *= Math.exp(cRate * dt);
    if (cam.dist >= cam.startDist) {
      // Re-aim at the top, not at the turnaround: the 0.4 aim invariant snaps the target the whole
      // way in one frame when the goal is 0.03 away in world units and cam.dist is 1e-13, leaving
      // the camera on featureless plane. Measured that way, 126 of 900 frames dark with coverage
      // at 0.0 percent for a hundred frames of the climb.
      cam.dist = cam.startDist; climbing = false; projDist = 1e300;
      if (targets.length > 1) setGoal(targetIdx + 1);
    }
  } else {
    cam.dist *= Math.exp(-cfg.rate * dt);
  }
  parked = false;
  // The autopilot descends and does not turn. Rotation is on the drag.
  rebaseAll();
  // A field has no exact self similarity to rebase into, so its aim is re-projected onto the
  // boundary by one Newton step on the distance estimate whenever the camera distance has halved.
  if (isPlane() && !climbing && cam.dist < projDist * 0.5) {
    const P = PRESETS[cfg.preset];
    const c = projectToPlane(cam.goal[0], cam.goal[2], job.power, job.c0x, job.c0y,
                             !!P.julia, Math.max(job.maxIter, 4000));
    if (isFinite(c[0] + c[1])) { cam.goal[0] = c[0]; cam.goal[2] = c[1]; }
    projDist = cam.dist;
  }
  if (isTerrain() && !climbing && cam.dist < projDist * 0.5) {
    const c = projectToBoundary(cam.goal[0], cam.goal[2], Math.max(job.maxIter, 2000));
    if (isFinite(c[0] + c[1])) { cam.goal[0] = c[0]; cam.goal[2] = c[1]; }
    projDist = cam.dist;
  }
}

let last = 0, fpsEma = 60, hudAcc = 1, paramKey = '', rebuildAcc = 0, hzEma = 0;
let frameCount = 0;

// The per frame refinement slice, in ms. A cut bounded at msCap ms of CPU arrives msCap/slice
// frames later, so the slice is a latency budget: doubling it either halves the wait or doubles
// the detail. An IFS cut costs 20 to 30 ms and lands in five frames; a field cut at a two megapixel
// window wants a hundred, hence 1.8x for a moving plane, at 45 fps instead of 60. A still camera
// waits for nothing, so it gets 5x on a plane and 3x on a 3D object, whose split threshold halves
// over the stillness ramp and so has four times the cells to spend it on. A converged plane cut at
// the opening view is about 450 ms of CPU: fifty frames to land at 9 ms a frame, eighteen at 25 ms.
// WORK_MS is in src/40_state.js because startJob needs it too.
function frameWork() {
  // The smallest slice while nothing is on screen: the first cut runs for the most frames.
  if (!built.valid) return WORK_MS;
  if (stillCuts <= 0) return isPlane() ? WORK_MS * 1.8 : WORK_MS;
  return WORK_MS * (isPlane() ? 5 : 3);
}
// isPlane, not isField: the terrain shares the field machinery, but its cuts run to 900 ms and
// are bounded by nothing, so a slice 1.8 times bigger divided its buildLead prediction by the
// same factor and put 41 of 900 frames dark.

function paramSignature() {
  return cfg.preset + '|' + cfg.budget + '|' + cfg.splitPx + '|' + cfg.precision + '|' +
    cfg.sigma + '|' + cfg.fog + '|' + cfg.relief + '|' + cfg.iters + '|' +
    cfg.contour + '|' + cfg.height + '|' + cfg.power + '|' + cfg.cx + '|' + cfg.cy + '|' +
    cfg.gauss;
}

// Cuts that have reached the screen, read by the probe: differencing consecutive frames of a
// descent cannot separate a moving edge from a whole new cut, so the probe differences the frames
// a cut lands on separately from the rest.
let cutSerial = 0;
function upload(n) {
  cutSerial++;
  if (n > 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
    // cutFloats(), not FLOATS: a plane cut writes seven floats an instance against thirteen, so
    // the larger stride would upload three quarters garbage.
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, instances.subarray(0, n * cutFloats()));
  }
  reprojection();
  hzEma = hzEma ? hzEma + (1 / Math.max(rebuildAcc, 1e-3) - hzEma) * 0.2
                : 1 / Math.max(rebuildAcc, 1e-3);
  rebuildAcc = 0;
}

function frame(t) {
  // A fixed step when one is asked for. Headless Chrome under a virtual time budget barely advances
  // rAF timestamps, so a wall clock dt freezes the camera; it also makes a capture reproducible.
  const dt = fixedDt > 0 ? fixedDt : (last ? Math.min(0.1, (t - last) / 1000) : 0.016);
  last = t;
  frameCount++;
  // Before anything reads the object: a switch here is exactly what a change in the object menu
  // does between two frames, and the signature test below then takes the new object's branch.
  autoTick(dt);

  // The resource tier decides the pixel ratio, the pixel ceiling and the refinement slice, and
  // the governor moves the tier from the measured frame time. See src/35_perf.js.
  perfFrame(dt);
  WORK_MS = perfNow().work;
  const vp = perfViewport(canvas.clientWidth, canvas.clientHeight);
  const vw = vp[0], vh = vp[1];
  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width = vw; canvas.height = vh;
    resizeTargets(vw, vh);
    dirty = true;
  }

  advanceCamera(dt);
  updateBasis();
  // The per pixel path has no cut: no signature, no reprojection, no stepping, no upload.
  if (isDirect()) {
    stats.logZoom = logZoom();
    stats.errPx = errPxNow();
    stillT = (dirty || Math.abs(zoomRate) > 1e-6) ? 0 : stillT + dt;
    stillCuts = stillPin >= 0 ? stillPin
      : (stillT < STILL_T0 ? 0
        : Math.min(STILL_MAX, 1 + Math.floor(Math.log2(stillT / STILL_T0))));
    dirty = false;
    // Moving here is `stillCuts === 0` and not the raw motion test, so the evaluation pin reaches
    // this path too. At zoom 1e6 the moving iteration cap is 3 647 against the 88 800 the view
    // needs, and half the frame comes back as the out of iterations colour.
    drawDirect(vw, vh, stillCuts === 0, 1000 / Math.max(fpsEma, 1e-3));
    if (typeof window !== 'undefined' && window.__probe) window.__probe(gl, vw, vh);
    fpsEma += (1 / Math.max(dt, 1e-4) - fpsEma) * 0.06;
    hudAcc += dt;
    if (hudAcc > 0.15) { hudAcc = 0; hud(vw, vh); }
    progress();
    requestAnimationFrame(frame);
    return;
  }
  const sig = paramSignature();
  const drift = reprojection();
  rebuildAcc += dt;
  // Stillness in seconds and the level that follows: level k is reached after STILL_T0 * 2^(k-1)
  // seconds. The level triggers the rebuild, so no step of the ramp is skipped. A count of still
  // cuts cannot do this, since it advances only on the periodic 0.5 s refresh, which a slower cut
  // never meets and which never fires at all in a headless page.
  const moving = dirty || !built.valid || drift > 0.04 || Math.abs(zoomRate) > 1e-6;
  stillT = moving ? 0 : stillT + dt;
  const wantStill = stillPin >= 0 ? stillPin
    : (stillT < STILL_T0 ? 0
      : Math.min(STILL_MAX, 1 + Math.floor(Math.log2(stillT / STILL_T0))));

  if (sig !== paramKey) {
    // A change of object must not block. The first cut of a new object is an ordinary stepped job,
    // not a synchronous buildCut: a plane cut is seconds of escape time iteration, and seconds
    // inside the frame callback freeze the whole tab. Until it lands, the new object's background
    // is what is on screen. The stillness ramp resets with it, since a clock driven ramp is at its
    // top by the time anyone reaches the menu, which for a plane object is a latency ceiling of
    // twenty three seconds and sixteen times the iteration cap.
    job.active = false;
    stillT = 0; stillCuts = 0;
    // `buildLead` comes from the last cut's build time, and the last cut belonged to a different
    // object that could be a hundred times cheaper or dearer, so it does not carry across.
    stats.buildMs = 0; buildLead = 0;
    // `built` still describes the previous object's splats, in the previous frame.
    built.valid = false; built.count = 0;
    startJob(vw, vh);
    if (stepJob(frameWork())) upload(finishJob());
    paramKey = sig; dirty = false;
  } else if (job.active) {
    if (stepJob(frameWork())) upload(finishJob());
  } else if (moving || wantStill !== stillCuts || rebuildAcc > (stillCuts > 0 ? 4 : 0.5)) {
    // The periodic refresh is a backstop; real changes set `dirty` or show up as drift. Rebuilding
    // a still plane object costs 450 ms of the 500 available, hence 4 s while still, 0.5 s moving.
    stillCuts = wantStill;
    startJob(vw, vh);
    dirty = false;
    if (stepJob(frameWork())) upload(finishJob());
  }
  draw(vw, vh, built.count);
  // The batch field program, linked once a session, after a frame has been drawn. See gfWarm.
  if (isPlane()) gfWarm();
  // Where a harness reads the frame that actually reached the screen, through the real rasterizer,
  // the 24 bit depth buffer and the float16 accumulation.
  if (typeof window !== 'undefined' && window.__probe) window.__probe(gl, vw, vh);

  fpsEma += (1 / Math.max(dt, 1e-4) - fpsEma) * 0.06;
  hudAcc += dt;
  if (hudAcc > 0.15) { hudAcc = 0; hud(vw, vh); }
  // Every frame rather than on the readout's 150 ms tick, where the bar steps rather than fills.
  progress();
  requestAnimationFrame(frame);
}

// The cut, which decides which splats exist and evaluates the escape time field for the plane
// objects, runs on the CPU in one JavaScript thread: no worker, no GPU compute path, which is what
// bounds a plane object's resolution, see knowledge/mandelbrot_speed.md. The raster runs on the
// GPU named here. WEBGL_debug_renderer_info is usually an ANGLE wrapper around the real device, so
// the useful part is in the middle; trim rather than truncate.
function gpuName() {
  let s = String(stats.gpu || '?');
  const m = /^ANGLE \((.*)\)$/.exec(s);
  if (m) s = m[1];
  s = s.replace(/\s*(Direct3D\d+|OpenGL|Vulkan)[^,]*$/i, '');
  s = s.replace(/\s*\((\d+x)?[0-9a-f]{4}\)/gi, '');
  const parts = s.split(',').map(t => t.trim()).filter(Boolean);
  // ANGLE repeats the vendor inside the device name: "NVIDIA, NVIDIA GeForce RTX 3080".
  if (parts.length > 1 && parts[1].toLowerCase().startsWith(parts[0].toLowerCase())) parts.shift();
  return parts.join(', ') || '?';
}

/* --------------------------- the progress bar ---------------------------- */
// Two waits on one bar, and the label says which. Building is `job.ms / job.msCap`, monotone
// within a cut and bounded, which splats over budget is not, since a converged cut can be a tenth
// of the budget. Sharpening is the accumulated stillness in the log of the time, because level k
// is reached at STILL_T0 * 2^(k-1) seconds. Neither is drawn for an IFS: a cut there is 8 to 14 ms
// and the bar would flash once a frame.
let progLast = -1, progWhat = '';
function progress() {
  const bar = el('prog');
  if (!bar) return;
  // The per pixel objects have no cut, and their band counter already reports their progress.
  if (!isField() || isDirect()) {
    if (!bar.hidden) { bar.hidden = true; progLast = -1; progWhat = ''; }
    return;
  }
  let u = -1, what = '';
  if (job.active && job.msCap > 0 && job.msCap < 1e8) {
    u = Math.min(1, job.ms / job.msCap);
    what = 'building';
  } else if (stillCuts < STILL_MAX && stillT > 0) {
    // the time the last level is reached at, and where this stillness sits in the log of it
    const tEnd = STILL_T0 * Math.pow(2, STILL_MAX - 1);
    u = Math.min(1, Math.log2(1 + stillT / STILL_T0) / Math.log2(1 + tEnd / STILL_T0));
    what = 'sharpening ' + (stillCuts + 1) + ' of ' + STILL_MAX;
  }
  if (u < 0) {
    if (!bar.hidden) { bar.hidden = true; progLast = -1; progWhat = ''; }
    return;
  }
  if (bar.hidden) bar.hidden = false;
  // Write only when something moved: a style write that changes nothing still costs a recalc.
  if (Math.abs(u - progLast) > 0.004) {
    progLast = u;
    const b = el('prog-b');
    if (b) b.style.width = (u * 100).toFixed(1) + '%';
  }
  if (what !== progWhat) {
    progWhat = what;
    const l = el('prog-l');
    if (l) l.textContent = what;
  }
}

function hud(vw, vh) {
  if (!el('h-splats')) return;
  const set = (id, v) => { const e = el(id); if (e) e.textContent = v; };
  set('h-gpu', gpuName());
  // The cut, and what it costs. `evals` is field evaluations for a plane object and is
  // meaningless for an IFS, where the cost is the tree walk itself.
  if (isDirect()) {
    // No cut to report, so report the work: pixels times the iteration cap bounds the frame cost.
    const work = vw * vh * dirIters;
    set('h-cut', 'GPU, per pixel, ' + fmt(vw * vh) + ' orbits x ' + fmt(dirIters) +
      ' iters = ' + fmt(work) + ', double single');
    set('l-splats', 'pixels');
    set('h-splats', fmt(vw * vh));
  } else {
    set('h-cut', 'CPU, 1 thread, ' + stats.buildMs.toFixed(0) + ' ms @ ' + hzEma.toFixed(0) + ' Hz' +
      (isField() ? ', ' + fmt(stats.evals) + ' cells, ' + fmt(stats.itersReal) + ' iters' : ''));
    // The Gaussian view raises the split threshold, so the splat count drops. The label says so
    // here because the caption that explains it is suppressed on a phone.
    set('l-splats', cfg.gauss ? 'splats (reduced)' : 'splats');
    set('h-splats', fmt(stats.splats));
  }
  set('h-object', PRESETS[cfg.preset].name);
  set('h-fps', fpsEma.toFixed(0));
  set('h-ms', (1000 / Math.max(fpsEma, 1e-3)).toFixed(1) + ' ms');
  set('h-build', stats.buildMs.toFixed(1) + ' ms @ ' + hzEma.toFixed(0) + ' Hz');
  set('h-res', vw + '×' + vh + ' ' + perfName());
  // The picker is written from the state, so a tier from the URL or the governor is what shows.
  const qsel = el('quality');
  if (qsel) qsel.value = perfAuto ? 'auto' : perfNow().name;
  const lz = stats.logZoom;
  set('h-zoom', '10^' + (lz > 999 ? lz.toFixed(0) : lz.toFixed(2)));
  set('h-depth', stats.dmin + '–' + stats.dmax);
  set('h-err', (stats.errPx < 0.001 ? '< 0.001 px' : stats.errPx.toFixed(3) + ' px') +
    (parked ? ' (parked)' : ''));
  const e = el('h-err'); if (e) e.dataset.warn = stats.errPx > 0.25 ? '1' : '0';
  set('h-fmt', stats.format);
  if (isTerrain()) {
    set('h-extra', fmt(stats.evals) + ' cells, ' + fmt(stats.iters) + ' iters, max ' + stats.maxIter);
  } else if (isDirect()) {
    set('h-extra', 'iteration cap ' + dirIters + ', still ' + stillCuts +
      ', band ' + dirBand + '/' + dirBands + (dirDone ? ' done' : '') +
      (dirSS > 1 ? ', ' + dirSS + 'x' + dirSS + ' AA' : ''));
  } else if (isPlane()) {
    // The iteration cap and the ramp position. The cap decides whether a region comes back resolved
    // or as one flat colour, and moves by a factor of sixteen when the camera stops.
    set('h-extra', 'maxIter ' + stats.maxIter + ', still ' + stillCuts +
      ', cache ' + (stats.fieldHit * 100).toFixed(0) + '%' +
      // Where the field came from: the difference between a cut that costs 8 seconds and one that
      // costs 0.9, and past GF_MAX_DEC the GPU path turns itself off.
      (stats.gfQueries > 0
        ? ', field GPU ' + fmt(stats.gfQueries) + ' in ' + stats.gfBatches + ' @ ' +
          stats.gfMs.toFixed(0) + ' ms'
        : ', field CPU' + (gfWhy ? ' (' + gfWhy + ')' : '')));
  } else {
    set('h-extra', (maps ? maps.length : 0) + ' maps, D = ' + mDim.toFixed(3) +
      ', ' + stats.rebases + ' rebases');
  }
}

/* ============================ the page's theme =========================== */
// The viewer normally runs as a same origin iframe on marcelpadilla.com and repeats the host's
// rule: light or dark from localStorage['theme'], the OS as the default. Reading it directly makes
// the first frame correct even if the parent never posts; the message channel below only keeps it
// correct afterwards. localStorage throws rather than returning null in a sandboxed iframe or with
// cookies blocked, and touching the property itself can throw, so the whole read is in the try.
function themeResolve() {
  let pref = 'system';
  try {
    const v = localStorage.getItem('theme');
    if (v === 'light' || v === 'dark') pref = v;
  } catch (e) { /* denied: follow the OS */ }
  if (pref !== 'system') return pref;
  return (typeof matchMedia !== 'undefined' &&
          matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
}
function themeApply(t) {
  if (typeof document === 'undefined') return;
  document.documentElement.setAttribute('data-vp-theme', t === 'light' ? 'light' : 'dark');
}
function themeInit() {
  if (typeof window === 'undefined') return;
  themeApply(themeResolve());
  // The storage event never fires in the frame that wrote it, so a same origin iframe hears nothing
  // when the parent's toggle writes localStorage; the parent has to post.
  // Only from the embedding page, and only one of the two names it is allowed to say. The origin
  // is not checked because the viewer is meant to be embeddable from anywhere; restricting the
  // sender to the parent and the payload to a closed set is what keeps that safe.
  window.addEventListener('message', (e) => {
    if (e.source !== window.parent) return;
    const d = e.data;
    if (d && (d.fsTheme === 'light' || d.fsTheme === 'dark')) themeApply(d.fsTheme);
  });
  // And the OS, for a viewer opened on its own with no stored choice.
  if (typeof matchMedia !== 'undefined') {
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const on = () => themeApply(themeResolve());
    if (mq.addEventListener) mq.addEventListener('change', on);
    else if (mq.addListener) mq.addListener(on);
  }
}

/* ================================ boot ================================== */
function boot() {
  const cv = el('gl');
  if (!initGL(cv)) {
    const warn = el('nogl');
    if (warn) warn.hidden = false;
    return;
  }
  themeInit();
  // Before applyURL, so an explicit gs= in the query still wins over what was remembered.
  cfg.gauss = gaussRecall();
  bindUI();
  if (!applyURL()) loadPreset(cfg.preset, false);
  // After the object is loaded, so the walk starts from what is on screen rather than from the
  // default preset that was there while the query was being read.
  if (autoOn) autoShuffle();
  setUIHidden(false);
  syncUI();
  attachInput(cv);
  requestAnimationFrame(frame);
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
}

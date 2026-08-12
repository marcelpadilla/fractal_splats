// Load the renderer into the REAL script realm, not a vm sandbox, and hand back its internals for
// testing. The sources are concatenated here exactly as build.mjs concatenates them, so the tests
// run straight from a clone and can never be checking a stale build.
//
// The realm matters for every timing measured through this file. `vm.createContext` gives the
// script a contextified global, and every reference it makes to a global builtin, `Math.sqrt`
// included, goes through an interceptor: measured that way the refinement kernel looks about eight
// times slower than it is. `runInThisContext` shares the real global, so what comes out is what the
// browser will see.
//
// The DOM is stubbed rather than emulated. Nothing under test draws, and the renderer already
// guards every element lookup, so a proxy that answers plausibly to whatever the UI layer asks for
// is enough.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const NOOP = () => {};
const stubEl = () => new Proxy({ dataset: {}, style: {} }, {
  get: (t, k) => {
    if (k in t) return t[k];
    if (k === 'getContext') return () => null;                 // no GPU here
    if (['addEventListener', 'select', 'appendChild', 'setPointerCapture',
         'setAttribute', 'removeAttribute', 'focus', 'blur'].includes(k)) return NOOP;
    if (k === 'classList') return { toggle: NOOP, contains: () => false, add: NOOP };
    return '';
  },
  set: (t, k, v) => { t[k] = v; return true; },
});

globalThis.requestAnimationFrame = NOOP;
globalThis.location = { search: '' };
if (!globalThis.navigator) globalThis.navigator = { clipboard: null };   // node 22 defines it
globalThis.history = { replaceState: NOOP };
globalThis.document = {
  readyState: 'complete',
  getElementById: () => stubEl(),
  querySelectorAll: () => [],
  createElement: () => stubEl(),
  addEventListener: NOOP,
};
globalThis.window = { addEventListener: NOOP, devicePixelRatio: 1 };

const EXPORTS = ['loadPreset', 'buildCut', 'startJob', 'stepJob', 'finishJob', 'advanceCamera',
  'pickTarget', 'setGoal', 'rebaseAll', 'rebaseStep', 'rebaseInto', 'rebaseNeeded', 'logZoom',
  'measureNorm', 'measureDimension', 'rootMoments', 'supportRadius', 'solveFixedPoint', 'refreshGoal',
  'mandelField', 'projectToBoundary', 'kernelConst', 'turnCamera', 'orthoBasis', 'updateBasis',
  'camFromAngles', 'mat3Mul', 'mat3MulT', 'mat3Vec', 'sym3Inv', 'frameWork',
  'cfg', 'cam', 'anchor', 'stats', 'pool', 'job', 'basis', 'emitted', 'instances', 'fld',
  'PRESETS', 'palette', 'repro', 'built', 'FLOATS', 'FLAT_FLOATS', 'FADE_BAND'];

const srcDir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'src');
const src = fs.readdirSync(srcDir).filter(f => f.endsWith('.js')).sort()
  .map(f => fs.readFileSync(path.join(srcDir, f), 'utf8'))
  .join(String.fromCharCode(10));
// Everything above runs in the script's own scope, so the only way out is a closure created
// inside it. `eval` here is deliberate: a test needs live bindings, not a snapshot.
const tail = '\n;globalThis.__fs = { get: n => eval(n), set: (n, v) => eval(n + " = v"), api: {' +
  EXPORTS.map(n => `${n}: typeof ${n} === 'undefined' ? undefined : ${n}`).join(', ') + '} };\n';
vm.runInThisContext(src + tail, { filename: 'fractal_splats' });

export const api = globalThis.__fs.api;
// Read or evaluate anything in the script scope, for the cases where a live binding is needed
// rather than the value captured above.
export const get = globalThis.__fs.get;
export const set = globalThis.__fs.set;

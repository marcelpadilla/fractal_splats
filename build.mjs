// Builds index.html from src/*.js and page.part.html.
//
//   node build.mjs
//
// The sources are concatenated in filename order into one script and inlined into the page, so the
// result is a single file that needs no server, no bundler and no network. They share one scope on
// purpose: the refinement kernel reads flat typed arrays out of the enclosing scope, and a module
// boundary would put a property load in the inner loop. Numbering is by layer, ten apart, so a new
// layer can be slotted in between two existing ones.
//
// index.html is committed, so the demo runs straight from a clone. Rebuild after editing src/.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));   // handles spaces in the path
const srcDir = path.join(here, 'src');

const parts = fs.readdirSync(srcDir).filter(f => f.endsWith('.js')).sort();
if (!parts.length) throw new Error('no sources in src/');
const app = parts.map(f => {
  const s = fs.readFileSync(path.join(srcDir, f), 'utf8');
  return '/* ==== src/' + f + ' ' + '='.repeat(Math.max(0, 60 - f.length)) + ' */\n' + s;
}).join('\n');

const part = fs.readFileSync(path.join(here, 'page.part.html'), 'utf8');
if (!part.includes('<!--#SCRIPT#-->')) throw new Error('page.part.html lost its script marker');
// A closing script tag anywhere in a source, even inside a string, would end the inlined block
// early and take the rest of the program with it as markup.
if (/<\/script/i.test(app)) throw new Error('a source contains a closing script tag');

const body = part.replace('<!--#SCRIPT#-->', '<script>\n' + app + '\n</script>');

const viewer = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fractal splats</title>
<meta name="description" content="Fractals rendered as Gaussian splats, refined on demand as the camera descends. An affine iterated function system with exact anchor rebasing, and a distance estimated terrain over the Mandelbrot set.">
</head>
<body>
${body}
</body>
</html>
`;

// Parse what is about to be written. A patch that emits a literal newline inside a string literal
// is a syntax error that takes the whole page down, and a black canvas is indistinguishable from a
// renderer bug. A parse is cheap and catches the entire class.
for (const block of (viewer.match(/<script>[\s\S]*?<\/script>/g) || [])) {
  const src = block.replace(/^<script>/, '').replace(/<\/script>$/, '');
  try { new Function(src); } catch (e) { throw new Error('index.html will not parse: ' + e.message); }
}

fs.writeFileSync(path.join(here, 'index.html'), viewer);
console.log('sources     ' + parts.join(' '));
console.log('index.html  ' + (viewer.length / 1024).toFixed(1) + ' kB');

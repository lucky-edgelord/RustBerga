// usage: node node-test.mjs <wasm> <model.bin> <vocab.spm> <lex.bin> < input.txt
import { readFileSync } from 'node:fs';
import { Translator } from './rustberger.mjs';
const [wasm, model, vocab, lex] = process.argv.slice(2);
const input = readFileSync(0, 'utf8');
let t0 = performance.now();
const tr = await Translator.create({ wasm: readFileSync(wasm), model: readFileSync(model), vocab: readFileSync(vocab), shortlist: readFileSync(lex) });
const load = performance.now() - t0;
tr.translate('warm up.');
t0 = performance.now();
const out = tr.translate(input);
const ms = performance.now() - t0;
process.stdout.write(out);
console.error(`simd128=${tr.simd} load ${load.toFixed(0)}ms translate ${ms.toFixed(0)}ms (${(input.split(/\s+/).filter(Boolean).length / (ms / 1000)).toFixed(1)} words/s)`);

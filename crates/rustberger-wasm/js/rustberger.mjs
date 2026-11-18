// Minimal loader for rustberger_wasm.wasm. Works in browsers (use in a Worker) and Node >= 18.
const enc = new TextEncoder();
const dec = new TextDecoder();

export class Translator {
  #x; #h;
  constructor(x, h) { this.#x = x; this.#h = h; }

  /** @param {{wasm: BufferSource|WebAssembly.Module, model: Uint8Array, vocab: Uint8Array, shortlist?: Uint8Array, beam?: number}} o */
  static async create({ wasm, model, vocab, shortlist, beam = 1 }) {
    const mod = wasm instanceof WebAssembly.Module ? wasm : await WebAssembly.compile(wasm);
    const { exports: x } = await WebAssembly.instantiate(mod, {});
    const put = (b) => { const p = x.rb_alloc(b.length); new Uint8Array(x.memory.buffer, p, b.length).set(b); return p; };
    const [pm, pv] = [put(model), put(vocab)];
    const sl = shortlist ?? new Uint8Array(0);
    const ps = sl.length ? put(sl) : 0;
    const h = x.rb_load(pm, model.length, pv, vocab.length, ps, sl.length, beam);
    x.rb_free(pm, model.length); x.rb_free(pv, vocab.length); if (ps) x.rb_free(ps, sl.length);
    if (h < 0) throw new Error(dec.decode(new Uint8Array(x.memory.buffer, x.rb_error_ptr(), x.rb_error_len())));
    const t = new Translator(x, h);
    t.simd = x.rb_has_simd128() === 1;
    return t;
  }

  translate(text) {
    const x = this.#x, b = enc.encode(text);
    const p = x.rb_alloc(b.length);
    new Uint8Array(x.memory.buffer, p, b.length).set(b);
    const rc = x.rb_translate(this.#h, p, b.length);
    x.rb_free(p, b.length);
    const view = (ptr, len) => dec.decode(new Uint8Array(x.memory.buffer, ptr, len));
    if (rc !== 0) throw new Error(view(x.rb_error_ptr(), x.rb_error_len()));
    return view(x.rb_result_ptr(), x.rb_result_len());
  }

  free() { this.#x.rb_unload(this.#h); }
}

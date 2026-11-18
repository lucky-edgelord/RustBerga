# RustBerga

A pure-Rust port of the [`app/`](https://github.com/browsermt/bergamot-translator/tree/main/app) CLI from
[bergamot-translator](https://github.com/browsermt/bergamot-translator): read text on stdin, translate it with a
Marian/Bergamot model, print the target text. No C++, no FFI, no BLAS. It also builds to a small WebAssembly module.

## Features

- Loads Mozilla *Firefox Translations* student models (Marian binary, `intgemm8`, SSRU decoder) and the same Bergamot YAML configs the C++ app uses.
- Own SentencePiece (unigram + `nmt_nfkc` charsmap) tokenizer: 330/330 sentences identical to python `sentencepiece`.
- Batched beam/greedy search, lexical shortlist, rayon parallelism (native).
- Hand-written int8 GEMM kernels with runtime CPU detection:

| Target | Kernel |
|---|---|
| x86-64 | AVX2, AVX-VNNI, AVX512-VNNI (256 and 512-bit), scalar fallback; all bit-exact vs scalar in tests |
| aarch64 | NEON, NEON `sdot` (inline asm, interleaved layout) |
| wasm32 | SIMD128 (build with `+simd128`) |
| GPU (native, opt-in) | `wgpu` (Vulkan/Metal/DX12/GL), WGSL 4x4 register-blocked int8 GEMM, `dot4I8Packed` when available |

- Float ops (attention, layer norm, softmax, SSRU) are SIMD-dispatched too.

## Build and run

```sh
cargo build --release -p rustberger-cli
echo "Hello, how are you today?" | target/release/rustberger --model-config-paths config.yml
# or: --model model.bin --vocab vocab.spm [--shortlist lex.bin] --cpu-threads 4 --beam 4 --stats
```

Models: download a Firefox Translations pair (e.g. `enes`) from Mozilla's
[firefox-translations-models](https://github.com/mozilla/firefox-translations-models) and unpack `model.*.bin`,
`vocab.*.spm`, `lex.*.bin`.

GPU: `cargo build --release -p rustberger-cli --features gpu`, then pass `--gpu`. Only matmuls with at least 48 rows are offloaded
(activations are quantised on the CPU and uploaded per call), so expect gains mainly on discrete or fast GPUs with large batches.

### WebAssembly

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --release -p rustberger-wasm --target wasm32-unknown-unknown
node crates/rustberger-wasm/js/node-test.mjs
```

The module (~336 KB) exposes a plain C ABI (`rb_load`, `rb_translate`, ...) wrapped by `js/rustberger.mjs`
(`Translator.create({wasm, model, vocab, shortlist, beam})`, `.translate(text)`, `.free()`). It is single-threaded and CPU-only.

## Benchmark vs C++ bergamot

Same model (Mozilla en-es tiny), same config and input (28 lines), C++ built without BLAS using its intgemm path,
median of 5 runs, 2-core x86-64 machine (noisy; treat as indicative).

| | C++ 1 thread | Rust 1 thread | C++ 2 threads | Rust 2 threads |
|---|---|---|---|---|
| Startup (s) | 0.190 | 0.092 | 0.188 | 0.075 |
| Total (s) | 0.645 | 0.428 | 0.632 | 0.244 |
| Translate only (s) | 0.455 | 0.336 | 0.444 | 0.169 |
| Peak RSS (MB) | 234 | 125 | 234 | 143 |

Translation is about **1.35x** (1 thread) and **2.6x** (2 threads) faster; startup about 2-2.5x faster; memory about half.
Reproduce with `scripts/bench.py` conventions: same YAML, `--cpu-threads N`.

## Accuracy

Outputs are not bit-identical to C++: a few sentences flip between near-tied beam hypotheses (a 0.1% scale perturbation does the same
in either implementation). Translations are of equivalent quality (about 98.5% textual similarity on the test set).

## Limitations (honest list)

- **NPU / WebNN: not implemented.** WebNN is still a W3C Candidate Recommendation Draft and Chrome origin-trial only. Browser WebGPU/WebNN need async readback,
  which the current synchronous C ABI cannot express; that would require an async redesign.
- Native GPU offload is partial (large-M matmuls only, per-call upload). Validated only on a software adapter (llvmpipe), not on real GPU hardware.
- ARM kernels cross-compile and the tile logic is unit-tested via an emulated `sdot`, but they have **not been run on ARM hardware**. `i8mm` is not used.
- Only int8 SSRU student models are supported. No HTML translation, alignments or quality estimation.
- The sentence splitter is rule-based and differs from C++ `ssplit`.

## Notes on BergaRust

[zh-ww/BergaRust](https://github.com/zh-ww/BergaRust) is an Axum server over the C++ library via FFI. Ideas worth borrowing later:
`whichlang` language detection, API-compatible endpoints, ISO-639-1 model keying.

## License

MPL-2.0 (same as bergamot-translator).

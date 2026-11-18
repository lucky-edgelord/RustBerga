//! Optional coarse timers (native only; `Instant` is unavailable on wasm32-unknown-unknown).
use std::sync::atomic::{AtomicU64, Ordering};

pub const ENCODE: usize = 0;
pub const DEC_LAYERS: usize = 1;
pub const DEC_LOGITS: usize = 2;
pub const SEARCH: usize = 3;
pub const TOKENIZE: usize = 4;
pub const QUANT: usize = 5;
pub const GEMM: usize = 6;
pub const NAMES: [&str; 7] = ["encoder", "decoder layers", "output layer", "beam search", "tokenize", "quantize(all)", "int8 gemm(all, excl. output)"];

static NS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

pub struct Timer(usize, #[cfg(not(target_arch = "wasm32"))] std::time::Instant);

pub fn start(slot: usize) -> Timer {
    Timer(slot, #[cfg(not(target_arch = "wasm32"))] std::time::Instant::now())
}

impl Drop for Timer {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        NS[self.0].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Accumulated seconds per slot (summed over threads, so can exceed wall-clock time).
pub fn seconds() -> [f64; 7] {
    let mut out = [0.0; 7];
    for (o, n) in out.iter_mut().zip(&NS) {
        *o = n.load(Ordering::Relaxed) as f64 * 1e-9;
    }
    out
}

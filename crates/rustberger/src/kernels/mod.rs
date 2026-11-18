//! Compute kernels: int8 GEMM with per-ISA SIMD back-ends plus small f32 helpers.
//!
//! Quantisation scheme (matches intgemm / Marian `int8shiftAlphaAll`):
//!   A_q = clamp(round_even(A * qa), -127, 127),  B_q stored at load time,
//!   C   = (A_q . B_q) / (qa * qb) + bias
//! Exact i32 accumulation is used, so results are deterministic across ISAs.

mod scalar;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod x86;

/// Weight matrix `[N][K]` (K padded with zeros to a multiple of 32, N padded to a multiple of 4).
pub struct PackedB {
    pub n: usize,
    pub k: usize,
    pub kp: usize,
    pub np: usize,
    pub data: Vec<i8>,
    pub qm: f32,
}

pub const KALIGN: usize = 32;

pub fn pad_k(k: usize) -> usize {
    (k + KALIGN - 1) / KALIGN * KALIGN
}

impl PackedB {
    /// `data` is `[n][k]` row-major int8.
    pub fn from_nk(data: &[i8], n: usize, k: usize, qm: f32) -> Self {
        let kp = pad_k(k);
        let np = (n + 3) / 4 * 4;
        let mut d = vec![0i8; np * kp];
        for r in 0..n {
            d[r * kp..r * kp + k].copy_from_slice(&data[r * k..(r + 1) * k]);
        }
        PackedB { n, k, kp, np, data: d, qm }
    }

    /// Gather a subset of rows (output columns) into a new packed matrix.
    pub fn gather_rows(&self, rows: &[u32]) -> PackedB {
        let n = rows.len();
        let np = (n + 3) / 4 * 4;
        let mut d = vec![0i8; np * self.kp];
        for (i, &r) in rows.iter().enumerate() {
            let r = r as usize;
            d[i * self.kp..(i + 1) * self.kp].copy_from_slice(&self.data[r * self.kp..(r + 1) * self.kp]);
        }
        PackedB { n, k: self.k, kp: self.kp, np, data: d, qm: self.qm }
    }
}

/// Which instruction set the int8 GEMM dispatches to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    Avx2,
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    AvxVnni,
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    Avx512Vnni,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "aarch64")]
    NeonDot,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Simd128,
}

pub fn detect_isa() -> Isa {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") && is_x86_feature_detected!("avx2") {
            return Isa::Avx512Vnni;
        }
        if is_x86_feature_detected!("avxvnni") && is_x86_feature_detected!("avx2") {
            return Isa::AvxVnni;
        }
        if is_x86_feature_detected!("avx2") {
            return Isa::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return Isa::NeonDot;
        }
        return Isa::Neon;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return Isa::Simd128;
    }
    #[allow(unreachable_code)]
    Isa::Scalar
}

pub fn isa() -> Isa {
    use std::sync::OnceLock;
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(|| match std::env::var("RUSTBERGER_ISA").ok().as_deref() {
        Some("scalar") => Isa::Scalar,
        _ => detect_isa(),
    })
}

/// Quantise `m x k` f32 rows into int8 rows with stride `kp` (zero padded).
pub fn quantize(a: &[f32], m: usize, k: usize, qm: f32, kp: usize, out: &mut Vec<i8>) {
    out.clear();
    out.resize(m * kp, 0);
    const MAGIC: f32 = 12582912.0; // 1.5 * 2^23: add/sub rounds to nearest-even
    for r in 0..m {
        let src = &a[r * k..(r + 1) * k];
        let dst = &mut out[r * kp..r * kp + k];
        for (d, &x) in dst.iter_mut().zip(src) {
            let y = (x * qm + MAGIC) - MAGIC;
            *d = y.clamp(-127.0, 127.0) as i8;
        }
    }
}

/// C[m][n] = (A_q . B_q^T) * (1 / (qa * qb)) + bias. `a` has row stride `b.kp`.
pub fn gemm(a: &[i8], m: usize, b: &PackedB, qa: f32, bias: Option<&[f32]>, out: &mut [f32]) {
    gemm_isa(isa(), a, m, b, qa, bias, out)
}

/// Same as [`gemm`] with an explicit instruction set (used by tests and benchmarks).
pub fn gemm_isa(isa: Isa, a: &[i8], m: usize, b: &PackedB, qa: f32, bias: Option<&[f32]>, out: &mut [f32]) {
    debug_assert!(a.len() >= m * b.kp);
    debug_assert!(out.len() >= m * b.n);
    if m == 0 || b.n == 0 {
        return;
    }
    let scale = 1.0 / (qa * b.qm);
    let tiles = b.np / 4;
    let work = m * b.np * b.kp;
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    if work >= (1 << 21) && tiles >= 8 {
        use rayon::prelude::*;
        let threads = rayon::current_num_threads().max(1);
        let chunk = ((tiles + threads * 2 - 1) / (threads * 2)).max(1);
        let ptr = SendPtr(out.as_mut_ptr());
        let ranges: Vec<(usize, usize)> = (0..tiles).step_by(chunk).map(|s| (s, (s + chunk).min(tiles))).collect();
        ranges.into_par_iter().for_each(|(t0, t1)| {
            let ptr = ptr;
            // SAFETY: tile ranges write disjoint output columns.
            unsafe { run(isa, a, m, b, scale, bias, ptr.0, t0, t1) };
        });
        return;
    }
    let _ = work;
    unsafe { run(isa, a, m, b, scale, bias, out.as_mut_ptr(), 0, tiles) };
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[allow(clippy::too_many_arguments)]
unsafe fn run(isa: Isa, a: &[i8], m: usize, b: &PackedB, scale: f32, bias: Option<&[f32]>, out: *mut f32, t0: usize, t1: usize) {
    match isa {
        Isa::Scalar => scalar::gemm(a, m, b, scale, bias, out, t0, t1),
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        Isa::Avx2 => x86::gemm_avx2(a, m, b, scale, bias, out, t0, t1),
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        Isa::AvxVnni => x86::gemm_avxvnni(a, m, b, scale, bias, out, t0, t1),
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        Isa::Avx512Vnni => x86::gemm_avx512vnni(a, m, b, scale, bias, out, t0, t1),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::gemm(a, m, b, scale, bias, out, t0, t1),
        #[cfg(target_arch = "aarch64")]
        Isa::NeonDot => neon::gemm_dot(a, m, b, scale, bias, out, t0, t1),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => wasm::gemm(a, m, b, scale, bias, out, t0, t1),
    }
}

// ---------------------------------------------------------------- f32 helpers

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// y += alpha * x
#[inline]
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (yy, xx) in y.iter_mut().zip(x) {
        *yy += alpha * xx;
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - mx).exp();
        s += *v;
    }
    let inv = 1.0 / s;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Marian layer norm over the last dimension: gamma * (x - mean) / sqrt(var + eps) + beta, eps = 1e-9.
pub fn layer_norm(x: &mut [f32], dim: usize, gamma: &[f32], beta: &[f32]) {
    for row in x.chunks_exact_mut(dim) {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let mut var = 0.0;
        for v in row.iter() {
            let d = v - mean;
            var += d * d;
        }
        var /= dim as f32;
        let inv = 1.0 / (var + 1e-9).sqrt();
        for ((v, g), b) in row.iter_mut().zip(gamma).zip(beta) {
            *v = (*v - mean) * inv * g + b;
        }
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    fn available() -> Vec<Isa> {
        let mut v = vec![Isa::Scalar];
        let d = detect_isa();
        if d != Isa::Scalar {
            v.push(d);
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("avx2") {
                v.push(Isa::Avx2);
            }
            if is_x86_feature_detected!("avxvnni") && is_x86_feature_detected!("avx2") {
                v.push(Isa::AvxVnni);
            }
        }
        v.dedup();
        v
    }

    #[test]
    fn int8_gemm_matches_scalar_on_all_isas() {
        let mut seed = 42;
        for &(m, n, k) in &[(1, 1, 5), (1, 7, 32), (3, 9, 100), (5, 256, 256), (17, 33, 1536), (64, 130, 257)] {
            let bdata: Vec<i8> = (0..n * k).map(|_| (lcg(&mut seed) % 255) as i32 as i8 as i32).map(|x| x.clamp(-127, 127) as i8).collect();
            let b = PackedB::from_nk(&bdata, n, k, 40.0);
            let af: Vec<f32> = (0..m * k).map(|_| (lcg(&mut seed) % 2000) as f32 / 1000.0 - 1.0).collect();
            let mut aq = Vec::new();
            quantize(&af, m, k, 100.0, b.kp, &mut aq);
            let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
            let mut reference = vec![0f32; m * n];
            gemm_isa(Isa::Scalar, &aq, m, &b, 100.0, Some(&bias), &mut reference);
            // exact integer cross-check of the scalar reference itself
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0i32;
                    for t in 0..k {
                        acc += aq[i * b.kp + t] as i32 * bdata[j * k + t] as i32;
                    }
                    let want = acc as f32 / (100.0 * 40.0) + bias[j];
                    assert!((reference[i * n + j] - want).abs() <= 1e-4 * want.abs().max(1.0));
                }
            }
            for isa in available() {
                let mut got = vec![0f32; m * n];
                gemm_isa(isa, &aq, m, &b, 100.0, Some(&bias), &mut got);
                assert_eq!(got, reference, "isa {isa:?} m={m} n={n} k={k}");
            }
        }
    }

    #[test]
    fn quantize_rounds_to_even_and_clamps() {
        let mut out = Vec::new();
        quantize(&[0.5, 1.5, 2.5, -0.5, -1.5, 1000.0, -1000.0], 1, 7, 1.0, 32, &mut out);
        assert_eq!(&out[..7], &[0, 2, 2, 0, -2, 127, -127]);
    }
}

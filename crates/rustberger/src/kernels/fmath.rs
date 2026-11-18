//! f32 helpers (attention, layer norm, softmax, gating, quantisation) compiled twice:
//! a portable version (auto-vectorised for NEON / wasm simd128 / SSE2) and an AVX2+FMA
//! version selected at runtime on x86.

/// Branch-free `exp` (Cephes polynomial, ~1e-7 relative error) that LLVM can vectorise.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    const MAGIC: f32 = 12582912.0;
    let x = x.clamp(-87.0, 88.0);
    let fx = (x * 1.442_695 + MAGIC) - MAGIC; // round to nearest
    let x = x - fx * 0.693_359_375 - fx * -2.121_944_4e-4;
    let z = x * x;
    let y = ((((1.987_569_15e-4 * x + 1.398_199_95e-3) * x + 8.333_451_9e-3) * x + 4.166_579_6e-2) * x + 1.666_666_5e-1) * x + 5.000_000_1e-1;
    let y = y * z + x + 1.0;
    y * f32::from_bits(((fx as i32 + 127) << 23) as u32)
}

#[inline(always)]
pub fn fast_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
}


/// Sum with 8 independent lanes (an ordered f32 reduction cannot be vectorised by LLVM).
#[inline(always)]
fn sum_slice(x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let c = x.chunks_exact(8);
    let rem = c.remainder();
    for ch in c {
        for i in 0..8 {
            acc[i] += ch[i];
        }
    }
    let mut s = 0.0;
    for a in acc {
        s += a;
    }
    for v in rem {
        s += v;
    }
    s
}

#[inline(always)]
fn max_slice(x: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 8];
    let c = x.chunks_exact(8);
    let rem = c.remainder();
    for ch in c {
        for i in 0..8 {
            acc[i] = if ch[i] > acc[i] { ch[i] } else { acc[i] };
        }
    }
    let mut m = f32::NEG_INFINITY;
    for a in acc {
        m = if a > m { a } else { m };
    }
    for &v in rem {
        m = if v > m { v } else { m };
    }
    m
}

/// sum((x - mean)^2) with 8 lanes
#[inline(always)]
fn sumsq_dev(x: &[f32], mean: f32) -> f32 {
    let mut acc = [0f32; 8];
    let c = x.chunks_exact(8);
    let rem = c.remainder();
    for ch in c {
        for i in 0..8 {
            let d = ch[i] - mean;
            acc[i] += d * d;
        }
    }
    let mut s = 0.0;
    for a in acc {
        s += a;
    }
    for v in rem {
        let d = v - mean;
        s += d * d;
    }
    s
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[inline]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

macro_rules! mv {
    ($vis:vis fn $name:ident / $avx:ident ( $($a:ident : $t:ty),* ) => $imp:ident) => {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $avx($($a: $t),*) {
            $imp($($a),*)
        }
        $vis fn $name($($a: $t),*) {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if has_avx2() {
                    return unsafe { $avx($($a),*) };
                }
            }
            $imp($($a),*)
        }
    };
}

// ------------------------------------------------------------------ quantise

#[inline(always)]
fn quantize_impl(a: &[f32], m: usize, k: usize, qm: f32, kp: usize, out: &mut Vec<i8>) {
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
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn quantize_avx2(a: &[f32], m: usize, k: usize, qm: f32, kp: usize, out: &mut Vec<i8>) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    out.clear();
    out.resize(m * kp, 0);
    let vq = _mm256_set1_ps(qm);
    let (lo, hi) = (_mm256_set1_ps(-127.0), _mm256_set1_ps(127.0));
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let cv = |p: *const f32| _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(p), vq), lo), hi));
    for r in 0..m {
        let src = a.as_ptr().add(r * k);
        let dst = out.as_mut_ptr().add(r * kp);
        let mut i = 0;
        while i + 32 <= k {
            let p01 = _mm256_packs_epi32(cv(src.add(i)), cv(src.add(i + 8)));
            let p23 = _mm256_packs_epi32(cv(src.add(i + 16)), cv(src.add(i + 24)));
            let bytes = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p01, p23), perm);
            _mm256_storeu_si256(dst.add(i) as *mut __m256i, bytes);
            i += 32;
        }
        const MAGIC: f32 = 12582912.0;
        while i < k {
            let y = (*src.add(i) * qm + MAGIC) - MAGIC;
            *dst.add(i) = y.clamp(-127.0, 127.0) as i8;
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn quantize_neon(a: &[f32], m: usize, k: usize, qm: f32, kp: usize, out: &mut Vec<i8>) {
    use std::arch::aarch64::*;
    out.clear();
    out.resize(m * kp, 0);
    let lo = vdupq_n_s8(-127);
    let cv = |p: *const f32| vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(p), qm)); // round-to-nearest-even, saturating
    for r in 0..m {
        let src = a.as_ptr().add(r * k);
        let dst = out.as_mut_ptr().add(r * kp);
        let mut i = 0;
        while i + 16 <= k {
            let h01 = vcombine_s16(vqmovn_s32(cv(src.add(i))), vqmovn_s32(cv(src.add(i + 4))));
            let h23 = vcombine_s16(vqmovn_s32(cv(src.add(i + 8))), vqmovn_s32(cv(src.add(i + 12))));
            let b = vmaxq_s8(vcombine_s8(vqmovn_s16(h01), vqmovn_s16(h23)), lo);
            vst1q_s8(dst.add(i), b);
            i += 16;
        }
        const MAGIC: f32 = 12582912.0;
        while i < k {
            let y = (*src.add(i) * qm + MAGIC) - MAGIC;
            *dst.add(i) = y.clamp(-127.0, 127.0) as i8;
            i += 1;
        }
    }
}

pub fn quantize(a: &[f32], m: usize, k: usize, qm: f32, kp: usize, out: &mut Vec<i8>) {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { quantize_neon(a, m, k, qm, kp, out) };
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { quantize_avx2(a, m, k, qm, kp, out) };
        }
    }
    #[allow(unreachable_code)]
    quantize_impl(a, m, k, qm, kp, out)
}

#[cfg(test)]
pub fn quantize_reference(a: &[f32], m: usize, k: usize, qm: f32, kp: usize, out: &mut Vec<i8>) {
    quantize_impl(a, m, k, qm, kp, out)
}

// ----------------------------------------------------------------- attention

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
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

/// Multi-head scaled dot-product attention; `q` is `tq x dim`, `k`/`v` are `tk x dim`.
#[inline(always)]
fn attend_impl(q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, heads: usize, dim: usize, out: &mut [f32], scores: &mut Vec<f32>) {
    let dk = dim / heads;
    let scale = 1.0 / (dk as f32).sqrt();
    scores.clear();
    scores.resize(tk, 0.0);
    for t in 0..tq {
        for h in 0..heads {
            let qh = &q[t * dim + h * dk..t * dim + (h + 1) * dk];
            for j in 0..tk {
                scores[j] = dot(qh, &k[j * dim + h * dk..j * dim + (h + 1) * dk]) * scale;
            }
            let mx = max_slice(scores);
            for s in scores.iter_mut() {
                *s = fast_exp(*s - mx);
            }
            let inv = 1.0 / sum_slice(scores);
            let o = &mut out[t * dim + h * dk..t * dim + (h + 1) * dk];
            o.fill(0.0);
            for j in 0..tk {
                let w = scores[j] * inv;
                let vh = &v[j * dim + h * dk..j * dim + (h + 1) * dk];
                for (oo, vv) in o.iter_mut().zip(vh) {
                    *oo += w * vv;
                }
            }
        }
    }
}
mv!(pub fn attend / attend_avx2 (q: &[f32], k: &[f32], v: &[f32], tq: usize, tk: usize, heads: usize, dim: usize, out: &mut [f32], scores: &mut Vec<f32>) => attend_impl);

// ---------------------------------------------------------------- layer norm

/// out = LayerNorm(out + residual) over the last dimension (Marian eps = 1e-9).
#[inline(always)]
fn add_ln_impl(out: &mut [f32], residual: &[f32], dim: usize, gamma: &[f32], beta: &[f32]) {
    for (row, res) in out.chunks_exact_mut(dim).zip(residual.chunks_exact(dim)) {
        for (o, r) in row.iter_mut().zip(res) {
            *o += r;
        }
        let mean = sum_slice(row) / dim as f32;
        let var = sumsq_dev(row, mean);
        let inv = 1.0 / (var / dim as f32 + 1e-9).sqrt();
        for ((v, g), b) in row.iter_mut().zip(gamma).zip(beta) {
            *v = (*v - mean) * inv * g + b;
        }
    }
}
mv!(pub fn add_layer_norm / add_layer_norm_avx2 (out: &mut [f32], residual: &[f32], dim: usize, gamma: &[f32], beta: &[f32]) => add_ln_impl);

// -------------------------------------------------------------------- misc

/// SSRU gate: c = s*c + (1-s)*xw with s = sigmoid(f); o = relu(c).
#[inline(always)]
fn ssru_impl(f: &[f32], xw: &[f32], c: &mut [f32], o: &mut [f32]) {
    for i in 0..c.len() {
        let s = fast_sigmoid(f[i]);
        let nc = s * c[i] + (1.0 - s) * xw[i];
        c[i] = nc;
        o[i] = nc.max(0.0);
    }
}
mv!(pub fn ssru_gate / ssru_gate_avx2 (f: &[f32], xw: &[f32], c: &mut [f32], o: &mut [f32]) => ssru_impl);

#[inline(always)]
fn relu_impl(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.max(0.0);
    }
}
mv!(pub fn relu_inplace / relu_inplace_avx2 (x: &mut [f32]) => relu_impl);

/// Row-wise log-softmax in place.
#[inline(always)]
fn log_softmax_impl(x: &mut [f32], width: usize) {
    let mut tmp = vec![0f32; width];
    for row in x.chunks_exact_mut(width) {
        let mx = max_slice(row);
        for (t, v) in tmp.iter_mut().zip(row.iter()) {
            *t = fast_exp(*v - mx);
        }
        let lse = sum_slice(&tmp).ln() + mx;
        for v in row.iter_mut() {
            *v -= lse;
        }
    }
}
mv!(pub fn log_softmax_rows / log_softmax_rows_avx2 (x: &mut [f32], width: usize) => log_softmax_impl);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_simd_matches_reference() {
        let mut seed = 99u64;
        for &(m, k) in &[(1usize, 5usize), (3, 31), (4, 32), (5, 33), (7, 256), (2, 1536), (3, 300)] {
            let a: Vec<f32> = (0..m * k)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let r = (seed >> 33) as f32 / (1u64 << 31) as f32; // [0,1)
                    (r - 0.5) * if seed & 7 == 0 { 400.0 } else { 6.0 } // include out-of-range values
                })
                .collect();
            let kp = crate::kernels::pad_k(k);
            let (mut x, mut y) = (Vec::new(), Vec::new());
            quantize(&a, m, k, 9.37, kp, &mut x);
            quantize_reference(&a, m, k, 9.37, kp, &mut y);
            assert_eq!(x, y, "m={m} k={k}");
        }
    }

    #[test]
    fn fast_exp_accuracy() {
        let mut worst = 0f32;
        let mut x = -30.0f32;
        while x < 30.0 {
            let (a, b) = (fast_exp(x), x.exp());
            worst = worst.max(((a - b) / b).abs());
            x += 0.0137;
        }
        assert!(worst < 3e-7, "worst relative error {worst}");
        assert!(fast_exp(-200.0) >= 0.0 && fast_exp(-200.0) < 1e-30);
        assert!(fast_exp(200.0).is_finite());
    }

    #[test]
    fn add_ln_matches_reference() {
        let dim = 64;
        let x: Vec<f32> = (0..dim * 3).map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0).collect();
        let r: Vec<f32> = (0..dim * 3).map(|i| ((i * 11 % 53) as f32 - 26.0) / 7.0).collect();
        let g: Vec<f32> = (0..dim).map(|i| 1.0 + i as f32 / 100.0).collect();
        let b: Vec<f32> = (0..dim).map(|i| i as f32 / 50.0).collect();
        let mut got = x.clone();
        add_layer_norm(&mut got, &r, dim, &g, &b);
        for row in 0..3 {
            let v: Vec<f64> = (0..dim).map(|i| (x[row * dim + i] + r[row * dim + i]) as f64).collect();
            let mean = v.iter().sum::<f64>() / dim as f64;
            let var = v.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / dim as f64;
            for i in 0..dim {
                let want = (v[i] - mean) / (var + 1e-9).sqrt() * g[i] as f64 + b[i] as f64;
                assert!((got[row * dim + i] as f64 - want).abs() < 2e-4);
            }
        }
    }
}

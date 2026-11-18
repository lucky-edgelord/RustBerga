#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::PackedB;

/// One multiply-accumulate step of 32 signed int8 pairs into 8 x i32 lanes.
trait Step {
    unsafe fn step(acc: __m256i, a: __m256i, b: __m256i) -> __m256i;
}

/// AVX2: |a| (unsigned) x sign(b, a) (signed) via maddubs, then pair-sum with madd.
/// Inputs are clamped to [-127, 127] so the i16 intermediate cannot saturate.
struct Avx2;
impl Step for Avx2 {
    #[inline(always)]
    unsafe fn step(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(a, a);
        let bs = _mm256_sign_epi8(b, a);
        let p = _mm256_maddubs_epi16(ax, bs);
        _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)))
    }
}

struct AvxVnni;
impl Step for AvxVnni {
    #[inline(always)]
    unsafe fn step(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(a, a);
        let bs = _mm256_sign_epi8(b, a);
        _mm256_dpbusd_avx_epi32(acc, ax, bs)
    }
}

struct Avx512Vnni;
impl Step for Avx512Vnni {
    #[inline(always)]
    unsafe fn step(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(a, a);
        let bs = _mm256_sign_epi8(b, a);
        _mm256_dpbusd_epi32(acc, ax, bs)
    }
}

#[inline(always)]
unsafe fn hsum4(a0: __m256i, a1: __m256i, a2: __m256i, a3: __m256i) -> __m128i {
    let t0 = _mm256_hadd_epi32(a0, a1);
    let t1 = _mm256_hadd_epi32(a2, a3);
    let t = _mm256_hadd_epi32(t0, t1);
    _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1))
}

#[inline(always)]
unsafe fn store4(v: __m128i, scale: f32, bias: Option<&[f32]>, j: usize, n: usize, dst: *mut f32) {
    let mut f = _mm_mul_ps(_mm_cvtepi32_ps(v), _mm_set1_ps(scale));
    if j + 4 <= n {
        if let Some(b) = bias {
            f = _mm_add_ps(f, _mm_loadu_ps(b.as_ptr().add(j)));
        }
        _mm_storeu_ps(dst.add(j), f);
    } else {
        let mut tmp = [0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), f);
        for q in 0..(n - j) {
            *dst.add(j + q) = tmp[q] + bias.map_or(0.0, |b| b[j + q]);
        }
    }
}

#[inline(always)]
unsafe fn gemm_generic<S: Step>(a: &[i8], m: usize, b: &PackedB, scale: f32, bias: Option<&[f32]>, out: *mut f32, t0: usize, t1: usize) {
    let kp = b.kp;
    let ap = a.as_ptr();
    let bp = b.data.as_ptr();
    for t in t0..t1 {
        let j = t * 4;
        let b0 = bp.add(j * kp);
        let b1 = b0.add(kp);
        let b2 = b1.add(kp);
        let b3 = b2.add(kp);
        let mut i = 0;
        // two A rows at a time against four B rows: 8 accumulators
        while i + 2 <= m {
            let a0p = ap.add(i * kp);
            let a1p = a0p.add(kp);
            let z = _mm256_setzero_si256();
            let (mut c00, mut c01, mut c02, mut c03) = (z, z, z, z);
            let (mut c10, mut c11, mut c12, mut c13) = (z, z, z, z);
            let mut k = 0;
            while k < kp {
                let a0 = _mm256_loadu_si256(a0p.add(k) as *const __m256i);
                let a1 = _mm256_loadu_si256(a1p.add(k) as *const __m256i);
                let v0 = _mm256_loadu_si256(b0.add(k) as *const __m256i);
                c00 = S::step(c00, a0, v0);
                c10 = S::step(c10, a1, v0);
                let v1 = _mm256_loadu_si256(b1.add(k) as *const __m256i);
                c01 = S::step(c01, a0, v1);
                c11 = S::step(c11, a1, v1);
                let v2 = _mm256_loadu_si256(b2.add(k) as *const __m256i);
                c02 = S::step(c02, a0, v2);
                c12 = S::step(c12, a1, v2);
                let v3 = _mm256_loadu_si256(b3.add(k) as *const __m256i);
                c03 = S::step(c03, a0, v3);
                c13 = S::step(c13, a1, v3);
                k += 32;
            }
            store4(hsum4(c00, c01, c02, c03), scale, bias, j, b.n, out.add(i * b.n));
            store4(hsum4(c10, c11, c12, c13), scale, bias, j, b.n, out.add((i + 1) * b.n));
            i += 2;
        }
        if i < m {
            let a0p = ap.add(i * kp);
            let z = _mm256_setzero_si256();
            let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
            let mut k = 0;
            while k < kp {
                let a0 = _mm256_loadu_si256(a0p.add(k) as *const __m256i);
                c0 = S::step(c0, a0, _mm256_loadu_si256(b0.add(k) as *const __m256i));
                c1 = S::step(c1, a0, _mm256_loadu_si256(b1.add(k) as *const __m256i));
                c2 = S::step(c2, a0, _mm256_loadu_si256(b2.add(k) as *const __m256i));
                c3 = S::step(c3, a0, _mm256_loadu_si256(b3.add(k) as *const __m256i));
                k += 32;
            }
            store4(hsum4(c0, c1, c2, c3), scale, bias, j, b.n, out.add(i * b.n));
        }
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn gemm_avx2(a: &[i8], m: usize, b: &PackedB, scale: f32, bias: Option<&[f32]>, out: *mut f32, t0: usize, t1: usize) {
    gemm_generic::<Avx2>(a, m, b, scale, bias, out, t0, t1)
}

#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn gemm_avxvnni(a: &[i8], m: usize, b: &PackedB, scale: f32, bias: Option<&[f32]>, out: *mut f32, t0: usize, t1: usize) {
    gemm_generic::<AvxVnni>(a, m, b, scale, bias, out, t0, t1)
}

#[target_feature(enable = "avx2,avx512vnni,avx512vl")]
pub unsafe fn gemm_avx512vnni(a: &[i8], m: usize, b: &PackedB, scale: f32, bias: Option<&[f32]>, out: *mut f32, t0: usize, t1: usize) {
    gemm_generic::<Avx512Vnni>(a, m, b, scale, bias, out, t0, t1)
}

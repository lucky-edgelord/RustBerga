//! WebAssembly SIMD128 int8 kernel (compile with `-C target-feature=+simd128`).
use core::arch::wasm32::*;

use super::PackedB;

#[inline(always)]
unsafe fn mac(acc: v128, a: v128, b: v128) -> v128 {
    let lo = i16x8_extmul_low_i8x16(a, b);
    let hi = i16x8_extmul_high_i8x16(a, b);
    let s = i32x4_add(i32x4_extadd_pairwise_i16x8(lo), i32x4_extadd_pairwise_i16x8(hi));
    i32x4_add(acc, s)
}

#[inline(always)]
unsafe fn hsum(v: v128) -> i32 {
    i32x4_extract_lane::<0>(v) + i32x4_extract_lane::<1>(v) + i32x4_extract_lane::<2>(v) + i32x4_extract_lane::<3>(v)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm(a: &[i8], m: usize, b: &PackedB, scale: f32, bias: Option<&[f32]>, out: *mut f32, t0: usize, t1: usize) {
    let kp = b.kp;
    let ap = a.as_ptr();
    let bp = b.data.as_ptr();
    for t in t0..t1 {
        let j = t * 4;
        let b0 = bp.add(j * kp);
        let (b1, b2, b3) = (b0.add(kp), b0.add(2 * kp), b0.add(3 * kp));
        for i in 0..m {
            let arow = ap.add(i * kp);
            let z = i32x4_splat(0);
            let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
            let mut k = 0;
            while k < kp {
                let av = v128_load(arow.add(k) as *const v128);
                c0 = mac(c0, av, v128_load(b0.add(k) as *const v128));
                c1 = mac(c1, av, v128_load(b1.add(k) as *const v128));
                c2 = mac(c2, av, v128_load(b2.add(k) as *const v128));
                c3 = mac(c3, av, v128_load(b3.add(k) as *const v128));
                k += 16;
            }
            let r = [hsum(c0), hsum(c1), hsum(c2), hsum(c3)];
            for q in 0..4 {
                if j + q < b.n {
                    *out.add(i * b.n + j + q) = r[q] as f32 * scale + bias.map_or(0.0, |bb| bb[j + q]);
                }
            }
        }
    }
}

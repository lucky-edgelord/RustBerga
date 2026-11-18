use core::arch::aarch64::*;

use super::PackedB;

#[inline(always)]
unsafe fn finish(r: [i32; 4], j: usize, n: usize, i: usize, scale: f32, bias: Option<&[f32]>, out: *mut f32) {
    for q in 0..4 {
        if j + q < n {
            *out.add(i * n + j + q) = r[q] as f32 * scale + bias.map_or(0.0, |bb| bb[j + q]);
        }
    }
}

#[inline(always)]
unsafe fn mac(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let hi = vmull_high_s8(a, b);
    vpadalq_s16(vpadalq_s16(acc, lo), hi)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm(a: &[i8], m: usize, b: &PackedB, scale: f32, bias: Option<&[f32]>, out: *mut f32, t0: usize, t1: usize) {
    let kp = b.kp;
    for t in t0..t1 {
        let j = t * 4;
        let bp = b.data.as_ptr().add(j * kp);
        for i in 0..m {
            let ar = a.as_ptr().add(i * kp);
            let z = vdupq_n_s32(0);
            let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
            let mut k = 0;
            while k < kp {
                let av = vld1q_s8(ar.add(k));
                c0 = mac(c0, av, vld1q_s8(bp.add(k)));
                c1 = mac(c1, av, vld1q_s8(bp.add(kp + k)));
                c2 = mac(c2, av, vld1q_s8(bp.add(2 * kp + k)));
                c3 = mac(c3, av, vld1q_s8(bp.add(3 * kp + k)));
                k += 16;
            }
            finish([vaddvq_s32(c0), vaddvq_s32(c1), vaddvq_s32(c2), vaddvq_s32(c3)], j, b.n, i, scale, bias, out);
        }
    }
}

pub struct NeonDot;

impl super::dot4::Dot4 for NeonDot {
    type B = int8x16_t;
    type Acc = int32x4_t;
    #[inline(always)]
    unsafe fn load(p: *const i8) -> int8x16_t {
        vld1q_s8(p)
    }
    #[inline(always)]
    unsafe fn zero() -> int32x4_t {
        vdupq_n_s32(0)
    }
    #[inline(always)]
    unsafe fn dot<const L: i32>(acc: int32x4_t, b: int8x16_t, a: int8x16_t) -> int32x4_t {
        let mut r = acc;
        core::arch::asm!(
            ".arch_extension dotprod",
            "sdot {r:v}.4s, {b:v}.16b, {a:v}.4b[{l}]",
            r = inout(vreg) r, b = in(vreg) b, a = in(vreg) a, l = const L,
            options(pure, nomem, nostack)
        );
        r
    }
    #[inline(always)]
    unsafe fn lanes(acc: int32x4_t) -> [i32; 4] {
        let mut o = [0i32; 4];
        vst1q_s32(o.as_mut_ptr(), acc);
        o
    }
}

/// Interleaved-layout kernel (4 rows x 8 columns per tile).
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_dot_il(a: &[i8], m: usize, b: &PackedB, scale: f32, bias: Option<&[f32]>, out: *mut f32, t0: usize, t1: usize) {
    super::dot4::gemm::<NeonDot>(a, m, b, scale, bias, out, t0, t1)
}

#![allow(dead_code)]
//! Portable tile kernel for CPUs with a 4-byte-group int8 dot product (ARM `sdot`).
//!
//! The tile logic is generic over [`Dot4`], whose only job is the 16-byte primitive
//! `acc.lane[i] += sum_t b[4i+t] * a[4L+t]` (exactly the semantics of `SDOT Vd.4S, Vn.16B, Vm.4B[L]`).
//! A software implementation lets the x86 test-suite exercise the very same loops, layout
//! arithmetic and tail handling that run on ARM hardware.
//!
//! Weight layout (`PackedB::il`, group size 4): `[col/4][k/4][col%4][k%4]`, i.e. 16 bytes per
//! (4 columns, 4 k) block. Each tile covers 8 columns (two groups) and up to 4 activation rows.

use super::PackedB;

pub trait Dot4 {
    type B: Copy;
    type Acc: Copy;
    /// Load 16 consecutive int8 values.
    unsafe fn load(p: *const i8) -> Self::B;
    unsafe fn zero() -> Self::Acc;
    unsafe fn dot<const L: i32>(acc: Self::Acc, b: Self::B, a: Self::B) -> Self::Acc;
    unsafe fn lanes(acc: Self::Acc) -> [i32; 4];
}

/// Software model of the primitive (used by tests on any architecture).
#[allow(dead_code)]
pub struct Emulated;

impl Dot4 for Emulated {
    type B = [i8; 16];
    type Acc = [i32; 4];
    unsafe fn load(p: *const i8) -> [i8; 16] {
        let mut v = [0i8; 16];
        std::ptr::copy_nonoverlapping(p, v.as_mut_ptr(), 16);
        v
    }
    unsafe fn zero() -> [i32; 4] {
        [0; 4]
    }
    unsafe fn dot<const L: i32>(mut acc: [i32; 4], b: [i8; 16], a: [i8; 16]) -> [i32; 4] {
        for i in 0..4 {
            for t in 0..4 {
                acc[i] += b[4 * i + t] as i32 * a[4 * L as usize + t] as i32;
            }
        }
        acc
    }
    unsafe fn lanes(acc: [i32; 4]) -> [i32; 4] {
        acc
    }
}

#[inline(always)]
unsafe fn tile<V: Dot4, const R: usize>(a: *const i8, kp: usize, row0: usize, b0: *const i8, b1: *const i8, col0: usize, n: usize, scale: f32, bias: Option<&[f32]>, out: *mut f32) {
    let z = V::zero();
    let mut acc = [[z; 2]; R];
    let mut k = 0;
    while k < kp {
        let av: [V::B; R] = core::array::from_fn(|r| V::load(a.add((row0 + r) * kp + k)));
        // 16 k-values == 4 k-groups == 64 bytes of each column group's stream
        let o = k * 4;
        let (p0, p1, p2, p3) = (V::load(b0.add(o)), V::load(b0.add(o + 16)), V::load(b0.add(o + 32)), V::load(b0.add(o + 48)));
        let (q0, q1, q2, q3) = (V::load(b1.add(o)), V::load(b1.add(o + 16)), V::load(b1.add(o + 32)), V::load(b1.add(o + 48)));
        for r in 0..R {
            acc[r][0] = V::dot::<0>(acc[r][0], p0, av[r]);
            acc[r][0] = V::dot::<1>(acc[r][0], p1, av[r]);
            acc[r][0] = V::dot::<2>(acc[r][0], p2, av[r]);
            acc[r][0] = V::dot::<3>(acc[r][0], p3, av[r]);
            acc[r][1] = V::dot::<0>(acc[r][1], q0, av[r]);
            acc[r][1] = V::dot::<1>(acc[r][1], q1, av[r]);
            acc[r][1] = V::dot::<2>(acc[r][1], q2, av[r]);
            acc[r][1] = V::dot::<3>(acc[r][1], q3, av[r]);
        }
        k += 16;
    }
    for r in 0..R {
        for g in 0..2 {
            let l = V::lanes(acc[r][g]);
            for j in 0..4 {
                let col = col0 + g * 4 + j;
                if col < n {
                    *out.add((row0 + r) * n + col) = l[j] as f32 * scale + bias.map_or(0.0, |b| b[col]);
                }
            }
        }
    }
}

/// Units `t0..t1` are 8-column blocks.
#[inline(always)]
pub unsafe fn gemm<V: Dot4>(a: &[i8], m: usize, b: &PackedB, scale: f32, bias: Option<&[f32]>, out: *mut f32, t0: usize, t1: usize) {
    debug_assert_eq!(b.il_group, 4);
    let kp = b.kp;
    let kg_bytes = (kp / 4) * 16; // bytes per 4-column group
    let ap = a.as_ptr();
    for t in t0..t1 {
        let col0 = t * 8;
        let b0 = b.il.as_ptr().add(2 * t * kg_bytes);
        let b1 = b0.add(kg_bytes);
        let mut i = 0;
        while i + 4 <= m {
            tile::<V, 4>(ap, kp, i, b0, b1, col0, b.n, scale, bias, out);
            i += 4;
        }
        if i + 2 <= m {
            tile::<V, 2>(ap, kp, i, b0, b1, col0, b.n, scale, bias, out);
            i += 2;
        }
        if i < m {
            tile::<V, 1>(ap, kp, i, b0, b1, col0, b.n, scale, bias, out);
        }
    }
}

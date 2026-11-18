use super::PackedB;

#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm(a: &[i8], m: usize, b: &PackedB, scale: f32, bias: Option<&[f32]>, out: *mut f32, t0: usize, t1: usize) {
    let kp = b.kp;
    for j in (t0 * 4)..(t1 * 4).min(b.n) {
        let br = &b.data[j * kp..(j + 1) * kp];
        for i in 0..m {
            let ar = &a[i * kp..(i + 1) * kp];
            let mut acc = 0i32;
            for (x, y) in ar.iter().zip(br) {
                acc += *x as i32 * *y as i32;
            }
            *out.add(i * b.n + j) = acc as f32 * scale + bias.map_or(0.0, |bb| bb[j]);
        }
    }
}

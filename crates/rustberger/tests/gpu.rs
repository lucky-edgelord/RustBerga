#![cfg(feature = "gpu")]
use rustberger::gpu;
use rustberger::kernels::{self, Isa, PackedB};

#[test]
fn gpu_gemm_matches_cpu() {
    let info = match gpu::enable() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: {e}");
            return;
        }
    };
    eprintln!("GPU adapter: {info}");
    let g = gpu::get().unwrap();
    let mut seed = 7u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    for &(m, n, k) in &[(5, 7, 40), (64, 256, 256), (100, 33, 1536), (17, 1000, 257)] {
        let bd: Vec<i8> = (0..n * k).map(|_| ((rnd() % 255) as i32 - 127) as i8).collect();
        let b = PackedB::from_nk(&bd, n, k, 50.0);
        let af: Vec<f32> = (0..m * k).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
        let mut aq = Vec::new();
        kernels::quantize(&af, m, k, 90.0, b.kp, &mut aq);
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let mut want = vec![0f32; m * n];
        kernels::gemm_isa(Isa::Scalar, &aq, m, &b, 90.0, Some(&bias), &mut want);
        let w = g.upload(&b);
        let got = g.gemm(&aq, m, &w, 1.0 / (90.0 * 50.0), Some(&bias));
        assert_eq!(got.len(), want.len());
        let maxerr = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(maxerr < 1e-3, "m={m} n={n} k={k} max abs err {maxerr}");
    }
}

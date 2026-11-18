//! WebAssembly build of Rustberger.
//!
//! A tiny C ABI (no wasm-bindgen needed) driven by `js/rustberger.mjs`. Build with
//! `RUSTFLAGS="-C target-feature=+simd128"` so the int8 GEMM uses WebAssembly SIMD.

use std::cell::RefCell;

use rustberger::config::Options;
use rustberger::Translator;

thread_local! {
    static TRANSLATORS: RefCell<Vec<Option<Translator>>> = const { RefCell::new(Vec::new()) };
    static RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn set_error(msg: String) {
    ERROR.with(|e| *e.borrow_mut() = msg.into_bytes());
}

#[no_mangle]
pub extern "C" fn rb_alloc(n: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(n.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `p` must come from `rb_alloc(n)` with the same `n`.
#[no_mangle]
pub unsafe extern "C" fn rb_free(p: *mut u8, n: usize) {
    drop(Vec::from_raw_parts(p, 0, n.max(1)));
}

/// Returns a translator handle (>= 0) or -1 (see `rb_error_*`). `slen == 0` means no shortlist.
///
/// # Safety
/// All pointers must be valid for the given lengths.
#[no_mangle]
pub unsafe extern "C" fn rb_load(model: *const u8, mlen: usize, vocab: *const u8, vlen: usize, sl: *const u8, slen: usize, beam: u32) -> i32 {
    let m = std::slice::from_raw_parts(model, mlen);
    let v = std::slice::from_raw_parts(vocab, vlen);
    let s = if slen > 0 { Some(std::slice::from_raw_parts(sl, slen)) } else { None };
    let opts = Options { beam_size: (beam as usize).max(1), ..Options::default() };
    match Translator::from_memory(m, v, s, opts) {
        Ok(t) => TRANSLATORS.with(|ts| {
            let mut ts = ts.borrow_mut();
            ts.push(Some(t));
            (ts.len() - 1) as i32
        }),
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn rb_unload(handle: i32) {
    TRANSLATORS.with(|ts| {
        if let Some(slot) = ts.borrow_mut().get_mut(handle as usize) {
            *slot = None;
        }
    });
}

/// Translate UTF-8 text; 0 on success (result via `rb_result_*`), -1 on error.
///
/// # Safety
/// `input` must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn rb_translate(handle: i32, input: *const u8, len: usize) -> i32 {
    let bytes = std::slice::from_raw_parts(input, len);
    let Ok(text) = std::str::from_utf8(bytes) else {
        set_error("input is not valid UTF-8".into());
        return -1;
    };
    TRANSLATORS.with(|ts| match ts.borrow().get(handle as usize) {
        Some(Some(t)) => match t.translate(text) {
            Ok(out) => {
                RESULT.with(|r| *r.borrow_mut() = out.into_bytes());
                0
            }
            Err(e) => {
                set_error(e.to_string());
                -1
            }
        },
        _ => {
            set_error("invalid translator handle".into());
            -1
        }
    })
}

#[no_mangle]
pub extern "C" fn rb_result_ptr() -> *const u8 {
    RESULT.with(|r| r.borrow().as_ptr())
}
#[no_mangle]
pub extern "C" fn rb_result_len() -> usize {
    RESULT.with(|r| r.borrow().len())
}
#[no_mangle]
pub extern "C" fn rb_error_ptr() -> *const u8 {
    ERROR.with(|r| r.borrow().as_ptr())
}
#[no_mangle]
pub extern "C" fn rb_error_len() -> usize {
    ERROR.with(|r| r.borrow().len())
}

/// 1 when this build was compiled with WebAssembly SIMD128 kernels.
#[no_mangle]
pub extern "C" fn rb_has_simd128() -> i32 {
    cfg!(target_feature = "simd128") as i32
}

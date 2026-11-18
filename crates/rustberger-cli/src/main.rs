//! `rustberger`: drop-in counterpart of bergamot-translator's `app/bergamot.cpp`.
//! Reads text from stdin, prints only the translation to stdout.

use std::io::{Read, Write};
use std::time::Instant;

use rustberger::config::Options;
use rustberger::Translator;

fn usage() -> ! {
    eprintln!(
        "usage: rustberger --model-config-paths <config.yml> [--cpu-threads N] [--stats] [--gpu]\n       rustberger --model <model.bin> --vocab <vocab.spm> [--shortlist <lex.bin>] [--beam N] [--cpu-threads N] [--stats]"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut cfg, mut model, mut vocab, mut sl) = (None, None, None, None);
    let (mut threads, mut beam, mut stats) = (None, None, false);
    let mut gpu = false;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].clone();
        let mut val = || {
            i += 1;
            args.get(i).cloned().unwrap_or_else(|| usage())
        };
        match flag.as_str() {
            "--model-config-paths" => cfg = Some(val()),
            "--model" => model = Some(val()),
            "--vocab" => vocab = Some(val()),
            "--shortlist" => sl = Some(val()),
            "--cpu-threads" => threads = Some(val().parse::<usize>().unwrap_or_else(|_| usage())),
            "--beam" => beam = Some(val().parse::<usize>().unwrap_or_else(|_| usage())),
            "--stats" => stats = true,
            "--gpu" => gpu = true,
            _ => usage(),
        }
        i += 1;
    }
    if let Some(t) = threads {
        if t > 0 {
            rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok();
        }
    }
    if gpu {
        #[cfg(feature = "gpu")]
        match rustberger::gpu::enable() {
            Ok(info) => eprintln!("gpu: {info}"),
            Err(e) => eprintln!("gpu: {e}; continuing on CPU"),
        }
        #[cfg(not(feature = "gpu"))]
        eprintln!("gpu: this binary was built without the `gpu` feature; continuing on CPU");
    }
    let t0 = Instant::now();
    let loaded = match (&cfg, &model, &vocab) {
        (Some(c), _, _) => Translator::from_config_file(c),
        (None, Some(m), Some(v)) => {
            let rd = |p: &String| std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
            let slb = sl.as_ref().map(rd);
            Translator::from_memory(&rd(m), &rd(v), slb.as_deref(), Options::default())
        }
        _ => usage(),
    };
    let mut tr = loaded.unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1)
    });
    if let Some(b) = beam {
        tr.opts.beam_size = b.max(1);
    }
    let t_load = t0.elapsed();

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("stdin must be UTF-8");
    let t1 = Instant::now();
    let out = tr.translate(&input).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1)
    });
    let t_tr = t1.elapsed();
    std::io::stdout().write_all(out.as_bytes()).unwrap();
    if stats {
        let words = input.split_whitespace().count();
        let p = rustberger::prof::seconds();
        let parts: Vec<String> = rustberger::prof::NAMES.iter().zip(p).map(|(n, s)| format!("{n} {s:.3}s")).collect();
        eprintln!("cpu-seconds by stage (search includes decoder stages): {}", parts.join(", "));
        eprintln!("load {:.3}s  translate {:.3}s  ({} words, {:.1} words/s)", t_load.as_secs_f64(), t_tr.as_secs_f64(), words, words as f64 / t_tr.as_secs_f64());
    }
}

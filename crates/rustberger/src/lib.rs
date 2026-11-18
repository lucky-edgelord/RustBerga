//! Rustberger: a pure-Rust port of the Bergamot (Marian student) translation engine.

pub mod config;
pub mod error;
pub mod kernels;
pub mod marian;
pub mod model;
pub mod search;
pub mod shortlist;
pub mod spm;
pub mod text;

use config::{Options, RawConfig};
use error::Result;
use model::Model;
use shortlist::Shortlist;
use spm::SentencePiece;

pub struct Translator {
    model: Model,
    spm: SentencePiece,
    shortlist: Option<Shortlist>,
    pub opts: Options,
}

struct Job {
    sent: usize,
    chunk: usize,
    ids: Vec<u32>, // includes trailing EOS
}

impl Translator {
    /// Load from a Bergamot/Marian YAML config; relative paths resolve against the config's directory.
    pub fn from_config_file(path: &str) -> Result<Translator> {
        let text = std::fs::read_to_string(path)?;
        let raw = RawConfig::parse(&text)?;
        let dir = std::path::Path::new(path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let resolve = |p: &str| -> std::path::PathBuf {
            let pp = std::path::Path::new(p);
            if pp.is_absolute() || pp.exists() {
                pp.to_path_buf()
            } else {
                dir.join(pp)
            }
        };
        let Some(m) = raw.models.first() else { crate::bail!("config has no `models`") };
        let Some(v) = raw.vocabs.first() else { crate::bail!("config has no `vocabs`") };
        let model = std::fs::read(resolve(m))?;
        let vocab = std::fs::read(resolve(v))?;
        let sl = match raw.shortlist_path() {
            Some(p) => Some(std::fs::read(resolve(&p))?),
            None => None,
        };
        Translator::from_memory(&model, &vocab, sl.as_deref(), raw.options())
    }

    pub fn from_memory(model: &[u8], vocab: &[u8], shortlist: Option<&[u8]>, opts: Options) -> Result<Translator> {
        let (model, _yml) = Model::from_bytes(model)?;
        let spm = SentencePiece::from_bytes(vocab)?;
        if spm.vocab_size() != model.vocab {
            crate::bail!("vocab size mismatch: sentencepiece has {} pieces, model expects {}", spm.vocab_size(), model.vocab);
        }
        let shortlist = match shortlist {
            Some(b) => Some(Shortlist::from_bytes(b)?),
            None => None,
        };
        Ok(Translator { model, spm, shortlist, opts })
    }

    /// Translate free text; sentence boundaries and inter-sentence whitespace are preserved.
    pub fn translate(&self, text: &str) -> Result<String> {
        let (segs, tail) = text::split(text);
        let src: Vec<&str> = segs.iter().map(|s| s.text.as_str()).collect();
        let out = self.translate_sentences(&src);
        let mut res = String::with_capacity(text.len() + text.len() / 4);
        for (seg, t) in segs.iter().zip(out) {
            res.push_str(&seg.gap);
            res.push_str(&t);
        }
        res.push_str(&tail);
        Ok(res)
    }

    /// Translate already-segmented sentences.
    pub fn translate_sentences(&self, sentences: &[&str]) -> Vec<String> {
        let eos = self.spm.eos_id;
        let max_chunk = self.opts.max_length_break.saturating_sub(1).max(1);
        let mut jobs: Vec<Job> = Vec::new();
        let mut chunks_per_sent = vec![0usize; sentences.len()];
        for (si, s) in sentences.iter().enumerate() {
            let ids = self.spm.encode(s);
            if ids.is_empty() {
                continue;
            }
            for (ci, c) in ids.chunks(max_chunk).enumerate() {
                let mut v = c.to_vec();
                v.push(eos);
                jobs.push(Job { sent: si, chunk: ci, ids: v });
                chunks_per_sent[si] = ci + 1;
            }
        }
        // longest first, then pack into batches bounded by padded word count
        let mut order: Vec<usize> = (0..jobs.len()).collect();
        order.sort_by(|&a, &b| jobs[b].ids.len().cmp(&jobs[a].ids.len()));
        let mut batches: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut cur_max = 0;
        for &j in &order {
            let l = jobs[j].ids.len();
            let m = cur_max.max(l);
            if !cur.is_empty() && m * (cur.len() + 1) > self.opts.mini_batch_words.max(l) {
                batches.push(std::mem::take(&mut cur));
                cur_max = 0;
            }
            cur_max = cur_max.max(l);
            cur.push(j);
        }
        if !cur.is_empty() {
            batches.push(cur);
        }

        let run = |batch: &Vec<usize>| -> Vec<(usize, Vec<u32>)> {
            let seqs: Vec<&[u32]> = batch.iter().map(|&j| jobs[j].ids.as_slice()).collect();
            let ids = self.shortlist.as_ref().map(|sl| sl.generate(seqs.iter().flat_map(|s| s.iter().copied()), self.model.vocab));
            let out = self.model.output_layer(ids);
            let enc = self.model.encode(&seqs);
            let dec = search::decode(&self.model, &enc, &out, eos, &self.opts);
            batch.iter().copied().zip(dec).collect()
        };

        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        let results: Vec<Vec<(usize, Vec<u32>)>> = {
            use rayon::prelude::*;
            batches.par_iter().map(run).collect()
        };
        #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
        let results: Vec<Vec<(usize, Vec<u32>)>> = batches.iter().map(run).collect();

        let mut pieces: Vec<Vec<Option<String>>> = chunks_per_sent.iter().map(|&c| vec![None; c]).collect();
        for (j, toks) in results.into_iter().flatten() {
            pieces[jobs[j].sent][jobs[j].chunk] = Some(self.spm.decode(&toks));
        }
        pieces
            .into_iter()
            .map(|ch| ch.into_iter().map(|c| c.unwrap_or_default()).collect::<Vec<_>>().join(" "))
            .collect()
    }
}

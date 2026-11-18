//! Batched beam search over the SSRU decoder (beam 1 degenerates to greedy decoding).

use crate::config::Options;
use crate::model::{Encoded, Model, OutputLayer};

struct Hyp {
    toks: Vec<u32>,
    score: f32,
    row: usize, // row in the previous step's state buffer
}

struct Sent {
    alive: Vec<Hyp>,
    finished: Vec<(f32, Vec<u32>)>, // (normalised score, tokens without EOS)
    max_len: usize,
    done: bool,
}

/// Decode every sentence of `enc`. Returns target token ids (EOS stripped) per sentence.
pub fn decode(model: &Model, enc: &Encoded, out: &OutputLayer, eos: u32, opts: &Options) -> Vec<Vec<u32>> {
    let n = enc.lens.len();
    let beam = opts.beam_size.max(1);
    let ss = model.state_size();
    let mut sents: Vec<Sent> = (0..n)
        .map(|s| Sent {
            alive: vec![Hyp { toks: vec![], score: 0.0, row: s }],
            finished: vec![],
            max_len: ((enc.lens[s] as f32 * opts.max_length_factor).ceil() as usize).max(2),
            done: false,
        })
        .collect();
    let mut cells = vec![0f32; n * ss];
    let width = out.width();
    let norm = |score: f32, len: usize| if opts.normalize > 0.0 { score / (len as f32).powf(opts.normalize) } else { score };

    let mut step = 0usize;
    loop {
        // gather alive rows
        let mut prev = Vec::new();
        let mut sent_of = Vec::new();
        let mut old_rows = Vec::new();
        let mut owner = Vec::new(); // (sentence, hyp index)
        for (s, st) in sents.iter().enumerate() {
            if st.done {
                continue;
            }
            for (hi, h) in st.alive.iter().enumerate() {
                prev.push(h.toks.last().copied());
                sent_of.push(s);
                old_rows.push(h.row);
                owner.push((s, hi));
            }
        }
        if prev.is_empty() {
            break;
        }
        let r = prev.len();
        let mut cur = vec![0f32; r * ss];
        for (i, &o) in old_rows.iter().enumerate() {
            cur[i * ss..(i + 1) * ss].copy_from_slice(&cells[o * ss..(o + 1) * ss]);
        }
        let pos = vec![step; r];
        let lp = model.step(enc, out, &prev, &pos, &sent_of, &mut cur);

        // select candidates per sentence
        let mut cands: Vec<Vec<(f32, usize, usize)>> = vec![Vec::new(); n]; // (score, row, col)
        let mut row_of_owner = vec![0usize; r];
        for i in 0..r {
            row_of_owner[i] = i;
        }
        for (i, &(s, hi)) in owner.iter().enumerate() {
            let st = &sents[s];
            let base = st.alive[hi].score;
            let row = &lp[i * width..(i + 1) * width];
            let last = step + 1 >= st.max_len;
            if last {
                // force termination
                if let Some(c) = (0..width).find(|&c| out.token(c) == eos) {
                    cands[s].push((base + row[c] - opts.word_penalty, i, c));
                }
                continue;
            }
            let mut top: Vec<(f32, usize)> = Vec::with_capacity(beam + 1);
            for (c, &v) in row.iter().enumerate() {
                if top.len() < beam || v > top[top.len() - 1].0 {
                    let p = top.iter().position(|t| v > t.0).unwrap_or(top.len());
                    top.insert(p, (v, c));
                    if top.len() > beam {
                        top.pop();
                    }
                }
            }
            for (v, c) in top {
                cands[s].push((base + v - opts.word_penalty, i, c));
            }
        }

        let mut next_cells: Vec<Vec<f32>> = Vec::new();
        let mut next_index = 0usize;
        // rebuild each sentence's alive set
        let mut staged: Vec<(usize, Vec<Hyp>)> = Vec::new();
        for s in 0..n {
            if sents[s].done {
                continue;
            }
            let mut c = std::mem::take(&mut cands[s]);
            c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            c.truncate(beam);
            let mut alive = Vec::new();
            for (score, row, col) in c {
                let (_, hi) = owner[row];
                let tok = out.token(col);
                let mut toks = sents[s].alive[hi].toks.clone();
                if tok == eos {
                    let len = toks.len() + 1;
                    sents[s].finished.push((norm(score, len), toks));
                } else {
                    toks.push(tok);
                    alive.push(Hyp { toks, score, row: row });
                }
            }
            staged.push((s, alive));
        }
        for (s, alive) in staged {
            let st = &mut sents[s];
            let best_fin = st.finished.iter().map(|f| f.0).fold(f32::NEG_INFINITY, f32::max);
            let stop = alive.is_empty()
                || (best_fin > f32::NEG_INFINITY && alive.iter().all(|h| norm(h.score, h.toks.len() + 1) < best_fin));
            st.alive = alive;
            if stop {
                st.done = true;
                st.alive.clear();
            }
        }
        // remap rows: hyp.row currently indexes `cur` (this step's rows) which holds the new states
        let _ = (&mut next_cells, &mut next_index);
        cells = cur;
        step += 1;
    }

    sents
        .into_iter()
        .map(|mut s| {
            s.finished.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            s.finished.into_iter().next().map(|f| f.1).unwrap_or_default()
        })
        .collect()
}

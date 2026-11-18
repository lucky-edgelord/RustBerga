//! Student transformer (6-layer encoder, SSRU decoder) executed with int8 GEMMs.

use std::collections::HashMap;

use crate::bail;
use crate::config::ModelYml;
use crate::error::Result;
use crate::kernels::{self, PackedB};
use crate::marian::{self, Item, TYPE_INTGEMM8};

const POS_CACHE: usize = 256;

pub struct Linear {
    pub w: PackedB,
    pub qa: f32,
    pub bias: Option<Vec<f32>>,
    #[cfg(feature = "gpu")]
    gpu_w: std::sync::OnceLock<crate::gpu::GpuWeights>,
}

impl Linear {
    /// `x` is `m x w.k` f32; result is `m x w.n`.
    pub fn apply(&self, x: &[f32], m: usize, scratch: &mut Vec<i8>, out: &mut Vec<f32>) {
        {
            let _q = crate::prof::start(crate::prof::QUANT);
            kernels::quantize(x, m, self.w.k, self.qa, self.w.kp, scratch);
        }
        let _g = crate::prof::start(crate::prof::GEMM);
        #[cfg(feature = "gpu")]
        if m >= crate::gpu::MIN_ROWS {
            if let Some(g) = crate::gpu::get() {
                let gw = self.gpu_w.get_or_init(|| g.upload(&self.w));
                *out = g.gemm(scratch, m, gw, 1.0 / (self.qa * self.w.qm), self.bias.as_deref());
                return;
            }
        }
        out.clear();
        out.resize(m * self.w.n, 0.0);
        kernels::gemm(scratch, m, &self.w, self.qa, self.bias.as_deref(), out);
    }
}

pub struct Ln {
    g: Vec<f32>,
    b: Vec<f32>,
}

struct EncLayer {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    ln_att: Ln,
    w1: Linear,
    w2: Linear,
    ln_ffn: Ln,
}

struct DecLayer {
    rnn_w: Linear,
    rnn_wf: Linear,
    ln_rnn: Ln,
    cq: Linear,
    ck: Linear,
    cv: Linear,
    co: Linear,
    ln_ctx: Ln,
    w1: Linear,
    w2: Linear,
    ln_ffn: Ln,
}

pub struct Model {
    pub dim: usize,
    heads: usize,
    pub vocab: usize,
    enc: Vec<EncLayer>,
    dec: Vec<DecLayer>,
    emb: Vec<f32>,
    emb_q: PackedB,
    out_qa: f32,
    out_bias: Vec<f32>,
    freqs: Vec<f32>,
    pos_table: Vec<f32>,
}

/// Encoder output for a batch of sentences (concatenated along the token axis).
pub struct Encoded {
    pub offsets: Vec<usize>,
    pub lens: Vec<usize>,
    cross_k: Vec<Vec<f32>>,
    cross_v: Vec<Vec<f32>>,
}

/// Output layer weights restricted to a shortlist (or the whole vocabulary).
pub struct OutputLayer {
    pub ids: Option<Vec<u32>>,
    w: PackedB,
    bias: Vec<f32>,
}

impl OutputLayer {
    pub fn width(&self) -> usize {
        self.w.n
    }
    pub fn token(&self, col: usize) -> u32 {
        match &self.ids {
            Some(v) => v[col],
            None => col as u32,
        }
    }
}

struct Loader {
    items: HashMap<String, Item>,
}

impl Loader {
    fn get(&self, name: &str) -> Result<&Item> {
        match self.items.get(name) {
            Some(i) => Ok(i),
            None => bail!("missing tensor `{name}` in model"),
        }
    }
    fn vec(&self, name: &str) -> Result<Vec<f32>> {
        Ok(self.get(name)?.as_f32())
    }
    fn linear(&self, w: &str, b: Option<&str>) -> Result<Linear> {
        let it = self.get(w)?;
        if it.ty != TYPE_INTGEMM8 {
            bail!("tensor `{w}` is not an intgemm8 matrix (type {}); only int8 models are supported", it.ty);
        }
        let (k, n) = (it.rows(), it.cols());
        let (data, qm) = it.as_i8_with_mult();
        let qa = self.get(&format!("{w}_QuantMultA"))?.as_f32()[0];
        let bias = match b {
            Some(b) => Some(self.vec(b)?),
            None => None,
        };
        Ok(Linear { w: PackedB::from_nk(data, n, k, qm), qa, bias, #[cfg(feature = "gpu")] gpu_w: std::sync::OnceLock::new() })
    }
    fn ln(&self, prefix: &str) -> Result<Ln> {
        Ok(Ln { g: self.vec(&format!("{prefix}_ln_scale"))?, b: self.vec(&format!("{prefix}_ln_bias"))? })
    }
}

impl Model {
    pub fn from_bytes(buf: &[u8]) -> Result<(Model, ModelYml)> {
        let items = marian::parse(buf)?;
        let yml_item = items.iter().find(|i| i.name == "special:model.yml");
        let Some(yml_item) = yml_item else { bail!("model has no embedded special:model.yml") };
        let end = yml_item.data.iter().position(|&b| b == 0).unwrap_or(yml_item.data.len());
        let yml: ModelYml = serde_yaml::from_slice(&yml_item.data[..end])?;
        if yml.decoder_autoreg != "rnn" || yml.dec_cell != "ssru" {
            bail!("unsupported decoder (autoreg={}, cell={}); only SSRU student models are supported", yml.decoder_autoreg, yml.dec_cell);
        }
        if !yml.tied_embeddings_all {
            bail!("only models with tied-embeddings-all are supported");
        }
        if yml.ffn_activation != "relu" {
            bail!("unsupported ffn activation `{}`", yml.ffn_activation);
        }
        let l = Loader { items: items.into_iter().map(|i| (i.name.clone(), i)).collect() };

        let wemb = l.get("Wemb")?;
        if wemb.ty != TYPE_INTGEMM8 {
            bail!("only int8 (intgemm8) embeddings are supported");
        }
        let (vocab, dim) = (wemb.rows(), wemb.cols());
        let (wdata, wqm) = wemb.as_i8_with_mult();
        let inv = 1.0 / wqm;
        let emb: Vec<f32> = wdata.iter().map(|&q| q as f32 * inv).collect();
        let emb_q = PackedB::from_nk(wdata, vocab, dim, wqm);

        let mut enc = Vec::new();
        for i in 1..=yml.enc_depth {
            let p = format!("encoder_l{i}");
            enc.push(EncLayer {
                wq: l.linear(&format!("{p}_self_Wq"), Some(&format!("{p}_self_bq")))?,
                wk: l.linear(&format!("{p}_self_Wk"), Some(&format!("{p}_self_bk")))?,
                wv: l.linear(&format!("{p}_self_Wv"), Some(&format!("{p}_self_bv")))?,
                wo: l.linear(&format!("{p}_self_Wo"), Some(&format!("{p}_self_bo")))?,
                ln_att: l.ln(&format!("{p}_self_Wo"))?,
                w1: l.linear(&format!("{p}_ffn_W1"), Some(&format!("{p}_ffn_b1")))?,
                w2: l.linear(&format!("{p}_ffn_W2"), Some(&format!("{p}_ffn_b2")))?,
                ln_ffn: l.ln(&format!("{p}_ffn_ffn"))?,
            });
        }
        let mut dec = Vec::new();
        for i in 1..=yml.dec_depth {
            let p = format!("decoder_l{i}");
            dec.push(DecLayer {
                rnn_w: l.linear(&format!("{p}_rnn_W"), None)?,
                rnn_wf: l.linear(&format!("{p}_rnn_Wf"), Some(&format!("{p}_rnn_bf")))?,
                ln_rnn: l.ln(&format!("{p}_rnn_ffn"))?,
                cq: l.linear(&format!("{p}_context_Wq"), Some(&format!("{p}_context_bq")))?,
                ck: l.linear(&format!("{p}_context_Wk"), Some(&format!("{p}_context_bk")))?,
                cv: l.linear(&format!("{p}_context_Wv"), Some(&format!("{p}_context_bv")))?,
                co: l.linear(&format!("{p}_context_Wo"), Some(&format!("{p}_context_bo")))?,
                ln_ctx: l.ln(&format!("{p}_context_Wo"))?,
                w1: l.linear(&format!("{p}_ffn_W1"), Some(&format!("{p}_ffn_b1")))?,
                w2: l.linear(&format!("{p}_ffn_W2"), Some(&format!("{p}_ffn_b2")))?,
                ln_ffn: l.ln(&format!("{p}_ffn_ffn"))?,
            });
        }
        let out_qa = l.get("none_QuantMultA")?.as_f32()[0];
        let out_bias = l.vec("decoder_ff_logit_out_b")?;

        let nt = dim / 2;
        let inc = (10000f32).ln() / (nt as f32 - 1.0);
        let freqs: Vec<f32> = (0..nt).map(|i| (i as f32 * -inc).exp()).collect();
        // sinusoidal position signal for the first POS_CACHE positions (longer ones are computed on demand)
        let mut pos_table = vec![0f32; POS_CACHE * dim];
        for p in 0..POS_CACHE {
            for i in 0..nt {
                let a = p as f32 * freqs[i];
                pos_table[p * dim + i] = a.sin();
                pos_table[p * dim + nt + i] = a.cos();
            }
        }
        Ok((Model { dim, heads: yml.heads, vocab, enc, dec, emb, emb_q, out_qa, out_bias, freqs, pos_table }, yml))
    }

    /// Output layer over the full vocabulary or a shortlist of target ids.
    pub fn output_layer(&self, ids: Option<Vec<u32>>) -> OutputLayer {
        match ids {
            Some(ids) => {
                let w = self.emb_q.gather_rows(&ids);
                let bias = ids.iter().map(|&i| self.out_bias[i as usize]).collect();
                OutputLayer { ids: Some(ids), w, bias }
            }
            None => OutputLayer {
                ids: None,
                w: self.emb_q.duplicate(),
                bias: self.out_bias.clone(),
            },
        }
    }

    fn add_position(&self, x: &mut [f32], pos: usize) {
        let d = self.dim;
        let nt = d / 2;
        let scale = (d as f32).sqrt();
        for v in x.iter_mut() {
            *v *= scale;
        }
        if pos < POS_CACHE {
            for (v, s) in x.iter_mut().zip(&self.pos_table[pos * d..(pos + 1) * d]) {
                *v += s;
            }
            return;
        }
        for i in 0..nt {
            let a = pos as f32 * self.freqs[i];
            x[i] += a.sin();
            x[nt + i] += a.cos();
        }
    }

    /// Encode a batch of token sequences (each already terminated by EOS).
    pub fn encode(&self, seqs: &[&[u32]]) -> Encoded {
        let _t = crate::prof::start(crate::prof::ENCODE);
        let d = self.dim;
        let lens: Vec<usize> = seqs.iter().map(|s| s.len()).collect();
        let mut offsets = Vec::with_capacity(seqs.len() + 1);
        let mut total = 0;
        for &l in &lens {
            offsets.push(total);
            total += l;
        }
        offsets.push(total);

        let mut x = vec![0f32; total * d];
        for (s, seq) in seqs.iter().enumerate() {
            for (t, &tok) in seq.iter().enumerate() {
                let row = &mut x[(offsets[s] + t) * d..(offsets[s] + t + 1) * d];
                row.copy_from_slice(&self.emb[tok as usize * d..(tok as usize + 1) * d]);
                self.add_position(row, t);
            }
        }

        let (mut qs, mut ks, mut vs, mut ctx) = (Vec::new(), Vec::new(), Vec::new(), vec![0f32; total * d]);
        let (mut tmp, mut scratch, mut scores) = (Vec::new(), Vec::new(), Vec::new());
        for layer in &self.enc {
            layer.wq.apply(&x, total, &mut scratch, &mut qs);
            layer.wk.apply(&x, total, &mut scratch, &mut ks);
            layer.wv.apply(&x, total, &mut scratch, &mut vs);
            for s in 0..seqs.len() {
                let (a, b) = (offsets[s] * d, offsets[s + 1] * d);
                kernels::attend(&qs[a..b], &ks[a..b], &vs[a..b], lens[s], lens[s], self.heads, d, &mut ctx[a..b], &mut scores);
            }
            layer.wo.apply(&ctx, total, &mut scratch, &mut tmp);
            add_residual_ln(&mut tmp, &x, d, &layer.ln_att);
            std::mem::swap(&mut x, &mut tmp);

            layer.w1.apply(&x, total, &mut scratch, &mut tmp);
            kernels::relu_inplace(&mut tmp);
            let mut h2 = Vec::new();
            layer.w2.apply(&tmp, total, &mut scratch, &mut h2);
            add_residual_ln(&mut h2, &x, d, &layer.ln_ffn);
            x = h2;
        }

        let mut cross_k = Vec::new();
        let mut cross_v = Vec::new();
        for layer in &self.dec {
            let (mut k, mut v) = (Vec::new(), Vec::new());
            layer.ck.apply(&x, total, &mut scratch, &mut k);
            layer.cv.apply(&x, total, &mut scratch, &mut v);
            cross_k.push(k);
            cross_v.push(v);
        }
        Encoded { offsets, lens, cross_k, cross_v }
    }

    /// Number of f32 recurrent-state values per decoder row.
    pub fn state_size(&self) -> usize {
        self.dec.len() * self.dim
    }

    /// One decoder step for `rows` hypotheses. `prev[r]` is the previously emitted token
    /// (`None` at position 0), `sent[r]` the sentence index in `enc`, `cells` the
    /// `rows x state_size` SSRU states (updated in place). Returns log-probabilities
    /// `rows x out.width()`.
    pub fn step(&self, enc: &Encoded, out: &OutputLayer, prev: &[Option<u32>], pos: &[usize], sent: &[usize], cells: &mut [f32]) -> Vec<f32> {
        let _t = crate::prof::start(crate::prof::DEC_LAYERS);
        let d = self.dim;
        let r = prev.len();
        let ss = self.state_size();
        let mut x = vec![0f32; r * d];
        for i in 0..r {
            let row = &mut x[i * d..(i + 1) * d];
            if let Some(t) = prev[i] {
                row.copy_from_slice(&self.emb[t as usize * d..(t as usize + 1) * d]);
            }
            self.add_position(row, pos[i]);
        }

        let (mut scratch, mut tmp, mut xw, mut f, mut q) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut scores = Vec::new();
        let mut ctx = vec![0f32; r * d];
        for (li, layer) in self.dec.iter().enumerate() {
            // SSRU: c = sigmoid(f) * c_prev + (1 - sigmoid(f)) * (x W);  out = relu(c)
            layer.rnn_w.apply(&x, r, &mut scratch, &mut xw);
            layer.rnn_wf.apply(&x, r, &mut scratch, &mut f);
            let mut o = vec![0f32; r * d];
            for i in 0..r {
                let c = &mut cells[i * ss + li * d..i * ss + (li + 1) * d];
                kernels::ssru_gate(&f[i * d..(i + 1) * d], &xw[i * d..(i + 1) * d], c, &mut o[i * d..(i + 1) * d]);
            }
            add_residual_ln(&mut o, &x, d, &layer.ln_rnn);
            x = o;

            // cross attention against the cached encoder keys / values
            layer.cq.apply(&x, r, &mut scratch, &mut q);
            for i in 0..r {
                let s = sent[i];
                let (a, b) = (enc.offsets[s] * d, enc.offsets[s + 1] * d);
                kernels::attend(&q[i * d..(i + 1) * d], &enc.cross_k[li][a..b], &enc.cross_v[li][a..b], 1, enc.lens[s], self.heads, d, &mut ctx[i * d..(i + 1) * d], &mut scores);
            }
            layer.co.apply(&ctx, r, &mut scratch, &mut tmp);
            add_residual_ln(&mut tmp, &x, d, &layer.ln_ctx);
            std::mem::swap(&mut x, &mut tmp);

            layer.w1.apply(&x, r, &mut scratch, &mut tmp);
            kernels::relu_inplace(&mut tmp);
            let mut h2 = Vec::new();
            layer.w2.apply(&tmp, r, &mut scratch, &mut h2);
            add_residual_ln(&mut h2, &x, d, &layer.ln_ffn);
            x = h2;
        }

        // output projection (tied embeddings) + log-softmax
        drop(_t);
        let _t = crate::prof::start(crate::prof::DEC_LOGITS);
        kernels::quantize(&x, r, d, self.out_qa, out.w.kp, &mut scratch);
        let n = out.w.n;
        let mut logits = vec![0f32; r * n];
        kernels::gemm(&scratch, r, &out.w, self.out_qa, Some(&out.bias), &mut logits);
        kernels::log_softmax_rows(&mut logits, n);
        logits
    }
}

/// out = LayerNorm(out + residual)
fn add_residual_ln(out: &mut [f32], residual: &[f32], dim: usize, ln: &Ln) {
    kernels::add_layer_norm(out, residual, dim, &ln.g, &ln.b);
}

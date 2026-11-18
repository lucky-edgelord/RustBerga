//! Subset of the Marian / Bergamot decoder configuration that this port honours.

use crate::error::Result;
use serde::Deserialize;
use serde_yaml::Value;

#[derive(Debug, Clone)]
pub struct Options {
    pub beam_size: usize,
    pub normalize: f32,
    pub word_penalty: f32,
    pub max_length_factor: f32,
    pub max_length_break: usize,
    pub mini_batch_words: usize,
    pub cpu_threads: usize,
    /// (first_num, best_num) if a shortlist is configured.
    pub shortlist: Option<(usize, usize)>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            beam_size: 1,
            normalize: 1.0,
            word_penalty: 0.0,
            max_length_factor: 2.0,
            max_length_break: 128,
            mini_batch_words: 1024,
            cpu_threads: 0,
            shortlist: None,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct RawConfig {
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub vocabs: Vec<String>,
    #[serde(default)]
    pub shortlist: Option<Vec<Value>>,
    #[serde(rename = "beam-size")]
    pub beam_size: Option<usize>,
    pub normalize: Option<f32>,
    #[serde(rename = "word-penalty")]
    pub word_penalty: Option<f32>,
    #[serde(rename = "max-length-factor")]
    pub max_length_factor: Option<f32>,
    #[serde(rename = "max-length-break")]
    pub max_length_break: Option<usize>,
    #[serde(rename = "mini-batch-words")]
    pub mini_batch_words: Option<usize>,
    #[serde(rename = "cpu-threads")]
    pub cpu_threads: Option<usize>,
}

impl RawConfig {
    pub fn parse(s: &str) -> Result<RawConfig> {
        Ok(serde_yaml::from_str(s)?)
    }

    pub fn shortlist_path(&self) -> Option<String> {
        match self.shortlist.as_ref()?.first()? {
            Value::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn options(&self) -> Options {
        let d = Options::default();
        let sl = self.shortlist.as_ref().and_then(|v| {
            if !matches!(v.first(), Some(Value::String(_))) {
                return None;
            }
            let num = |i: usize| v.get(i).and_then(|x| x.as_u64()).map(|x| x as usize);
            Some((num(1).unwrap_or(100), num(2).unwrap_or(100)))
        });
        Options {
            beam_size: self.beam_size.unwrap_or(d.beam_size).max(1),
            normalize: self.normalize.unwrap_or(d.normalize),
            word_penalty: self.word_penalty.unwrap_or(d.word_penalty),
            max_length_factor: self.max_length_factor.unwrap_or(d.max_length_factor),
            max_length_break: self.max_length_break.unwrap_or(d.max_length_break),
            mini_batch_words: self.mini_batch_words.unwrap_or(d.mini_batch_words),
            cpu_threads: self.cpu_threads.unwrap_or(d.cpu_threads),
            shortlist: sl,
        }
    }
}

/// The architecture description embedded in the model (`special:model.yml`).
#[derive(Debug, Clone, Deserialize)]
pub struct ModelYml {
    #[serde(rename = "dim-emb")]
    pub dim_emb: usize,
    #[serde(rename = "enc-depth")]
    pub enc_depth: usize,
    #[serde(rename = "dec-depth")]
    pub dec_depth: usize,
    #[serde(rename = "transformer-heads")]
    pub heads: usize,
    #[serde(rename = "transformer-dim-ffn")]
    pub dim_ffn: usize,
    #[serde(rename = "transformer-ffn-activation", default = "relu")]
    pub ffn_activation: String,
    #[serde(rename = "transformer-decoder-autoreg", default)]
    pub decoder_autoreg: String,
    #[serde(rename = "dec-cell", default)]
    pub dec_cell: String,
    #[serde(rename = "tied-embeddings-all", default)]
    pub tied_embeddings_all: bool,
    #[serde(rename = "dim-vocabs", default)]
    pub dim_vocabs: Vec<usize>,
}

fn relu() -> String {
    "relu".into()
}

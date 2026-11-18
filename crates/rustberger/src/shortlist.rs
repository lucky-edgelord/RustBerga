//! Binary lexical shortlist (`lex.*.s2t.bin`): restricts output logits to likely target words.

use crate::bail;
use crate::error::Result;

pub struct Shortlist {
    first_num: usize,
    offsets: Vec<u64>,
    lists: Vec<u32>,
}

impl Shortlist {
    pub fn from_bytes(buf: &[u8]) -> Result<Shortlist> {
        if buf.len() < 48 {
            bail!("shortlist too short");
        }
        let rd = |i: usize| u64::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
        let (first_num, w2o, sl) = (rd(2) as usize, rd(4) as usize, rd(5) as usize);
        let expected = 48 + w2o * 8 + sl * 4;
        if expected != buf.len() {
            bail!("shortlist header claims {expected} bytes but file has {}", buf.len());
        }
        let offsets = buf[48..48 + w2o * 8].chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let lists = buf[48 + w2o * 8..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(Shortlist { first_num, offsets, lists })
    }

    /// Sorted, deduplicated target ids for a batch of source ids; size padded to a multiple of 8.
    pub fn generate(&self, src_ids: impl Iterator<Item = u32>, trg_vocab: usize) -> Vec<u32> {
        let mut trg = vec![false; trg_vocab];
        let mut seen = vec![false; self.offsets.len().saturating_sub(1).max(1)];
        for t in trg.iter_mut().take(self.first_num) {
            *t = true;
        }
        for s in src_ids {
            let s = s as usize;
            if s + 1 >= self.offsets.len() || seen[s] {
                continue;
            }
            seen[s] = true;
            for j in self.offsets[s]..self.offsets[s + 1] {
                if let Some(t) = trg.get_mut(self.lists[j as usize] as usize) {
                    *t = true;
                }
            }
        }
        let mut ones = trg.iter().filter(|&&b| b).count();
        let mut i = self.first_num;
        while i < trg_vocab && ones % 8 != 0 {
            if !trg[i] {
                trg[i] = true;
                ones += 1;
            }
            i += 1;
        }
        trg.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i as u32).collect()
    }
}

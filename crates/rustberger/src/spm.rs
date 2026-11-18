//! Pure-Rust SentencePiece (unigram) processor: model-proto reader, the
//! precompiled-charsmap (`nmt_nfkc`) normaliser and Viterbi encoding/decoding.

use crate::bail;
use crate::error::Result;

const SPACE: &str = "\u{2581}"; // ▁
const UNK_PENALTY: f32 = 10.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

struct Piece {
    text: String,
    score: f32,
    ty: PieceType,
}

pub struct SentencePiece {
    pieces: Vec<Piece>,
    pub unk_id: u32,
    pub eos_id: u32,
    pub bos_id: i32,
    pub pad_id: i32,
    trie: Trie,
    unk_score: f32,
    charsmap: Option<CharsMap>,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

// ------------------------------------------------------------------ protobuf

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Reader { b, p: 0 }
    }
    fn done(&self) -> bool {
        self.p >= self.b.len()
    }
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            if self.p >= self.b.len() || shift > 63 {
                bail!("bad varint in sentencepiece model");
            }
            let c = self.b[self.p];
            self.p += 1;
            v |= ((c & 0x7f) as u64) << shift;
            if c & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
        }
    }
    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()? as usize;
        if self.p + n > self.b.len() {
            bail!("truncated sentencepiece model");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    /// Returns (field number, wire type); caller must consume the payload.
    fn tag(&mut self) -> Result<(u32, u8)> {
        let t = self.varint()?;
        Ok(((t >> 3) as u32, (t & 7) as u8))
    }
    fn skip(&mut self, wt: u8) -> Result<()> {
        match wt {
            0 => {
                self.varint()?;
            }
            1 => self.p += 8,
            2 => {
                self.bytes()?;
            }
            5 => self.p += 4,
            _ => bail!("unsupported protobuf wire type {wt}"),
        }
        Ok(())
    }
    fn f32(&mut self) -> Result<f32> {
        if self.p + 4 > self.b.len() {
            bail!("truncated float");
        }
        let v = f32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        Ok(v)
    }
}

// --------------------------------------------------------------- charsmap

/// darts-clone double array + normalised-string pool, as emitted by SentencePiece.
struct CharsMap {
    units: Vec<u32>,
    pool: Vec<u8>,
}

impl CharsMap {
    fn parse(blob: &[u8]) -> Result<CharsMap> {
        if blob.len() < 4 {
            bail!("charsmap too short");
        }
        let tsz = u32::from_le_bytes(blob[..4].try_into().unwrap()) as usize;
        if 4 + tsz > blob.len() || tsz % 4 != 0 {
            bail!("corrupt charsmap");
        }
        let units = blob[4..4 + tsz].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(CharsMap { units, pool: blob[4 + tsz..].to_vec() })
    }

    /// Longest-prefix match: returns (value offset into pool, bytes consumed).
    fn longest_prefix(&self, key: &[u8]) -> Option<(usize, usize)> {
        let u = &self.units;
        let off = |unit: u32| ((unit >> 10) << ((unit & 0x200) >> 6)) as usize;
        let mut pos = 0usize;
        let mut unit = *u.first()?;
        pos ^= off(unit);
        let mut best = None;
        for (i, &c) in key.iter().enumerate() {
            pos ^= c as usize;
            unit = *u.get(pos)?;
            if unit & ((1 << 31) | 0xff) != c as u32 {
                break;
            }
            pos ^= off(unit);
            if (unit >> 8) & 1 == 1 {
                let leaf = *u.get(pos)?;
                best = Some(((leaf & 0x7fff_ffff) as usize, i + 1));
            }
        }
        best
    }

    fn normalized(&self, off: usize) -> &[u8] {
        let end = self.pool[off..].iter().position(|&b| b == 0).map_or(self.pool.len(), |p| off + p);
        &self.pool[off..end]
    }
}

// ------------------------------------------------------------------- trie

/// Compact byte trie over piece strings.
struct Trie {
    // node -> (first_edge, edge_count, piece_id or u32::MAX)
    nodes: Vec<(u32, u32, u32)>,
    edges: Vec<(u8, u32)>,
}

impl Trie {
    fn build(words: &[(Vec<u8>, u32)]) -> Trie {
        #[derive(Default)]
        struct N {
            kids: std::collections::BTreeMap<u8, usize>,
            id: Option<u32>,
        }
        let mut tmp = vec![N::default()];
        for (w, id) in words {
            let mut cur = 0;
            for &b in w {
                cur = match tmp[cur].kids.get(&b) {
                    Some(&n) => n,
                    None => {
                        tmp.push(N::default());
                        let n = tmp.len() - 1;
                        tmp[cur].kids.insert(b, n);
                        n
                    }
                };
            }
            tmp[cur].id = Some(*id);
        }
        let mut nodes = Vec::with_capacity(tmp.len());
        let mut edges = Vec::new();
        // flatten preserving node indices
        for n in &tmp {
            let start = edges.len() as u32;
            for (&b, &c) in &n.kids {
                edges.push((b, c as u32));
            }
            nodes.push((start, n.kids.len() as u32, n.id.unwrap_or(u32::MAX)));
        }
        Trie { nodes, edges }
    }

    #[inline]
    fn child(&self, node: u32, b: u8) -> Option<u32> {
        let (s, c, _) = self.nodes[node as usize];
        let e = &self.edges[s as usize..(s + c) as usize];
        e.binary_search_by_key(&b, |x| x.0).ok().map(|i| e[i].1)
    }
}

// ----------------------------------------------------------------- impl

fn utf8_len(b: u8) -> usize {
    match b {
        0..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

impl SentencePiece {
    pub fn from_bytes(buf: &[u8]) -> Result<SentencePiece> {
        let mut r = Reader::new(buf);
        let mut pieces = Vec::new();
        let (mut unk, mut bos, mut eos, mut pad) = (0i32, 1i32, 2i32, -1i32);
        let mut model_type = 1u64;
        let mut charsmap_blob: Vec<u8> = Vec::new();
        let (mut dummy, mut rm_ws, mut esc) = (true, true, true);
        while !r.done() {
            let (f, wt) = r.tag()?;
            match (f, wt) {
                (1, 2) => {
                    let mut pr = Reader::new(r.bytes()?);
                    let (mut text, mut score, mut ty) = (String::new(), 0f32, 1u64);
                    while !pr.done() {
                        let (pf, pw) = pr.tag()?;
                        match (pf, pw) {
                            (1, 2) => text = String::from_utf8_lossy(pr.bytes()?).into_owned(),
                            (2, 5) => score = pr.f32()?,
                            (3, 0) => ty = pr.varint()?,
                            _ => pr.skip(pw)?,
                        }
                    }
                    let ty = match ty {
                        1 => PieceType::Normal,
                        2 => PieceType::Unknown,
                        3 => PieceType::Control,
                        4 => PieceType::UserDefined,
                        5 => PieceType::Unused,
                        6 => PieceType::Byte,
                        _ => PieceType::Normal,
                    };
                    pieces.push(Piece { text, score, ty });
                }
                (2, 2) => {
                    let mut tr = Reader::new(r.bytes()?);
                    while !tr.done() {
                        let (tf, tw) = tr.tag()?;
                        match (tf, tw) {
                            (3, 0) => model_type = tr.varint()?,
                            (40, 0) => unk = tr.varint()? as i64 as i32,
                            (41, 0) => bos = tr.varint()? as i64 as i32,
                            (42, 0) => eos = tr.varint()? as i64 as i32,
                            (43, 0) => pad = tr.varint()? as i64 as i32,
                            _ => tr.skip(tw)?,
                        }
                    }
                }
                (3, 2) => {
                    let mut nr = Reader::new(r.bytes()?);
                    while !nr.done() {
                        let (nf, nw) = nr.tag()?;
                        match (nf, nw) {
                            (2, 2) => charsmap_blob = nr.bytes()?.to_vec(),
                            (3, 0) => dummy = nr.varint()? != 0,
                            (4, 0) => rm_ws = nr.varint()? != 0,
                            (5, 0) => esc = nr.varint()? != 0,
                            _ => nr.skip(nw)?,
                        }
                    }
                }
                (_, w) => r.skip(w)?,
            }
        }
        if model_type != 1 {
            bail!("only unigram SentencePiece models are supported (model_type={model_type})");
        }
        if pieces.is_empty() {
            bail!("empty sentencepiece model");
        }
        let min_score = pieces.iter().filter(|p| p.ty == PieceType::Normal).map(|p| p.score).fold(f32::INFINITY, f32::min);
        let words: Vec<(Vec<u8>, u32)> = pieces
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p.ty, PieceType::Normal | PieceType::UserDefined))
            .map(|(i, p)| (p.text.as_bytes().to_vec(), i as u32))
            .collect();
        let charsmap = if charsmap_blob.is_empty() { None } else { Some(CharsMap::parse(&charsmap_blob)?) };
        Ok(SentencePiece {
            trie: Trie::build(&words),
            unk_score: min_score - UNK_PENALTY,
            pieces,
            unk_id: unk.max(0) as u32,
            eos_id: eos.max(0) as u32,
            bos_id: bos,
            pad_id: pad,
            charsmap,
            add_dummy_prefix: dummy,
            remove_extra_whitespaces: rm_ws,
            escape_whitespaces: esc,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// One normalisation step at the head of `input`: (normalised bytes, bytes consumed).
    fn normalize_prefix<'a>(&'a self, input: &'a [u8]) -> (&'a [u8], usize) {
        if let Some(cm) = &self.charsmap {
            if let Some((off, len)) = cm.longest_prefix(&input[..input.len().min(64)]) {
                return (cm.normalized(off), len);
            }
        }
        let n = utf8_len(input[0]).min(input.len());
        (&input[..n], n)
    }

    pub fn normalize(&self, text: &str) -> String {
        let mut input = text.as_bytes();
        if self.remove_extra_whitespaces {
            while !input.is_empty() {
                let (p, n) = self.normalize_prefix(input);
                if p != b" " {
                    break;
                }
                input = &input[n..];
            }
            if input.is_empty() {
                return String::new();
            }
        }
        let mut out: Vec<u8> = Vec::with_capacity(input.len() + 8);
        let add_ws = |out: &mut Vec<u8>| out.extend_from_slice(if self.escape_whitespaces { SPACE.as_bytes() } else { b" " });
        if self.add_dummy_prefix {
            add_ws(&mut out);
        }
        let mut prev_space = self.remove_extra_whitespaces;
        while !input.is_empty() {
            let (mut sp, n) = self.normalize_prefix(input);
            while prev_space && sp.first() == Some(&b' ') {
                sp = &sp[1..];
            }
            for &c in sp {
                if self.escape_whitespaces && c == b' ' {
                    out.extend_from_slice(SPACE.as_bytes());
                } else {
                    out.push(c);
                }
            }
            if !sp.is_empty() {
                prev_space = sp.last() == Some(&b' ');
            }
            input = &input[n..];
        }
        if self.remove_extra_whitespaces {
            let sp = if self.escape_whitespaces { SPACE.as_bytes() } else { b" " };
            while out.ends_with(sp) {
                out.truncate(out.len() - sp.len());
            }
        }
        String::from_utf8(out).unwrap_or_default()
    }

    /// Text -> piece ids (no EOS appended).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let norm = self.normalize(text);
        let s = norm.as_bytes();
        let n = s.len();
        if n == 0 {
            return vec![];
        }
        #[derive(Clone, Copy)]
        struct Best {
            score: f32,
            start: i32,
            id: u32,
        }
        let mut best = vec![Best { score: f32::NEG_INFINITY, start: -1, id: 0 }; n + 1];
        best[0].score = 0.0;
        let mut pos = 0;
        while pos < n {
            let clen = utf8_len(s[pos]).min(n - pos);
            let base = best[pos].score;
            if base == f32::NEG_INFINITY {
                pos += clen;
                continue;
            }
            let mut has_single = false;
            let mut node = 0u32;
            let mut i = pos;
            while i < n {
                match self.trie.child(node, s[i]) {
                    Some(c) => node = c,
                    None => break,
                }
                i += 1;
                let id = self.trie.nodes[node as usize].2;
                if id != u32::MAX {
                    let cand = base + self.pieces[id as usize].score;
                    if best[i].start < 0 || cand > best[i].score {
                        best[i] = Best { score: cand, start: pos as i32, id };
                    }
                    if i - pos == clen {
                        has_single = true;
                    }
                }
            }
            if !has_single {
                let cand = base + self.unk_score;
                let e = pos + clen;
                if best[e].start < 0 || cand > best[e].score {
                    best[e] = Best { score: cand, start: pos as i32, id: self.unk_id };
                }
            }
            pos += clen;
        }
        let mut out = Vec::new();
        let mut e = n;
        while e > 0 {
            let b = best[e];
            out.push(b.id);
            e = b.start as usize;
        }
        out.reverse();
        // SentencePiece merges runs of consecutive unknown characters into one <unk>.
        out.dedup_by(|b, a| *a == self.unk_id && *b == self.unk_id);
        out
    }

    /// Piece ids -> text (control pieces are dropped, unknowns render as ` ⁇ `).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut s = String::new();
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else { continue };
            match p.ty {
                PieceType::Control => {}
                PieceType::Unknown => s.push_str(" \u{2047} "),
                _ => s.push_str(&p.text),
            }
        }
        let s = s.replace(SPACE, " ");
        s.strip_prefix(' ').map(str::to_owned).unwrap_or(s)
    }
}

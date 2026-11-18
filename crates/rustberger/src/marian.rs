//! Reader for the Marian binary model format (`*.bin`), including the
//! architecture-agnostic `intgemm8` layout used by Firefox Translations models.

use crate::bail;
use crate::error::Result;

pub const TYPE_FLOAT32: u64 = 1028;
pub const TYPE_INTGEMM8: u64 = 16641;

pub struct Item {
    pub name: String,
    pub ty: u64,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Item {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn rows(&self) -> usize {
        self.elements() / self.cols()
    }
    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }
    pub fn as_f32(&self) -> Vec<f32> {
        self.data[..self.elements() * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    /// For `intgemm8` items: the int8 payload (stored transposed, `[cols][rows]`
    /// row-major, i.e. `[N][K]` for a `K x N` weight) and its quantisation multiplier.
    pub fn as_i8_with_mult(&self) -> (&[i8], f32) {
        let n = self.elements();
        let q = f32::from_le_bytes(self.data[n..n + 4].try_into().unwrap());
        // SAFETY: i8 and u8 have identical layout.
        let s = unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const i8, n) };
        (s, q)
    }
}

fn rd_u64(b: &[u8], p: &mut usize) -> Result<u64> {
    if *p + 8 > b.len() {
        bail!("model truncated");
    }
    let v = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap());
    *p += 8;
    Ok(v)
}

pub fn parse(buf: &[u8]) -> Result<Vec<Item>> {
    let mut p = 0usize;
    let ver = rd_u64(buf, &mut p)?;
    if ver != 1 {
        bail!("unsupported binary model version {ver}");
    }
    let n = rd_u64(buf, &mut p)? as usize;
    let mut hdr = Vec::with_capacity(n);
    for _ in 0..n {
        hdr.push([
            rd_u64(buf, &mut p)? as usize,
            rd_u64(buf, &mut p)?  as usize,
            rd_u64(buf, &mut p)? as usize,
            rd_u64(buf, &mut p)? as usize,
        ]);
    }
    let mut items: Vec<Item> = Vec::with_capacity(n);
    for h in &hdr {
        let nl = h[0];
        if p + nl > buf.len() {
            bail!("model truncated (names)");
        }
        let name = String::from_utf8_lossy(&buf[p..p + nl.saturating_sub(1)]).into_owned();
        p += nl;
        items.push(Item { name, ty: h[1] as u64, shape: vec![], data: vec![] });
    }
    for (it, h) in items.iter_mut().zip(&hdr) {
        let sl = h[2];
        if p + 4 * sl > buf.len() {
            bail!("model truncated (shapes)");
        }
        it.shape = (0..sl)
            .map(|i| i32::from_le_bytes(buf[p + 4 * i..p + 4 * i + 4].try_into().unwrap()) as usize)
            .collect();
        p += 4 * sl;
    }
    let off = rd_u64(buf, &mut p)? as usize;
    p += off;
    for (it, h) in items.iter_mut().zip(&hdr) {
        let dl = h[3];
        if p + dl > buf.len() {
            bail!("model truncated (data for {})", it.name);
        }
        it.data = buf[p..p + dl].to_vec();
        p += dl;
    }
    Ok(items)
}

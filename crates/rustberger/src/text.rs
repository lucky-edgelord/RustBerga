//! Sentence segmentation that preserves the original whitespace between sentences.

pub struct Segment {
    /// Whitespace that precedes the sentence in the source text (copied verbatim to the output).
    pub gap: String,
    pub text: String,
}

const NONBREAKING: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "inc", "ltd", "co", "corp", "gen", "col", "lt", "sgt", "capt", "rev", "hon",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "e.g", "i.e", "cf", "fig", "approx", "dept", "est", "p.m", "a.m", "u.s",
];

fn is_closer(c: char) -> bool {
    matches!(c, '"' | '\'' | ')' | ']' | '}' | '»' | '”' | '’' | '›' | '」' | '』' | '）')
}

fn is_cjk_term(c: char) -> bool {
    matches!(c, '。' | '！' | '？' | '｡')
}

/// Returns the segments and the trailing whitespace of the text.
pub fn split(text: &str) -> (Vec<Segment>, String) {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte = |i: usize| if i < n { chars[i].0 } else { text.len() };
    let mut segs = Vec::new();
    let mut i = 0;
    let mut gap_start = 0usize;
    while i < n {
        // skip whitespace to find the sentence start
        while i < n && chars[i].1.is_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        let start = i;
        let mut end = i; // exclusive, last non-ws char + 1
        while i < n {
            let c = chars[i].1;
            if c == '\n' {
                break;
            }
            if !c.is_whitespace() {
                end = i + 1;
            }
            let term = matches!(c, '.' | '!' | '?' | '…') || is_cjk_term(c);
            if term {
                // swallow repeated terminators and closing punctuation
                let mut j = i + 1;
                while j < n && (matches!(chars[j].1, '.' | '!' | '?' | '…') || is_closer(chars[j].1) || is_cjk_term(chars[j].1)) {
                    j += 1;
                }
                let at_end = j >= n;
                let next_ws = !at_end && chars[j].1.is_whitespace();
                let mut boundary = at_end || is_cjk_term(c) && !at_end;
                if !boundary && next_ws {
                    // look at what follows the whitespace
                    let mut k = j;
                    while k < n && chars[k].1.is_whitespace() && chars[k].1 != '\n' {
                        k += 1;
                    }
                    let next = if k < n { Some(chars[k].1) } else { None };
                    let nl = next.map_or(true, |x| x == '\n');
                    let upper_or_other = next.map_or(true, |x| !x.is_lowercase());
                    if c == '.' {
                        let word_start = (0..i).rev().find(|&q| chars[q].1.is_whitespace()).map_or(0, |q| q + 1);
                        let word: String = chars[word_start.max(start)..i].iter().map(|x| x.1).collect::<String>().trim_start_matches(|ch: char| !ch.is_alphanumeric()).to_lowercase();
                        let single_letter = word.chars().count() == 1 && word.chars().all(|ch| ch.is_alphabetic());
                        let numeric = !word.is_empty() && word.chars().all(|ch| ch.is_ascii_digit());
                        let nonbreak = NONBREAKING.contains(&word.as_str()) || single_letter || (numeric && !upper_or_other);
                        boundary = (nl || upper_or_other) && !nonbreak;
                    } else {
                        boundary = nl || upper_or_other;
                    }
                }
                if boundary {
                    end = j;
                    i = j;
                    break;
                }
            }
            i += 1;
        }
        let gap = text[gap_start..byte(start)].to_string();
        segs.push(Segment { gap, text: text[byte(start)..byte(end)].to_string() });
        gap_start = byte(end);
        i = i.max(end);
    }
    (segs, text[gap_start..].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(t: &str) {
        let (s, tail) = split(t);
        let mut o = String::new();
        for x in &s {
            o.push_str(&x.gap);
            o.push_str(&x.text);
        }
        o.push_str(&tail);
        assert_eq!(o, t);
    }

    #[test]
    fn splits_and_preserves_whitespace() {
        let t = "Hello world. This is Dr. Smith's test!  Is it ok? yes\n\nNew para.  ";
        roundtrip(t);
        let (s, tail) = split(t);
        let v: Vec<&str> = s.iter().map(|x| x.text.as_str()).collect();
        assert_eq!(v, ["Hello world.", "This is Dr. Smith's test!", "Is it ok? yes", "New para."]);
        assert_eq!(tail, "  ");
        assert_eq!(s[2].gap, "  ");
        assert_eq!(s[3].gap, "\n\n");
    }

    #[test]
    fn cjk_and_edge_cases() {
        for t in ["", "   ", "\n", "一。二！三", "no terminator", "1. Item one\n2. Item two", "He said \"Stop.\" Then left.", "3.14 is pi. ok"] {
            roundtrip(t);
        }
        assert_eq!(split("一。二！三").0.len(), 3);
        assert_eq!(split("3.14 is pi. ok").0.len(), 1);
    }
}

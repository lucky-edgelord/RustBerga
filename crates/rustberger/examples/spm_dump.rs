use std::io::BufRead;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let sp = rustberger::spm::SentencePiece::from_bytes(&std::fs::read(&a[1]).unwrap()).unwrap();
    for line in std::io::stdin().lock().lines() {
        let l = line.unwrap();
        let ids = sp.encode(&l);
        println!("{}\t{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "), sp.decode(&ids));
    }
}

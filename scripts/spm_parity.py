import subprocess, sys, sentencepiece as spm
model='/home/claude/models/vocab.enes.spm'
s=spm.SentencePieceProcessor(model_file=model)
cases = [
 "Hello, how are you?", "  leading and   multiple    spaces  ", "tab\tseparated", "Café ﬁne ① ２０２４ ＡＢＣ", "日本語のテキスト and 中文",
 "a😀😀b", "𓀀 hieroglyph", "https://example.com/path?x=1&y=2", "The quick brown fox jumps over the lazy dog.",
 "Dr. Smith went to Washington. He arrived at 5 p.m.", "naïve façade coöperate", "x²+y²=z² ½ ™ ©", "", " ", "   ", "a", "Ünïcödé", "Привет, мир!", "العربية", "한국어 문장",
 "Mixed nbsp and　ideographic space", "line1\nline2", "…—–“quoted”‘single’", "ǅ ǈ Ǌ", "ﷺ", "é combining", "​zero width", "ＨＥＬＬＯ ｗｏｒｌｄ",
 "100% sure! #hashtag @user", "C++ and C# and .NET", "O'Neill's e-mail: o'neill@example.org",
]
import random
random.seed(1)
words="the of and to in is that for it as was with be by on not he I this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split()
for _ in range(300):
    cases.append(" ".join(random.choice(words) for _ in range(random.randint(1,25))).capitalize()+random.choice([".","?","!",""]))
open('/home/claude/scripts/cases.txt','w').write("\n".join(c for c in cases if "\n" not in c))
cases=[c for c in cases if "\n" not in c]
out=subprocess.run(['/home/claude/rustberger/target/release/examples/spm_dump',model],input="\n".join(cases)+"\n",capture_output=True,text=True).stdout.split("\n")
bad=0
for c,line in zip(cases,out):
    ids,_,dec=line.partition("\t")
    ref=s.encode(c)
    mine=[int(x) for x in ids.split()]
    refdec=s.decode(ref)
    if mine!=ref or dec!=refdec:
        bad+=1
        if bad<=8: print("MISMATCH",repr(c),"\n  ref ",ref,repr(refdec),"\n  mine",mine,repr(dec))
print(f"{len(cases)-bad}/{len(cases)} identical")

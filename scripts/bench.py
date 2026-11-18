import os, subprocess, sys, time, statistics, resource, json
M='/home/claude/models'
C='/home/claude/cpp_build/app/bergamot'
R='/home/claude/rustberger/target/release/rustberger'
base=open(f'{M}/bench.yml').read()
def cfg(threads):
    p=f'{M}/bench_t{threads}.yml'
    open(p,'w').write(base.replace('cpu-threads: 0',f'cpu-threads: {0 if threads==1 else threads}'))
    return p
def run(cmd, inp):
    t=time.perf_counter()
    r=subprocess.run(cmd,stdin=open(inp),capture_output=True,cwd=M)
    dt=time.perf_counter()-t
    return dt, r.stdout, r.returncode
def rss(cmd, inp):
    pid_r=subprocess.Popen(cmd,stdin=open(inp),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,cwd=M)
    _,st,ru=os.wait4(pid_r.pid,0)
    return ru.ru_maxrss/1024
res={}
for threads in (1,2):
    c=cfg(threads)
    cmds={'C++ (bergamot)':[C,'--model-config-paths',c],'Rust (rustberger)':[R,'--model-config-paths',c,'--cpu-threads',str(threads)]}
    for name,cmd in cmds.items():
        run(cmd,f'{M}/tiny.txt')  # warm page cache
        start=[run(cmd,f'{M}/tiny.txt')[0] for _ in range(5)]
        full=[]; out=None
        for _ in range(5):
            dt,o,rc=run(cmd,f'{M}/bench_en.txt'); assert rc==0,(name,rc); full.append(dt); out=o
        res[(threads,name)]=dict(startup=statistics.median(start),total=statistics.median(full),best=min(full),rss=rss(cmd,f'{M}/bench_en.txt'),out=out)
words=len(open(f'{M}/bench_en.txt').read().split())
print(f'input: {words} words\n')
print(f'{"threads":>7} {"engine":<20}{"startup":>9}{"total":>9}{"translate":>11}{"words/s":>9}{"peak RSS":>10}')
for (t,n),v in res.items():
    tr=v['total']-v['startup']
    print(f'{t:>7} {n:<20}{v["startup"]:>8.3f}s{v["total"]:>8.3f}s{tr:>10.3f}s{words/tr:>9.0f}{v["rss"]:>8.0f}MB')
for t in (1,2):
    a=res[(t,'C++ (bergamot)')]; b=res[(t,'Rust (rustberger)')]
    print(f'threads={t}: Rust translate-time speedup vs C++ = {(a["total"]-a["startup"])/(b["total"]-b["startup"]):.2f}x ; end-to-end = {a["total"]/b["total"]:.2f}x')
a=res[(1,'C++ (bergamot)')]['out'].decode().split('\n'); b=res[(1,'Rust (rustberger)')]['out'].decode().split('\n')
same=sum(x==y for x,y in zip(a,b)); print(f'\nline-exact output agreement: {same}/{len(a)} lines ({len(a)} vs {len(b)})')
open('/home/claude/models/out_cpp.txt','wb').write(res[(1,'C++ (bergamot)')]['out']); open('/home/claude/models/out_rust.txt','wb').write(res[(1,'Rust (rustberger)')]['out'])
json.dump({f'{k[0]}|{k[1]}':{x:y for x,y in v.items() if x!='out'} for k,v in res.items()},open('/home/claude/models/bench_result.json','w'),indent=1)

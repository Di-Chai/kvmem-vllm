"""CPU-only FP8 index finalization; same compiled library, interleaved off/on."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np

from vllm_kvmem import _native


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if not (1 <= args.records <= 256 and 1 <= args.tokens <= 256 and
            1 <= args.scoring_layers <= args.layers <= 32 and
            1 <= args.heads <= 16 and 1 <= args.dim <= 512 and
            5 <= args.rounds <= 100 and 1 <= args.warmup <= 20):
        raise ValueError('Bounded CPU component geometry required')
    rng = np.random.default_rng(941)
    sources = [rng.integers(0, 256, (args.layers,args.tokens,args.heads,args.dim),dtype=np.uint8)
               for _ in range(args.records)]
    for raw in sources:
        raw[raw == 127] = 126
        raw[raw == 255] = 254
    outputs = {flag:[np.empty((args.scoring_layers,args.heads,args.dim),dtype=np.float32)
                    for _ in sources] for flag in (False,True)}
    jobs = {flag:[(raw,out,args.tokens,True) for raw,out in zip(sources,outputs[flag])]
            for flag in outputs}
    library = Path(_native.__file__)
    sha = digest(library)
    calls = {False:[], True:[]}
    for trial in range(args.warmup + args.rounds):
        for flag in ((False,True) if trial%2 == 0 else (True,False)):
            cpu = time.process_time()
            wall = time.perf_counter()
            used = _native.finalize_mean_batch(jobs[flag],vectorized=flag)
            seconds = time.perf_counter()-wall
            cpu_seconds = time.process_time()-cpu
            expected = (args.records if _native.finalize_mean_simd_supported() and args.heads*args.dim >= 8 else 0)
            assert used == (expected if flag else None)
            if trial >= args.warmup:
                calls[flag].append(dict(seconds=seconds,cpu_seconds=cpu_seconds,vectorized_records=used))
        for off,on in zip(outputs[False],outputs[True]):
            np.testing.assert_array_equal(off.view(np.uint32),on.view(np.uint32))
    assert digest(library) == sha, 'Compiled library changed during the component comparison'
    summary = {}
    for flag,values in calls.items():
        times = [r['seconds']*1000 for r in values]
        summary['vectorized' if flag else 'scalar'] = dict(
            median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),
            min_ms=min(times),max_ms=max(times),
            median_process_cpu_ms=float(np.median([r['cpu_seconds']*1000 for r in values])))
    quota = Path('/sys/fs/cgroup/cpu.max')
    output = dict(diagnostic_only=True,geometry={k:getattr(args,k) for k in
        ('records','tokens','layers','scoring_layers','heads','dim')},
        warmup_rounds=args.warmup,measured_rounds=args.rounds,
        equality='FP32 sums bitwise equal after every interleaved round',
        source_bytes=sum(r.nbytes for r in sources),
        scored_source_bytes=args.records*args.scoring_layers*args.tokens*args.heads*args.dim,
        extra_workers=0,simd_supported=_native.finalize_mean_simd_supported(),
        library=str(library.resolve()),library_sha256=sha,
        bindings_sha256=digest(Path(__file__).resolve().parents[1]/'csrc/bindings.cpp'),
        machine=platform.machine(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_quota=quota.read_text().strip() if quota.exists() else None,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        summary=summary,samples={'vectorized' if k else 'scalar':v for k,v in calls.items()},
        caution='Only sealed FP8 index finalization. Pipeline overlap and full-model throughput require separate tests.')
    output['median_speedup'] = summary['scalar']['median_ms']/summary['vectorized']['median_ms']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,median_speedup=output['median_speedup'],output=str(args.output)),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name,default in [('records',64),('tokens',32),('layers',17),('scoring-layers',16),
                         ('heads',4),('dim',256),('rounds',30),('warmup',6)]:
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())

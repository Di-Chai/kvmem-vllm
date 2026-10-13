"""Measure foreground sealed-page submission and completion, with CPU cache.

Source records are allocated before timing. Real namespace/generation admission
and CPU cache copies run, but NVMe I/O is replaced by a discard-only operation.
Run without model requests; these timings are not serving throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch

from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.cached_storage import CachedArchive
from vllm_kvmem.config import KVMemConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error('repeats must be positive')
    torch.set_num_threads(1)
    record_bytes = 16 * 128 * 4 * 256 * 3
    report = dict(scope='CPU sealed flush; preallocated sources; includes CPU cache; discard-only NVMe; no GPU',
                  record_bytes=record_bytes, repeats=args.repeats, cases={})
    with tempfile.TemporaryDirectory(prefix='kvmem-seal-') as directory:
        for blocks in (1, 8):
            archive = CachedArchive(str(Path(directory)/str(blocks)), record_bytes, blocks+2,
                cache_bytes=blocks*record_bytes, namespace_bytes=blocks*record_bytes)
            tier = archive.tier
            class DiscardIo:
                def __getattr__(self, name):
                    return getattr(tier, name)
                def io(self, spans, payload, write):
                    assert write
                    return dict(bytes=0, syscalls=0, duration_ns=0)
            archive.tier = DiscardIo()
            archive.executor.submit(lambda: None).result()
            raw = torch.ones((16, blocks*128, 4, 256), dtype=torch.bfloat16)
            values = torch.full(raw.shape, 7, dtype=torch.uint8)
            expected = hashlib.sha256(raw[:, :128].contiguous().view(torch.uint8).numpy().tobytes()
                                      + values[:, :128].contiguous().numpy().tobytes()).hexdigest()
            samples = {mode: dict(foreground_ms=[], completion_ms=[]) for mode in ('sync','async')}
            cfg = KVMemConfig(select_budget=1024, gen_budget=1024, repository_blocks=32,
                              sink_blocks=1, recent_blocks=0)
            try:
                for repeat in range(args.repeats+1):
                    for mode in (('sync','async') if repeat % 2 else ('async','sync')):
                        archive.async_seal = mode == 'async'
                        authority = RequestAuthority(f'{repeat}-{mode}', cfg, archive, 16, 4, 256,
                                                     resident_records=blocks+2)
                        flush = authority.flush
                        authority.flush = lambda: True
                        assert authority.append(raw, values, history_begin=0, local_rope_begin=0)
                        authority.flush = flush
                        started = time.perf_counter()
                        assert flush()
                        submitted = time.perf_counter()
                        for job, _ in authority.writes:
                            job.result(timeout=30)
                        authority.poll()
                        completed = time.perf_counter()
                        assert all(r.digest == expected and r.archived for r in authority.records.values())
                        assert authority.close()
                        if repeat:
                            samples[mode]['foreground_ms'].append((submitted-started)*1000)
                            samples[mode]['completion_ms'].append((completed-started)*1000)
            finally:
                assert archive.close()
            report['cases'][str(blocks)] = {mode:{phase:dict(median=statistics.median(rows),
                min=min(rows), max=max(rows), samples=rows) for phase,rows in phases.items()}
                for mode,phases in samples.items()}
    report['all_digests_verified'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

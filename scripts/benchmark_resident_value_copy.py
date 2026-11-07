"""Exclusive-GPU copy diagnosis; never a model-throughput measurement."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace


def run(output):
    import torch
    import benchmark_block_graph as lifecycle
    from run_recall_comparison import gpu_lock
    from vllm_kvmem.split_assembly import gather_resident_values
    if output.exists():
        raise ValueError('Preserve the original measurement')
    rows = []
    with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
        if lifecycle.gpu_processes():
            raise ValueError('GPU belongs to another experiment')
        torch.manual_seed(571)
        for tokens in (16384, 36864):
            page, heads, dim = 1616, 4, 256
            pages = (tokens+page-1)//page + 2
            cache = torch.randint(0, 256, (pages, page, heads, 2, dim), device='cuda', dtype=torch.uint8)
            table = torch.randperm(pages, device='cuda', dtype=torch.int32)
            bases = torch.randperm(tokens//32, device='cuda', dtype=torch.int64)*32
            layer = SimpleNamespace(value=cache[:, :, :, 1], kernel_page_table=table)
            destinations = [torch.empty(tokens*heads*dim, device='cuda', dtype=torch.uint8) for _ in range(2)]
            def copy(tiled):
                return gather_resident_values((layer,), bases, 32, destinations[int(tiled)], tiled=tiled)
            for tiled in (False, True):
                for _ in range(10):
                    copy(tiled)
            torch.cuda.synchronize()
            if not torch.equal(destinations[0], destinations[1]):
                raise ValueError('Tiled copy changed GPU Value bytes')
            measurements = {False: [], True: []}
            for tiled in (False, True, True, False):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    copy(tiled)
                end.record(); end.synchronize()
                measurements[tiled].append(start.elapsed_time(end)/50)
            rows.append(dict(tokens=tokens, byte_equal=True,
                             original_ms=measurements[False], tiled_ms=measurements[True],
                             original_mean_ms=statistics.mean(measurements[False]),
                             tiled_mean_ms=statistics.mean(measurements[True])))
    module = Path(__file__).resolve().parents[1]/'src/vllm_kvmem/split_assembly.py'
    result = dict(rows=rows, split_assembly_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                  scope='CUDA copy-only ABBA for one layer with native page1616 and interleaved KV. '
                        'Not assembly wall time or model/prefill throughput.')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())

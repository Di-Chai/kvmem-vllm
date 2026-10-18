"""CPU metadata microbenchmark; no GPU work or serving performance claims.

Build two identical frames before measurement, then alternate old Python and
experimental native preparation. Existing completion/commit logic is unchanged.
Nested Python-stage clocks are diagnostic overhead, included and reported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from vllm_kvmem import _native
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import Phase, RequestMemory
from vllm_kvmem.restore_timing import RestoreTiming


class Ready:
    def query(self):
        return True


def build(tokens, block_tokens, gen_tokens, repository_blocks):
    config = KVMemConfig(block_tokens=block_tokens, select_budget=tokens,
        gen_budget=gen_tokens, repository_blocks=repository_blocks,
        sink_blocks=0, recent_blocks=0)
    memory = RequestMemory('microbenchmark', config)
    assert memory.begin_append(tokens, 0) is not None
    memory.attach_completion(Ready())
    assert memory.commit_append(tokens)
    return memory, memory.checkpoint_window()


def measure(memory, checkpoint, arm, observe=True):
    timing = RestoreTiming(enabled=observe)
    started, cpu_started = time.perf_counter(), time.thread_time()
    if arm == 'old_python':
        plan = memory.restore_window(checkpoint, timing=timing)
    else:
        with timing.stage('native_restore_frame'):
            plan = memory._store.restore_frame(checkpoint.prefix, checkpoint.records)
        # Explicit microbenchmark adapter, not a serving default.
        memory._plan, memory.phase, memory._submitted = plan, Phase.ASSEMBLING, False
    cpu_end, end = time.thread_time(), time.perf_counter()
    prepared = dict(wall_seconds=end-started, thread_cpu_seconds=cpu_end-cpu_started)
    started, cpu_started = time.perf_counter(), time.thread_time()
    memory.attach_completion(Ready())
    assert memory.commit_selection()
    cpu_end, end = time.thread_time(), time.perf_counter()
    committed = dict(wall_seconds=end-started, thread_cpu_seconds=cpu_end-cpu_started)
    if tuple(memory._store.working_records()) != checkpoint.records:
        raise AssertionError('Restored frame differs from its exact checkpoint')
    return dict(arm=arm, prepare=prepared, unchanged_commit=committed, timing=timing.summary())


def summary(rows):
    result = {}
    for arm in ('old_python', 'native_adapter'):
        for stage in ('prepare', 'unchanged_commit'):
            for metric in ('wall_seconds', 'thread_cpu_seconds'):
                values = sorted(row[stage][metric] for row in rows if row['arm'] == arm)
                result.setdefault(arm, {}).setdefault(stage, {})[metric] = dict(
                    median=statistics.median(values), maximum=max(values),
                    p95=values[min(len(values)-1, int(.95 * len(values)))])
    return result


def run(args):
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError('Require positive iterations and nonnegative warmup')
    if not hasattr(_native.Store, 'restore_frame'):
        raise RuntimeError('Native extension must be rebuilt before this standalone CPU benchmark')
    frames = {arm: build(args.tokens, args.block_tokens, args.gen_tokens, args.repository_blocks)
              for arm in ('old_python', 'native_adapter')}
    rows = []
    try:
        if frames['old_python'][1].records != frames['native_adapter'][1].records:
            raise AssertionError('Frames differ before measurement')
        for _ in range(args.warmup):
            for arm, (memory, checkpoint) in frames.items():
                measure(memory, checkpoint, arm, observe=False)
        for iteration in range(args.iterations):
            arms = list(frames) if iteration % 2 == 0 else list(reversed(frames))
            for arm in arms:
                memory, checkpoint = frames[arm]
                rows.append(dict(iteration=iteration, **measure(memory, checkpoint, arm)))
        result = dict(diagnostic_only=True, performance_claim_allowed=False,
            scope='CPU-only metadata on already-built equal dense frames; no KV payload, model, GPU, '
                  'concurrent engine, GC-policy change or serving throughput. Nested old-Python stage '
                  'clocks add diagnostic overhead; their intervals overlap and must not be summed.',
            settings=vars(args) | {'output': str(args.output)}, summary=summary(rows), samples=rows,
            native_extension=str(Path(_native.__file__).resolve()),
            native_extension_sha256=hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    finally:
        for memory, _ in frames.values():
            if not memory.close():
                raise RuntimeError('CPU benchmark unexpectedly retained an incomplete transaction')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=131072)
    parser.add_argument('--block-tokens', type=int, default=32)
    parser.add_argument('--gen-tokens', type=int, default=32768)
    parser.add_argument('--repository-blocks', type=int, default=32768)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())

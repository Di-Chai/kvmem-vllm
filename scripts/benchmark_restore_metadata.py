"""Compare restore message metadata on a saved real plan without GPU work.

The saved plan supplies actual message coordinates. Live KV ranges are a
synthetic bounded tail, and no model, active KV bytes, or Agent is executed.
Alternate unchanged and optimized Python modules in one CPU process.
"""
import argparse
import cProfile
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType, SimpleNamespace


PACKAGE = Path(__file__).resolve().parents[1] / 'src/vllm_kvmem'
SOURCES = ('prompt_spans.py', 'block_graph.py', 'query_boundary_history.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, directory):
    package = ModuleType(name)
    package.__path__ = [str(directory), str(PACKAGE)]
    sys.modules[name] = package
    return SimpleNamespace(**{key:importlib.import_module(name+'.'+key) for key in
                               ('prompt_spans', 'block_graph', 'query_boundary_history')})


class Frame:
    def __init__(self, modules, plan, live_records):
        self.modules = modules
        self.position = plan['prompt_tokens'] - 1
        self.binding = SimpleNamespace(tokens=range(plan['prompt_tokens']), message_spans=(),
            locality_spans=tuple((row['begin'], row['end'], row['role'])
                                for row in plan['kvmem_locality_spans']),
            locality_provenance=plan['kvmem_locality_provenance'])
        first = max(0, (plan['prompt_tokens']-live_records*32+31)//32*32)
        records = {i:SimpleNamespace(history_begin=begin, count=min(32, plan['prompt_tokens']-begin))
                   for i,begin in enumerate(range(first,plan['prompt_tokens'],32))}
        self.history = modules.query_boundary_history.QueryBoundaryHistory(
            'metadata-benchmark', 32, initial_position=self.position)
        self.authority = SimpleNamespace(block_graph=None, query_boundaries=self.history,
                                        records=records, history_end=self.position)
        modules.prompt_spans.install_selection_metadata(self.authority, self.binding)
        self.saved = self.history.snapshot()

    def exercise(self):
        incoming, _ = self.modules.prompt_spans.graph_metadata(self.binding)
        self.modules.block_graph._merge_spans(self.saved.message_spans,
            self.saved.message_end_known, incoming, self.position)
        self.history.restore(self.saved)
        self.modules.prompt_spans.install_selection_metadata(
            self.authority, self.binding, validated_prefix=self.position)

    def state(self):
        h = self.history
        return (h.message_spans, h.message_end_known, h.locality_provenance,
                h.accepted_through, h.runtime_epoch, h.branch_id, h.revision)

    def close(self):
        self.saved.close()
        self.history.close()


def summary(values):
    values = sorted(values)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=values[min(len(values)-1,int(.95*len(values)))], maximum=max(values))


def run(args):
    if args.output.exists() or args.iterations < 1 or args.warmup < 0 or args.live_records < 1:
        raise ValueError('Require a new output and valid sample counts')
    baseline = args.baseline.resolve()
    if any(not (baseline/rel).is_file() for rel in SOURCES):
        raise ValueError('Baseline must contain the three unchanged metadata modules')
    plan = json.loads(args.fixture.read_text())
    frames = {name:Frame(load('_restore_metadata_'+name, path), plan, args.live_records)
              for name,path in [('before',baseline),('after',PACKAGE)]}
    samples = []
    try:
        assert frames['before'].state() == frames['after'].state()
        for iteration in range(args.warmup+args.iterations):
            order = list(frames) if iteration % 2 == 0 else list(reversed(frames))
            for name in order:
                start, cpu = time.perf_counter(), time.thread_time()
                frames[name].exercise()
                elapsed, consumed = time.perf_counter()-start, time.thread_time()-cpu
                if iteration >= args.warmup:
                    samples.append(dict(iteration=iteration-args.warmup, arm=name,
                                        wall_seconds=elapsed, thread_cpu_seconds=consumed))
            assert frames['before'].state() == frames['after'].state(), 'Metadata or epoch changed'
        profiles = {}
        for name,frame in frames.items():
            profiler = cProfile.Profile()
            profiler.runcall(frame.exercise)
            top = sorted(profiler.getstats(), key=lambda entry:entry.totaltime, reverse=True)[:20]
            profiles[name] = [dict(function=(str(entry.code) if isinstance(entry.code,str) else
                                            entry.code.co_filename+':'+str(entry.code.co_firstlineno)+':'+entry.code.co_name),
                                   calls=entry.callcount, inclusive_seconds=entry.totaltime,
                                   self_seconds=entry.inlinetime) for entry in top]
        assert frames['before'].state() == frames['after'].state()
        timings = {name:{metric:summary([s[metric] for s in samples if s['arm']==name])
                         for metric in ('wall_seconds','thread_cpu_seconds')} for name in frames}
        result = dict(scope='CPU message metadata only. Actual plan coordinates; synthetic bounded live tail. '
                      'Excludes checkpoint token identity, model state, GPU copies, engine scheduling and Agent execution. '
                      'Does not measure serving throughput or the full restore interval.',
            at=time.time(),fixture=str(args.fixture.resolve()),fixture_sha256=sha(args.fixture),
            prompt_tokens=plan['prompt_tokens'],message_spans=len(plan['kvmem_locality_spans']),
            live_records=len(frames['after'].authority.records),state_identical=True,
            timings=timings,samples=samples,profiles=profiles,
            wall_median_reduction_fraction=1-timings['after']['wall_seconds']['median']/timings['before']['wall_seconds']['median'],
            source_sha256={name:{rel:sha(path/rel) for rel in SOURCES}
                           for name,path in [('before',baseline),('after',PACKAGE)]})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n')
        print(json.dumps({key:result[key] for key in ('prompt_tokens','message_spans','state_identical',
            'timings','wall_median_reduction_fraction')}) )
    finally:
        for frame in frames.values():frame.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--live-records',type=int,default=8192)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=30)
    run(parser.parse_args())

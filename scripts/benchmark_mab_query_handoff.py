"""Finite cold/warm MAB diagnostic; never resumes or scores the formal cohort."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from audit_layerwise_gilopt import compare, read, save_new, sha, tree

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
SOURCE = REPO/'artifacts/mab-pressure-optimized-20261105-r3/formal'


def insert_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Frozen source anchor is not unique: '+before[:80])
    return text.replace(before, after)


def method_source(code, name):
    nodes = [n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, name
    node = nodes[0]
    return ''.join(code.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def instrument_restore(content):
    original = method_source(content,'_restore')
    measured = insert_once(original,'    def _restore(self, request_id):\n',
        '    def _restore(self, request_id):\n        restore_started = time.perf_counter()\n        restore_stages = {}\n')
    measured = insert_once(measured,'            assembly = assembler.assemble(authority, plan)\n',
        "            stage_started = time.perf_counter()\n            restore_stages['claim_and_prepare'] = stage_started - restore_started\n            assembly = assembler.assemble(authority, plan)\n")
    for anchor, stage in [('            restore_sync(self, request_id, checkpoint)\n','shared_kv_restore'),
                          ('            event = self.model_adapter.restore(checkpoint.model, request_id)\n','layerwise_kv_restore'),
                          ('            self._verify_restored(request_id, checkpoint)\n','recurrent_restore')]:
        measured = insert_once(measured,anchor,
            '            now = time.perf_counter()\n'+f"            restore_stages['{stage}'] = now - stage_started\n"+
            '            stage_started = now\n'+anchor)
    measured = insert_once(measured,'            self._verify_restored(request_id, checkpoint)\n',
        "            self._verify_restored(request_id, checkpoint)\n            restore_stages['verification'] = time.perf_counter() - stage_started\n")
    measured += '''
        output = getattr(self.runner, '_kvmem_probe_output', None)
        if output is not None:
            import json
            with (output/'sync_restore_timing.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(request_id=request_id,
                    started_at=restore_started, finished_at=time.perf_counter(),
                    host_stages_seconds=restore_stages,
                    scope='Host wall intervals including existing CUDA waits; not exclusive GPU kernel time.'))+'\\n')'''
    return insert_once(content,original,measured)


def prepare(root, qualification=None, sample_start=2):
    if root.exists():
        raise ValueError('Preserve previous diagnostic')
    assert tree(SOURCE/'runtime-v1') == read(SOURCE/'runtime-proof.json')
    root.mkdir(parents=True)
    fixture = read(SOURCE/'fixture.json')
    # Full original inputs, one group so causal/batch numerics are controlled.
    rows = [r for r in fixture['rows'] if r['id'].startswith('005_')][sample_start:sample_start+4]
    qualification_files = {}
    if qualification is not None:
        gate = read(qualification/'equivalence.json')
        assert gate['passed'] and gate['cases'] == 4
        assert read(qualification/'status.json')['status'] == 'completed'
        for arm in ('baseline','candidate'):
            assert tree(qualification/arm/'runtime-v1') == read(qualification/arm/'runtime-proof.json')
            assert not read(qualification/arm/'run/cleanup.json')['gpu_processes']
            for name in ('runtime-proof.json','execution-audit.json','inference-performance.json'):
                qualification_files[str(qualification/arm/name)] = sha(qualification/arm/name)
        qualification_files[str(qualification/'equivalence.json')] = sha(qualification/'equivalence.json')
        rows += [r for r in fixture['rows'] if r['id'].startswith('006_')][sample_start:sample_start+4]
    assert len(rows) == (8 if qualification else 4)
    ids = {r['id'] for r in rows}
    fixture.update(rows=rows, case_ids=[r['id'] for r in rows], cases=len(rows),
                   qualification=qualification is None, restore_digest_qualification=qualification is None,
                   diagnostic_only=True, performance_observation=True)
    fixture['bindings'] = [b for b in fixture['bindings'] if b['id'] in ids]
    profile = read(SOURCE/'profile.json')
    assert [profile['kvmem'][k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')] == [36864,16384,32,41,0]
    code = (REPO/'src/vllm_kvmem/async_refresh.py').read_text()
    klass = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == 'AsyncRefresh')
    fn = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'poll_query_completions')
    function = ''.join(code.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno])+'\n\n'
    for arm, enabled in (('baseline', False), ('candidate', True)):
        dest = root/arm
        dest.mkdir()
        runtime = dest/'runtime-v1'
        shutil.copytree(SOURCE/'runtime-v1', runtime, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        path = runtime/'src/vllm_kvmem/async_refresh.py'
        content = path.read_text()
        content = insert_once(content, '        self.trace_lock=threading.Lock()\n',
                              '        self.trace_lock=threading.Lock()\n        self.query_post_forward_reap = '+str(enabled)+'\n')
        content = insert_once(content, '    def _forward(self, job, tokens, begin, *, next_token=False):\n',
                              function+'    def _forward(self, job, tokens, begin, *, next_token=False):\n')
        path.write_text(content)
        path = runtime/'src/vllm_kvmem/connector.py'
        content = insert_once(path.read_text(), '        self.pressure_completion_poll = None\n',
                              '        self.pressure_completion_poll = None\n        self.query_completion_poll = None\n')
        content = insert_once(content, '            self.pressure_completion_poll()\n',
                              '            self.pressure_completion_poll()\n        if self.query_completion_poll is not None:\n            self.query_completion_poll()\n')
        path.write_text(content)
        path = runtime/'src/vllm_kvmem/runner.py'
        content = insert_once(path.read_text(), '                if timing is not None:\n                    connector.step_timings = timing.poll\n',
                              "                connector.query_completion_poll = (refresh.poll_query_completions if refresh is not None and getattr(refresh, 'query_post_forward_reap', False) else None)\n                if timing is not None:\n                    connector.step_timings = timing.poll\n")
        path.write_text(content)
        before = read(SOURCE/'runtime-proof.json')
        if qualification is not None:
            # The new diagnostic is identical in both arms and introduces no
            # extra GPU fence. Preserve all other frozen session-worker code.
            path = runtime/'src/vllm_kvmem/session_worker.py'
            content = path.read_text()
            path.write_text(instrument_restore(content))
        after = tree(runtime)
        changed = sorted(k for k in after if after[k] != before.get(k))
        expected = ['src/vllm_kvmem/async_refresh.py','src/vllm_kvmem/connector.py','src/vllm_kvmem/runner.py']
        if qualification is not None: expected += ['src/vllm_kvmem/session_worker.py']
        assert changed == expected
        save_new(dest/'fixture.json', fixture)
        save_new(dest/'profile.json', profile)
        save_new(dest/'verification-profile.json', dict(profile, verify_session_restore=True, verify_session_restore_digest=True))
        save_new(dest/'runtime-proof.json', after)
        files = {str(dest/n):sha(dest/n) for n in ('fixture.json','profile.json','verification-profile.json','runtime-proof.json')}
        files.update({b['request']:b['sha256'] for b in fixture['bindings']})
        save_new(dest/'files.json', files)
    shutil.copy2(__file__, root/'launch.py')
    save_new(root/'plan.json', dict(source=str(SOURCE), cases=len(rows), warm_cases=6 if qualification else 3,
        controls_unchanged=True, formal_resume=False, external_grading=False,
        qualification_only=qualification is None, qualification_files=qualification_files,
        launch_sha256=sha(root/'launch.py'),
        files={str(root/a/'files.json'):sha(root/a/'files.json') for a in ('baseline','candidate')}))


def run_arm(root, arm):
    dest = root/arm
    plan = read(root/'plan.json')
    assert sha(root/'launch.py') == plan['launch_sha256']
    assert sha(dest/'files.json') == plan['files'][str(dest/'files.json')]
    for name, digest in plan.get('qualification_files',{}).items():
        assert sha(Path(name)) == digest, name
    sys.path[:0] = [str(dest/'runtime-v1/src'),str(dest/'runtime-v1/scripts')]
    spec = importlib.util.spec_from_file_location('frozen_memory', dest/'runtime-v1/scripts/run_layerwise_memory.py')
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)
    def verify(path):
        assert path == dest
        for name, digest in read(dest/'files.json').items():
            assert sha(Path(name)) == digest, name
        assert tree(dest/'runtime-v1') == read(dest/'runtime-proof.json')
        return read(dest/'fixture.json'),read(dest/'profile.json')
    memory.verify = verify
    memory.run(dest)


def run(root):
    if (root/'process.json').exists():
        raise ValueError('Do not repeat first answers')
    save_new(root/'process.json', dict(pid=os.getpid(),at=time.time()))
    from run_recall_comparison import termination_guard, stop_owned
    child = None
    with termination_guard():
        try:
            for arm in ('baseline','candidate'):
                (root/'status.json').write_text(json.dumps(dict(status='running',active=arm)))
                with (root/(arm+'.log')).open('x') as log:
                    child = subprocess.Popen([sys.executable,'-B',str(root/'launch.py'),'arm','--root',str(root),'--arm',arm],
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                child.wait()
                if child.returncode:
                    raise RuntimeError(arm+' failed; retained evidence, no automatic retry')
            result = compare(root/'baseline',root/'candidate')
            save_new(root/'equivalence.json', result)
            summarize(root)
            (root/'status.json').write_text(json.dumps(dict(status='completed',equivalent=result['passed'])))
        except BaseException as error:
            (root/'status.json').write_text(json.dumps(dict(status='failed',error=repr(error))))
            raise
        finally:
            stop_owned(child)


def summarize(root):
    arms = {}
    for arm in ('baseline','candidate'):
        dest = root/arm
        performance = read(dest/'inference-performance.json')
        work = read(dest/'confirmed-work-performance.json')
        records = [read(p) for p in (dest/'run/answers/cases').glob('*/result.json')]
        warm = [r for r in records if r['prefix_reused']]
        gpu = []
        for line in (dest/'run/gpu.jsonl').open():
            row = json.loads(line)
            if performance['started_at'] <= row.get('at',0) <= performance['completed_at']:
                gpu.extend(d['gpu_utilization_percent'] for d in row.get('devices',[]) if d.get('gpu_utilization_percent') is not None)
        events = [json.loads(l) for l in (dest/'run/server/capture/async_refresh.jsonl').open()]
        arms[arm] = dict(
            inference_wall_seconds=performance['inference_wall_seconds'],
            confirmed_prefill_tokens=work['completed']['public_prefill_tokens'],
            confirmed_prefill_tokens_per_inference_second=work['public_prefill_tokens_per_inference_second'],
            completed=len(records),warm_completed=len(warm),
            warm_mean_seconds=statistics.mean(r['seconds'] for r in warm),
            warm_median_seconds=statistics.median(r['seconds'] for r in warm),
            output_tokens=sum(r['usage']['completion_tokens'] for r in records),
            gpu_mean_percent=statistics.mean(gpu) if gpu else None,
            gpu_sample_count=len(gpu),
            query_post_forward_completions=sum(e['stage']=='query_completion_reaped' for e in events),
            kv_byte_oracle_enabled=read(dest/'fixture.json')['restore_digest_qualification'])
    a,b = (arms[k] for k in ('baseline','candidate'))
    save_new(root/'performance-summary.json',dict(arms=arms,
        exact_equivalence=read(root/'equivalence.json')['passed'],
        observed_tps_change=b['confirmed_prefill_tokens_per_inference_second']/a['confirmed_prefill_tokens_per_inference_second']-1,
        warm_latency_change=b['warm_mean_seconds']/a['warm_mean_seconds']-1,
        scope='Matched full-input MAB mini-workload, includes cold/warm restore, Query, maintenance and output. Excludes model load. Executed CUDA-fenced public prefill tokens only; never counts reused P prefixes as computed work.',
        caveats=['One A/B is exploratory, not a stable gain.','Oracle-enabled qualification timing is not serving throughput.','Group count may be below configured max concurrency 8.','No automatic formal resume or external grading.']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run','arm'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('baseline','candidate'))
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--sample-start', type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare': prepare(root,args.qualification.resolve() if args.qualification else None,args.sample_start)
    elif args.action == 'run': run(root)
    else: run_arm(root,args.arm)

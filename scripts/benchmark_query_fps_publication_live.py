"""Two predetermined full-input live checks for the FPS publication change.

The original twelve-case fixture is retained intact. This is a two-case
integration/performance measurement, not a new complete quality benchmark.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np

import test_ruler_fps_halo_answers as halo

ROOT = Path('/data/chaidi/kvmem-vllm-experiments/query-fps-publication-live-20261023-r1')
REFERENCE = Path('/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/ruler-key-fps-halo-20261023-r1')
CASES = ('niah_multikey_2-0008', 'niah_multikey_2-0009')
read, save, sha, require = halo.read, halo.save, halo.sha, halo.require


def prepare(root, reference):
    require(read(reference/'execution-audit.json')['status'] == 'passed', 'Previous live audit missing')
    require(read(reference/'status.json')['status'] == 'completed', 'Previous live run incomplete')
    halo.prepare(root, halo.SOURCE_ROOT.resolve(), optimized=True)
    profile = read(root/halo.ARM/'profile.json')
    require(profile == read(reference/halo.ARM/'profile.json'), 'Live performance profile changed')
    files = {name: sha(reference/name) for name in ('fixture.json', 'execution-audit.json',
        'status.json', halo.ARM+'/profile.json', halo.ARM+'/cleanup.json')}
    for case in CASES:
        base = Path(halo.ARM)/'cases'/case
        manifest = read(reference/base/'hashes.json')
        halo.verify_files(reference/base, manifest)
        files.update({str(base/name): digest for name, digest in manifest.items()})
    plan = dict(case_ids=list(CASES), reference=str(reference), reference_files=files,
        scope='Two predetermined original full-256K cases; efficiency/integration check, not full RULER.',
        policy='Preserve inputs, seeds, sampling, profile, answers and failures; no retries.')
    save(root/'measurement-plan.json', plan)
    save(root/'measurement-proof.json', {'measurement-plan.json': sha(root/'measurement-plan.json')})
    save(root/'status.json', dict(status='prepared', cases=2, scope=plan['scope']))


def verify(root):
    halo.verify_files(root/'runtime', read(root/'runtime-proof.json')['files'])
    halo.verify_files(root, read(root/'prepared-proof.json'))
    halo.verify_files(root, read(root/'measurement-proof.json'))
    plan = read(root/'measurement-plan.json')
    require(tuple(plan['case_ids']) == CASES, 'Predetermined live subset changed')
    halo.verify_files(Path(plan['reference']), plan['reference_files'])
    return plan


def compare(reference, case, result, evidence):
    old = read(reference/halo.ARM/'cases'/case/'evidence.json')
    a, b = old['query_inputs'][0], evidence['query_inputs'][0]
    seeds_equal = all(a[key] == b[key] for key in ('starts', 'counts', 'score_mask',
        'transport_mask', 'mandatory_starts', 'budget', 'raw_scores'))
    oq = next(q for q in old['bounded'] if q['selection'] == 'query')
    nq = next(q for q in evidence['bounded'] if q['selection'] == 'query')
    same_selection = oq['selected_history_ranges'] == nq['selected_history_ranges']
    # Do not retry if generation or the comparison differs. Preserve evidence.
    return dict(id=case, seeds_exactly_equal=seeds_equal, selection_equal=same_selection,
        previous_fps=oq['query_key_fps_scoring'], candidate_fps=nq['query_key_fps_scoring'],
        official_case_score=result['official_case_score'], value_covered=result['value_covered'],
        completion_tokens=result['completion_tokens'], request_seconds=result['seconds'])


def run(root):
    require(Path(__file__).resolve() == root/'runtime/scripts'/Path(__file__).name,
        'Use the frozen live measurement controller')
    plan = verify(root)
    with (root/'attempt.json').open('x') as output:
        json.dump(dict(pid=os.getpid(), at=time.time()), output)
    fixture = read(root/'fixture.json'); base = root/halo.ARM; runtime = root/'runtime'
    profile = read(base/'profile.json'); capture = base/'server/capture'
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_', 'VLLM_', 'DEEPSEEK_'))}
    env.update(PYTHONPATH=str(runtime/'src'), PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false', MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', VLLM_LOG_STATS_INTERVAL='5')
    for toolkit in ('/usr/local/cuda-13', '/usr/local/cuda'):
        if (Path(toolkit)/'bin/nvcc').exists():
            env.update(CUDA_HOME=toolkit, PATH=toolkit+'/bin'+os.pathsep+env.get('PATH','')); break
    server = None; results = []; comparisons = []
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a+') as lock, halo.lifecycle.cleanup_on_termination():
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not halo.lifecycle.gpu_processes(), 'Foreign GPU owner; do not stop it')
            try:
                command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve', '--config',
                    str(base/'profile.json'), '--run-dir', str(base/'server')]
                with (base/'server.log').open('x') as log:
                    server = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                save(base/'server-process.json', dict(pid=server.pid, command=command))
                save(root/'status.json', dict(status='loading', pid=server.pid, at=time.time()))
                opener = halo.client.urllib.request.build_opener(halo.client.urllib.request.ProxyHandler({}))
                deadline = time.monotonic()+1200
                while True:
                    require(server.poll() is None, 'Live FPS server exited')
                    try:
                        with opener.open(f"http://127.0.0.1:{profile['port']}/health", timeout=1) as response:
                            if response.status == 200: break
                    except OSError:
                        pass
                    require(time.monotonic() < deadline, 'Live FPS server startup timeout')
                    time.sleep(1)
                rows = {row['id']: row for row in fixture['rows']}
                for case_id in CASES:
                    row = rows[case_id]; case = base/'cases'/case_id; case.mkdir(parents=True, exist_ok=False)
                    save(root/'status.json', dict(status='testing', case=case_id, completed=len(results), expected=2, at=time.time()))
                    source = Path(fixture['input_directory'])/(case_id+'.i32')
                    require(sha(source) == row['served_sha256'], 'Original input changed')
                    tokens = np.fromfile(source, dtype='<i4').tolist()
                    sampling = dict(fixture['sampling'], seed=row['seed']); session = 'fps-publication-'+uuid.uuid4().hex
                    save(case/'request.json', dict(sampling=sampling, session_id=session, input_sha256=row['served_sha256'],
                        query_span=row['served_query'], profile_sha256=sha(base/'profile.json')))
                    response = halo.client.stream(profile, session, tokens, row['served_query'], sampling, 3600)
                    save(case/'response.json', response)
                    result, evidence = halo.finish_case(root, fixture, row, profile, tokens, session, response, capture)
                    save(case/'result.json', result); save(case/'evidence.json', evidence)
                    (capture/'block_graph_inputs.jsonl').rename(case/'query-inputs.jsonl')
                    manifest = {name:sha(case/name) for name in (*halo.CASE_FILES, 'query-inputs.jsonl')}
                    save(case/'hashes.json', manifest); halo.verify_files(case, manifest)
                    comparisons.append(compare(Path(plan['reference']), case_id, result, evidence)); results.append(result)
                    save(root/'progress.json', dict(completed=len(results), expected=2, comparisons=comparisons))
                    print(json.dumps(dict(case=case_id, completed=len(results), score=result['official_case_score'])), flush=True)
            finally:
                remaining = halo.lifecycle.stop_owned_groups((server,)); gpu = sorted(halo.lifecycle.gpu_processes())
                save(base/'cleanup.json', dict(remaining_owned_groups=remaining, gpu_processes=gpu))
                require(not remaining and not gpu, 'Live FPS cleanup incomplete')
        verify(root)
        for case_id in CASES:
            case = base/'cases'/case_id; halo.verify_files(case, read(case/'hashes.json'))
            row = next(row for row in fixture['rows'] if row['id'] == case_id)
            request = read(case/'request.json')
            require(request['sampling'] == dict(fixture['sampling'], seed=row['seed']) and
                request['query_span'] == row['served_query'] and request['input_sha256'] == row['served_sha256'] and
                request['profile_sha256'] == sha(base/'profile.json'), 'Final request binding changed')
            require(sha(Path(fixture['input_directory'])/(case_id+'.i32')) == row['served_sha256'],
                'Original input changed during live measurement')
            evidence = read(case/'evidence.json'); recorded = deepcopy(read(case/'result.json')['selection_audit'])
            q = next(q for q in evidence['bounded'] if q['selection']=='query')
            g = next(g for g in evidence['graph'] if g['phase']=='query')
            replay = halo.audit_selection(q, g, evidence['query_inputs'][0], profile)
            for audit in (recorded, replay): audit['propagation'].pop('scoring_seconds', None)
            require(replay == recorded, 'Saved native group replay changed')
        save(root/'execution-audit.json', dict(status='passed', cases=2, frozen_source_and_inputs=True,
            per_case_execution_audit=True, native_group_picker_replay=True, gpu_cleanup=True,
            previous_answers_unchanged=True, scope=plan['scope']))
        save(root/'summary.json', dict(cases=2, comparisons=comparisons, scope=plan['scope'],
            timing_limit='Historical live intervals, not interleaved whole-model timing. Fixed-Q/K interleaved component timing is separate.'))
        save(root/'status.json', dict(status='completed', cases=2, at=time.time()))
    except BaseException as error:
        save(root/'failure.json', dict(error=repr(error), completed=len(results), at=time.time()))
        save(root/'status.json', dict(status='failed', completed=len(results), error=repr(error), at=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.root.resolve(), args.reference.resolve())
    else: run(args.root.resolve())

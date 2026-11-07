"""Frozen ABBA serving comparison with independent long-history session owners.

Cold requests prime four distinct complete prefixes. Warm phases offer two or
four concurrent HTTP requests, using the same twelve original queries in each.
Normal native scheduling is retained; measured batches prove achieved overlap.
Output mismatches are retained and compared with A/A and B/B repeatability.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from threading import Barrier
import time
import urllib.error
import urllib.request
import uuid

from audit_layerwise_gilopt import read, save_new, sha, tree
from benchmark_layerwise_no_probe import metrics, save

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO/'artifacts/mab-pressure-optimized-20261105-r3/formal'
GROUPS = ('005', '006', '007', '008')


def percentile(values, q):
    values = sorted(values)
    rank = (len(values)-1)*q
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo] + (values[hi]-values[lo])*(rank-lo)


def client_peak(responses):
    edges = [(r['started_at'], 1) for r in responses]
    edges += [(r['started_at']+r['seconds'], -1) for r in responses]
    active = peak = 0
    for _, change in sorted(edges):
        active += change
        peak = max(peak, active)
    return peak


def phase_summary(responses, offered):
    wall = max(r['started_at']+r['seconds'] for r in responses)-min(r['started_at'] for r in responses)
    tokens = sum(len(r['output_ids']) for r in responses)
    return dict(requests=len(responses), output_tokens=tokens, wall_seconds=wall,
        output_tokens_per_second=tokens/wall, requests_per_second=len(responses)/wall,
        ttft_p50_seconds=percentile([r['ttft_seconds'] for r in responses], .5),
        ttft_p95_seconds=percentile([r['ttft_seconds'] for r in responses], .95),
        request_p50_seconds=percentile([r['seconds'] for r in responses], .5),
        offered_concurrency=offered, observed_client_peak=client_peak(responses))


def prepare(root):
    from benchmark_memory import source
    if root.exists():
        raise ValueError('Use a new campaign directory')
    fixture = read(SOURCE/'fixture.json')
    groups = {g: [r for r in fixture['rows'] if r['id'].startswith(g+'_')][:4] for g in GROUPS}
    if any(len(rows) != 4 for rows in groups.values()):
        raise ValueError('Each independent history requires four original questions')
    prefixes = {}
    for g, rows in groups.items():
        prefix = min(r['query_begin'] for r in rows)//1024*1024
        inputs = [source(row)['tokens'][:prefix] for row in rows]
        if any(tokens != inputs[0] for tokens in inputs[1:]):
            raise ValueError('Questions do not share the claimed exact prefix: '+g)
        prefixes[g] = prefix
    rows = [row for group in groups.values() for row in group]
    ids = {r['id'] for r in rows}
    fixture.update(rows=rows, cases=len(rows), prefixes=prefixes, owners=list(GROUPS))
    fixture['bindings'] = [b for b in fixture['bindings'] if b['id'] in ids]
    profile = read(SOURCE/'profile.json')
    profile.update(port=19165, served_model_name='no-probe-concurrency',
                   verify_session_restore=False, verify_session_restore_digest=False)
    kv = profile['kvmem']
    reservations = sum(((max(r['input_tokens'] for r in group)+2*kv['gen_budget']+31)//32)*32*17*4*256*2
                       for group in groups.values())
    if reservations >= kv['cpu_read_cache_bytes']-16*1024**3:
        raise ValueError('Four retained histories exceed the measured host memory budget')
    root.mkdir(parents=True)
    for name in ('src', 'scripts'):
        shutil.copytree(REPO/name, root/'runtime'/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    save_new(root/'fixture.json', fixture)
    save_new(root/'profile.json', profile)
    save_new(root/'runtime-proof.json', tree(root/'runtime'))
    files = {str(SOURCE/n): sha(SOURCE/n) for n in ('fixture.json', 'profile.json')}
    files.update({r['source']: r['source_cache_sha256'] for r in rows})
    files.update({b['request']: b['sha256'] for b in fixture['bindings']})
    trials = [dict(name='r1-sequential', mode='sequential', levels=[2, 4]),
              dict(name='r2-no-probe', mode='no-probe', levels=[2, 4]),
              dict(name='r3-no-probe', mode='no-probe', levels=[4, 2]),
              dict(name='r4-sequential', mode='sequential', levels=[4, 2])]
    save_new(root/'plan.json', dict(trials=trials, files=files,
        fixture_sha256=sha(root/'fixture.json'), profile_sha256=sha(root/'profile.json'),
        cpu_reservation_bytes=reservations, client_credit_bytes=kv['cpu_read_cache_bytes']-16*1024**3,
        scope='Four distinct complete MAB histories, original sampling and output caps; '
              'four concurrent cold requests then the same twelve warm queries at C2 and C4. '
              'ABBA with balanced level order; native scheduling unchanged. '
              'Diagnostic byte/Q hashing disabled. No accuracy grading, retries, or input truncation.',
        timing='Client SSE first nonempty token_ids. Phase wall spans first HTTP start to final HTTP completion, '
               'including between-wave terminal waits; excludes loading and evidence analysis. '
               'Warm phases exclude cold priming; cold throughput is reported separately.'))
    save_new(root/'status.json', dict(status='prepared'))


def verify(root):
    plan = read(root/'plan.json')
    if tree(root/'runtime') != read(root/'runtime-proof.json'):
        raise ValueError('Frozen runtime changed')
    for name in ('fixture', 'profile'):
        if sha(root/(name+'.json')) != plan[name+'_sha256']:
            raise ValueError('Frozen '+name+' changed')
    for path, digest in plan['files'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Frozen source or request changed: '+path)
    return plan


def stream_request(profile, payload, barrier):
    # Serialize all long prompts before releasing this wave's barrier.
    request = urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
        data=json.dumps(dict(payload, stream=True, stream_options={'include_usage': True})).encode(),
        headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    barrier.wait(timeout=120)
    started = time.perf_counter()
    result = dict(output_ids=[], text='', usage=None, response_id=None, chunks=[], finish_reason=None)
    first = None
    done = False
    try:
        with opener.open(request, timeout=3600) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                result['response_id'] = event.get('id') or result['response_id']
                result['usage'] = event.get('usage') or result['usage']
                for choice in event.get('choices', []):
                    ids = choice.get('token_ids') or []
                    if ids:
                        at = time.perf_counter()-started
                        if first is None:
                            first = at
                        result['chunks'].append([at, len(ids)])
                        result['output_ids'].extend(ids)
                    result['text'] += choice.get('text') or ''
                    result['finish_reason'] = choice.get('finish_reason') or result['finish_reason']
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: '+error.read().decode()) from None
    if (not done or first is None or not result['usage'] or
            result['usage']['completion_tokens'] != len(result['output_ids']) or
            result['usage']['prompt_tokens'] != len(payload['prompt'])):
        raise RuntimeError('Incomplete concurrent response or token accounting mismatch')
    result.update(started_at=started, seconds=time.perf_counter()-started, ttft_seconds=first)
    return result


def collect_evidence(capture, results):
    by_rid = {r['response']['response_id']: {} for r in results}
    for filename, kind in [('layerwise_query.jsonl', 'events'), ('bounded_refresh.jsonl', 'refresh')]:
        for line in (capture/filename).open():
            event = json.loads(line)
            matches = [rid for rid in by_rid if event.get('request_id', '').startswith(rid+'-')]
            if len(matches) > 1:
                raise RuntimeError('Overlapping response identities')
            if matches:
                by_rid[matches[0]].setdefault(kind, []).append(event)
    for result in results:
        evidence = by_rid[result['response']['response_id']]
        layers = [e for e in evidence['events'] if e['event'] == 'layer_selected']
        query = [e for e in evidence['refresh'] if e['selection'] == 'query']
        if len(query) != 1 or [e['layer'] for e in layers] != list(range(16)):
            raise RuntimeError('Concurrent owner lacks one complete sixteen-layer Query selection')
        q = query[0]
        if result['mode'] == 'no-probe':
            if not q.get('no_probe') or any(q[k] != 0 for k in
                    ('private_forward_steps', 'state_restore_count', 'global_assembly_bytes')):
                raise RuntimeError('Candidate executed private/global work')
        elif q['private_forward_steps'] <= 0:
            raise RuntimeError('Sequential control skipped its private Query')
        restores = [e for e in evidence['events'] if e['event'] == 'pressure_prefix_restored']
        if len(restores) != int(result['phase'] != 'cold-c4'):
            raise RuntimeError('Concurrent owner did not restore its own frozen prefix')
        if result['terminal'].get('checkpoint_tokens') != result['prefix']:
            raise RuntimeError('Concurrent owner lost its frozen prefix checkpoint')
        save_new(result['directory']/'evidence.json', dict(evidence, layers=layers))


def native_batches(capture, results, directory):
    steps = [json.loads(line) for line in (capture/'model_step_latency.jsonl').open()]
    summaries = {}
    for phase in sorted({r['phase'] for r in results}):
        ids = {r['response']['response_id'] for r in results if r['phase'] == phase}
        selected = []
        for step in steps:
            geometry = [g for g in step['batch_geometry']
                        if any(g['request_id'].startswith(rid+'-') for rid in ids)]
            if geometry:
                selected.append(dict(sequence=step['sequence'], started_at=step['started_at'],
                    owners=len(geometry), prefill_owners=sum(g['prefill'] for g in geometry),
                    decode_owners=sum(not g['prefill'] for g in geometry), geometry=geometry))
        if not selected:
            raise RuntimeError('Missing native batch evidence')
        save_new(directory/phase/'native-batches.json', selected)
        summaries[phase] = dict(max_owners=max(s['owners'] for s in selected),
            max_prefill_owners=max(s['prefill_owners'] for s in selected),
            max_decode_owners=max(s['decode_owners'] for s in selected),
            batches_by_owners=dict(Counter(s['owners'] for s in selected)))
    return summaries


def trial(root, name):
    from benchmark_query_fps_prebuild_live import server_environment
    from run_layerwise_memory import normalize_cuda_alias
    from run_recall_comparison import gpu_lock, stop_owned, wait_health, termination_guard
    from benchmark_block_graph import gpu_processes
    from benchmark_memory import source, wait_terminal
    from graph_v2_protocol import release_owner
    plan = verify(root)
    spec = next(t for t in plan['trials'] if t['name'] == name)
    directory = root/name
    directory.mkdir()
    runtime = root/'runtime'
    profile, fixture = read(root/'profile.json'), read(root/'fixture.json')
    save_new(directory/'profile.json', profile)
    flags = ['--layerwise-query-diagnostic', 'sequential', '--layerwise-query-optimized',
             '--layerwise-query-native-batches']
    flags += ['--'+k.replace('_', '-') for k, v in fixture['execution_flags'].items() if v]
    if spec['mode'] == 'no-probe':
        flags += ['--layerwise-query-no-probe']
    command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve', '--config', str(directory/'profile.json'),
               '--run-dir', str(directory/'server'), *flags]
    env = normalize_cuda_alias(server_environment(runtime))
    capture = directory/'server/capture'
    groups = {g: [r for r in fixture['rows'] if r['id'].startswith(g+'_')] for g in GROUPS}
    inputs = {r['id']: source(r)['tokens'] for r in fixture['rows']}
    bindings = {b['id']: b for b in fixture['bindings']}
    sessions = {g: 'no-probe-c-'+uuid.uuid4().hex for g in GROUPS}
    phases = [('cold-c4', 4, [0])] + [('warm-c'+str(c), c, [1, 2, 3]) for c in spec['levels']]
    server = None
    results, summaries, received = [], {}, []
    with termination_guard(), gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
        if gpu_processes():
            raise RuntimeError('Another GPU owner is active')
        try:
            with (directory/'server.log').open('x') as log:
                server = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            save_new(directory/'process.json', dict(pid=server.pid, command=command))
            wait_health(server, profile)
            for phase, concurrency, queries in phases:
                (directory/phase).mkdir()
                save(directory/'progress.json', dict(phase=phase, completed=0))
                before = metrics(profile)
                phase_results = []
                for index in queries:
                    for begin in range(0, len(GROUPS), concurrency):
                        owners = GROUPS[begin:begin+concurrency]
                        barrier = Barrier(len(owners))
                        wave = []
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            for owner in owners:
                                row = groups[owner][index]
                                tokens, binding = inputs[row['id']], bindings[row['id']]
                                prefix = fixture['prefixes'][owner]
                                payload = dict(binding['sampling'], model=profile['served_model_name'],
                                    prompt=tokens, max_tokens=binding['max_tokens'],
                                    thinking_token_budget=binding['thinking_token_budget'], return_token_ids=True,
                                    session_id=sessions[owner], cache_salt=sessions[owner],
                                    kvmem_query_span=[row['query_begin'], row['query_end']],
                                    kvmem_replay_span=[prefix, row['query_end']], kvmem_reselect='force')
                                case = directory/phase/row['id']
                                case.mkdir()
                                save_new(case/'request.json', dict({k: v for k, v in payload.items() if k != 'prompt'},
                                    input_tokens=len(tokens), token_sha256=row['token_sha256']))
                                result = dict(id=row['id'], phase=phase, mode=spec['mode'],
                                              directory=case, prefix=prefix)
                                wave.append((result, pool.submit(stream_request, profile, payload, barrier)))
                            failures = []
                            for result, future in wave:
                                try:
                                    result['response'] = future.result()
                                    received.append(result)
                                except BaseException as error:
                                    save_new(result['directory']/'failure.json', dict(error=repr(error)))
                                    failures.append(repr(error))
                            if failures:
                                raise RuntimeError('Concurrent wave failed: '+repr(failures))
                        for result, _ in wave:
                            result['terminal'] = wait_terminal(capture, result['response']['response_id'])
                            if result['terminal'].get('checkpoint_tokens') != result['prefix']:
                                raise RuntimeError('Concurrent owner lost its frozen prefix checkpoint')
                            phase_results.append(result)
                        save(directory/'progress.json', dict(phase=phase, completed=len(phase_results)))
                        print(json.dumps(dict(trial=name, phase=phase, completed=len(phase_results),
                            wave_ttft=[r['response']['ttft_seconds'] for r, _ in wave])), flush=True)
                after = metrics(profile)
                response_rows = [r['response'] for r in phase_results]
                summary = phase_summary(response_rows, concurrency)
                summary['cases'] = [dict(id=r['id'], **{k: r['response'][k] for k in
                    ('started_at', 'seconds', 'ttft_seconds')}, output_tokens=len(r['response']['output_ids'])) for r in phase_results]
                summary['mtp_metrics'] = {k: after.get(k, 0)-before.get(k, 0) for k in set(before)|set(after)}
                summaries[phase] = summary
                for result in phase_results:
                    save_new(result['directory']/'response.json', result['response'])
                    save_new(result['directory']/'terminal.json', result['terminal'])
                save_new(directory/phase/'summary.json', summary)
                results.extend(phase_results)
            collect_evidence(capture, results)
            batches = native_batches(capture, results, directory)
            save_new(directory/'summary.json', dict(mode=spec['mode'], phases=summaries, native_batches=batches))
            releases = {g: release_owner(profile, capture, sessions[g], inputs[groups[g][-1]['id']][-1]) for g in GROUPS}
            save_new(directory/'release.json', releases)
        finally:
            stop_owned(server)
            for result in received:
                for key in ('response', 'terminal'):
                    path = result['directory']/(key+'.json')
                    if key in result and not path.exists():
                        save_new(path, result[key])
            save(directory/'cleanup.json', dict(remaining_gpu_processes=sorted(gpu_processes())))


def compare(root, left, right):
    checks = []
    for phase in ('cold-c4', 'warm-c2', 'warm-c4'):
        for path in sorted((root/left/phase).glob('*/response.json')):
            dirs = [path.parent, root/right/phase/path.parent.name]
            responses = [read(d/'response.json') for d in dirs]
            evidence = [read(d/'evidence.json') for d in dirs]
            selections = [[e['selected_starts'] for e in data['layers']] for data in evidence]
            pressure = [[[layer['selected_starts'] for layer in e['layers']]
                         for e in data['events'] if e['event'] == 'pressure_selected'] for data in evidence]
            a, b = [r['output_ids'] for r in responses]
            same = a == b
            mismatch = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))) if not same else None
            checks.append(dict(phase=phase, id=path.parent.name, same_output=same,
                same_text=responses[0]['text'] == responses[1]['text'],
                same_query_selection=selections[0] == selections[1], same_pressure=pressure[0] == pressure[1],
                layer_set_overlap_mean=statistics.mean(len(set(a)&set(b))/max(1, len(set(a)|set(b)))
                    for a, b in zip(*selections)), first_output_difference=mismatch,
                output_lengths=[len(a), len(b)]))
    return dict(left=left, right=right, checks=checks,
                output_matches=sum(c['same_output'] for c in checks), total=len(checks))


def summarize(root):
    plan = read(root/'plan.json')
    arms = {}
    for mode in ('sequential', 'no-probe'):
        runs = [read(root/t['name']/'summary.json') for t in plan['trials'] if t['mode'] == mode]
        arms[mode] = {}
        for phase in ('cold-c4', 'warm-c2', 'warm-c4'):
            phases = [r['phases'][phase] for r in runs]
            cases = [c for p in phases for c in p['cases']]
            wall = sum(p['wall_seconds'] for p in phases)
            arms[mode][phase] = dict(requests=len(cases), runs=len(runs),
                wall_mean_seconds=wall/len(runs), output_tokens=sum(c['output_tokens'] for c in cases),
                requests_per_second=len(cases)/wall,
                output_tokens_per_second=sum(c['output_tokens'] for c in cases)/wall,
                ttft_p50_seconds=percentile([c['ttft_seconds'] for c in cases], .5),
                ttft_p95_seconds=percentile([c['ttft_seconds'] for c in cases], .95),
                native_batch_peaks=[r['native_batches'][phase] for r in runs],
                client_peaks=[p['observed_client_peak'] for p in phases],
                mtp_metrics={k: sum(p['mtp_metrics'].get(k, 0) for p in phases) for k in phases[0]['mtp_metrics']})
    comparisons = {}
    for phase in ('cold-c4', 'warm-c2', 'warm-c4'):
        a, b = (arms[m][phase] for m in ('sequential', 'no-probe'))
        comparisons[phase] = {k+'_change': b[k]/a[k]-1 for k in
            ('output_tokens_per_second', 'requests_per_second', 'ttft_p50_seconds', 'ttft_p95_seconds')}
    save_new(root/'performance.json', dict(arms=arms, comparisons=comparisons,
        scope=plan['scope'], timing=plan['timing']))
    pairs = [('r1-sequential', 'r2-no-probe'), ('r4-sequential', 'r3-no-probe'),
             ('r1-sequential', 'r4-sequential'), ('r2-no-probe', 'r3-no-probe')]
    save_new(root/'consistency.json', dict(comparisons=[compare(root, a, b) for a, b in pairs],
        note='First two pairs are A/B; last two measure same-mode repeatability. '
             'Preserve every first response; asynchronous batches and nonzero-temperature sampling can differ.'))


def run(root):
    from benchmark_block_graph import gpu_processes
    from run_recall_comparison import stop_owned, termination_guard
    plan = verify(root)
    save_new(root/'controller.json', dict(pid=os.getpid(), at=time.time()))
    child = None
    with termination_guard():
        try:
            while gpu_processes():
                save(root/'status.json', dict(status='waiting_for_gpu', at=time.time()))
                time.sleep(30)
            for spec in plan['trials']:
                name = spec['name']
                save(root/'status.json', dict(status='running', trial=name, at=time.time()))
                with (root/(name+'.log')).open('x') as log:
                    child = subprocess.Popen([sys.executable, '-B', str(root/'runtime/scripts'/Path(__file__).name),
                        'trial', '--root', str(root), '--name', name], stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                if child.wait():
                    raise RuntimeError('Trial failed; preserve evidence: '+name)
                child = None
            summarize(root)
            save(root/'status.json', dict(status='completed', at=time.time()))
        except BaseException as error:
            save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise
        finally:
            stop_owned(child)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'trial', 'summarize'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root)
    elif args.action == 'run':
        run(root)
    elif args.action == 'trial':
        trial(root, args.name)
    else:
        summarize(root)

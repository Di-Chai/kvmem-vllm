"""Frozen twelve-case Key FPS-4 answer experiment, without graph propagation.

The failure-enriched cohort, inputs, seeds and sampling match the completed
16/8 overlap/no-graph experiment. Preparation never starts a model. Execution
uses a frozen runtime and refuses every previously attempted campaign.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace as NS
import uuid

import numpy as np

import benchmark_block_graph as lifecycle
from benchmark_ruler import save, sha
import qualify_graph_niah as client
import ruler_protocol as protocol
from test_ruler_retrieval_answers import coverage, events, read


ARM = 'key_fps_4'
OVERLAP_ARM = 'overlap_16_8'
CASE_FILES = ('request.json', 'response.json', 'result.json', 'evidence.json')
SCOPE = 'The same 12-case failure/control cohort; not complete-RULER accuracy.'
DEFAULT_ROOT = Path('/data/chaidi/kvmem-vllm-experiments/ruler-key-fps-4-20261023-r1')
SOURCE_ROOT = Path('/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/ruler-retrieval-answer-20261023-r1')
OVERLAP_ROOT = Path('/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/ruler-overlap-16-8-no-graph-20261023-r1')
FIXTURE_FIELDS = ('rows', 'input_directory', 'sampling', 'starts_in_thinking', 'upstream_path', 'tokenizer_sha256')
CASE_IDS = tuple('niah_multikey_2-'+number for number in
    ('0008', '0009', '0000', '0013', '0003', '0010', '0011', '0012', '0015', '0018', '0019', '0005'))


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_files(directory, proof):
    for name, digest in proof.items():
        require(sha(directory/name) == digest, f'Immutable experiment evidence changed: {name}')


def profile_difference(reference, candidate):
    """Only the Query scorer and listening port may differ from overlap alpha=0."""
    before, after = deepcopy(reference), deepcopy(candidate)
    changes = {}
    for key, left, right in (('port', before.pop('port'), after.pop('port')),
            ('kvmem.query_score_mode', before['kvmem'].pop('query_score_mode'),
                after['kvmem'].pop('query_score_mode'))):
        if left != right:
            changes[key] = dict(before=left, after=right)
    require(before == after, 'FPS profile differs beyond query_score_mode and port')
    return changes


def profile_for_fps(original):
    k = original['kvmem']
    require(k['query_score_mode'] == OVERLAP_ARM and k['graph_alpha'] == 0 and
            k['graph_mode'] == 'multi' and k['graph_scoring_version'] == 'v2' and
            k['graph_aggregation'] == 'sum' and k['graph_trace'] and
            k['guided_query_tokens'] == 0 and not k['query_history'],
            'Expected completed overlap/no-graph Query policy')
    require((k['select_budget'], k['gen_budget'], k['block_tokens'], k['sink_blocks'],
             k['recent_blocks']) == (36864, 16384, 32, 41, 0), 'Original KV/fixed-keep budget differs')
    require(original['draft_method'] == 'mtp' and original['draft_tokens'] == 4 and
            original['execution_mode'] == 'compiled_graph' and
            original['prefill_chunk_tokens'] == 2048 and original['max_num_seqs'] == 1 and
            original['max_num_batched_tokens'] == 16424, 'Original model execution geometry differs')
    require(k['pressure_policy'] == 'attention' and k['prefill_attention'] and
            not k['prefill_physical_coalescing'] and k['archive_backend'] == 'cpu',
            'Original pressure or archive policy differs')
    profile = deepcopy(original)
    profile['port'] = 18866
    profile['kvmem']['query_score_mode'] = ARM
    profile_difference(original, profile)
    return profile


def reference_proof(reference, fixture, arms, *, candidate=None):
    """Verify immutable completed answers and retain each original file hash."""
    require(read(reference/'status.json')['status'] == 'completed' and
            read(reference/'execution-audit.json')['status'] == 'passed',
            'Reference experiment is not completed and audited')
    previous = read(reference/'fixture.json')
    require(all(previous[name] == fixture[name] for name in FIXTURE_FIELDS),
            'Reference changes the original fixture, seed or sampling')
    names = ['fixture.json', 'status.json', 'execution-audit.json']
    if (reference/'runtime-proof.json').exists():
        verify_files(reference/'runtime', read(reference/'runtime-proof.json')['files'])
        names.append('runtime-proof.json')
    if (reference/'prepared-proof.json').exists():
        verify_files(reference, read(reference/'prepared-proof.json'))
        names.append('prepared-proof.json')
    if candidate is not None:
        require(tuple(arms) == (OVERLAP_ARM,), 'Expected only the completed overlap/no-graph reference')
        before = read(reference/OVERLAP_ARM/'profile.json')
        require(candidate == profile_for_fps(before), 'FPS candidate changed the paired profile')
        names.append('target-spans.json')
    proof = {name: sha(reference/name) for name in names}
    snapshots = {}
    for arm in arms:
        cleanup = read(reference/arm/'cleanup.json')
        require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'],
                'Reference experiment has not released the GPU')
        scores = read(reference/arm/'scores.json')
        require(scores['cases'] == len(fixture['rows']), 'Reference scores do not cover the frozen cohort')
        for name in ('profile.json', 'scores.json', 'cleanup.json'):
            relative = arm+'/'+name; proof[relative] = sha(reference/relative)
        snapshots[arm] = {}
        for row in fixture['rows']:
            case = reference/arm/'cases'/row['id']
            manifest = read(case/'hashes.json')
            require(set(CASE_FILES).issubset(manifest), 'Reference lacks complete answer evidence')
            verify_files(case, manifest)
            proof[str((case/'hashes.json').relative_to(reference))] = sha(case/'hashes.json')
            proof.update({str((case/name).relative_to(reference)):digest for name,digest in manifest.items()})
            snapshots[arm][row['id']] = dict(root=str(case),
                response=read(case/'response.json'), result=read(case/'result.json'),
                sha256={name:manifest[name] for name in ('response.json', 'result.json')})
    result = dict(root=str(reference), files=proof, identical_fixture_fields=list(FIXTURE_FIELDS))
    if candidate is not None:
        result['profile_difference'] = profile_difference(before, candidate)
    return result, snapshots


def prepare(root, source, overlap_reference):
    require(not root.exists(), 'Use a new FPS experiment root')
    fixture = read(source/'fixture.json')
    require(tuple(row['id'] for row in fixture['rows']) == CASE_IDS, 'Expected exactly the original twelve cases in order')
    require(fixture['sampling']['max_tokens'] == 8192, 'Original output quota changed')
    profile = profile_for_fps(read(overlap_reference/OVERLAP_ARM/'profile.json'))
    controls, saved_controls = reference_proof(source, fixture, ('mean_k', 'token_sum'))
    paired, saved_overlap = reference_proof(overlap_reference, fixture, (OVERLAP_ARM,), candidate=profile)
    for row in fixture['rows']:
        path = Path(fixture['input_directory'])/(row['id']+'.i32')
        require(sha(path) == row['served_sha256'] and path.stat().st_size == row['served_input_tokens']*4,
                'Original complete input changed')
    repo = Path(__file__).resolve().parents[1]
    from vllm_kvmem.serve import build_launch
    build_launch(profile, repo, root/ARM/'preflight-only')
    targets = read(overlap_reference/'target-spans.json')
    require(set(targets) == set(CASE_IDS), 'Missing offline-only reference spans')
    root.mkdir(parents=True)
    runtime = root/'runtime'
    for name in ('src/vllm_kvmem', 'scripts', 'csrc', 'third_party/qw3'):
        shutil.copytree(repo/name, runtime/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'setup.py'):
        shutil.copy2(repo/name, runtime/name)
    runtime_proof = {str(p.relative_to(runtime)):sha(p) for p in runtime.rglob('*') if p.is_file()}
    save(root/'runtime-proof.json', dict(files=runtime_proof))
    old_runtime = read(overlap_reference/'runtime-proof.json')['files']
    paired['runtime_changed_files'] = {name:dict(before=old_runtime.get(name), after=runtime_proof.get(name))
        for name in sorted(set(old_runtime)|set(runtime_proof)) if old_runtime.get(name) != runtime_proof.get(name)}
    fixture.update(answer_reference=str(source), answer_reference_sha256=sha(source/'fixture.json'),
        overlap_reference=str(overlap_reference), graph_alpha=0., query_score_mode=ARM,
        purpose=SCOPE,
        difference='Public Query uses four raw-Key farthest-first representatives per original 32-token block; '
            'representative maximum precedes block softmax; graph alpha=0; no post-picker neighbor expansion.',
        comparison_limit='The overlap/no-graph profile differs only in Query scorer and service port. '
            'All controls are previous real model answers, preserved without rerunning or inferring new answers. '
            'Separate frozen runtimes and cache-building costs are recorded; this twelve-case '
            'failure-enriched cohort is not complete-RULER accuracy.')
    save(root/'fixture.json', fixture)
    save(root/'reference-proof.json', controls)
    save(root/'overlap-reference-proof.json', paired)
    save(root/'reference-answers.json', dict(**saved_controls, **saved_overlap))
    save(root/'target-spans.json', targets)
    save(root/ARM/'profile.json', profile)
    prepared_files = ['fixture.json', 'reference-proof.json', 'overlap-reference-proof.json',
        'reference-answers.json', 'target-spans.json', ARM+'/profile.json', 'runtime-proof.json']
    save(root/'prepared-proof.json', {name:sha(root/name) for name in prepared_files})
    save(root/'status.json', dict(status='prepared', cases=12, arms=1, arm=ARM, graph_alpha=0., at=time.time()))


def audit_selection(query, graph, inputs, profile):
    """Recompute transport and native picking from exported, answer-blind seeds."""
    import analyze_block_graph as offline
    from vllm_kvmem.graph_scoring import score_graph
    alpha = profile['kvmem']['graph_alpha']
    require(type(alpha) in (int, float) and alpha == 0, 'FPS experiment requires graph alpha=0')
    stats = query.get('query_key_fps_scoring', {})
    require(query.get('query_scoring_method') == ARM and stats.get('method') == ARM and
            (stats.get('representatives_per_block'), stats.get('block_tokens')) == (4, 32),
            'Four-representative raw-Key FPS public Query scoring was not executed')
    require(stats.get('selection') == 'raw_key_farthest_first' and
            stats.get('normalization') == 'block softmax after per-layer/per-Query/per-head representative max',
            'FPS representative selection or reduction order differs')
    for record in (graph, inputs):
        require(record.get('seed_source') == ARM and record.get('scoring_version') == 'v2' and
                record.get('mode') == 'multi' and record.get('aggregation') == 'sum' and
                record.get('alpha') == alpha, 'Requested V2 SUM propagation differs')
    if alpha == 0:
        require(graph.get('propagation_applied') is False and graph.get('identity') is True,
                'Alpha-zero Query must explicitly report no graph propagation')
    require(abs(math.fsum(inputs['raw_scores'])-query['query_score_tokens']) < 1e-3,
            'Query seed mass differs from scored Query rows')
    ranges = query['selected_history_ranges']
    selected = {begin for begin,_ in ranges}
    original_counts = dict(zip(inputs['starts'], inputs['counts']))
    require(len(selected) == len(ranges) and
            all(begin % 32 == 0 and 0 < count <= 32 for begin,count in ranges) and
            all(original_counts.get(begin) == count for begin,count in ranges) and
            inputs['budget'] == profile['kvmem']['select_budget'] and
            sum(count for _,count in ranges) <= profile['kvmem']['select_budget'],
            'Native block geometry or selected budget differs')
    sink = profile['kvmem']['sink_blocks']*32
    require(coverage(ranges, (0, sink)), 'Required sink content was lost')
    config, snapshot = offline._validate_record(inputs)
    memory, handles = offline._native_history(config, inputs['starts'], inputs['counts'])
    try:
        offline._validate_native_masks(inputs, memory, handles)
        mandatory = tuple(handles[start].slot for start in inputs['mandatory_starts'])
        with score_graph(inputs['raw_scores'], inputs['starts'], inputs['transport_mask'], snapshot,
                mode='multi', scoring_version='v2', aggregation='sum', alpha=alpha,
                cross_message_weight=inputs['cross_message_weight'], block_tokens=32) as propagated:
            if alpha == 0:
                require(list(propagated.scores) == list(inputs['raw_scores']) and
                        propagated.diagnostics['identity'] is True,
                        'Alpha-zero graph changed the original FPS seed scores')
            slots = memory.preview_selection({handles[start].slot:score
                for start,score in zip(inputs['starts'], propagated.scores)}, mandatory,
                action=False, prefill_scores=None, max_tokens=inputs['budget'])
            picked = {start for start,handle in handles.items() if handle.slot in slots}
            require(selected == picked, 'Observed selection differs from native SUM picker replay')
            diagnostics = dict(propagated.diagnostics)
    finally:
        memory.close()
    return dict(native_picker_replay_equal=True, selected_tokens=sum(n for _,n in ranges),
        sink_tokens=sink, no_post_picker_neighbors=True, key_fps_stats=stats, propagation=diagnostics,
        graph_alpha=alpha, propagation_applied=not diagnostics['identity'])


def finish_case(root, fixture, row, profile, tokens, session, response, capture):
    from tokenizers import Tokenizer
    terminal = client.wait_retired(capture, response['response_id'])
    require(terminal['history_end'] == len(tokens)+len(response['output_ids'])-1 and
            not terminal.get('checkpoint_tokens'), 'Accepted input/output extent or retirement differs')
    released = client.wait_owner_released(capture, session, response, terminal)
    bounded = events(capture, 'bounded_refresh.jsonl', response)
    queries = [e for e in bounded if e['selection'] == 'query']
    pressure = [e for e in bounded if e['selection'] == 'pressure']
    require(len(queries) == 1 and queries[0]['query_span'] == row['served_query'] and pressure and
            all(not e['guided_query_tokens'] for e in bounded), 'Public Query or unguided pressure differs')
    graph = events(capture, 'block_graph.jsonl', response)
    query_graph = [e for e in graph if e['phase'] == 'query']
    pressure_graph = [e for e in graph if e['phase'] == 'pressure']
    require(len(query_graph) == 1 and len(pressure_graph) == len(pressure) and
            all(e.get('seed_source') == 'prefill' and e.get('pressure_policy') == 'attention' and
                not e.get('propagation_applied') for e in pressure_graph), 'Prefill cumulative attention changed')
    inputs = [e for e in events(capture, 'block_graph_inputs.jsonl', response) if e['phase'] == 'query']
    require(len(inputs) == 1, 'Expected exactly one complete exported public Query seed/graph record')
    selection_audit = audit_selection(queries[0], query_graph[0], inputs[0], profile)
    action = events(capture, 'action_attention.jsonl', response)
    tokenizer = Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    expected = client.q.expected_output_samples(NS(convert_tokens_to_ids=tokenizer.token_to_id),
        profile, tokens, response['output_ids'])
    require(len(action) == 1 and action[0]['samples'] == expected,
            'Private/rejected/thinking samples contaminated attention')
    gold = read(root/'target-spans.json')[row['id']]
    scorer = protocol.load_scorer(Path(fixture['upstream_path']))
    answer = protocol.final_answer(response['text'], starts_in_thinking=fixture['starts_in_thinking'])
    config = scorer[0][protocol.TASK_FAMILIES[row['task']]]
    score = config['metric_fn']([scorer[1](answer['text'], config)], [row['outputs']])
    ranges = queries[0]['selected_history_ranges']
    result = dict(id=row['id'], arm=ARM, official_case_score=score, answer=answer['text'],
        thinking_complete=answer['thinking_complete'], finish_reason=response['finish_reason'],
        completion_tokens=len(response['output_ids']), seconds=response['seconds'],
        value_covered=coverage(ranges, gold['token_span']),
        record_covered=coverage(ranges, [gold['key_token_span'][0],gold['token_span'][1]]),
        target_value_span=gold['token_span'], pressure_count=len(pressure), selection_audit=selection_audit)
    return result, dict(bounded=bounded, graph=graph, query_inputs=inputs, action=action,
        terminal=terminal, owner_release=released)


def verify_campaign(root, fixture):
    verify_files(root/'runtime', read(root/'runtime-proof.json')['files'])
    verify_files(root, read(root/'prepared-proof.json'))
    require(tuple(row['id'] for row in fixture['rows']) == CASE_IDS and fixture['graph_alpha'] == 0 and
            fixture['query_score_mode'] == ARM, 'Frozen cohort or FPS policy changed')
    profile = read(root/ARM/'profile.json')
    controls = read(root/'reference-proof.json'); paired = read(root/'overlap-reference-proof.json')
    require(controls['root'] == fixture['answer_reference'] and paired['root'] == fixture['overlap_reference'],
            'Reference binding changed')
    verify_files(Path(controls['root']), controls['files']); verify_files(Path(paired['root']), paired['files'])
    current_controls, old_answers = reference_proof(Path(controls['root']), fixture, ('mean_k', 'token_sum'))
    current_paired, overlap_answers = reference_proof(Path(paired['root']), fixture, (OVERLAP_ARM,), candidate=profile)
    require(current_controls == controls and current_paired['files'] == paired['files'] and
            current_paired['profile_difference'] == paired['profile_difference'], 'Reference audit changed')
    require(read(root/'reference-answers.json') == dict(**old_answers, **overlap_answers),
            'Preserved reference answers differ from their original immutable files')
    hashes = {}
    scorer = protocol.load_scorer(Path(fixture['upstream_path']))
    for row in fixture['rows']:
        case = root/ARM/'cases'/row['id']
        manifest = read(case/'hashes.json')
        require(set((*CASE_FILES, 'query-inputs.jsonl')).issubset(manifest), 'FPS case evidence incomplete')
        verify_files(case, manifest)
        request = read(case/'request.json')
        require(request['sampling'] == dict(fixture['sampling'], seed=row['seed']) and
                request['query_span'] == row['served_query'] and request['input_sha256'] == row['served_sha256'],
                'Request changed original sampling, Query or input')
        require(sha(Path(fixture['input_directory'])/(row['id']+'.i32')) == row['served_sha256'],
                'Original fixture input changed')
        response = read(case/'response.json'); result = read(case/'result.json')
        answer = protocol.final_answer(response['text'], starts_in_thinking=fixture['starts_in_thinking'])
        scoring = scorer[0][protocol.TASK_FAMILIES[row['task']]]
        expected = scoring['metric_fn']([scorer[1](answer['text'], scoring)], [row['outputs']])
        require(result['official_case_score'] == expected and result['answer'] == answer['text'],
                'Saved result differs from official score recomputation')
        require(request['profile_sha256'] == sha(root/ARM/'profile.json'), 'Request used a different profile')
        evidence = read(case/'evidence.json')
        queries = [e for e in evidence['bounded'] if e['selection'] == 'query']
        graphs = [e for e in evidence['graph'] if e['phase'] == 'query']
        require(len(queries) == len(graphs) == len(evidence['query_inputs']) == 1,
                'Saved Query execution evidence incomplete')
        replay = audit_selection(queries[0], graphs[0], evidence['query_inputs'][0], profile)
        recorded = deepcopy(result['selection_audit'])
        # CPU audit duration is measured again; all semantic evidence must match.
        replay['propagation'].pop('scoring_seconds', None)
        recorded['propagation'].pop('scoring_seconds', None)
        require(replay == recorded, 'Final native picker audit changed')
        hashes.update({str((case/name).relative_to(root)):digest for name,digest in manifest.items()})
    cleanup = read(root/ARM/'cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'], 'GPU cleanup incomplete')
    return dict(status='passed', cases=12, unchanged_frozen_runtime=True, case_hashes=hashes,
        original_inputs_seeds_sampling_verified=True, native_picker_replay_verified=True,
        official_scores_recomputed=True, no_post_picker_neighbors=True,
        pressure_cumulative_attention=True, guided=False, graph_alpha=0., propagation_applied=False,
        query_score_mode=ARM, gpu_cleanup_verified=True, reference_answers_unchanged=True,
        overlap_reference_unchanged=True, overlap_profile_difference=paired['profile_difference'], scope=SCOPE)


def run(root):
    require(Path(__file__).resolve() == root/'runtime/scripts'/Path(__file__).name,
            'Run the frozen runtime/scripts/test_ruler_fps_answers.py controller')
    verify_files(root/'runtime', read(root/'runtime-proof.json')['files'])
    verify_files(root, read(root/'prepared-proof.json'))
    for name in ('reference-proof.json', 'overlap-reference-proof.json'):
        reference = read(root/name)
        verify_files(Path(reference['root']), reference['files'])
    with (root/'attempt.json').open('x') as marker:
        json.dump(dict(pid=os.getpid(), at=time.time(), policy='No automatic retries'), marker)
    fixture = read(root/'fixture.json'); base = root/ARM; profile = read(base/'profile.json')
    runtime = root/'runtime'; capture = base/'server/capture'; results = []; server = None
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_', 'VLLM_', 'DEEPSEEK_'))}
    env.update(PYTHONPATH=str(runtime/'src'), PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
        MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', VLLM_LOG_STATS_INTERVAL='5')
    for toolkit in ('/usr/local/cuda-13', '/usr/local/cuda'):
        if (Path(toolkit)/'bin/nvcc').exists():
            env.update(CUDA_HOME=toolkit, PATH=toolkit+'/bin'+os.pathsep+env.get('PATH','')); break
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a+') as lock, lifecycle.cleanup_on_termination():
            deadline = time.monotonic()+7200
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB); break
                except BlockingIOError:
                    require(time.monotonic() < deadline, 'GPU coordination lock timeout')
                    time.sleep(5)
            require(not lifecycle.gpu_processes(), 'Foreign GPU owner; do not stop it')
            try:
                command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve', '--config',
                    str(base/'profile.json'), '--run-dir', str(base/'server')]
                with (base/'server.log').open('x') as log:
                    server = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                save(base/'server-process.json', dict(pid=server.pid, command=command))
                save(root/'status.json', dict(status='loading', arm=ARM, pid=server.pid, at=time.time()))
                startup = time.monotonic()+1200
                opener = client.urllib.request.build_opener(client.urllib.request.ProxyHandler({}))
                while True:
                    require(server.poll() is None, 'FPS answer model exited')
                    try:
                        with opener.open(f"http://127.0.0.1:{profile['port']}/health", timeout=1) as response:
                            if response.status == 200: break
                    except OSError:
                        pass
                    require(time.monotonic() < startup, 'FPS answer model startup timeout')
                    time.sleep(1)
                for row in fixture['rows']:
                    case = base/'cases'/row['id']; case.mkdir(parents=True, exist_ok=False)
                    save(root/'status.json', dict(status='testing', arm=ARM, case=row['id'],
                        completed=len(results), expected=12, at=time.time()))
                    source = Path(fixture['input_directory'])/(row['id']+'.i32')
                    require(sha(source) == row['served_sha256'], 'Original input changed')
                    tokens = np.fromfile(source, dtype='<i4').tolist()
                    sampling = dict(fixture['sampling'], seed=row['seed'])
                    session = 'fps-answer-'+uuid.uuid4().hex
                    save(case/'request.json', dict(sampling=sampling, session_id=session,
                        input_sha256=row['served_sha256'], query_span=row['served_query'],
                        profile_sha256=sha(base/'profile.json')))
                    response = client.stream(profile, session, tokens, row['served_query'], sampling, 3600)
                    save(case/'response.json', response)
                    result,evidence = finish_case(root, fixture, row, profile, tokens, session, response, capture)
                    save(case/'evidence.json', evidence); save(case/'result.json', result)
                    # The server opens this diagnostic log for each append.
                    # After independent-owner retirement, retain its entire log
                    # per case so twelve complete graph dumps fit the log quota.
                    (capture/'block_graph_inputs.jsonl').rename(case/'query-inputs.jsonl')
                    save(case/'hashes.json', {name:sha(case/name) for name in (*CASE_FILES, 'query-inputs.jsonl')})
                    results.append(result)
                    save(base/'progress.json', dict(completed=len(results), expected=12, results=results, at=time.time()))
                    print(json.dumps(result), flush=True)
            finally:
                remaining = lifecycle.stop_owned_groups((server,))
                gpu = sorted(lifecycle.gpu_processes())
                save(base/'cleanup.json', dict(remaining_owned_groups=remaining, gpu_processes=gpu))
                require(not remaining and not gpu, 'FPS model GPU cleanup incomplete')
        audit = verify_campaign(root, fixture)
        save(root/'execution-audit.json', audit)
        scores = dict(cases=len(results), mean_case_score=sum(r['official_case_score'] for r in results)/len(results),
            full_value_coverage=sum(r['value_covered'] for r in results), results=results, scope=SCOPE)
        save(base/'scores.json', scores)
        references = {arm:read(Path(fixture['answer_reference'])/arm/'scores.json') for arm in ('mean_k', 'token_sum')}
        references[OVERLAP_ARM] = read(Path(fixture['overlap_reference'])/OVERLAP_ARM/'scores.json')
        save(root/'summary.json', dict(key_fps=scores, previous_answers=references,
            graph_alpha=0., comparison_limit=fixture['comparison_limit'], scope=SCOPE))
        save(base/'status.json', dict(status='completed', cases=12, at=time.time()))
        save(root/'status.json', dict(status='completed', cases=12, arms=1, arm=ARM,
            graph_alpha=fixture['graph_alpha'], at=time.time()))
    except BaseException as error:
        save(root/'failure.json', dict(error=repr(error), completed=len(results), at=time.time()))
        save(root/'status.json', dict(status='failed', completed=len(results), error=repr(error), at=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--source', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--overlap-reference', type=Path, default=OVERLAP_ROOT)
    arguments = parser.parse_args()
    if arguments.action == 'prepare':
        prepare(arguments.root.resolve(), arguments.source.resolve(), arguments.overlap_reference.resolve())
    else:
        run(arguments.root.resolve())

"""Run only 16/8 overlap MeanK plus V2 multi SUM on the frozen 12-case cohort.

This is the prior failure-enriched diagnostic cohort, not full RULER accuracy.
Preparation never starts a model. Execution must use this script's frozen copy;
an attempted campaign cannot be rerun or overwrite a completed answer.
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


ARM = 'overlap_16_8'
DEFAULT_ROOT = Path('/data/chaidi/kvmem-vllm-experiments/ruler-overlap-16-8-20261023-r1')
DEFAULT_NO_GRAPH_ROOT = Path('/data/chaidi/kvmem-vllm-experiments/ruler-overlap-16-8-no-graph-20261023-r1')
SOURCE_ROOT = Path('/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/ruler-retrieval-answer-20261023-r1')
SCOPE = 'The same 12-case failure/control cohort; not complete-RULER accuracy.'
CASE_FILES = ('request.json', 'response.json', 'result.json', 'evidence.json')


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_alpha(value):
    require(type(value) in (int, float) and value in (0, .2), 'Only graph alpha 0 or 0.2 is qualified')
    return float(value)


def profile_for_overlap(original, graph_alpha=.2):
    graph_alpha = validate_alpha(graph_alpha)
    profile = deepcopy(original)
    profile['port'] = 18865 if graph_alpha == 0 else 18864
    profile['kvmem'].update(query_score_mode=ARM, query_window_cache_bytes=2147483648,
        graph_mode='multi', graph_scoring_version='v2', graph_aggregation='sum',
        graph_alpha=graph_alpha, graph_trace=True, guided_query_tokens=0, query_history=False)
    k = profile['kvmem']
    require((k['select_budget'], k['gen_budget'], k['block_tokens'], k['sink_blocks'],
             k['recent_blocks']) == (36864, 16384, 32, 41, 0), 'Original KV/fixed-keep budget differs')
    require(profile['draft_method'] == 'mtp' and profile['draft_tokens'] == 4 and
            profile['prefill_chunk_tokens'] == 2048 and profile['max_num_seqs'] == 1 and
            profile['max_num_batched_tokens'] == 16424, 'Original model execution geometry differs')
    require(k['pressure_policy'] == 'attention' and k['prefill_attention'] and
            not k['prefill_physical_coalescing'] and k['archive_backend'] == 'cpu',
            'Original pressure or archive policy differs')
    return profile


def verify_files(directory, proof):
    for name, digest in proof.items():
        require(sha(directory/name) == digest, f'Immutable experiment evidence changed: {name}')


def profile_difference(reference, candidate):
    """The paired overlap ablation may change only alpha and the service port."""
    before, after = deepcopy(reference), deepcopy(candidate)
    changes = {}
    for key, left, right in (('port', reference['port'], candidate['port']),
            ('kvmem.graph_alpha', reference['kvmem']['graph_alpha'], candidate['kvmem']['graph_alpha'])):
        if left != right:
            changes[key] = dict(before=left, after=right)
    before.pop('port'); after.pop('port')
    before['kvmem'].pop('graph_alpha'); after['kvmem'].pop('graph_alpha')
    require(before == after, 'Paired overlap profiles differ beyond graph_alpha and port')
    return changes


def overlap_reference_proof(reference, fixture, profile):
    require(read(reference/'status.json')['status'] == 'completed' and
            read(reference/'execution-audit.json')['status'] == 'passed',
            'Prior overlap experiment is not completed and audited')
    previous = read(reference/'fixture.json')
    fields = ('rows', 'input_directory', 'sampling', 'starts_in_thinking', 'upstream_path', 'tokenizer_sha256')
    require(all(previous[name] == fixture[name] for name in fields),
            'Overlap reference changes the original fixture, seed or sampling')
    previous_profile = read(reference/ARM/'profile.json')
    require(previous_profile['kvmem']['query_score_mode'] == ARM and
            previous_profile['kvmem']['graph_alpha'] == .2, 'Expected the completed alpha=0.2 overlap reference')
    delta = profile_difference(previous_profile, profile)
    cleanup = read(reference/ARM/'cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'],
            'Prior overlap experiment has not released the GPU')
    proof = {name:sha(reference/name) for name in ('fixture.json', 'status.json', 'execution-audit.json',
        'runtime-proof.json', 'prepared-proof.json', 'target-spans.json', ARM+'/profile.json',
        ARM+'/scores.json', ARM+'/cleanup.json')}
    verify_files(reference/'runtime', read(reference/'runtime-proof.json')['files'])
    verify_files(reference, read(reference/'prepared-proof.json'))
    for row in fixture['rows']:
        case = reference/ARM/'cases'/row['id']
        manifest = read(case/'hashes.json')
        require(set(CASE_FILES).issubset(manifest), 'Overlap reference lacks complete answer evidence')
        verify_files(case, manifest)
        proof[str((case/'hashes.json').relative_to(reference))] = sha(case/'hashes.json')
        proof.update({str((case/name).relative_to(reference)):digest for name,digest in manifest.items()})
    return dict(root=str(reference), files=proof, profile_difference=delta,
        identical_fixture_fields=list(fields), reference_alpha=.2, candidate_alpha=profile['kvmem']['graph_alpha'])


def prepare(root, source, graph_alpha=.2, overlap_reference=None):
    graph_alpha = validate_alpha(graph_alpha)
    require(not root.exists(), 'Use a new overlap experiment root')
    fixture = read(source/'fixture.json')
    require(len(fixture['rows']) == 12 and len({r['id'] for r in fixture['rows']}) == 12,
            'Expected exactly the previous twelve cases')
    require(read(source/'status.json')['status'] == 'completed' and
            read(source/'execution-audit.json')['status'] == 'passed', 'Prior answer comparison is not audited')
    require(fixture['sampling']['max_tokens'] == 8192, 'Original output quota changed')
    reference_proof = {name:sha(source/name) for name in
        ('fixture.json', 'execution-audit.json', 'mean_k/profile.json', 'mean_k/scores.json',
         'token_sum/scores.json', 'mean_k/cleanup.json', 'token_sum/cleanup.json')}
    for arm in ('mean_k', 'token_sum'):
        cleanup = read(source/arm/'cleanup.json')
        require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'],
                'Prior answer arm has not released the GPU')
        for row in fixture['rows']:
            case = source/arm/'cases'/row['id']
            hashes = read(case/'hashes.json')
            verify_files(case, hashes)
            reference_proof.update({str((case/name).relative_to(source)):digest
                                    for name,digest in hashes.items()})
    for row in fixture['rows']:
        path = Path(fixture['input_directory'])/(row['id']+'.i32')
        require(sha(path) == row['served_sha256'] and path.stat().st_size//4 == row['served_input_tokens'],
                'Original complete input changed')
    repo = Path(__file__).resolve().parents[1]
    profile = profile_for_overlap(read((overlap_reference/ARM/'profile.json') if overlap_reference else
        (source/'mean_k/profile.json')), graph_alpha)
    paired_proof = overlap_reference_proof(overlap_reference, fixture, profile) if overlap_reference else None
    from vllm_kvmem.serve import build_launch
    build_launch(profile, repo, root/ARM/'preflight-only')
    target_source = repo/'artifacts/ruler-failures-20261022/analysis.json'
    targets = {r['id']:r['references'][0]['spans'][0] for r in read(target_source)['rows']
               if r['id'] in {row['id'] for row in fixture['rows']}}
    require(len(targets) == 12, 'Missing offline-only reference spans')
    root.mkdir(parents=True)
    runtime = root/'runtime'
    for name in ('src/vllm_kvmem', 'scripts', 'csrc', 'third_party/qw3'):
        shutil.copytree(repo/name, runtime/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'setup.py'):
        shutil.copy2(repo/name, runtime/name)
    save(root/'runtime-proof.json', dict(files={str(p.relative_to(runtime)):sha(p)
        for p in runtime.rglob('*') if p.is_file()}))
    fixture.update(answer_reference=str(source), answer_reference_sha256=sha(source/'fixture.json'),
        graph_alpha=graph_alpha, overlap_reference=str(overlap_reference) if overlap_reference else None,
        purpose=SCOPE, difference='Only new arm: 16/8 window MeanK, window-to-block maximum before '
        f'block softmax, then V2 multi SUM alpha={graph_alpha:g}; no post-picker neighbor expansion.',
        comparison_limit=('Paired overlap profiles differ only in graph_alpha and service port; '
            'the previous alpha=0.2 overlap answers are immutable and are not rerun.' if paired_proof else
            'Existing MeanK/token_sum controls used alpha=0; comparisons do not isolate '
            'window scoring from the additional requested graph propagation.' if graph_alpha else
            'Existing MeanK/token_sum controls also used alpha=0; this is a failure-enriched cohort '
            'with separate frozen runtimes, not complete-RULER accuracy.'))
    save(root/'fixture.json', fixture)
    save(root/'reference-proof.json', dict(root=str(source), files=reference_proof,
        target_source=str(target_source), target_source_sha256=sha(target_source)))
    save(root/'target-spans.json', targets)
    save(root/ARM/'profile.json', profile)
    prepared_files = ['fixture.json', 'reference-proof.json', 'target-spans.json', ARM+'/profile.json', 'runtime-proof.json']
    if paired_proof:
        old_runtime = read(overlap_reference/'runtime-proof.json')['files']
        new_runtime = read(root/'runtime-proof.json')['files']
        paired_proof['runtime_changed_files'] = {name:dict(before=old_runtime.get(name), after=new_runtime.get(name))
            for name in sorted(set(old_runtime)|set(new_runtime)) if old_runtime.get(name) != new_runtime.get(name)}
        save(root/'overlap-reference-proof.json', paired_proof)
        prepared_files.append('overlap-reference-proof.json')
    save(root/'prepared-proof.json', {name:sha(root/name) for name in prepared_files})
    save(root/'status.json', dict(status='prepared', cases=12, arms=1, arm=ARM, graph_alpha=graph_alpha, at=time.time()))


def audit_selection(query, graph, inputs, profile):
    """Recompute transport and native picking from exported, answer-blind seeds."""
    import analyze_block_graph as offline
    from vllm_kvmem.graph_scoring import score_graph
    alpha = validate_alpha(profile['kvmem']['graph_alpha'])
    stats = query.get('query_window_scoring', {})
    require(query.get('query_scoring_method') == ARM and stats.get('method') == ARM and
            (stats.get('window_tokens'), stats.get('window_stride'), stats.get('block_tokens')) == (16, 8, 32),
            '16/8 public Query scoring was not executed')
    require(0 < stats['valid_windows'] <= stats['geometry_windows'] and stats['fallback_blocks'] >= 0 and
            stats['normalization'] == 'block softmax after per-layer/per-Query/per-head window max',
            'Window statistics or reduction order differs')
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
    require(len(selected) == len(ranges) and
            all(begin % 32 == 0 and 0 < count <= 32 for begin,count in ranges) and
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
                        'Alpha-zero graph changed the original overlap seed scores')
            slots = memory.preview_selection({handles[start].slot:score
                for start,score in zip(inputs['starts'], propagated.scores)}, mandatory,
                action=False, prefill_scores=None, max_tokens=inputs['budget'])
            picked = {start for start,handle in handles.items() if handle.slot in slots}
            require(selected == picked, 'Observed selection differs from native SUM picker replay')
            diagnostics = dict(propagated.diagnostics)
    finally:
        memory.close()
    return dict(native_picker_replay_equal=True, selected_tokens=sum(n for _,n in ranges),
        sink_tokens=sink, no_post_picker_neighbors=True, window_stats=stats, propagation=diagnostics,
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
    reference = read(root/'reference-proof.json')
    verify_files(Path(reference['root']), reference['files'])
    paired = None
    if fixture.get('overlap_reference'):
        paired = read(root/'overlap-reference-proof.json')
        require(paired['root'] == fixture['overlap_reference'], 'Overlap reference binding changed')
        verify_files(Path(paired['root']), paired['files'])
        current = overlap_reference_proof(Path(paired['root']), fixture, read(root/ARM/'profile.json'))
        require(current['profile_difference'] == paired['profile_difference'] and
                current['files'] == paired['files'], 'Overlap reference audit changed')
    hashes = {}
    scorer = protocol.load_scorer(Path(fixture['upstream_path']))
    for row in fixture['rows']:
        case = root/ARM/'cases'/row['id']
        manifest = read(case/'hashes.json'); verify_files(case, manifest)
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
        require(result['selection_audit']['graph_alpha'] == fixture['graph_alpha'] and
                (fixture['graph_alpha'] != 0 or result['selection_audit']['propagation_applied'] is False),
                'Case Query propagation differs from the prepared arm')
        hashes.update({str((case/name).relative_to(root)):digest for name,digest in manifest.items()})
    cleanup = read(root/ARM/'cleanup.json')
    require(not cleanup['remaining_owned_groups'] and not cleanup['gpu_processes'], 'GPU cleanup incomplete')
    return dict(status='passed', cases=12, unchanged_frozen_runtime=True, case_hashes=hashes,
        original_inputs_seeds_sampling_verified=True, native_picker_replay_verified=True,
        official_scores_recomputed=True,
        no_post_picker_neighbors=True, pressure_cumulative_attention=True, guided=False,
        graph_alpha=fixture['graph_alpha'], gpu_cleanup_verified=True, reference_answers_unchanged=True,
        overlap_reference_unchanged=paired is not None,
        overlap_profile_difference=paired['profile_difference'] if paired else None, scope=SCOPE)


def run(root):
    require(Path(__file__).resolve() == root/'runtime/scripts'/Path(__file__).name,
            'Run the frozen runtime/scripts/test_ruler_overlap_answers.py controller')
    verify_files(root/'runtime', read(root/'runtime-proof.json')['files'])
    verify_files(root, read(root/'prepared-proof.json'))
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
                    require(server.poll() is None, 'Overlap answer model exited')
                    try:
                        with opener.open(f"http://127.0.0.1:{profile['port']}/health", timeout=1) as response:
                            if response.status == 200: break
                    except OSError:
                        pass
                    require(time.monotonic() < startup, 'Overlap answer model startup timeout')
                    time.sleep(1)
                for row in fixture['rows']:
                    case = base/'cases'/row['id']; case.mkdir(parents=True, exist_ok=False)
                    save(root/'status.json', dict(status='testing', arm=ARM, case=row['id'],
                        completed=len(results), expected=12, at=time.time()))
                    source = Path(fixture['input_directory'])/(row['id']+'.i32')
                    require(sha(source) == row['served_sha256'], 'Original input changed')
                    tokens = np.fromfile(source, dtype='<i4').tolist()
                    sampling = dict(fixture['sampling'], seed=row['seed'])
                    session = 'overlap-answer-'+uuid.uuid4().hex
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
                require(not remaining and not gpu, 'Overlap model GPU cleanup incomplete')
        audit = verify_campaign(root, fixture)
        save(root/'execution-audit.json', audit)
        scores = dict(cases=len(results), mean_case_score=sum(r['official_case_score'] for r in results)/len(results),
            full_value_coverage=sum(r['value_covered'] for r in results), results=results, scope=SCOPE)
        save(base/'scores.json', scores)
        references = {arm:read(Path(fixture['answer_reference'])/arm/'scores.json') for arm in ('mean_k', 'token_sum')}
        save(root/'summary.json', dict(overlap=scores, previous_answers=references,
            graph_alpha=fixture['graph_alpha'],
            previous_overlap=(read(Path(fixture['overlap_reference'])/ARM/'scores.json')
                if fixture.get('overlap_reference') else None),
            comparison_limit=fixture['comparison_limit'], scope=SCOPE))
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
    parser.add_argument('--root', type=Path)
    parser.add_argument('--source', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--graph-alpha', type=float, choices=(0., .2), default=.2,
        help='Preparation only; execution always uses the frozen profile (default: 0.2)')
    parser.add_argument('--overlap-reference', type=Path,
        help='Completed alpha=0.2 overlap root whose answers remain immutable')
    arguments = parser.parse_args()
    root = arguments.root or (DEFAULT_NO_GRAPH_ROOT if arguments.graph_alpha == 0 else DEFAULT_ROOT)
    if arguments.action == 'prepare':
        prepare(root.resolve(), arguments.source.resolve(), arguments.graph_alpha,
            arguments.overlap_reference.resolve() if arguments.overlap_reference else None)
    else:
        run(root.resolve())

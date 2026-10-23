"""Frozen twelve-case FPS-4 + budgeted adjacent-block atomic group experiment.

Preparation requires an explicit optimized/reference FPS implementation choice.
Old answers remain immutable; runtime, inputs and every answer are hash-audited.
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


import importlib.util

_spec = importlib.util.spec_from_file_location('fps_previous_controller',
    Path(__file__).with_name('test_ruler_fps_answers.py'))
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
require, verify_files = previous.require, previous.verify_files
CASE_FILES, CASE_IDS, FIXTURE_FIELDS, SCOPE = previous.CASE_FILES, previous.CASE_IDS, previous.FIXTURE_FIELDS, previous.SCOPE
ARM = 'key_fps_4_halo_1'
SCORE_MODE = 'key_fps_4'
DEFAULT_ROOT = Path('/data/chaidi/kvmem-vllm-experiments/ruler-key-fps-halo-20261023-r1')
SOURCE_ROOT = Path('/home/chaidi/kvmem-freetoken/kvmem-vllm/artifacts/ruler-key-fps-4-20261023-r1')
GROUP_KEYS = ('mode','radius','atomic_groups','post_picker_expansion','coordinate',
    'anchor_starts','groups','group_scores','group_members')
GROUP_STATS = ('selected_tokens','selected_starts','baseline_selected_starts','added_starts',
    'displaced_starts','covered_anchor_starts')


def profile_difference(reference,candidate):
    before,after=deepcopy(reference),deepcopy(candidate)
    fields=[('port',before.pop('port'),after.pop('port'))]
    for name,default in (('query_fps_halo_blocks',0),('query_fps_optimized',False)):
        fields.append(('kvmem.'+name,before['kvmem'].pop(name,default),after['kvmem'].pop(name,default)))
    require(before==after,'Halo profile differs beyond port, query_fps_halo_blocks and query_fps_optimized')
    return {name:dict(before=a,after=b) for name,a,b in fields if a!=b}


def profile_for_halo(original,*,optimized):
    require(type(optimized) is bool,'Specify optimized=True or False explicitly')
    k=original['kvmem']
    require(k['query_score_mode']==SCORE_MODE and k.get('query_fps_halo_blocks',0)==0 and
        k.get('query_fps_optimized',False) is False,'Expected the completed unoptimized FPS-4/no-halo reference')
    require((k['select_budget'],k['gen_budget'],k['block_tokens'],k['sink_blocks'],k['recent_blocks'])==
        (36864,16384,32,41,0),'Original KV/fixed-keep budget differs')
    require(k['graph_mode']=='multi' and k['graph_scoring_version']=='v2' and k['graph_aggregation']=='sum' and
        k['graph_alpha']==0 and k['graph_trace'] and k['guided_query_tokens']==0 and not k['query_history'],
        'Original Query/identity policy differs')
    require(original['draft_method']=='mtp' and original['draft_tokens']==4 and original['execution_mode']=='compiled_graph' and
        original['prefill_chunk_tokens']==2048 and original['max_num_seqs']==1 and original['max_num_batched_tokens']==16424,
        'Original execution geometry differs')
    require(k['pressure_policy']=='attention' and k['prefill_attention'] and
        not k['prefill_physical_coalescing'] and k['archive_backend']=='cpu','Original prefill policy differs')
    profile=deepcopy(original);profile['port']=18867
    profile['kvmem'].update(query_fps_halo_blocks=1,query_fps_optimized=optimized)
    profile_difference(original,profile)
    return profile


def reference_proof(source,fixture,profile):
    old_fixture=read(source/'fixture.json')
    require(all(old_fixture[key]==fixture[key] for key in FIXTURE_FIELDS),'Original cohort, seed or sampling changed')
    previous.verify_campaign(source,old_fixture)
    proof,snapshots=previous.reference_proof(source,fixture,(SCORE_MODE,))
    for name in ('target-spans.json','reference-answers.json','reference-proof.json','overlap-reference-proof.json'):
        proof['files'][name]=sha(source/name)
    inherited=read(source/'reference-answers.json')
    require(set(inherited)=={'mean_k','token_sum','overlap_16_8'},'Previous frozen references incomplete')
    snapshots.update(inherited)
    require(profile==profile_for_halo(read(source/SCORE_MODE/'profile.json'),
        optimized=profile['kvmem']['query_fps_optimized']),'Prepared halo profile differs')
    proof['profile_difference']=profile_difference(read(source/SCORE_MODE/'profile.json'),profile)
    return proof,snapshots


def prepare(root,source,*,optimized):
    require(not root.exists(),'Use a new halo experiment root')
    fixture=read(source/'fixture.json')
    require(tuple(row['id'] for row in fixture['rows'])==CASE_IDS and fixture['sampling']['max_tokens']==8192,
        'Expected the original twelve-case cohort and output budget')
    profile=profile_for_halo(read(source/SCORE_MODE/'profile.json'),optimized=optimized)
    proof,snapshots=reference_proof(source,fixture,profile)
    for row in fixture['rows']:
        path=Path(fixture['input_directory'])/(row['id']+'.i32')
        require(sha(path)==row['served_sha256'] and path.stat().st_size==row['served_input_tokens']*4,
            'Original full input changed')
    repo=Path(__file__).resolve().parents[1]
    from vllm_kvmem.serve import build_launch
    build_launch(profile,repo,root/ARM/'preflight-only')
    targets=read(source/'target-spans.json')
    require(set(targets)==set(CASE_IDS),'Missing offline evaluation spans')
    root.mkdir(parents=True);runtime=root/'runtime'
    for name in ('src/vllm_kvmem','scripts','csrc','third_party/qw3'):
        shutil.copytree(repo/name,runtime/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('pyproject.toml','setup.py'): shutil.copy2(repo/name,runtime/name)
    files={str(path.relative_to(runtime)):sha(path) for path in runtime.rglob('*') if path.is_file()}
    save(root/'runtime-proof.json',dict(files=files))
    old_files=read(source/'runtime-proof.json')['files']
    proof['runtime_changed_files']={name:dict(before=old_files.get(name),after=files.get(name))
        for name in sorted(set(old_files)|set(files)) if old_files.get(name)!=files.get(name)}
    fixture.update(fps_reference=str(source),query_score_mode=SCORE_MODE,query_fps_halo_blocks=1,
        query_fps_optimized=optimized,graph_alpha=0.,purpose=SCOPE,
        difference='FPS-4 seeds; original logical left/self/right block groups ranked by anchor score, '
            'accepted atomically inside the unchanged KV budget. No post-picker expansion.',
        comparison_limit='The prior FPS-4 answers are immutable. Profile changes only halo=1, '
            'the explicitly selected implementation flag and the service port. Separate generated trajectories '
            'and source revisions prevent treating all wall-time changes as a causal optimization benefit.')
    for name,item in [('fixture.json',fixture),('reference-proof.json',proof),('reference-answers.json',snapshots),
            ('target-spans.json',targets),(ARM+'/profile.json',profile)]:save(root/name,item)
    names=['fixture.json','reference-proof.json','reference-answers.json','target-spans.json',ARM+'/profile.json','runtime-proof.json']
    save(root/'prepared-proof.json',{name:sha(root/name) for name in names})
    save(root/'status.json',dict(status='prepared',cases=12,arm=ARM,query_fps_optimized=optimized,at=time.time()))


def expected_groups(inputs):
    """Independent positional oracle; never calls the production group builder."""
    starts,counts,scores=inputs['starts'],inputs['counts'],inputs['raw_scores']
    require(len(starts)==len(counts)==len(scores) and starts==sorted(set(starts)), 'Invalid group source index')
    spans=inputs['graph'].get('message_spans',[])
    identities=[]
    for start,count in zip(starts,counts):
        matches=[i for i,item in enumerate(spans) if item[0]<=start and start+count<=item[1]]
        require(len(matches)<=1,'Ambiguous overlapping message identity')
        identities.append(matches[0] if matches else None)
    groups,members=[],[]
    for i,(start,count) in enumerate(zip(starts,counts)):
        lo=hi=i
        def same(j):
            return not spans or identities[i] is not None and identities[j]==identities[i]
        if i and starts[i-1]+counts[i-1]==start and same(i-1):lo=i-1
        if i+1<len(starts) and start+count==starts[i+1] and same(i+1):hi=i+1
        groups.append([starts[lo],starts[hi]+counts[hi]])
        members.append(starts[lo:hi+1])
    return dict(mode='adjacent_1',radius=1,atomic_groups=True,post_picker_expansion=False,
        coordinate='original_logical_tokens_half_open',anchor_starts=list(starts),groups=groups,
        group_scores=list(scores),group_members=members)


def audit_selection(query,graph,inputs,profile):
    import analyze_block_graph as offline
    from vllm_kvmem.graph_scoring import score_graph
    k=profile['kvmem'];stats=query.get('query_key_fps_scoring',{})
    require(k['query_score_mode']==SCORE_MODE and k['graph_alpha']==0 and k['query_fps_halo_blocks']==1 and
        type(k['query_fps_optimized']) is bool,'Requested FPS halo policy differs')
    require(query.get('query_scoring_method')==SCORE_MODE and stats.get('method')==SCORE_MODE and
        stats.get('representatives_per_block')==4 and stats.get('block_tokens')==32 and
        stats.get('selection')=='raw_key_farthest_first' and
        stats.get('normalization')=='block softmax after per-layer/per-Query/per-head representative max',
        'FPS scorer or reduction order differs')
    require(stats.get('optimized') is k['query_fps_optimized'],'Actual FPS implementation differs from explicit profile choice')
    for event in (graph,inputs):
        require(event.get('seed_source')==SCORE_MODE and event.get('mode')=='multi' and
            event.get('scoring_version')=='v2' and event.get('aggregation')=='sum' and event.get('alpha')==0,
            'Query graph identity policy differs')
    require(graph.get('identity') is True and graph.get('propagation_applied') is False,'Graph was not explicit identity')
    require(abs(math.fsum(inputs['raw_scores'])-query['query_score_tokens'])<1e-3,'Query seed mass differs')
    expected=expected_groups(inputs);metadata=inputs.get('query_group_selection',{})
    require(all(metadata.get(key)==expected[key] for key in GROUP_KEYS),'Exported halo groups differ from independent positional oracle')
    summary=query.get('query_group_selection',{})
    for key in ('mode','radius','atomic_groups','post_picker_expansion','selected_tokens'):
        require(key in summary and summary[key]==metadata.get(key),'Bounded group summary differs')
    require(all(value==metadata.get(key) for key,value in summary.items()),'Bounded group summary conflicts with complete export')
    ranges=query['selected_history_ranges'];actual={start for start,_ in ranges}
    require(len(actual)==len(ranges) and inputs['budget']==k['select_budget'] and
        len(actual)*32<=k['select_budget'],'Atomic group block budget differs')
    counts=dict(zip(inputs['starts'],inputs['counts']))
    require(all(counts.get(start)==count for start,count in ranges),'Original selected block extents differ')
    require(coverage(ranges,(0,k['sink_blocks']*32)),'Fixed sink content was lost')
    cfg,snapshot=offline._validate_record(inputs)
    memory,handles=offline._native_history(cfg,inputs['starts'],inputs['counts'])
    try:
        offline._validate_native_masks(inputs,memory,handles)
        mandatory=tuple(handles[start].slot for start in inputs['mandatory_starts'])
        with score_graph(inputs['raw_scores'],inputs['starts'],inputs['transport_mask'],snapshot,mode='multi',
                scoring_version='v2',aggregation='sum',alpha=0,cross_message_weight=inputs['cross_message_weight'],block_tokens=32) as output:
            require(list(output.scores)==list(inputs['raw_scores']) and output.diagnostics['identity'],
                'Identity graph changed original FPS seeds')
            baseline=memory.preview_selection({handles[start].slot:score for start,score in zip(inputs['starts'],output.scores)},
                mandatory,action=False,prefill_scores=None,max_tokens=inputs['budget'])
            baseline={start for start,handle in handles.items() if handle.slot in baseline}
            selected=memory.preview_group_selection(expected['groups'],expected['group_scores'],mandatory,max_tokens=inputs['budget'])
            selected={start for start,handle in handles.items() if handle.slot in selected}
            require(selected==actual,'Observed selection differs from native atomic group picker replay')
            fixed_slots=memory._store.fixed_selection_blocks(list(mandatory),pressure=False,budget=inputs['budget'])
            fixed={start for start,handle in handles.items() if handle.slot in fixed_slots}
            diagnostics=dict(output.diagnostics)
    finally:
        require(memory.close(),'Offline group selection metadata did not release')
    actual_stats=dict(selected_tokens=sum(n for _,n in ranges),selected_starts=sorted(actual),
        baseline_selected_starts=sorted(baseline),added_starts=sorted(actual-baseline),displaced_starts=sorted(baseline-actual),
        covered_anchor_starts=[start for start,members in zip(expected['anchor_starts'],expected['group_members']) if set(members)<=actual],
        selected_blocks=len(actual),charged_tokens=32*len(actual),fixed_starts=sorted(fixed),budget=inputs['budget'],
        unused_block_slots=inputs['budget']//32-len(actual),all_fit=len(inputs['starts'])<=inputs['budget']//32)
    covered_members={start for members in expected['group_members'] if set(members)<=actual for start in members}
    require(fixed<=actual and actual-fixed<=covered_members,'Native result contains a missing fixed block or partial orphan group')
    require(all(metadata.get(key)==value for key,value in actual_stats.items()),'Exported group accounting or baseline differs')
    return dict(native_group_picker_replay_equal=True,native_picker_replay_equal=True,
        independent_group_geometry_verified=True,atomic_groups=True,before_picker_expansion=True,
        post_picker_expansion=False,no_post_picker_neighbors=True,selected_tokens=actual_stats['selected_tokens'],
        charged_block_tokens=32*len(actual),sink_tokens=k['sink_blocks']*32,key_fps_stats=stats,
        graph_alpha=0.,propagation_applied=False,propagation=diagnostics,groups=actual_stats,
        query_fps_optimized=k['query_fps_optimized'])


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
    require(tuple(row['id'] for row in fixture['rows'])==CASE_IDS and fixture['query_score_mode']==SCORE_MODE and
        fixture['graph_alpha']==0 and fixture['query_fps_halo_blocks']==1,'Frozen halo cohort or policy changed')
    profile=read(root/ARM/'profile.json');proof=read(root/'reference-proof.json')
    require(fixture['query_fps_optimized'] is profile['kvmem']['query_fps_optimized'],
        'Frozen FPS implementation choice differs')
    require(proof['root']==fixture['fps_reference'],'FPS reference binding changed')
    verify_files(Path(proof['root']),proof['files'])
    current,snapshots=reference_proof(Path(proof['root']),fixture,profile)
    require(current['files']==proof['files'] and current['profile_difference']==proof['profile_difference'],
        'FPS reference audit changed')
    require(read(root/'reference-answers.json')==snapshots,'Frozen old answers changed')
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
    return dict(status='passed',cases=12,unchanged_frozen_runtime=True,case_hashes=hashes,
        original_inputs_seeds_sampling_verified=True,native_picker_replay_verified=True,
        native_group_picker_replay_verified=True,independent_group_geometry_verified=True,
        official_scores_recomputed=True,no_post_picker_neighbors=True,before_picker_expansion=True,
        pressure_cumulative_attention=True,guided=False,graph_alpha=0.,propagation_applied=False,
        query_score_mode=SCORE_MODE,query_fps_halo_blocks=1,query_fps_optimized=fixture['query_fps_optimized'],
        gpu_cleanup_verified=True,reference_answers_unchanged=True,profile_difference=proof['profile_difference'],scope=SCOPE)


def run(root):
    require(Path(__file__).resolve() == root/'runtime/scripts'/Path(__file__).name,
            'Run the frozen runtime/scripts/test_ruler_fps_halo_answers.py controller')
    verify_files(root/'runtime', read(root/'runtime-proof.json')['files'])
    verify_files(root, read(root/'prepared-proof.json'))
    for name in ('reference-proof.json',):
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
                    session = 'fps-halo-answer-'+uuid.uuid4().hex
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
        references={arm:dict(cases=12,correct=sum(case['result']['official_case_score']==100 for case in entries.values()))
            for arm,entries in read(root/'reference-answers.json').items()}
        save(root/'summary.json',dict(key_fps_halo=scores,previous_answers=references,
            graph_alpha=0.,query_fps_optimized=fixture['query_fps_optimized'],
            comparison_limit=fixture['comparison_limit'],scope=SCOPE))
        save(base/'status.json', dict(status='completed', cases=12, at=time.time()))
        save(root/'status.json', dict(status='completed', cases=12, arms=1, arm=ARM,
            graph_alpha=fixture['graph_alpha'], at=time.time()))
    except BaseException as error:
        save(root/'failure.json', dict(error=repr(error), completed=len(results), at=time.time()))
        save(root/'status.json', dict(status='failed', completed=len(results), error=repr(error), at=time.time()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--source',type=Path,default=SOURCE_ROOT)
    parser.add_argument('--fps-optimized',choices=('true','false'),
        help='Required for prepare; frozen execution never changes the selected implementation')
    args=parser.parse_args()
    if args.action=='prepare':
        require(args.fps_optimized is not None,'prepare requires an explicit --fps-optimized true or false')
        prepare(args.root.resolve(),args.source.resolve(),optimized=args.fps_optimized=='true')
    else:run(args.root.resolve())

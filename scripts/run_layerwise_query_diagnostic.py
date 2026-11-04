"""Fresh Mean-K diagnostics: paired failures or thirteen tasks, two cases each."""
import argparse
import csv
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid

import recall_comparison_protocol as protocol
import ruler_protocol


def save(path, value):
    protocol.save(path, value)


def prepare(root, cohort='fail10'):
    repo = Path(__file__).resolve().parents[1]
    source = repo/'artifacts/deletion-prefix-130-20261102-r1/campaign-v1'
    fixture = protocol.read(source/'fixture.json')
    if cohort == 'balanced26':
        ids = [r['id'] for r in protocol.execution_rows(fixture['rows']) if r['index'] < 2]
        ruler_protocol.validate_cohort([r for r in fixture['rows'] if r['id'] in ids], 2)
        arms = ['sequential']
    elif cohort == 'fail10':
        with (repo/'artifacts/ruler-130-history-20261102-r1/per-case-130.csv').open(encoding='utf-8-sig') as f:
            failed = sorted(r['sample_id'] for r in csv.DictReader(f)
                            if r['record_id'] == 'R012' and r['task'].startswith('niah_')
                            and float(r['score']) < 100)
        ids, arms = failed[:10], ['global', 'sequential']
    else:
        raise ValueError('Unknown diagnostic cohort')
    if len(ids) != (26 if cohort == 'balanced26' else 10) or root.exists():
        raise ValueError('Need a fresh diagnostic root and the exact requested sample matrix')
    root.mkdir(parents=True)
    runtime = root/'runtime-v1'
    reference = repo/'artifacts/layerwise-meank-fail10-20261102-r4'
    previous = reference/'runtime-v1' if cohort == 'balanced26' else repo/'artifacts/deletion-prefix-130-20261102-r1/runtime-v1'
    shutil.copytree(previous, runtime, ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    if cohort == 'fail10':
        shutil.copy2(repo/'src/vllm_kvmem/layerwise_query.py', runtime/'src/vllm_kvmem/layerwise_query.py')
        runner = runtime/'src/vllm_kvmem/runner.py'
        text = runner.read_text()
        anchor = '            self._kvmem_layers = install_qwen_capture(self.model)\n'
        if text.count(anchor) != 1:
            raise ValueError('Frozen runner capture hook is ambiguous')
        runner.write_text(text.replace(anchor, anchor +
            "            if os.getenv('KVMEM_LAYERWISE_QUERY_MODE'):\n"
            "                from .layerwise_query import install_layerwise_query\n"
            "                install_layerwise_query(self)\n"))
        serve = runtime/'src/vllm_kvmem/serve.py'
        text = serve.read_text()
        anchor = '    args = parser.parse_args(argv)\n'
        text = text.replace(anchor,
            "    parser.add_argument('--layerwise-query-diagnostic', choices=('global', 'sequential'))\n"+anchor)
        anchor = '    plan = build_launch(profile, base, run_dir)\n'
        text = text.replace(anchor,anchor+
            "    if args.layerwise_query_diagnostic:\n"
            "        if profile.get('execution_mode') != 'eager' or profile.get('max_num_seqs') != 1:\n"
            "            raise ValueError('Layerwise diagnostic requires serial eager execution')\n"
            "        plan['environment']['KVMEM_LAYERWISE_QUERY_MODE'] = args.layerwise_query_diagnostic\n")
        serve.write_text(text)
    shutil.copy2(__file__, runtime/'scripts'/Path(__file__).name)
    rows = [next(r for r in fixture['rows'] if r['id'] == sid) for sid in ids]
    (root/'inputs').mkdir()
    for row in rows:
        source_input = source/'dataset/inputs'/(row['id']+'.i32')
        if protocol.sha(source_input) != row['served_sha256']:
            raise ValueError('Original prefixed input hash differs')
        shutil.copy2(source_input, root/'inputs'/source_input.name)
    shutil.copytree(source/'upstream', root/'upstream')
    profile = protocol.read(reference/'profile.json' if cohort == 'balanced26' else
        repo/'artifacts/answer-prefix-recall-20261102-r1/campaign-v2/profiles/mean_k.json')
    profile.pop('recall_experiment', None)
    profile.update(port=19021, served_model_name='layerwise-meank-fail10', execution_mode='eager',
                   max_num_seqs=1, verify_session_restore=False, verify_session_restore_digest=False,
                   cudagraph_capture_sizes=None)
    if cohort == 'balanced26':
        profile.update(port=19022,served_model_name='layerwise-meank-balanced26')
    kv = profile['kvmem']
    kv.update(capture_opaque_projection=False, query_replay_backend='eager', query_replay_cobatch=False,
              graph_mode='off', graph_alpha=0., query_score_mode='mean_k')
    for key in ('model','tokenizer','hf_config_path'):
        profile[key] = str(Path(profile[key]).resolve())
    if (kv['select_budget'],kv['gen_budget'],kv['block_tokens'],kv['sink_blocks'],kv['recent_blocks']) != (36864,16384,32,41,0):
        raise ValueError('Original KV controls changed')
    save(root/'profile.json', profile)
    frozen = {k:deepcopy(v) for k,v in fixture.items() if k not in ('rows',)}
    frozen.update(rows=rows, qualification_ids=ids, source_fixture=str(source/'fixture.json'),
                  source_fixture_sha256=protocol.sha(source/'fixture.json'), workers=1,
                  diagnostic_scope='Ten original Mean-K NIAH failures; paired fresh answers, not benchmark accuracy.',
                  selection_rule='Lexicographically first ten failed NIAH IDs from historical R012 same130 registry.',
                  arms=arms, input_directory='inputs', upstream_path='upstream',
                  output_cap=128, algorithm='Mean-K head mean / Query row sum; global layer mean versus per-layer selection',
                  fixed_controls=dict(active_kv=36864,gen_kv=16384,thinking=False,answer_prefix='official',
                    prefill_pressure='accumulated_attention',decode_pressure='accumulated_attention'),
                  diagnostic_limits=['Serial eager execution in BOTH arms; timing is not a production performance comparison.',
                    'Per-layer KV contents differ only after the final Query; per-layer capacity and total cache allocation unchanged.',
                    'One public turn only; any subsequent refresh after layerwise selection fails explicitly.',
                    'Historical failures were selected under an earlier thinking-enabled protocol; this fresh baseline may recover some.',
                    'This comparison changes both layer aggregation and causal selection order; frozen-Q layerwise sets are recorded separately.'])
    frozen['cohort'] = cohort
    if cohort == 'balanced26':
        frozen.update(cases=26,tasks=13,samples_per_task=2,pilot_samples_per_task=2,
            diagnostic_scope='Sequential per-layer Mean-K on original indices 0 and 1 of all thirteen RULER tasks; fresh independent answers.',
            selection_rule='First two original fixed-list samples per task; no selection by score.',
            algorithm_runtime_reference=str(reference),
            algorithm_runtime_reference_proof_sha256=protocol.sha(reference/'runtime-proof.json'),
            methods=['mean_k_sequential_layerwise'],pilot_selected_methods=['mean_k_sequential_layerwise'],
            value_recall_reference=False,full_value_comparison=False,pressure_attention_hybrid=False,
            value_query_optimization=False,answer_prefix_pilot=False,
            execution_profile='layerwise-meank-v1-balanced26',runtime_source=str(runtime),
            experiment_scope='Fresh 13x2 sequential layerwise Mean-K; existing cumulative-attention pressure; fixed original inputs, per-task official limits and unchanged KV budgets.',
            output_cap='official_per_task',output_caps=deepcopy(fixture['task_max_tokens']),
            diagnostic_limits=[
                'Same serial eager per-layer implementation as the completed ten-failure diagnostic; no algorithm changes.',
                'One public turn only; subsequent refresh after layer-specific selection is unsupported.',
                'No new global Mean-K control in this campaign; earlier compiled/eight-owner results are historical references.',
                'Two cases per task provide a small-batch signal, not a stable full-benchmark estimate.'])
    save(root/'fixture.json', frozen)
    proof = {str(p.relative_to(runtime)):protocol.sha(p) for p in runtime.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    save(root/'runtime-proof.json', proof)
    save(root/'prepared.json', dict(at=time.time(),ids=ids,expected_runs=len(rows)*len(arms),
        profile_sha256=protocol.sha(root/'profile.json'),fixture_sha256=protocol.sha(root/'fixture.json'),
        runtime_proof_sha256=protocol.sha(root/'runtime-proof.json')))
    save(root/'status.json',dict(status='prepared',at=time.time(),completed=0,expected=len(rows)*len(arms)))
    print(json.dumps(dict(status='prepared',root=str(root),ids=ids)),flush=True)


def verify(root):
    proof = protocol.read(root/'prepared.json')
    for name in ('profile','fixture','runtime-proof'):
        key = name.replace('-','_')+'_sha256'
        if protocol.sha(root/(name+'.json')) != proof[key]:
            raise ValueError('Frozen diagnostic configuration changed: '+name)
    for name, digest in protocol.read(root/'runtime-proof.json').items():
        if protocol.sha(root/'runtime-v1'/name) != digest:
            raise ValueError('Frozen source changed: '+name)


def run(root, arms, smoke=False):
    verify(root)
    runtime = root/'runtime-v1'
    sys.path[:0] = [str(runtime/'src'), str(runtime/'scripts')]
    import qualify_graph_niah as client
    import benchmark_block_graph as lifecycle
    from run_recall_comparison import wait_health, stop_owned, gpu_lock, termination_guard
    fixture, profile = protocol.read(root/'fixture.json'), protocol.read(root/'profile.json')
    if not arms or len(arms)!=len(set(arms)) or not set(arms)<=set(fixture['arms']):
        raise ValueError('Run only the frozen, explicitly requested methods')
    scorer = ruler_protocol.load_scorer(root/'upstream')
    phase = 'smoke' if smoke else 'formal'
    rows = fixture['rows'][:1] if smoke else fixture['rows']
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('KVMEM_','VLLM_'))}
    environment.update(PYTHONPATH=str(runtime/'src'),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',CUDA_HOME='/usr/local/cuda',
        PATH='/usr/local/cuda/bin:'+str(Path(sys.executable).parent)+':'+os.environ['PATH'])
    with termination_guard(),gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
        if lifecycle.gpu_processes():
            raise RuntimeError('GPU has another owner; leave it untouched')
        for arm in arms:
            verify(root)
            environment['KVMEM_LAYERWISE_QUERY_MODE'] = arm
            output = root/phase/arm
            output.mkdir(parents=True,exist_ok=True)
            attempts = output/'server-attempts';attempts.mkdir(exist_ok=True)
            attempt = attempts/f'{len(list(attempts.iterdir()))+1:04d}';attempt.mkdir()
            capture = attempt/'server/capture'
            command = [sys.executable,'-B','-m','vllm_kvmem','serve','--config',str(root/'profile.json'),
                       '--run-dir',str(attempt/'server'),'--layerwise-query-diagnostic',arm]
            if fixture.get('layerwise_optimized'):
                command += ['--layerwise-query-optimized']
                if fixture.get('layerwise_full_diagnostics'):
                    command += ['--layerwise-query-full-diagnostics']
            if fixture.get('layerwise_prefill_pressure'):
                command += ['--layerwise-prefill-pressure']
            server = None
            try:
                save(root/'status.json',dict(status='loading',phase=phase,arm=arm,at=time.time()))
                with (attempt/'server.log').open('x') as log:
                    server=subprocess.Popen(command,cwd=runtime,env=environment,stdout=log,
                                            stderr=subprocess.STDOUT,start_new_session=True)
                save(attempt/'process.json',dict(pid=server.pid,command=command,mode=arm,at=time.time()))
                save(root/'process.json',dict(pid=os.getpid(),server_pid=server.pid,phase=phase,arm=arm,at=time.time()))
                wait_health(server,profile)
                for index,row in enumerate(rows):
                    case=output/'cases'/row['id'];case.mkdir(parents=True,exist_ok=True)
                    if (case/'result.json').exists():
                        continue
                    if (case/'response.json').exists():
                        raise RuntimeError('Completed response awaiting audit; preserve it without regeneration')
                    save(root/'status.json',dict(status='running',phase=phase,arm=arm,
                        sample_id=row['id'],completed_in_arm=index,expected_in_arm=len(rows),at=time.time()))
                    import numpy as np
                    path=root/'inputs'/(row['id']+'.i32')
                    if protocol.sha(path)!=row['served_sha256']:
                        raise ValueError('Original input changed')
                    tokens=np.fromfile(path,dtype='<i4').tolist()
                    sampling=protocol.case_sampling(fixture,row)
                    if smoke:sampling=dict(sampling,max_tokens=1)
                    session='layerwise-'+arm+'-'+uuid.uuid4().hex
                    save(case/'request.json',dict(sample_id=row['id'],session=session,input_sha256=row['served_sha256'],
                        sampling=sampling,query_span=row['served_query'],fixture_sha256=protocol.sha(root/'fixture.json')))
                    response=client.stream(profile,session,tokens,row['served_query'],sampling,3600)
                    save(case/'response.json',response)
                    terminal=client.wait_retired(capture,response['response_id'])
                    if terminal['history_end'] != row['served_input_tokens']+len(response['output_ids'])-1:
                        raise RuntimeError('Accepted input/output token accounting differs')
                    released=client.wait_owner_released(capture,session,response,terminal)
                    events=[json.loads(line) for line in (capture/'layerwise_query.jsonl').read_text().splitlines()]
                    events=[e for e in events if e.get('request_id','').startswith(response['response_id'])]
                    layers=[e for e in events if e['event']=='layer_selected']
                    if len(layers)!=16 or [e['layer'] for e in layers]!=list(range(16)):
                        raise RuntimeError('Not all sixteen actual attention layers selected in order')
                    if any(e['selected_tokens']>36864 or e['charged_tokens']>36864 for e in layers):
                        raise RuntimeError('Layerwise Active budget exceeded')
                    bounds=[json.loads(line) for line in (capture/'bounded_refresh.jsonl').read_text().splitlines()]
                    bounds=[e for e in bounds if e['request_id'].startswith(response['response_id'])]
                    pressures=[e for e in bounds if e['selection']=='pressure']
                    pressure_audit=[e for e in events if e['event']=='pressure_selected']
                    pressure_method = ('accumulated_attention_per_layer' if fixture.get('layerwise_prefill_pressure')
                                       else 'accumulated_attention')
                    if (not pressures or len(pressure_audit)!=len(pressures) or
                            any(e['actual_method']!=pressure_method for e in pressure_audit)):
                        raise RuntimeError('Prefill pressure did not use accumulated attention')
                    if fixture.get('layerwise_prefill_pressure'):
                        for event in pressure_audit:
                            selected=event['layers']
                            if ([e['layer'] for e in selected]!=list(range(16)) or
                                    any(e['charged_tokens']>36864 or e['selected_tokens']!=event['selected_tokens'] or
                                        e['accepted_score_end']>event['accepted_history'] or e['fallback_reason'] is not None
                                        for e in selected)):
                                raise RuntimeError('Layer pressure evidence, accepted boundary or budget differs')
                    save(case/'evidence.json',dict(layers=layers,events=events,refresh= bounds,
                        terminal=terminal,release=released,mode=arm))
                    score=protocol.score_case(row,response,scorer,starts_in_thinking=False)
                    result=dict(id=row['id'],task=row['task'],index=row['index'],arm=arm,phase=phase,
                        input_sha256=row['served_sha256'],response_sha256=protocol.sha(case/'response.json'),
                        evidence_sha256=protocol.sha(case/'evidence.json'),text=response['text'],
                        finish_reason=response['finish_reason'],seconds=response['seconds'],
                        ttft_seconds=response['ttft_seconds'],completion_tokens=len(response['output_ids']),
                        max_tokens=sampling['max_tokens'],**score)
                    save(case/'result.json',result)
                    print(json.dumps({k:result[k] for k in ('id','arm','phase','official_case_score','seconds','finish_reason')},ensure_ascii=False),flush=True)
                save(output/'completed.json',dict(cases=len(rows),at=time.time()))
            except BaseException as error:
                save(root/'status.json',dict(status='failed',phase=phase,arm=arm,error=repr(error),at=time.time()))
                raise
            finally:
                stop_owned(server)
                save(attempt/'cleanup.json',dict(pid=server.pid if server else None,
                    returncode=server.returncode if server else None,at=time.time()))
        verify(root)
        if fixture.get('cohort') in ('balanced26', 'ruler260') and not smoke:
            analyze_balanced26(root)
        save(root/'status.json',dict(status='smoke_completed' if smoke else 'completed',phase=phase,
            arms=arms,cases_per_arm=len(rows),at=time.time()))


def analyze_balanced26(root):
    verify(root)
    fixture=protocol.read(root/'fixture.json')
    per_task_count = fixture.get('samples_per_task', 2)
    case_count = len(fixture['rows'])
    ruler_protocol.validate_cohort(fixture['rows'],per_task_count)
    scorer=ruler_protocol.load_scorer(root/'upstream')
    results=[];predictions={};response_ids=set()
    for row in fixture['rows']:
        case=root/'formal/sequential/cases'/row['id']
        result=protocol.read(case/'result.json')
        response=protocol.read(case/'response.json')
        request=protocol.read(case/'request.json')
        evidence=protocol.read(case/'evidence.json')
        if (protocol.sha(case/'response.json')!=result['response_sha256'] or
            protocol.sha(case/'evidence.json')!=result['evidence_sha256'] or
            protocol.sha(root/'inputs'/(row['id']+'.i32'))!=row['served_sha256'] or
            request['sampling']!=protocol.case_sampling(fixture,row) or
            request['input_sha256']!=row['served_sha256'] or request['query_span']!=row['served_query'] or
            request['fixture_sha256']!=protocol.sha(root/'fixture.json')):
            raise ValueError('Original frozen inputs, answers, controls or evidence changed')
        rescored=protocol.score_case(row,response,scorer,
            starts_in_thinking=fixture.get('starts_in_thinking', False))
        if rescored['official_case_score']!=result['official_case_score']:
            raise ValueError('Saved official score differs on rescore')
        if response['response_id'] in response_ids:
            raise ValueError('A response was reused across samples')
        response_ids.add(response['response_id'])
        if (len(evidence['layers'])!=16 or any(
            layer['charged_tokens']>36864 or layer['selected_tokens']>36864 for layer in evidence['layers'])):
            raise ValueError('Per-layer budget or causal-order evidence incomplete')
        armed=[e for e in evidence['events'] if e['event']=='armed']
        if len(armed)!=1 or armed[0]['actual_method']!='mean_k_sequential_layerwise':
            raise ValueError('A sample did not execute sequential layerwise Mean-K')
        predictions[(row['task'],row['index'])]=response['text']
        results.append(result)
    official=ruler_protocol.score_suite(fixture['rows'],predictions,scorer,
        samples_per_task=per_task_count,starts_in_thinking=fixture.get('starts_in_thinking', False))
    per_task={task:dict(values,truncated=sum(r['finish_reason']=='length' for r in results if r['task']==task),
                        max_tokens=protocol.case_sampling(fixture,
                            next(r for r in fixture['rows'] if r['task']==task))['max_tokens'])
              for task,values in official['per_task'].items()}
    output=dict(official,method='mean_k_sequential_layerwise',per_task=per_task,
        thinking=fixture.get('starts_in_thinking', False),
        answer_prefix=fixture.get('answer_prefix_policy', 'official'),active_kv=36864,gen_kv=16384,
        thinking_incomplete=sum(r.get('thinking_complete') is False for r in results),
        truncated=sum(r['finish_reason']=='length' for r in results),
        truncation_rate=sum(r['finish_reason']=='length' for r in results)/case_count,
        median_seconds=statistics.median(r['seconds'] for r in results),
        total_inference_seconds=sum(r['seconds'] for r in results),
        independent_answers=case_count,official_rescore_verified=True,
        scope=fixture['diagnostic_scope'],limits=fixture['diagnostic_limits'])
    save(root/'analysis/comparison.json',output)
    with (root/'analysis/per-task.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['task','cases','official_score','truncated','max_tokens'])
        for task,row in per_task.items():writer.writerow([task,row['cases'],row['score'],row['truncated'],row['max_tokens']])
    with (root/'analysis/per-case.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['id','task','index','official_score','finish_reason','completion_tokens','max_tokens','seconds'])
        for r in results:writer.writerow([r['id'],r['task'],r['index'],r['official_case_score'],r['finish_reason'],r['completion_tokens'],r['max_tokens'],r['seconds']])
    if protocol.sha(Path(fixture['source_fixture']))!=fixture['source_fixture_sha256']:
        raise ValueError('The original source fixture changed')
    save(root/'completion-audit.json',dict(status='passed',cases=case_count,tasks=13,samples_per_task=per_task_count,
        independent_responses=len(response_ids),full_attention_layer_selections=16*case_count,
        runtime_and_configuration_hashes_unchanged=True,official_rescore_verified=True,
        original_source_fixture_unchanged=True,formal_answers_resampled=0,at=time.time()))
    print(json.dumps(dict(status='scored',cases=case_count,macro_score=official['macro_score'],
        truncated=output['truncated'],per_task={k:v['score'] for k,v in per_task.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','analyze'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arms',nargs='+',choices=('global','sequential'))
    parser.add_argument('--cohort',choices=('fail10','balanced26'),default='fail10')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':prepare(root,args.cohort)
    elif args.action=='analyze':analyze_balanced26(root)
    else:run(root,args.arms or protocol.read(root/'fixture.json')['arms'],args.smoke)

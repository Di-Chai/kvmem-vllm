"""Cold off/collect client with separate, non-fabricated off evidence."""
import argparse
import ast
import json
import math
from pathlib import Path
import re
import statistics
import time
from types import SimpleNamespace

from tokenizers import Tokenizer
from vllm_kvmem.offline_queue import OfflineQueue, OfflineTask

import benchmark_graph_collection as control
import graph_v2_protocol as protocol
import qualify_block_graph as q
import qualify_graph_niah as old
import qualify_graph_v2_performance as counters

require, sha, save = protocol.require, protocol.sha, old.save


def audit_launch(profile, capture, variant):
    control.validate_profile(profile,variant)
    root = capture.parent.parent
    manifest=json.loads((root/'manifest.json').read_text())
    require(manifest.get('schema')==control.SCHEMA and manifest['arm']==variant and
        manifest['native_identity']==control.native_identity(),'Frozen native source/variant differs')
    require(sha(root/'expected-launch.json')==manifest['expected_launch_sha256'],'Expected launch changed')
    expected=json.loads((root/'expected-launch.json').read_text())
    actual=json.loads((capture.parent/'launch.json').read_text())
    require(all(actual.get(k)==expected[k] for k in ('profile','vllm_argv')),
            'Actual launch or worker payload differs from frozen expectation')
    # This helper audits actual FP8/MTP4/compiled/K/G/chunk argv and environment,
    # without calling the old pressure=false profile/selection validator.
    result=old.audit_launch(profile,capture)
    plan=json.loads(actual['environment']['KVMEM_REQUEST_PLAN_CONFIG'])
    require(plan.get('pressure_policy')=='attention','Actual RequestPlan pressure policy changed')
    geometry=json.loads((capture/'cache-geometry.json').read_text())
    require((geometry.get('select_budget'),geometry.get('gen_budget'),geometry.get('historical_tokens'),
        geometry.get('cpu_archive_bytes_per_token'),geometry.get('cpu_offload_bytes'))==
        (36864,16384,524288,34816,100000000000),'Actual native/cache geometry changed')
    return dict(result,variant=variant,native_identity=manifest['native_identity'],geometry=geometry,
        aligned_mean_k=dict(contract=protocol.SCORE_DOMAIN,evidence='Frozen native score-frame implementation plus observed MeanK backend/selection; off has no graph-domain event.'),
        attention_runtime_audit='Deferred until actual requests initialize ActionRuntime')


def audit_attention_runtime(capture):
    # ActionRuntime is initialized lazily by the first actual forward. Reading
    # this file in the pre-request launch audit races that initialization.
    action=json.loads((capture/'action_config.json').read_text())
    require(action.get('prefill_attention') is True and action.get('action_pressure') is True and
        action.get('fraction')==0 and action.get('scope')=='non-thinking' and
        action.get('async_action_metadata') is False and action.get('native_action_metadata') is False,
        'Actual attention collection or pressure was disabled/changed')
    return dict(status='passed',scope='Actual ActionRuntime after the original cohort completed',
                attention_configuration=action)


def selection_budget(event, profile):
    kv=profile['kvmem']; ranges=event.get('selected_history_ranges')
    require(isinstance(ranges,list) and ranges and type(event.get('history_begin')) is int,
            'Missing actual selected history ranges')
    end=total=0
    for item in ranges:
        require(isinstance(item,list) and len(item)==2 and all(type(x) is int for x in item),'Invalid history range')
        start,count=item
        require(start>=end and start%kv['block_tokens']==0 and 0<count<=kv['block_tokens'] and
            start+count<=event['history_begin'],'Overlapping,invalid or future selected history')
        end=start+count; total+=count
    require(type(event.get('selected_tokens')) is int and 0<event['selected_tokens']==total<=kv['select_budget'],
            'Query/pressure selected-token budget differs')
    require(type(event.get('query_tokens')) is int and 0<event['query_tokens']<=kv['gen_budget'] and
        event['selected_tokens']+event['query_tokens']<=kv['select_budget']+kv['gen_budget'],
        'Query/replay window exceeds K/G')
    for block in (event.get('mandatory_fit') or {}).get('hard_blocks',[]):
        require(type(block) is int and old.covered(block*kv['block_tokens'],(block+1)*kv['block_tokens'],ranges),
                'Selected window dropped an actual hard mandatory block')


def audit_events(refresh, graph, action, profile, variant, query):
    control.validate_profile(profile,variant)
    queries=[e for e in refresh if e.get('selection')=='query']
    pressure=[e for e in refresh if e.get('selection')=='pressure']
    require(len(queries)==1 and pressure,'Long original input requires one Query and actual pressure selections')
    selected=queries[0]
    require(selected.get('query_span')==list(query) and selected.get('mean_k_backend')=='qw3_cuda' and
        selected.get('query_history_enabled') is False and selected.get('guided_query_tokens')==0 and
        selected.get('action_attention_enabled') is True and selected.get('action_history_pages')==0 and
        selected.get('action_fraction')==0,'MeanK/Query changed or independent case inherited output statistics')
    require(selected.get('prefill_attention_samples',0)>0 and action.get('prefill_samples')==selected['prefill_attention_samples'],
            'Missing prefill collection or replay polluted accepted attention samples')
    require(len(refresh)==len(queries)+len(pressure),'Unexpected selection phase')
    for event in refresh:
        selection_budget(event,profile)
        require(event.get('raw_cache_hit_tokens')==0 and not event.get('cancelled'),'Disabled raw-K hit or cancelled selection')
    if variant=='off':
        require(not graph and 'block_graph' not in action,'Graph collection/state leaked into off')
        return dict(query=selected,pressure=pressure,graph=[],
            pressure_seed_evidence=dict(exact_runtime_seed_observed=False,
                observed_attention_source_fields=[e.get('attention_source') for e in pressure],
                limitation='Off has no graph_selection_report. attention_source is a live diagnostic, not the exact working-candidate pressure seed.'),
            graph_evidence=dict(expected=False,observed=False),
            aligned_domain_event_observed=False)
    evidence=protocol.audit_events(refresh,graph,profile,'A',query,require_pressure=True)
    require(all(e.get('selection_unchanged') is True for e in graph if e.get('phase')=='query'),
            'Collect did not preserve its native MeanK selection')
    stats=action.get('block_graph')
    require(isinstance(stats,dict) and stats.get('prefill_samples',0)>0 and
        stats.get('observed_sources',0)>0 and 0<stats.get('total_bytes',0)<=profile['kvmem']['graph_max_bytes_per_session'],
        'Collect did not maintain real accepted graph state within budget')
    return dict(evidence,pressure_seed_evidence=dict(exact_runtime_seed_observed=True,
        sources=[e['seed_source'] for e in pressure]),graph_evidence=dict(expected=True,observed=True),
        aligned_domain_event_observed=True)


def finish_case(args,profile,row,scorer,tokenizer,pending):
    result,response,tokens=pending
    directory=args.output/'cases'/row['id']
    terminal=old.wait_retired(args.capture,response['response_id'])
    end=len(tokens)+len(response['output_ids'])-bool(response['output_ids'])
    require(terminal['history_end']==end and not terminal.get('checkpoint_tokens'),
            'Incomplete accepted history or cross-case checkpoint retention')
    release=old.wait_owner_released(args.capture,result['session'],response,terminal)
    save(directory/'release.json',release)
    match=lambda e:q.native_matches(response,e)
    actions=[e for e in q.records(args.capture/'action_attention.jsonl') if match(e)]
    require(len(actions)==1,'Missing or duplicate per-case attention completion')
    action=actions[0]
    expected=q.expected_output_samples(SimpleNamespace(convert_tokens_to_ids=tokenizer.token_to_id),profile,tokens,response['output_ids'])
    require(action.get('samples')==expected,'Accepted-output sample oracle differs; replay/rejection or cross-case leakage')
    evidence=audit_events([e for e in q.records(args.capture/'bounded_refresh.jsonl') if match(e)],
        [e for e in q.records(args.capture/'block_graph.jsonl') if match(e)],action,profile,args.arm,row['served_query'])
    if args.arm=='collect':
        stats=action['block_graph']
        require(stats.get('accepted_through')==end and 0<=stats.get('completed_through',-1)<=end and
            stats.get('output_samples')==expected,'Collect retained unaccepted/replayed/cross-case output samples')
    save(directory/'evidence.json',dict(evidence,action=action,terminal=terminal))
    result.update(status='completed',evaluation=scorer(response['text'],row['expected']),
        completion_tokens=len(response['output_ids']),seconds=response['seconds'],ttft_seconds=response['ttft_seconds'],
        hit_output_cap=len(response['output_ids'])==fixture_output_limit(args),
        query_span=row['served_query'],pressure_seed_observed=evidence['pressure_seed_evidence']['exact_runtime_seed_observed'])
    save(directory/'result.json',result)
    return result


def fixture_output_limit(args):
    return json.loads(args.fixture.read_text())['sampling']['max_tokens']


def archive_contract(archive,variant,profile):
    require(archive.get('archive_backend')=='cpu' and archive.get('repository_gc') is False and
        all(archive.get(k)==0 for k in ('disk_read_bytes','disk_write_bytes','nvme_capacity_bytes','block_graph_bytes')),
        'CPU-only authority or final graph release differs')
    require(archive.get('block_graph_limit_bytes')==(0 if variant=='off' else profile['kvmem']['graph_max_total_bytes']),
            'Off allocated a graph budget or collect resource contract changed')


def run_niah(args,profile,fixture):
    require(fixture.get('schema')==protocol.SCHEMA and fixture.get('suite')=='niah' and
        fixture.get('scope')=='smoke8' and len(fixture['rows'])==8 and fixture['workers']==8 and
        all(row['nominal_input_tokens']==262144 for row in fixture['rows']),'Require the original eight256K cohort')
    require(sha(Path(profile['tokenizer'])/'tokenizer.json')==fixture['tokenizer_sha256'],'Tokenizer changed')
    tree=ast.parse(fixture['scorer_source'])
    require(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name=='score','Invalid original scorer')
    scope={'re':re}; exec(compile(tree,'frozen-original-niah-score','exec'),scope)
    tokenizer=Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    queue=OfflineQueue([OfflineTask(row['id'],row,old.history_bytes(row,fixture,profile)) for row in fixture['rows']],
        max_inflight=8,history_capacity_bytes=fixture['cpu_payload_credits_bytes'])
    observed=[]
    def on_result(result):
        state=queue.states[result['id']]
        result.update(cpu_queue_seconds=state['admitted_at']-state['queued_at'],
            retirement_seconds=state['completed_at']-state['inference_done_at'])
        save(args.output/'cases'/result['id']/'result.json',result); observed.append(result)
        save(args.output/'progress.json',dict(completed=len(observed),expected=8))
    def on_state(state):
        save(args.output/'queue.json',state)
        with (args.output/'queue_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=time.monotonic(),**{k:v for k,v in state.items() if k!='tasks'}))+'\n')
    begin=time.monotonic()
    results=queue.run(lambda task:old.infer_case(args,fixture,profile,task.payload,tokenizer),
        lambda task,pending:finish_case(args,profile,task.payload,scope['score'],tokenizer,pending),
        on_result=on_result,on_state=on_state,
        on_failure=lambda error:save(args.output/'client-failure.json',dict(error=repr(error))))
    released=q.wait_released(args.capture,[row['session'] for row in results],timeout=180)
    archive_contract(released['archive'],args.arm,profile)
    end=time.monotonic()
    require(len(results)==8 and queue.peak_inflight==8 and queue.inflight==queue.reserved_bytes==0,
            'Missing eight-owner admission/completion or leaked CPU credits')
    steps=[e for e in q.records(args.capture/'model_step_latency.jsonl') if begin<=e['started_at']<=end]
    if args.arm=='off':
        require(not q.records(args.capture/'block_graph.jsonl') and not q.records(args.capture/'block_graph_inputs.jsonl'),
                'Off unexpectedly produced graph state/trace')
    return dict(status='passed',schema=control.SCHEMA,variant=args.arm,all_owned_sessions_released=True,
        fixture_sha256=sha(args.fixture),measurement_started_at=begin,measurement_finished_at=end,wall_seconds=end-begin,
        cpu_admission_peak=queue.peak_inflight,cpu_admission_peak_bytes=queue.peak_bytes,
        actual_concurrency=protocol.audit_actual_concurrency(steps),final_archive=released['archive'],
        step_summary=q.summarize_steps(args.capture,begin,end),results=sorted(results,key=lambda r:r['id']),
        macro_coverage=statistics.fmean(r['evaluation']['normalized_coverage'] for r in results),
        exact=sum(r['evaluation']['normalized_exact_numeric_set'] for r in results),
        measurement_protocol='Cold eight-case cohort through release; no warmup,300s window or replacement workload.')


def measurements(output,capture,profile,result):
    cases={}
    actions=[]
    for row in result['results']:
        directory=output/'cases'/row['id']
        response=json.loads((directory/'response.json').read_text())
        evidence=json.loads((directory/'evidence.json').read_text()); actions.append(evidence['action'])
        selected=[dict(phase=e['selection'],ranges=e['selected_history_ranges']) for e in
            [evidence['query'],*evidence['pressure']]]
        require(len(response['output_ids'])==row['completion_tokens'],'Accepted output count differs')
        cases[row['id']]=dict(input_sha256=row['input_sha256'],input_tokens=row['input_tokens'],seed=row['seed'],
            query_span=row['query_span'],output_ids_sha256=counters.digest(response['output_ids']),
            selection_sha256=counters.digest(selected),output_tokens=row['completion_tokens'],
            attention_samples={k:evidence['action'][k] for k in ('samples','prefill_samples')},
            seconds=row['seconds'],ttft_seconds=row['ttft_seconds'],evaluation=row['evaluation'],
            exact_pressure_seed_observed=evidence['pressure_seed_evidence']['exact_runtime_seed_observed'])
    begin,end,wall=(result[k] for k in ('measurement_started_at','measurement_finished_at','wall_seconds'))
    require(len(cases)==8 and all(type(v) in (int,float) and math.isfinite(v) for v in (begin,end,wall)) and
        wall>0 and math.isclose(end-begin,wall,rel_tol=1e-9,abs_tol=1e-5),'Invalid cold measurement window')
    refresh=[e for e in q.records(capture/'bounded_refresh.jsonl') if begin<=e['refresh_started_at']<=end]
    fields=('graph_prepare_seconds','graph_prepare_calls','graph_commit_seconds','graph_commit_calls')
    maxima={key:max((r[key] for r in actions if key in r),default=None) for key in fields}
    return dict(schema=control.SCHEMA,variant=profile['kvmem']['graph_mode'],cases=cases,
        wall_seconds=wall,measurement_started_at=begin,measurement_finished_at=end,
        actual_concurrency=result['actual_concurrency'],step_summary=result['step_summary'],
        quality=dict(macro_coverage=result['macro_coverage'],exact=result['exact']),
        useful_input_tokens_per_second=sum(r['input_tokens'] for r in cases.values())/wall,
        accepted_output_tokens_per_second=sum(r['output_tokens'] for r in cases.values())/wall,
        rates_scope='Useful input and accepted output over cold cohort wall; neither is pure decode throughput.',
        h2d_payload_bytes=counters.total(refresh,'assembly_uploaded_bytes'),
        host_prepare_seconds=counters.total(refresh,'assembly_host_prepare_seconds'),
        service=counters.interval_evidence(refresh,'refresh_started_at','refresh_finished_at'),
        collection_exercised=all(r.get('block_graph',{}).get('observed_sources',0)>0 for r in actions),
        off_has_no_graph_state=profile['kvmem']['graph_mode']=='off' and all('block_graph' not in r for r in actions) and
            result['final_archive']['block_graph_limit_bytes']==0,
        graph_engine_cumulative_observed_max=maxima,
        counter_scope='Engine cumulative graph timings use observed maxima, never per-owner sums. These overlapping host scopes are not GPU idle.',
        gpu_idle_inferred=False,performance_gain_established=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','capture','output','fixture'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=control.VARIANTS,required=True)
    args=parser.parse_args(argv)
    profile,fixture=json.loads(args.config.read_text()),json.loads(args.fixture.read_text())
    require(not args.output.exists(),'Never overwrite or retry an existing client attempt')
    args.output.mkdir(parents=True)
    try:
        save(args.output/'launch-audit.json',audit_launch(profile,args.capture,args.arm))
        result=run_niah(args,profile,fixture)
        save(args.output/'attention-runtime-audit.json',audit_attention_runtime(args.capture))
        save(args.output/'collection.json',measurements(args.output,args.capture,profile,result))
        result['collection_sha256']=sha(args.output/'collection.json')
        save(args.output/'results.json',result)
    except BaseException as error:
        save(args.output/'client-failure.json',dict(error=repr(error),at=time.time())); raise
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','final_archive')}),flush=True)


if __name__=='__main__': main()

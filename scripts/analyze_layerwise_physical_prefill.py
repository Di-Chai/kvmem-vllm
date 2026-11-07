"""Full-input semantic-boundary audit and measured pure-prefill comparison."""
import argparse
from pathlib import Path
from statistics import mean

from analyze_mab_cold_prefill import lines, measure
from audit_layerwise_gilopt import read, save_new, canonical_signature, tree


def logical_steps(root, measurement):
    capture=root/'run/server/capture'
    groups={}
    path=capture/'logical_prefill_publication.jsonl'
    if path.exists():
        for group in lines(path):
            assert group['count']==group['retained']==sum(s['count'] for s in group['spans'])
            assert all(s['count']==1024 and s['sampled_rows']==32 and
                       s['sampled_row_indices']==list(range(31,1024,32)) for s in group['spans'])
            assert group['history_begin'] not in groups
            groups[group['history_begin']]=group
    logical=[];used=set()
    for step in lines(capture/'model_step_latency.jsonl'):
        if not measurement['begin_first_model_step']<=step['started_at']<measurement['end_query_refresh_dispatch']:
            continue
        for g in step['batch_geometry']:
            count=g['query_tokens'];start=g['logical_seq_len']-count;local=g['active_seq_len']-count
            if start in groups:
                group=groups[start];assert group['count']==count and group['local_begin']==local
                used.add(start)
                logical.extend((s['history_begin'],s['local_begin'],s['count']) for s in group['spans'])
            else:logical.append((start,local,count))
    assert used==set(groups), 'Every physical receipt must belong to the measured history'
    return logical,dict(physical_groups=len(groups),logical_chunks=sum(len(g['spans']) for g in groups.values()),
        maximum_physical_tokens=max((g['count'] for g in groups.values()),default=1024))


def audit(baseline,candidate):
    measurements=[measure(p) for p in (baseline,candidate)]
    a,b=measurements
    assert (a['sample_id'],a['executed_prefill_tokens'])==(b['sample_id'],b['executed_prefill_tokens'])
    expanded=[logical_steps(p,m) for p,m in zip((baseline,candidate),measurements)]
    assert expanded[0][0]==expanded[1][0], 'Original logical scheduling boundaries changed'
    assert tree(baseline/'runtime-v1')==tree(candidate/'runtime-v1')
    cases=[p/'run/answers/cases'/a['sample_id'] for p in (baseline,candidate)]
    results=[read(p/'result.json') for p in cases]
    signatures=[canonical_signature(r,read(p/'events.json')) for r,p in zip(results,cases)]
    pressures=[s['pressure'] for s in signatures]
    assert len(pressures[0])==len(pressures[1])==a['pressure_events']==b['pressure_events']
    overlap=[]
    for old,new in zip(*pressures):
        for key in ('accepted_history','method','fallback','event_type','query_source','selected_tokens'):
            assert old[key]==new[key],key
        assert len(old['layers'])==len(new['layers'])==16
        layer_overlap=[]
        for x,y in zip(old['layers'],new['layers']):
            for key in ('layer','charged_tokens','accepted_score_end','query_samples','sampled_chunks'):
                assert x[key]==y[key],key
            left,right=set(x['selected_starts']),set(y['selected_starts'])
            layer_overlap.append(len(left&right)/len(left))
        overlap.append(layer_overlap)
    native=[read(p/'run/server/capture/native_attention_stats.json') for p in (baseline,candidate)]
    for n in native:
        assert n['disabled_reason'] is None and n['validation_failures']==0
        assert not any('fallback' in k and v for k,v in n['layer_phase_counts'].items())
    assert native[0]['allocated_bytes']==native[1]['allocated_bytes']
    return dict(baseline=a,candidate=b,throughput_gain_percent=100*(b['executed_tokens_per_second']/a['executed_tokens_per_second']-1),
        physical=expanded[1][1],original_logical_steps_equal=True,logical_steps=len(expanded[0][0]),
        pressure_boundaries_method_budget_and_all_layer_sampling_equal=True,
        first_pressure_mean_block_overlap=mean(overlap[0]),all_pressure_mean_block_overlap=mean(v for event in overlap for v in event),
        native_stats_bytes=native[0]['allocated_bytes'],native_validation_failures=0,
        same_output_tokens=read(cases[0]/'response.json')['choices'][0]['token_ids']==read(cases[1]/'response.json')['choices'][0]['token_ids'],
        same_final_answer=results[0]['answer']==results[1]['answer'],
        temporary_cpu_score_download_ceiling_mib=[read(p/'profile.json')['kvmem'].get(
            'action_download_max_bytes',64<<20)//(1<<20) for p in (baseline,candidate)],
        formal_resume=False,external_grading=False,
        caveat='One matched original cold input; excludes Query/decode/load. Changed batch shape can change scores and answers. No benchmark accuracy or multiowner gain established.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.baseline,args.candidate);save_new(args.output,report)
    print({k:report[k] for k in ['throughput_gain_percent','physical','original_logical_steps_equal','same_output_tokens','same_final_answer']})

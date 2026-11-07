"""Report all predetermined runs, geometry effects and same-shape checks."""
import argparse
from pathlib import Path
from statistics import mean
from collections import defaultdict

from audit_layerwise_gilopt import read,save_new,sha,tree,compare
from analyze_mab_cold_prefill import lines


def analyze(root):
    plan=read(root/'plan.json');rows=[];controls=[];fixtures=[]
    capacity=plan.get('capacity_test',False)
    for arm in plan['order']:
        ar=root/arm;c=ar/'candidate'
        assert read(ar/'status.json')['status']=='completed'
        assert tree(c/'runtime-v1')==read(c/'runtime-proof.json')
        assert not read(c/'run/cleanup.json')['gpu_processes']
        profile=read(c/'profile.json');cfg=profile['kvmem']
        chunks=cfg.pop('prefill_physical_max_chunks');early=cfg.pop('prefill_action_before_mtp')
        assert chunks==((16 if capacity else 14) if arm[0] in 'bd' else 8)
        assert early==(capacity or arm[0] in 'cd')
        controls.append(profile);fixtures.append(read(c/'fixture.json'))
        audit=read(ar/'semantic-audit.json');m=audit['measurement'];t=m['pressure_timings']
        native=read(c/'run/server/capture/native_attention_stats.json')
        progress=read(c/'run/server/capture/progress.json')
        steps=[s for s in lines(c/'run/server/capture/model_step_latency.jsonl')
            if m['begin_first_model_step']<=s['started_at']<m['end_query_refresh_dispatch']]
        gaps=defaultdict(list)
        for previous,current in zip(steps,steps[1:]):
            if current['previous_sequence']!=previous['sequence']:continue
            key=':'.join(str(sum(g['query_tokens'] for g in s['batch_geometry'])) for s in (previous,current))
            gaps[key].append(current['previous_end_to_model_ms'])
        row=dict(arm=arm,physical_chunks=chunks,early_statistics=early,measurement=m,
            physical=audit['physical'],mean_pressure_block_overlap=audit['mean_pressure_block_overlap'],
            equivalence_to_reference=read(ar/'equivalence.json')['equal_counts'],
            logical_steps_equal=audit['logical_steps_equal'],
            pressure_mean_ms=t['suspended_to_ready']['mean_seconds']*1000,
            assembly_mean_ms=t['selection_prepared_to_assembly_submitted']['mean_seconds']*1000,
            native_stats_bytes=native['allocated_bytes'],
            early_statistics_counts=progress.get('prefill_action_before_mtp_stats',{}),
            forward_transition_gaps={k:dict(count=len(v),mean_ms=mean(v),sum_ms=sum(v)) for k,v in gaps.items()},
            mean_forward_tokens=m['executed_prefill_tokens']/len(steps),
            max_forward_tokens=max(sum(g['query_tokens'] for g in s['batch_geometry']) for s in steps))
        rows.append(row)
    assert all(p==controls[0] for p in controls)
    assert all(f==fixtures[0] for f in fixtures)
    assert len({r['native_stats_bytes'] for r in rows})==1
    def pool(letter):
        pairs=[r for r in rows if r['arm'][0]==letter]
        assert len(pairs)==2
        tokens=sum(r['measurement']['executed_prefill_tokens'] for r in pairs)
        seconds=sum(r['measurement']['phase_seconds'] for r in pairs)
        return dict(tokens=tokens,seconds=seconds,tokens_per_second=tokens/seconds,
            pressure_mean_ms=mean(r['pressure_mean_ms'] for r in pairs),
            assembly_mean_ms=mean(r['assembly_mean_ms'] for r in pairs),
            model_event_mean_seconds=mean(r['measurement']['model_event_seconds'] for r in pairs))
    pooled={k:pool(k) for k in sorted({a[0] for a in plan['order']})}
    for k in pooled:pooled[k]['gain_vs_a_percent']=100*(pooled[k]['tokens_per_second']/pooled['a']['tokens_per_second']-1)
    checks={}
    for a,b in [('a1','a2'),('b1','b2'),('c1','c2'),('d1','d2'),
                ('a1','c1'),('b1','d1'),('a1','b1'),('c1','d1')]:
        if a not in plan['order'] or b not in plan['order']:continue
        checks[a+':'+b]=compare(root/a/'candidate',root/b/'candidate')
    for key in ('a1:a2','b1:b2','c1:c2','d1:d2','a1:c1','b1:d1'):
        if key in checks:assert checks[key]['passed'],key
    old=compare(Path(plan['baseline']),root/'a1/candidate')
    saved=read(root/'formal-preservation.json')
    for p,digest in saved.items():assert sha(Path(p))==digest
    return dict(rows=rows,pooled=pooled,exact_checks=checks,previous_capacity_check=old,
        early_statistics_gain_at_8k_percent=None if capacity else 100*(pooled['c']['tokens_per_second']/pooled['a']['tokens_per_second']-1),
        early_statistics_gain_at_14k_percent=None if capacity else 100*(pooled['d']['tokens_per_second']/pooled['b']['tokens_per_second']-1),
        same_model_inputs_budgets_sampling=True,native_stats_bytes=rows[0]['native_stats_bytes'],
        preserved_formal_files=len(saved),formal_resume=False,external_grading=False,
        scope=plan['scope'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root;report=analyze(root);save_new(root/'comparison.json',report)
    print(report['pooled'])

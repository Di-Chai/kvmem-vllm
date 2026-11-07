"""Compare paired complete-input throughput after byte and output validation."""
import argparse
from pathlib import Path
from statistics import mean

from audit_layerwise_gilopt import read,save_new,sha,tree
from analyze_mab_cold_prefill import measure
from analyze_layerwise_physical_prefill import logical_steps


def analyze(root):
    plan=read(root/'plan.json');rows=[];controls=[];steps=[]
    assert read(root/'status.json')['status']=='completed'
    for arm in plan['order']:
        ar=root/arm;c=ar/'candidate';enabled=arm.startswith('b')
        assert read(ar/'status.json')['status']=='completed'
        assert read(ar/'equivalence.json')['passed']
        assert tree(c/'runtime-v1')==read(c/'runtime-proof.json')
        assert not read(c/'run/cleanup.json')['gpu_processes']
        profile=read(c/'profile.json');assert profile['kvmem'].pop('cpu_reuse_record_views')==enabled
        controls.append((profile,read(c/'fixture.json')))
        m=measure(c);steps.append(logical_steps(c,m)[0])
        progress=read(c/'run/server/capture/progress.json')
        publication=progress['publication_metrics'].get('background/bulk',{})
        rows.append(dict(arm=arm,enabled=enabled,measurement=m,
            view_counts=progress['cpu_record_view_stats'],
            publication_logical_seconds=publication.get('logical_transaction_seconds'),
            publication_claim_seconds=publication.get('claim_seconds'),
            all_blocks_and_answer_equal=True))
    assert all(c==controls[0] for c in controls)
    assert all(s==steps[0] for s in steps)
    def pool(enabled):
        rs=[r for r in rows if r['enabled']==enabled]
        tokens=sum(r['measurement']['executed_prefill_tokens'] for r in rs)
        seconds=sum(r['measurement']['phase_seconds'] for r in rs)
        return dict(tokens=tokens,seconds=seconds,tokens_per_second=tokens/seconds,
            publication_logical_seconds_mean=mean(r['publication_logical_seconds'] for r in rs),
            publication_claim_seconds_mean=mean(r['publication_claim_seconds'] for r in rs),
            pressure_seconds_mean=mean(r['measurement']['pressure_timings']['suspended_to_ready']['total_seconds'] for r in rs),
            model_event_seconds_mean=mean(r['measurement']['model_event_seconds'] for r in rs))
    a,b=pool(False),pool(True)
    for p,digest in read(root/'formal-preservation.json').items():assert sha(Path(p))==digest
    return dict(rows=rows,baseline=a,optimized=b,
        throughput_gain_percent=(b['tokens_per_second']/a['tokens_per_second']-1)*100,
        all_pressure_query_blocks_and_full_outputs_equal=True,
        controls_equal_except_execution_flag=True,logical_steps_equal=True,
        formal_resumed=False,external_grading=False,
        limitations=['One original cold MAB input, one live owner, fixed ABBA; not a campaign/multiowner estimate.',
            'Host publication durations overlap GPU/model work and are not additive foreground idle time.',
            'Pure prefill includes maintenance; excludes loading, first untimed 1K, final Query and decode.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.root);save_new(args.root/'comparison.json',result)
    print({k:result[k] for k in ('baseline','optimized','throughput_gain_percent')})

"""Compare finite unprofiled cold-input runs and preserve stage attribution."""
import argparse
from pathlib import Path
from statistics import mean

from audit_layerwise_gilopt import read,save_new,tree
from analyze_mab_cold_prefill import measure,lines
from analyze_layerwise_physical_prefill import logical_steps


def summarize(root):
    order=read(root/'plan.json')['order'];rows=[];controls=[];logical=[]
    for arm in order:
        ar=root/arm;c=ar/'candidate'
        assert read(ar/'status.json')['status']=='completed'
        eq=read(ar/'equivalence.json');assert eq['passed'] and eq['cases']==1
        assert tree(c/'runtime-v1')==read(c/'runtime-proof.json')
        assert not read(c/'run/cleanup.json')['gpu_processes']
        controls.append((read(c/'fixture.json'),read(c/'profile.json')))
        m=measure(c);logical.append(logical_steps(c,m)[0]);capture=c/'run/server/capture'
        bounds=[x for x in lines(capture/'bounded_refresh.jsonl') if x['selection']=='pressure']
        assert len(bounds)==m['pressure_events']==23
        stats=[x['assembly_read_stats'] for x in bounds]
        prepared=sum(x.get('layerwise_prepared_read_chunks',0) for x in stats)
        assert bool(prepared)==(arm in ('b1','b2'))
        assert all(x.get('layerwise_prepared_read_chunks',0)==x['layerwise_paired_read_chunks'] for x in stats) if prepared else True
        native=read(capture/'native_attention_stats.json')
        assert native['disabled_reason'] is None and native['validation_failures']==0
        assert not any('fallback' in k and v for k,v in native['layer_phase_counts'].items())
        t=m['pressure_timings']
        rows.append(dict(arm=arm,measurement=m,
            true_queue_mean_ms=t['assembly_queued_to_assembly_worker_started']['mean_seconds']*1000,
            worker_prepare_mean_ms=t['assembly_worker_started_to_assembly_started']['mean_seconds']*1000,
            assembly_mean_ms=t['selection_prepared_to_assembly_submitted']['mean_seconds']*1000,
            pressure_mean_ms=t['suspended_to_ready']['mean_seconds']*1000,
            uploaded_bytes=sum(x['assembly_uploaded_bytes'] for x in bounds),
            paired_chunks=sum(x['layerwise_paired_read_chunks'] for x in stats),prepared_chunks=prepared,
            prepared_sources=sum(x.get('layerwise_prepared_source_blocks',0) for x in stats),
            full_blocks_and_output_equal=True))
    assert all(c==controls[0] for c in controls)
    assert all(s==logical[0] for s in logical)
    assert len({r['uploaded_bytes'] for r in rows})==1
    def pool(names):
        xs=[r for r in rows if r['arm'] in names]
        tokens=sum(r['measurement']['executed_prefill_tokens'] for r in xs)
        seconds=sum(r['measurement']['phase_seconds'] for r in xs)
        return dict(tokens=tokens,seconds=seconds,tokens_per_second=tokens/seconds,
            worker_prepare_mean_ms=mean(r['worker_prepare_mean_ms'] for r in xs),
            assembly_mean_ms=mean(r['assembly_mean_ms'] for r in xs),
            pressure_mean_ms=mean(r['pressure_mean_ms'] for r in xs))
    a,b=pool(('a1','a2')),pool(('b1','b2'))
    return dict(rows=rows,baseline=a,optimized=b,
        throughput_gain_percent=(b['tokens_per_second']/a['tokens_per_second']-1)*100,
        pressure_cost_reduction_percent=(1-b['pressure_mean_ms']/a['pressure_mean_ms'])*100,
        equivalent=True,original_logical_steps_equal=True,all_controls_equal=True,
        same_uploaded_bytes=True,formal_resume=False,external_grading=False,
        scope='Same complete original single cold MAB history. Fixed a1, metadata, b1, b2, a2 order; 8K merging, 36K/16K budgets, original sampling. Pure prefill excludes load/Query/decode. Single-stage metadata measurement is not a repeated speedup claim; no multiowner or benchmark accuracy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    a=p.parse_args();x=summarize(a.root);save_new(a.root/'comparison.json',x)
    print({k:x[k] for k in ('baseline','optimized','throughput_gain_percent','pressure_cost_reduction_percent','equivalent')})

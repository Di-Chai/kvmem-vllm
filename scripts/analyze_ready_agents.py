"""Read-only review of ready-agent fixed windows and retained-history evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


core=load('ready_analysis_core','capacity_scaling_workload.py')
accounting=load('ready_analysis_accounting','analyze_deepswe_capacity.py')


def records(path):
    return core.read_records(path)


def geometry(body):
    rows=[r for r in body.splitlines() if r.startswith('vllm:cache_config_info{')]
    if len(rows)!=1:return None
    labels=dict(re.findall(r'(\w+)="([^"\\]*)"',rows[0]))
    keys=('block_size','num_gpu_blocks')
    return {k:int(labels[k]) for k in keys} if all(labels.get(k,'').isdigit() for k in keys) else None


def capacity_proof(traces,steps,native_geometry,window):
    """Unknown GPU membership uses the shortest possible subset, never all HTTP."""
    if native_geometry is None:return dict(available=False)
    pool,block=native_geometry['num_gpu_blocks'],native_geometry['block_size']
    matched=above=0;best=None
    for step in steps:
        at=step['started_at'];n=step['decode_requests']
        if not(window['begin']<=at<window['end']) or not n or step['prefill_requests'] or step['prefill_tokens']:
            continue
        possible=[r for r in traces if r['begin']-.02<=at<r['end']+.02]
        if len(possible)<n or len({r['session'] for r in possible})!=len(possible):continue
        # Each request has independent cache salt. Ignore generated growth,
        # recurrent/MTP overhead, and select the N smallest possible histories.
        pages=sorted((r['input_tokens']+block-1)//block for r in possible)[:n]
        need=sum(pages);matched+=1;above+=need>pool
        row=dict(at=at,decode_requests=n,possible_http_requests=len(possible),
                 native_attention_pages_lower_bound=need,native_pool_pages=pool)
        if best is None or need>best['native_attention_pages_lower_bound']:best=row
    return dict(available=True,matched_steps=matched,steps_exceeding_native_pool=above,best=best,
        scope='Only observed KVMem pure public GPU steps. Conservative N-shortest HTTP superset, '
              '20ms expanded boundaries, input-only full-attention demand; not native measured batch or full256K capacity.')


def history_proof(traces,publications):
    reports=[]
    for row in traces:
        if row['kind'] not in ('warmup','workload'):continue
        matches=[r for r in publications if core.native_matches(row,r['request_id'])]
        expected=row['input_tokens']+len(row['output_ids'])-1
        passed=bool(matches) and all(r.get('history_end')==r.get('checkpoint_tokens')==expected
            and r.get('checkpoint_kind')=='M' for r in matches)
        reports.append(dict(request_id=row['request_id'],expected_history=expected,
                            publication_matches=len(matches),passed=passed))
    return dict(passed=bool(reports) and all(r['passed'] for r in reports),requests=reports,
        scope='Complete logical history and terminal checkpoint publication; payload byte correctness is a separate GPU oracle.')


def review(root):
    m=json.loads((root/'manifest.json').read_text())
    out=dict(scope=m['scope'],status=json.loads((root/'status.json').read_text()),arms={})
    ns=records(root/'native/native_samples.jsonl')
    geom=next((v for r in ns if (v:=geometry(r['raw_metrics'])) is not None),None)
    for arm in m['arms']:
        d=root/arm
        if not(d/'results.json').exists():continue
        result=json.loads((d/'results.json').read_text())
        window=json.loads((d/'tools/window.json').read_text())
        traces=json.loads((d/'tools/traces.json').read_text())
        samples=records(d/'native_samples.jsonl')
        before=max((r for r in samples if r['finished_at']<=window['begin']),key=lambda r:r['finished_at'],default=None)
        after=min((r for r in samples if r['started_at']>=window['end']),key=lambda r:r['started_at'],default=None)
        pf=None
        if before and after:
            pf=accounting.prefill_count(accounting.metrics(after['raw_metrics']))-accounting.prefill_count(accounting.metrics(before['raw_metrics']))
        item=dict(output_tokens_per_second=result['output_tokens_per_second'],output_tokens=result['output_tokens'],
            gpu=result['gpu'],preemptions=result['preemptions'],execution_profile=result['execution_profile'],
            public_prefill_outer_counter_delta=pf,
            public_prefill_outer_counter_delta_per_window_second=pf/(window['end']-window['begin']) if pf is not None else None,
            prefill_scope='Native completed-prompt credits at outer sampling boundaries; not GPU prefill kernel throughput.',
            client_token_reconciled=result['native_client_tokens_equal'])
        if arm!='native':
            capture=d/'server/capture'
            item['capacity_proof']=capacity_proof(traces,records(capture/'model_step_latency.jsonl'),geom,window)
            item['full_history_publication']=history_proof(traces,records(d/'history_publications.jsonl'))
            item['peak_cpu_authority_bytes']=max((r.get('cpu_authority_bytes',0) for r in records(d/'history_samples.jsonl')),default=None)
        out['arms'][arm]=item
    arms=out['arms']
    if 'kvmem-ready' in arms and 'native' in arms:
        gain=arms['kvmem-ready']['output_tokens_per_second']/arms['native']['output_tokens_per_second']-1
        out['candidate_over_native_gain']=gain
        out['target_25percent_observed']=gain>=.25
    if 'kvmem-ready' in arms and 'kvmem-base' in arms:
        out['candidate_over_kvmem_base_gain']=arms['kvmem-ready']['output_tokens_per_second']/arms['kvmem-base']['output_tokens_per_second']-1
    out['limitations']='Single fixed-window deployment observations. External trace replay, own output histories, not task solve-rate equivalence or pure decode kernel speed.'
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    result=review(a.run);(a.run/'reviewed.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))

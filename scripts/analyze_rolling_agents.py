"""Audit natural-stop external-trace performance without treating length as speed."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rolling_ready_analysis', ROOT/'scripts/analyze_ready_agents.py')
ready = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ready
spec.loader.exec_module(ready)


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, maximum=None)
    def quantile(q):
        pos = (len(values)-1)*q
        lo = int(pos)
        return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(pos-lo)
    return dict(count=len(values), mean=sum(values)/len(values), p50=quantile(.5),
                p95=quantile(.95), maximum=values[-1])


def summarize(traces, events, steps, refreshes, restores, window, block_tokens=32):
    lo, hi = window['begin'], window['end']
    duration = hi-lo
    if duration <= 0:
        raise ValueError('Measurement window must be positive')
    offered = [r for r in traces if r['kind']=='workload' and lo <= r['begin'] < hi]
    complete = [r for r in offered if r['status']=='completed' and r['end'] < hi]
    drained = [r for r in offered if r['status']=='completed' and r['end'] >= hi]
    tokens = sum(n for r in offered for at,n in r['timeline'] if lo <= at < hi)
    accepted_lengths = [len(r['output_ids']) for r in offered if r['status']=='completed']
    if any(r.get('generation_mode')!='natural' for r in offered):
        raise ValueError('Natural performance audit requires natural workload records')
    length_caps = sum(r.get('finish_reason')=='length' for r in offered if r['status']=='completed')
    step_time = batch_time = 0.
    weighted_histogram = {}
    for r in steps:
        if not r['decode_requests'] or r['prefill_requests'] or r['prefill_tokens']:
            continue
        overlap = max(0., min(hi,r['started_at']+r['step_ms']/1000)-max(lo,r['started_at']))
        if not overlap:
            continue
        n = r['decode_requests']
        weighted_histogram[str(n)] = weighted_histogram.get(str(n),0.)+overlap
        step_time += overlap
        batch_time += n*overlap
    refresh = [r for r in refreshes if lo <= r['refresh_started_at'] < hi]
    restore = [r for r in restores if lo <= r['started'] < hi]
    positions = dict(post_prompt=0, final_prompt_block=0, earlier_prompt=0, unmatched=0)
    for r in refresh:
        if r.get('selection')!='pressure':
            continue
        matches = [t for t in traces if ready.core.native_matches(t,r['request_id'])]
        if len(matches)!=1:
            positions['unmatched'] += 1
            continue
        remaining = matches[0]['input_tokens']-r['history_begin']
        key = 'post_prompt' if remaining <= 0 else 'final_prompt_block' if remaining <= block_tokens else 'earlier_prompt'
        positions[key] += 1
    uploaded = sum(r.get('assembly_uploaded_bytes',0) for r in refresh)
    restore_bytes = sum(r.get('uploaded_bytes',0)+r.get('model_bytes',0) for r in restore)
    refresh_wait = sum(r['refresh_finished_at']-r['refresh_started_at'] for r in refresh)
    restore_wait = sum(r['finished']-r['started'] for r in restore)
    def normalized(count):
        return dict(per_100_completed_turns=count*100/len(complete) if complete else None,
                    per_100_offered_turns=count*100/len(offered) if offered else None,
                    per_1000_window_output_tokens=count*1000/tokens if tokens else None)
    epoch_ends = [r for r in events if r.get('event')=='epoch_ended' and lo <= r['monotonic'] < hi]
    return dict(window_seconds=duration, output_tokens=tokens, output_tokens_per_second=tokens/duration,
        offered_turns=len(offered), completed_turns_in_window=len(complete),
        completed_turns_per_second=len(complete)/duration, drained_turns=len(drained),
        pending_at_deadline=len(offered)-len(complete),
        completed_in_window_output_lengths=distribution([len(r['output_ids']) for r in complete]),
        offered_cohort_final_output_lengths=distribution(accepted_lengths),
        offered_cohort_length_cap_fraction=length_caps/len(accepted_lengths) if accepted_lengths else None,
        offered_cohort_finish_reasons={reason:sum(r.get('finish_reason')==reason for r in offered)
                                     for reason in ('stop','length')},
        epoch_ends={reason:sum(r.get('reason')==reason for r in epoch_ends)
                    for reason in ('context_capacity','declared_turn_limit')},
        pure_decode_telemetry_available=bool(steps),
        pure_decode_step_seconds=step_time if steps else None,
        pure_decode_batch_time_weighted=batch_time/step_time if step_time else None,
        pure_decode_batch_seconds=weighted_histogram if steps else None,
        pressure_refresh_positions=positions, refresh_count=len(refresh),
        refresh_count_normalized=normalized(len(refresh)),
        refresh_uploaded_bytes=uploaded, refresh_bytes_normalized=normalized(uploaded),
        refresh_owner_wait_seconds=refresh_wait, refresh_wait_normalized=normalized(refresh_wait),
        restore_count=len(restore), restore_uploaded_bytes=restore_bytes,
        restore_owner_wait_seconds=restore_wait, restore_wait_normalized=normalized(restore_wait),
        scope='Only requests offered in [T0,T1); in-window output and completed-turn rates are distinct. '
              'Offered-cohort lengths include normal drain, without cancellation. Maintenance is counted '
              'by start time and owner spans can overlap; their sum is not GPU idle. Pure-decode weighting '
              'uses GPU-event duration with approximate CPU-start boundary clipping, excluding mixed prefill/decode; native unavailable '
              'is not zero. Pressure position uses logical history relative to full prompt, not kernel labels. '
              'Frozen external messages do not establish on-policy task quality.')


def review(root):
    result = ready.review(root)
    for arm,item in result['arms'].items():
        d = root/arm
        window = json.loads((d/'tools/window.json').read_text())
        traces = json.loads((d/'tools/traces.json').read_text())
        capture = d/'server/capture'
        natural = summarize(traces,ready.records(d/'tools/events.jsonl'),
            ready.records(capture/'model_step_latency.jsonl'),
            ready.records(capture/'bounded_refresh.jsonl'),ready.records(capture/'async_restore.jsonl'),window)
        if natural['output_tokens'] != item['output_tokens']:
            raise ValueError(f'{arm}: natural audit output accounting mismatch')
        item['natural_workload'] = natural
    arms=result['arms']
    candidate=arms.get('rolling-candidate-v2')
    if candidate:
        result['natural_comparisons']={}
        for name in ('native','pipeline-base'):
            if name not in arms:
                continue
            before,after=arms[name]['natural_workload'],candidate['natural_workload']
            result['natural_comparisons'][name]={
                key+'_relative_change':after[key]/before[key]-1 if before[key] else None
                for key in ('output_tokens_per_second','completed_turns_per_second')}
    result['interpretation'] = ('Token-rate gains alone are insufficient: compare completed-turn rate, '
        'output lengths, actual batch and normalized maintenance. Longer natural outputs and fewer epoch '
        'restarts can change rates. Single order/window is provisional; paired reverse-order repetition '
        'and observed capacity evidence are required before a robust performance claim.')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = review(args.run)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2))

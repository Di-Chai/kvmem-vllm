"""Compare eight original agents alone versus together using equal per-agent windows.

Server/runtime/budgets stay identical. Serial runs each source task alone for60s;
parallel runs all eight for60s. Every slice gets its own two-turn warmup, normal
drain and explicit session release. The serial controller envelope includes
intermediate warmup/drain and is NEVER the performance denominator.
"""
import argparse
import asyncio
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/benchmark_agent_concurrency.py'
ARMS = ('serial', 'parallel')
DURATION = 60
KIND = 'serial_parallel_agent_windows_v1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


chain = module('agent_concurrency_chain', ROOT/'scripts/benchmark_capacity_chain.py')
cap, ready, work = chain.cap, chain.ready, chain.work
audit = module('agent_concurrency_audit', ROOT/'scripts/analyze_ready_agents.py')


def prepare(args):
    root, base = args.output.resolve(), args.reference.resolve()
    cap.require(not root.exists(), 'Use a fresh directory')
    gate = module('agent_concurrency_reference', base/'frozen/scripts/benchmark_capacity_chain.py')
    original = gate.verify(base)
    arm = 'rolling-candidate-v2'
    cap.require(original['arms'] == [arm] and
        cap.read(base/'status.json')['status'] == 'completed_pending_review' and
        cap.read(base/arm/'deployment-acceptance.json')['accepted'], 'Require accepted NVFP4 reference')
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(base/'frozen', root/'frozen', ignore=ignore)
    shutil.copyfile(ROOT/ENTRY, root/'frozen'/ENTRY)
    for name in ('fixture.json', 'installed-native-identity.json'):
        shutil.copyfile(base/name, root/name)
    profile = dict(cap.read(base/(arm+'.json')), port=args.port)
    for name in ARMS:
        shutil.copytree(base/'runtime'/arm, root/'runtime'/name, ignore=ignore)
        cap.save(root/(name+'.json'), profile)
    m = copy.deepcopy(original)
    m.update(experiment_kind=KIND, created_at=time.time(), arms=list(ARMS), port=args.port,
        concurrency_reference=str(base), concurrency_reference_sha256=cap.sha(base/'manifest.json'),
        source_sha256=cap.hashes(root/'frozen'),
        runtime_sha256={name:cap.hashes(root/'runtime'/name) for name in ARMS},
        profiles_sha256={name:cap.sha(root/(name+'.json')) for name in ARMS},
        workload=dict(original['workload'], duration_seconds=DURATION),
        scope='Original eight G16 tasks each alone for60s then all eight for60s. '
              'Same server max_num_seqs8, model, CPU history, K64/G32/256Ki, sampling and recorded '
              'tool waits. Two warmup turns per slice; normal drain; no clipping. Not on-policy '
              'DeepSWE quality, fixed output work, or a repeated randomized trial. '
              'Serial rate sums output over8*60s; parallel rate uses60s.')
    cap.save(root/'manifest.json', m)
    cap.save(root/'status.json', dict(status='prepared', at=time.time()))


def verify(root):
    m = ready.verify(root)
    cap.require(m['experiment_kind'] == KIND and m['arms'] == list(ARMS), 'Wrong experiment')
    base = Path(m['concurrency_reference'])
    cap.require(cap.sha(base/'manifest.json') == m['concurrency_reference_sha256'], 'Reference changed')
    original = cap.read(base/'manifest.json')
    cap.require(m['fixture_sha256'] == original['fixture_sha256'], 'Original eight tasks changed')
    expected = dict(cap.read(base/'rolling-candidate-v2.json'), port=m['port'])
    for arm in ARMS:
        cap.require(cap.read(root/(arm+'.json')) == expected, 'Serving configuration changed')
        cap.require(m['runtime_sha256'][arm] == original['runtime_sha256']['rolling-candidate-v2'],
                    'Production runtime changed')
    cap.require(m['workload'] == dict(original['workload'], duration_seconds=DURATION),
                'Sampling, warmup, natural limits or tool waits changed')
    cap.require(m['budget'] == original['budget'], 'CPU budget changed')
    chain.installed_native(root/'installed-native-identity.json')
    return m


def remap(rows, user):
    return [dict(r, agent=user) if 'agent' in r else dict(r) for r in rows]


def decode_intervals(steps, public, traces, window, guard=.5):
    """Intersect confirmed no-prefill runs with at least one generating client.

    Consecutive decode dispatches alone do not exclude all-agents-in-tools idle
    gaps. The stream must have delivered its first token and remain in flight.
    Exclude each interval's left-edge event so the token opening an interval is
    not credited to its subsequent duration.
    """
    lo, hi = window['begin'], window['end']
    events = [(at,n) for r in traces if r['kind']=='workload' for at,n in r['timeline']]
    active=[]
    for a,b in sorted((r['timeline'][0][0],r['end']) for r in traces
                      if r['kind']=='workload' and r['timeline']):
        if active and a<=active[-1][1]:active[-1]=(active[-1][0],max(b,active[-1][1]))
        else:active.append((a,b))
    runs, start = [], None
    for i,s in enumerate(steps):
        pure = s['decode_requests']>0 and not s['prefill_requests'] and not s['prefill_tokens']
        if pure and start is None: start=i
        if not pure and start is not None:
            runs.append((start,i)); start=None
    windows=[]
    for ia,ib in runs:
        a,b=max(lo,steps[ia]['started_at']),min(hi,steps[ib]['started_at'])
        if b-a < 2+2*guard: continue
        confirm=next((r for r in public if a<=r['at']<b and
            r['completed']['public_prefill_tokens']==r['submitted']['public_prefill_tokens']),None)
        if confirm is None: continue
        a=max(a+guard,confirm['at']); b-=guard
        if b-a < 2: continue
        for x,y in active:
            left,right=max(a,x),min(b,y)
            if right-left<2:continue
            area=sum(max(0,min(right,steps[j+1]['started_at'])-max(left,steps[j]['started_at']))*
                     steps[j]['decode_requests'] for j in range(ia,ib))
            windows.append(dict(begin=left,end=right,seconds=right-left,
                output_tokens=sum(n for at,n in events if left<at<=right and at<hi),
                batch_request_seconds=area))
    seconds=sum(r['seconds'] for r in windows); tokens=sum(r['output_tokens'] for r in windows)
    return dict(windows=windows,seconds=seconds,output_tokens=tokens,
        tokens_per_second=tokens/seconds if seconds else None,
        mean_decode_batch=sum(r['batch_request_seconds'] for r in windows)/seconds if seconds else None,
        scope='Post-hoc public-decode intervals>=2s,0.5s edge exclusion and prior-prefill completion '
              'fence; additionally intersect with union of first-token-to-stream-end intervals. '
              'Excludes no-client/tool-wait gaps, includes maintenance with a client generating. '
              'Not fixed-batch kernel throughput; batch is dispatch-count weighted by wall intervals.')


def summarize(root):
    result={}
    for arm in ARMS:
        d=root/arm; slices=[d/'tools'/f'agent-{i}' for i in range(8)] if arm=='serial' else [d/'tools']
        steps=ready.records(d/'server/capture/model_step_latency.jsonl')
        public=ready.records(d/'server/capture/public_work.jsonl')
        gpu=ready.records(d/'gpu_samples.jsonl'); items=[]
        for index,path in enumerate(slices):
            r=cap.read(path/'results.json'); rows=cap.read(path/'traces.json');w=cap.read(path/'window.json')
            n=1 if arm=='serial' else 8
            first=[min((r for r in rows if r['kind']=='warmup' and r['agent']==i),key=lambda r:r['begin']) for i in range(n)]
            cold_seconds=max(r['timeline'][0][0] for r in first)-min(r['begin'] for r in first)
            cold_tokens=sum(r['input_tokens'] for r in first)
            assert all(sum(n for _,n in r['timeline'])==len(r['output_ids'])==r['usage']['completion_tokens'] for r in rows)
            tokens=sum(n for r in rows if r['kind']=='workload' for at,n in r['timeline'] if w['begin']<=at<w['end'])
            assert tokens==r['output_tokens']
            items.append(dict(source_agent=index if arm=='serial' else None,seconds=w['end']-w['begin'],
                output_tokens=tokens,output_tokens_per_second=tokens/(w['end']-w['begin']),
                completed_turns=r['requests_completed_in_window'],drain_seconds=r['drain_seconds'],
                cold_input_tokens=cold_tokens,cold_seconds=cold_seconds,
                actual_prefill_tokens=sum(s['prefill_tokens'] for s in steps if w['begin']<=s['started_at']<w['end']),
                decode=decode_intervals(steps,public,rows,w),
                gpu=ready.bench.gpu_summary(gpu,w['begin'],w['end'])))
        seconds=sum(r['seconds'] for r in items); tokens=sum(r['output_tokens'] for r in items)
        dec_seconds=sum(r['decode']['seconds'] for r in items);dec_tokens=sum(r['decode']['output_tokens'] for r in items)
        result[arm]=dict(slices=items,seconds=seconds,output_tokens=tokens,output_tokens_per_second=tokens/seconds,
            per_agent_output_tokens_per_second=tokens/seconds/(8 if arm=='parallel' else 1),
            completed_turns=sum(r['completed_turns'] for r in items),
            cold_prefill_tokens_per_second=sum(r['cold_input_tokens'] for r in items)/sum(r['cold_seconds'] for r in items),
            decode_seconds=dec_seconds,decode_tokens=dec_tokens,
            decode_tokens_per_second=dec_tokens/dec_seconds if dec_seconds else None)
    result['aggregate_throughput_ratio']=result['parallel']['output_tokens_per_second']/result['serial']['output_tokens_per_second']
    result['scope']=('Equal60-second opportunity per source agent. Serial pooled480s versus parallel60s. '
        'Natural outputs/context evolution differ; one serial-first order, no task-quality claim. '
        'Cold prefill and drain excluded from formal output rate and separately reported.')
    cap.save(root/'comparison.json',result)
    return result


async def run(args):
    root=args.output.resolve();m=verify(root)
    cap.require(cap.read(root/'status.json')['status']=='prepared','No automatic retry')
    original=work.workload
    async def measured(client,profile,fixture,directory,values):
        if values.run_tag=='parallel':
            return await chain.released_workload(original,client,profile,fixture,directory,values,m['release_qualification'])
        directory.mkdir(parents=True)
        all_rows,all_events,phases=[],[],[]
        for i,plan in enumerate(fixture['agent_plans']):
            phase=directory/f'agent-{i}'
            cap.save(root/'progress.json',dict(arm='serial',source_agent=i,phase='warming_or_measuring',at=time.time()))
            selected=SimpleNamespace(**{**vars(values),'concurrency':1,'run_tag':f'serial-{i}'})
            one=dict(fixture,agent_plans=[plan])
            before=await values.native_quiescent(stage=f'before-{i}',not_before=time.monotonic())
            r=await chain.released_workload(original,client,profile,one,phase,selected,m['release_qualification'])
            after=await values.native_quiescent(stage=f'after-{i}',not_before=time.monotonic())
            a,b=before['native_metrics']['values'],after['native_metrics']['values']
            cap.require(b['vllm:generation_tokens_total']-a['vllm:generation_tokens_total']==r['total_stream_output_tokens']
                and b['vllm:num_preemptions_total']==0,'Serial slice token accounting/preemption failed')
            cap.save(phase/'native-reconciliation.json',dict(passed=True,before=before,after=after))
            all_rows.extend(remap(cap.read(phase/'traces.json'),i))
            all_events.extend(remap(ready.records(phase/'events.jsonl'),i))
            phases.append(r)
        cap.save(directory/'traces.json',all_rows)
        with (directory/'events.jsonl').open('x') as stream:
            for row in sorted(all_events,key=lambda r:r.get('monotonic',0)):stream.write(json.dumps(row)+'\n')
        begin,end=min(r['begin'] for r in all_rows),max(r['end'] for r in all_rows)
        cap.save(directory/'window.json',dict(begin=begin,end=end,t0_monotonic=begin,t1_monotonic=end,
            scope='Controller envelope including all cold preparations and drains; NEVER used as formal performance window'))
        return dict(status='completed',drained=True,explicit_session_release_verified=True,
            total_stream_output_tokens=sum(r['total_stream_output_tokens'] for r in phases),
            slice_paths=[f'agent-{i}' for i in range(8)],scope='Only per-slice60s windows define serial performance')
    work.workload=measured
    task=asyncio.current_task();loop=asyncio.get_running_loop()
    for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,task.cancel)
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for arm in ARMS:
                cap.require(verify(root)==m,'Frozen input changed')
                cap.save(root/'progress.json',dict(arm=arm,phase='starting',at=time.time()))
                await ready.run_arm(root,arm,m,prelaunch_check=chain.prelaunch_admission)
                await asyncio.to_thread(chain.post_stop_proof,root,arm)
                d=root/arm;rows=cap.read(d/'tools/traces.json')
                cap.save(d/'own-history-proof.json',chain.own_history_proof(cap.read(root/'fixture.json'),rows,ready.records(d/'tools/events.jsonl')))
                history=audit.history_proof(rows,ready.records(d/'history_publications.jsonl'))
                cap.require(history['passed'],'Full CPU history failed')
                cap.save(d/'full-history-proof.json',history)
                verdict=chain.deployment_verdict(root,arm,m)
                cap.save(d/'deployment-acceptance.json',verdict)
                cap.require(verdict['accepted'],'GPU headroom/preemption failed')
        summarize(root)
        cap.save(root/'status.json',dict(status='completed_pending_review',at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise
    finally:
        work.workload=original
        for sig in (signal.SIGTERM,signal.SIGINT):loop.remove_signal_handler(sig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare-only',action='store_true');g.add_argument('--run-prepared',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    p.add_argument('--port',type=int,default=18590)
    args=p.parse_args()
    if args.prepare_only:prepare(args)
    else:asyncio.run(run(args))


if __name__=='__main__':main()

"""Paired four-agent qualification before a fresh formal DeepSWE first16 run.

Fixed input/tool IDs, output lengths and tool waits; each turn appends its own
accepted output IDs. This synthetic test measures scheduling, not task quality.
Reuses the owned-server/foreign-GPU guard from qualify_coordination.
"""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import statistics


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('agent_coordination', ROOT/'scripts/qualify_coordination.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
spec = importlib.util.spec_from_file_location('agent_lifecycle', ROOT/'scripts/agent_lifecycle_checks.py')
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)
ARMS = {
    'A0': {},
    'A1': {'active_resident_ttl_ms':5000., 'active_resident_max_fraction':.5},
    'P1': {'active_resident_pressure_only':True, 'active_resident_max_fraction':.5, 'offline_agent_policy':True},
    'Pttl': {'active_resident_ttl_ms':5000., 'active_resident_max_fraction':.5, 'offline_agent_policy':True},
    # Correctness-only fault injection: four 32K views exceed the idle cap.
    # Pressure-only retention separates revocation from separately tested expiry.
    'Ppressure': {'active_resident_pressure_only':True, 'active_resident_max_fraction':.075, 'offline_agent_policy':True},
    'A2': {'active_resident_ttl_ms':5000., 'active_resident_max_fraction':.5, 'prefill_batch_tokens':512},
    'A3': {'active_resident_ttl_ms':5000., 'active_resident_max_fraction':.5, 'prefill_batch_tokens':512,
           'maintenance_transfer_bytes_per_second':8 * 1024**3, 'maintenance_transfer_burst_bytes':64 * 1024**2},
}


def source_hashes():
    paths = [Path(__file__), ROOT/'scripts/qualify_coordination.py', ROOT/'scripts/check_coordination_restore.py',
             ROOT/'scripts/agent_lifecycle_checks.py',
             *sorted((ROOT/'src/vllm_kvmem').glob('*.py')),
             *sorted((ROOT/'src/vllm_kvmem').glob('*.so'))]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


async def workload(client, profile, fixture, case, directory, args):
    import time
    sessions = [f'coordination-{i}' for i in range(args.concurrency)]
    directory.mkdir()
    await asyncio.gather(*(q.call(client, profile, [fixture['eos']], s, 1, query=False) for s in sessions))
    before_metrics = (await client.get(f"http://127.0.0.1:{profile['port']}/metrics")).text
    capture = directory.parent/'server/capture'
    before_resources = read_json(capture/'scheduler_resources.json')
    started = time.perf_counter()
    results = []
    barrier = asyncio.Barrier(args.concurrency) if args.retirement_barrier else None
    mixed_start = bool(getattr(args, 'coalescing_mixed_start', False))
    first_token_events = [asyncio.Event() for _ in range(args.concurrency-1)] if mixed_start else []

    async def agent(i, session):
        length=args.context_tokens * (2 if mixed_start and i==args.concurrency-1 else 1)
        tokens = fixture['prompt'][:length]
        if args.distinct_users:
            prefix=fixture['user_prefixes'][i]
            tokens=prefix+tokens[len(prefix):]
        for turn in range(args.turns if case == 'agents' else 1):
            if turn==0 and mixed_start and i==args.concurrency-1:
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in first_token_events)),timeout=300)
            if turn:
                # Includes short hot returns, a long result and a TTL miss.
                wait = fixture['waits'][i][(turn-1) % len(fixture['waits'][i])]
                await asyncio.sleep(wait)
                count = fixture['tool_sizes'][(turn-1) % len(fixture['tool_sizes'])]
                tool=fixture['tool'][:count]
                if args.distinct_users:
                    prefix=fixture['user_prefixes'][i]
                    tool=prefix+tool[len(prefix):]
                tokens = tokens + tool
            row = await q.call(client, profile, tokens, session, args.output_tokens, turn=turn,
                ready=first_token_events[i] if mixed_start and turn==0 and i<len(first_token_events) else None)
            results.append(row)
            q.save(directory/'progress.json', {'completed_requests':len(results),
                   'output_tokens':sum(len(r['output_ids']) for r in results), 'at':time.time()})
            q.save(directory/f'agent-{i}-turn-{turn}.json', row)
            tokens = tokens + row['output_ids']
            if turn == 0 and barrier is not None:
                if await barrier.wait() == 0:
                    state = await lifecycle.quiescent(capture)
                    q.save(directory/'retirement-barrier.json', state)
                await barrier.wait()

    if case == 'mixed':
        events = [asyncio.Event() for _ in sessions[:-1]]
        async with asyncio.TaskGroup() as group:
            peers = [group.create_task(q.call(client, profile, fixture['prompt'][:32768], session,
                args.output_tokens*4, ready=events[i])) for i,session in enumerate(sessions[:-1])]
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in events)), timeout=300)
            target = group.create_task(q.call(client, profile, fixture['prompt'][:args.context_tokens], sessions[-1],
                                             args.output_tokens))
        results = [t.result() for t in peers] + [target.result()]
    else:
        await asyncio.gather(*(agent(i,s) for i,s in enumerate(sessions)))
    finished = max(r['end'] for r in results)
    q.save(directory/'traces.json', results)
    summary = q.summarize(results, finished-started)
    if case == 'decode':
        begin, end = max(r['timeline'][0][0] for r in results), min(r['timeline'][-1][0] for r in results)
        if end > begin:
            n = sum(n for r in results for t,n in r['timeline'] if begin < t <= end)
            summary['common_decode'] = {'seconds':end-begin, 'tokens':n, 'tokens_per_second':n/(end-begin)}
    after_metrics = (await client.get(f"http://127.0.0.1:{profile['port']}/metrics")).text
    (directory/'metrics-before.txt').write_text(before_metrics)
    (directory/'metrics-after.txt').write_text(after_metrics)
    q.save(directory/'resources-before.json', before_resources)
    q.save(directory/'resources-after.json', read_json(capture/'scheduler_resources.json'))
    q.save(directory/'diagnostics.json', q.collect_diagnostics(directory.parent/'server', started=started, finished=finished))
    # Cleanup is excluded from workload time, but required to release complete
    # CPU histories before the next case. No capacity-driven history eviction.
    await asyncio.gather(*(q.call(client, profile, [fixture['eos']], s, 1, query=False) for s in sessions))
    if args.verify_restores:
        q.save(directory/'resources-cleanup.json', await lifecycle.quiescent(capture))
    if args.lifecycle_checks and profile['kvmem']['offline_agent_policy']:
        await lifecycle.exercise(client, profile, fixture, capture, directory.parent/'lifecycle', q)
    archive = read_json(capture/'archive.json')
    if archive.get('disk_read_bytes',0) or archive.get('disk_write_bytes',0):
        raise RuntimeError('CPU-only qualification unexpectedly used KV disk I/O')
    q.save(directory/'results.json', summary)
    return summary


def comparisons(measured, arms, cases):
    result = {}
    for arm in arms:
        if arm == 'A0': continue
        pairs = sorted(i for i,a in measured if a == arm and (i,'A0') in measured)
        for case in cases:
            ratios = [measured[i,arm][case]['output_tokens_per_second'] /
                      measured[i,'A0'][case]['output_tokens_per_second'] for i in pairs]
            gaps = [measured[i,arm][case]['output_event_gap_p95_seconds'] /
                    measured[i,'A0'][case]['output_event_gap_p95_seconds'] for i in pairs]
            result[arm+'/'+case] = {'pairs':len(pairs), 'throughput_ratios':ratios,
                'throughput_ratio_median':statistics.median(ratios) if ratios else None,
                'event_gap_p95_ratio_median':statistics.median(gaps) if gaps else None,
                'consistent_improvement':len(pairs)>=3 and all(r>1 for r in ratios),
                'scope':'Paired synthetic agent rounds; requires formal first16 validation separately'}
            if case == 'decode':
                common = [measured[i,arm][case]['common_decode']['tokens_per_second'] /
                          measured[i,'A0'][case]['common_decode']['tokens_per_second'] for i in pairs
                          if measured[i,arm][case].get('common_decode') and measured[i,'A0'][case].get('common_decode')]
                result[arm+'/'+case].update(common_decode_ratios=common,
                    common_decode_ratio_median=statistics.median(common) if common else None)
    return result


async def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('pair-*')):
        raise ValueError('Use a fresh output directory, never overwrite measurements')
    from vllm_kvmem.serve import build_launch
    if args.verify_restores and args.cases != ['agents']:
        raise ValueError('Byte-oracle qualification uses only the multi-round agents case')
    prefill_batch_tokens = getattr(args, 'prefill_batch_tokens', 0)
    if type(prefill_batch_tokens) is not int or prefill_batch_tokens < 0:
        raise ValueError('Shared prefill qualification budget must be a nonnegative integer (0 disables)')
    if prefill_batch_tokens and (not args.verify_restores or
            any(arm not in ('P1', 'Ppressure') for arm in args.arms) or
            getattr(args, 'phase_policy', False) or getattr(args, 'prefill_coalescing', False)):
        raise ValueError('Shared prefill qualification requires continuous P1/Ppressure byte oracles without phase/coalescing')
    coalescing=bool(getattr(args,'prefill_coalescing',False))
    if coalescing and (not args.verify_restores or getattr(args,'phase_policy',False) or
            any(arm not in ('P1','Ppressure') for arm in args.arms)):
        raise ValueError('Coalescing qualification requires continuous P1/Ppressure byte oracles')
    if getattr(args,'coalescing_mixed_start',False) and (not coalescing or args.arms!=['P1'] or
            args.retirement_barrier or args.context_tokens!=32768 or args.output_tokens<2048):
        raise ValueError('Coalescing mixed start requires the fixed long-decode P1 byte workload')
    if getattr(args, 'phase_policy', False) and (not args.verify_restores or
            any(arm not in ('P1', 'Ppressure') for arm in args.arms)):
        raise ValueError('Phase qualification requires byte oracles and coordinated P1/Ppressure arms')
    if (getattr(args, 'phase_policy', False) and 'P1' in args.arms and
            getattr(args, 'retirement_barrier', False) and getattr(args, 'distinct_users', False)):
        raise ValueError('Phase P1 long-wait isolation and zero-wait pressure barrier require separate runs')
    if (args.lifecycle_checks or args.retirement_barrier or args.distinct_users or {'Ppressure','Pttl'} & set(args.arms)) and not args.verify_restores:
        raise ValueError('Lifecycle fault injection must not enter performance measurements')
    minimum=65536 if any(ARMS[a].get('active_resident_ttl_ms',0) for a in args.arms) else 32768
    if args.lifecycle_checks and args.context_tokens < minimum:
        raise ValueError('Lifecycle cancellation requires a stable old Query and sufficient fixture tokens')
    if 'Ppressure' in args.arms and (not args.retirement_barrier or args.context_tokens != 32768):
        raise ValueError('Pressure correctness requires the fixed 32K retirement-barrier workload')
    base = json.loads(args.config.read_text())
    if prefill_batch_tokens:
        from vllm_kvmem.config import KVMemConfig
        block = base['kvmem'].get('block_tokens', KVMemConfig.block_tokens)
        if type(block) is not int or block <= 0 or prefill_batch_tokens % block:
            raise ValueError('Shared prefill qualification budget must be positive and block aligned')
    if (args.concurrency != base['max_num_seqs'] or not base['kvmem'].get('cpu_history_coordination') or
            not base['kvmem'].get('async_restore') or base['kvmem'].get('archive_backend') != 'cpu'):
        raise ValueError('Use the existing coordinated CPU-history profile and its actual concurrency')
    profiles = {}
    for arm in args.arms:
        profile = json.loads(json.dumps(base))
        profile['kvmem'].update(prefill_batch_tokens=0, active_resident_ttl_ms=0., active_resident_pressure_only=False,
            maintenance_transfer_bytes_per_second=0, adaptive_prefill=False, step_latency_interval=4,
            offline_agent_policy=False)
        profile['kvmem'].update(ARMS[arm])
        if prefill_batch_tokens:
            # Apply after arm defaults; an explicit correctness cap must reach
            # the real launcher and saved profile without being reset to zero.
            profile['kvmem']['prefill_batch_tokens'] = prefill_batch_tokens
        profile['kvmem']['offline_phase_policy'] = bool(getattr(args, 'phase_policy', False))
        profile['kvmem']['offline_prefill_coalescing'] = coalescing
        if coalescing:
            profile['kvmem']['step_latency_interval']=1
        profile['verify_session_restore'] = args.verify_restores
        build_launch(profile, args.config.resolve().parent, output/'validation'/arm)
        q.save(output/f'{arm}.json', profile)
        profiles[arm] = profile
    if not (output/'fixture.json').exists():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base['tokenizer'], local_files_only=True)
        text = ''.join(f'Record {i}: worker {i%8} observed value {(i*104729)%1000003}. Keep the evidence.\n'
                       for i in range(20000))
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < args.context_tokens + 16384:
            raise ValueError('Insufficient synthetic context')
        prompt_length=args.context_tokens*(2 if getattr(args,'coalescing_mixed_start',False) else 1)
        q.save(output/'fixture.json', {'prompt':ids[:prompt_length], 'tool':ids[-16384:],
            'tool_sizes':[512,512,16384,512],
            'waits':[[.05,.25,1.,6.], [.25,1.,6.,.05], [1.,6.,.05,.25], [6.,.05,.25,1.]],
            'eos':tokenizer.eos_token_id})
    fixture = read_json(output/'fixture.json')
    if args.distinct_users and 'user_prefixes' not in fixture:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(base['tokenizer'],local_files_only=True)
        fixture['user_prefixes']=[tokenizer.encode(f'Unique tenant {i}: evidence identifier USER_{i}.\n',
            add_special_tokens=False) for i in range(args.concurrency)]
        if (len({tuple(p) for p in fixture['user_prefixes']})!=args.concurrency or
                max(map(len,fixture['user_prefixes']))>min(args.context_tokens,min(fixture['tool_sizes']))):
            raise ValueError('Distinct tenant identifiers do not fit the fixed workload')
        q.save(output/'fixture.json',fixture)
    if args.retirement_barrier:
        fixture['waits'] = [[0.] for _ in range(args.concurrency)]
        fixture['tool_sizes'] = [512]
        q.save(output/'fixture.json', fixture)
    q.source_hashes = source_hashes
    q.workload = workload
    q.save(output/'manifest.json', {'kind':'agent_scheduling', 'arms':args.arms, 'cases':args.cases,
        'pairs':args.pairs, 'source_sha256':source_hashes(), 'config':str(args.config.resolve()),
        'fixture_sha256':q.digest(fixture), 'workload':{k:v for k,v in vars(args).items() if k not in ('config','output')},
        'scope':'Fixed synthetic multi-turn loads; not DeepSWE scores. Accepted output IDs remain exact per arm.'})
    if args.prepare_only:
        q.save(output/'status.json', {'status':'prepared', 'gpu_started':False})
        return
    measured = {}
    lock = open('/tmp/kvmem-coordination-qualification.lock', 'w')
    try:
        # Wait cooperatively for the existing benchmark; never terminate it.
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                q.save(output/'status.json', {'status':'waiting_for_owned_benchmark_lock'})
                await asyncio.sleep(10)
        for pair in range(args.pairs):
            order = list(args.arms) if pair % 2 == 0 else list(reversed(args.arms))
            for arm in order:
                measured[pair,arm] = await q.run_arm(args, profiles[arm], fixture, output, arm, pair)
                diagnostics = read_json(output/f'pair-{pair:02d}-{arm}/diagnostics.json')
                if arm != 'A0' and 'agents' in args.cases:
                    files = list((output/f'pair-{pair:02d}-{arm}/server').rglob('async_restore.jsonl'))
                    rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
                    if not any(r.get('resident_active_tokens',0) for r in rows):
                        raise RuntimeError('Hot arm never restored a resident active view')
                q.save(output/'comparison.json', comparisons(measured,args.arms,args.cases))
        q.save(output/'status.json', {'status':'completed', 'arms':args.arms, 'pairs':args.pairs})
        if args.verify_restores:
            checker_spec = importlib.util.spec_from_file_location('agent_restore_gate', ROOT/'scripts/check_coordination_restore.py')
            checker = importlib.util.module_from_spec(checker_spec)
            checker_spec.loader.exec_module(checker)
            q.save(output/'restore-gate.json', checker.check(output))
    except BaseException as error:
        q.save(output/'status.json', {'status':'failed', 'error':repr(error)})
        raise
    finally:
        lock.close()


async def entry(args):
    task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    await main(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--verify-restores',action='store_true')
    p.add_argument('--prefill-batch-tokens',type=int,default=0,
                   help='Correctness-only: shared ordinary-prefill cap for continuous P1/Ppressure (default: unchanged arms)')
    p.add_argument('--phase-policy',action='store_true',
                   help='Correctness-only: enable the public phase gate in P1/Ppressure arms')
    p.add_argument('--prefill-coalescing',action='store_true',help='Correctness-only: continuous prefill coalescing')
    p.add_argument('--coalescing-mixed-start',action='store_true',
                   help='Correctness-only: start a 64K target after three 32K peers enter long decode')
    p.add_argument('--lifecycle-checks',action='store_true')
    p.add_argument('--retirement-barrier',action='store_true')
    p.add_argument('--distinct-users',action='store_true',help='Correctness-only: distinct tenant evidence in each history')
    p.add_argument('--arms',nargs='+',choices=ARMS,default=['A0','P1'],
                   help='Default: baseline and unified offline policy. A1/A2/A3 are explicit component diagnostics.')
    p.add_argument('--cases',nargs='+',choices=('decode','agents','mixed'),default=['decode','agents','mixed'])
    p.add_argument('--pairs',type=int,default=3)
    p.add_argument('--concurrency',type=int,choices=(4,),default=4)
    p.add_argument('--context-tokens',type=int,default=141870)
    p.add_argument('--output-tokens',type=int,default=1024)
    p.add_argument('--turns',type=int,default=5)
    p.add_argument('--idle-seconds',type=float,default=120.)
    a = p.parse_args()
    if ('A0' not in a.arms and not a.verify_restores) or a.pairs < 1 or a.turns < 2 or min(a.context_tokens,a.output_tokens) <= 0:
        p.error('Require A0 for performance, positive lengths/pairs and at least two tool turns')
    asyncio.run(entry(a))

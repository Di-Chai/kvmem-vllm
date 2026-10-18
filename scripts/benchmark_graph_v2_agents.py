"""Freeze and run one or three alternating V2 D natural fixed8-agent pairs."""
import argparse
import asyncio
import difflib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import graph_v2_agent_checks as checks
import qualify_session_release as release

ROOT = Path(__file__).resolve().parents[1]
cap, ready, work, chain = checks.cap, checks.ready, checks.work, checks.chain
require, read, save, sha = cap.require, cap.read, cap.save, cap.sha
DEFAULT_PROFILE = ROOT/'artifacts/graph-overhead-20261018/gpu-r2/after/profile.json'
DEFAULT_FIXTURE = ROOT/'artifacts/calibrated-fp8-agents-20261018-candidate-r3/fixture.json'
DEFAULT_QUALIFICATIONS = ROOT/'artifacts/graph-v2-sessions-20261018'


def prepare(args):
    root = args.output.resolve()
    require(not root.exists(), 'Use a new output directory; no overwrite or implicit retry')
    root.mkdir(parents=True)
    save(root/'status.json', dict(status='preparing', at=time.time()))
    try:
        candidate = args.candidate or 'raw-k-cache'
        jobs = checks.job_order(args.pairs)
        configs = checks.profiles(read(args.profile), args.port, candidate)
        baseline, changed = configs.values()
        proof = checks.fixture_proof(args.fixture)
        identity = checks.pairs.shared.model_identity(baseline)
        from transformers import AutoTokenizer
        require(AutoTokenizer.from_pretrained(baseline['tokenizer'], local_files_only=True).eos_token_id ==
                read(args.fixture)['eos'], 'Original fixture EOS differs from the actual tokenizer')
        shutil.copyfile(args.fixture, root/'fixture.json')
        frozen = root/'frozen'
        for name in ('src','scripts','csrc'):
            shutil.copytree(ROOT/name, frozen/name, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        vendor = frozen/'third_party/qw3'
        for name in ('include','src'):
            shutil.copytree(ROOT/'third_party/qw3'/name, vendor/name)
        shutil.copyfile(ROOT/'third_party/qw3/PROVENANCE.json', vendor/'PROVENANCE.json')
        sources = cap.hashes(frozen)
        require(any(k.startswith('src/vllm_kvmem/_native') and k.endswith('.so') for k in sources),
                'Current native extension is absent from frozen source')
        qualifications = {}
        for mode, source in (('sync',args.sync_qualification),('async',args.async_qualification)):
            target = root/'qualifications'/mode
            manifest = read(source/'manifest.json')
            for name in (*checks.QUALIFICATION_FILES, *('runtime/'+k for k in manifest['runtime_sha256'])):
                destination = target/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source/name, destination)
            report = checks.qualification_proof(target, baseline, sources, mode == 'sync', runtime_root=frozen)
            report['original_path'] = str(source.resolve())
            qualifications[mode] = report
            for name in report['source_extensions']:
                diff = ''.join(difflib.unified_diff((target/'runtime'/name).read_text().splitlines(True),
                    (frozen/name).read_text().splitlines(True), fromfile='qualified/'+name, tofile='proposed/'+name))
                destination = target/'source_extensions'/Path(name).with_suffix('.diff')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(diff)
        save(root/'qualification-checklist.json', qualifications)
        from vllm_kvmem.serve import build_launch
        profile_hashes, runtime_hashes, launches = {}, {}, {}
        (root/'runtime').mkdir()
        for job in jobs:
            name = job['id']
            profile = baseline if job['role'] == 'baseline' else changed
            save(root/(name+'.json'), profile)
            launch = build_launch(profile, root, root/name/'server')
            launches[name] = {key:launch[key] for key in ('profile','vllm_argv')}
            profile_hashes[name] = sha(root/(name+'.json'))
            # Every fresh service uses exactly the same immutable snapshot.
            # Output/capture locations are outside this shared source tree.
            (root/'runtime'/name).symlink_to(Path('../frozen'), target_is_directory=True)
            runtime_hashes[name] = sources
        save(root/'expected-launches.json', launches)
        patch = subprocess.check_output(['git','diff','HEAD','--','src','csrc','third_party','scripts'],cwd=ROOT)
        (root/'source.diff').write_bytes(patch)
        status = subprocess.check_output(['git','status','--porcelain','--','src','csrc','third_party','scripts'],cwd=ROOT,text=True)
        save(root/'manifest.json', dict(kind='ready_agents_v1', experiment_kind=checks.SCHEMA, created_at=time.time(),
            source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_dirty=bool(status.strip()), source_worktree_status=status, source_diff_sha256=sha(root/'source.diff'),
            source_sha256=sources, runtime_sha256=runtime_hashes, profiles_sha256=profile_hashes,
            expected_launches_sha256=sha(root/'expected-launches.json'), fixture_sha256=sha(root/'fixture.json'),
            fixture_source=str(args.fixture.resolve()), fixture_proof=proof,
            qualification_sha256=cap.hashes(root/'qualifications'), qualification_checklist_sha256=sha(root/'qualification-checklist.json'),
            candidate=candidate, changed_profile_field='kvmem.'+checks.pairs.CANDIDATES[candidate]['field'],
            variants=checks.pairs.CANDIDATES[candidate]['variants'], pairs=args.pairs, jobs=jobs,
            arms=[job['id'] for job in jobs], workload=checks.WORKLOAD, model_identity=identity,
            port=args.port, idle_seconds=10, budget=checks.host_budget(), host_admission=chain.HOST_ADMISSION,
            minimum_gpu_free_mib=4096, qualify=False, native_source=cap.native_source_identity(),
            candidate_prequalified=False, current_profile_prequalified=False,
            python=dict(executable=str(Path(sys.executable).resolve()), version=sys.version),
            scope='Fixed original eight independent agents; natural own outputs, fixed300s after two warmups, legal full256K/16-turn epochs. Fresh service per job, no retry.'))
        save(root/'status.json', dict(status='prepared', at=time.time()))
        print(json.dumps(dict(prepared=str(root), candidate=candidate, jobs=[j['id'] for j in jobs], gpu_started=False)),flush=True)
    except BaseException as error:
        save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise


def validate(root):
    m = read(root/'manifest.json')
    require(m.get('experiment_kind') == checks.SCHEMA and m.get('kind') == 'ready_agents_v1', 'Wrong agent protocol')
    require(m['jobs'] == checks.job_order(m['pairs']) and m['arms'] == [j['id'] for j in m['jobs']] and
            m['workload'] == checks.WORKLOAD, 'Alternating order or fixed workload changed')
    candidate = m['candidate']
    require(m['variants'] == checks.pairs.CANDIDATES[candidate]['variants'], 'Frozen candidate contract changed')
    require(cap.hashes(root/'frozen') == m['source_sha256'], 'Frozen source changed')
    require(sha(root/'source.diff') == m['source_diff_sha256'] and sha(root/'fixture.json') == m['fixture_sha256'] == checks.FIXTURE_SHA,
            'Frozen source/fixture identity changed')
    require(sha(root/'expected-launches.json') == m['expected_launches_sha256'] and
        sha(root/'qualification-checklist.json') == m['qualification_checklist_sha256'] and
        cap.hashes(root/'qualifications') == m['qualification_sha256'], 'Frozen launch/qualification evidence changed')
    require(m['budget'] == checks.host_budget() and m['host_admission'] == chain.HOST_ADMISSION and
        m['minimum_gpu_free_mib'] == 4096 and m.get('candidate_prequalified') is False and
        m.get('current_profile_prequalified') is False and m['qualify'] is False, 'Admission or qualification scope changed')
    require(m['native_source'] == cap.native_source_identity(), 'Installed native vLLM source changed')
    require(m['python'] == dict(executable=str(Path(sys.executable).resolve()),version=sys.version), 'Interpreter changed')
    roles = {}
    for job in m['jobs']:
        name = job['id']
        require(sha(root/(name+'.json')) == m['profiles_sha256'][name] and
            cap.hashes(root/'runtime'/name) == m['runtime_sha256'][name] == m['source_sha256'], 'Profile/runtime changed')
        profile = read(root/(name+'.json'))
        require(job['role'] not in roles or profile == roles[job['role']], 'A repeated profile changed')
        roles[job['role']] = profile
    configs = dict(zip(m['variants'], (roles['baseline'],roles['candidate'])))
    checks.validate_profiles(configs, candidate)
    checklist = read(root/'qualification-checklist.json')
    for mode in ('sync','async'):
        proof = checks.qualification_proof(root/'qualifications'/mode,roles['baseline'],m['source_sha256'], mode=='sync', runtime_root=root/'frozen')
        require(proof == {k:v for k,v in checklist[mode].items() if k!='original_path'}, 'Qualification coverage changed')
    for group in ('config_sha256','tokenizer_sha256'):
        for name,expected in m['model_identity'][group].items(): require(sha(name)==expected,'Model/tokenizer changed')
    for name,expected in m['model_identity']['weight_stat_identity'].items():
        st = Path(name).stat()
        require(dict(size=st.st_size,mtime_ns=st.st_mtime_ns)==expected,'Model weight identity changed')
    return m


async def monitored_workload(original, client, profile, fixture, directory, values):
    """Add full-history/resource evidence, never alter the original request plan."""
    import pynvml
    pynvml.nvmlInit()
    device = pynvml.nvmlDeviceGetHandleByIndex(0)
    selected = SimpleNamespace(**{**vars(values),'explicit_session_release':True})
    async def monitor():
        target = directory.parent/'agent_counters.jsonl'
        while True:
            path = values.capture_directory/'archive.json'
            archive = read(path) if path.exists() else None
            if archive is not None: chain.require_terminal_history(archive)
            memory = pynvml.nvmlDeviceGetMemoryInfo(device)
            require(memory.free >= 4096*1024**2, 'Device memory headroom below4GiB')
            row = dict(at=time.monotonic(),wall_at=time.time(),
                archive={k:v for k,v in (archive or {}).items() if k.startswith('seal_') or k in
                    ('session_checkpoint_deferrals','session_checkpoint_payload_bytes','cpu_authority_bytes')},
                nvml=dict(gpu_used_mib=memory.used/1024**2,gpu_free_mib=memory.free/1024**2,
                    gpu_utilization_percent=pynvml.nvmlDeviceGetUtilizationRates(device).gpu,
                    gpu_power_w=pynvml.nvmlDeviceGetPowerUsage(device)/1000))
            with target.open('a') as stream: stream.write(json.dumps(row)+'\n')
            await asyncio.sleep(1)
    try:
        result = await ready.supervise_monitored(original(client,profile,fixture,directory,selected),
                                               monitor(),monitor_name='full history/NVML budget')
        rows = read(directory/'traces.json')
        closes = [r for r in rows if r['kind']=='cleanup']
        require(closes and all(r['explicit_session_release'] for r in closes), 'Epoch cleanup omitted explicit release')
        # This profile deliberately disables idle active residency. Its absence
        # in telemetry is not a live lease; verify the actual frozen launch
        # before permitting that single optional field to be absent.
        job_directory = directory.parent
        checks.actual_profile(job_directory,
            read(job_directory.parent/'expected-launches.json')[job_directory.name])
        require(profile['kvmem'].get('active_resident_pressure_only') is False and
            profile['kvmem'].get('active_resident_ttl_ms') == 0,
            'Missing residency telemetry is allowed only for an explicitly disabled module')
        proof = await release.wait_release(values.capture_directory,max(r['begin'] for r in closes),
                                           require_active_residency=False)
        chain.require_terminal_history(proof['archive'])
        save(directory/'explicit-release-proof.json',proof)
        result['explicit_session_release_verified'] = True
        return result
    finally:
        pynvml.nvmlShutdown()


async def guarded_arm(root, name, manifest):
    task = asyncio.create_task(ready.run_arm(root,name,manifest,prelaunch_check=chain.prelaunch_admission))
    cancelled = False
    try:
        while not task.done():
            try: await asyncio.shield(task)
            except asyncio.CancelledError:
                if not cancelled: task.cancel(); cancelled = True
        if cancelled: raise asyncio.CancelledError
        return task.result()
    finally:
        (root/name).mkdir(exist_ok=True)
        # The server owner's awaited finally has already stopped its process
        # group. This bounded, read-only device check must finish even under a
        # repeated cancellation; its two device probes each have a10s timeout.
        chain.post_stop_proof(root,name)


def audit_job(root, job, manifest, result):
    directory = root/job['id']
    profile, fixture = read(root/(job['id']+'.json')), read(root/'fixture.json')
    rows, events = read(directory/'tools/traces.json'), ready.records(directory/'tools/events.jsonl')
    history = chain.own_history_proof(fixture,rows,events)
    require(result.get('explicit_session_release_verified') and result.get('native_client_tokens_equal') and
        result['preemptions']==0 and result.get('drained'), 'Unreleased, preempted or incomplete workload')
    chain.require_terminal_history(result['archive_final'])
    publications = ready.records(directory/'history_publications.jsonl') + result['archive_final'].get('completed', [])
    for archive in publications:
        chain.require_terminal_history(dict(session_checkpoint_deferrals=0,completed=[archive]))
    for row in rows:
        if row['kind']=='cleanup': continue
        matching = [entry for entry in publications if checks.counters.q.native_matches(row,entry)]
        expected_end = row['input_tokens']+len(row['output_ids'])-bool(row['output_ids'])
        require(matching and any(entry.get('history_end') == expected_end and
            entry.get('checkpoint_kind')=='M' and entry.get('checkpoint_tokens')==entry['history_end'] for entry in matching),
            'Missing complete accepted terminal CPU history/checkpoint publication')
    report = dict(history=history, actual_profile=checks.actual_profile(directory,
        read(root/'expected-launches.json')[job['id']]), graph=checks.graph_evidence(directory/'server/capture',fixture,rows,profile),
        measured=checks.measurements(directory,profile,result))
    save(directory/'v2-evidence.json',report)
    return report


def comparison(root, manifest):
    reports = {job['id']:read(root/job['id']/'v2-evidence.json') for job in manifest['jobs']}
    pairs = []
    for number in range(1,manifest['pairs']+1):
        a,b = (reports[f'p{number}-{role}'] for role in ('baseline','candidate'))
        ar,br = a['measured'],b['measured']
        field = {'raw-k-cache':'raw_cache_exercised','raw-k-cache-window':'raw_cache_exercised',
                 'cpu-direct-record-seal':'seal','async-action-metadata':'action_metadata'}[manifest['candidate']]
        observed = br[field]
        if field=='seal': observed=observed['direct_path_observed']
        if field=='action_metadata': observed=observed['async_path_observed']
        # Different progress yields different request counts. Compare all
        # shared logical turns; never call changed own outputs a pure speedup.
        def turns(role):
            rows = read(root/f'p{number}-{role}'/'tools/traces.json')
            return {(r['agent'],r['epoch'],r['turn']):r for r in rows if r['kind']!='cleanup'}
        left,right = turns('baseline'),turns('candidate')
        common = left.keys() & right.keys()
        identical = bool(common) and all(all(left[k][f]==right[k][f] for f in
            ('input_sha256','output_sha256','query_plan')) for k in common)
        selections = [{(r['agent'],r['epoch'],r['turn']):r['selection_sha256'] for r in evidence['graph']['requests']}
                      for evidence in (a,b)]
        selection_equal = bool(common) and all(selections[0].get(k) == selections[1].get(k) and
            k in selections[0] and k in selections[1] for k in common)
        conditions = dict(candidate_exercised=bool(observed), common_outputs_identical=identical,
            common_selections_identical=selection_equal, positive_output=ar['output_tokens']>0 and br['output_tokens']>0,
            native_c8_observed=all(r['native']['actual_c8_compute_observed'] for r in (ar,br)),
            latency_measurement_complete=all(r['latency']['measurement_complete'] for r in (ar,br)))
        if manifest['candidate']=='cpu-direct-record-seal':
            conditions['seal_measurement_complete']=all(r['seal']['measurement_complete'] for r in (ar,br))
        eligible = all(conditions.values())
        tails = {name:br['latency'][name]/ar['latency'][name] if ar['latency'].get(name) and
            br['latency'].get(name) is not None else None for name in
            ('request_p95_seconds','ttft_p95_seconds','intertoken_event_gap_p95_seconds')}
        ratio = br['output_tokens_per_second']/ar['output_tokens_per_second'] if ar['output_tokens_per_second'] else None
        pairs.append(dict(pair=number,order=['baseline','candidate'] if number%2 else ['candidate','baseline'],
            candidate_path_observed=observed, common_turns=len(common),common_own_inputs_outputs_identical=identical,
            common_selection_sequences_identical=selection_equal,
            comparison_conditions=conditions,inconclusive_reasons=[k for k,v in conditions.items() if not v],
            tail_latency_ratios_candidate_over_baseline=tails,
            observed_accepted_output_rate_ratio=ratio, usable_speedup_ratio=ratio if eligible else None,
            verdict='eligible_observation' if eligible else 'inconclusive'))
    eligible = all(p['verdict']=='eligible_observation' for p in pairs)
    median = statistics.median(p['usable_speedup_ratio'] for p in pairs) if eligible else None
    return dict(schema=checks.SCHEMA,candidate=manifest['candidate'],pairs=pairs,
        verdict='screening_observation' if eligible and len(pairs)==1 else 'repeated_observation' if eligible else 'inconclusive',
        geometric_mean_usable_ratio=math.prod(p['usable_speedup_ratio'] for p in pairs)**(1/len(pairs)) if eligible else None,
        median_usable_ratio=median, median_five_percent_screen_passed=median is not None and len(pairs)==3 and median>=1.05,
        tail_latency_gate='pending_manual_review_of_consistent_material_regression',
        merge_gain_qualified=False, performance_gain_established=False,
        statistical_significance_claimed=False, candidate_prequalified=False,
        scope='Recorded external tool replay with own natural outputs; throughput evidence, not agent task-quality evaluation. Preserve failures and divergent histories; no automatic reruns.')


async def run(args):
    root = args.output.resolve()
    with (root/'.campaign.lock').open('a+') as campaign:
        fcntl.flock(campaign,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = validate(root)
        require(ROOT.resolve() == (root/'frozen').resolve(), 'Run the frozen agent controller')
        require(read(root/'status.json')['status']=='prepared', 'No retry of started, failed, interrupted or completed campaigns')
        require(all(not (root/j['id']).exists() for j in manifest['jobs']), 'A job already exists; prepare a new campaign')
        require(args.candidate in (None,manifest['candidate']), 'Requested candidate differs from frozen campaign')
        loop, active = asyncio.get_running_loop(),asyncio.current_task()
        original, original_gpu = work.workload, ready.bench.q.gpu_state
        def gpu_state():
            import pynvml
            result = original_gpu()
            pynvml.nvmlInit()
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
                require(memory.free >= 4096*1024**2, 'All-phase GPU memory headroom below4GiB')
                result['free_mib'] = memory.free/1024**2
            finally: pynvml.nvmlShutdown()
            return result
        for sig in (signal.SIGINT,signal.SIGTERM): loop.add_signal_handler(sig,active.cancel)
        work.workload = lambda *values: monitored_workload(original,*values)
        ready.bench.q.gpu_state = gpu_state
        try:
            with open('/tmp/kvmem-coordination-qualification.lock','a+') as device:
                fcntl.flock(device,fcntl.LOCK_EX|fcntl.LOCK_NB)
                save(root/'status.json',dict(status='started',at=time.time()))
                for job in manifest['jobs']:
                    result = await guarded_arm(root,job['id'],manifest)
                    audit_job(root,job,manifest,result)
                save(root/'comparison.json',comparison(root,manifest))
                save(root/'status.json',dict(status='completed_pending_review',at=time.time()))
        except BaseException as error:
            save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
            raise
        finally:
            work.workload = original
            ready.bench.q.gpu_state = original_gpu
            for sig in (signal.SIGINT,signal.SIGTERM): loop.remove_signal_handler(sig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','validate','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',choices=tuple(checks.pairs.CANDIDATES))
    parser.add_argument('--pairs',type=int,choices=(1,3),default=1)
    parser.add_argument('--profile',type=Path,default=DEFAULT_PROFILE)
    parser.add_argument('--fixture',type=Path,default=DEFAULT_FIXTURE)
    parser.add_argument('--sync-qualification',type=Path,default=DEFAULT_QUALIFICATIONS/'sync-r1')
    parser.add_argument('--async-qualification',type=Path,default=DEFAULT_QUALIFICATIONS/'async-r1')
    parser.add_argument('--port',type=int,default=18686)
    args = parser.parse_args(argv)
    if args.mode=='prepare': return prepare(args)
    if args.mode=='validate':
        m=validate(args.output.resolve())
        print(json.dumps(dict(valid=True,candidate=m['candidate'],pairs=m['pairs'],gpu_started=False)),flush=True)
        return
    asyncio.run(run(args))


if __name__=='__main__': main()

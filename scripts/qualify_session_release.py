"""Qualify explicit session release across more namespaces than cache slots.

Five waves of eight independent namespaces exercise full-history continuation,
native MTP state restoration and fenced release. The final wave uses the whole
recorded long prompt; the shorter waves expose the fixed model-checkpoint cost.
This is correctness qualification, not a DeepSWE score or a throughput window.
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


ROOT = Path(__file__).resolve().parents[1]
USERS = 8
WAVES = 5
SHORT_CONTEXT = 8192
OUTPUT_TOKENS = 512
TOOL_TOKENS = 128
spec = importlib.util.spec_from_file_location('session_release_shared', ROOT/'scripts/qualify_rolling_window.py')
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)
cap, work, core = shared.cap, shared.work, shared.core


def qualification_profile(source, port):
    profile = copy.deepcopy(source)
    cap.require(profile.get('model_format') == 'safetensors' and
        profile.get('execution_mode') == 'compiled_graph' and
        profile.get('draft_method') == 'mtp' and profile.get('draft_tokens') == 4 and
        profile.get('max_num_seqs') == USERS and profile.get('max_model_len') == 196608,
        'Preserve NVFP4/MTP4/graph and eight-client deployment geometry')
    kv = profile['kvmem']
    cap.require(kv.get('select_budget') == 65536 and kv.get('gen_budget') == 32768 and
        kv.get('block_tokens') == 32 and kv.get('archive_backend') == 'cpu' and
        kv.get('cpu_read_cache_bytes') == 100_000_000_000 and
        kv.get('session_capacity') == 32 and kv.get('session_checkpoint_bytes') == 4*1024**3,
        'Preserve the observed CPU-history and checkpoint capacity')
    cap.require(kv.get('session_checkpoint_policy') == 'pm' and
        kv.get('session_capture_terminal') is True and kv.get('rolling_active_window') is True and
        kv.get('cpu_history_coordination') is True and kv.get('active_resident_pressure_only') is True,
        'Require terminal checkpoints, coordinated CPU history and pressure-managed residency')
    profile.update(port=port, verify_session_restore=True)
    return profile


def validate_fixture(fixture, profile):
    tokens = fixture.get('tokens')
    cap.require(isinstance(tokens, list) and len(tokens) == 99826 and
        all(type(n) is int and n >= 0 for n in tokens), 'Require the complete recorded 99826-token prompt')
    cap.require(type(fixture.get('eos')) is int and fixture['eos'] >= 0, 'Tokenizer EOS required')
    bound = len(tokens)+2*OUTPUT_TOKENS+TOOL_TOKENS
    cap.require(bound <= profile['max_model_len'], 'Qualification history exceeds model ceiling')
    archive = USERS*((bound+31)//32*32+64)*cap.ARCHIVE_BYTES_PER_TOKEN
    return dict(archive_bytes=archive, maximum_history_tokens=bound,
        required_available_bytes=archive+USERS*cap.INDEX_BYTES_PER_REQUEST+32*1024**3,
        minimum_free_bytes=8*1024**3)


def validate_archive(archive):
    deferred = archive.get('session_checkpoint_deferrals')
    cap.require(type(deferred) is int and deferred == 0, 'Checkpoint deferral or missing accounting')
    cap.require(archive.get('archive_backend') == 'cpu' and
        archive.get('repository_gc') is False and
        archive.get('disk_read_bytes') == archive.get('disk_write_bytes') == 0,
        'Complete CPU-only history contract changed')


def completely_released(state, archive):
    validate_archive(archive)
    required = ('session_checkpoints', 'session_checkpoint_payload_bytes', 'cpu_authority_bytes',
                'retiring_requests', 'session_retiring_requests', 'pending_downloads')
    # Missing fields must not turn a stale or incomplete snapshot into proof.
    if any(type(archive.get(k)) is not int or archive[k] != 0 for k in required):
        return False
    cpu = state.get('cpu_history_admission', {})
    residency = state.get('active_residency', {})
    return (state.get('session_views') == [] and cpu.get('leases') == 0 and
        cpu.get('reserved_bytes') == 0 and residency.get('idle_views') == 0 and
        residency.get('idle_blocks') == 0 and state.get('free_blocks') == state.get('total_blocks', 0)-1)


async def wait_release(capture, not_before, timeout=180):
    deadline = time.monotonic()+timeout
    while True:
        state = await work.wait_retirement(capture, timeout=max(.1,deadline-time.monotonic()),
                                           not_before=not_before)
        archive = cap.read(capture/'archive.json')
        if completely_released(state, archive):
            return dict(resources=state, archive=archive)
        if time.monotonic() >= deadline:
            raise TimeoutError('Explicit close retained CPU history, model snapshots or GPU leases')
        await asyncio.sleep(.05)


def restore_evidence(capture, resumes):
    cap.require(not (capture/'restore_mismatch.json').exists(), 'Restore byte mismatch')
    targets = core.read_records(capture/'verified_restores.jsonl')
    states = core.read_records(capture/'verified_state_restores.jsonl')
    proofs = []
    for row in resumes:
        target = [r for r in targets if core.native_matches(row,r['request_id'])]
        state = [r for r in states if core.native_matches(row,r['request_id'])]
        cap.require(target and state and all(r.get('byte_equal') is True and
            r.get('working_attention_layers',0) > 0 and r.get('mtp_attention_layers',0) > 0 and
            r.get('target_kv_bytes',0) > 0 and
            r.get('oracle_kind') in ('native_terminal_window','fp8_immutable_rebuild') for r in target) and
            all(r.get('recurrent_byte_equal') is True for r in state),
            'Each continuation requires target/MTP and recurrent restore byte proofs')
        proofs.append(dict(request_id=row['request_id'],target=target,recurrent=state))
    return proofs


async def workload(client, profile, fixture, directory, capture):
    directory.mkdir()
    recorder = work.Recorder(directory, USERS)
    proofs, releases, resumes, occupancy, namespaces = [], [], [], [], []
    try:
        for wave in range(WAVES):
            wave_begin = time.monotonic()
            sessions = [f'release-qualification-{wave}-{i}' for i in range(USERS)]
            namespaces.extend(sessions)
            prompt = fixture['tokens'] if wave == WAVES-1 else fixture['tokens'][:SHORT_CONTEXT]
            histories = {}

            async def call(i, turn, close=False):
                tokens = ([fixture['eos']] if close else list(prompt) if turn == 0
                          else histories[i]+fixture['tokens'][:TOOL_TOKENS])
                row = await work.stream_call(client,profile,tokens,sessions[i],1 if close else OUTPUT_TOKENS,
                    recorder,agent=i,epoch=wave,turn=turn,kind='cleanup' if close else 'workload',
                    request_id=f'release-{wave}-{i}-{turn}',new_input_tokens=len(tokens) if turn == 0 else TOOL_TOKENS,
                    query=not close,query_plan=None if close else dict(kvmem_reselect='auto',
                        kvmem_prefill_window='pressure',kvmem_mandatory_spans=[]),release_session=close)
                if not close:
                    histories[i] = tokens+row['output_ids']
                return row

            for turn in range(2):
                rows = await asyncio.gather(*(call(i,turn) for i in range(USERS)))
                state = await work.wait_retirement(capture,timeout=180,
                    not_before=max(r['begin'] for r in rows),
                    expected_views={sessions[i]:len(histories[i])-1 for i in range(USERS)})
                archive = cap.read(capture/'archive.json')
                validate_archive(archive)
                cap.require(archive.get('session_checkpoints') == USERS and
                    archive.get('session_checkpoint_payload_bytes',0) > 0,
                    'Wave must retain exactly eight real model checkpoint owners before release')
                cap.save(directory/f'wave-{wave}-turn-{turn}.json',dict(resources=state,archive=archive))
                proofs.extend(shared.history_evidence(rows,archive.get('completed',[])))
                if turn:
                    resumes.extend(rows)
                    restore_evidence(capture,rows)
            steps = [r for r in core.read_records(capture/'model_step_latency.jsonl')
                if r['started_at'] >= wave_begin and r.get('decode_requests') == USERS and
                not r.get('prefill_tokens')]
            cap.require(steps, 'Wave did not execute an actual eight-request decode batch')
            occupancy.append(dict(wave=wave,pure_eight_request_steps=len(steps)))
            closed = await asyncio.gather(*(call(i,2,True) for i in range(USERS)))
            released = await wait_release(capture,max(r['begin'] for r in closed))
            cap.save(directory/f'wave-{wave}-released.json',released)
            publications = released['archive'].get('completed',[])
            for row in closed:
                matching = [p for p in publications if core.native_matches(row,p['request_id'])]
                cap.require(matching and all(p.get('checkpoint_kind') not in ('P','M') for p in matching),
                            'Release request created a replacement model checkpoint')
            releases.append(dict(wave=wave,namespaces=sessions,all_resources_released=True))
        cap.require(len(set(namespaces)) == USERS*WAVES > profile['kvmem']['session_capacity'],
                    'Qualification did not exceed the namespace capacity')
        return dict(status='passed',history_proofs=proofs,restore_proofs=restore_evidence(capture,resumes),
            releases=releases,actual_batch_evidence=occupancy,namespace_count=len(namespaces),
            total_stream_output_tokens=sum(len(r['output_ids']) for r in recorder.rows),
            cleanup_completed=True,checkpoint_deferrals=0,
            scope='Five waves, forty namespaces, own-output continuations and a full 99826-token final wave. '
                  'Correctness only; short waves intentionally retain the same fixed model-checkpoint cost.')
    finally:
        cap.save(directory/'traces.json',recorder.rows)
        recorder.progress()
        recorder.stream.close()


def prepare(args):
    root, candidate = args.output.resolve(), args.candidate.resolve()
    cap.require(not root.exists(), 'Use a fresh qualification output')
    profile = qualification_profile(cap.read(args.config),args.port)
    fixture = cap.read(args.fixture)
    budget = validate_fixture(fixture,profile)
    root.mkdir(parents=True)
    for folder in ('src','csrc','scripts'):
        shutil.copytree(ROOT/folder if folder == 'scripts' else candidate/folder,root/'frozen'/folder,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(candidate,root/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    cap.save(root/'profile.json',profile)
    cap.save(root/'fixture.json',fixture)
    cap.save(root/'manifest.json',dict(kind='session_release_qualification_v1',at=time.time(),
        source_sha256=cap.hashes(root/'frozen'),runtime_sha256=cap.hashes(root/'runtime'),
        profile_sha256=cap.sha(root/'profile.json'),fixture_sha256=cap.sha(root/'fixture.json'),
        fixture_source=str(args.fixture.resolve()),fixture_source_sha256=cap.sha(args.fixture),
        runtime_source=str(candidate),native_source=cap.native_source_identity(),budget=budget,
        idle_seconds=15,concurrency=USERS,waves=WAVES,short_context=SHORT_CONTEXT,
        output_tokens=OUTPUT_TOKENS,tool_tokens=TOOL_TOKENS,
        scope='Correctness qualification; preserve completed and failed real DeepSWE attempts.'))
    cap.save(root/'status.json',dict(status='prepared',at=time.time()))


def verify(root):
    cap.require(ROOT.resolve() == (root/'frozen').resolve(), 'Execute the frozen qualification')
    m = cap.read(root/'manifest.json')
    cap.require(m.get('kind') == 'session_release_qualification_v1', 'Unexpected qualification kind')
    cap.require(cap.hashes(ROOT) == m['source_sha256'] and cap.hashes(root/'runtime') == m['runtime_sha256'],
                'Frozen source or runtime changed')
    cap.require(cap.sha(root/'profile.json') == m['profile_sha256'] and
        cap.sha(root/'fixture.json') == m['fixture_sha256'] and cap.native_source_identity() == m['native_source'],
        'Profile, fixture or installed native source changed')
    profile = cap.read(root/'profile.json')
    cap.require(qualification_profile(profile,profile['port']) == profile and
        validate_fixture(cap.read(root/'fixture.json'),profile) == m['budget'], 'Qualification contract changed')
    cap.require((m['concurrency'],m['waves'],m['short_context'],m['output_tokens'],m['tool_tokens']) ==
        (USERS,WAVES,SHORT_CONTEXT,OUTPUT_TOKENS,TOOL_TOKENS), 'Workload constants changed')
    return m


async def run(args):
    import psutil
    root = args.output.resolve()
    m = verify(root)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared', 'No automatic restart')
    owner = psutil.Process()
    cap.save(root/'owner.json',dict(pid=owner.pid,create_time=owner.create_time(),argv=sys.argv,at=time.time()))
    loop, current = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):
        loop.add_signal_handler(sig,current.cancel)
    shared.workload = workload
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            await shared.run_server(root,m)
        cap.save(root/'status.json',dict(status='passed',at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json',dict(status='failed',at=time.time(),error=repr(error),no_automatic_retry=True))
        raise
    finally:
        for sig in (signal.SIGINT,signal.SIGTERM):
            loop.remove_signal_handler(sig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--candidate',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--fixture',type=Path)
    p.add_argument('--port',type=int,default=18620)
    p.add_argument('--run-prepared',action='store_true')
    args = p.parse_args()
    if args.run_prepared:
        asyncio.run(run(args))
    else:
        cap.require(all((args.candidate,args.config,args.fixture)), 'Candidate, profile and fixture required')
        cap.require(1024 <= args.port <= 65535, 'Invalid dedicated server port')
        prepare(args)

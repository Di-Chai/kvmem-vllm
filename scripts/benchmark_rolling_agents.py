"""Frozen natural-EOS agent replay with independently qualified KVMem runtimes.

Prepare without GPU access; run only the frozen entry point after BOTH byte
qualifications finish. This replays recorded external messages and waits with
each model's own outputs, not on-policy DeepSWE task quality. Analyze the result
with frozen/scripts/analyze_rolling_agents.py; output length changes must be
considered alongside token rate, completed turns, batches and maintenance.
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
ARMS = ('native', 'pipeline-base', 'rolling-candidate-v2')
USERS, TURNS, WARMUP, LIMIT, CEILING = 8, 16, 2, 32000, 262144
EXPERIMENT = 'rolling_agents_natural_v1'
SAMPLING = dict(temperature=.6, top_p=.95, top_k=20, min_p=0,
                presence_penalty=0, repetition_penalty=1, seed=1000,
                thinking_token_budget=4096, ignore_eos=False, skip_special_tokens=False)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


ready = module('natural_rolling_ready', 'benchmark_ready_agents.py')
qual = module('natural_rolling_qualification', 'qualify_rolling_window.py')
cap, work = ready.cap, ready.work


def profiles(source, port):
    """Identical deployment controls; only candidate enables rolling pressure."""
    p = cap.capacity_profile(source, port)
    kv = p['kvmem']
    cap.require(all(kv.get(k) is True for k in ('request_planning', 'async_refresh',
        'async_restore', 'cpu_history_coordination', 'session_capture_terminal', 'action_pressure')),
        'Require coordinated asynchronous pressure and terminal capture')
    cap.require(kv.get('session_checkpoint_policy') == 'pm' and
                not kv.get('session_capture_decode', False), 'Require terminal PM checkpoints')
    p.update(verify_session_restore=False, max_num_seqs=USERS)
    kv.update(resident_restore_slots=4, step_latency_interval=1)
    kv.pop('rolling_active_window', None)  # The older baseline has no such option.
    result = {arm: copy.deepcopy(p) for arm in ARMS}
    result[ARMS[2]]['kvmem']['rolling_active_window'] = True
    return result


def fixture_bounds(fixture, profile):
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.request_plan import RequestPlan
    cap.require(type(fixture.get('eos')) is int and fixture['eos'] >= 0, 'Validated EOS required')
    cap.require(len(fixture.get('agent_plans', [])) == USERS, 'Require exactly the original first eight users')
    plans = work.validate_agent_plans(fixture, concurrency=USERS, warmup_turns=WARMUP,
        epoch_turns=TURNS, gen_budget=32768, enforce_output_budget=False)
    selection = fixture.get('selection', {})
    cap.require(selection.get('users') == USERS and selection.get('turns') == TURNS and
        selection.get('manifest_order') == 'first users without reward/timing selection',
        'Require declared original first-eight consecutive-turn selection')
    cap.require(len({p.get('source_task') for p in plans}) == USERS and
        all(p.get('source_task') and type(p.get('source_turn')) is int for p in plans),
        'Independent source task/turn provenance required')
    queries = []
    for plan in plans:
        tokens = plan['initial_tokens']
        cap.require(len(tokens)+LIMIT <= CEILING, 'Complete initial input plus generation limit exceeds ceiling')
        query = RequestPlan.parse(plan['query_plan'], len(tokens), 32, 32768,
                                  profile['kvmem'].get('guided_query_tokens', 0))
        cap.require(query.query is not None and query.replay_end-query.replay_begin <=
                    profile['query_replay_budget'], 'Complete Query exceeds replay budget')
        query.mandatory_fit(len(tokens), KVMemConfig(select_budget=65536, gen_budget=32768,
            block_tokens=32, sink_blocks=profile['kvmem'].get('sink_blocks', 0),
            recent_blocks=profile['kvmem'].get('recent_blocks', 0)))
        cap.require(len(tokens)+1+len(plan['turns'][1]['tool_tokens'])+LIMIT <= CEILING,
                    'Even minimum natural output leaves no room for second warmup; no clipping allowed')
        for i, turn in enumerate(plan['turns'][:TURNS]):
            cap.require(turn.get('source_turn') == plan['source_turn']+i, 'Source turns must be consecutive')
            if i:
                cap.require(turn['tool_tokens'] and turn.get('external_message_proof'),
                            'Require complete original external tool messages')
        queries.append(dict(source_task=plan['source_task'], source_turn=plan['source_turn'],
            initial_tokens=len(tokens), query_span=list(query.query)))
    # Natural outputs are unknown. Never reuse recorded output quotas to make
    # the CPU admission estimate smaller. Include two block staging margins.
    archive = USERS*(CEILING+64)*cap.ARCHIVE_BYTES_PER_TOKEN
    cap.require(archive <= 100_000_000_000, 'Complete CPU historical budget exceeds configured cap')
    index, extra = USERS*cap.INDEX_BYTES_PER_REQUEST, 32*1024**3
    return dict(history_limits=[CEILING]*USERS, archive_bytes=archive,
        mean_k_index_bytes=index, extra_bytes=extra, minimum_free_bytes=8*1024**3,
        required_available_bytes=archive+index+extra, initial_queries=queries,
        scope='Full per-user historical ceiling, independent of recorded or observed output lengths')


def verify_provenance(fixture):
    provenance = fixture.get('provenance', {})
    for key in ('source_file_sha256', 'tokenizer_file_sha256'):
        entries = provenance.get(key)
        cap.require(isinstance(entries, dict) and entries, 'Missing frozen source/tokenizer provenance')
        for filename, digest in entries.items():
            path = Path(filename)
            cap.require(path.is_absolute() and path.is_file() and cap.sha(path) == digest,
                        f'Original provenance changed: {filename}')


def qualification_identity(path, arm, expected_runtime, native_source):
    """Read-only identity check, usable before correctness execution finishes."""
    m = cap.read(path/'manifest.json')
    rolling = arm == ARMS[2]
    cap.require(m.get('kind') == ('rolling_window_qualification_v1' if rolling else 'ready_agents_v1'),
                'Unexpected correctness qualification kind')
    cap.require(cap.hashes(path/'frozen') == m['source_sha256'], 'Qualification frozen source changed')
    if rolling:
        actual = cap.hashes(path/'runtime')
        cap.require(actual == m['runtime_sha256'], 'Rolling qualification runtime changed')
        cap.require(cap.sha(path/'profile.json') == m['profile_sha256'], 'Rolling qualification profile changed')
        p = cap.read(path/'profile.json')
        cap.require(p == qual.qualification_profile(p, p['port']), 'Rolling qualification controls changed')
    else:
        actual = cap.hashes(path/'runtime/kvmem-ready')
        cap.require(actual == m['runtime_sha256']['kvmem-ready'], 'Pipeline qualification runtime changed')
        cap.require(cap.sha(path/'kvmem-ready.json') == m['profiles_sha256']['kvmem-ready'],
                    'Pipeline qualification profile changed')
        p = cap.read(path/'kvmem-ready.json')
        cap.require(p.get('verify_session_restore') is True and
                    p['kvmem'].get('resident_restore_slots') == 4, 'Pipeline oracle controls missing')
    cap.require(actual == expected_runtime, 'Performance runtime does not match exact qualified candidate')
    cap.require(cap.sha(path/'fixture.json') == m['fixture_sha256'], 'Qualification fixture changed')
    cap.require(m['native_source'] == native_source, 'Qualification native implementation differs')
    return dict(path=str(path), manifest_sha256=cap.sha(path/'manifest.json'), runtime_sha256=actual)


def validate_completed_qualification(path, arm):
    cap.require(cap.read(path/'status.json')['status'] == 'completed_pending_review',
                f'{arm} GPU qualification has not completed; performance launch forbidden')
    rolling = arm == ARMS[2]
    capture = path/('server/capture' if rolling else 'kvmem-ready/server/capture')
    cap.require(not (capture/'restore_mismatch.json').exists(), 'GPU restore byte mismatch')
    result_path = path/('results.json' if rolling else 'kvmem-ready/results.json')
    result = cap.read(result_path)
    if rolling:
        cap.require(result.get('status') == 'passed' and result.get('cleanup_completed') is True,
                    'Rolling qualification failed or did not clean up')
        rows = cap.read(path/'workload/traces.json')
        cap.require(all(any(r['kind'] == 'cleanup' and r['agent'] == user and len(r['output_ids']) == 1
            for r in rows) for user in range(qual.USERS)), 'Rolling session cleanup traces missing')
        oracle = qual.oracle_evidence(capture, rows)
        cap.require(all(r.get('oracle_kind') == 'native_stable_rows_plus_fp8_rebuild'
            for r in oracle['verified_rolling_refreshes']), 'Require candidate-v2 immutable mixed-row oracle')
        history = qual.history_evidence([r for r in rows if r['kind'] == 'workload'],
                                       cap.read(path/'workload/history_publications.json'))
        reconciliation = result.get('native_reconciliation', {})
        cap.require(reconciliation.get('passed') is True and reconciliation.get('preemptions') == 0
            and reconciliation.get('native_tokens') == reconciliation.get('client_tokens') ==
            result.get('total_stream_output_tokens') == sum(len(r['output_ids']) for r in rows),
            'Rolling accepted output reconciliation failed')
        proof = dict(oracle=oracle, history=history)
    else:
        cap.require(result.get('status') == 'completed' and result.get('drained') is True and
            result.get('native_client_tokens_equal') is True and result.get('preemptions') == 0,
            'Pipeline qualification did not drain/reconcile cleanly')
        archive = result.get('archive_final', {})
        cap.require(archive.get('archive_backend') == 'cpu' and archive.get('repository_gc') is False and
            archive.get('disk_read_bytes') == archive.get('disk_write_bytes') == 0,
            'Pipeline qualification did not retain complete CPU history')
        targets = ready.records(capture/'verified_restores.jsonl')
        states = ready.records(capture/'verified_state_restores.jsonl')
        cap.require(targets and states and all(r.get('byte_equal') is True and
            r.get('working_attention_layers', 0)>0 and r.get('mtp_attention_layers', 0)>0 and
            r.get('target_kv_bytes', 0)>0 for r in targets) and
            all(r.get('recurrent_byte_equal') is True for r in states), 'Pipeline target/MTP/recurrent byte proof missing')
        cap.require({r['request_id'] for r in targets} == {r['request_id'] for r in states},
                    'Pipeline target/recurrent request proofs differ')
        rows = cap.read(path/'kvmem-ready/tools/traces.json')
        resumes = [r for r in rows if r['kind'] in ('warmup','workload') and r['turn'] > 0]
        cap.require(resumes and all(any(qual.core.native_matches(row,r['request_id']) for r in targets)
            for row in resumes), 'Pipeline continuation lacks per-request byte proof')
        peak = cap.read(capture/'scheduler_resources.json')['restore_admission']['peak_occupied']['resident']
        cap.require(peak > 1, 'Pipeline qualification lacks independent restore overlap')
        proof = dict(target_restores=len(targets), recurrent_restores=len(states), peak_resident_permits=peak)
    return dict(**proof, result_sha256=cap.sha(result_path))


def qualification_proofs(manifest):
    proofs = {}
    for arm in ARMS[1:]:
        expected = manifest['qualifications'][arm]
        path = Path(expected['path'])
        actual = qualification_identity(path, arm, manifest['runtime_sha256'][arm], manifest['native_source'])
        cap.require(actual == expected, 'Qualification provenance changed after preparation')
        proofs[arm] = dict(identity=actual, evidence=validate_completed_qualification(path, arm))
    return proofs


def prepare(args):
    root = args.output.resolve()
    cap.require(not root.exists(), 'Use a fresh output directory')
    configs = profiles(cap.read(args.config), args.port)
    fixture = cap.read(args.fixture)
    from transformers import AutoTokenizer
    eos = AutoTokenizer.from_pretrained(configs['native']['tokenizer'], local_files_only=True).eos_token_id
    cap.require(type(eos) is int and eos >= 0 and fixture.get('eos', eos) == eos,
                'Fixture must use the local tokenizer EOS')
    fixture['eos'] = eos
    budget = fixture_bounds(fixture, configs['native'])
    verify_provenance(fixture)
    native_source = cap.native_source_identity()
    sources = {ARMS[1]: args.baseline.resolve(), ARMS[2]: args.candidate.resolve()}
    runtime_hashes = {arm: cap.hashes(path) for arm, path in sources.items()}
    qualifications = {arm: qualification_identity(path.resolve(), arm, runtime_hashes[arm], native_source)
        for arm, path in ((ARMS[1], args.pipeline_qualification), (ARMS[2], args.rolling_qualification))}
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    for folder in ('src', 'scripts', 'csrc'):
        shutil.copytree(ROOT/folder, root/'frozen'/folder, ignore=ignore)
    for arm, source in sources.items():
        shutil.copytree(source, root/'runtime'/arm, ignore=ignore)
        cap.require(cap.hashes(root/'runtime'/arm) == runtime_hashes[arm], 'Runtime changed during freeze')
    for arm, p in configs.items():
        cap.save(root/(arm+'.json'), p)
    cap.save(root/'fixture.json', fixture)
    cap.save(root/'manifest.json', dict(kind='ready_agents_v1', experiment_kind=EXPERIMENT,
        created_at=time.time(), source_sha256=cap.hashes(root/'frozen'), runtime_sha256=runtime_hashes,
        runtime_sources={arm:str(path) for arm,path in sources.items()}, qualifications=qualifications,
        profiles_sha256={arm:cap.sha(root/(arm+'.json')) for arm in ARMS},
        fixture_sha256=cap.sha(root/'fixture.json'), fixture_source=str(args.fixture.resolve()),
        fixture_source_sha256=cap.sha(args.fixture), native_source=native_source,
        workload=dict(concurrency=USERS, warmup_turns=WARMUP, epoch_turns=TURNS,
            duration_seconds=300, mode='tools', generation_mode='natural', generation_limit=LIMIT),
        generation_sampling=SAMPLING, arms=list(ARMS), port=args.port, idle_seconds=10,
        budget=budget, qualify=False,
        scope='Natural EOS, identical complete recorded external messages/waits, own accepted outputs. '
            '32000-token generation cap; finite 16-turn epochs end normally before the next complete '
            'input+32000 exceeds 256K. No clipping, padding, deadline cancellation, SSD or history recomputation. '
            'Two warmup turns, fixed 300-second window and normal drain. Fresh native deployment comparison; '
            'not on-policy DeepSWE correctness. Single order is provisional until paired repetition.'))
    cap.save(root/'status.json', dict(status='prepared', at=time.time(),
        launch_gate='Both exact-runtime GPU qualifications must complete before any performance launch'))


def verify(root):
    m = ready.verify(root)
    cap.require(m.get('experiment_kind') == EXPERIMENT and m.get('qualify') is False and
        m.get('arms') == list(ARMS) and m.get('generation_sampling') == SAMPLING,
        'Natural experiment identity or oracle controls changed')
    cap.require(m['workload'] == dict(concurrency=USERS, warmup_turns=WARMUP, epoch_turns=TURNS,
        duration_seconds=300, mode='tools', generation_mode='natural', generation_limit=LIMIT),
        'Natural experiment workload changed')
    expected = profiles(cap.read(root/'native.json'), m['port'])
    cap.require(all(cap.read(root/(arm+'.json')) == expected[arm] for arm in ARMS),
                'Arm controls differ beyond qualified rolling option')
    fixture = cap.read(root/'fixture.json')
    cap.require(fixture_bounds(fixture, expected['native']) == m['budget'], 'Full historical budget changed')
    source = Path(m['fixture_source'])
    cap.require(source.is_file() and cap.sha(source) == m['fixture_source_sha256'], 'Source fixture changed')
    verify_provenance(fixture)
    return m


async def run(args):
    root = args.output.resolve()
    m = verify(root)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared', 'No automatic restart')
    # This must precede GPU polling, server launch, and any status mutation.
    proofs = qualification_proofs(m)
    cap.save(root/'qualification-proof.json', dict(at=time.time(), qualifications=proofs))
    loop, active = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, active.cancel)
    started, monotonic = time.time(), time.monotonic()
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            results = {}
            for arm in ARMS:
                verify(root)
                qualification_proofs(m)
                results[arm] = await ready.run_arm(root, arm, m)
                cap.save(root/'comparison.json', {key:dict(output_tps=value['output_tokens_per_second'],
                    completed_turns_per_second=value['requests_completed_in_window']/300,
                    gpu=value['gpu'], preemptions=value['preemptions']) for key,value in results.items()})
            analysis = module('natural_rolling_analysis', 'analyze_rolling_agents.py')
            cap.save(root/'analysis.json', analysis.review(root))
            cap.save(root/'status.json', dict(status='completed_pending_review', at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        cap.save(root/'timing.json', dict(started_wall=started, finished_wall=time.time(),
            elapsed_seconds=time.monotonic()-monotonic, scope='All arms including warmup and normal drain'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    for key in ('config', 'fixture', 'baseline', 'candidate', 'pipeline-qualification', 'rolling-qualification'):
        parser.add_argument('--'+key, type=Path)
    parser.add_argument('--port', type=int, default=18534)
    args = parser.parse_args()
    if args.run_prepared:
        return asyncio.run(run(args))
    cap.require(all((args.config,args.fixture,args.baseline,args.candidate,
        args.pipeline_qualification,args.rolling_qualification)), 'Explicit inputs and both qualifications required')
    cap.require(0 < args.port < 65536, 'Invalid port')
    prepare(args)


if __name__ == '__main__':
    main()

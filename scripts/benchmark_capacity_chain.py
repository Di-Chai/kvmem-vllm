"""Independent G16 context-epoch capacity comparison, never an old-control relabel.

Reuse the original frozen natural workload and qualified rolling runtime. The
native arm can observe actual InputBatch membership with an independently
qualified worker extension. A separately declared native-only measurement can
omit that extension to check its overhead; it cannot claim actual batch evidence.
Preparation requires completed qualification and never launches a GPU process.
"""
import argparse
import ast
import asyncio
import copy
import ctypes
import fcntl
import gc
import importlib.metadata
import importlib.util
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/benchmark_capacity_chain.py'
KIND = 'capacity_g16_context_epoch_v1'
ARMS = ('native', 'rolling-candidate-v2')
ARM_SELECTIONS = {'both': list(ARMS), 'native': [ARMS[0]], 'kvmem': [ARMS[1]]}
HOST_ADMISSION = dict(timeout_seconds=600, sample_seconds=2, consecutive_samples=3,
    launch_margin_bytes=0, phase='after_gpu_process_probe')
WORKLOAD = dict(concurrency=8, warmup_turns=2, epoch_turns=16,
    duration_seconds=300, mode='tools', generation_mode='natural', generation_limit=16384)
EXTRA = (ENTRY, 'scripts/qualify_native_batch_observer.py',
    'scripts/qualify_action_metadata_upload.py', 'scripts/verify_g16_context_epoch_fixture.py',
    'scripts/prepare_g16_context_epoch_fixture.py', 'scripts/freeze_g16_long_history_fixture.py',
    'scripts/benchmark_ready_agents.py')
CACHE_SCOPE = 'Per-user-per-epoch cache_salt; native APC remains enabled within each epoch. No cross-user or cross-epoch prefix sharing.'
SHARED_CACHE_SCOPE = 'Native default APC without cache_salt: identical token prefixes may be reused across users and epochs. Own accepted output histories remain independent.'
RELEASE_TRANSPORT = 'scripts/timed_agent_workload.py'
RELEASE_RUNTIME_FILES = frozenset('src/vllm_kvmem/'+name for name in
    ('capacity.py', 'coordination.py', 'http_sessions.py', 'scheduler.py', 'session_control.py'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


ready = module('capacity_chain_ready', ROOT/'scripts/benchmark_ready_agents.py')
cap, work = ready.cap, ready.work
build_native_launch = ready.native.launch.build_launch


def helper(name):
    # The fixture helpers use sibling imports; never substitute a working-tree
    # helper when this entry is running from its frozen directory.
    directory = str(ROOT/'scripts')
    if not sys.path or sys.path[0] != directory:
        sys.path.insert(0, directory)
    return module('capacity_chain_'+name, ROOT/'scripts'/name)


def observer_enabled(manifest):
    enabled = manifest.get('native_observer_enabled', True)
    cap.require(type(enabled) is bool, 'Native observer mode must be an explicit boolean')
    cap.require(enabled or manifest.get('arms') == ['native'],
                'Uninstrumented confirmation is a separately declared native-only arm')
    return enabled


def native_cache_scope(manifest):
    shared = manifest.get('native_shared_prefix_cache', False)
    cap.require(type(shared) is bool, 'Native prefix sharing must be an explicit boolean')
    cap.require(not shared or manifest.get('arms') == ['native'],
                'Shared-prefix confirmation is a separately declared native-only arm')
    return SHARED_CACHE_SCOPE if shared else CACHE_SCOPE


class SharedNativePrefixClient:
    """Use default native APC for this public fixture, leaving tokens untouched.

    This adapter is only installed for the declared native-only control. The
    frozen transport still owns request IDs, histories, sampling, timing and
    output accounting. KVMem session namespaces are never sent through here.
    """
    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def stream(self, method, url, **kwargs):
        cap.require(method.upper() == 'POST' and url.endswith('/v1/completions'),
                    'Unexpected endpoint in native prefix-sharing control')
        payload = kwargs.get('json')
        cap.require(isinstance(payload, dict) and isinstance(payload.get('cache_salt'), str)
                    and payload['cache_salt'] and
                    not any(k.startswith('kvmem_') for k in payload),
                    'Expected a native completion with no KVMem namespace controls')
        return self.client.stream(method, url, **dict(kwargs,
            json={k:v for k,v in payload.items() if k != 'cache_salt'}))


def profiles(reference, port, native_max_num_seqs=8, native_batch_tokens=None,
             native_gpu_memory_utilization=.915):
    cap.require(type(port) is int and 1024 <= port <= 65535, 'Dedicated loopback port required')
    cap.require(type(native_max_num_seqs) is int and native_max_num_seqs in (7,8),
        'Only original eight-slot or capacity-pressure seven-slot native admission is supported')
    cap.require(native_batch_tokens is None or
        (type(native_batch_tokens) is int and native_batch_tokens in (2048,4096)),
        'Explicit native prefill tuning supports only2048 or4096 total batch tokens')
    cap.require(type(native_gpu_memory_utilization) is float and
        native_gpu_memory_utilization in (.915,.93),
        'Native memory control supports only declared .915 or .93 fractions')
    result = {}
    for arm in ARMS:
        p = copy.deepcopy(cap.read(reference/(arm+'.json')))
        cap.require(p.get('max_num_seqs') == 8 and p.get('max_model_len') == 262144
            and p.get('gpu_memory_utilization') == .81 and p.get('draft_method') == 'mtp'
            and p.get('draft_tokens') == 4 and p.get('execution_mode') == 'compiled_graph',
            'Require original eight-slot compiled MTP4 deployment')
        kv = p['kvmem']
        cap.require(kv.get('select_budget') == 65536 and kv.get('gen_budget') == 32768
            and kv.get('archive_backend') == 'cpu' and kv.get('cpu_read_cache_bytes') == 100_000_000_000
            and p.get('verify_session_restore') is False,
            'Keep K64/G32, complete CPU history and original performance controls')
        cap.require(not any(kv.get(k, False) for k in ('async_action_metadata',
            'early_capture_slot_release', 'native_action_metadata', 'action_profiling', 'mean_k_prewarm')),
            'Unrelated runtime candidates are forbidden')
        if arm != 'native':
            cap.require(kv.get('rolling_active_window') is True, 'Require qualified rolling-v2')
        p['port'] = port
        if arm == 'native':
            p['gpu_memory_utilization'] = native_gpu_memory_utilization
            if native_max_num_seqs != 8 or native_batch_tokens is not None:
                p['max_num_seqs'] = native_max_num_seqs
                # Native serving executes no private Query replay. Remove its
                # derived allowance before explicitly tuning native prefill;
                # a slots-only change keeps the original16424 token budget.
                p['query_replay_budget'] = None
                p['max_num_batched_tokens'] = native_batch_tokens or 16424
        expected_batch = native_batch_tokens if arm == 'native' and native_batch_tokens is not None else 16424
        effective = build_native_launch(p, port=port, async_scheduling=True, environ={})['native_profile']
        cap.require(effective['max_num_batched_tokens'] == expected_batch and effective['enable_prefix_caching']
            and effective['async_scheduling'] and effective['kv_cache_dtype'] == 'fp8_e4m3',
            f'Actual batch{expected_batch} and native async/APC/FP8 must match the declared deployment')
        result[arm] = p
    return result


def profile_delta(reference, port, native_max_num_seqs=8, native_batch_tokens=None,
                  native_gpu_memory_utilization=.915):
    configs = profiles(reference, port, native_max_num_seqs, native_batch_tokens,
                       native_gpu_memory_utilization)
    return {arm:{key:[before.get(key),value] for key,value in configs[arm].items()
                 if before.get(key) != value}
            for arm in ARMS for before in [cap.read(reference/(arm+'.json'))]}


def reference_proof(root, expected=None):
    if expected is not None:
        cap.require(cap.sha(root/'manifest.json') == expected, 'Original natural reference changed')
    original = module('capacity_chain_reference', root/'frozen/scripts/benchmark_rolling_agents.py')
    manifest = original.verify(root)
    cap.require(cap.read(root/'status.json').get('status') == 'completed_pending_review',
        'Original natural reference must have completed')
    cap.require(manifest['workload'] == dict(WORKLOAD, generation_limit=32000),
        'Reference is not the completed original natural32000 protocol')
    proofs = original.qualification_proofs(manifest)
    return manifest, proofs


def installed_native(path):
    identity = cap.read(path); source = Path(identity['source_directory'])
    actual = Path(importlib.metadata.distribution('vllm').locate_file('vllm')).resolve()
    cap.require(source.resolve() == actual and identity.get('files'), 'Full native installation lock required')
    files = {str(p):cap.sha(p) for p in sorted(source.rglob('*')) if p.is_file() and p.suffix in ('.py', '.so')}
    cap.require(files == identity['files'], 'Native installation changed')
    return identity


def device_identity():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.reserved',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    cap.require(len(rows) == 1, 'One measured GPU required')
    uuid, total, reserved = [s.strip() for s in rows[0].split(',')]
    return dict(uuid=uuid, total_mib=int(total), reserved_mib=int(reserved))


def fixture_bounds(fixture, profile):
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.request_plan import RequestPlan
    contract = helper('prepare_g16_context_epoch_fixture.py')
    cap.require(fixture.get('experiment_kind') == contract.EXPERIMENT
        and fixture.get('context_epoch_contract') == contract.CONTRACT
        and fixture.get('protocol') == contract.strict.LIMITS, 'Independent context-epoch contract changed')
    cap.require(len(fixture.get('agent_plans', [])) == 8, 'Exactly the frozen eight tasks are required')
    plans = work.validate_agent_plans(fixture, concurrency=8, warmup_turns=2,
        epoch_turns=16, gen_budget=32768, enforce_output_budget=False)
    cap.require([(p['source_task'], p['source_turn']) for p in plans] == contract.strict.EXPECTED,
        'No task replacement or source-start reselection')
    queries = []
    config = KVMemConfig(select_budget=65536, gen_budget=32768, block_tokens=32,
        sink_blocks=profile['kvmem'].get('sink_blocks', 0), recent_blocks=profile['kvmem'].get('recent_blocks', 0))
    for plan in plans:
        length = len(plan['initial_tokens'])
        cap.require(length+16384 <= 262144 and length+len(plan['turns'][1]['tool_tokens'])+16384 <= 262144,
            'Complete initial/second warmup input cannot fit; no clipping or reduced cap')
        q = RequestPlan.parse(plan['query_plan'], length, 32, 32768, profile['kvmem'].get('guided_query_tokens', 0))
        cap.require(q.query is not None and q.replay_end-q.replay_begin <= profile['query_replay_budget'],
            'Complete Query must fit the original replay budget')
        q.mandatory_fit(length, config)
        queries.append(dict(source_task=plan['source_task'], source_turn=plan['source_turn'],
            initial_tokens=length, query_span=list(q.query)))
    archive = 8*(262144+64)*cap.ARCHIVE_BYTES_PER_TOKEN
    index, extra = 8*cap.INDEX_BYTES_PER_REQUEST, 32*1024**3
    cap.require(archive <= 100_000_000_000 and archive+index+extra == 113834459136,
        'Full eight-user CPU admission must not shrink with shorter outputs')
    return dict(history_limits=[262144]*8, archive_bytes=archive, mean_k_index_bytes=index,
        extra_bytes=extra, minimum_free_bytes=8*1024**3, required_available_bytes=archive+index+extra,
        initial_queries=queries, scope='Full eight-user historical ceiling, staging/index and32GiB extra; no prediction from source/output lengths')


def fixture_proof(directory, profile):
    fixture, qualification, verification = helper('verify_g16_context_epoch_fixture.py').verify_prepared(directory)
    for key in ('source_file_sha256', 'tokenizer_file_sha256'):
        entries = fixture['provenance'].get(key)
        cap.require(isinstance(entries, dict) and entries, 'Complete input provenance required')
        for name, digest in entries.items():
            path = Path(name)
            cap.require(path.is_absolute() and path.is_file() and cap.sha(path) == digest, 'Fixture provenance changed: '+name)
    budget = fixture_bounds(fixture, profile)
    identity = {name:cap.sha(directory/name) for name in
        ('fixture.json', 'qualification.json', 'verification.json', 'capacity-premises.json')}
    return fixture, budget, identity


def observer_proof(root, expected_sha=None):
    root = Path(root)
    gate = module('capacity_chain_observer_gate', root/'frozen/scripts/qualify_native_batch_observer.py')
    digest = cap.sha(root/'diagnostics/native_executed_batch_observer.py')
    if expected_sha is not None:
        cap.require(digest == expected_sha, 'Observer revision changed')
    return gate, gate.completed_proof(root, digest)


def no_cleanup(root):
    cap.require(not list(root.rglob('cleanup-required.json')), 'Unresolved owned process cleanup')


def post_stop_proof(root, arm):
    # The inherited owner stopper waits on the server parent. Verify the device
    # separately: an absent cleanup marker alone cannot prove worker exit.
    state = ready.bench.q.gpu_state()
    proof = dict(at=time.time(), observed_gpu_pids=state['pids'], gpu_empty=not state['pids'],
        scope='Post-owned-stop device observation. Any remaining PID fails closed; never kill an external or unverified process.')
    cap.save(root/arm/'post-stop.json', proof)
    if state['pids']:
        cap.save(root/arm/'cleanup-required.json', proof)
    cap.require(not state['pids'], 'GPU processes remain after owned server stop; manual ownership review required')
    return proof


def observer_model_contract(root, profile):
    qualified = cap.read(root/'profile.json')
    for name in ('model_format', 'model', 'tokenizer', 'hf_config_path', 'draft_method', 'draft_tokens',
                 'draft_quantization', 'execution_mode', 'max_model_len', 'linear_backend_per_quant'):
        cap.require(qualified.get(name) == profile.get(name), 'Observer qualification model contract changed: '+name)


def release_runtime_map(original, qualified):
    production = lambda mapping: {k:v for k,v in mapping.items() if k.startswith(('src/', 'csrc/'))}
    before, after = production(original), production(qualified)
    cap.require(before.keys() == after.keys(), 'Release runtime production file set changed')
    changed = {k for k in before if before[k] != after[k]}
    cap.require(changed == RELEASE_RUNTIME_FILES,
                'Release runtime must contain only the qualified pressure and session-release fixes')
    return {**original, **after}


def release_transport_hash(qualified, candidate):
    # The qualified HTTP implementation, token accounting, request validation
    # and all other helpers retain identical ASTs. Only workload cleanup
    # orchestration gains the opt-in close and its empty final-view expectation.
    before, after = (ast.parse(p.read_text()) for p in (qualified,candidate))
    find = lambda tree: [i for i,node in enumerate(tree.body)
                        if isinstance(node,ast.AsyncFunctionDef) and node.name == 'workload']
    old, new = find(before), find(after)
    cap.require(len(old) == len(new) == 1, 'Exactly one timed workload required')
    after.body[new[0]] = before.body[old[0]]
    cap.require(ast.dump(before) == ast.dump(after), 'Qualified release transport or helpers changed')
    return cap.sha(candidate)


def release_proof(directory, original_runtime, profile):
    directory = Path(directory).resolve()
    gate = module('capacity_chain_release', directory/'frozen/scripts/qualify_session_release.py')
    qualified = gate.verify(directory)
    cap.require(cap.read(directory/'status.json').get('status') == 'passed',
                'Session release GPU qualification has not passed')
    result = cap.read(directory/'results.json')
    counts = (len(result.get('history_proofs', [])), len(result.get('restore_proofs', [])),
              len(result.get('releases', [])), len(result.get('actual_batch_evidence', [])))
    cap.require(result.get('status') == 'passed' and result.get('namespace_count') == 40
        and result.get('cleanup_completed') is True and result.get('checkpoint_deferrals') == 0
        and counts == (80,40,5,5), 'Incomplete session-release qualification')
    reconciled = result.get('native_reconciliation', {})
    cap.require(reconciled.get('passed') is True and reconciled.get('preemptions') == 0
        and reconciled.get('client_tokens') == reconciled.get('native_tokens') == 41000,
        'Release qualification output accounting failed')
    for row in result['restore_proofs']:
        cap.require(row.get('target') and row.get('recurrent') and
            all(r.get('byte_equal') is True and r.get('working_attention_layers',0)>0 and
                r.get('mtp_attention_layers',0)>0 for r in row['target']) and
            all(r.get('recurrent_byte_equal') is True for r in row['recurrent']),
            'Release qualification lacks per-continuation byte evidence')
    for wave in range(5):
        raw = cap.read(directory/f'workload/wave-{wave}-released.json')
        cap.require(gate.completely_released(raw['resources'],raw['archive']),
                    'Release qualification retained resources')
    qualified_profile = cap.read(directory/'profile.json')
    cap.require(qualified_profile.get('max_model_len') == 196608 and profile.get('max_model_len') == 262144,
                'Declare the192Ki qualification to256Ki measurement ceiling change')
    expected = dict(qualified_profile,port=profile['port'],verify_session_restore=False,max_model_len=262144)
    cap.require(profile == expected, 'Measurement changed an unqualified release deployment field')
    runtime = release_runtime_map(original_runtime, qualified['runtime_sha256'])
    return dict(path=str(directory),manifest_sha256=cap.sha(directory/'manifest.json'),
        results_sha256=cap.sha(directory/'results.json'),runtime_sha256=runtime,
        qualified_transport_sha256=qualified['source_sha256'][RELEASE_TRANSPORT],
        transport_sha256=release_transport_hash(directory/'frozen'/RELEASE_TRANSPORT,ROOT/RELEASE_TRANSPORT),
        declared_ceiling_delta=[196608,262144],
        scope='GPU-qualified production fixes; original256Ki comparison ceiling retained. '
              'Qualification histories reach100977 tokens, not256Ki; measurement must independently '
              'prove every completed full history, zero deferrals and final release.')


def require_terminal_history(archive):
    cap.require(type(archive.get('session_checkpoint_deferrals')) is int and
        archive['session_checkpoint_deferrals'] == 0, 'Checkpoint deferral violates no-recompute contract')
    for row in archive.get('completed', []):
        if (row.get('checkpoint_kind') in ('P','M') and
                (row.get('terminal_prefix') or {}).get('status') in ('FINISHED_STOPPED','FINISHED_LENGTH_CAPPED')):
            cap.require(row['checkpoint_kind'] == 'M' and row.get('history_end') == row.get('checkpoint_tokens'),
                        'Completed session did not retain its full terminal history')


async def release_history_monitor(capture):
    while True:
        path = capture/'archive.json'
        if path.exists():
            require_terminal_history(cap.read(path))
        await asyncio.sleep(1)


async def released_workload(original, client, profile, fixture, directory, values, qualification):
    """Only cleanup gains release intent; all measured inputs/waits stay original."""
    selected = SimpleNamespace(**{**vars(values),'explicit_session_release':True})
    result = await ready.supervise_monitored(
        original(client,profile,fixture,directory,selected),
        release_history_monitor(values.capture_directory), monitor_name='full CPU history')
    rows = cap.read(directory/'traces.json')
    closes = [r for r in rows if r['kind'] == 'cleanup']
    cap.require(closes and all(r.get('explicit_session_release') is True for r in closes),
                'Every KVMem epoch cleanup must explicitly release its namespace')
    gate = module('capacity_chain_final_release',
        Path(qualification['path'])/'frozen/scripts/qualify_session_release.py')
    released = await gate.wait_release(values.capture_directory,max(r['begin'] for r in closes))
    require_terminal_history(released['archive'])
    cap.save(directory/'explicit-release-proof.json',released)
    result['explicit_session_release_verified'] = True
    return result


def prepare(args):
    root, reference = args.output.resolve(), args.reference.resolve()
    cap.require(not root.exists(), 'Use a fresh directory; no retry or overwrite')
    original, proofs = reference_proof(reference)
    native_slots = getattr(args, 'native_max_num_seqs', 8)
    native_batch = getattr(args, 'native_batch_tokens', None)
    native_memory = getattr(args, 'native_gpu_memory_utilization', .915)
    selected = ARM_SELECTIONS[getattr(args, 'arms', 'both')]
    observed_native = observer_enabled(dict(arms=selected,
        native_observer_enabled=getattr(args, 'native_observer', True)))
    shared_native = getattr(args, 'native_shared_prefix_cache', False)
    cache_scope = native_cache_scope(dict(arms=selected,native_shared_prefix_cache=shared_native))
    configs = profiles(reference, args.port, native_slots, native_batch, native_memory)
    gate, observed = observer_proof(args.observer_qualification.resolve())  # Before creating output.
    observer_model_contract(args.observer_qualification, configs['native'])
    fixture, budget, identity = fixture_proof(args.fixture_directory.resolve(), configs[ARMS[1]])
    release = None
    if getattr(args, 'release_qualification', None) is not None:
        cap.require(selected == [ARMS[1]], 'Fresh qualified release runs only the KVMem arm')
        release = release_proof(args.release_qualification, original['runtime_sha256'][ARMS[1]], configs[ARMS[1]])
    native = installed_native(args.installed_native)
    cap.require(cap.sha(args.installed_native) == observed['installed_native_sha256'],
        'Observer qualification used another native installation lock')
    device = device_identity()
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache')
    shutil.copytree(reference/'frozen', root/'frozen', ignore=ignore)
    for name in EXTRA:
        source = (args.observer_qualification/'frozen'/name if name in EXTRA[1:3] else ROOT/name)
        shutil.copyfile(source, root/'frozen'/name)
    shutil.copytree(reference/'runtime'/ARMS[1], root/'runtime'/ARMS[1], ignore=ignore)
    if release is not None:
        qualification = Path(release['path'])
        shutil.copyfile(ROOT/RELEASE_TRANSPORT,root/'frozen'/RELEASE_TRANSPORT)
        for name in RELEASE_RUNTIME_FILES:
            shutil.copyfile(qualification/'runtime'/name,root/'runtime'/ARMS[1]/name)
    shutil.copytree(args.observer_qualification/'diagnostics', root/'diagnostics', ignore=ignore)
    shutil.copyfile(args.fixture_directory/'fixture.json', root/'fixture.json')
    shutil.copyfile(args.installed_native, root/'installed-native-identity.json')
    for arm, p in configs.items(): cap.save(root/(arm+'.json'), p)
    source = Path(native['source_directory'])
    cap.save(root/'observer.json', dict(schema='native_batch_observer_v1', enabled=True,
        output_directory=str(root/'native/observation'),
        source_sha256={name:native['files'][str(source/name)] for name in gate.SOURCE_FILES}))
    manifest = dict(kind='ready_agents_v1', experiment_kind=KIND, created_at=time.time(),
        reference=str(reference), reference_manifest_sha256=cap.sha(reference/'manifest.json'),
        source_sha256=cap.hashes(root/'frozen'), extra_source_sha256={n:cap.sha(root/'frozen'/n) for n in EXTRA},
        runtime_sha256={ARMS[1]:original['runtime_sha256'][ARMS[1]]},
        profiles_sha256={arm:cap.sha(root/(arm+'.json')) for arm in ARMS},
        fixture_sha256=cap.sha(root/'fixture.json'), fixture_directory=str(args.fixture_directory.resolve()),
        fixture_identity=identity, workload=WORKLOAD, generation_sampling=original['generation_sampling'],
        budget=budget, arms=selected, host_admission=HOST_ADMISSION,
        qualify=False, port=args.port, idle_seconds=original['idle_seconds'],
        native_source=original['native_source'], installed_native_sha256=cap.sha(root/'installed-native-identity.json'),
        device_memory_identity=device, minimum_device_free_mib=4096,
        observer_qualification=str(args.observer_qualification.resolve()), observer_qualification_proof=observed,
        diagnostic_sha256=cap.hashes(root/'diagnostics'), observer_config_sha256=cap.sha(root/'observer.json'),
        native_cache_scope=cache_scope,
        native_shared_prefix_cache=shared_native,
        native_max_num_seqs=native_slots,
        native_batch_tokens=native_batch,
        native_gpu_memory_utilization=native_memory,
        native_observer_enabled=observed_native,
        declared_profile_delta=profile_delta(reference, args.port, native_slots, native_batch, native_memory),
        protocol_delta=dict(reference_request_cap=32000, request_cap=16384, source_cohort='Fixed independent G16 eight tasks/starts',
            epochs='Maximum16; each arm own full output context guard, no strict16 guarantee; original strict16 rejection retained'),
        capacity_scope='Record actual allocator pool and worker membership under the declared APC mode. '
            'Do not treat planning pages as an instantaneous lower bound or assume a batch ceiling from an older pool. '
            +cache_scope,
        scope='Independent natural G16 context-epoch replay: exact qualified rolling-v2 .81 versus native '
            f'{native_memory} async/APC '
            'with explicitly declared native batch budget and observer mode. Original inputs/tools/waits/sampling, '
            'own complete accepted IDs,2warmup/300s, full CPU admission, normal drain; not original first8/32000 or task quality.')
    manifest['scope'] += (' Native throughput includes the qualified InputBatch observer overhead.'
        if observed_native else ' Native worker has no batch observer; no actual batch membership claim from this run.')
    manifest['scope'] += ' '+cache_scope
    if release is not None:
        manifest['release_qualification'] = release
        manifest['runtime_sha256'] = {ARMS[1]:release['runtime_sha256']}
        manifest['extra_source_sha256'][RELEASE_TRANSPORT] = release['transport_sha256']
        manifest['scope'] += ' KVMem uses the separately declared qualified pressure/session-release fixes and explicit epoch cleanup.'
    cap.require(cap.hashes(root/'runtime'/ARMS[1]) == manifest['runtime_sha256'][ARMS[1]], 'Runtime changed during freezing')
    cap.save(root/'manifest.json', manifest)
    cap.save(root/'reference-qualification-proof.json', proofs)
    cap.save(root/'status.json', dict(status='prepared', gpu_started=False, at=time.time()))


def verify(root):
    m = ready.verify(root)
    observer_enabled(m)
    cap.require(m.get('experiment_kind') == KIND and m['arms'] in ARM_SELECTIONS.values() and m['qualify'] is False
        and m['workload'] == WORKLOAD and m['native_cache_scope'] == native_cache_scope(m),
        'Independent G16 protocol changed; oldfirst8/32000 identity is not valid here')
    cap.require(m.get('host_admission') == HOST_ADMISSION, 'Host admission waiting policy changed')
    reference = Path(m['reference']); original, _ = reference_proof(reference, m['reference_manifest_sha256'])
    release = m.get('release_qualification')
    expected_extra = set(EXTRA) | ({RELEASE_TRANSPORT} if release is not None else set())
    cap.require(set(m['extra_source_sha256']) == expected_extra
        and m['source_sha256'] == {**original['source_sha256'], **m['extra_source_sha256']},
        'Original benchmark sources changed beyond explicit wrapper/helpers')
    expected_runtime = release['runtime_sha256'] if release is not None else original['runtime_sha256'][ARMS[1]]
    cap.require(m['runtime_sha256'] == {ARMS[1]:expected_runtime}
        and m['generation_sampling'] == original['generation_sampling']
        and m['idle_seconds'] == original['idle_seconds'] and m['native_source'] == original['native_source'],
        'Qualified runtime, sampling or native source changed')
    native_slots = m.get('native_max_num_seqs', 8)
    native_batch = m.get('native_batch_tokens')
    native_memory = m.get('native_gpu_memory_utilization', .915)
    expected = profiles(reference, m['port'], native_slots, native_batch, native_memory)
    if release is not None:
        cap.require(m['arms'] == [ARMS[1]] and release == release_proof(
            release['path'],original['runtime_sha256'][ARMS[1]],expected[ARMS[1]]), 'Release proof changed')
        cap.require(m['extra_source_sha256'][RELEASE_TRANSPORT] == release['transport_sha256'],
                    'Explicit release transport differs from GPU qualification')
    cap.require(all(cap.read(root/(arm+'.json')) == expected[arm] for arm in ARMS), 'Undeclared profile change')
    delta = profile_delta(reference, m['port'], native_slots, native_batch, native_memory)
    cap.require(m['declared_profile_delta'] == delta, 'Deployment delta declaration changed')
    fixture, budget, identity = fixture_proof(Path(m['fixture_directory']), expected[ARMS[1]])
    cap.require(identity == m['fixture_identity'] and m['fixture_sha256'] == identity['fixture.json']
        and budget == m['budget'], 'Complete fixture or full CPU budget changed')
    cap.require(cap.hashes(root/'diagnostics') == m['diagnostic_sha256']
        and cap.sha(root/'observer.json') == m['observer_config_sha256'], 'Observer or configuration changed')
    gate, proof = observer_proof(Path(m['observer_qualification']), m['diagnostic_sha256']['native_executed_batch_observer.py'])
    observer_model_contract(Path(m['observer_qualification']), expected['native'])
    cap.require(proof == m['observer_qualification_proof'], 'Completed observer proof changed')
    for name in EXTRA[1:3]:
        cap.require(cap.sha(root/'frozen'/name) == cap.sha(Path(m['observer_qualification'])/'frozen'/name),
            'Observer helper differs from qualified version')
    cap.require(cap.sha(root/'installed-native-identity.json') == m['installed_native_sha256'] == proof['installed_native_sha256'],
        'Native installation lock changed')
    native = installed_native(root/'installed-native-identity.json'); source = Path(native['source_directory'])
    cap.require(cap.read(root/'observer.json') == dict(schema='native_batch_observer_v1', enabled=True,
        output_directory=str(root/'native/observation'),
        source_sha256={n:native['files'][str(source/n)] for n in gate.SOURCE_FILES}), 'Observer ownership/hook sources changed')
    cap.require(m['minimum_device_free_mib'] == 4096 and device_identity() == m['device_memory_identity'],
        'Physical device or all-phase headroom contract changed')
    return m


def own_history_proof(fixture, rows, events):
    """Complete own IDs and legal finite epochs, including zero-output EOS."""
    cap.require(len(fixture['agent_plans']) == 8 and rows
        and len({r['request_id'] for r in rows}) == len(rows), 'Eight users and unique complete requests required')
    verified = []
    for user, plan in enumerate(fixture['agent_plans']):
        own = sorted((r for r in rows if r['agent'] == user), key=lambda r:r['begin'])
        cap.require(own and own[0]['kind'] == own[-1]['kind'] == 'cleanup'
            and len({r['session'] for r in own}) == 1, 'Initial/final normal cleanup and stable user identity required')
        previous = None; cleaned = True; warmup = []
        for r in own:
            cap.require(r['status'] == 'completed' and r['end'] >= r['begin']
                and all(type(x) is int and x >= 0 for x in r['output_ids']), 'Incomplete accepted stream')
            count = len(r['output_ids'])
            if r['kind'] == 'cleanup':
                tokens = [fixture['eos']]
                cap.require(count == r['declared_generation_ceiling'] == 1 and r['generation_mode'] == 'fixed', 'Cleanup must remain fixed one token')
                cleaned = True
            else:
                cap.require(r['kind'] in ('warmup', 'workload') and r['generation_mode'] == 'natural'
                    and r['declared_generation_ceiling'] == 16384 and count <= 16384
                    and r['finish_reason'] in ('stop', 'length')
                    and (r['finish_reason'] != 'length' or count == 16384), 'Natural request ceiling or completion changed')
                turn, epoch = r['turn'], r['epoch']
                cap.require(type(turn) is int and 0 <= turn < 16 and type(epoch) is int and epoch >= 0, 'Invalid finite epoch')
                if previous is None:
                    cap.require(turn == epoch == 0, 'Initial epoch must begin at original source start')
                    tokens = list(plan['initial_tokens'])
                elif epoch == previous['epoch']:
                    cap.require(not cleaned and turn == previous['turn']+1, 'History skipped or unexpectedly cleaned')
                    tokens = previous['tokens']+plan['turns'][turn]['tool_tokens']
                else:
                    cap.require(cleaned and epoch == previous['epoch']+1 and turn == 0, 'Epoch replacement requires normal cleanup')
                    next_turn = previous['turn']+1
                    next_length = None if next_turn == 16 else len(previous['tokens'])+len(plan['turns'][next_turn]['tool_tokens'])
                    reason = 'declared_turn_limit' if next_turn == 16 else 'context_capacity'
                    cap.require(next_turn == 16 or next_length+16384 > 262144, 'Premature history reset before context/turn limit')
                    matches = [e for e in events if e.get('event') == 'epoch_ended' and e.get('agent') == user and e.get('epoch') == previous['epoch']]
                    cap.require(len(matches) == 1 and matches[0]['reason'] == reason
                        and matches[0]['completed_turns'] == next_turn and matches[0]['retained_history_tokens'] == len(previous['tokens'])
                        and matches[0]['next_prompt_tokens'] == next_length
                        and matches[0]['generation_limit'] == 16384 and matches[0]['max_model_len'] == 262144,
                        'Context/turn epoch-end evidence changed')
                    tokens = list(plan['initial_tokens'])
                cap.require(len(tokens)+16384 <= 262144, 'A sent request exceeded full-context guard')
                cap.require(r['source_metadata'] == dict(source_task=plan['source_task'], source_turn=plan['turns'][turn]['source_turn']), 'Frozen external turn exposure changed')
                if r['kind'] == 'warmup': warmup.append((epoch, turn))
                previous = dict(epoch=epoch, turn=turn, tokens=tokens+r['output_ids']); cleaned = False
            cap.require(r['input_tokens'] == len(tokens) and r['input_sha256'] == work.digest(tokens)
                and r['usage']['prompt_tokens'] == len(tokens) and r['usage']['completion_tokens'] == count
                and r['output_sha256'] == work.digest(r['output_ids']), 'Full own-history/accepted IDs do not reconcile')
            verified.append(r['request_id'])
        cap.require(warmup == [(0, 0), (0, 1)], 'Every user must complete two original warmup turns')
    cap.require(len(verified) == len(rows), 'Unexpected agent outside frozen cohort')
    return dict(passed=True, requests=len(verified), complete_own_outputs=True, normal_cleanup=True,
        warmup_requests=16, request_ceiling=16384, maximum_epoch_turns=16,
        scope='Complete input hashes and own accepted IDs; legal context/turn epochs, never pressure history deletion')


def deployment_verdict(root, arm, m):
    samples = ready.records(root/arm/'gpu_samples.jsonl')
    cap.require(samples, 'All-phase device samples required')
    peak = max(samples, key=lambda r:r['memory_mib']); device = m['device_memory_identity']
    free = device['total_mib']-device['reserved_mib']-peak['memory_mib']
    native = ready.records(root/arm/'native_samples.jsonl')
    cap.require(native, 'Actual native metrics required')
    preemptions = max(r['values']['vllm:num_preemptions_total'] for r in native)
    cache = [line for r in native for line in r['raw_metrics'].splitlines() if line.startswith('vllm:cache_config_info{')]
    cap.require(cache, 'Actual allocator pool must be observed; do not assume1192 pages')
    geometry = []
    for line in cache:
        labels = dict(re.findall(r'(\w+)="([^"\\]*)"', line))
        cap.require(all(labels.get(k, '').isdigit() and int(labels[k]) > 0
            for k in ('block_size', 'num_gpu_blocks')), 'Actual allocator geometry cannot be parsed')
        geometry.append({k:int(labels[k]) for k in ('block_size', 'num_gpu_blocks')})
    cap.require(all(g == geometry[0] for g in geometry), 'Allocator geometry changed during run')
    return dict(accepted=free >= 4096 and preemptions == 0,
        reason=None if free >= 4096 and preemptions == 0 else 'headroom_below4GiB' if free < 4096 else 'preemptions',
        minimum_sampled_free_mib=free,
        required_free_mib=4096, peak_wall=peak['wall_at'], maximum_used_mib=peak['memory_mib'],
        full_process_preemptions=preemptions, actual_cache_config_info=sorted(set(cache)),
        actual_configured_pool=geometry[0], measured_null_or_reserved_pages=None,
        scope='All-phase NVML observations can miss short peaks. Negative verdict preserved without retry. '
            'Cache config is measured capacity, not live request occupancy. '+native_cache_scope(m))


def observer_postflight(root, gate):
    d = root/'native'; logs = list((d/'observation').glob('worker-*/events.jsonl'))
    cap.require(len(logs) == 1, 'Exactly one actual worker log required')
    initial, seal, final = (cap.read(d/name) for name in ('observer-before.json', 'observer-seal.json', 'observer-final.json'))
    summary = cap.read(logs[0].parent/'summary.json')
    for value in (seal, final, summary):
        cap.require(value.get('complete_coverage') is True and value.get('explicit_seal') is True
            and value.get('post_close_dispatches') == 0, 'Incomplete observer coverage or model work after seal')
    cap.require(all(summary[k] == final[k] == seal[k] for k in ('dispatches', 'completed', 'failed', 'dropped', 'observer_errors')),
        'Observer final counts changed after seal')
    events, rows, window = ready.records(logs[0]), cap.read(d/'tools/traces.json'), cap.read(d/'tools/window.json')
    coverage = gate.observation_proof(events, rows, seal, initial, max_requests=8, window=window)
    coverage['max_actual_batch_scope'] = 'Lifetime, all kinds, including warmup and cleanup; never formal capacity evidence'
    return dict(coverage=coverage, formal_workload=formal_native_membership(events, rows, window,
        cap.read(root/'fixture.json')), scope='Exact worker IDs for native only; raw sampler counts are not accepted client tokens')


def formal_native_membership(events, rows, window, fixture):
    """Separate long workload decode from warmup/cleanup in completed dispatches."""
    by_response = {r['response_id']:r for r in rows}
    completed = {r['execution_id'] for r in events if r['kind'] == 'completed' and r.get('executed') is True}
    pure = {}; mixed = {}; records = []; excluded = 0
    for record in events:
        if record['kind'] != 'dispatch': continue
        batch = record['batch']; at = batch['dispatch_monotonic']
        if not window['begin'] <= at < window['end']: continue
        cap.require(batch['execution_id'] in completed, 'Only completed dispatches may establish actual batch')
        owners = []
        for identity in batch['req_ids']:
            match = re.fullmatch(r'(.+)-0-[0-9a-f]{8}', identity)
            cap.require(match is not None and match[1] in by_response, 'Actual worker identity lacks complete HTTP history')
            owners.append(by_response[match[1]])
        cap.require(len(owners) == len(batch['is_prefilling']) == len(batch['prompt_lengths']), 'Incomplete worker metadata')
        selected = [i for i, row in enumerate(owners) if not batch['is_prefilling'][i]
            and row['kind'] == 'workload'
            and row['input_tokens'] >= len(fixture['agent_plans'][row['agent']]['initial_tokens'])]
        if not selected:
            excluded += 1; continue
        cap.require(len({owners[i]['agent'] for i in selected}) == len(selected), 'Duplicate agent in actual decode cohort')
        n = len(selected); whole = n == len(owners)
        target = pure if whole else mixed; target[str(n)] = target.get(str(n), 0)+1
        records.append(dict(execution_id=batch['execution_id'], dispatch_monotonic=at,
            pure_long_workload_decode=whole, workload_decode_members=[dict(
                worker_request_id=batch['req_ids'][i], request_id=owners[i]['request_id'],
                agent=owners[i]['agent'], epoch=owners[i]['epoch'], turn=owners[i]['turn'],
                input_history_tokens=owners[i]['input_tokens']) for i in selected],
            other_members=[dict(kind=row['kind'], is_prefilling=batch['is_prefilling'][i])
                           for i,row in enumerate(owners) if i not in selected]))
    return dict(pure_long_workload_decode_histogram=pure,
        mixed_workload_decode_subset_histogram=mixed, excluded_steps_without_long_workload_decode=excluded,
        pure_eight_long_workload_decode_steps=pure.get('8', 0), actual_membership=records,
        scope='Completed worker dispatches started in[T0,T1); workload-only long histories, excluding warmup/cleanup. '
            'Mixed prefill/cleanup steps stay separate. Counts, not GPU-time weights or accepted-token counts.')


def conservative_kvmem_eight(steps, rows, window, fixture, margin=.02):
    """Existing K telemetry has counts, not worker IDs. Do not invent membership."""
    eligible = []; total = 0
    for step in steps:
        if step.get('decode_requests') != 8 or step.get('prefill_requests') or step.get('prefill_tokens'): continue
        begin = step['started_at']; end = begin+step['step_ms']/1000
        if not window['begin'] <= begin < end <= window['end']: continue
        total += 1
        if any(r['kind'] != 'workload' and r['begin']-margin < end and r['end']+margin > begin for r in rows): continue
        active = [r for r in rows if r['kind'] == 'workload' and r['begin']+margin <= begin
            and r['end']-margin >= end and r['input_tokens'] >= len(fixture['agent_plans'][r['agent']]['initial_tokens'])]
        if len(active) != 8 or len({r['agent'] for r in active}) != 8: continue
        eligible.append(dict(started_at=begin, host_anchored_estimated_end=end, step_ms=step['step_ms'],
            possible_client_members=[dict(request_id=r['request_id'], agent=r['agent'],
                epoch=r['epoch'], turn=r['turn'], input_history_tokens=r['input_tokens']) for r in active]))
    return dict(formal_pure_eight_steps=total, conservative_long_workload_eight_steps=len(eligible),
        eligible_steps=eligible, membership='inferred_from_count_and_exclusive_HTTP_intervals_not_worker_IDs',
        scope='Only full host-anchored step estimates within[T0,T1), all8 distinct long-workload clients in flight '
            'throughout, no warmup/cleanup overlap with20ms margin. CUDA-event duration plus host start is only '
            'an interval estimate, not exact physical GPU timing or occupancy. No inferred throughput/capacity theorem.')


def release_controller_heap():
    """Release only this controller's unused Python/libc heap between arms."""
    collected = gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
        trimmed = bool(trim(0))
    except (AttributeError, OSError):
        trimmed = None
    return dict(collected=collected, malloc_trim_released=trimmed,
        scope='Own controller only; no system cache eviction, foreign process changes or reduced budgets')


async def wait_host_headroom(root, arm, manifest):
    """Wait above the full-history floor, including transient launch checks.

    This wait runs after the process-based GPU checks. Optional additional
    margin is never deducted from the history budget. The final host check and
    live memory guard remain in force.
    """
    policy = manifest['host_admission']
    required = manifest['budget']['required_available_bytes']
    margin = policy.get('launch_margin_bytes', 0)
    cap.require(type(margin) is int and margin >= 0, 'Invalid host launch margin')
    admission_floor = required+margin
    deadline = time.monotonic()+policy['timeout_seconds']
    proof = dict(arm=arm, required_available_bytes=required,
        launch_margin_bytes=margin, admission_floor_bytes=admission_floor,
        cleanup=release_controller_heap(), samples=[], passed=False)
    target = root/(arm+'-host-admission.json')
    stable = 0
    try:
        while True:
            available = cap.host_memory()['MemAvailable']
            stable = stable+1 if available >= admission_floor else 0
            proof['samples'].append(dict(at=time.time(), available_bytes=available,
                consecutive_qualifying_samples=stable))
            if stable >= policy['consecutive_samples']:
                proof['passed'] = True
                return proof
            cap.save(target, proof)
            cap.save(root/'status.json', dict(status='waiting_host_headroom', arm=arm,
                at=time.time(), available_bytes=available, required_available_bytes=required,
                admission_floor_bytes=admission_floor))
            remaining = deadline-time.monotonic()
            cap.require(remaining > 0, 'Insufficient host headroom after bounded wait; full budget unchanged')
            await asyncio.sleep(min(policy['sample_seconds'], remaining))
    finally:
        cap.save(target, proof)


async def prelaunch_admission(root, arm, manifest):
    # Initialize the in-process device API before measuring host memory. Unlike
    # a second nvidia-smi invocation, the final check does not fork a process
    # between the stable host samples and run_arm's final memory floor check.
    import pynvml
    pynvml.nvmlInit()
    try:
        device = pynvml.nvmlDeviceGetHandleByIndex(0)
        await wait_host_headroom(root, arm, manifest)
        cap.require(not pynvml.nvmlDeviceGetComputeRunningProcesses(device),
                    'Foreign GPU process appeared during host admission')
    finally:
        pynvml.nvmlShutdown()


async def run(args):
    root = args.output.resolve(); m = verify(root)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared', 'No automatic retry')
    observed_native = observer_enabled(m)
    gate = helper('qualify_native_batch_observer.py')
    old_wait, old_launch, old_workload = ready.bench.q.wait_idle, ready.native.launch.build_launch, work.workload
    active_arm = None
    async def checked_wait(*values, **kwargs):
        await old_wait(*values, **kwargs)
        cap.require(verify(root) == m, 'Identity changed while waiting for GPU')
    def launch(profile, **kwargs):
        original = old_launch(profile, **kwargs)
        if active_arm != 'native' or not observed_native: return original
        actual = gate.inject_observer(original, profile, root/'diagnostics', root/'observer.json')
        cap.save(root/'native/observer-launch.json', {k:v for k,v in actual.items() if k != 'environment'})
        return dict(original, **actual)
    async def native_workload(client, profile, fixture, directory, values):
        shared = m.get('native_shared_prefix_cache', False)
        if shared:
            client = SharedNativePrefixClient(client)
        result = await old_workload(client, profile, fixture, directory, values)
        if shared:
            result['native_cache_salt_scope'] = native_cache_scope(m)
            result['native_transport_delta'] = 'Only omit cache_salt from native completion POSTs'
            cap.save(directory/'results.json', result)
        return result
    async def workload(client, profile, fixture, directory, values):
        if not profile.get('native_vllm'):
            if m.get('release_qualification') is not None:
                return await released_workload(old_workload,client,profile,fixture,directory,values,
                                               m['release_qualification'])
            return await old_workload(client, profile, fixture, directory, values)
        if not observed_native:
            return await native_workload(client, profile, fixture, directory, values)
        url = f"http://{profile['host']}:{profile['port']}"
        cap.save(root/'native/observer-before.json', await gate.rpc(client, url, 'native_executed_batch_snapshot'))
        result = await native_workload(client, profile, fixture, directory, values)
        final = await values.native_quiescent(stage='observer_seal', not_before=time.monotonic())
        cap.require(final['native_metrics']['values']['vllm:generation_tokens_total'] == result['total_stream_output_tokens'],
            'Native/client counts must reconcile before seal after all EOS cleanup')
        cap.save(root/'native/observer-seal.json', await gate.rpc(client, url, 'native_executed_batch_seal'))
        cap.save(root/'native/observer-final.json', await gate.rpc(client, url, 'native_executed_batch_snapshot'))
        return result
    ready.bench.q.wait_idle, ready.native.launch.build_launch, work.workload = checked_wait, launch, workload
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM): loop.add_signal_handler(sig, task.cancel)
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            for arm in m.get('arms', ARMS):
                cap.require(verify(root) == m, 'Identity changed before next arm')
                active_arm = arm
                await ready.run_arm(root, arm, m, prelaunch_check=prelaunch_admission)
                await asyncio.to_thread(post_stop_proof, root, arm)
                directory = root/arm; rows = cap.read(directory/'tools/traces.json')
                proof = own_history_proof(cap.read(root/'fixture.json'), rows, ready.records(directory/'tools/events.jsonl'))
                cap.save(directory/'own-history-proof.json', proof)
                cap.save(directory/'deployment-acceptance.json', deployment_verdict(root, arm, m))
                if arm == 'native' and observed_native:
                    cap.save(directory/'observer-proof.json', observer_postflight(root, gate))
                no_cleanup(root)
        analysis = module('capacity_chain_analysis', ROOT/'scripts/analyze_rolling_agents.py').review(root)
        selected = list(m.get('arms', ARMS))
        cap.require(set(analysis['arms']) == set(selected), 'Analysis arms do not match executed arms')
        if ARMS[1] in selected:
            cap.require(analysis['arms'][ARMS[1]].get('full_history_publication', {}).get('passed') is True,
                'Complete CPU history/terminal M proof missing')
            kd = root/ARMS[1]
            analysis['kvmem_eight_long_workload'] = conservative_kvmem_eight(
                ready.records(kd/'server/capture/model_step_latency.jsonl'), cap.read(kd/'tools/traces.json'),
                cap.read(kd/'tools/window.json'), cap.read(root/'fixture.json'))
        if 'native' in selected:
            if observed_native:
                analysis['native_executed_batch'] = cap.read(root/'native/observer-proof.json')
                analysis['native_executed_batch']['throughput_scope'] = 'Observer-instrumented native; counts only, not time-weighted batch or uninstrumented speed'
            else:
                analysis['native_executed_batch'] = dict(available=False,
                    throughput_scope='Native confirmation without worker batch observer; no actual batch membership measured')
        analysis['deployment_acceptance'] = {arm:cap.read(root/arm/'deployment-acceptance.json') for arm in selected}
        analysis['comparison_complete'] = selected == list(ARMS)
        analysis['comparison_deployment_eligible'] = analysis['comparison_complete'] and all(
            v['accepted'] for v in analysis['deployment_acceptance'].values())
        analysis['deployment_scope'] = 'Raw rates remain visible on negative headroom/preemption results; not an accepted deployment comparison, no automatic retry.'
        cap.save(root/'analysis.json', analysis)
        cap.save(root/'status.json', dict(status='completed_pending_review', at=time.time(),
            comparison_complete=analysis['comparison_complete'],
            comparison_deployment_eligible=analysis['comparison_deployment_eligible']))
    except BaseException as error:
        cap.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise
    finally:
        ready.bench.q.wait_idle, ready.native.launch.build_launch, work.workload = old_wait, old_launch, old_workload
        for sig in (signal.SIGINT, signal.SIGTERM): loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true'); mode.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    for name in ('reference', 'fixture-directory', 'observer-qualification', 'installed-native'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--port', type=int, default=18574)
    parser.add_argument('--arms', choices=tuple(ARM_SELECTIONS), default='both',
        help='Execute a fresh independent arm without repeating or copying an already completed control')
    parser.add_argument('--native-max-num-seqs', type=int, choices=(7,8), default=8,
        help='Keep eight clients and KVMem slots; optionally admit seven native requests to reduce capacity-pressure recompute')
    parser.add_argument('--native-batch-tokens', type=int, choices=(2048,4096),
        help='Explicit native-only prefill batch tuning; omit to retain the original effective16424 budget')
    parser.add_argument('--native-observer', action=argparse.BooleanOptionalAction, default=True,
        help='Disable only for a fresh native-only throughput confirmation; actual batch counts will be unavailable')
    parser.add_argument('--native-shared-prefix-cache', action='store_true',
        help='Native-only control using default APC without per-user/epoch cache salts; original inputs and own output history remain unchanged')
    parser.add_argument('--native-gpu-memory-utilization',type=float,choices=(.915,.93),default=.915,
        help='Declared native allocator fraction; all-phase4GiB free-memory and zero-preemption acceptance remain mandatory')
    parser.add_argument('--release-qualification', type=Path,
        help='KVMem-only: use the GPU-qualified pressure/release fixes and explicit epoch cleanup')
    args = parser.parse_args()
    if args.run_prepared: return asyncio.run(run(args))
    cap.require(all((args.reference, args.fixture_directory, args.observer_qualification, args.installed_native)),
        'Explicit original reference, new G16 fixture, completed observer qualification and native lock required')
    prepare(args)


if __name__ == '__main__':
    main()

"""Freeze and qualify asynchronous Action metadata, never launch during prepare.

Run the four exact CUDA tests first, then the unchanged two-user rolling model
qualification. This is correctness evidence, not a throughput measurement.
Real engine cancellation is explicitly outside this qualification's coverage.
"""
import argparse
import asyncio
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from time import monotonic
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/qualify_action_metadata_upload.py'
HARNESS_TEST = 'tests/test_qualify_action_metadata_upload.py'
KIND = 'action_metadata_upload_qualification_v1'
FILES = tuple('src/vllm_kvmem/'+name for name in
              ('action_attention.py', 'action_metadata_upload.py', 'options.py'))
GPU_TEST = 'test_action_metadata_upload_gpu.py'
CUDA_CASES = 4
ACTION_EVIDENCE_VERSION = 2
CUDA_TIMEOUT_SECONDS = 600
ARENA_BYTES, MAX_LEASES = 8*1024**2, 128


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def module(path):
    spec = importlib.util.spec_from_file_location('action_upload_rolling_reference', path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def reference_proof(reference):
    m = read(reference/'manifest.json')
    require(m.get('kind') == 'rolling_window_qualification_v1' and
            read(reference/'status.json').get('status') == 'completed_pending_review',
            'Require completed original rolling qualification')
    require(inventory(reference/'frozen') == m['source_sha256'] and
            inventory(reference/'runtime') == m['runtime_sha256'], 'Original qualified source changed')
    require(sha(reference/'profile.json') == m['profile_sha256'] and
            sha(reference/'fixture.json') == m['fixture_sha256'], 'Original profile/fixture changed')
    result = read(reference/'results.json')
    require(result.get('status') == 'passed' and result.get('cleanup_completed') is True and
            result.get('native_reconciliation', {}).get('passed') is True and
            result['native_reconciliation'].get('preemptions') == 0,
            'Original qualification lacks completed reconciled cleanup')
    require(m.get('concurrency') == 2 and m.get('output_quotas') == [1024, 4096, 1024],
            'Keep the original two-user qualification quotas')
    require(not (reference/'server/capture/restore_mismatch.json').exists(),
            'Original qualification contains a restore mismatch')
    return m


def candidate_proof(candidate, expected_base, gpu_test):
    m = read(candidate/'manifest.json')
    require(m.get('status') == 'cpu_verified_gpu_pending' and
            m.get('cpu_validation', {}).get('isolated_candidate', {}).get('passed', 0) > 0,
            'Require isolated CPU-verified candidate provenance')
    require(set(m.get('production_files', {})) == set(FILES), 'Candidate must change exactly three files')
    require(inventory(Path(m['base'])) == expected_base, 'Candidate base differs from qualified runtime')
    require(sha(candidate/'candidate.patch') == m['patch_sha256'], 'Candidate patch changed')
    require(m.get('default') is False and m.get('enable') == {'kvmem.async_action_metadata':True},
            'Candidate must be default-off and enable only async Action metadata')
    require(m.get('limits') == dict(combined_pinned_gpu_payload_bytes=ARENA_BYTES,
                                   max_pending_leases=MAX_LEASES), 'Arena limits changed')
    overlay = inventory(candidate/'overlay')
    require(set(overlay) == set(FILES), 'Overlay includes unrelated source')
    expected = dict(expected_base)
    for name in FILES:
        item = m['production_files'][name]
        require(item.get('before_sha256') == expected_base.get(name) and
                overlay[name] == item['after_sha256'], 'Overlay before/after identity mismatch: '+name)
        expected[name] = item['after_sha256']
    require(m.get('cuda_cases_expected') == CUDA_CASES and
            m.get('tests', {}).get('tests/'+GPU_TEST) == sha(gpu_test), 'CUDA oracle identity/count changed')
    return m, expected


def qualification_profile(source, port):
    p = copy.deepcopy(source)
    require(p.get('execution_mode') == 'compiled_graph' and p.get('draft_method') == 'mtp' and
            p.get('draft_tokens') == 4 and p.get('verify_session_restore') is True and
            p.get('max_num_seqs') == 2 and p.get('max_model_len') == 262144,
            'Keep compiled MTP4 and two-user real restore oracle controls')
    kv = p['kvmem']
    require(kv.get('select_budget') == 32768 and kv.get('gen_budget') == 4096 and
            kv.get('rolling_active_window') is True and kv.get('archive_backend') == 'cpu' and
            kv.get('active_resident_max_fraction') == 1e-6 and
            kv.get('cpu_read_cache_bytes') == 100_000_000_000,
            'Keep original active/history budgets and forced cold restore')
    require(all(kv.get(k) is True for k in ('action_attention', 'prefill_attention', 'action_pressure')),
            'Require complete original output/prefill Action scoring')
    require(not any(kv.get(k, False) for k in
                    ('early_capture_slot_release', 'native_action_metadata', 'action_profiling', 'mean_k_prewarm')),
            'Unrelated candidate overlay is forbidden')
    p['port'] = port
    kv['async_action_metadata'] = True
    return p


def action_tokenizer_identity(profile):
    path = Path(profile['tokenizer'])/'tokenizer_config.json'
    decoder = read(path)['added_tokens_decoder']
    ids = []
    for name in ('<tool_call>','</tool_call>','<think>','</think>','<|im_start|>'):
        matches = [int(key) for key,value in decoder.items() if value.get('content') == name]
        require(len(matches) == 1 and matches[0] >= 0, 'Missing or ambiguous Action scope token')
        ids.append(matches[0])
    require(len(set(ids)) == 5, 'Action scope token IDs overlap')
    return dict(tokenizer_config_sha256=sha(path),token_ids=ids)


def prepare(args):
    root, reference, candidate = args.output.resolve(), args.reference.resolve(), args.candidate.resolve()
    require(not root.exists(), 'Use a fresh qualification directory')
    require(1024 <= args.port <= 65535, 'Invalid dedicated server port')
    ref = reference_proof(reference)
    gpu_test = args.gpu_tests.resolve()
    provenance, expected = candidate_proof(candidate, ref['runtime_sha256'], gpu_test)
    profile = qualification_profile(read(reference/'profile.json'), args.port)
    tokenizer_identity = action_tokenizer_identity(profile)
    identity = read(args.installed_native)
    require(identity.get('source_directory') and identity.get('files'), 'Require complete installed native identity')
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(reference/'frozen', root/'frozen', ignore=ignore)
    shutil.copyfile(Path(__file__), root/'frozen'/ENTRY)
    (root/'frozen/tests').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/HARNESS_TEST, root/'frozen'/HARNESS_TEST)
    shutil.copytree(reference/'runtime', root/'runtime', ignore=ignore)
    for relative in FILES:
        shutil.copyfile(candidate/'overlay'/relative, root/'runtime'/relative)
    require(inventory(root/'runtime') == expected, 'Prepared runtime differs beyond reviewed three-file overlay')
    (root/'oracle').mkdir()
    shutil.copyfile(gpu_test, root/'oracle'/GPU_TEST)
    (root/'provenance').mkdir()
    for name in ('manifest.json', 'candidate.patch'):
        shutil.copyfile(candidate/name, root/'provenance'/name)
    shutil.copyfile(args.installed_native, root/'installed-native-identity.json')
    shutil.copyfile(reference/'fixture.json', root/'fixture.json')
    save(root/'profile.json', profile)
    save(root/'manifest.json', dict(kind=KIND, at=time.time(), reference=str(reference),
        reference_manifest_sha256=sha(reference/'manifest.json'),
        reference_result_sha256=sha(reference/'results.json'), candidate_source=str(candidate),
        candidate_manifest_sha256=sha(candidate/'manifest.json'), candidate_patch_sha256=provenance['patch_sha256'],
        source_sha256=inventory(root/'frozen'), runtime_sha256=expected,
        oracle_sha256=inventory(root/'oracle'), provenance_sha256=inventory(root/'provenance'),
        installed_native_sha256=sha(root/'installed-native-identity.json'),
        profile_sha256=sha(root/'profile.json'), fixture_sha256=sha(root/'fixture.json'),
        native_source=ref['native_source'], budget=ref['budget'], idle_seconds=ref['idle_seconds'],
        output_quotas=ref['output_quotas'], concurrency=ref['concurrency'], cuda_cases_expected=CUDA_CASES,
        action_evidence_version=ACTION_EVIDENCE_VERSION,
        action_tokenizer_identity=tokenizer_identity,
        cuda_timeout_seconds=CUDA_TIMEOUT_SECONDS,
        scope='Four exact CUDA cases, then unchanged two-user 1024/4096/1024 rolling model qualification '
              'with FP8 target/MTP/recurrent cold-restore oracles, Action histories and H=M. Not performance.',
        cancellation_scope='Low-level in-flight lease exception only. No real engine request cancellation injection.'))
    save(root/'status.json', dict(status='prepared', at=time.time(), gpu_started=False))


def verify(root, *, frozen_entry=True):
    m = read(root/'manifest.json')
    if frozen_entry:
        require(Path(__file__).resolve() == (root/'frozen'/ENTRY).resolve(), 'Run the frozen qualification entry')
    require(m.get('kind') == KIND and m.get('cuda_cases_expected') == CUDA_CASES and
            m.get('action_evidence_version') == ACTION_EVIDENCE_VERSION and
            m.get('cuda_timeout_seconds') == CUDA_TIMEOUT_SECONDS,
            'Unexpected Action upload qualification kind or oracle count')
    for directory, key in (('frozen', 'source_sha256'), ('runtime', 'runtime_sha256'),
                           ('oracle', 'oracle_sha256'), ('provenance', 'provenance_sha256')):
        require(inventory(root/directory) == m[key], 'Frozen '+directory+' changed')
    for filename, key in (('profile.json', 'profile_sha256'), ('fixture.json', 'fixture_sha256'),
                          ('installed-native-identity.json', 'installed_native_sha256')):
        require(sha(root/filename) == m[key], 'Frozen '+filename+' changed')
    reference = Path(m['reference'])
    require(sha(reference/'manifest.json') == m['reference_manifest_sha256'] and
            sha(reference/'results.json') == m['reference_result_sha256'], 'Reference provenance changed')
    ref = reference_proof(reference)
    candidate = Path(m['candidate_source'])
    require(sha(candidate/'manifest.json') == m['candidate_manifest_sha256'], 'Candidate provenance changed')
    _, expected = candidate_proof(candidate, ref['runtime_sha256'], root/'oracle'/GPU_TEST)
    require(expected == m['runtime_sha256'] and m['candidate_patch_sha256'] == sha(candidate/'candidate.patch'),
            'Candidate identity differs from frozen runtime')
    p = read(root/'profile.json')
    require(p == qualification_profile(read(reference/'profile.json'), p['port']), 'Profile changed beyond flag/port')
    require(m.get('action_tokenizer_identity') == action_tokenizer_identity(p),
            'Frozen Action tokenizer identity changed')
    require(m['fixture_sha256'] == ref['fixture_sha256'] and m['budget'] == ref['budget'] and
            m['native_source'] == ref['native_source'] and m['output_quotas'] == [1024,4096,1024] and
            m['concurrency'] == 2 and m['idle_seconds'] == ref['idle_seconds'], 'Original workload controls changed')
    return m


def installed_native_proof(root):
    identity = read(root/'installed-native-identity.json')
    source = Path(identity['source_directory'])
    spec = importlib.util.find_spec('vllm')
    require(spec is not None and Path(spec.origin).resolve().parent == source.resolve(),
            'Python would import another native vLLM installation')
    actual = {str(p):sha(p) for p in source.rglob('*') if p.is_file() and p.suffix in ('.py','.so')}
    require(actual == identity['files'], 'Installed native execution source changed')


def numerical_evidence(path):
    tree = ET.parse(path).getroot()
    suites = [tree] if tree.tag == 'testsuite' else list(tree.findall('testsuite'))
    totals = {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    require(totals == dict(tests=CUDA_CASES, failures=0, errors=0, skipped=0),
            'All four exact CUDA cases must pass without skips')
    cases = [c.get('name') for s in suites for c in s.findall('testcase')]
    require(len(cases) == CUDA_CASES, 'CUDA XML lacks four actual test case records')
    return dict(**totals, cases=cases, xml_sha256=sha(path))


def native_matches(row, request_id):
    response = row.get('response_id')
    return isinstance(response,str) and re.fullmatch(re.escape(response)+r'-0-[0-9a-f]{8}',request_id) is not None


def action_expected_samples(root, rows, config):
    """Use the frozen host-only C++ parser on the complete accepted history.

    The last generated token has not yet been consumed by the target model;
    require its exact exclusion to agree with the published terminal M state.
    No target/draft CUDA code is imported or executed by this oracle.
    """
    manifest, profile, fixture = (read(root/name) for name in ('manifest.json','profile.json','fixture.json'))
    kv = profile['kvmem']
    require(config.get('scope') == kv.get('action_scope') and
            config.get('interval') == kv.get('action_interval'), 'Action parser settings changed')
    paths = list((root/'runtime/src/vllm_kvmem').glob('_native*.so'))
    require(len(paths) == 1, 'Require one frozen native Action parser')
    path = paths[0]
    native_hash = sha(path)
    require(manifest['runtime_sha256'].get(str(path.relative_to(root/'runtime'))) == native_hash,
            'Frozen native Action parser changed')
    tokenizer_identity = action_tokenizer_identity(profile)
    # Old failed r1 can be inspected offline, but verify()/the performance gate
    # require version 2 and the tokenizer identity frozen before GPU admission.
    frozen_tokenizer = manifest.get('action_evidence_version') == ACTION_EVIDENCE_VERSION
    if frozen_tokenizer:
        require(manifest.get('action_tokenizer_identity') == tokenizer_identity,
                'Frozen Action tokenizer identity changed')
    ids = tokenizer_identity['token_ids']
    # Pybind types are process-global. The model harness may already have
    # imported this exact binary for RequestPlan; loading a second alias then
    # registers ActionParser twice. Reuse the canonical host-only module and
    # verify its bytes against the frozen oracle, never accept another build.
    native = importlib.import_module('vllm_kvmem._native')
    require(sha(Path(native.__file__)) == native_hash, 'Loaded native Action parser differs from frozen oracle')
    digest = lambda tokens: hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
    publications = read(root/'workload/history_publications.json')
    require(manifest.get('output_quotas') == [1024,4096,1024] and
            len(rows) == 6 and {r['agent'] for r in rows} == {0,1} and
            len({r['request_id'] for r in rows}) == 6, 'Action oracle requires six original model requests')
    expected, sessions = {}, set()
    for user, plan in enumerate(fixture['agent_plans'][:2]):
        own = sorted((r for r in rows if r['agent'] == user),key=lambda r:r['turn'])
        require([r['turn'] for r in own] == [0,1,2] and len({r['session'] for r in own}) == 1,
                'Action history turn order or session changed')
        sessions.add(own[0]['session'])
        tokens = list(plan['initial_tokens'])
        for turn,row in enumerate(own):
            if turn:
                tokens.extend(plan['turns'][turn]['tool_tokens'])
            output = row['output_ids']
            require(row.get('status') == 'completed' and row.get('epoch') == 0 and
                    row['input_tokens'] == len(tokens) and row['input_sha256'] == digest(tokens) and
                    len(output) == row['usage']['completion_tokens'] == manifest['output_quotas'][turn] and
                    row['output_sha256'] == digest(output) and
                    all(type(t) is int and t >= 0 for t in tokens+output),
                    'Action oracle full own-history/output chain changed')
            consumed = output[:-1]
            boundary = len(tokens)+len(consumed)
            marks = [p for p in publications if native_matches(row,p['request_id'])]
            require(marks and all(p.get('history_end') == p.get('checkpoint_tokens') == boundary and
                    p.get('checkpoint_kind') == 'M' for p in marks),
                    'Action oracle consumed-output boundary lacks complete terminal M history')
            marker = next((t for t in reversed(tokens) if t in ids[2:]),None)
            thinking = marker == ids[2]
            parser = native.ActionParser(config['scope'],config['interval'],ids[:4],thinking)
            samples = 0
            for at in range(0,len(consumed),32):
                chunk = consumed[at:at+32]
                parser.prepare(chunk)
                samples += parser.commit(len(chunk)).bit_count()
            expected[row['request_id']] = dict(samples=samples,consumed_tokens=len(consumed),
                initial_thinking=thinking,input_sha256=row['input_sha256'],output_sha256=row['output_sha256'])
            tokens.extend(output)
    require(len(sessions) == 2 and len(expected) == 6, 'Action history user namespaces collapsed')
    return expected,dict(native_parser_sha256=native_hash,**tokenizer_identity,
        tokenizer_identity_frozen_before_run=frozen_tokenizer,
        consumed_boundary='own output_ids[:-1], verified against full terminal M')


def action_evidence(capture, rows):
    config = read(capture/'action_config.json')
    require(config.get('async_action_metadata') is True and config.get('prefill_attention') is True and
            config.get('action_pressure') is True and len(config.get('layers',[])) == 3,
            'Actual Action configuration lacks enabled three-layer upload/scoring path')
    actions, prefills = records(capture/'action_attention.jsonl'), records(capture/'prefill_attention.jsonl')
    ordinary = [r for r in rows if r['kind'] == 'workload']
    require(len(ordinary) == 6, 'Require all six original model requests')
    expected, parser_proof = action_expected_samples(capture.parent.parent,ordinary,config)
    proofs = []
    previous, positive = {}, {}
    for row in sorted(ordinary,key=lambda r:(r['agent'],r['turn'])):
        matches = [a for a in actions if native_matches(row,a['request_id'])]
        require(len(matches) == 1, 'Missing or duplicated terminal Action publication')
        a = matches[0]
        old = previous.get(row['agent'],dict(rounds=0,score_sum=0,positive_pages=0,
                                            prefill_samples=0,prefill_chunks=0,prefill_end=0))
        samples = expected[row['request_id']]['samples']
        require(a.get('samples') == samples and a.get('rounds') == old['rounds']+(samples>0) and
                math.isfinite(a.get('score_sum',float('nan'))), 'Incomplete output/prefill Action scoring chain')
        if samples:
            require(a['score_sum'] > 0 and a.get('positive_pages',0) > 0,
                    'Positive Action samples lost their scores')
            positive[row['agent']] = positive.get(row['agent'],0)+1
        else:
            require(all(a.get(k) == old[k] for k in ('rounds','score_sum','positive_pages')),
                    'Zero eligible Action round changed historical scores')
        own_prefill = [p for p in prefills if native_matches(row,p['request_id'])]
        require(own_prefill, 'No committed prefill Action observations for this request')
        n,c,end = old['prefill_samples'],old['prefill_chunks'],old['prefill_end']
        for p in own_prefill:
            require(type(p.get('query_samples')) is int and p['query_samples'] > 0,
                    'Empty committed prefill Action observations')
            n += p['query_samples']; c += 1
            require(p.get('samples') == n and p.get('chunks') == c and p.get('pages',0) > 0 and
                    end < p.get('end',0) <= row['input_tokens'], 'Prefill Action cumulative scoring chain changed')
            end = p['end']
        require(a.get('prefill_samples') == n and a.get('prefill_chunks') == c and
                a.get('prefill_end') == end and a.get('prefill_pages') == own_prefill[-1]['pages'],
                'Incomplete output/prefill Action scoring chain')
        previous[row['agent']] = a
        proofs.append(dict(request_id=row['request_id'],samples=a['samples'],rounds=a['rounds'],
                           score_sum=a['score_sum'],prefill_samples=a['prefill_samples'],prefill_end=a['prefill_end'],
                           expected=expected[row['request_id']]))
    require(set(positive) == {0,1}, 'Each user must exercise actual output Action scoring')
    snapshots = [a.get('metadata_upload_total') for a in actions]
    require(snapshots and all(isinstance(s,dict) for s in snapshots), 'Missing actual upload accounting')
    previous = 0
    for s in snapshots:
        require(s.get('submissions',0) >= previous and s['submissions'] > 0 and s.get('uploaded_bytes',0)>0,
                'No actual asynchronous upload submissions or counters regressed')
        previous = s['submissions']
        require(not s.get('fallbacks') and s.get('quarantined') is False and
                all(s.get(k) == 0 for k in ('failed_submissions','fence_failures','query_failures')),
                'Model qualification used fallback or encountered an upload failure')
        require(s.get('max_combined_arena_bytes') == ARENA_BYTES and s.get('max_pending_leases') == MAX_LEASES and
                s.get('arena_host_bytes') == s.get('arena_gpu_bytes') == ARENA_BYTES//2 and
                0 <= s.get('pending_leases',-1) <= s.get('peak_pending_leases',-1) <= MAX_LEASES and
                0 <= s.get('pending_payload_bytes',-1) <= s.get('peak_pending_payload_bytes',-1) <= ARENA_BYTES and
                s.get('completed',-1)+s['pending_leases'] == s['submissions'], 'Upload arena/credit accounting violated')
    progress = read(capture/'progress.json')
    require(progress.get('full_graph_decode_calls',0)>0, 'No real full CUDA graph decode evidence')
    return dict(version=ACTION_EVIDENCE_VERSION,parser=parser_proof,requests=proofs,
        positive_output_rounds_by_user={str(k):v for k,v in positive.items()},
        upload_final=snapshots[-1],full_graph_decode_calls=progress['full_graph_decode_calls'],
        pending_scope='Completed leases may remain in the bounded arena until the next upload poll; '
                      'the model process is fully stopped after qualification.')


def child_gpu_tests(root):
    m = verify(root)
    owner = read(root/'owner.json')
    require(read(root/'status.json').get('status') == 'low_level_running' and os.getppid() == owner['pid'] and
            os.environ.get('KVMEM_ACTION_UPLOAD_QUAL_OWNER') == owner['token'], 'GPU tests require the owned qualifier')
    import importlib
    for relative in FILES:
        name = 'vllm_kvmem.'+Path(relative).stem
        loaded = importlib.import_module(name)
        require(Path(loaded.__file__).resolve() == (root/'runtime'/relative).resolve(), 'Wrong candidate module imported')
    import pytest
    code = pytest.main(['-c','/dev/null','-q','-p','no:cacheprovider',str(root/'oracle'/GPU_TEST),
                        '--junitxml='+str(root/'oracle.xml')])
    require(code == 0, 'CUDA test subprocess failed')
    numerical_evidence(root/'oracle.xml')


async def low_level(root, m, qual, token):
    import psutil
    await qual.bench.q.wait_idle(root,m['idle_seconds'])
    require(not qual.bench.q.gpu_state()['pids'], 'Foreign GPU process before low-level tests')
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith(('VLLM_','KVMEM_'))}
    env.update(PYTHONPATH=str(root/'runtime/src'),PYTHONDONTWRITEBYTECODE='1',
               KVMEM_ACTION_UPLOAD_QUAL_OWNER=token,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command = [sys.executable,str(root/'frozen'/ENTRY),'--child-gpu-tests','--output',str(root)]
    # Publish the child admission state before spawn; the child may execute
    # before this controller receives its PID and writes the owner record.
    save(root/'status.json',dict(status='low_level_running',at=time.time()))
    with (root/'oracle.log').open('x') as log:
        started = monotonic()
        process = subprocess.Popen(command,cwd=root/'oracle',env=env,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        try:
            # Cleanup owns the process from the instant Popen succeeds, even
            # if recording its identity or status fails.
            save(root/'low-level-owner.json',dict(pid=process.pid,create_time=psutil.Process(process.pid).create_time()))
            save(root/'status.json',dict(status='low_level_running',at=time.time(),pid=process.pid))
            while process.poll() is None:
                elapsed = monotonic()-started
                if elapsed >= m['cuda_timeout_seconds']:
                    save(root/'low-level-timeout.json',dict(pid=process.pid,elapsed_seconds=elapsed,
                        timeout_seconds=m['cuda_timeout_seconds'],at=time.time(),
                        scope='Child startup, CUDA tests and first Triton compilation; no retry.'))
                    raise TimeoutError('Low-level CUDA qualification exceeded its total deadline')
                state = await asyncio.to_thread(qual.bench.q.gpu_state)
                require(not(set(state['pids'])-qual.bench.q.owned_pids(process)), 'Foreign GPU process during oracle')
                require(qual.cap.host_memory()['MemAvailable'] >= m['budget']['minimum_free_bytes'],
                        'Host free memory below qualification minimum')
                with (root/'low-level-gpu-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(at=time.time(),**state))+'\n')
                await asyncio.sleep(1)
            require(process.returncode == 0, 'Low-level CUDA oracle failed; model launch forbidden')
        finally:
            await asyncio.to_thread(qual.bench.q.stop_owned,process)
    proof = numerical_evidence(root/'oracle.xml')
    save(root/'low-level-proof.json',proof)
    return proof


async def run(args):
    import psutil
    root = args.output.resolve()
    m = verify(root)
    require(read(root/'status.json').get('status') == 'prepared', 'No automatic qualification restart')
    installed_native_proof(root)
    # RequestPlan must be resolved from the frozen candidate before importing
    # any helper that could load the older editable installation.
    sys.path.insert(0,str(root/'runtime/src'))
    import vllm_kvmem
    require(Path(vllm_kvmem.__file__).resolve() == (root/'runtime/src/vllm_kvmem/__init__.py').resolve(),
            'Qualification controller imported another KVMem runtime')
    qual = module(root/'frozen/scripts/qualify_rolling_window.py')
    require(qual.cap.native_source_identity() == m['native_source'], 'Native capacity geometry source changed')
    require(qual.validate_fixture(read(root/'fixture.json'),read(root/'profile.json')) == m['budget'],
            'Original request plans/history bounds changed')
    owner = psutil.Process()
    token = uuid.uuid4().hex
    save(root/'owner.json',dict(pid=owner.pid,create_time=owner.create_time(),token=token,at=time.time()))
    original_wait = qual.bench.q.wait_idle
    async def checked_wait(*values, **kwargs):
        await original_wait(*values,**kwargs)
        require(verify(root) == m, 'Frozen inputs changed during GPU admission')
        installed_native_proof(root)
    qual.bench.q.wait_idle = checked_wait
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,task.cancel)
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            low = await low_level(root,m,qual,token)
            require(verify(root) == m, 'Frozen inputs changed after numerical oracle')
            result = await qual.run_server(root,m)
            save(root/'model-results.json',result)
            rows = read(root/'workload/traces.json')
            require(result.get('status') == 'passed' and result.get('cleanup_completed') is True and
                    result.get('native_reconciliation',{}).get('passed') is True and
                    result['native_reconciliation'].get('preemptions') == 0,
                    'Real model did not pass and drain/reconcile')
            result.update(low_level=low,action_upload=action_evidence(root/'server/capture',rows),
                          cancellation_validated=False,low_level_lease_exception_validated=True)
            require(verify(root) == m, 'Frozen inputs changed during model qualification')
            installed_native_proof(root)
            save(root/'results.json',result)
        save(root/'status.json',dict(status='completed_pending_review',at=time.time(),
                                    real_engine_cancellation_validated=False))
    except BaseException as error:
        save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
    finally:
        qual.bench.q.wait_idle = original_wait
        for sig in (signal.SIGINT,signal.SIGTERM):loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--run-prepared',action='store_true')
    mode.add_argument('--verify-prepared',action='store_true')
    mode.add_argument('--child-gpu-tests',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--output',required=True,type=Path)
    for name in ('reference','candidate','gpu-tests','installed-native'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--port',type=int,default=18564)
    args = parser.parse_args()
    if args.run_prepared:return asyncio.run(run(args))
    if args.child_gpu_tests:return child_gpu_tests(args.output.resolve())
    if args.verify_prepared:
        verify(args.output.resolve())
        print(json.dumps(dict(status='verified_not_launched')))
        return
    require(all((args.reference,args.candidate,args.gpu_tests,args.installed_native)),
            'Explicit original qualification, candidate, exact GPU tests and installed native identity required')
    prepare(args)


if __name__ == '__main__':
    main()

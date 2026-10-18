"""One natural eight-agent performance arm after exact Action-upload qualification.

The original frozen workload/launcher/analysis is reused. Preparation itself
requires the new four-case CUDA and real-model qualification to have passed;
ordinary rolling-window qualification is not sufficient.
"""
import argparse
import asyncio
import copy
import fcntl
import importlib.util
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/benchmark_action_metadata_upload.py'
QUALIFIER = 'scripts/qualify_action_metadata_upload.py'
PARSER_IMPORT_ERROR = 'ImportError(\'generic_type: type "ActionParser" is already registered!\')'
ARM = 'rolling-candidate-v2'
KIND = 'action_metadata_upload_performance_v1'
QUALIFICATION_KIND = 'action_metadata_upload_qualification_v1'
FILES = {'src/vllm_kvmem/'+name+'.py' for name in ('action_attention','action_metadata_upload','options')}


def module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


ready = module('action_upload_performance_ready',ROOT/'scripts/benchmark_ready_agents.py')
cap = ready.cap


def performance_profile(source,port):
    cap.require(type(port) is int and 1024 <= port <= 65535,'Invalid dedicated performance port')
    kv = source['kvmem']
    cap.require(source.get('max_num_seqs') == 8 and source.get('max_model_len') == 262144
        and source.get('gpu_memory_utilization') == .81 and source.get('draft_method') == 'mtp'
        and source.get('draft_tokens') == 4 and source.get('execution_mode') == 'compiled_graph'
        and source.get('verify_session_restore') is False and kv.get('select_budget') == 65536
        and kv.get('gen_budget') == 32768 and kv.get('rolling_active_window') is True
        and kv.get('archive_backend') == 'cpu' and kv.get('cpu_read_cache_bytes') == 100_000_000_000,
        'Require original natural eight-agent .81 K64/G32 compiled MTP4 deployment')
    cap.require(all(kv.get(name) is True for name in ('action_attention','prefill_attention','action_pressure'))
        and not any(kv.get(name,False) for name in ('async_action_metadata','early_capture_slot_release',
            'native_action_metadata','action_profiling','mean_k_prewarm')),
        'Reference must retain original Action scoring without unrelated candidates')
    result = copy.deepcopy(source)
    result['port'] = port
    result['kvmem']['async_action_metadata'] = True
    return result


def reference_proof(reference,expected=None):
    if expected is not None:
        cap.require(cap.sha(reference/'manifest.json') == expected,'Natural reference manifest changed')
    natural = module('action_upload_natural_reference',reference/'frozen/scripts/benchmark_rolling_agents.py')
    manifest = natural.verify(reference)
    natural.qualification_proofs(manifest)
    cap.require(manifest['workload'] == dict(concurrency=8,warmup_turns=2,epoch_turns=16,
        duration_seconds=300,mode='tools',generation_mode='natural',generation_limit=32000),
        'Original natural workload changed')
    return manifest


def no_cleanup(root):
    cap.require(not list(root.rglob('cleanup-required.json')),'Unresolved owned cleanup')


def qualification_streams(model,root,result):
    """Recheck six complete original histories, both EOS cleanups and counters."""
    rows = cap.read(root/'workload/traces.json')
    fixture = cap.read(root/'fixture.json')
    cap.require(len(rows) == 8 and len({r['request_id'] for r in rows}) == 8,
                'Require exactly six model turns and two cleanups')
    plans = fixture['agent_plans'][:2]
    cleanups = []
    for user,plan in enumerate(plans):
        own = sorted((r for r in rows if r['agent'] == user),key=lambda r:r['begin'])
        cap.require([r['kind'] for r in own] == ['workload']*3+['cleanup'], 'Qualification turn order changed')
        tokens = list(plan['initial_tokens'])
        for turn,row in enumerate(own[:3]):
            if turn:
                tokens.extend(plan['turns'][turn]['tool_tokens'])
            count = (1024,4096,1024)[turn]
            cap.require(row['turn'] == turn and row['epoch'] == 0 and row['status'] == 'completed'
                and row['input_tokens'] == len(tokens) and row['input_sha256'] == model.work.digest(tokens)
                and len(row['output_ids']) == row['usage']['completion_tokens'] == count
                and row['output_sha256'] == model.work.digest(row['output_ids']),
                'Qualification output or full own-history chain changed')
            tokens.extend(row['output_ids'])
        cleanup = own[-1]
        cap.require(cleanup['status'] == 'completed' and cleanup['input_tokens'] == 1
            and cleanup['input_sha256'] == model.work.digest([fixture['eos']])
            and len(cleanup['output_ids']) == cleanup['usage']['completion_tokens'] == 1,
            'Qualification EOS cleanup incomplete')
        cap.require(len({r['session'] for r in own}) == 1,'Qualification namespace changed')
        cleanups.append(cleanup)
    cap.require(len({r['session'] for r in cleanups}) == 2,'Qualification user namespaces collapsed')
    resources = cap.read(root/'workload/resources-after.json')
    cap.require(model.work.views_match(resources,{r['session']:1 for r in cleanups})
        and resources['at'] >= max(r['begin'] for r in cleanups)
        and all(not resources[key] for key in ('requests','refresh_pending','connector_pending','owned_blocks','reservations'))
        and all(not resources['cpu_history_admission'][key] for key in ('active','retiring','waiting')),
        'Qualification did not publish normal session retirement')
    archive = cap.read(root/'server/capture/archive.json')
    cap.require(archive.get('archive_backend') == 'cpu' and archive.get('repository_gc') is False
        and archive.get('disk_read_bytes') == archive.get('disk_write_bytes') == 0,
        'Qualification history was not complete CPU-only')
    total = sum(len(r['output_ids']) for r in rows)
    native = result.get('native_reconciliation',{})
    cap.require(native.get('passed') is True and native.get('preemptions') == 0
        and native.get('native_tokens') == native.get('client_tokens') == result.get('total_stream_output_tokens') == total,
        'Qualification native/client outputs do not reconcile')
    return rows,dict(model_requests=6,cleanup_requests=2,accepted_tokens=total,normal_cleanup=True)


def completed_qualification(root,base_hashes,native_source,review=None):
    saved = cap.read(root/'manifest.json')
    cap.require(saved.get('kind') == QUALIFICATION_KIND and saved.get('cuda_cases_expected') == 4 and
        saved.get('action_evidence_version') == 2,
        'Require new Action-upload qualification; rolling-only proof cannot admit performance')
    terminal = cap.read(root/'status.json')
    repaired = review is not None
    cap.require((not repaired and terminal.get('status') == 'completed_pending_review') or
                (repaired and terminal.get('status') == 'failed' and terminal.get('error') == PARSER_IMPORT_ERROR),
                'Action-upload qualification must pass before even preparing performance')
    gate = module('action_upload_completed_gate',root/'frozen/scripts/qualify_action_metadata_upload.py')
    manifest = gate.verify(root)
    gate.installed_native_proof(root)
    cap.require(manifest['native_source'] == native_source,'Qualification used another native implementation')
    candidate,expected = gate.candidate_proof(Path(manifest['candidate_source']),base_hashes,root/'oracle'/gate.GPU_TEST)
    cap.require(set(candidate['production_files']) == FILES and manifest['runtime_sha256'] == expected,
                'Qualification runtime is not exactly the reviewed three-file overlay')
    result_file = root/('model-results.json' if repaired else 'results.json')
    result = cap.read(result_file)
    cap.require(result.get('status') == 'passed' and result.get('cleanup_completed') is True,
                'Action-upload model qualification failed or did not clean up')
    low = gate.numerical_evidence(root/'oracle.xml')
    cap.require(low['tests'] == 4 and cap.read(root/'low-level-proof.json') == low and
                (repaired or result.get('low_level') == low),
                'Four exact CUDA cases were not all passed without skips')
    model = gate.module(root/'frozen/scripts/qualify_rolling_window.py')
    rows,streams = qualification_streams(model,root,result)
    review_identity = None
    if repaired:
        review = review.resolve()
        saved_review = cap.read(review)
        cap.require(saved_review.get('original_status') == terminal and
            saved_review.get('qualification_manifest_sha256') == cap.sha(root/'manifest.json') and
            saved_review.get('original_model_result_sha256') == cap.sha(result_file) and
            saved_review.get('verifier_sha256') == cap.sha(ROOT/QUALIFIER),
            'Offline review does not bind the original completed GPU evidence and corrected verifier')
        corrected = module('action_upload_offline_verifier',ROOT/QUALIFIER)
        actions = corrected.action_evidence(root/'server/capture',rows)
        cap.require(saved_review.get('actions') == actions and saved_review.get('low_level') == low,
                    'Offline Action evidence changed or no longer passes')
        # Preserve the original failed attempt. Only its final host-only parser
        # import failed; every GPU, ownership, history and scoring gate is still
        # recomputed below. No generic failed-qualification bypass is allowed.
        result = dict(result,low_level=low,action_upload=actions,cancellation_validated=False,
                      low_level_lease_exception_validated=True)
        review_identity = dict(path=str(review),sha256=cap.sha(review),
            verifier_sha256=cap.sha(ROOT/QUALIFIER),original_status=terminal,
            scope='Offline completion after exact parser-import failure; no GPU rerun, no original records rewritten')
    else:
        actions = gate.action_evidence(root/'server/capture',rows)
    cap.require(result.get('action_upload') == actions,'Actual Action/prefill upload evidence changed')
    oracle = model.oracle_evidence(root/'server/capture',rows)
    cap.require(result.get('oracle') == oracle,'Cold target/MTP/recurrent and accepted-boundary oracle changed')
    ordinary = [r for r in rows if r['kind'] == 'workload']
    histories = model.history_evidence(ordinary,cap.read(root/'workload/history_publications.json'))
    cap.require(len(histories) == len(result.get('history_proofs',[])) == 6 and
        {(r['request_id'],r['history_tokens']) for r in histories} ==
        {(r['request_id'],r['history_tokens']) for r in result['history_proofs']},
        'Complete historical KV/terminal M publication changed')
    cap.require(result.get('cancellation_validated') is False
        and result.get('low_level_lease_exception_validated') is True,
        'Do not claim real-engine cancellation from low-level lease exception tests')
    no_cleanup(root)
    identity = dict(path=str(root),manifest_sha256=cap.sha(root/'manifest.json'),
        result_sha256=cap.sha(result_file),runtime_sha256=expected,
        installed_native_sha256=manifest['installed_native_sha256'],candidate_manifest_sha256=manifest['candidate_manifest_sha256'],
        candidate_patch_sha256=manifest['candidate_patch_sha256'],offline_review=review_identity)
    return identity,dict(low_level=low,streams=streams,actions=actions,oracle=oracle,histories=histories,
        real_engine_cancellation_validated=False,offline_review=review_identity)


def prepare(args):
    root,reference,qualification = args.output.resolve(),args.reference.resolve(),args.qualification.resolve()
    cap.require(not root.exists() and reference not in root.parents and qualification not in root.parents,
                'Use a fresh independent performance directory')
    original = reference_proof(reference)
    identity,proof = completed_qualification(qualification,original['runtime_sha256'][ARM],original['native_source'],
        getattr(args,'qualification_review',None))
    profile = performance_profile(cap.read(reference/(ARM+'.json')),args.port)
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache')
    shutil.copytree(reference/'frozen',root/'frozen',ignore=ignore)
    shutil.copyfile(Path(__file__),root/'frozen'/ENTRY)
    shutil.copyfile(ROOT/QUALIFIER,root/'frozen'/QUALIFIER)
    shutil.copytree(qualification/'runtime',root/'runtime'/ARM,ignore=ignore)
    shutil.copyfile(reference/'fixture.json',root/'fixture.json')
    cap.save(root/(ARM+'.json'),profile)
    manifest = copy.deepcopy(original)
    manifest.update(experiment_kind=KIND,created_at=time.time(),arms=[ARM],reference=str(reference),
        reference_manifest_sha256=cap.sha(reference/'manifest.json'),metadata_qualification=identity,
        runtime_sha256={ARM:identity['runtime_sha256']},source_sha256=cap.hashes(root/'frozen'),
        entry_sha256=cap.sha(root/'frozen'/ENTRY),qualifier_sha256=cap.sha(root/'frozen'/QUALIFIER),
        profiles_sha256={ARM:cap.sha(root/(ARM+'.json'))},port=args.port,
        scope='Original natural8 fixture, own accepted IDs, tool messages/waits, sampling,2warmup/300s/16turn epochs, '
              '32000 natural ceiling and normal drain. Runtime changes only reviewed3 files, profile only '
              'async_action_metadata=true and port. No profiling, startup prewarm, early release or native verification changes. '
              'Requires new four-case CUDA and cold real-model Action qualification; not on-policy task quality.')
    cap.require(cap.hashes(root/'runtime'/ARM) == identity['runtime_sha256'],'Candidate changed during freezing')
    cap.save(root/'manifest.json',manifest)
    cap.save(root/'qualification-proof.json',dict(at=time.time(),identity=identity,evidence=proof))
    cap.save(root/'status.json',dict(status='prepared',at=time.time(),gpu_started=False))


def verify(root):
    manifest = ready.verify(root)
    cap.require(manifest.get('experiment_kind') == KIND and manifest['arms'] == [ARM]
        and manifest.get('qualify') is False,'Not the isolated metadata-upload performance arm')
    reference = Path(manifest['reference'])
    original = reference_proof(reference,manifest['reference_manifest_sha256'])
    sources = dict(original['source_sha256'])
    sources[ENTRY] = manifest['entry_sha256']
    sources[QUALIFIER] = manifest['qualifier_sha256']
    cap.require(manifest['source_sha256'] == sources,'Original workload source changed beyond performance entry')
    for key in ('workload','generation_sampling','budget','native_source','idle_seconds','fixture_sha256'):
        cap.require(manifest[key] == original[key],'Natural reference '+key+' changed')
    cap.require(cap.read(root/(ARM+'.json')) == performance_profile(cap.read(reference/(ARM+'.json')),manifest['port']),
                'Profile changed beyond upload flag/port')
    review = manifest['metadata_qualification'].get('offline_review')
    identity,proof = completed_qualification(Path(manifest['metadata_qualification']['path']),
        original['runtime_sha256'][ARM],original['native_source'],Path(review['path']) if review else None)
    cap.require(identity == manifest['metadata_qualification']
        and manifest['runtime_sha256'] == {ARM:identity['runtime_sha256']},'Qualified candidate identity changed')
    return manifest,proof


def upload_observations(root):
    capture = root/ARM/'server/capture'
    config = cap.read(capture/'action_config.json')
    cap.require(config.get('async_action_metadata') is True,'Performance server did not enable metadata upload')
    records = ready.records(capture/'action_attention.jsonl')
    snapshots = [r.get('metadata_upload_total') for r in records]
    cap.require(snapshots and all(isinstance(s,dict) for s in snapshots),'Missing actual metadata upload snapshots')
    cap.require(all(s.get('quarantined') is False and all(s.get(k) == 0 for k in
        ('failed_submissions','fence_failures','query_failures')) for s in snapshots),'Metadata upload became poisoned/failed')
    cap.require(snapshots[-1].get('submissions',0) > 0,'No real async metadata uploads occurred')
    # Terminal snapshots have no timestamp and are cumulative across owners.
    # They cannot establish an exact formal-window counter delta or GPU idle.
    return dict(enabled=True,terminal_snapshots=len(snapshots),final=snapshots[-1],
        fallback_observed=any(bool(s.get('fallbacks')) for s in snapshots),
        formal_counter_delta=None,scope='All-run terminal cumulative counters include warmup and drain. '
        'No timestamped formal upload delta is available; host waits are not GPU-idle measurements.')


async def run(args):
    root = args.output.resolve()
    manifest,proof = verify(root)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared','No automatic retry')
    cap.save(root/'qualification-proof.json',dict(at=time.time(),identity=manifest['metadata_qualification'],evidence=proof))
    original_wait = ready.bench.q.wait_idle
    async def checked_wait(*values,**kwargs):
        await original_wait(*values,**kwargs)
        current,evidence = verify(root)
        cap.require(current == manifest,'Performance identity changed during idle wait')
        cap.save(root/'qualification-proof.json',dict(at=time.time(),identity=current['metadata_qualification'],evidence=evidence))
    ready.bench.q.wait_idle = checked_wait
    loop,task = asyncio.get_running_loop(),asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):
        loop.add_signal_handler(sig,task.cancel)
    started = time.time()
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            await ready.run_arm(root,ARM,manifest)
        cap.save(root/'upload-observations.json',upload_observations(root))
        analysis = module('action_upload_natural_analysis',ROOT/'scripts/analyze_rolling_agents.py')
        review = analysis.review(root)
        cap.save(root/'analysis.json',review)
        cap.require(review.get('arms',{}).get(ARM,{}).get('full_history_publication',{}).get('passed') is True,
                    'Complete CPU history/terminal M proof missing')
        no_cleanup(root)
        cap.save(root/'status.json',dict(status='completed_pending_review',at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
    finally:
        ready.bench.q.wait_idle = original_wait
        for sig in (signal.SIGINT,signal.SIGTERM):
            loop.remove_signal_handler(sig)
        cap.save(root/'timing.json',dict(started_wall=started,finished_wall=time.time(),scope='Launch,2warmup,300s and normal drain'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--run-prepared',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--qualification-review',type=Path,
        help='Explicit offline review of the exact completed-model parser-import failure only')
    parser.add_argument('--port',type=int,default=18570)
    args = parser.parse_args()
    if args.run_prepared:
        return asyncio.run(run(args))
    cap.require(args.reference is not None and args.qualification is not None,
                'Explicit original natural reference and completed metadata qualification required')
    prepare(args)


if __name__ == '__main__':
    main()

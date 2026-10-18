"""Frozen natural-eight-agent performance arm for qualified early slot release.

Reuse the reference's workload, sampling, launcher and analysis unchanged.
Both the exact-runtime GPU byte qualification and actual shared-slot reuse
qualification must pass before GPU polling/launch and again after idle waiting.
"""
import argparse
import asyncio
from collections import Counter
import copy
import fcntl
import importlib.util
from pathlib import Path
import re
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/benchmark_capture_feedback.py'
ARM = 'rolling-candidate-v2'
EXPERIMENT = 'capture_feedback_performance_v1'
FILES = {'src/vllm_kvmem/capture_runtime.py', 'src/vllm_kvmem/capture_feedback.py',
         'src/vllm_kvmem/options.py', 'tests/test_capture_feedback.py'}
FEEDBACK_BYTES = 268435456
PUBLICATION_COUNTERS = (
    'copy_only_slot_wait_calls', 'copy_only_slot_wait_seconds', 'detach_budget_deferrals',
    'source_releases_before_semantic', 'source_reuse_before_semantic_packets',
    'shared_source_reuse_before_semantic_packets', 'source_copy_seconds',
)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


ready = module('capture_feedback_ready', ROOT/'scripts/benchmark_ready_agents.py')
cap = ready.cap


def performance_profile(reference, port):
    cap.require(type(port) is int and 1024 <= port <= 65535, 'Invalid dedicated performance port')
    cap.require(reference.get('max_num_seqs') == 8 and reference.get('max_model_len') == 262144 and
        reference.get('verify_session_restore') is False and reference['kvmem'].get('rolling_active_window') is True and
        reference['kvmem'].get('async_cpu_scatter') is True and reference['kvmem'].get('split_capture_dma') is True and
        not reference['kvmem'].get('action_profiling', False), 'Require original natural rolling deployment without profiling')
    p = copy.deepcopy(reference)
    p['port'] = port
    p['kvmem'].update(early_capture_slot_release=True, early_capture_feedback_bytes=FEEDBACK_BYTES)
    return p


def reference_proof(reference, expected_manifest=None):
    if expected_manifest is not None:
        cap.require(cap.sha(reference/'manifest.json') == expected_manifest, 'Reference manifest changed')
    natural = module('capture_feedback_reference', reference/'frozen/scripts/benchmark_rolling_agents.py')
    m = natural.verify(reference)
    natural.qualification_proofs(m)
    return natural, m


def runtime_contract(base, candidate, manifest):
    cap.require(set(manifest['files']) == FILES and manifest.get('default_enabled') is False,
                'Require only the reviewed four-file early-release overlay')
    expected = cap.hashes(base)
    for name, row in manifest['files'].items():
        cap.require(expected.get(name) == row['base_sha256'], 'Early-release base runtime differs')
        cap.require(cap.sha(candidate/name) == row['candidate_sha256'], 'Early-release candidate file changed')
        expected[name] = row['candidate_sha256']
    cap.require(cap.hashes(candidate) == expected, 'Unrelated runtime changes are not qualified')
    return expected


def reuse_identity(path, runtime_hashes, native_source, byte_identity):
    saved = cap.read(path/'manifest.json')
    cap.require(saved.get('kind') == 'early_capture_reuse_qualification_v1' and
        cap.hashes(path/'frozen') == saved['source_sha256'], 'Reuse qualification frozen source changed')
    probe = module('capture_feedback_reuse', path/'frozen/scripts/qualify_capture_reuse.py')
    m = probe.verify(path)
    cap.require(m['runtime_sha256'] == runtime_hashes and m['native_source'] == native_source and
        m['byte_qualification'] == byte_identity, 'Reuse qualification used different runtime or byte qualification')
    return probe, dict(path=str(path), manifest_sha256=cap.sha(path/'manifest.json'), runtime_sha256=runtime_hashes)


def completed_reuse_proof(probe, root):
    cap.require(cap.read(root/'status.json')['status'] == 'completed_pending_review',
                'Actual reuse qualification has not passed; performance launch forbidden')
    result = cap.read(root/'results.json')
    cap.require(result.get('status') == 'passed' and result.get('complete_rounds') is True and
        result.get('cleanup_completed') is True and not result.get('context_capacity_shortfalls') and
        result.get('actual_workload_requests') == 16, 'Reuse coverage insufficient or incomplete; performance forbidden')
    directory, capture = root/'workload', root/'server/capture'
    rows, fixture = cap.read(directory/'traces.json'), cap.read(root/'fixture.json')
    cap.require(len(rows) == 24 and sum(r['kind'] == 'workload' for r in rows) == 16,
                'Reuse qualification needs all sixteen natural requests and eight cleanups')
    output = probe.output_evidence(rows, ready.records(directory/'events.jsonl'), fixture)
    history = probe.qual.history_evidence(rows, cap.read(directory/'history_publications.json'))
    cleanup = [r for r in rows if r['kind'] == 'cleanup']
    cap.require(all(r['input_sha256'] == probe.work.digest([fixture['eos']]) for r in cleanup),
                'Reuse cleanup did not reset each session with the original EOS')
    reuse = probe.reuse_evidence(ready.records(capture/'capture_feedback_events.jsonl'),
        ready.records(capture/'publication_samples.jsonl'), rows, not_before=max(r['begin'] for r in cleanup))
    cap.require(reuse['covered'] and reuse['shared_slot_reuse_proofs'], 'No actual shared-slot generation reuse')
    tokens = sum(len(r['output_ids']) for r in rows)
    reconciliation = result.get('native_reconciliation', {})
    cap.require(reconciliation.get('passed') is True and reconciliation.get('preemptions') == 0 and
        reconciliation.get('native_tokens') == reconciliation.get('client_tokens') == tokens ==
        result.get('total_stream_output_tokens'), 'Reuse accepted outputs do not reconcile with native counters')
    archive = cap.read(directory/'archive-after.json')
    cap.require(archive.get('archive_backend') == 'cpu' and archive.get('repository_gc') is False and
        archive.get('disk_read_bytes') == archive.get('disk_write_bytes') == 0, 'Reuse history was not complete CPU-only')
    resources = cap.read(directory/'resources-after.json')
    cap.require(probe.work.views_match(resources, {r['session']:1 for r in cleanup}) and
        all(not resources[key] for key in ('requests', 'refresh_pending', 'connector_pending',
            'owned_blocks', 'reservations')) and all(not resources['cpu_history_admission'][key]
            for key in ('active', 'retiring', 'waiting')), 'Reuse session cleanup did not fully drain')
    return dict(result_sha256=cap.sha(root/'results.json'), accepted_tokens=tokens,
                output_proofs=output, history_proofs=history, reuse=reuse, cleanup_completed=True)


def qualification_proofs(m, natural, *, require_completed):
    byte = m['early_qualifications']['byte']
    path = Path(byte['path'])
    actual = natural.qualification_identity(path, ARM, m['runtime_sha256'][ARM], m['native_source'])
    cap.require(actual == byte, 'Byte qualification identity changed')
    kv = cap.read(path/'profile.json')['kvmem']
    cap.require(kv.get('early_capture_slot_release') is True and
        kv.get('early_capture_feedback_bytes') == FEEDBACK_BYTES, 'Byte qualification omitted exact early-release settings')
    reuse = m['early_qualifications']['reuse']
    probe, actual_reuse = reuse_identity(Path(reuse['path']), m['runtime_sha256'][ARM], m['native_source'], byte)
    cap.require(actual_reuse == reuse, 'Reuse qualification identity changed')
    if not require_completed:
        return dict(byte=byte, reuse=reuse, completion_required_before_launch=True)
    return dict(byte=dict(identity=byte, evidence=natural.validate_completed_qualification(path, ARM)),
        reuse=dict(identity=reuse, evidence=completed_reuse_proof(probe, Path(reuse['path']))))


def window_release_evidence(events, samples, window, rows=()):
    """Audit actual physical reuse in the original window; never infer GPU idle."""
    lo, hi = window['begin'], window['end']
    cap.require(hi > lo, 'Invalid formal performance window')
    phase = lambda at: 'warmup' if at < lo else ('formal' if at < hi else 'drain')
    responses = {r['response_id']:r['kind'] for r in rows if r.get('response_id')}
    cap.require(len(responses) == sum(bool(r.get('response_id')) for r in rows), 'Duplicate trace response IDs')
    def packet_kind(event):
        kinds = set()
        for request_id in event.get('request_ids', []):
            match = re.fullmatch(r'(.+)-0-[0-9a-f]{8}',request_id)
            kinds.add(responses.get(match[1], 'unknown') if match else 'unknown')
        if not kinds or 'unknown' in kinds:
            return 'unknown'
        return next(iter(kinds)) if len(kinds) == 1 else 'mixed'
    stages = {}
    for event in events:
        cap.require(event.get('lane') == 'bulk' and type(event.get('slot')) is int and event['slot'] >= 0 and
            type(event.get('generation')) is int and event['generation'] > 0, 'Invalid physical source identity')
        stages.setdefault((event['lane'], event['slot'], event['generation']), []).append(event)
    proofs = []
    for key, current in stages.items():
        submitted = [e for e in current if e['stage'] == 'source_submitted']
        cap.require(len(submitted) == 1, 'Missing or duplicate source generation submission')
        new = submitted[0]
        pending = new.get('pending_previous_generations', [])
        cap.require(len(pending) == len(set(pending)), 'Duplicate pending previous generation')
        for previous in pending:
            old = stages.get((key[0], key[1], previous), [])
            starts = [e for e in old if e['stage'] == 'source_submitted']
            releases = [e for e in old if e['stage'] == 'source_released']
            completes = [e for e in old if e['stage'] == 'semantic_complete']
            cap.require(len(starts) == len(releases) == len(completes) == 1,
                'A reused generation lacks complete source/semantic lifecycle')
            start, release, complete = starts[0], releases[0], completes[0]
            copied = [e for e in old if e['stage'] == 'source_copied' and e['at'] <= release['at']]
            indices = [i for e in copied for i in e['copied_indices']]
            cap.require(start.get('request_ids') and sorted(indices) == list(range(len(start['request_ids']))) and
                previous < key[2] and start['at'] <= release['at'] <= new['at'] < complete['at'] and
                release.get('semantic_pending', 0) > 0,
                'Slot was reused before complete copy or after semantic completion')
            old_kind, new_kind = packet_kind(start), packet_kind(new)
            kind = old_kind if old_kind == new_kind else (
                'unknown' if 'unknown' in (old_kind,new_kind) else 'mixed')
            proofs.append(dict(lane=key[0], slot=key[1], previous_generation=previous,
                next_generation=key[2], previous_owner_count=len(start['request_ids']),
                next_owner_count=len(new['request_ids']), source_released_at=release['at'],
                reused_at=new['at'], semantic_complete_at=complete['at'], phase=phase(new['at']),
                previous_packet_kind=old_kind,next_packet_kind=new_kind,trace_kind=kind,
                shared_previous_packet=len(start['request_ids']) > 1, feedback_bytes=release['feedback_bytes']))
    def counts(selected):
        identity = lambda p: (p['lane'], p['slot'], p['next_generation'])
        return dict(actual_reuse_submissions=len({identity(p) for p in selected}),
            shared_packet_reuse_submissions=len({identity(p) for p in selected if p['shared_previous_packet']}),
            previous_generation_pairs=len(selected))
    phases = {name:counts([p for p in proofs if p['phase'] == name]) for name in ('warmup','formal','drain')}
    kinds = ('workload','warmup','cleanup','mixed','unknown')
    for name in phases:
        phases[name]['by_trace_kind'] = {kind:counts([p for p in proofs if p['phase'] == name and
            p['trace_kind'] == kind]) for kind in kinds}
    early = Counter(phase(e['at']) for e in events if e['stage'] == 'source_released' and
                    e.get('semantic_pending', 0) > 0)
    for name in phases:
        phases[name]['early_source_releases'] = early[name]

    ordered = sorted(samples, key=lambda sample:sample['at'])
    for sample in ordered:
        state = sample['feedback']
        cap.require(0 <= state['detached_packets'] <= state['packet_limit'] and
            0 <= state['detached_owners'] <= state['owner_limit'] <= 8 and
            0 <= state['detached_bytes'] <= state['byte_limit'] == FEEDBACK_BYTES,
            'Detached feedback credit bound exceeded')
    def snapshot(sample, boundary):
        if sample is None:
            return None
        metrics = sample['publication_metrics']
        by_lane = {key:value for key,value in metrics.items() if key.endswith('/bulk')}
        feedback = copy.deepcopy(sample['feedback'])
        for singular, used, limit in (('packets','detached_packets','packet_limit'),
                ('owners','detached_owners','owner_limit'),('bytes','detached_bytes','byte_limit')):
            feedback['available_'+singular] = feedback[limit]-feedback[used]
        return dict(at=sample['at'], offset_seconds=sample['at']-boundary, feedback=feedback,
            counters={name:sum(value.get(name,0) for value in by_lane.values()) for name in PUBLICATION_COUNTERS},
            counters_by_thread_lane=by_lane)
    def boundary(at):
        before = [s for s in ordered if s['at'] <= at]
        after = [s for s in ordered if s['at'] >= at]
        return dict(before=snapshot(before[-1] if before else None,at),
                    after=snapshot(after[0] if after else None,at))
    boundaries = dict(T0=boundary(lo),T1=boundary(hi))
    def delta(start, end):
        if start is None or end is None or end['at'] <= start['at']:
            return None
        values = {name:end['counters'][name]-start['counters'][name] for name in PUBLICATION_COUNTERS}
        cap.require(all(value >= -1e-9 for value in values.values()), 'Publication counters reset within formal run')
        return dict(begin=start['at'],end=end['at'],begin_offset_from_T0_seconds=start['at']-lo,
                    end_offset_from_T1_seconds=end['at']-hi,counters=values)
    inner = delta(boundaries['T0']['after'],boundaries['T1']['before'])
    outer = delta(boundaries['T0']['before'],boundaries['T1']['after'])
    physical_covered = phases['formal']['shared_packet_reuse_submissions'] > 0
    covered = phases['formal']['by_trace_kind']['workload']['shared_packet_reuse_submissions'] > 0
    observed_phases = [name for name,value in phases.items() if value['shared_packet_reuse_submissions']]
    coverage = 'formal_shared_workload_reuse_observed' if covered else (
        'formal_shared_reuse_only_nonworkload_or_unknown' if physical_covered else (
            'outside_formal_window_only' if observed_phases else 'no_shared_reuse_observed'))
    return dict(window=dict(begin=lo,end=hi,duration_seconds=hi-lo),
        formal_shared_physical_reuse_covered=physical_covered,formal_shared_workload_reuse_covered=covered,
        coverage=coverage,shared_reuse_observed_phases=observed_phases,phases=phases,
        whole_run=counts(proofs),physical_reuse_proofs=proofs,publication_sample_count=len(ordered),
        boundary_snapshots=boundaries,counter_deltas=dict(inner_sample_interval=inner,outer_sample_interval=outer),
        scope='Actual reuse is classified by new source_submitted host timestamp in [T0,T1); '
              'a release without later physical slot reuse is not coverage. Counts identify distinct new '
              'lane/slot/generations; pairs also count multiple old pending generations. '
              'Trace kinds additionally separate workload, warmup, cleanup, mixed and unknown packets. '
              'Formal agent coverage requires both old and new packets to contain only workload requests; '
              'an epoch cleanup during the formal window cannot establish that coverage. Kind breakdowns '
              'can overlap for a new generation reusing multiple old generations. '
              'Warmup and drain are reported separately; raw whole-run events remain unchanged. '
              'Boundary cumulative-counter deltas cover the stated sample intervals, not exact T0/T1. '
              'Counters accrue when host operations finish; waits can cross boundaries and overlap. '
              'Copy-only wait seconds and other host spans are not GPU idle or measured time saved. '
              'Deferrals count attempts, not unique packets. Missing boundary observations yield null deltas. '
              'Uncovered formal bottlenecks do not justify workload changes or automatic retries.')


def save_window_release_evidence(root):
    directory = root/ARM
    capture = directory/'server/capture'
    sources = dict(window=directory/'tools/window.json',
        events=capture/'capture_feedback_events.jsonl',samples=capture/'publication_samples.jsonl',
        traces=directory/'tools/traces.json')
    report = window_release_evidence(ready.records(sources['events']),ready.records(sources['samples']),
                                    cap.read(sources['window']),cap.read(sources['traces']))
    report['source_sha256'] = {key:cap.sha(path) if path.exists() else None for key,path in sources.items()}
    cap.save(root/'formal-capture-feedback.json',report)


def prepare(args):
    root, reference, candidate_root = args.output.resolve(), args.reference.resolve(), args.candidate.resolve()
    cap.require(not root.exists(), 'Use a fresh performance directory')
    natural, ref = reference_proof(reference)
    candidate = cap.read(candidate_root/'manifest.json')
    cap.require(cap.sha(candidate_root/'changes.patch') == candidate['patch_sha256'], 'Early-release patch changed')
    expected = runtime_contract(reference/'runtime'/ARM, candidate_root/'runtime', candidate)
    p = performance_profile(cap.read(reference/(ARM+'.json')), args.port)
    byte = natural.qualification_identity(args.byte_qualification.resolve(), ARM, expected, ref['native_source'])
    _, reuse = reuse_identity(args.reuse_qualification.resolve(), expected, ref['native_source'], byte)
    m = copy.deepcopy(ref)
    m.update(experiment_kind=EXPERIMENT, created_at=time.time(), arms=[ARM], reference=str(reference),
        reference_manifest_sha256=cap.sha(reference/'manifest.json'), runtime_sha256={ARM:expected},
        early_qualifications=dict(byte=byte, reuse=reuse), port=args.port,
        scope='Formal natural eight-agent replay: original two warmup turns,300-second window,16-turn epochs, '
              '32000-token natural ceiling, complete original tool messages/waits and normal drain. '
              'Only early_capture_slot_release=true,256MiB feedback credit and dedicated port change. '
              'No Action profiling or extra unqualified native optimization. Requires exact byte and actual '
              'shared-slot reuse qualifications. Compare generation lengths and completed turns alongside token rates; '
              'not on-policy agent quality or a pure decode-kernel benchmark.')
    qualification_proofs(m, natural, require_completed=False)
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache')
    shutil.copytree(reference/'frozen', root/'frozen', ignore=ignore)
    shutil.copyfile(ROOT/ENTRY, root/'frozen'/ENTRY)
    shutil.copytree(candidate_root/'runtime', root/'runtime'/ARM, ignore=ignore)
    shutil.copyfile(reference/'fixture.json', root/'fixture.json')
    shutil.copyfile(candidate_root/'manifest.json', root/'candidate-manifest.json')
    shutil.copyfile(candidate_root/'changes.patch', root/'changes.patch')
    cap.save(root/(ARM+'.json'), p)
    m.update(source_sha256=cap.hashes(root/'frozen'), entry_sha256=cap.sha(root/'frozen'/ENTRY),
        profiles_sha256={ARM:cap.sha(root/(ARM+'.json'))},
        candidate_manifest_sha256=cap.sha(root/'candidate-manifest.json'),
        candidate_source=str(candidate_root), candidate_patch_sha256=candidate['patch_sha256'])
    cap.require(cap.hashes(root/'runtime'/ARM) == expected, 'Exact runtime changed during freeze')
    cap.save(root/'manifest.json', m)
    cap.save(root/'status.json', dict(status='prepared', at=time.time(), gpu_started=False,
        launch_gate='Completed exact-runtime byte and actual shared-slot reuse qualifications required'))


def verify(root, *, require_completed=True):
    m = ready.verify(root)
    cap.require(m.get('experiment_kind') == EXPERIMENT and m.get('arms') == [ARM] and
        m.get('qualify') is False, 'Not the isolated early-release performance arm')
    reference = Path(m['reference'])
    natural, ref = reference_proof(reference, m['reference_manifest_sha256'])
    expected_source = dict(ref['source_sha256'])
    expected_source[ENTRY] = m['entry_sha256']
    cap.require(m['source_sha256'] == expected_source, 'Original frozen helpers changed beyond the performance entry')
    for key in ('workload', 'generation_sampling', 'budget', 'native_source', 'idle_seconds', 'fixture_sha256'):
        cap.require(m[key] == ref[key], 'Reference '+key+' changed')
    cap.require(cap.read(root/(ARM+'.json')) == performance_profile(cap.read(reference/(ARM+'.json')), m['port']),
                'Profile changed beyond early flag,256MiB and port')
    cap.require(cap.sha(root/'candidate-manifest.json') == m['candidate_manifest_sha256'], 'Candidate manifest changed')
    candidate = cap.read(root/'candidate-manifest.json')
    cap.require(cap.sha(root/'changes.patch') == m['candidate_patch_sha256'] == candidate['patch_sha256'],
                'Reviewed early-release patch changed')
    original = Path(m['candidate_source'])
    cap.require(cap.sha(original/'manifest.json') == m['candidate_manifest_sha256'] and
        cap.sha(original/'changes.patch') == m['candidate_patch_sha256'], 'Original candidate identity changed')
    expected = runtime_contract(reference/'runtime'/ARM, root/'runtime'/ARM, candidate)
    cap.require(cap.hashes(original/'runtime') == expected and m['runtime_sha256'] == {ARM:expected},
                'Candidate differs from exact reviewed runtime')
    proofs = qualification_proofs(m, natural, require_completed=require_completed)
    return m, proofs


async def run(args):
    root = args.output.resolve()
    m, proofs = verify(root)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared', 'No automatic restart')
    cap.save(root/'qualification-proof.json', dict(at=time.time(), qualifications=proofs))
    original_wait = ready.bench.q.wait_idle
    async def checked_wait(*values, **kw):
        await original_wait(*values, **kw)
        current, latest = verify(root)
        cap.require(current == m, 'Performance identity changed while waiting')
        cap.save(root/'qualification-proof.json', dict(at=time.time(), qualifications=latest))
    ready.bench.q.wait_idle = checked_wait
    loop, active = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, active.cancel)
    started, monotonic = time.time(), time.monotonic()
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            await ready.run_arm(root, ARM, m)
        save_window_release_evidence(root)
        analysis = module('capture_feedback_analysis', ROOT/'scripts/analyze_rolling_agents.py')
        cap.save(root/'analysis.json', analysis.review(root))
        cap.save(root/'status.json', dict(status='completed_pending_review', at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise
    finally:
        ready.bench.q.wait_idle = original_wait
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        cap.save(root/'timing.json', dict(started_wall=started, finished_wall=time.time(),
            elapsed_seconds=time.monotonic()-monotonic, scope='Launch, warmup, formal300s window and normal drain'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    for name in ('reference', 'candidate', 'byte-qualification', 'reuse-qualification'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--port', type=int, default=18550)
    args = parser.parse_args()
    if args.run_prepared:
        return asyncio.run(run(args))
    cap.require(all((args.reference, args.candidate, args.byte_qualification, args.reuse_qualification)),
                'Explicit reference, candidate and both qualification directories required')
    prepare(args)


if __name__ == '__main__':
    main()

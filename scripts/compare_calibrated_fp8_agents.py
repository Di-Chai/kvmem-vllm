"""Read-only comparison of the two freshly calibrated FP8 replay arms.

--prepared-only audits hashes and protocol without producing performance results.
The default mode refuses unfinished, diagnostic or unqualified runs. It never
imports vLLM, torch or a benchmark entry point and never probes the live GPU.
Existing frozen protocol functions are AST-extracted with restricted globals to
recheck history, byte-oracle and public-work evidence with the original rules.
"""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

HOST_SOURCE = Path(__file__).with_name('host_admission_calibration.py')
host_spec = importlib.util.spec_from_file_location('offline_host_calibration', HOST_SOURCE)
host = importlib.util.module_from_spec(host_spec)
host_spec.loader.exec_module(host)

ARM = 'rolling-candidate-v2'
PRODUCTION_DELTA = {
    'src/vllm_kvmem/authority.py', 'csrc/bindings.cpp',
    'src/vllm_kvmem/_native.cpython-312-x86_64-linux-gnu.so',
}
ALLOCATOR_DELTA = {'src/vllm_kvmem/serve.py'}
ALLOCATOR_POLICY = dict(garbage_collection_threshold=.9, per_process_memory_fraction=.99)
MEASUREMENT_FUNCTIONS = ('checked_workload', 'separated_report', 'finish_cleanup',
                         'guarded_arm', 'run_qualification', 'run')
ALLOCATOR_PREPARATION_FUNCTIONS = {'context_profile', 'selected_profile', 'host_contract',
                                  'prepare', 'verify', 'main'}
CONTROLLER_DELTA = {'scripts/benchmark_aligned_fp8_agents.py'}
RUNTIME_SCRIPT_DELTA = CONTROLLER_DELTA | {'scripts/benchmark_sealed_mean_append.py'}
IMMUTABLE = (
    'workload', 'generation_sampling', 'budget', 'fixture_sha256', 'fixture_directory',
    'fixture_identity', 'weight_model_identity', 'host_admission', 'arms',
    'idle_seconds', 'native_source', 'installed_native_sha256', 'device_memory_identity',
    'minimum_device_free_mib', 'release_qualification', 'protocol_delta',
    'declared_profile_delta', 'new_runtime_reference', 'new_runtime_reference_sha256', 'host_calibration',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def delta(before, after):
    return {k: [before.get(k), after.get(k)] for k in sorted(before.keys() | after.keys())
            if before.get(k) != after.get(k)}


def calibrated_context_profile(manifest, profile):
    result = dict(profile)
    value = result.pop('cuda_allocator_conf', None)
    policy = manifest.get('cuda_allocator_policy')
    require((policy is None and value is None) or
            (policy == 'native_gc_90' and value == ALLOCATOR_POLICY),
            'Calibration has an undeclared or changed allocator policy')
    return result


def native_matches(row, request_id):
    response = row.get('response_id')
    return isinstance(response, str) and re.fullmatch(
        re.escape(response) + r'-0-[0-9a-f]{8}', request_id) is not None


def digest(tokens):
    # Match the frozen timed_agent_workload.digest, including JSON separators.
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def distribution(values):
    values = sorted(values)
    def q(fraction):
        if not values:
            return None
        position = (len(values)-1)*fraction
        lo = int(position)
        return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(position-lo)
    return dict(count=len(values), sum=sum(values), p50=q(.5), p95=q(.95),
                maximum=max(values, default=None))


class Audit:
    def __init__(self):
        self.evidence = {}

    def hash(self, path):
        path = path.resolve()
        value = sha(path)
        old = self.evidence.setdefault(str(path), value)
        require(old == value, 'Evidence changed during audit: '+str(path))
        return value

    def read(self, path):
        self.hash(path)
        return json.loads(path.read_text())

    def rows(self, path):
        return list(self.iter_rows(path))

    def iter_rows(self, path):
        self.hash(path)
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    def tree(self, root, expected):
        paths = {str(p.relative_to(root)): p for p in root.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        require(paths.keys() == expected.keys(), 'Frozen tree file set changed: '+str(root))
        require(all(self.hash(p) == expected[k] for k, p in paths.items()),
                'Frozen tree content changed: '+str(root))

    def function(self, path, names, globals_):
        """Load only named pure functions; do not execute imports/module setup."""
        self.hash(path)
        module = ast.parse(path.read_text(), filename=str(path))
        selected = [node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name in names]
        require({node.name for node in selected} == set(names), 'Frozen pure checker missing')
        namespace = dict(globals_)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
        return namespace

    def frozen_run(self, root, *, completed):
        m = self.read(root/'manifest.json')
        require(m.get('experiment_kind') == 'aligned_fp8_agents_v1' and
                m.get('performance_eligible') is True and m.get('diagnostic_steps') == 0 and
                m.get('diagnostic_reference') is None and m.get('arms') == [ARM],
                'Only unprofiled aligned FP8 performance runs qualify')
        self.tree(root/'runtime'/ARM, m['runtime_sha256'][ARM])
        self.tree(root/'frozen', m['source_sha256'])
        require(self.hash(root/'fixture.json') == m['fixture_sha256'], 'Fixture changed')
        for name, expected in m['profiles_sha256'].items():
            require(self.hash(root/(name+'.json')) == expected, 'Profile changed')
        profile = self.read(root/(ARM+'.json'))
        require(profile.get('profiler_config') is None, 'Profiler is not performance evidence')
        self.alignment(m['alignment_qualification'], m['runtime_sha256'][ARM], completed)
        self.qualification(root, m, profile, completed)
        if m.get('host_calibration'):
            self.calibrated_host(m, profile)
        if completed:
            require(self.read(root/'status.json').get('status') == 'completed_pending_review',
                    'Performance run is not complete: '+str(root))
        return m, profile

    def calibrated_host(self, manifest, profile):
        binding = manifest['host_calibration']
        path = Path(binding['path'])
        require(self.hash(path) == binding['sha256'], 'Host calibration binding changed')
        require(self.hash(HOST_SOURCE) == manifest['source_sha256'].get('scripts/host_admission_calibration.py'),
                'Offline calibration checker differs from the frozen controller')
        report = self.read(path)
        source = Path(report['source_run'])
        sm = self.read(source/'manifest.json')
        require(sm.get('host_calibration') is None and sm['budget']['extra_bytes'] == 32*1024**3,
                'Calibration must derive from the accepted original32GiB source')
        # This validates complete own outputs, terminal CPU history, byte
        # qualification, deployment and explicit release as well as source hashes.
        self.measured(source)
        names = ('manifest.json', 'status.json', ARM+'/results.json', ARM+'/deployment-acceptance.json',
                 ARM+'/own-history-proof.json', ARM+'/tools/explicit-release-proof.json',
                 ARM+'/tools/window.json', ARM+'/tools/traces.json', ARM+'/tools/events.jsonl',
                 ARM+'/history_publications.jsonl', ARM+'/post-stop.json')
        proof = dict(path=str(source), evidence_sha256={name:self.hash(source/name) for name in names})
        require(host.source_report(source, sm, proof, self.hash) == report and
                host.calibrated_budget(sm['budget'], report) == manifest['budget'],
                'Calibrated allowance cannot be reproduced from the accepted memory peak')
        require(sm['workload'] == manifest['workload'] and
                sm['fixture_sha256'] == manifest['fixture_sha256'] and
                sm['device_memory_identity'] == manifest['device_memory_identity'] and
                calibrated_context_profile(sm,self.read(source/(ARM+'.json'))) ==
                dict(calibrated_context_profile(manifest,profile),port=sm['port']),
                'Calibration was used for another model/device/replay')

    def alignment(self, binding, runtime, completed):
        root = Path(binding['path'])
        require(self.hash(root/'manifest.json') == binding['manifest_sha256'], 'Alignment binding changed')
        m = self.read(root/'manifest.json')
        self.tree(root/'runtime', m['runtime_sha256'])
        require(m.get('qualification_script') == 'qualify_context_alignment.py' and
                {k:v for k,v in m['runtime_sha256'].items() if k.startswith('src/')} ==
                {k:v for k,v in runtime.items() if k.startswith('src/')}, 'Alignment server source differs')
        require(self.hash(root/'profile.json') == m['profile_sha256'] and
                self.hash(Path(m['fixture'])) == m['fixture_sha256'], 'Alignment fixture/profile changed')
        if completed:
            result, cleanup = self.read(root/'qualification/results.json'), self.read(root/'cleanup.json')
            require(self.read(root/'status.json').get('status') == result.get('status') == 'passed' and
                    result.get('all_owned_sessions_released') is True and
                    cleanup.get('remaining_owned_process_groups') == {} and
                    cleanup.get('remaining_gpu_pids') == [], 'Alignment qualification or cleanup failed')

    def qualification(self, performance, pm, profile, completed):
        binding = pm['fp8_runtime_qualification']
        root = Path(binding['path'])
        require(self.hash(root/'manifest.json') == binding['manifest_sha256'], 'FP8 qualification binding changed')
        m = self.read(root/'manifest.json')
        require(m['runtime_sha256'] == pm['runtime_sha256'][ARM], 'Qualification runtime differs')
        self.tree(root/'runtime', m['runtime_sha256'])
        self.tree(root/'frozen', m['source_sha256'])
        require(self.hash(root/'profile.json') == m['profile_sha256'] and
                self.hash(root/'fixture.json') == m['fixture_sha256'] and
                m['fixture_source_sha256'] == pm['fixture_sha256'], 'FP8 qualification fixture/profile changed')
        cap = SimpleNamespace(require=require)
        checkers = self.function(root/'frozen/scripts/qualify_rolling_window.py',
            ('qualification_profile', 'oracle_evidence', 'history_evidence'), dict(
                cap=cap, copy=__import__('copy'), USERS=2, COUNTS=(1024,4096,1024),
                shared=SimpleNamespace(records=self.rows),
                core=SimpleNamespace(native_matches=native_matches)))
        qp = self.read(root/'profile.json')
        require(qp == checkers['qualification_profile'](profile, qp['port']) and
                m.get('output_quotas') == [1024,4096,1024] and m.get('concurrency') == 2,
                'Exact FP8 qualification protocol changed')
        if not completed:
            return
        result = self.read(root/'results.json')
        counts = result.get('native_reconciliation', {})
        require(self.read(root/'status.json').get('status') == 'completed_pending_review' and
                result.get('status') == 'passed' and result.get('cleanup_completed') is True and
                counts.get('passed') is True and counts.get('preemptions') == 0 and
                counts.get('native_tokens', 0) > 0 and counts['native_tokens'] == counts.get('client_tokens'),
                'FP8 qualification not passed or token counts differ')
        rows = self.read(root/'workload/traces.json')
        require(sum(len(r['output_ids']) for r in rows) == result['total_stream_output_tokens'] ==
                counts['client_tokens'] and all(r['status'] == 'completed' and
                len(r['output_ids']) == ((1024,4096,1024)[r['turn']] if r['kind'] == 'workload' else 1)
                for r in rows), 'FP8 fixed-quota accepted outputs differ')
        require(checkers['oracle_evidence'](root/'server/capture', rows) == result.get('oracle'),
                'FP8 byte oracle evidence changed')
        work = [r for r in rows if r['kind'] == 'workload']
        proofs = []
        for turn in range(3):
            archive = self.read(root/f'workload/archive-turn-{turn}.json')
            proofs.extend(checkers['history_evidence'](
                [r for r in work if r['turn'] == turn], archive.get('completed', [])))
        require(len(work) == 6 and proofs == result.get('history_proofs'),
                'FP8 complete CPU history proofs changed')
        self.qualification_cleanup(root, pm)

    def qualification_cleanup(self, root, performance_manifest):
        # An exactly reused qualification keeps its original protected owner.
        # Do not rewrite an old cleanup receipt to name the new performance run.
        owner = self.read(root.parent/'manifest.json')
        expected = dict(path=str(root),manifest_sha256=self.hash(root/'manifest.json'))
        require(owner.get('fp8_runtime_qualification') == expected and
                owner['runtime_sha256'][ARM] == performance_manifest['runtime_sha256'][ARM],
                'Reused FP8 qualification has another source/owner')
        self.tree(root.parent/'frozen',owner['source_sha256'])
        self.tree(root.parent/'runtime'/ARM,owner['runtime_sha256'][ARM])
        stop = self.read(root/'process/post-stop.json')
        binding = self.read(root/'process/qualification-cleanup.json')
        require(stop.get('gpu_empty') is True and stop.get('observed_gpu_pids') == [] and binding == dict(
            protocol='aligned_fp8_qualification_cleanup_v1',
            qualification_manifest_sha256=self.hash(root/'manifest.json'),
            performance_manifest_sha256=self.hash(root.parent/'manifest.json'),
            post_stop_sha256=self.hash(root/'process/post-stop.json')), 'Unbound or failed FP8 process cleanup')

    def unchanged_measurement_controller(self, before, after):
        def bodies(path):
            self.hash(path)
            nodes = ast.parse(path.read_text()).body
            result = {n.name: ast.dump(n, include_attributes=False) for n in nodes
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                      n.name in MEASUREMENT_FUNCTIONS}
            require(set(result) == set(MEASUREMENT_FUNCTIONS), 'Measurement controller function missing')
            remaining = []
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ALLOCATOR_PREPARATION_FUNCTIONS:
                    continue
                if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                        isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'CUDA_CACHE_GC'):
                    expected = ast.parse('CUDA_CACHE_GC = dict(garbage_collection_threshold=.9, per_process_memory_fraction=.99)').body[0]
                    require(ast.dump(node) == ast.dump(expected), 'Allocator controller policy constant changed')
                    continue
                remaining.append(ast.dump(node, include_attributes=False))
            return result, remaining
        require(bodies(before) == bodies(after), 'Allocator ablation changed actual measurement/cleanup code')

    def protocol(self, baseline, candidate):
        bm,bp = self.frozen_run(baseline, completed=False)
        cm,cp = self.frozen_run(candidate, completed=False)
        require(bm.get('host_calibration') is not None and cm.get('host_calibration') is not None,
                'Both performance arms must use the calibrated protocol')
        require(all(bm[k] == cm[k] for k in IMMUTABLE), 'A/B immutable protocol differs')
        allocator = cm.get('cuda_allocator_policy')
        if allocator is None:
            require(bp == dict(cp, port=bp['port']), 'A/B profile differs beyond local port')
        else:
            require(allocator == 'native_gc_90' and bm.get('cuda_allocator_policy') is None and
                    bp.get('cuda_allocator_conf') is None and cp == dict(bp, port=cp['port'],
                    cuda_allocator_conf=ALLOCATOR_POLICY), 'Undeclared allocator/profile change')
        require(cm.get('baseline',{}).get('path') == str(baseline), 'Candidate is not bound to this baseline')
        for name, expected in cm['baseline']['evidence_sha256'].items():
            require(self.hash(baseline/name) == expected, 'Candidate baseline evidence binding changed')
        changed = delta(bm['runtime_sha256'][ARM], cm['runtime_sha256'][ARM])
        production = {k:v for k,v in changed.items() if k.startswith(('src/','csrc/'))}
        expected = PRODUCTION_DELTA | (ALLOCATOR_DELTA if allocator else set())
        require(set(production) == expected and production == cm['baseline']['production_delta'],
                'Production changes differ from the declared sealed-copy/allocator files')
        require(set(changed)-set(production) == RUNTIME_SCRIPT_DELTA,
                'Unexpected benchmark runtime source changes')
        controllers = delta(bm['source_sha256'],cm['source_sha256'])
        if allocator:
            require(set(controllers) <= CONTROLLER_DELTA, 'Unexpected allocator controller change')
            self.unchanged_measurement_controller(baseline/'frozen/scripts/benchmark_aligned_fp8_agents.py',
                candidate/'frozen/scripts/benchmark_aligned_fp8_agents.py')
        else:
            require(not controllers, 'Unexpected benchmark/controller source changes')
        bq = self.read(Path(bm['fp8_runtime_qualification']['path'])/'manifest.json')
        cq = self.read(Path(cm['fp8_runtime_qualification']['path'])/'manifest.json')
        require(all(bq[k] == cq[k] for k in ('fixture_sha256','budget','output_quotas','concurrency','native_source')),
                'A/B FP8 qualification protocol differs')
        return dict(production_delta=production, nonproduction_runtime_delta={k:v for k,v in changed.items()
            if k not in production}, controller_delta=delta(bm['source_sha256'],cm['source_sha256']),
            identical_protocol_fields=list(IMMUTABLE), profile_delta={'port':[bp['port'],cp['port']],
                **({'cuda_allocator_conf':[None,ALLOCATOR_POLICY]} if allocator else {})},
            fixture_sha256=bm['fixture_sha256'], workload=bm['workload'],
            alignment_scope='Alignment freezes matching src including native.so; csrc is bound by performance/FP8 runtimes.')

    def measured(self, root):
        m, _ = self.frozen_run(root, completed=True)
        d = root/ARM
        result, acceptance = self.read(d/'results.json'), self.read(d/'deployment-acceptance.json')
        history = self.read(d/'own-history-proof.json')
        stop = self.read(d/'post-stop.json')
        require(result.get('status') == 'completed' and result.get('drained') is True and
                result.get('native_client_tokens_equal') is True and result.get('preemptions') == 0 and
                result.get('explicit_session_release_verified') is True and acceptance.get('accepted') is True and
                acceptance.get('full_process_preemptions') == 0 and
                stop.get('gpu_empty') is True and stop.get('observed_gpu_pids') == [],
                'Completed-run accounting, deployment or cleanup failed')
        released = self.read(d/'tools/explicit-release-proof.json')
        require(all(released['resources'].get(k) == 0 for k in ('requests','refresh_pending','connector_pending','owned_blocks')) and
                all(released['archive'].get(k) == 0 for k in ('cpu_cache_used_bytes','cpu_authority_bytes',
                    'session_checkpoint_payload_bytes','session_checkpoints','active_requests','archive_used_slots',
                    'session_checkpoint_deferrals','retiring_requests','session_retiring_requests','pending_downloads')),
                'Session/CPU history was not fully released')
        if m.get('host_calibration'):
            envelope = host.observed_envelope(m['budget'],
                self.read(root/(ARM+'-host-admission.json')), self.rows(d/'gpu_samples.jsonl'))
            envelope['calibration'] = m['host_calibration']
            require(self.read(root/'host-memory-envelope.json') == envelope,
                    'Measured calibrated host guard evidence differs')
        rows, window = self.read(d/'tools/traces.json'), self.read(d/'tools/window.json')
        events = self.rows(d/'tools/events.jsonl')
        checker = self.function(root/'frozen/scripts/benchmark_capacity_chain.py', ('own_history_proof',),
            dict(cap=SimpleNamespace(require=require),work=SimpleNamespace(digest=digest)))
        require(checker['own_history_proof'](self.read(root/'fixture.json'), rows, events) == history,
                'Complete own-history protocol proof differs')
        publication_checker = self.function(root/'frozen/scripts/analyze_ready_agents.py',
            ('history_proof',), dict(core=SimpleNamespace(native_matches=native_matches)))
        require(publication_checker['history_proof'](rows,self.rows(d/'history_publications.jsonl'))['passed'],
                'Complete terminal M publication proof missing')
        require(all(result['archive_final'].get(k) == 0 for k in ('disk_read_bytes','disk_write_bytes',
                'session_checkpoint_deferrals','cpu_cache_used_bytes')), 'CPU-only history contract failed')
        last = None
        for snapshot in self.iter_rows(d/'native_samples.jsonl'):
            last = snapshot['values']
            require(last['vllm:num_preemptions_total'] == 0, 'Native logs record preemption')
        total = sum(len(r['output_ids']) for r in rows)
        require(last and last['vllm:generation_tokens_total'] == total == result['total_stream_output_tokens'] and
                last['vllm:num_requests_running'] == last['vllm:num_requests_waiting'] == 0,
                'Native final drain or accepted token accounting differs')
        lo,hi = window['begin'],window['end']
        require(hi-lo == m['workload']['duration_seconds'] == result['duration_seconds'] == 300 and
                window['agents'] == 8 and window['warmup_turns'] == 2, 'Fixed 300s C8 window changed')
        for r in rows:
            require(sum(n for _,n in r['timeline']) == len(r['output_ids']) and
                    all(r['begin'] <= at <= r['end'] and type(n) is int and n > 0 for at,n in r['timeline']) and
                    all(a[0] <= b[0] for a,b in zip(r['timeline'],r['timeline'][1:])),
                    'Accepted timeline does not reconcile')
        offered = [r for r in rows if r['kind'] == 'workload' and lo <= r['begin'] < hi]
        complete = [r for r in offered if r['end'] < hi]
        tokens = sum(n for r in offered for at,n in r['timeline'] if lo <= at < hi)
        require(tokens == result['output_tokens'] and len(offered) == result['requests_started_in_window'] and
                len(complete) == result['requests_completed_in_window'] and
                math.isclose(tokens/(hi-lo),result['output_tokens_per_second'],rel_tol=1e-12),
                'Fixed-window accepted output accounting differs')
        counters = self.function(root/'frozen/scripts/benchmark_timed_agents.py',
                                ('counter_bracket','public_work_summary'), {})
        work = counters['public_work_summary'](self.rows(d/'server/capture/public_work.jsonl'),lo,hi)
        require(work == result['public_work'] and work.get('complete_brackets') is True,
                'Public prefill counter evidence differs or lacks brackets')
        steps = [r for r in self.rows(d/'server/capture/model_step_latency.jsonl') if lo <= r['started_at'] < hi]
        require(steps, 'Missing model-step telemetry')
        groups,bins = defaultdict(list),defaultdict(list)
        for r in steps:
            mode = 'mixed' if r['decode_requests'] and r['prefill_tokens'] else ('prefill_only'
                if r['prefill_tokens'] else 'decode_only' if r['decode_requests'] else 'maintenance_only')
            groups[mode].append(r)
            if mode == 'decode_only':
                bins[(r['decode_requests'],r.get('decode_context_bucket'))].append(r)
        def stats(group):
            return {k:distribution([r[k] for r in group]) for k in
                    ('model_ms','maintenance_ms','maintenance_host_ms','step_ms')}
        return dict(window=window, accepted_output_tokens=tokens,
            output_tokens_per_wall_second=tokens/(hi-lo), offered_turns=len(offered), completed_turns=len(complete),
            completed_turns_per_second=len(complete)/(hi-lo),
            offered_cohort_final_output_lengths=distribution([len(r['output_ids']) for r in offered]),
            offered_cohort_length_cap_fraction=sum(r['finish_reason']=='length' for r in offered)/len(offered),
            epoch_end_reasons={reason:sum(r.get('event')=='epoch_ended' and r.get('reason')==reason and
                lo <= r['monotonic'] < hi for r in events) for reason in ('context_capacity','declared_turn_limit')},
            public_prefill_tokens_bounds=work['completed_prefill_tokens_in_window_bounds'],
            public_prefill_tokens_per_wall_second_bounds=work['completed_prefill_tokens_per_second_bounds'],
            model_step_groups={key:stats(value) for key,value in groups.items()},
            decode_only_bins={str(key):stats(value) for key,value in bins.items()},
            nvml=result['gpu'], drain_seconds=result['drain_seconds'], preemptions=0,
            configured_kv_pool=acceptance['actual_configured_pool'],
            observed_ttft_p95_seconds=result['ttft_observed_p95_seconds'],
            output_event_gap_p95_seconds=result['output_event_gap_p95_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--prepared-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate, output = args.baseline.resolve(), args.candidate.resolve(), args.output.resolve()
    require(not output.exists() and not any(output.is_relative_to(r) for r in (baseline,candidate)),
            'Output must be a new file outside the original run directories')
    audit = Audit()
    protocol = audit.protocol(baseline,candidate)
    report = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                  baseline=str(baseline), candidate=str(candidate), protocol=protocol,
                  mode='prepared_protocol_only' if args.prepared_only else 'single_pair_completed_comparison')
    if not args.prepared_only:
        before,after = audit.measured(baseline),audit.measured(candidate)
        if 'cuda_allocator_conf' in protocol['profile_delta']:
            require(before['configured_kv_pool'] == after['configured_kv_pool'],
                    'Allocator ablation changed the actual configured KV pool')
        report.update(baseline_metrics=before, candidate_metrics=after,
            relative_changes={k:after[k]/before[k]-1 if before[k] else None for k in
                              ('output_tokens_per_wall_second','completed_turns_per_second')},
            matched_decode_bins={k:dict(baseline=before['decode_only_bins'][k],candidate=after['decode_only_bins'][k])
                                 for k in sorted(before['decode_only_bins'].keys() & after['decode_only_bins'].keys())},
            limitations=[
                'One A/B order is provisional; no statistical significance or robust speedup claim without repeated reverse order.',
                'Accepted output / full wall window includes tool waits, public prefill, restore and maintenance; it is not pure decode kernel throughput.',
                'Public prefill bounds count actual public prompt positions, excluding restored KV and private Query work; mixed-batch token rates are not isolated prefill kernel rates.',
                'Model event spans include MTP/sampling/capture. Maintenance event/wall durations do not measure CPU-only work or GPU idle.',
                'Matched decode bins control count/context bucket only; MTP acceptance and restore overlap may differ. Natural outputs change turn/epoch exposure.',
                'NVML samples do not establish SM occupancy or exact idle intervals. Fixed external tool replay is not on-policy task solve accuracy.',
            ])
    else:
        report['performance_claim'] = None
        report['completion_claim'] = None
    audit.hash(Path(__file__))
    report['evidence_sha256'] = audit.evidence
    with output.open('x') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output),mode=report['mode'],
        production_changed_files=len(protocol['production_delta']),
        evidence_files=len(audit.evidence))))


if __name__ == '__main__':
    main()

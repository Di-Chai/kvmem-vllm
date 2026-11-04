"""Byte-tested Query optimizations, followed by exact full-input qualification.

This finite queue never resumes an attempted arm or replaces saved answers.
GPU component tests and model inference both require the previous owner to exit.
"""
import argparse
from copy import deepcopy
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import recall_comparison_protocol as p
from run_layerwise_query_diagnostic import verify

REPO = Path(__file__).resolve().parents[1]
MODULES = ('layerwise_query.py', 'layerwise_pressure_efficiency.py', 'cuda_scoring.py')
STAGES = ('batched_reads', 'packed_assembly', 'prepared_index')


def released(queue):
    """Check actual handles; terminal metadata alone cannot release ownership."""
    if p.read(queue/'status.json')['status'] not in ('completed', 'failed'):
        raise RuntimeError('Previous queue is still active')
    for name in ('process.json', 'child.json'):
        path = queue/name
        if not path.exists():
            raise RuntimeError('Missing previous process ownership record')
        stat = Path('/proc')/str(p.read(path)['pid'])/'stat'
        if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] not in ('Z', 'X'):
            raise RuntimeError('Previous queue process has not exited')
    import pynvml
    pynvml.nvmlInit()
    if pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(0)):
        raise RuntimeError('GPU has another owner')


def verify_components(root):
    plan = p.read(root/'plan.json')
    if p.sha(root/'source-proof.json') != plan['source_proof_sha256']:
        raise ValueError('Frozen component manifest changed')
    for name, digest in p.read(root/'source-proof.json').items():
        if p.sha(root/name) != digest:
            raise ValueError('Frozen component source changed: '+name)
    return plan


def test_counts(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    return dict(tests=len(cases), failures=sum(c.find('failure') is not None for c in cases),
                errors=sum(c.find('error') is not None for c in cases),
                skipped=sum(c.find('skipped') is not None for c in cases))


def component_acceptance(root):
    plan = verify_components(root)
    expected = dict(tests=plan.get('expected_tests',16), failures=0, errors=0, skipped=0)
    audit = p.read(root/'gpu-audit.json')
    if (not audit['passed'] or audit['returncode'] != 0 or
            audit['counts'] != expected or
            audit['source_proof_sha256'] != p.sha(root/'source-proof.json') or
            audit['plan_sha256'] != p.sha(root/'plan.json') or
            audit['tests_sha256'] != p.sha(root/'gpu-tests.xml') or
            test_counts(root/'gpu-tests.xml') != audit['counts']):
        raise ValueError('Missing matching, complete GPU component equality evidence')
    return audit


def components(root, previous_queue):
    """One exclusive GPU attempt, including an independent frozen score oracle."""
    from run_recall_comparison import gpu_lock, termination_guard
    from benchmark_query_fps_prebuild_live import server_environment
    from run_layerwise_memory import normalize_cuda_alias
    plan = verify_components(root)
    pointer = p.read(previous_queue.parent/'value-recall-active.json')
    if pointer.get('queue_root') != str(previous_queue):
        raise RuntimeError('Another experiment owns GPU scheduling')
    if any((root/name).exists() for name in ('gpu-test-attempt.json', 'gpu-test.log', 'gpu-audit.json')):
        raise ValueError('Preserve the component attempt; use a fresh version for a technical retry')
    with termination_guard(), gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
        released(previous_queue)
        environment = normalize_cuda_alias(server_environment(root/'runtime-v1'))
        environment.update(plan['environment'])
        command = plan['command']+['--junitxml='+str(root/'gpu-tests.xml')]
        started = time.time()
        p.save(root/'gpu-test-attempt.json', dict(pid=os.getpid(), started_at=started,
            command=command, previous_queue=str(previous_queue), plan_sha256=p.sha(root/'plan.json')))
        with (root/'gpu-test.log').open('x') as log:
            result = subprocess.run(command, cwd=root/'runtime-v1', env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        verify_components(root)
        counts = test_counts(root/'gpu-tests.xml') if (root/'gpu-tests.xml').exists() else {}
        passed = result.returncode == 0 and counts == dict(tests=plan.get('expected_tests',16), failures=0, errors=0, skipped=0)
        p.save(root/'gpu-audit.json', dict(passed=passed, returncode=result.returncode, counts=counts,
            seconds=time.time()-started, plan_sha256=p.sha(root/'plan.json'),
            source_proof_sha256=p.sha(root/'source-proof.json'),
            tests_sha256=p.sha(root/'gpu-tests.xml') if counts else None,
            scope=plan.get('scope', 'Component equality only; full-input pressure/Query/output qualification is still required.')))
        if not passed:
            raise RuntimeError('GPU byte/score equality failed or tests were skipped; preserve the evidence')


def source_audit(source):
    from run_layerwise_pressure_efficiency import signature, full_prefix_candidate_proof
    verify(source)
    report = p.read(source/'equivalence.json')
    if not report['passed'] or report['cases'] != 8:
        raise ValueError('Source must pass all eight original full-input comparisons')
    baseline = Path(report['baseline']); verify(baseline)
    fixture = p.read(source/'fixture.json')
    if (fixture['layerwise_workers'] != 1 or not fixture['layerwise_prefill_pressure'] or
            len(fixture['rows']) != 8):
        raise ValueError('Expected the fixed-order serial full-input source')
    checks = []
    for row in fixture['rows']:
        cases = [r/'formal/sequential/cases'/row['id'] for r in (baseline, source)]
        evidence = [p.read(c/'evidence.json') for c in cases]
        responses = [p.read(c/'response.json') for c in cases]
        requests = [p.read(c/'request.json') for c in cases]
        if (signature(evidence[0]) != signature(evidence[1]) or
                any(responses[0][k] != responses[1][k] for k in ('output_ids', 'text')) or
                any(requests[0][k] != requests[1][k] for k in ('sampling', 'input_sha256', 'query_span'))):
            raise ValueError('Source failed the current strict scope/output audit')
        for case in cases:
            result = p.read(case/'result.json')
            if any(result[k+'_sha256'] != p.sha(case/(k+'.json')) for k in ('response', 'evidence')):
                raise ValueError('Source first answer or evidence changed')
        checks.append(dict(id=row['id'], proofs=[full_prefix_candidate_proof(e,p.read(r/'profile.json'))
            for e,r in zip(evidence, (baseline, source))]))
    profile = p.read(source/'profile.json')
    if profile['kvmem'].get('action_native_scores', 'off') != 'off':
        if not p.read(source/'native-reuse-audit.json')['passed']:
            raise ValueError('Native reuse source did not pass its execution audit')
    if fixture.get('cpu_region_gather_min_bytes') is not None or profile['kvmem'].get('cpu_direct_record_seal',False):
        from run_layerwise_matched_performance import archive_backend_audit
        archive_backend_audit(profile,p.read(source/'formal/sequential/server-attempts/0001/server/capture/archive.json'),
                              fixture.get('cpu_region_gather_min_bytes'))
    return checks


def serve_flags(path):
    text = path.read_text()
    anchor = '    args = parser.parse_args(argv)\n'
    if text.count(anchor) != 1 or any(flag in text for flag in (
            '--layerwise-packed-assembly','--layerwise-prepared-index','--layerwise-batched-reads')):
        raise ValueError('Unexpected source Query optimization flags')
    text = text.replace(anchor,
        "    parser.add_argument('--layerwise-packed-assembly', action='store_true')\n"
        "    parser.add_argument('--layerwise-prepared-index', action='store_true')\n"
        "    parser.add_argument('--layerwise-batched-reads', action='store_true')\n"+anchor)
    anchor = "            plan['environment']['KVMEM_LAYERWISE_QUERY_OPTIMIZED'] = '1'\n"
    if text.count(anchor) != 1:
        raise ValueError('Missing optimized Query environment anchor')
    text = text.replace(anchor, anchor+
        "        if args.layerwise_batched_reads:\n"
        "            if not args.layerwise_query_optimized: raise ValueError('Batched reads require optimized Query')\n"
        "            plan['environment']['KVMEM_LAYERWISE_BATCHED_READS'] = '1'\n"
        "        if args.layerwise_packed_assembly:\n"
        "            if not args.layerwise_query_optimized: raise ValueError('Packed assembly requires optimized Query')\n"
        "            plan['environment']['KVMEM_LAYERWISE_PACKED_ASSEMBLY'] = '1'\n"
        "        if args.layerwise_prepared_index:\n"
        "            if not args.layerwise_query_optimized: raise ValueError('Prepared index requires optimized Query')\n"
        "            plan['environment']['KVMEM_LAYERWISE_PREPARED_INDEX'] = '1'\n")
    path.write_text(text)


def prepare(root, source, component_root):
    if root.exists():
        raise ValueError('Use a new version; preserve all previous attempts')
    checks = source_audit(source)
    component_acceptance(component_root)
    # Keep the same native score kernel and assembly dependencies tested above.
    for name in ('csrc/qw3_mean_k.cu', 'csrc/qw3_mean_k_kernel.cuh', 'csrc/qw3_mean_k_tiled.cuh',
                 'src/vllm_kvmem/split_assembly.py', 'src/vllm_kvmem/assembly.py'):
        if p.sha(source/'runtime-v1'/name) != p.sha(component_root/'runtime-v1'/name):
            raise ValueError('Component dependency differs from the accepted source: '+name)
    root.mkdir(parents=True)
    stages = []
    for index, name in enumerate(STAGES):
        arm = root/name; arm.mkdir(); runtime = arm/'runtime-v1'
        shutil.copytree(source/'runtime-v1', runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for folder in ('inputs', 'upstream'):
            shutil.copytree(source/folder, arm/folder)
        for module in MODULES:
            shutil.copy2(component_root/'runtime-v1/src/vllm_kvmem'/module, runtime/'src/vllm_kvmem'/module)
        shutil.copy2(REPO/'scripts/run_concurrent_layerwise_ruler.py', runtime/'scripts/run_concurrent_layerwise_ruler.py')
        serve_flags(runtime/'src/vllm_kvmem/serve.py')
        fixture, profile = deepcopy(p.read(source/'fixture.json')), deepcopy(p.read(source/'profile.json'))
        fixture.update(layerwise_batched_reads=True, layerwise_packed_assembly=index >= 1, layerwise_prepared_index=index == 2,
            qualification_only=True, cohort='query-execution-full-input8')
        profile.update(port=19090+index, served_model_name='query-efficiency-'+name)
        p.save(arm/'fixture.json', fixture); p.save(arm/'profile.json', profile)
        p.save(arm/'runtime-proof.json', {str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
            if f.is_file() and '__pycache__' not in f.parts})
        p.save(arm/'prepared.json', dict(at=time.time(), expected_runs=8,
            profile_sha256=p.sha(arm/'profile.json'), fixture_sha256=p.sha(arm/'fixture.json'),
            runtime_proof_sha256=p.sha(arm/'runtime-proof.json')))
        verify(arm)
        p.save(arm/'status.json', dict(status='prepared', completed=0, expected=8))
        stages.append(dict(name=name, root=str(arm), prepared_sha256=p.sha(arm/'prepared.json')))
    shutil.copy2(__file__, root/'launch.py')
    shutil.copy2(REPO/'scripts/run_layerwise_pressure_efficiency.py', root/'comparison-code.py')
    source_files = {str(source/n):p.sha(source/n) for n in
        ('prepared.json', 'equivalence.json', 'fixture.json', 'profile.json', 'runtime-proof.json')}
    for row in p.read(source/'fixture.json')['rows']:
        case = source/'formal/sequential/cases'/row['id']
        source_files.update({str(case/n):p.sha(case/n) for n in
            ('request.json', 'response.json', 'evidence.json', 'result.json')})
    p.save(root/'plan.json', dict(source=str(source), component_root=str(component_root),
        component_audit_sha256=p.sha(component_root/'gpu-audit.json'), source_files=source_files,
        stages=stages, source_checks=checks, launch_sha256=p.sha(root/'launch.py'),
        comparator_sha256=p.sha(root/'comparison-code.py'), scheduled_monitor=False,
        no_automatic_retry=True, expected_cases=8,
        acceptance='Identical complete-input pressure candidates/blocks, Query scope/blocks and full tokens/text.'))
    p.save(root/'status.json', dict(status='prepared', at=time.time()))


def run(root):
    from run_recall_comparison import termination_guard, stop_owned
    plan = p.read(root/'plan.json'); source = Path(plan['source'])
    if (p.sha(root/'launch.py') != plan['launch_sha256'] or
            p.sha(root/'comparison-code.py') != plan['comparator_sha256']):
        raise ValueError('Frozen qualification launcher changed')
    for name,digest in plan['source_files'].items():
        if p.sha(Path(name)) != digest:
            raise ValueError('Accepted source evidence changed')
    if p.sha(Path(plan['component_root'])/'gpu-audit.json') != plan['component_audit_sha256']:
        raise ValueError('Component acceptance changed')
    spec = importlib.util.spec_from_file_location('query_efficiency_comparator', root/'comparison-code.py')
    comparator = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparator)
    pointer = root.parent/'value-recall-active.json'
    with termination_guard(), (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if p.read(pointer).get('queue_root') != str(source.parent):
            raise RuntimeError('Another experiment owns GPU scheduling')
        released(source.parent)
        if any((Path(stage['root'])/'driver.log').exists() for stage in plan['stages']):
            raise ValueError('Do not rerun attempted arms')
        p.save(root/'process.json', dict(pid=os.getpid(), at=time.time()))
        p.save(pointer, dict(queue_root=str(root), root=str(root), status=str(root/'status.json'),
            process=str(root/'process.json'), experiment_type='query-execution-equivalence', at=time.time()))
        for stage in plan['stages']:
            arm = Path(stage['root']); verify(arm)
            if p.sha(arm/'prepared.json') != stage['prepared_sha256']:
                raise ValueError('Frozen candidate changed')
            p.save(root/'status.json', dict(status='running', stage=stage['name'], active_root=str(arm), at=time.time()))
            command = [sys.executable, '-B', str(arm/'runtime-v1/scripts/run_concurrent_layerwise_ruler.py'), '--root', str(arm)]
            with (arm/'driver.log').open('x') as log:
                child = subprocess.Popen(command, cwd=arm/'runtime-v1', stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            p.save(root/'child.json', dict(pid=child.pid, stage=stage['name'], command=command, at=time.time()))
            try:
                returncode = child.wait()
            except BaseException:
                stop_owned(child)
                raise
            if returncode != 0:
                raise RuntimeError('Technical qualification failure; preserve all first answers')
            report = comparator.compare(arm, source)
            capture = arm/'formal/sequential/server-attempts/0001/server/capture'
            stats = p.read(capture/'native_attention_stats.json'); counts = stats['layer_phase_counts']
            good = (len(stats['producer_layers']) == 16 and not stats['disabled_reason'] and
                not stats['validation_failures'] and counts.get('reuse:prefill',0) > 0 and
                not any(v for k,v in counts.items() if k.startswith('fallback:prefill')))
            p.save(arm/'native-reuse-audit.json', dict(passed=good, statistics=stats))
            backend = []
            for row in p.read(arm/'fixture.json')['rows']:
                evidence = p.read(arm/'formal/sequential/cases'/row['id']/'evidence.json')
                for layer in evidence['layers']:
                    assembly = layer['assembly_backend']
                    backend.append(dict(id=row['id'], layer=layer['layer'], **assembly,
                        prepared_candidate_index=layer['prepared_candidate_index']))
            exercised = len(backend) == 128 and all(
                b['chunk_blocks'] == 256 and
                (b['packed_chunks'] > 0) == (stage['name'] != 'batched_reads') and
                b['prepared_candidate_index'] == (stage['name'] == 'prepared_index')
                for b in backend)
            p.save(arm/'query-backend-audit.json', dict(passed=exercised, layers=backend))
            if not report['passed'] or not good or not exercised:
                raise RuntimeError('Exact equality, native reuse or intended Query backend failed')
        p.save(root/'status.json', dict(status='completed', stages=len(plan['stages']), at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('components', 'prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--components', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    try:
        if args.action == 'components': components(root, args.source.resolve())
        elif args.action == 'prepare': prepare(root, args.source.resolve(), args.components.resolve())
        else: run(root)
    except BaseException as error:
        if args.action == 'run' and root.exists():
            p.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise

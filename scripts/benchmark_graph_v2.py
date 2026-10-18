"""Freeze V2 A/C/D benchmarks; default preparation is eight 256K NIAH cases.

Preparation never starts a service or touches CUDA. Run only a fresh frozen
controller; full NIAH180/MAB1116/independent-LME102 is an explicit scope.
"""
import argparse
import ast
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import benchmark_block_graph as bench
import benchmark_niah as niah
import graph_v2_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ('benchmark_graph_v2.py', 'qualify_graph_v2.py', 'graph_v2_protocol.py',
    'benchmark_block_graph.py', 'benchmark_niah.py', 'qualify_graph_niah.py',
    'qualify_block_graph.py', 'benchmark_memory.py', 'benchmark_longmemeval.py',
    'analyze_graph_v2.py', 'graph_v2_evaluation.py', 'analyze_memory.py', 'analyze_longmemeval.py')
sha, require, write = protocol.sha, protocol.require, bench.write


def cpu_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Missing available host memory observation')


def freeze_niah(args, root, profile):
    source = args.reference.resolve()
    subset = args.scope == 'smoke8'
    settings = SimpleNamespace(reference=source, config=root/'niah-A.json', output=root/'dataset/niah',
        lengths=[262144] if subset else [65536, 131072, 262144],
        depths=[10, 90] if subset else [10, 30, 50, 70, 90], seeds=[1000] if subset else [1000, 2000, 3000],
        max_output=8192, workers=8, cpu_headroom_gib=16, resume=False)
    rows = niah.prepare(settings, niah.normalized_config(profile))
    require(len(rows) == (8 if subset else 180), 'Frozen NIAH cohort differs')
    original = (source/'run.py').read_text()
    node = next(node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name == 'score')
    return dict(schema=protocol.SCHEMA, suite='niah', scope=args.scope, rows=rows,
        input_directory=str(settings.output/'inputs'), workers=8,
        tokenizer_sha256=sha(Path(profile['tokenizer'])/'tokenizer.json'),
        source_cohort_sha256=sha(source/'cohort.json'), source_scorer_sha256=sha(source/'run.py'),
        scorer_source=ast.get_source_segment(original, node),
        cpu_payload_credits_bytes=100000000000-16*1024**3, request_timeout_seconds=3600,
        sampling=dict(max_tokens=8192, temperature=1., top_p=.95, top_k=20, min_p=0,
            presence_penalty=0, repetition_penalty=1, thinking_token_budget=4096),
        score_domain_version=protocol.SCORE_DOMAIN, original_input_intact=True,
        gold_used_for_selection=False, locality_metadata='unknown: original token-only source',
        source_cases_per_length={str(length):sum(row['nominal_input_tokens'] == length for row in rows)
                                 for length in settings.lengths})


def freeze_reference(args, root, suite, profile):
    """Bind exact compressed source and request samples; no re-encoding."""
    import benchmark_memory as memory
    source = args.memory_reference.resolve()
    original = json.loads((source/'cohort.json').read_text())
    rows = [deepcopy(row) for row in original['cases'] if row['suite'] == suite]
    require(len(rows) == (1116 if suite == 'memoryagentbench' else 102) and
            len({row['id'] for row in rows}) == len(rows), 'Original reference cohort size/identity differs')
    sampling = {row['id']:json.loads((source/'results/attention/cases'/suite/row['id']/
        'request_config.json').read_text()) for row in rows}
    capacity = protocol.preflight_reference(rows, sampling, profile, suite)
    destination = root/'dataset'/suite
    destination.mkdir(parents=True)
    bindings = []
    for row in rows:
        # Validate tokens, UTF-8 prompt and public Query coordinates before the
        # source becomes an immutable campaign input.
        original_source = memory.source(row)
        require(original_source['max_tokens'] == sampling[row['id']]['max_tokens'],
            'Original source and request output limits differ')
        frozen = destination/'sources'/(row['id']+'.json.gz')
        frozen.parent.mkdir(exist_ok=True)
        shutil.copyfile(row['source'], frozen)
        original_path = row['source']
        row['source'] = str(frozen)
        request = source/'results/attention/cases'/suite/row['id']/'request_config.json'
        target = destination/'results/attention/cases'/suite/row['id']/'request_config.json'
        target.parent.mkdir(parents=True)
        shutil.copyfile(request, target)
        bindings.append(dict(id=row['id'], original_source=original_path,
            source_sha256=sha(frozen), request_sha256=sha(target), token_sha256=row['token_sha256']))
    write(destination/'cohort.json', dict(cases=rows))
    from graph_v2_evaluation import freeze_reference_evidence
    evaluation_sources = freeze_reference_evidence(source, destination, suite, rows)
    return dict(schema=protocol.SCHEMA, suite=suite, scope='full', reference=str(destination),
        cases=len(rows), rows=rows, bindings=bindings, workers=8,
        source_cohort_sha256=sha(source/'cohort.json'), gold_used_for_selection=False,
        score_domain_version=protocol.SCORE_DOMAIN,
        capacity_preflight=capacity,
        evaluation_reference_sha256=evaluation_sources,
        protocol='frozen-P independent questions' if suite == 'memoryagentbench' else 'cold independent histories',
        locality_metadata='unknown: original token-only source')


def model_identity(profile):
    directory = Path(profile['model'])
    config = json.loads((directory/'config.json').read_text())
    quant = config.get('quantization_config', {})
    groups = quant.get('config_groups', {}).values()
    require(quant.get('quant_method') == 'modelopt' and any(
        group.get('weights', {}).get('num_bits') == 4 and
        group.get('input_activations', {}).get('num_bits') == 4 for group in groups),
        'Benchmark requires the NVFP4 model configuration, not a filename inference')
    weights = {str(path.resolve()):dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
               for path in directory.glob('*.safetensors')}
    require(weights, 'Missing native safetensors model weights')
    return dict(config_sha256={str(path.resolve()):sha(path) for path in directory.glob('*.json')},
        tokenizer_sha256={str(path.resolve()):sha(path) for path in Path(profile['tokenizer']).glob('*.json')},
        weight_stat_identity=weights, quantization='NVFP4 modelopt; FP8 KV verified in actual argv')


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write(root/'status.json', dict(status='preparing', at=time.time()))
    try:
        base = json.loads(args.profile.read_text())
        identity = model_identity(base)
        suites = ('niah',) if args.scope == 'smoke8' else protocol.SUITES
        configurations = {suite:protocol.profiles(base, args.port, suite) for suite in suites}
        from vllm_kvmem.serve import build_launch
        for suite, arms in configurations.items():
            for arm, profile in arms.items():
                build_launch(profile, ROOT, root/(suite+'-'+arm)/'server')
                write(root/(suite+'-'+arm+'.json'), profile)
        fixtures = {'niah': freeze_niah(args, root, configurations['niah']['A'])}
        for suite in suites[1:]:
            fixtures[suite] = freeze_reference(args, root, suite, configurations[suite]['A'])
        evaluation_sha256 = None
        if args.scope == 'full':
            from graph_v2_evaluation import freeze_evaluation
            evaluation_sha256 = freeze_evaluation(root)
        jobs, runtime_hashes = [], None
        for suite in suites:
            fixture = root/(suite+'-fixture.json')
            write(fixture, fixtures[suite])
            for arm in protocol.ARMS:
                job = suite+'-'+arm
                target = root/job
                bench.prepare(SimpleNamespace(output=target, fixture=fixture, arm=arm,
                    profile=root/(job+'.json'), qualifier=ROOT/'scripts/qualify_graph_v2.py',
                    synchronous=False, port=args.port, client_timeout_seconds=7200 if args.scope == 'smoke8' else 86400))
                for name in HELPERS:
                    shutil.copyfile(ROOT/'scripts'/name, target/'runtime/scripts'/name)
                manifest = json.loads((target/'manifest.json').read_text())
                hashes = {str(path.relative_to(target/'runtime')):sha(path)
                          for path in (target/'runtime').rglob('*') if path.is_file()}
                require(any(name.startswith('src/vllm_kvmem/_native') and name.endswith('.so') for name in hashes),
                    'Frozen runtime lacks the current built native extension')
                if runtime_hashes is not None:
                    require(hashes == runtime_hashes, 'Source changed while freezing the campaign')
                runtime_hashes = hashes
                manifest.update(runtime_sha256=hashes, schema=protocol.SCHEMA, suite=suite)
                write(target/'manifest.json', manifest)
                jobs.append(dict(id=job, suite=suite, arm=arm, cases=len(fixtures[suite]['rows']),
                    manifest_sha256=sha(target/'manifest.json')))
        frozen = root/'frozen'
        frozen.mkdir()
        for name in HELPERS:
            shutil.copyfile(ROOT/'scripts'/name, frozen/name)
            require(sha(frozen/name) == runtime_hashes['scripts/'+name],
                    'Controller changed while freezing the campaign: '+name)
        patch = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src', 'csrc', 'third_party', 'scripts'], cwd=ROOT)
        (root/'source.diff').write_bytes(patch)
        worktree = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'csrc', 'third_party', 'scripts'],
                                          cwd=ROOT, text=True)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        inputs = {str(path.relative_to(root)):sha(path) for path in (root/'dataset').rglob('*') if path.is_file()}
        maximum = max(niah_row['served_input_tokens'] for niah_row in fixtures['niah']['rows'])+8192+16384
        credit = ((maximum+31)//32)*32*17*4*256*2
        require(8*credit <= fixtures['niah']['cpu_payload_credits_bytes'], 'CPU credits cannot admit eight intact histories')
        manifest = dict(schema=protocol.SCHEMA, scope=args.scope, jobs=jobs, created_at=time.time(),
            source_git_commit=commit, source_diff_sha256=sha(root/'source.diff'),
            source_dirty=bool(worktree.strip()), source_worktree_status=worktree,
            runtime_sha256=runtime_hashes,
            frozen_sha256={path.name:sha(path) for path in frozen.iterdir()},
            fixture_sha256={suite:sha(root/(suite+'-fixture.json')) for suite in suites},
            input_sha256=inputs, model_identity=identity, score_domain_version=protocol.SCORE_DOMAIN,
            evaluation_sha256=evaluation_sha256,
            workers=8, start_cpu_available_bytes=8*credit+24*1024**3,
            minimum_cpu_available_bytes=8*1024**3, minimum_gpu_free_mib=4096,
            allowed_arm_differences=['kvmem.graph_mode', 'kvmem.pressure_policy'],
            scope_note='smoke8 is functional/performance qualification, not the full quality benchmark')
        write(root/'manifest.json', manifest)
        write(root/'status.json', dict(status='prepared', at=time.time()))
        print(json.dumps(dict(prepared=str(root), scope=args.scope, jobs=jobs,
            run=str(frozen/'benchmark_graph_v2.py'), gpu_started=False)), flush=True)
    except BaseException as error:
        write(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise


def validate(root):
    manifest = json.loads((root/'manifest.json').read_text())
    require(manifest.get('schema') == protocol.SCHEMA, 'Unknown campaign schema')
    require(sha(root/'source.diff') == manifest['source_diff_sha256'], 'Source patch identity changed')
    for name, expected in manifest['frozen_sha256'].items():
        require(sha(root/'frozen'/name) == expected, 'Frozen controller changed: '+name)
    for name, expected in manifest['input_sha256'].items():
        require(sha(root/name) == expected, 'Frozen input/sample changed: '+name)
    for suite, expected in manifest['fixture_sha256'].items():
        require(sha(root/(suite+'-fixture.json')) == expected, 'Frozen fixture changed')
        configs = {arm:json.loads((root/(suite+'-'+arm)/'profile.json').read_text()) for arm in protocol.ARMS}
        protocol.validate_profiles(configs, suite)
    if manifest.get('scope') == 'full':
        from graph_v2_evaluation import validate_evaluation
        validate_evaluation(root, manifest.get('evaluation_sha256'))
    for job in manifest['jobs']:
        directory = root/job['id']
        require(sha(directory/'manifest.json') == job['manifest_sha256'], 'Job manifest changed')
        frozen = json.loads((directory/'manifest.json').read_text())
        require(frozen['runtime_sha256'] == manifest['runtime_sha256'], 'Arms use different runtimes')
        require(sha(directory/'profile.json') == frozen['profile_sha256'] and
                sha(directory/'fixture.json') == frozen['fixture_sha256'], 'Job profile/fixture changed')
        for name, expected in frozen['runtime_sha256'].items():
            require(sha(directory/'runtime'/name) == expected, 'Frozen runtime changed: '+name)
    identity = manifest['model_identity']
    for group in ('config_sha256', 'tokenizer_sha256'):
        for name, expected in identity[group].items():
            require(sha(name) == expected, 'Model/tokenizer identity changed')
    for name, expected in identity['weight_stat_identity'].items():
        stat = Path(name).stat()
        require(dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == expected, 'Model weight file identity changed')
    return manifest


def run(args):
    root = args.output.resolve()
    # Claim the campaign before inspecting its status. Two controllers must
    # never both consume the same prepared jobs, including different arms.
    with (root/'.campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_locked(args, root)


def _run_locked(args, root):
    manifest = validate(root)
    require(Path(__file__).resolve() == root/'frozen/benchmark_graph_v2.py', 'Run the frozen controller')
    require(json.loads((root/'status.json').read_text())['status'] in ('prepared', 'partial'),
        'A failed/interrupted campaign cannot be retried; prepare a new attempt')
    selected = [job for job in manifest['jobs'] if job['arm'] in args.arms and job['suite'] in args.suites]
    require(selected, 'No prepared jobs match the requested arm/suite')
    pending = []
    for job in selected:
        status = json.loads((root/job['id']/'status.json').read_text())['status']
        require(status in ('prepared', 'passed'), 'Failed/interrupted job is retained: '+job['id'])
        if status == 'prepared':
            pending.append(job)
    require(pending, 'All selected jobs already passed; nothing will be silently rerun')
    write(root/'status.json', dict(status='running', jobs=[job['id'] for job in pending], at=time.time()))
    try:
        with bench.cleanup_on_termination() as begin_cleanup:
            for job in pending:
                require(cpu_available() >= manifest['start_cpu_available_bytes'], 'Insufficient host headroom for C8 histories')
                target = root/job['id']
                command = [sys.executable, '-B', str(target/'runtime/scripts/benchmark_graph_v2.py'),
                           'arm-run', '--output', str(target)]
                process = None
                try:
                    with (target/'controller.log').open('x') as log, (target/'resources.jsonl').open('x') as samples:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
                        while process.poll() is None:
                            available = cpu_available()
                            fields = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu,power.draw',
                                '--format=csv,noheader,nounits'], text=True).strip().split(',')
                            free, util, power = [float(value.strip()) for value in fields]
                            samples.write(json.dumps(dict(at=time.time(), cpu_available_bytes=available,
                                gpu_free_mib=free, gpu_utilization=util, gpu_power_w=power))+'\n')
                            samples.flush()
                            require(available >= manifest['minimum_cpu_available_bytes'] and
                                    free >= manifest['minimum_gpu_free_mib'], 'Runtime memory guard reached')
                            time.sleep(1)
                    require(process.returncode == 0, 'Job failed; retained logs: '+job['id'])
                finally:
                    if process is not None:
                        require(not bench.stop_owned_groups((process,)), 'Owned controller did not stop')
            begin_cleanup()
        complete = all(json.loads((root/job['id']/'status.json').read_text())['status'] == 'passed'
                       for job in manifest['jobs'])
        write(root/'status.json', dict(status='completed' if complete else 'partial', at=time.time()))
    except BaseException as error:
        write(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'validate', 'arm-run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=('smoke8', 'full'), default='smoke8')
    parser.add_argument('--profile', type=Path, default=ROOT/'artifacts/graph-overhead-20261018/gpu-r2/after/profile.json')
    parser.add_argument('--reference', type=Path,
        default=Path('/data/chaidi/qw3-experiments/cacheblend_20260925/official_template_20260927'))
    parser.add_argument('--memory-reference', type=Path,
        default=Path('/home/chaidi/qw3/benchmark/attention_multilayer_prefix_reuse_20261012_r2'))
    parser.add_argument('--port', type=int, default=18678)
    parser.add_argument('--arms', nargs='+', choices=protocol.ARMS, default=list(protocol.ARMS))
    parser.add_argument('--suites', nargs='+', choices=protocol.SUITES, default=list(protocol.SUITES))
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'run':
        run(args)
    elif args.action == 'arm-run':
        require(Path(__file__).resolve() == args.output.resolve()/'runtime/scripts/benchmark_graph_v2.py',
                'Arm runner must use its frozen runtime')
        bench.run(args)
    else:
        result = validate(args.output.resolve())
        print(json.dumps(dict(status='validated', scope=result['scope'], jobs=result['jobs'])))


if __name__ == '__main__':
    main()

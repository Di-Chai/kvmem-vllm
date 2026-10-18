"""Prepare a frozen MeanK cache/numerical qualification; GPU execution is explicit.

Two independent Python processes load identical kernels from different absolute
package roots. A third executes the unchanged original CUDA numerical oracle.
This qualifies loader reuse, not serving readiness or a performance speedup.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
CUDA_FILES = ('qw3_mean_k.cu', 'qw3_mean_k_kernel.cuh', 'qw3_mean_k_tiled.cuh')
PYTHON_FILES = ('__init__.py', 'cuda_scoring.py', 'retrieval.py')
PATCH_FILES = ('src/vllm_kvmem/runner.py', 'src/vllm_kvmem/cuda_scoring.py',
               'tests/test_mean_k_startup.py', 'tests/test_cuda_scoring_source_cache.py')
ORACLE = 'test_original_cuda_scoring.py'
ENTRY = 'qualify_mean_k_startup.py'
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'TORCH_CUDA_ARCH_LIST', 'CUDA_HOME', 'CXX',
            'CC', 'CUDAHOSTCXX', 'MAX_JOBS', 'TORCH_EXTENSIONS_DIR')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def patch_provenance(candidate, workspace):
    manifest = read(candidate / 'manifest.json')
    require(manifest.get('patch_sha256') == sha(candidate / 'candidate.patch'), 'Candidate patch hash mismatch')
    require(set(manifest.get('files', {})) == set(PATCH_FILES), 'Unexpected candidate patch scope')
    require(manifest.get('status') == 'cpu_verified_gpu_pending', 'Candidate lacks CPU verification provenance')
    for relative in PATCH_FILES:
        require(manifest['files'][relative].get('after_sha256') == sha(workspace / relative),
                'Candidate source mismatch: ' + relative)
    require(manifest.get('cpu_validation', {}).get('passed', 0) > 0, 'Missing candidate CPU validation')
    return manifest


def prepare(output, candidate, workspace=ROOT):
    output, candidate, workspace = output.resolve(), candidate.resolve(), workspace.resolve()
    require(not output.exists(), 'Use a fresh qualification directory')
    provenance = patch_provenance(candidate, workspace)
    oracle = workspace / 'tests' / ORACLE
    require(oracle.is_file(), 'Original numerical oracle is missing')
    source = workspace / 'src/vllm_kvmem/cuda_sources'
    if not source.exists():
        source = workspace / 'csrc'
    # Read all source bytes before creating the prepared directory.
    kernels = {name: (source / name).read_bytes() for name in CUDA_FILES}
    python = {name: (workspace / 'src/vllm_kvmem' / name).read_bytes() for name in PYTHON_FILES}
    output.mkdir(parents=True)
    for label in ('a', 'b'):
        package = output / 'snapshots' / label / 'vllm_kvmem'
        (package / 'cuda_sources').mkdir(parents=True)
        for name, content in python.items():
            (package / name).write_bytes(content)
        for name, content in kernels.items():
            (package / 'cuda_sources' / name).write_bytes(content)
    (output / 'frozen').mkdir()
    shutil.copyfile(Path(__file__), output / 'frozen' / ENTRY)
    (output / 'oracle').mkdir()
    shutil.copyfile(oracle, output / 'oracle' / ORACLE)
    proof = output / 'provenance'
    proof.mkdir()
    for name in ('candidate.patch', 'manifest.json'):
        shutil.copyfile(candidate / name, proof / name)
    for relative in PATCH_FILES:
        path = proof / 'after' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / relative, path)
    immutable = {name: inventory(output / name) for name in ('snapshots', 'frozen', 'oracle', 'provenance')}
    save(output / 'manifest.json', dict(kind='mean_k_startup_qualification_v1', immutable=immutable,
        candidate_source=str(candidate), candidate_patch_sha256=provenance['patch_sha256'],
        oracle_sha256=sha(oracle), source_sha256={name: hashlib.sha256(data).hexdigest() for name, data in kernels.items()},
        python_executable=str(Path(sys.executable).resolve()),
        torch_distribution_version=importlib.metadata.version('torch'), expected_oracle_tests=9,
        flags=dict(extra_cuda_cflags=['-O3'], verbose=False),
        build_root_policy='New dedicated temporary directory under this qualification root; never the old shared cache. Retained for audit.',
        scope='Independent-process build cache reuse plus original CUDA numerical oracle. '
              'No server startup, first-kernel warmup, model, throughput, or cancellation-quality claim.',
        remaining='The runner preload hook is retained as patch provenance and CPU-tested separately; '
                  'a real serving readiness test is not part of this loader qualification.'))
    save(output / 'status.json', dict(status='prepared', at=time.time()))
    return verify(output, frozen_entry=False)


def verify(output, *, frozen_entry=True):
    output = output.resolve()
    manifest = read(output / 'manifest.json')
    require(manifest.get('kind') == 'mean_k_startup_qualification_v1', 'Wrong qualification kind')
    if frozen_entry:
        require(Path(__file__).resolve() == output / 'frozen' / ENTRY, 'Use the frozen entry point')
    require(set(manifest.get('immutable', {})) == {'snapshots', 'frozen', 'oracle', 'provenance'},
            'Missing immutable source provenance')
    for name, expected in manifest['immutable'].items():
        require(inventory(output / name) == expected, 'Frozen input changed: ' + name)
    provenance = patch_provenance(output / 'provenance', output / 'provenance' / 'after')
    require(provenance['patch_sha256'] == manifest['candidate_patch_sha256'], 'Patch identity changed')
    a, b = (output / 'snapshots' / label / 'vllm_kvmem' for label in ('a', 'b'))
    require(a.resolve() != b.resolve() and inventory(a) == inventory(b), 'Require two distinct identical snapshots')
    expected_paths = set(PYTHON_FILES) | {'cuda_sources/' + name for name in CUDA_FILES}
    require(set(inventory(a)) == expected_paths, 'Snapshot files differ from minimal qualification scope')
    require(sha(a / 'cuda_scoring.py') == provenance['files']['src/vllm_kvmem/cuda_scoring.py']['after_sha256'],
            'Snapshot loader is not the candidate')
    require({name: sha(a / 'cuda_sources' / name) for name in CUDA_FILES} == manifest['source_sha256'],
            'Kernel source identity changed')
    require(sha(output / 'oracle' / ORACLE) == manifest['oracle_sha256'] and manifest['expected_oracle_tests'] == 9,
            'Original numerical oracle changed')
    require(manifest['flags'] == dict(extra_cuda_cflags=['-O3'], verbose=False), 'Compiler flags changed')
    require(str(Path(sys.executable).resolve()) == manifest['python_executable'] and
            importlib.metadata.version('torch') == manifest['torch_distribution_version'], 'Python/Torch environment changed')
    return manifest


def file_identity(path):
    state = path.stat()
    return dict(sha256=sha(path), inode=state.st_ino, mtime_ns=state.st_mtime_ns, size=state.st_size)


def build_evidence(build_root):
    directories = [p for p in build_root.iterdir() if p.is_dir()]
    require(len(directories) == 1, 'Expected exactly one content-addressed build')
    directory = directories[0]
    ninja = directory / 'build.ninja'
    require(ninja.is_file(), 'Missing Ninja build evidence')
    inputs = sorted((directory / 'sources').glob('*/*'))
    require({p.name for p in inputs} == set(CUDA_FILES) and len(inputs) == 3, 'Missing stable kernel/header inputs')
    artifacts = [p for p in directory.rglob('*') if p.suffix in ('.o', '.so')]
    require(any(p.suffix == '.o' for p in artifacts) and any(p.suffix == '.so' for p in artifacts),
            'Missing compiled object or shared library')
    cu = next(p for p in inputs if p.suffix == '.cu')
    require(str(cu) in ninja.read_text(), 'Ninja does not compile the stable input path')
    return dict(build_directory=str(directory), stable_source=str(cu), ninja=file_identity(ninja),
        inputs={str(p): file_identity(p) for p in inputs},
        artifacts={str(p): file_identity(p) for p in sorted(artifacts)})


def unchanged_build(first, second):
    require(first == second, 'Second process changed stable inputs, Ninja, object or library identity/mtime')


def numerical_evidence(path, expected=9):
    tree = ET.parse(path).getroot()
    suites = [tree] if tree.tag == 'testsuite' else list(tree.findall('testsuite'))
    totals = {key: sum(int(s.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    require(totals == dict(tests=expected, failures=0, errors=0, skipped=0),
            'Original CUDA oracle must pass all tests without skips')
    require(sum(len(s.findall('testcase')) for s in suites) == expected, 'Missing individual oracle results')
    return totals


def gpu_snapshot():
    def query(*args):
        return subprocess.check_output(['nvidia-smi', *args], text=True, timeout=10).strip()
    apps = query('--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits')
    rows = query('--query-gpu=uuid,name,driver_version,utilization.gpu,memory.used', '--format=csv,noheader,nounits').splitlines()
    require(len(rows) == 1, 'This qualification requires the isolated single-GPU deployment')
    values = [v.strip() for v in rows[0].split(',')]
    require(not apps and int(values[-2]) <= 10 and int(values[-1]) <= 1024, 'GPU is busy; leave existing jobs untouched')
    return dict(uuid=values[0], name=values[1], driver=values[2], utilization=int(values[3]), memory_mib=int(values[4]))


def child_load(output, label):
    manifest = verify(output)
    owner = read(output / 'owner.json')
    require(read(output / 'status.json')['status'] == 'running' and
            os.environ.get('KVMEM_MEANK_QUAL_OWNER') == owner['token'] and os.getppid() == owner['pid'],
            'Child load requires its live qualification owner')
    build_root = Path(owner['build_root'])
    require(build_root.parent == output and os.environ.get('KVMEM_CUDA_BUILD_DIR') == str(build_root),
            'Child must use the dedicated build root')
    package = output / 'snapshots' / label / 'vllm_kvmem'
    import torch
    import vllm_kvmem.cuda_scoring as scoring
    require(Path(scoring.__file__).resolve() == package / 'cuda_scoring.py', 'Wrong snapshot imported')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'CUDA numerical qualification requires one visible device')
    props = torch.cuda.get_device_properties(0)
    module = scoring.kernel()
    require(hasattr(module, 'score'), 'Compiled kernel API is missing')
    evidence = build_evidence(build_root)
    require(all(evidence['inputs'][str(Path(evidence['stable_source']).parent / name)]['sha256'] == digest
                for name, digest in manifest['source_sha256'].items()), 'Stable inputs differ from frozen sources')
    require(str(package) not in (Path(evidence['build_directory']) / 'build.ninja').read_text(),
            'Ninja still embeds the frozen package path')
    save(output / ('load-' + label + '.json'), dict(pid=os.getpid(), package=str(package), build=evidence,
        torch_version=torch.__version__, torch_cuda=torch.version.cuda,
        device=dict(name=props.name, capability=[props.major, props.minor], total_memory=props.total_memory),
        environment={key: os.environ.get(key) for key in ENV_KEYS}))


def owned_process(argv, env, cwd, log):
    with log.open('w') as stream:
        process = subprocess.Popen(argv, env=env, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            require(process.wait() == 0, 'Qualification subprocess failed: ' + str(log))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def run(output):
    import psutil
    output = output.resolve()
    manifest = verify(output)
    require(read(output / 'status.json')['status'] == 'prepared', 'No automatic restart')
    require(psutil.virtual_memory().available >= 8 * 1024**3, 'Insufficient free CPU memory')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Qualification interrupted')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            idle = []
            for index in range(11):
                idle.append(gpu_snapshot())
                if index < 10:
                    time.sleep(1)
            require(verify(output) == manifest, 'Prepared inputs changed during admission')
            build_root = Path(tempfile.mkdtemp(prefix='build-', dir=output))
            owner = dict(pid=os.getpid(), create_time=psutil.Process().create_time(), token=uuid.uuid4().hex,
                         build_root=str(build_root), at=time.time())
            save(output / 'owner.json', owner)
            save(output / 'status.json', dict(status='running', at=time.time()))
            base = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'KVMEM_CUDA_BUILD_DIR': str(build_root),
                    'KVMEM_MEANK_QUAL_OWNER': owner['token'], 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
            for label in ('a', 'b'):
                env = {**base, 'PYTHONPATH': str(output / 'snapshots' / label)}
                owned_process([sys.executable, str(output / 'frozen' / ENTRY), '--output', str(output),
                               '--child-load', label], env, output, output / ('load-' + label + '.log'))
            a, b = read(output / 'load-a.json'), read(output / 'load-b.json')
            require(a['pid'] != b['pid'] and a['package'] != b['package'], 'Loads were not independent source processes')
            require(all(a[key] == b[key] for key in ('torch_version', 'torch_cuda', 'device', 'environment')),
                    'Build environment changed between processes')
            unchanged_build(a['build'], b['build'])
            env = {**base, 'PYTHONPATH': str(output / 'snapshots' / 'b')}
            owned_process([sys.executable, '-m', 'pytest', '-c', '/dev/null', '-q', '-p', 'no:cacheprovider',
                           str(output / 'oracle' / ORACLE), '--junitxml=' + str(output / 'oracle.xml')],
                          env, output / 'oracle', output / 'oracle.log')
            numerical = numerical_evidence(output / 'oracle.xml', manifest['expected_oracle_tests'])
            unchanged_build(b['build'], build_evidence(build_root))
            require(verify(output) == manifest, 'Frozen inputs changed during qualification')
            save(output / 'results.json', dict(status='passed', cache_reuse=True, numerical=numerical,
                first=a, second=b, idle_admission=idle, candidate_patch_sha256=manifest['candidate_patch_sha256'],
                scope=manifest['scope'], remaining=manifest['remaining']))
            save(output / 'status.json', dict(status='passed', at=time.time()))
    except BaseException as error:
        save(output / 'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    mode.add_argument('--child-load', choices=('a', 'b'), help=argparse.SUPPRESS)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    if args.child_load:
        child_load(args.output.resolve(), args.child_load)
    elif args.run_prepared:
        run(args.output)
    else:
        require(args.candidate is not None, 'Explicit candidate provenance required')
        prepare(args.output, args.candidate)


if __name__ == '__main__':
    main()

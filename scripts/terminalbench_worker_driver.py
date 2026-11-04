"""Run one explicitly handed-off remote cohort, without model ownership or retries."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

try:
    from .terminalbench_verifier_cache import preflight_cache_inputs
except ImportError:
    from terminalbench_verifier_cache import preflight_cache_inputs


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024**2):
            h.update(data)
    return h.hexdigest()


def verify_memory_admission(root):
    """Exercise the packaged launcher, not an unrelated workspace import."""
    manifest = read(root/'manifest.json')
    launcher = root/'runtime/scripts/run_terminalbench_job.py'
    spec = importlib.util.spec_from_file_location('_qualified_tb_launcher', launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'MemoryAdmission'):
        raise ValueError('Packaged launcher does not implement declared memory admission')
    events = []
    gate = module.MemoryAdmission(root, manifest, events.append)
    gate.refresh()
    gib = 1024**3
    expected = manifest['container_memory_admission_bytes']
    reserve = manifest['container_memory_host_reserve_bytes']
    if gate.budget != expected or module.HOST_MEMORY_RESERVE_BYTES != reserve:
        raise ValueError('Effective memory admission differs from the manifest')
    # This case caught the former hard-coded 24-GiB launcher despite a 48-GiB
    # manifest. Also prove both capacity guards remain in force.
    request = min(8*gib, expected)
    available = max(gate.total, reserve + request)
    if (not gate.admits(expected-request, request, available) or
            gate.admits(expected, 1, available) or
            gate.admits(0, request, reserve+request-1)):
        raise ValueError('Packaged launcher failed memory admission boundary checks')
    receipt = dict(at=time.time(), passed=True, launcher_sha256=sha(launcher),
                   configured_bytes=expected, effective_bytes=gate.budget,
                   host_reserve_bytes=reserve, history=events)
    save(root/'memory-admission-qualification.json', receipt)
    return receipt


def validate(root):
    proof, handoff = read(root/'worker-provenance.json'), read(root/'handoff.json')
    config, manifest = read(root/'formal-candidate.json'), read(root/'manifest.json')
    selected = sorted(Path(t['path']).name for t in config['tasks'])
    if (not handoff.get('original_executor_inactive') or
            not handoff.get('original_completed_evidence_preserved') or
            selected != sorted(handoff['remaining_tasks']) or
            selected != sorted(manifest['launched_task_names']) or len(set(selected)) != len(selected)):
        raise ValueError('Missing exact exclusive task ownership handoff')
    if config['n_attempts'] != 1 or config['retry']['max_retries'] != 0:
        raise ValueError('No automatic task retries')
    for rel, digest in proof['copied_files'].items():
        if sha(root/rel) != digest:
            raise ValueError('Prepared worker changed: '+rel)
    smoke = read(root/'smoke-remote-events/summary.json')
    if smoke.get('passed') != 1 or smoke.get('errors') != 0:
        raise ValueError('Remote end-to-end smoke did not pass')
    if (root/'jobs/formal-remote').exists():
        raise ValueError('Never repeat an attempted remote cohort')
    verify_memory_admission(root)
    preflight_cache_inputs(root, manifest)
    return selected


def main(root):
    selected = validate(root)
    owner = psutil.Process()
    with (root/'formal-launch.json').open('x') as stream:
        json.dump(dict(at=time.time(), pid=owner.pid, created=owner.create_time(),
                       tasks=selected, handoff_sha256=sha(root/'handoff.json')), stream, indent=2)
    env = {k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
    env.update(HARBOR_TELEMETRY='0', OPENAI_API_KEY='local-kvmem',
               LITELLM_LOCAL_MODEL_COST_MAP='True', PYTHONUNBUFFERED='1')
    args = [sys.executable, '-B', str(root/'runtime/scripts/run_terminalbench_job.py'),
            str(root), str(root/'formal-candidate.json')]
    with (root/'formal.log').open('xb') as log:
        child = subprocess.Popen(args, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log)
        save(root/'formal-process.json', dict(pid=child.pid,
             created=psutil.Process(child.pid).create_time(), argv=args, at=time.time()))
        save(root/'worker-state.json', dict(status='formal', at=time.time(), expected=len(selected)))
        code = child.wait()
    save(root/'worker-exit.json', dict(at=time.time(), returncode=code, retries=0))
    save(root/'worker-state.json', dict(status='completed' if code == 0 else 'failed',
                                      at=time.time(), returncode=code))


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    try:
        main(root)
    except BaseException as error:
        save(root/'worker-driver-error.json', dict(at=time.time(), error=repr(error)))
        raise

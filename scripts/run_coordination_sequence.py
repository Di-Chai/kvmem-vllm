"""Run a frozen GPU qualification, require restore evidence, then benchmark.

The child harness owns GPU-idle detection and stops only its own server. This
controller adds a durable, fail-closed correctness gate between the two jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(snapshot, prepared, output):
    manifest = json.loads((prepared/'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen qualification differs from CPU-prepared source: {name}')
    config = Path(manifest['config'])
    output.mkdir()
    shutil.copy2(prepared/'fixture.json', output/'fixture.json')
    # Own the input config too; later workspace edits cannot change queued arms.
    shutil.copy2(config, output/'input-profile.json')
    # Original model/tokenizer paths in these prepared profiles are absolute.
    profile = json.loads((output/'input-profile.json').read_text())
    for key in ('model', 'tokenizer', 'hf_config_path', 'draft_model'):
        if profile.get(key) and not Path(profile[key]).is_absolute():
            raise ValueError('Queued test profiles must use absolute local model paths')
    command = [sys.executable, str(snapshot/'scripts/qualify_coordination.py'),
               '--config', str(output/'input-profile.json'), '--output', str(output)]
    for key, value in manifest['workload'].items():
        if key == 'prepare_only':
            continue
        command.append('--' + key.replace('_', '-'))
        command.extend(str(v) for v in value) if isinstance(value, list) else command.append(str(value))
    return command


def main(args):
    import fcntl
    with open('/tmp/kvmem-coordination-sequence.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = args.output.resolve()
        root.mkdir(exist_ok=False, parents=True)
        snapshot = args.snapshot.resolve()
        correctness = prepare(snapshot, args.correctness_prepared.resolve(), root/'correctness')
        performance = prepare(snapshot, args.performance_prepared.resolve(), root/'performance')
        gate = root/'check_coordination_restore.py'
        shutil.copy2(Path(__file__).with_name(gate.name), gate)
        commands = [('correctness', correctness), ('restore_gate',
            [sys.executable, str(gate), '--input', str(root/'correctness'),
             '--output', str(root/'restore-gate.json')]), ('performance', performance)]
        save(root/'sequence.json', dict(snapshot=str(snapshot), commands=commands,
             policy='Wait for exclusive GPU idle; do not terminate foreign work; performance requires verified restores'))
        environment = {**os.environ, 'PYTHONPATH':str(snapshot/'src'), 'TOKENIZERS_PARALLELISM':'false'}
        stopped = False
        child = None

        def stop(signum, frame):
            nonlocal stopped
            stopped = True
            if child is not None and child.poll() is None:
                child.terminate()  # Harness handles its own server in finally.

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            for stage, command in commands:
                if stopped:
                    raise InterruptedError('Sequence cancelled')
                with (root/f'{stage}.log').open('w') as log:
                    child = subprocess.Popen(command, cwd=snapshot, env=environment,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    save(root/'status.json', dict(status='running', stage=stage, pid=child.pid, at=time.time()))
                    code = child.wait()
                if stopped:
                    raise InterruptedError('Sequence cancelled')
                if code:
                    raise RuntimeError(f'{stage} exited {code}; inspect {stage}.log')
                if stage == 'restore_gate' and json.loads((root/'restore-gate.json').read_text()).get('passed') is not True:
                    raise RuntimeError('Restore gate did not pass')
            save(root/'status.json', dict(status='completed', at=time.time()))
        except BaseException as error:
            save(root/'status.json', dict(status='interrupted' if stopped else 'failed',
                 stage=stage, error=str(error), at=time.time()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'correctness-prepared', 'performance-prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())

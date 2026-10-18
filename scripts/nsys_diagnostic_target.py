"""Identify an owned frozen diagnostic, await authorization, then exec it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import psutil


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify(root, output, nonce):
    if ROOT != root / 'frozen' or output.parent != root:
        raise ValueError('Target must come from this diagnostic frozen tree')
    claim = read(root / 'nsys-claim.json')
    if claim.get('nonce') != nonce or claim.get('capture_output') != str(output):
        raise ValueError('Target does not match the exclusive collector claim')
    if digest(root / 'manifest.json') != claim['manifest_sha256']:
        raise ValueError('Diagnostic manifest changed before target exec')
    manifest = read(root / 'manifest.json')
    for name in ('nsys_diagnostic_target.py', 'diagnose_restore_stages.py'):
        if digest(ROOT / 'scripts' / name) != manifest['source_sha256']['scripts/' + name]:
            raise ValueError('Frozen diagnostic entry point changed')
    if read(root / 'status.json').get('status') != 'prepared':
        raise ValueError('Target requires a fresh prepared diagnostic')
    return claim


def run(args):
    root, output = args.diagnostic_output.resolve(), args.capture_output.resolve()
    verify(root, output, args.nonce)
    process = psutil.Process()
    parents = []
    for parent in process.parents():
        try:
            parents.append(dict(pid=parent.pid, create_time=parent.create_time()))
        except psutil.NoSuchProcess:
            pass  # nsys may reparent its target; ancestry is evidence, not the ownership gate.
    identity = dict(pid=process.pid, create_time=process.create_time(), nonce=args.nonce,
        diagnostic_output=str(root), capture_output=str(output), parents=parents,
        helper=str(Path(__file__).resolve()), executable=sys.executable,
        manifest_sha256=digest(root / 'manifest.json'), at=time.monotonic(), wall=time.time())
    save(output / 'target.json', identity)
    # Cancellation before identity discovery cannot launch a late GPU job.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ack = read(output / 'allow-exec.json')
        if (ack.get('nonce'), ack.get('pid'), ack.get('create_time')) == (
                args.nonce, process.pid, identity['create_time']):
            verify(root, output, args.nonce)
            os.chdir(ROOT)
            entry = ROOT / 'scripts/diagnose_restore_stages.py'
            os.execv(sys.executable, [sys.executable, str(entry), '--run-prepared', '--output', str(root)])
        time.sleep(.05)
    raise TimeoutError('No matching wrapper authorization; diagnostic was not started')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-output', type=Path, required=True)
    parser.add_argument('--capture-output', type=Path, required=True)
    parser.add_argument('--nonce', required=True)
    run(parser.parse_args())

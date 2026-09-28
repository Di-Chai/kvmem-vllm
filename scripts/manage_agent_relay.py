"""Own a dedicated Docker bridge address and metrics relay; no GPU/model work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream-port', type=int, required=True)
    parser.add_argument('--image', required=True, help='Existing task image with Python3; no unrelated container is reused')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('agent_api_relay.py').resolve()
    snapshot = output / 'agent_api_relay.py'
    shutil.copy2(script, snapshot)
    name = 'kvmem-vllm-agent-' + uuid.uuid4().hex[:12]
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    network = container = None
    record = {'name': name, 'manager_pid': os.getpid(), 'upstream_port': args.upstream_port, 'state': 'starting'}
    record['relay_source_sha256'] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    state_path = output / 'resources.json'

    def save():
        temporary = output / 'resources.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(state_path)

    try:
        image = json.loads(docker('image', 'inspect', args.image))[0]
        record['image_id'] = image['Id']
        network = docker('network', 'create', '--label', 'kvmem-vllm.owner=' + name, name)
        record['network_id'] = network
        save()
        data = json.loads(docker('network', 'inspect', network))[0]
        address = data['IPAM']['Config'][0]['Gateway']
        container = docker('run', '--rm', '-d', '--pull', 'never', '--name', name,
            '--label', 'kvmem-vllm.owner=' + name, '--network', 'host', '--entrypoint', 'python3',
            '-v', f'{snapshot}:/opt/kvmem/agent_api_relay.py:ro',
            '-v', f'{output}:/records', image['Id'], '/opt/kvmem/agent_api_relay.py',
            '--listen-host', address, '--listen-port', '443', '--upstream-port', str(args.upstream_port),
            '--output', '/records/requests.jsonl')
        record.update(container_id=container, api_base=f'http://{address}:443/v1')
        save()
        # Proxies on the host are irrelevant to a private bridge health check.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 30
        while not stopped.is_set():
            try:
                with opener.open(record['api_base'] + '/models', timeout=2) as response:
                    record['models'] = json.load(response)
                break
            except OSError:
                running = json.loads(docker('inspect', container))[0]['State']['Running']
                if not running or time.monotonic() > deadline:
                    raise RuntimeError('Owned relay did not become ready')
                stopped.wait(.2)
        record['state'] = 'ready' if not stopped.is_set() else 'interrupted'
        save()
        print(json.dumps(record), flush=True)
        stopped.wait()
    except BaseException as error:
        record['failure'] = repr(error)
        raise
    finally:
        errors = []
        if container:
            try:
                docker('stop', '--time', '5', container)
            except subprocess.CalledProcessError as error:
                errors.append(str(error))
        if network:
            try:
                docker('network', 'rm', network)
            except subprocess.CalledProcessError as error:
                errors.append(str(error))
        record.update(state='cleanup_error' if errors else ('failed' if 'failure' in record else 'stopped'), cleanup_errors=errors)
        save()


if __name__ == '__main__':
    main()

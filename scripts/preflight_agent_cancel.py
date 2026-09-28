"""Disconnect one real non-streaming request and prove native archive/page release.

Separate synthetic active-prefill diagnostic, not agent quality/performance or a
compressed-refresh cancellation gate. Run on an otherwise idle backend.
"""
import argparse
import http.client
import json
from pathlib import Path
import socket
import time
import urllib.parse
import urllib.request
import uuid


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine-evidence', type=Path, required=True)
    parser.add_argument('--relay-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    engine, relay = read(args.engine_evidence), read(args.relay_evidence)
    if (engine.get('state') != 'ready' or relay.get('state') != 'ready'
            or engine.get('port') != relay.get('upstream_port')):
        raise ValueError('Cancellation preflight requires the same ready backend and relay')
    if args.output.exists():
        raise FileExistsError('Cancellation evidence must not be overwritten')
    capture = args.engine_evidence.parent / 'capture'
    if read(capture / 'scheduler_resources.json').get('requests') != 0:
        raise RuntimeError('Cancellation preflight requires an idle backend after tool preflight')
    request_id = 'kvmem-cancel-' + uuid.uuid4().hex
    prefix = 'chatcmpl-' + request_id
    payload = {'request_id': request_id, 'model': 'qualification', 'temperature': 0, 'max_tokens': 8192,
        'messages': [{'role': 'user', 'content': ('Repository notes: preserve immutable source keys and accepted history.\n' * 400)
                     + 'Explain the implementation and write a detailed test plan.'}]}
    endpoint = urllib.parse.urlparse(relay['api_base'])
    conn = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=180)
    report = {'scope': __doc__, 'status': 'failed', 'request': payload,
              'engine_manager_pid': engine['manager_pid'], 'relay_manager_pid': relay['manager_pid']}
    try:
        conn.request('POST', endpoint.path + '/chat/completions', json.dumps(payload).encode(),
                     {'Content-Type': 'application/json', 'x-kvmem-task-id': 'api-cancel-preflight'})
        deadline = time.monotonic() + 180
        while True:
            active = read(capture / 'progress.json').get('accepted_history', {})
            matching = {k: v for k, v in active.items() if k.startswith(prefix + '-')}
            if len(matching) == 1 and next(iter(matching.values())) >= 512:
                report['observed_active_capture'] = matching
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Request did not reach active accepted capture')
            time.sleep(.05)
        conn.sock.shutdown(socket.SHUT_RDWR)
        conn.close()
        report['disconnected_monotonic'] = time.monotonic()
        deadline = time.monotonic() + 60
        while True:
            archive = read(capture / 'archive.json')
            owned = read(capture / 'scheduler_resources.json')
            done = [r for r in archive.get('completed', []) if r['request_id'].startswith(prefix + '-')]
            if (len(done) == 1 and owned.get('requests') == 0 and owned.get('owned_blocks') == 0
                    and owned.get('reservations') == [] and archive.get('archive_used_slots') == 0
                    and archive.get('pending_downloads') == 0 and archive.get('retiring_requests') == 0):
                if (done[0]['terminal_prefix']['status'] != 'FINISHED_ABORTED'
                        or done[0]['history_end'] != done[0]['terminal_prefix']['history_end']):
                    raise RuntimeError('Disconnected native request did not retire at its exact aborted prefix')
                report.update(archive_completion=done[0], scheduler_resources=owned,
                              release_seconds=time.monotonic() - report['disconnected_monotonic'])
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Disconnected request retained native resources')
            time.sleep(.05)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(engine['api_base'].removesuffix('/v1') + '/health', timeout=5):
            pass
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        conn.close()
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'release_seconds': report['release_seconds']}))


if __name__ == '__main__':
    main()

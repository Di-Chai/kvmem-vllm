"""Native HTTP session isolation/cancellation gate, not an agent benchmark.

Populate independent checkpoints serially, restore them in concurrent requests,
then compare serial continuations against their original exact token streams.
Concurrent token differences are retained separately: native batching can change
floating-point reduction order, so this gate does not assert concurrent quality.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import socket
import threading
import time
import urllib.parse
import urllib.request
import uuid

from qualify_sessions import output_tokens, settle


def reuse_delta(before, after, count, positions):
    hits = after['scheduler']['sessions']['hits'] - before['scheduler']['sessions']['hits']
    tokens = after['archive']['session_restored_tokens'] - before['archive']['session_restored_tokens']
    if hits != count or tokens != sum(positions):
        raise RuntimeError(f'Incorrect session imports: hits={hits}, tokens={tokens}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    evidence = json.loads(args.engine_evidence.read_text())
    command = evidence['command']
    options = json.loads(command[command.index('--kv-transfer-config')+1])['kv_connector_extra_config']['kvmem']
    if (evidence['state'] != 'ready' or options.get('session_capacity', 0) < args.concurrency or
            not options.get('session_checkpoint_bytes') or not options.get('async_archive_retirement')):
        raise ValueError('Need a ready session backend with sufficient checkpoint entries')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    capture = args.engine_evidence.resolve().parent / 'capture'
    report = {'status': 'running', 'scope': __doc__, 'concurrency': args.concurrency,
              'engine': evidence, 'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'calls': {}, 'phases': {}}
    def save():
        (out / 'results.json').write_text(json.dumps(report, indent=2))
    def snapshot():
        state = settle(capture)
        if (state['archive']['session_checkpoints'] > options['session_capacity'] or
                state['scheduler']['sessions']['entries'] > options['session_capacity'] or
                state['archive']['session_checkpoint_payload_bytes'] > options['session_checkpoint_bytes']):
            raise RuntimeError('Session entry or payload quota exceeded')
        return state
    def read_archive():
        return json.loads((capture / 'archive.json').read_text())
    def complete(tokens, session, request_id=None, cache_salt=None, output=64):
        payload = {'model': 'qualification', 'prompt': tokens, 'session_id': session,
                   'temperature': 0, 'seed': 104729, 'max_tokens': output, 'return_token_ids': True}
        if request_id:
            payload['request_id'] = request_id
        if cache_salt is not None:
            payload['cache_salt'] = cache_salt
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        start = time.monotonic()
        req = urllib.request.Request(evidence['api_base']+'/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=600) as response:
            result = json.load(response)
        output_tokens(result)
        return {'result': result, 'wall_seconds': time.monotonic()-start}
    def batch(name, prompts, sessions, positions):
        before = snapshot()
        peak = 0
        peak_active = 0
        observations = []
        observation_errors = []
        stop = threading.Event()
        def observe():
            nonlocal peak, peak_active
            try:
                while not stop.is_set():
                    state = json.loads((capture/'scheduler_resources.json').read_text())
                    progress = json.loads((capture/'progress.json').read_text())
                    peak = max(peak, state['requests'])
                    active = len(progress['accepted_history'])
                    peak_active = max(peak_active, active)
                    observations.append({'at': time.monotonic(), 'requests': state['requests'],
                                         'active_authorities': active})
                    stop.wait(.01)
            except Exception as error:
                observation_errors.append(repr(error))
        monitor = threading.Thread(target=observe, daemon=True)
        monitor.start()
        try:
            with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                futures = [executor.submit(complete, p, s) for p, s in zip(prompts, sessions)]
                results = [f.result() for f in futures]
        finally:
            stop.set(); monitor.join()
            (out/(name+'_observations.json')).write_text(json.dumps(observations))
        after = snapshot()
        if observation_errors:
            raise RuntimeError(f'Native concurrency observation failed: {observation_errors}')
        reuse_delta(before, after, len(prompts), positions)
        report['phases'][name] = {'before': before, 'after': after, 'peak_native_requests': peak,
                                  'peak_active_authorities': peak_active}
        report['calls'][name] = results
        save()
        return results, peak_active

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(command[command.index('--tokenizer')+1], local_files_only=True)
        sessions = ['session-concurrency-'+uuid.uuid4().hex for _ in range(args.concurrency)]
        prompts = []
        for i in range(args.concurrency):
            history = tokenizer.encode((f'Repository {i}: preserve utility behavior and test every change.\n')*600)
            question = tokenizer.encode(f'\nWrite a Python function to compute the sum of multiples of {i+2}.\n')
            prompts.append(history[:4096-len(question)]+question)
        (out/'inputs.json').write_text(json.dumps({'sessions': sessions, 'prompts': prompts}))
        controls, positions = [], []
        for i, (prompt, session) in enumerate(zip(prompts, sessions)):
            before = snapshot()
            response = complete(prompt, session)
            after = snapshot()
            new = after['archive']['completed'][len(before['archive']['completed']):]
            if len(new) != 1 or new[0].get('retirement_strategy') != 'checkpoint_rewind':
                raise RuntimeError('Serial population failed to retain its exact checkpoint')
            positions.append(new[0]['checkpoint_tokens'])
            controls.append(response)
            report['calls'][f'populate_{i}'] = response
            save()
        warm, peak = batch('concurrent_restore', prompts, sessions, positions)
        if peak < args.concurrency:
            raise RuntimeError('No actual native concurrency observed')
        report['concurrent_exact_tokens'] = [output_tokens(x['result']) == output_tokens(y['result'])
                                             for x, y in zip(controls, warm)]
        for i, (prompt, session) in enumerate(zip(prompts, sessions)):
            before = snapshot()
            result = complete(prompt, session)
            after = snapshot()
            reuse_delta(before, after, 1, [positions[i]])
            report['calls'][f'isolated_restore_{i}'] = result
            save()
            if output_tokens(result['result']) != output_tokens(controls[i]['result']):
                raise RuntimeError(f'Isolated session {i} changed tokens after concurrent page reuse')
        same, peak = batch('same_session_overlap', [prompts[0]]*2, [sessions[0]]*2, [positions[0]]*2)
        overlap = report['phases']['same_session_overlap']
        waits = (overlap['after']['scheduler']['sessions']['waits'] -
                 overlap['before']['scheduler']['sessions']['waits'])
        if peak != 1 or not waits or any(output_tokens(x['result']) != output_tokens(controls[0]['result']) for x in same):
            raise RuntimeError('Same-session requests were not serialized with exact continuation')

        # Cancel a restored decode after its first SSE event. Check the native
        # terminal cause, not just a successful socket close, then resume again.
        before = snapshot()
        request_id = 'session-cancel-'+uuid.uuid4().hex
        endpoint = urllib.parse.urlparse(evidence['api_base'])
        connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=180)
        payload = {'model': 'qualification', 'prompt': prompts[0], 'session_id': sessions[0],
                   'request_id': request_id, 'temperature': 0, 'seed': 104729,
                   'max_tokens': 8192, 'stream': True}
        try:
            connection.request('POST', endpoint.path+'/completions', json.dumps(payload),
                               {'Content-Type': 'application/json'})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(response.read().decode())
            while True:
                line = response.readline()
                if not line:
                    raise RuntimeError('Stream ended before a cancellation event')
                if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                    report['cancel_first_event'] = json.loads(line[6:])
                    break
            connection.sock.shutdown(socket.SHUT_RDWR)
        finally:
            connection.close()
        after = snapshot()
        reuse_delta(before, after, 1, [positions[0]])
        terminal = [r for r in read_archive()['completed'] if r['request_id'].startswith('cmpl-'+request_id+'-')]
        if len(terminal) != 1 or terminal[0]['terminal_prefix']['status'] != 'FINISHED_ABORTED':
            raise RuntimeError('Restored request did not end through native cancellation')
        report['cancellation'] = {'before': before, 'after': after, 'terminal': terminal[0]}
        result = complete(prompts[0], sessions[0])
        recovered = snapshot()
        reuse_delta(after, recovered, 1, [positions[0]])
        report['calls']['after_cancel'] = result
        if output_tokens(result['result']) != output_tokens(controls[0]['result']):
            raise RuntimeError('Cancellation changed the next exact continuation')
        # The same visible session ID in another gateway isolation scope must
        # cold-compute. At a full pool this also exercises fenced idle eviction.
        other_scope = 'different-scope-'+uuid.uuid4().hex
        result = complete(prompts[0], sessions[0], cache_salt=other_scope)
        isolated_scope = snapshot()
        reuse_delta(recovered, isolated_scope, 0, [])
        if output_tokens(result['result']) != output_tokens(controls[0]['result']):
            raise RuntimeError('Cold isolation-scope continuation differs')
        report['calls']['different_cache_salt'] = result
        report['phases']['different_cache_salt'] = {'before': recovered, 'after': isolated_scope}
        recovered = isolated_scope
        if options['session_capacity'] == args.concurrency:
            # Serial verification order, followed by the session-0 overlap and
            # cancellation phases, makes session 1 the least recently retired.
            result = complete(prompts[1], sessions[1])
            after_eviction = snapshot()
            reuse_delta(recovered, after_eviction, 0, [])
            if output_tokens(result['result']) != output_tokens(controls[1]['result']):
                raise RuntimeError('Evicted session cold recomputation differs')
            report['calls']['evicted_session'] = result
            report['phases']['evicted_session'] = {'before': recovered, 'after': after_eviction}
            recovered = after_eviction
        # A shorter prompt explicitly misses its old checkpoint and is too
        # short to create another. Exercise reclamation through normal HTTP.
        for i, session in enumerate(sessions):
            report['calls'][f'release_{i}'] = complete(prompts[i][:512], session, output=8)
            snapshot()
        report['calls']['release_other_scope'] = complete(prompts[0][:512], sessions[0],
                                                          cache_salt=other_scope, output=8)
        recovered = snapshot()
        if options['session_capacity'] == args.concurrency and any(recovered['archive'][key] for key in
                ('session_checkpoints', 'session_checkpoint_payload_bytes', 'archive_used_slots')):
            raise RuntimeError('Owned prompt sessions did not fully reclaim their checkpoints and archives')
        report['status'] = 'passed'
        report['final_state'] = recovered
    except BaseException as error:
        report['status'], report['failure'] = 'failed', repr(error)
        raise
    finally:
        save()
    print(json.dumps({'status': report['status'], 'concurrent_exact_tokens': report['concurrent_exact_tokens']}))


if __name__ == '__main__':
    main()

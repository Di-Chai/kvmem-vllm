"""Actual HTTP prompt-state continuation gate on an explicitly session-enabled backend.

Compares unchanged greedy requests before/after native pages are recycled, and
checks a changed prefix takes a cold path. This is a correctness gate with observed
timings, not a substitute for concurrent agent quality or matched throughput runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid


def settle(capture, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            archive = json.loads((capture / 'archive.json').read_text())
            scheduler = json.loads((capture / 'scheduler_resources.json').read_text())
            if (not archive['active_requests'] and not archive['retiring_requests'] and
                    not archive.get('session_retiring_requests', 0) and not archive['pending_downloads'] and
                    not scheduler['requests'] and not scheduler['owned_blocks'] and
                    not scheduler['reservations'] and not scheduler['refresh_pending'] and
                    scheduler['free_blocks'] == scheduler['total_blocks'] - 1):
                return {'archive': archive, 'scheduler': scheduler}
        except (OSError, ValueError):
            pass
        time.sleep(.05)
    raise TimeoutError('Native request pages or authority retirement did not settle')


def output_tokens(response):
    choices = response.get('choices', [])
    if len(choices) != 1 or not isinstance(choices[0].get('token_ids'), list):
        raise RuntimeError('Missing exact native completion token IDs')
    tokens = choices[0]['token_ids']
    if (not tokens or any(type(t) is not int or t < 0 for t in tokens) or
            len(tokens) != response['usage']['completion_tokens']):
        raise RuntimeError('Completion IDs disagree with native usage')
    return tokens


def validate_reuse(control, cold, warm, before, after):
    expected = output_tokens(control)
    if output_tokens(cold) != expected or output_tokens(warm) != expected:
        raise RuntimeError('Greedy token IDs changed after prompt state reuse')
    imported = after['archive']['session_restored_tokens'] - before['archive']['session_restored_tokens']
    prompt = warm['usage']['prompt_tokens']
    if not 0 < imported < prompt or imported % 128:
        raise RuntimeError('No sealed partial prompt import was observed')
    hits = after['scheduler']['sessions']['hits'] - before['scheduler']['sessions']['hits']
    if hits != 1:
        raise RuntimeError('Expected exactly one acknowledged native session hit')
    return imported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Frozen JSON input_ids, preserved in full')
    parser.add_argument('--output-tokens', type=int, default=64)
    parser.add_argument('--model', default='qualification', help='Native served model name')
    parser.add_argument('--api-key-env', help='Environment variable containing a tenant API key')
    parser.add_argument('--reuse-only', action='store_true',
                        help='Compare one cold populate against restore; omit separate cold and changed-prefix controls')
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise ValueError('The requested API key environment variable is empty')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = json.loads(args.engine_evidence.read_text())
    command = evidence['command']
    options = json.loads(command[command.index('--kv-transfer-config')+1])['kv_connector_extra_config']['kvmem']
    draft = json.loads(command[command.index('--speculative-config')+1])
    if (evidence['state'] != 'ready' or options.get('session_capacity', 0) < 2 or
            not options.get('session_checkpoint_bytes') or not options.get('async_archive_retirement') or
            command[command.index('--kv-cache-dtype')+1] != 'fp8_e4m3' or
            'Q8_0' not in command[command.index('serve')+1] or
            draft.get('method') != 'dflash' or draft.get('num_speculative_tokens') != 7 or
            draft.get('kv_cache_dtype') != 'bfloat16' or 'DFlash2' not in draft.get('model', '') or
            options.get('select_budget', 131072) != 131072 or options.get('gen_budget', 65536) != 65536 or
            options.get('block_tokens', 128) != 128):
        raise ValueError('Session gate requires the explicit mainline Q8/FP8 K128K/G64K backend')
    capture = args.engine_evidence.resolve().parent / 'capture'
    (out / 'engine_evidence.json').write_text(json.dumps(evidence, indent=2))
    if args.input:
        content = args.input.read_bytes()
        prompt = json.loads(content)['input_ids']
        input_info = {'path': str(args.input.resolve()), 'sha256': hashlib.sha256(content).hexdigest()}
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(command[command.index('--tokenizer')+1], local_files_only=True)
        history = tokenizer.encode('Repository note: utility functions must preserve existing behavior and pass their tests.\n' * 512)
        question = tokenizer.encode('\nWrite a Python function to merge overlapping intervals.\n')
        prompt = history[:4096-len(question)] + question
        input_info = {'scope': 'synthetic 4096-token functional checkpoint fixture'}
    if len(prompt) < 2048 or any(type(t) is not int or t < 0 for t in prompt):
        raise ValueError('Session gate needs a multi-chunk token-ID prompt')
    (out / 'input.json').write_text(json.dumps({'input_ids': prompt}))
    report = {'status': 'running', 'scope': 'HTTP exact prompt-state continuation; not agent/performance acceptance',
              'input': input_info, 'prompt_tokens': len(prompt), 'calls': {},
              'independent_cold_control': not args.reuse_only,
              'changed_prefix_check': not args.reuse_only}
    def save():
        (out / 'results.json').write_text(json.dumps(report, indent=2))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def complete(name, tokens, session=None, output=None):
        payload = {'model': args.model, 'prompt': tokens, 'temperature': 0, 'seed': 104729,
                   'max_tokens': args.output_tokens if output is None else output, 'return_token_ids': True}
        if session:
            payload['session_id'] = session
        started = time.monotonic()
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = 'Bearer ' + api_key
        request = urllib.request.Request(evidence['api_base']+'/completions', data=json.dumps(payload).encode(),
                                         headers=headers)
        with opener.open(request, timeout=3600) as response:
            result = json.load(response)
        elapsed = time.monotonic() - started
        state = settle(capture)
        report['calls'][name] = {'result': result, 'wall_seconds': elapsed, 'state': state}
        save()
        print(json.dumps({'phase': name, 'wall_seconds': elapsed, 'usage': result['usage']}), flush=True)
        return result, state
    session = 'checkpoint-gate-' + uuid.uuid4().hex
    try:
        save()
        control = None
        if not args.reuse_only:
            control, _ = complete('cold_control', prompt)
        cold, populated = complete('session_populate', prompt, session)
        if not populated['archive']['session_checkpoints']:
            raise RuntimeError('Worker did not retain a prompt checkpoint')
        # Give native allocator/model scratch a different request before import.
        complete('page_reuse_neighbour', prompt[-2048:], output=16)
        warm, restored = complete('session_reuse', prompt, session)
        report['imported_tokens'] = validate_reuse(
            cold if control is None else control, cold, warm, populated, restored)
        if args.reuse_only:
            report['status'] = 'passed'
            return
        changed = list(prompt)
        changed[0] = prompt[1] if prompt[1] != prompt[0] else prompt[-1]
        if changed == prompt:
            raise ValueError('Prefix-miss fixture did not change an input token')
        changed_control, _ = complete('changed_control', changed)
        changed_session, changed_state = complete('changed_session', changed, session)
        if output_tokens(changed_control) != output_tokens(changed_session):
            raise RuntimeError('Prefix-miss recomputation changed greedy output')
        if changed_state['scheduler']['sessions']['hits'] != restored['scheduler']['sessions']['hits']:
            raise RuntimeError('Changed token prefix was incorrectly reused')
        report['status'] = 'passed'
    except BaseException as error:
        report['status'], report['failure'] = 'failed', repr(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()

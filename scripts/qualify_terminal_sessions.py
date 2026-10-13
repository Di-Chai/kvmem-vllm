"""Check accepted terminal KV preservation and an independent greedy cold control."""
import argparse
import json
from pathlib import Path
import time

import httpx
from qualify_multi_user import load_users, validate_api_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-base', required=True, help='Authenticated turn gateway origin')
    parser.add_argument('--backend-base', required=True, help='Owned loopback native engine origin')
    parser.add_argument('--users-file', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='kvmem-meank')
    parser.add_argument('--session', default='terminal-quality')
    parser.add_argument('--first-output-tokens', type=int, default=257)
    parser.add_argument('--prompt-tokens', type=int, default=4200)
    parser.add_argument('--append-tokens', type=int, default=32)
    parser.add_argument('--capture', type=Path, help='Owned engine diagnostic directory')
    parser.add_argument('--skip-cold-control', action='store_true',
        help='Qualify restoration bytes only; requires the real engine restoration oracle')
    args = parser.parse_args()
    gateway, backend = validate_api_base(args.api_base), validate_api_base(args.backend_base)
    user = load_users(args.users_file)[0]
    source = json.loads(args.input.read_text())['input_ids']
    if not 50 <= args.prompt_tokens <= len(source) or not 0 < args.append_tokens <= len(source):
        raise ValueError('Frozen prompt or append extent is invalid')
    if args.skip_cold_control and args.capture is None:
        raise ValueError('Byte-only qualification requires --capture with the restoration oracle enabled')
    prompt = (source[:4150] + source[-50:] if args.prompt_tokens == 4200
              else source[:args.prompt_tokens])
    # Exact tail tokens provide a reproducible second input without re-tokenizing.
    append = source[-args.append_tokens:]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'calls': {}, 'status': 'running', 'prompt_tokens': len(prompt)}
    def save():
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    with httpx.Client(timeout=1800, trust_env=False) as client:
        def turn(name, version, extra, output):
            start = time.monotonic()
            response = client.post(gateway + '/v1/kvmem/sessions/' + args.session + '/turn',
                json={'version': version, 'append_token_ids': extra, 'max_tokens': output,
                      'ignore_eos': True, 'waiting_for_tool': True},
                headers={'Authorization': 'Bearer ' + user['api_key']})
            report['calls'][name] = {'status': response.status_code, 'seconds': time.monotonic() - start}
            save()
            if response.is_error:
                (args.output / (name + '-error.json')).write_text(response.text)
            response.raise_for_status()
            value = response.json()
            (args.output / (name + '.json')).write_text(json.dumps(value))
            return value
        try:
            first = turn('first', 0, prompt, args.first_output_tokens)
            second = turn('resume', 1, append, 128)
            if args.capture is not None:
                evidence = args.capture / 'verified_restores.jsonl'
                notices = [json.loads(line) for line in evidence.read_text().splitlines()]
                request_id = second['native_response']['id'] + '-0'
                matches = [n for n in notices if n['request_id'].startswith(request_id)]
                if len(matches) != 1 or not matches[0]['byte_equal']:
                    raise AssertionError('Missing unique real-model restoration byte oracle')
                report['restored_state'] = matches[0]
                report['restore_integrity_status'] = 'passed'
            report['expected_first_checkpoint_tokens'] = len(prompt) + args.first_output_tokens - 1
            if args.skip_cold_control:
                report.update(status='passed', scope='terminal_checkpoint_gpu_byte_integrity',
                              cold_control='not_run', final_history_tokens=second['history_tokens'])
                save()
                print(json.dumps(report, indent=2))
                return
            full = prompt + first['native_response']['choices'][0]['token_ids'] + append
            start = time.monotonic()
            response = client.post(backend + '/v1/completions', json={'model': args.model,
                'prompt': full, 'max_tokens': 128, 'ignore_eos': True, 'temperature': 0,
                'seed': 104729, 'return_token_ids': True})
            response.raise_for_status()
            cold = response.json(); (args.output / 'cold.json').write_text(json.dumps(cold))
            report['calls']['cold'] = {'status': 200, 'seconds': time.monotonic() - start}
            report['greedy_tokens_equal'] = second['native_response']['choices'][0]['token_ids'] == cold['choices'][0]['token_ids']
            report['status'] = 'passed' if report['greedy_tokens_equal'] else 'failed'
        except Exception as error:
            report.update(status='failed', error_type=type(error).__name__)
            save()
            raise
        save()
    print(json.dumps(report, indent=2))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

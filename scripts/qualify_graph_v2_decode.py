"""Qualify a final frozen Graph V2 profile in a shared C8 decode window.

This fixed-output diagnostic is separate from original answer-quality tests.
It reuses exact streamed-ID accounting and strict native confirmation; it does
not infer accepted output from speculative proposals or sum per-user rates.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys
import time


SCHEMA = 'kvmem-graph-v2-fixed-decode-1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_fixture(fixture, profile):
    if fixture.get('schema') != SCHEMA or fixture.get('concurrency') != 8:
        raise ValueError('Require the declared eight-request diagnostic fixture')
    kvmem = profile['kvmem']
    if fixture.get('output_tokens') != kvmem['gen_budget'] or fixture['output_tokens'] != 16384:
        raise ValueError('Diagnostic output must retain the original 16K generation budget')
    count = fixture['input_tokens']
    query = fixture['query_span']
    if type(count) is not int or count <= kvmem['select_budget'] + kvmem['gen_budget']:
        raise ValueError('Require a complete long input that actually exercises compression')
    if count + fixture['output_tokens'] > profile['max_model_len']:
        raise ValueError('Complete input and generation exceed the logical context limit')
    if (not isinstance(query, list) or len(query) != 2 or
            any(type(value) is not int for value in query) or not 0 <= query[0] < query[1] <= count):
        raise ValueError('Invalid explicit original Query coordinates')
    if fixture.get('min_window_seconds') != 30 or fixture.get('min_native_coverage') != .9:
        raise ValueError('Do not weaken the shared-window qualification gate')
    path = Path(fixture['input_i32'])
    if path.stat().st_size != 4 * count or digest(path) != fixture['input_i32_sha256']:
        raise ValueError('Frozen original input changed')


def validate_identities(sessions, responses):
    if (len(sessions) != 8 or len(set(sessions)) != 8 or len(responses) != 8 or
            {row['client_index'] for row in responses} != set(range(8))):
        raise ValueError('Diagnostic request identities differ')
    identifiers = [row.get('response_id') for row in responses]
    if any(not isinstance(value,str) or not value for value in identifiers) or len(set(identifiers)) != 8:
        raise ValueError('Missing or duplicate native response identities')


def absolute_window(batch):
    origins = [sample['absolute_monotonic_at']-sample['at'] for sample in batch['native_samples']
        if 'absolute_monotonic_at' in sample]
    if not origins or any(not math.isfinite(value) for value in origins) or max(origins)-min(origins) > 1e-6:
        raise ValueError('Cannot bind relative stream window to native monotonic time')
    window = batch['steady_decode_window']
    return origins[0]+window['start'], origins[0]+window['end']


def retire_completed_streams(code, report, sessions, responses, input_tokens, retire, release):
    """Drain completed HTTP requests even when the strict snapshot gate failed.

    The original gate still fails after retirement. Otherwise the controller can
    stop the engine while its last asynchronous retirement is still publishing.
    """
    if len(report['batches']) != 1:
        raise RuntimeError('Exact shared C8 decode/native qualification failed;retain first attempt')
    validate_identities(sessions, responses)
    try:
        terminals = [retire(row['response_id']) for row in responses]
    finally:
        released = release()
    if any(row['history_end'] != input_tokens+16384-1 for row in terminals):
        raise RuntimeError('Final complete CPU accepted history differs')
    if any(row.get('checkpoint_tokens') for row in terminals):
        raise RuntimeError('Independent diagnostic retained a checkpoint')
    if code or report['status'] != 'passed':
        raise RuntimeError('Exact shared C8 decode/native qualification failed;retain first attempt')
    return report['batches'][0], released


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'capture', 'output', 'fixture'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=('D',), required=True)
    args = parser.parse_args(argv)
    profile = json.loads(args.config.read_text())
    fixture = json.loads(args.fixture.read_text())
    validate_fixture(fixture, profile)
    import graph_v2_protocol as protocol
    import qualify_block_graph as lifecycle
    import qualify_graph_niah as retired
    import qualify_graph_v2 as launch
    import qualify_decode as decode
    protocol.validate_profile(profile, args.arm, 'niah')
    args.output.mkdir(parents=True, exist_ok=False)
    lifecycle.save(args.output/'launch-audit.json', launch.audit_launch(profile, args.capture, args.arm))
    tokens = array.array('i')
    tokens.frombytes(Path(fixture['input_i32']).read_bytes())
    if sys.byteorder != 'little':
        tokens.byteswap()
    if tokens.itemsize != 4 or len(tokens) != fixture['input_tokens'] or min(tokens) < 0:
        raise ValueError('Invalid original int32 token IDs')
    prompt = args.output/'input.json'
    lifecycle.save(prompt, dict(input_ids=tokens.tolist()))
    labels = args.output/'local-client-labels.json'
    lifecycle.save(labels, dict(users=[dict(id=f'owner-{index}',api_key=f'local-decode-label-{index}')
        for index in range(8)]))
    # Native serving has no API-key policy in the reference profile. These are
    # client labels,not evidence of eight authenticated accounts.
    sessions, responses = [], []
    original_payload, original_read, original_public = decode.make_payload, decode.read_sample, decode.public_evidence

    def payload(options, content):
        result = original_payload(options, content)
        sessions.append(result['session_id'])
        result.update(cache_salt=result['session_id'], kvmem_reselect='force', thinking_token_budget=4096)
        return result

    def read(capture):
        result = original_read(capture)
        result['absolute_monotonic_at'] = result['at']
        return result

    def public(batch):
        responses.extend(dict(client_index=row['client_index'], response_id=row.get('_response_id'))
            for row in batch['requests'])
        return original_public(batch)

    decode.make_payload, decode.read_sample, decode.public_evidence = payload, read, public
    result = None
    began = time.monotonic()
    try:
        code = decode.main(['--api-base',f"http://127.0.0.1:{profile['port']}/v1",'--model',profile['served_model_name'],
            '--users-file',str(labels),'--output',str(args.output/'decode'),'--capture',str(args.capture),
            '--endpoint','completions','--prompt-token-file',str(prompt),'--prompt-sessions',
            '--query-span',*[str(value) for value in fixture['query_span']],
            '--concurrency','8','--rounds','1','--output-tokens','16384','--trim-tokens','64',
            '--min-window-seconds','30','--min-native-coverage','.9','--timeout','1800'])
        ended = time.monotonic()
        report = json.loads((args.output/'decode/results.json').read_text())
        batch, released = retire_completed_streams(code, report, sessions, responses, len(tokens),
            lambda response_id: retired.wait_retired(args.capture, response_id),
            lambda: lifecycle.wait_released(args.capture, sessions, timeout=180))
        window = batch['steady_decode_window']
        start, end = absolute_window(batch)
        steps = [row for row in lifecycle.records(args.capture/'model_step_latency.jsonl')
            if start <= row['started_at'] and row['submitted_at'] <= end]
        if not steps or any(row['prefill_tokens'] or row['prefill_requests'] or row['decode_requests'] != 8 for row in steps):
            raise RuntimeError('Native completed rows do not independently corroborate C8 decode')
        result = dict(status='passed', cases=8, arm=args.arm, schema=SCHEMA,
            all_owned_sessions_released=True, input_tokens=len(tokens), output_tokens_each=16384,
            source_fixture_sha256=digest(args.fixture), actual_concurrency=protocol.audit_actual_concurrency(steps),
            shared_decode_window=window, native_confirmation=batch['native_confirmation'],
            completed_native_submission_rows_inside_window=len(steps), measurement_started_at=began,
            measurement_finished_at=ended, phase_summary=lifecycle.summarize_steps(args.capture,began,ended),
            requests=responses, sessions=sessions, final_archive=released['archive'],
            scope='Fixed-output synthetic diagnostic;temperature0,ignore_eos,original full256K input and unchanged frozen D profile. '
                'Accepted SSE IDs/usage supply shared-wall throughput;native commit snapshots and completed rows corroborate owners/batch. '
                'CPU submission bounds are not exact GPU kernel boundaries. Not agent/answer quality or authenticated-user acceptance. '
                'Prefill GPU-event rate includes replay/model pipeline work;not unique input throughput.')
        lifecycle.save(args.output/'results.json',result)
    except BaseException as error:
        lifecycle.save(args.output/'client-failure.json',dict(error_type=type(error).__name__,error=str(error),at=time.time()))
        raise
    finally:
        decode.make_payload, decode.read_sample, decode.public_evidence = original_payload, original_read, original_public
    print(json.dumps({key:value for key,value in result.items() if key not in
        ('native_confirmation','final_archive','requests','sessions','shared_decode_window')},ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Decode statistics and synthetic streaming transport; no model/GPU load."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest


scripts = Path(__file__).parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('decode_gate', scripts / 'qualify_decode.py')
gate = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(scripts))
try:
    spec.loader.exec_module(gate)
finally:
    sys.path.remove(str(scripts))


def empty_trace():
    return {'client_index': 0, 'outcome': 'failure', 'started_at': 0, 'timeline': [],
            'observed_tokens': 0, 'saw_token_ids': False, '_token_digest': hashlib.sha256()}


def trace(index, points, *, response_id=None):
    return {'client_index': index, 'outcome': 'success',
            '_response_id': response_id or f'cmpl-{index}',
            'observed_tokens': points[-1][1],
            'timeline': [{'at': at, 'tokens': tokens,
                          'delta_tokens': tokens - (points[i - 1][1] if i else 0)}
                         for i, (at, tokens) in enumerate(points)]}


def test_role_and_prompt_tokens_are_not_generated_tokens_and_tpot_excludes_first_burst():
    value = empty_trace()
    gate.consume_event(value, {'id': 'cmpl-a', 'prompt_token_ids': [10] * 100,
                              'choices': [{'delta': {'role': 'assistant'}}]}, .5)
    assert not value['timeline']
    gate.consume_event(value, {'id': 'cmpl-a', 'choices': [{'token_ids': [1, 2, 3, 4]}]}, 1)
    gate.consume_event(value, {'id': 'cmpl-a', 'choices': [{'token_ids': [5, 6, 7, 8],
                                                         'finish_reason': 'length'}]}, 3)
    gate.consume_event(value, {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 8}}, 4)
    gate.finish_trace(value, 8)
    assert value['observed_tokens'] == 8 and len(value['timeline']) == 2
    assert value['ttft_seconds'] == 1
    assert value['decode_seconds'] == 2
    assert value['tokens_after_first_burst'] == 4
    assert value['tpot_seconds'] == .5
    assert value['timeline'][0]['token_ids'] == [1, 2, 3, 4]


@pytest.mark.parametrize('change,error', [
    ({'saw_token_ids': False}, 'missing_streamed_token_ids'),
    ({'completion_tokens': None}, 'missing_final_usage'),
    ({'completion_tokens': 7}, 'token_ids_usage_mismatch'),
    ({'finish_reason': 'stop'}, 'unexpected_finish_reason'),
])
def test_missing_or_inconsistent_token_evidence_fails(change, error):
    value = empty_trace()
    gate.consume_event(value, {'choices': [{'token_ids': list(range(8)), 'finish_reason': 'length'}],
                              'usage': {'prompt_tokens': 20, 'completion_tokens': 8}}, 1)
    value.update(change)
    with pytest.raises(gate.DecodeError, match=error):
        gate.finish_trace(value, 8)


def test_shared_trimmed_window_counts_tokens_not_events_and_retains_stalls():
    # Request B prefills two seconds later; no A prefill/early decode enters the
    # shared window. Both then stall from t=3 to t=8; those five seconds count.
    a = trace(0, [(0, 4), (1, 8), (3, 16), (8, 24), (9, 32), (10, 40)])
    b = trace(1, [(2, 4), (3, 8), (8, 16), (9, 24), (10, 32), (11, 40)])
    window = gate.decode_window([a, b], 8, 1, 2)
    assert window['status'] == 'passed'
    assert (window['start'], window['end'], window['seconds']) == (3, 9, 6)
    assert window['output_tokens'] == 32
    assert window['output_tokens_per_second'] == pytest.approx(32 / 6)
    assert [item['tokens'] for item in window['per_request']] == [16, 16]
    assert [item['token_events'] for item in window['per_request']] == [2, 2]


def test_tail_crossing_burst_cannot_include_trimmed_tokens():
    value = trace(0, [(0, 7), (1, 14), (2, 21), (3, 28), (4, 35), (5, 40)])
    window = gate.decode_window([value], 8, .5, 2)
    assert window['status'] == 'passed'
    assert window['start'] == 1  # First whole burst reaching >=8, all 14 omitted.
    assert window['end'] == 3  # 35 would cross the final 8-token exclusion.
    assert window['output_tokens'] == 14
    assert window['per_request'][0]['end_tokens'] <= 40 - 8


def test_full_window_excludes_terminal_burst_and_invalid_overlap_does_not_pass():
    a = trace(0, [(1, 3), (2, 7), (3, 11), (4, 15)])
    b = trace(1, [(2, 3), (3, 7), (4, 11), (5, 15)])
    window = gate.decode_window([a, b], 0, .5, 1)
    assert window['status'] == 'passed'
    assert window['start'] == 2 and window['end'] == 4
    assert window['output_tokens'] == 8  # Neither request's t=4 burst counted.
    assert gate.decode_window([a, b], 0, 3, 1)['error'] == 'window_too_short'
    assert gate.decode_window([a, b], 0, .5, 2)['error'] == 'not_all_requests_have_enough_progress'
    c = trace(2, [(10, 3), (11, 7)])
    assert gate.decode_window([a, c], 0, .1, 1)['error'] == 'no_shared_decode_window'


def native_sample(at, a, b, *, prefill=100):
    return {'at': at, 'prefill': prefill, 'decode': at * 2, 'prefill_raced': False,
            'progress': {'accepted_history': {'cmpl-0-0': a, 'cmpl-1-0': b}, 'retiring_requests': 0},
            'scheduler': {'requests': 2, 'refresh_pending': 0, 'connector_pending': 0}}


def test_native_confirmation_excludes_prefill_and_terminal_and_keeps_stalls():
    samples = [native_sample(0, 80, 80, prefill=80), native_sample(1, 100, 100),
               native_sample(2, 108, 108), native_sample(3, 116, 116),
               native_sample(4, 116, 116), native_sample(5, 124, 124)]
    traces = [trace(index, [(1, 8), (5, 40)]) for index in range(2)]
    window = {'status': 'passed', 'start': 2, 'end': 4}
    result = gate.native_confirmation(samples, traces, window, 2, 2)
    assert result['status'] == 'passed'
    assert result['interval_count'] == 2 and result['seconds'] == 2
    assert result['accepted_input_tokens'] == 16
    assert result['accepted_input_tokens_per_second'] == 8
    # Matching owner count from unrelated requests is insufficient.
    traces[0]['_response_id'] = 'cmpl-foreign'
    assert gate.native_confirmation(samples, traces, window, 2, 1)['status'] == 'failed'


def native_fixture():
    samples = [native_sample(at, 100 + at * 8, 100 + at * 8) for at in range(101)]
    traces = [trace(index, [(1, 8), (99, 800)]) for index in range(2)]
    window = {'status': 'passed', 'start': 2, 'end': 98}
    return samples, traces, window


def test_native_confirmation_requires_ninety_percent_window_coverage_by_default():
    samples, traces, window = native_fixture()
    result = gate.native_confirmation(samples, traces, window, 2, 3)
    assert result['status'] == 'passed' and result['sse_window_coverage_fraction'] == 1
    sparse = gate.native_confirmation(samples[2:5], traces, window, 2, 1)
    assert sparse['interval_count'] == 1
    assert sparse['status'] == 'insufficient_evidence'
    assert sparse['required_coverage_fraction'] == .9
    assert gate.native_confirmation(samples[2:5], traces, window, 2, 1, .005)['status'] == 'passed'


@pytest.mark.parametrize('contamination,reason', [
    ('prefill', 'prefill_increment_during_window'),
    ('refresh', 'refresh_during_window'),
    ('read_error', 'sample_read_error'),
])
def test_interior_contamination_cannot_pass_by_discarding_short_intervals(contamination, reason):
    samples, traces, window = native_fixture()
    if contamination == 'prefill':
        for sample in samples[50:]:
            sample['prefill'] += 1
    elif contamination == 'refresh':
        samples[50]['scheduler']['refresh_pending'] = 1
    else:
        samples[50] = {'at': 50, 'error': 'JSONDecodeError'}
    result = gate.native_confirmation(samples, traces, window, 2, 3)
    assert result['sse_window_coverage_fraction'] > .9
    assert result['status'] == 'failed'
    assert any(reason in item['reasons'] for item in result['violations'])
    if contamination == 'read_error':
        assert result['violations'][0]['error'] == 'JSONDecodeError'


def test_nonatomic_boundary_prefill_is_recorded_as_ambiguous_not_interior():
    samples, traces, window = native_fixture()
    samples[2]['at'], samples[2]['read_started_at'] = 2.01, 1.99
    samples[2]['prefill_raced'] = True
    for sample in samples[2:]:
        sample['prefill'] += 1
    result = gate.native_confirmation(samples, traces, window, 2, 3)
    assert result['status'] == 'passed'
    assert result['sse_window_coverage_fraction'] > .9
    assert result['boundary_observations'] and not result['violations']
    samples[2]['read_started_at'] = 2.001
    result = gate.native_confirmation(samples, traces, window, 2, 3)
    assert result['status'] == 'failed'
    assert any('prefill_raced_read' in item['reasons'] for item in result['violations'])


def test_prefill_increment_from_window_start_is_not_lost_as_first_sample():
    samples, traces, window = native_fixture()
    for sample in samples[3:]:
        sample['prefill'] += 1
    result = gate.native_confirmation(samples, traces, window, 2, 3)
    assert result['status'] == 'failed'
    assert any('prefill_increment_during_window' in item['reasons'] for item in result['violations'])


def test_token_digest_does_not_depend_on_speculative_chunk_boundaries():
    a, b = empty_trace(), empty_trace()
    gate.consume_event(a, {'choices': [{'token_ids': [1, 23, 4]}]}, 1)
    gate.consume_event(b, {'choices': [{'token_ids': [1]}]}, 1)
    gate.consume_event(b, {'choices': [{'token_ids': [23, 4]}]}, 2)
    assert a['_token_digest'].hexdigest() == b['_token_digest'].hexdigest()


@pytest.mark.parametrize('token_count', [None, 81, 262101])
def test_async_streaming_batch_keeps_exact_ids_and_concurrent_lifecycle(tmp_path, monkeypatch, token_count):
    token_prompt, input_count = token_count is not None, token_count or 81
    active, peak = 0, 0
    prompts = []

    class Stream(httpx.AsyncByteStream):
        def __init__(self, request_id):
            self.request_id = request_id

        async def __aiter__(self):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)

            def event(data):
                return ('data: ' + json.dumps({'id': self.request_id, **data}) + '\n\n').encode()

            try:
                yield event({'prompt_token_ids': list(range(input_count)),
                             'choices': [{'delta': {'role': 'assistant'}}]})
                for index in range(0, 64, 4):
                    await asyncio.sleep(.004)
                    yield event({'choices': [{'token_ids': list(range(index, index + 4)),
                                               'finish_reason': 'length' if index == 60 else None}]})
                yield event({'choices': [], 'usage': {'prompt_tokens': input_count, 'completion_tokens': 64}})
                yield b'data: [DONE]\n\n'
            finally:
                active -= 1

    async def handler(request):
        value = json.loads(request.content)
        prompts.append(value['prompt'] if token_prompt else value['messages'][0]['content'])
        assert value['return_token_ids'] and value['ignore_eos']
        assert value['temperature'] == 0
        if token_prompt:
            assert request.url.path.endswith('/completions') and 'messages' not in value
        else:
            assert value['chat_template_kwargs']['enable_thinking'] is False
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              stream=Stream('cmpl-' + str(len(prompts))))

    original_client = httpx.AsyncClient
    monkeypatch.setattr(gate.httpx, 'AsyncClient',
                        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    args = gate.parse_args(['--users-file', str(tmp_path / 'unused'), '--output', str(tmp_path / 'out'),
                           '--concurrency', '2', '--output-tokens', '64', '--trim-tokens', '8',
                           '--min-window-seconds', '.01', '--min-events-per-request', '2'] +
                           (['--prompt-token-file', 'unused.json'] if token_prompt else []))
    batch = asyncio.run(gate.run_batch(args, [{'api_key': 'secret-one'}, {'api_key': 'secret-two'}],
                                      2, list(range(input_count)) if token_prompt else gate.DEFAULT_PROMPT))
    assert batch['status'] == 'passed' and peak == 2 and active == 0
    assert prompts[0] == prompts[1]
    for request in batch['requests']:
        assert request['observed_tokens'] == 64 and request['first_burst_tokens'] == 4
        assert sum(len(point['token_ids']) for point in request['timeline']) == 64
    public = gate.public_evidence(batch)
    assert '_response_id' not in json.dumps(public)
    assert 'secret-one' not in json.dumps(public)


def test_cli_rejects_empty_decode_windows_and_nonfinite_timings(tmp_path):
    common = ['--users-file', str(tmp_path / 'users'), '--output', str(tmp_path / 'out')]
    for extra in (['--output-tokens', '128', '--trim-tokens', '64'], ['--trim-tokens', '0'],
                  ['--min-window-seconds', 'nan'], ['--min-events-per-request', '1'],
                  ['--min-native-coverage', '0'], ['--min-native-coverage', '1.1'],
                  ['--min-native-coverage', 'nan']):
        with pytest.raises(SystemExit):
            gate.parse_args(common + extra)


def test_long_prompt_echo_exceeds_default_limit_without_becoming_generated_output():
    prompt = [2**32-1] * 262101
    event = {'id': 'cmpl-long', 'prompt_token_ids': prompt,
        'choices': [{'token_ids': [11, 22], 'finish_reason': 'length'}],
        'usage': {'prompt_tokens': len(prompt), 'completion_tokens': 2}}
    raw = ('data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n').encode()
    async def collect(limit):
        response = httpx.Response(200, content=raw)
        return [value async for value in gate.stream_events(response, max_event_chars=limit)]
    with pytest.raises(gate.DecodeError, match='sse_event_too_large'):
        asyncio.run(collect(1024*1024))
    values = asyncio.run(collect(1024*1024 + 12*len(prompt)))
    assert values == [event, None]
    value = empty_trace()
    gate.consume_event(value, values[0], 1)
    gate.finish_trace(value, 2)
    assert value['observed_tokens'] == 2 and value['prompt_tokens'] == len(prompt)
    assert value['timeline'][0]['token_ids'] == [11, 22]


def test_long_prompt_event_limit_still_rejects_unbounded_metadata():
    limit = 1024*1024 + 12*262101
    async def collect():
        response = httpx.Response(200, content=('data: ' + 'x'*limit + '\n\n').encode())
        return [value async for value in gate.stream_events(response, max_event_chars=limit)]
    with pytest.raises(gate.DecodeError, match='sse_event_too_large'):
        asyncio.run(collect())


def test_exact_token_prompt_uses_completions_and_records_source_hash(tmp_path):
    source = tmp_path / 'prompt.json'
    raw = b'{ "input_ids": [42, 0, 151644, 19], "description": "fixture" }\n'
    source.write_bytes(raw)
    args = gate.parse_args(['--users-file', 'unused', '--output', str(tmp_path / 'out'),
                           '--prompt-token-file', str(source)])
    prompt, metadata, frozen = gate.load_prompt(args)
    assert args.endpoint == 'completions'
    assert prompt == [42, 0, 151644, 19] and frozen == raw
    assert metadata['input_tokens'] == 4
    assert metadata['input_file_sha256'] == hashlib.sha256(raw).hexdigest()
    assert metadata['token_ids_sha256'] == hashlib.sha256(b'[42,0,151644,19]').hexdigest()
    payload = gate.make_payload(args, prompt)
    assert payload['prompt'] == prompt and 'messages' not in payload
    assert 'chat_template_kwargs' not in payload
    with pytest.raises(SystemExit):
        gate.parse_args(['--users-file', 'unused', '--output', 'unused',
                         '--prompt-token-file', str(source), '--endpoint', 'chat/completions'])


@pytest.mark.parametrize('tokens', [[], [-1], [True], [[1]], ['1'], None])
def test_token_prompt_rejects_ambiguous_or_invalid_ids(tmp_path, tokens):
    source = tmp_path / 'prompt.json'
    source.write_text(json.dumps({'input_ids': tokens}))
    args = gate.parse_args(['--users-file', 'unused', '--output', 'unused',
                           '--prompt-token-file', str(source)])
    with pytest.raises(ValueError, match='input_ids'):
        gate.load_prompt(args)


def test_queued_requests_have_http_throughput_without_a_full_client_decode_window():
    args = gate.parse_args(['--users-file', 'unused', '--output', 'unused', '--allow-queued',
                           '--output-tokens', '64', '--trim-tokens', '8',
                           '--min-window-seconds', '.1', '--min-events-per-request', '2'])
    traces = []
    for index in range(32):
        value = trace(index, [(index * 10 + at, at * 8) for at in range(1, 9)])
        value.update(completion_tokens=64, prompt_tokens=100, ttft_seconds=index * 10 + 1,
                     elapsed_seconds=index * 10 + 8, tpot_seconds=7 / 56)
        traces.append(value)
    result = gate.batch_windows(args, traces, [], 32, 320)
    assert result['status'] == 'passed'
    assert result['steady_decode_window']['status'] == 'not_applicable'
    assert result['common_decode_window']['status'] == 'not_applicable'
    assert result['native_confirmation']['status'] == 'not_applicable'
    assert result['end_to_end']['successful_output_tokens_per_second'] == 2048 / 320
    assert result['end_to_end']['success'] == 32
    args.allow_queued = False
    assert gate.batch_windows(args, traces, [], 32, 320)['status'] == 'failed'
    args.allow_queued = True
    traces[-1]['outcome'] = 'rejected'
    result = gate.batch_windows(args, traces, [], 32, 320)
    assert result['status'] == 'failed' and result['end_to_end']['rejected'] == 1


def test_overlapping_streams_are_not_claimed_as_full_native_decode_cohort():
    args = gate.parse_args(['--users-file', 'unused', '--output', 'unused', '--capture', 'unused',
                           '--allow-queued', '--output-tokens', '64', '--trim-tokens', '8',
                           '--min-window-seconds', '.1', '--min-events-per-request', '2'])
    traces = []
    for index in range(2):
        value = trace(index, [(at, at * 8) for at in range(1, 9)])
        value.update(completion_tokens=64, prompt_tokens=100, ttft_seconds=1,
                     elapsed_seconds=8, tpot_seconds=7 / 56)
        traces.append(value)
    result = gate.batch_windows(args, traces, [], 2, 8)
    assert result['status'] == 'passed'
    assert result['steady_decode_window']['status'] == 'not_applicable'
    assert result['native_confirmation']['confirmation_attempt']['status'] == 'insufficient_evidence'

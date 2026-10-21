"""CPU-only provenance and geometry for an observed full-input Nsight trace."""
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def complete_geometry(steps, response_id, input_tokens):
    """Keep every original public prefill row, including the small final tails."""
    require(isinstance(response_id, str) and bool(response_id), 'Missing observed response identity')
    result, current, owner = [], 0, None
    for step in steps:
        geometry = step.get('batch_geometry', ())
        matches = [row for row in geometry if
            (row.get('request_id') == response_id or row.get('request_id', '').startswith(response_id+'-'))
            and row.get('prefill') is True]
        if not matches:
            continue
        require(len(matches) == len(geometry) == 1 and not step.get('atomic_prefill_tokens', 0) and
                not step.get('decode_requests', 0), 'Observed prefill must be one ordinary public owner')
        row = matches[0]
        if owner is None:
            owner = row['request_id']
        require(row['request_id'] == owner, 'Observed prefill owner changed')
        count = row['query_tokens']
        require(type(count) is int and 0 < count <= 16384, 'Unsupported observed physical chunk')
        current += count
        require(row['logical_seq_len'] == row['active_seq_len'] == current and
                row.get('offset', 0) == 0, 'Observed prefill has a gap, overlap, or refreshed active window')
        # Older frozen KVMem timers omit this newer native probe field. The
        # validated full input length and exact cumulative coordinates determine
        # the terminal boundary; if explicitly logged, it must agree.
        terminal = current == input_tokens
        if 'prefill_terminal' in row:
            require(type(row['prefill_terminal']) is bool and row['prefill_terminal'] == terminal,
                    'Observed prefill terminal geometry differs from the complete input')
        result.append(dict(query_tokens=count, logical_seq_len=current, active_seq_len=current,
                           prefill_terminal=terminal))
    require(bool(result) and current == input_tokens and result[-1]['prefill_terminal'],
            'Observed public prefill does not cover the complete input')
    return result


def capture_geometry(geometry, delay, limit):
    # Installed WorkerProfiler.step() runs BEFORE execute_model. delay=1 starts
    # at the first step; delay=2 starts at the second. Keep the old CLI meaning.
    require(type(delay) is int and 0 <= delay <= 1024 and type(limit) is int and 1 <= limit <= 64,
            'Invalid bounded observed capture window')
    begin = max(0, delay-1)
    selected = geometry[begin:begin+limit]
    require(len(selected) == limit and all(not row['prefill_terminal'] for row in selected),
            'Capture window must contain the requested nonterminal public prefill steps')
    return begin, selected


def observed_shape_plan(source, case_path, *, delay, limit):
    source, case_path = Path(source).resolve(), Path(case_path).resolve()
    require(case_path.parent.parent == source/'requests' and case_path.name == 'case.json',
            'Observed chunks require a case from this completed source request')
    evidence = {}
    def load(path, *, lines=False):
        data = path.read_bytes()
        evidence[str(path.relative_to(source))] = hashlib.sha256(data).hexdigest()
        return [json.loads(line) for line in data.splitlines()] if lines else json.loads(data)
    case = load(case_path)
    status, manifest, suite, profile = (load(source/name) for name in
        ('status.json', 'manifest.json', 'suite.json', 'profile.json'))
    require(status.get('status') == 'completed' and status.get('protocol_passed') is True,
            'Observed source did not complete its protocol')
    require(evidence['profile.json'] == manifest['profile_sha256'] and
            evidence['suite.json'] == manifest['suite_sha256'], 'Observed source profile/suite changed')
    require(suite.get('serial') is True and suite.get('prefix_caching') is False,
            'Observed source must be serial with prefix caching disabled')
    case_id = case_path.parent.name
    definitions = [item for item in suite['cases'] if item['id'] == case_id]
    require(len(definitions) == 1 and definitions[0]['warmup'] is False and
            definitions[0]['case_sha256'] == evidence[str(case_path.relative_to(source))],
            'Observed case must be an unchanged formal source request')
    result, response, policy = (load(case_path.parent/name) for name in
        ('result.json', 'response.json', 'evidence.json'))
    receipts = [item['result'] for item in status['results'] if item['id'] == case_id]
    require(receipts == [result], 'Observed result differs from the completed source receipt')
    require(case['row']['served_input_tokens'] == result.get('input_tokens') ==
            response['usage']['prompt_tokens'] == 32768 and result.get('status') == 'completed' and
            result['input_sha256'] == case['row']['served_sha256'] and
            result['completion_tokens'] == response['usage']['completion_tokens'] == len(response['output_ids']),
            'Observed request input/output accounting differs')
    data = Path(case['input_path']).read_bytes()
    require(len(data) == 32768*4 and hashlib.sha256(data).hexdigest() == case['row']['served_sha256'],
            'Observed input bytes differ from the complete frozen token input')
    require(policy.get('refresh') == [] and not result.get('pressure_count') and not result.get('query_count'),
            'Observed full32K request unexpectedly refreshed its window')
    require(profile['prefill_chunk_tokens'] == 2048 and profile['max_num_batched_tokens'] == 16424,
            'Observed source changed the logical chunk/native batch policy')
    steps = load(source/'run/server/capture/model_step_latency.jsonl', lines=True)
    geometry = complete_geometry(steps, response['response_id'], 32768)
    begin, selected = capture_geometry(geometry, delay, limit)
    chunks = [row['query_tokens'] for row in geometry]
    maximum = max(chunks)
    require(maximum >= 2048 and maximum % 2048 == 0, 'Observed native cap must retain whole logical chunks')
    return dict(mode='observed_full32k', source_case=case_id, source_response_id=response['response_id'],
        chunks=chunks, geometry=geometry, max_chunk=maximum, capture_begin_index=begin,
        capture_geometry=selected, source_evidence_sha256=evidence,
        note='Actual complete public input; native forces these chunks, KVMem retains its frozen scheduler. '
             'The entire measured request and selected NVTX shapes must reproduce this geometry.')


def observed_native_argv(argv, shape_plan):
    """Only widen the existing diagnostic cap after deriving real source chunks."""
    result = list(argv)
    option = '--long-prefill-token-threshold'
    maximum = shape_plan['max_chunk']
    require(type(maximum) is int and 2048 <= maximum <= 16384 and maximum % 2048 == 0 and
            all(type(n) is int and 0 < n <= maximum for n in shape_plan['chunks']) and
            sum(shape_plan['chunks']) == 32768, 'Invalid observed native chunk schedule')
    require(result.count(option) == 1 and result.index(option)+1 < len(result) and
            result[result.index(option)+1] == '2048',
            'Expected the unchanged native diagnostic 2K cap')
    result[result.index(option)+1] = str(maximum)
    return result


def verify_observed_trace(steps, response_id, shape_plan, shapes, limit):
    actual = complete_geometry(steps, response_id, 32768)
    require(actual == shape_plan['geometry'], 'Actual complete prefill differs from the frozen observed block table')
    expected = shape_plan['capture_geometry']
    require(len(shapes) == len(expected) == limit, 'Trace has a different number of observed prefill steps')
    for shape, row in zip(shapes, expected):
        require(shape['context_requests'] == 1 and shape['generation_requests'] == shape['generation_tokens'] == 0 and
                shape['scheduled_tokens'] == shape['query_tokens'] == row['query_tokens'] and
                shape['logical_seq_len'] == row['logical_seq_len'],
                'NVTX capture differs from the frozen observed step window')
        shape.update(active_seq_len=row['active_seq_len'], actual_geometry_verified=True)

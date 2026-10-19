"""Fixed C8 workload for attention and graph-collection cost attribution.

All variants use recency pressure and unpropagated Mean-K Query selection.
This is diagnostic overhead screening, not an answer-quality comparison.
"""
import argparse
import array
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import graph_v2_protocol as protocol
import qualify_block_graph as lifecycle
import qualify_decode as decode
import qualify_graph_niah as native
from qualify_graph_v2_decode import validate_fixture, validate_identities, absolute_window

VARIANTS = {'disabled': (False, 'off'), 'attention': (True, 'off'), 'collect': (True, 'collect')}


def profiles(base):
    result = {}
    for variant, (enabled, mode) in VARIANTS.items():
        value = deepcopy(base)
        value['kvmem'].update(pressure_policy='recency', action_attention=enabled,
            prefill_attention=enabled, graph_mode=mode, action_fraction=0.,
            action_profiling=False, graph_trace=False)
        validate_profile(value, variant)
        result[variant] = value
    return result


def validate_profile(profile, variant):
    enabled, mode = VARIANTS[variant]
    kv = profile['kvmem']
    expected = dict(select_budget=36864, gen_budget=16384, block_tokens=32,
        action_attention=enabled, prefill_attention=enabled, graph_mode=mode,
        pressure_policy='recency', action_fraction=0., query_history=False,
        guided_query_tokens=0, graph_trace=False, action_profiling=False,
        step_latency_interval=1, mean_k_backend='qw3_cuda', archive_backend='cpu')
    for name, value in expected.items():
        if kv.get(name) != value:
            raise ValueError('Cost profile differs: '+name)
    if (profile.get('execution_mode') != 'compiled_graph' or profile.get('draft_method') != 'mtp'
            or profile.get('draft_tokens') != 4 or profile.get('max_num_seqs') != 8
            or profile.get('max_num_batched_tokens') != 16424):
        raise ValueError('Keep compiled native C8/MTP4/batch geometry')


def audit_launch(profile, capture, variant):
    validate_profile(profile, variant)
    proof = native.audit_launch(profile, capture)
    launch = json.loads((capture.parent/'launch.json').read_text())
    argv = launch['vllm_argv']
    transfer = json.loads(argv[argv.index('--kv-transfer-config')+1])
    actual = transfer['kv_connector_extra_config']['kvmem']
    for key in ('action_attention', 'prefill_attention', 'graph_mode', 'pressure_policy',
                'action_fraction', 'query_history', 'graph_alpha', 'graph_scoring_version'):
        lifecycle.require(actual[key] == profile['kvmem'][key], 'Worker setting differs: '+key)
    return dict(proof, cost_variant=variant, pressure_policy='recency', graph_propagation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'capture', 'output', 'fixture'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=VARIANTS, required=True)
    args = parser.parse_args()
    profile = json.loads(args.config.read_text())
    fixture = json.loads(args.fixture.read_text())
    validate_profile(profile, args.arm)
    validate_fixture(fixture, profile)
    args.output.mkdir(parents=True, exist_ok=False)
    lifecycle.save(args.output/'launch-audit.json', audit_launch(profile, args.capture, args.arm))
    values = array.array('i'); values.frombytes(Path(fixture['input_i32']).read_bytes())
    if sys.byteorder != 'little': values.byteswap()
    tokens = values.tolist()
    prompt = args.output/'input.json'
    lifecycle.save(prompt, dict(input_ids=tokens))
    labels = args.output/'local-labels.json'
    lifecycle.save(labels, dict(users=[dict(id=f'owner-{i}', api_key=f'local-cost-label-{i}') for i in range(8)]))
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
    began = time.monotonic()
    try:
        code = decode.main(['--api-base', f"http://127.0.0.1:{profile['port']}/v1",
            '--model', profile['served_model_name'], '--users-file', str(labels),
            '--output', str(args.output/'decode'), '--capture', str(args.capture),
            '--endpoint', 'completions', '--prompt-token-file', str(prompt), '--prompt-sessions',
            '--query-span', *map(str, fixture['query_span']), '--concurrency', '8', '--rounds', '1',
            '--output-tokens', '16384', '--trim-tokens', '64', '--min-window-seconds', '30',
            '--min-native-coverage', '.9', '--timeout', '1800'])
        ended = time.monotonic()
        report = json.loads((args.output/'decode/results.json').read_text())
        lifecycle.require(code == 0 and report['status'] == 'passed', 'Strict C8 decode qualification failed')
        batch = report['batches'][0]
        start, end = absolute_window(batch)
        steps = [r for r in lifecycle.records(args.capture/'model_step_latency.jsonl')
            if start <= r['started_at'] and r['submitted_at'] <= end]
        lifecycle.require(steps and all(r['decode_requests'] == 8 and not r['prefill_requests']
            and not r['prefill_tokens'] for r in steps), 'Completed native steps must confirm C8 decode')
        validate_identities(sessions, responses)
        terminals = [native.wait_retired(args.capture, r['response_id']) for r in responses]
        lifecycle.require(all(r['history_end'] == len(tokens)+16384-1 and not r.get('checkpoint_tokens')
            for r in terminals), 'Complete accepted history/checkpoint differs')
        released = lifecycle.wait_released(args.capture, sessions, timeout=180)
        lifecycle.require(not released['archive']['disk_read_bytes'] and not released['archive']['disk_write_bytes'],
            'Historical KV must stay in CPU memory')
        actions = lifecycle.records(args.capture/'action_attention.jsonl')
        graphs = lifecycle.records(args.capture/'block_graph.jsonl')
        enabled, mode = VARIANTS[args.arm]
        lifecycle.require(bool(actions) == enabled, 'Attention instrumentation was not exercised as declared')
        if enabled:
            lifecycle.require(len(actions) == 8 and all(r['samples'] > 0 and r['prefill_samples'] > 0 for r in actions),
                'Require actual output and prefill samples for every owner')
        if mode == 'off':
            lifecycle.require(not graphs and not any('block_graph' in r for r in actions), 'Unexpected graph work')
        else:
            lifecycle.require(len(graphs) == 8 and all(r.get('selection_unchanged') is True for r in graphs),
                'Collect must leave Query selection unchanged')
            lifecycle.require(all(r.get('block_graph', {}).get('observed_sources', 0) > 0 for r in actions),
                'Require accepted graph construction for every owner')
        requests = batch['requests']
        cold_seconds = max(r['first_token_at'] for r in requests)-min(r['started_at'] for r in requests)
        result = dict(status='passed', all_owned_sessions_released=True, variant=args.arm,
            cases=8, input_tokens=len(tokens), output_tokens_each=16384,
            shared_decode_window=batch['steady_decode_window'], end_to_end=batch['end_to_end'],
            cold_preparation_seconds=cold_seconds, effective_input_tokens_per_second=8*len(tokens)/cold_seconds,
            native_confirmation=batch['native_confirmation'], completed_C8_steps=len(steps),
            actual_concurrency=protocol.audit_actual_concurrency(steps), requests=responses,
            phase_summary=lifecycle.summarize_steps(args.capture, began, ended),
            final_archive=released['archive'], measurement_started_at=began, measurement_finished_at=ended,
            scope='Single cold cost screen. Identical recency pressure/unpropagated Mean-K; full256K input and16K accepted output. '
                'Compare actual selection/output identities and MTP acceptance before causal attribution. Not quality or agent throughput.')
        lifecycle.save(args.output/'results.json', result)
    except BaseException as error:
        lifecycle.save(args.output/'client-failure.json', dict(error_type=type(error).__name__, error=str(error), at=time.time()))
        raise
    finally:
        decode.make_payload, decode.read_sample, decode.public_evidence = original_payload, original_read, original_public


if __name__ == '__main__':
    main()

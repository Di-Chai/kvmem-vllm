"""Audit saved experiment evidence without launching any inference requests."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path

from analyze_gpu import prometheus, read, save
from protocol import quantile


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (actual, expected)


def verify(root, partial=False):
    base = root / 'serving-r1'
    rows = read(base / 'summary.json')
    expected = {f'l{n}-o{o}-c{c}-{k}' for n, o, c, k in itertools.product(
        (4096, 8192, 16384, 32768), (256, 512, 1024), (1, 2, 4, 8, 16), ('observer', 'pair'))}
    names = [r['name'] for r in rows]
    assert len(names) == len(set(names)) and set(names) <= expected
    if not partial:
        assert set(names) == expected
        assert read(base / 'status.json') == dict(status='completed', cells=120)
    counts = Counter()
    anomalies = []
    for cell in rows:
        path = base / cell['name']
        assert cell['name'] == f"l{cell['input_length']}-o{cell['output_length']}-c{cell['concurrency']}-{cell['kind']}"
        assert read(path / 'summary.json') == cell
        arms = ('full', 'reduced') if cell['kind'] == 'pair' else ('observer',)
        request_names = {f'{i}-{arm}.json' for i in range(cell['concurrency']) for arm in arms}
        assert {p.name for p in path.glob('*.json') if p.name != 'summary.json'} == request_names
        results = [read(path / name) for name in sorted(request_names)]
        assert {(r['pair_id'], r['arm']) for r in results} == {
            (i, arm) for i in range(cell['concurrency']) for arm in arms}
        ok = [r for r in results if r['success']]
        assert cell['submitted'] == len(results)
        assert cell['completed'] == len(ok)
        assert cell['failed'] == len(results) - len(ok)
        counts.update(cells=1, submitted=len(results), completed=len(ok), failed=len(results)-len(ok))
        for r in results:
            expected_input = cell['input_length'] - (32 if r['arm'] == 'reduced' else 0)
            assert r['input_tokens'] == expected_input
            assert r['elapsed_seconds'] > 0
            assert r['ended'] >= r['started']
            if not r['success']:
                assert r.get('error'), 'A failed request must retain its error'
                continue
            assert r['usage']['prompt_tokens'] == expected_input
            assert r['requested_output_tokens'] == r['actual_output_tokens'] == cell['output_length']
            assert r['usage']['completion_tokens'] == len(r['token_ids']) == cell['output_length']
            assert r['usage']['total_tokens'] == expected_input + cell['output_length']
            assert sum(e['tokens'] for e in r['events']) == cell['output_length']
            offsets = [e['offset_seconds'] for e in r['events']]
            assert offsets == sorted(offsets)
            assert 0 < r['client_ttft_seconds'] <= r['elapsed_seconds']
            close(r['client_ttft_seconds'], offsets[0])
            close(r['client_decode_seconds'], offsets[-1] - offsets[0])
            for key in ('time_to_first_token_ms', 'generation_time_ms', 'queue_time_ms'):
                assert r['metrics'][key] >= 0
            counts['output_tokens'] += len(r['token_ids'])
        close(cell['aggregate_output_tokens_per_wall_second'],
              sum(r['actual_output_tokens'] for r in ok) / cell['wall_seconds'])
        if ok:
            close(cell['ttft_p50_seconds'], quantile([r['client_ttft_seconds'] for r in ok], .5))
            close(cell['ttft_p95_seconds'], quantile([r['client_ttft_seconds'] for r in ok], .95))
            close(cell['queue_p95_ms'], quantile([r['metrics']['queue_time_ms'] for r in ok], .95))
            close(cell['per_request_token_weighted_decode_tps'],
                  sum(r['actual_output_tokens']-1 for r in ok)*1000 /
                  sum(r['metrics']['generation_time_ms'] for r in ok))
        before, after = [prometheus(path / f'metrics-{part}.txt') for part in ('before', 'after')]
        observed = after['vllm:request_prefill_time_seconds_count'] - before['vllm:request_prefill_time_seconds_count']
        if observed != len(ok):
            anomalies.append(dict(cell=cell['name'], metric='prefill_histogram_count',
                                  observed=observed, completed=len(ok)))
        # Failed/disconnected requests can still contribute engine counters.
        # Preserve discrepancies rather than silently calling the samples exact.
        if not cell['failed']:
            for metric, wanted in [('vllm:generation_tokens_total', sum(len(r['token_ids']) for r in ok)),
                                   ('vllm:prompt_tokens_total', sum(r['input_tokens'] for r in ok))]:
                observed = after[metric] - before[metric]
                if observed != wanted:
                    anomalies.append(dict(cell=cell['name'], metric=metric, observed=observed, expected=wanted))
    if not partial:
        assert counts['submitted'] == 1116
        if not counts['failed']:
            assert counts['output_tokens'] == 666624
    native = base / 'native-systemone'
    native_rows = read(native / 'summary.json') if (native / 'summary.json').exists() else []
    native_expected = {f'l{n}-c{c}' for n, c in itertools.product((4096, 8192, 16384, 32768), (1, 2, 4, 8, 16))}
    native_names = [r['name'] for r in native_rows]
    assert len(native_names) == len(set(native_names)) and set(native_names) <= native_expected
    if not partial:
        assert set(native_names) == native_expected
        assert read(native / 'status.json') == dict(status='completed', cells=20)
    native_counts = Counter()
    for cell in native_rows:
        path = native / cell['name']
        assert cell['name'] == f"l{cell['target_input']}-c{cell['concurrency']}"
        assert read(path / 'summary.json') == cell
        results = [read(path / f'response-{i}.json') for i in range(cell['concurrency'])]
        assert len(list(path.glob('response-*.json'))) == cell['concurrency']
        ok = [r for r in results if r['success']]
        assert cell['submitted'] == len(results) and cell['completed'] == len(ok)
        assert cell['failed'] == len(results)-len(ok)
        for r in ok:
            body = r['body']
            assert body['usage']['input_tokens'] == cell['target_input']
            assert body['usage']['output_tokens'] == 1
            assert set(body['answers']) == {'goal_coverage'}
            assert body['answers']['goal_coverage']['choice'] in ('documented', 'absent', 'uncertain')
        native_counts.update(cells=1, submitted=len(results), completed=len(ok), failed=len(results)-len(ok))
        close(cell['labels_per_wall_second'], len(ok)/cell['wall_seconds'])
        if ok:
            close(cell['http_latency_p95_seconds'], quantile([r['seconds'] for r in ok], .95))
    if not partial:
        assert native_counts['submitted'] == 124
    semantic = read(root / 'judge-r2/status.json')
    assert semantic['controls_passed'] is False and semantic['semantic_metric_validated'] is False
    report = dict(structural_audit_passed=True, full_matrix_present=not partial,
                  generative=dict(counts), native=dict(native_counts), metric_counter_discrepancies=anomalies,
                  semantic_controls_passed=False, semantic_metric_validated=False,
                  explanation='Request and aggregate consistency audit; neither human semantic labels nor deletion safety validation.')
    save(root / ('verification-partial.json' if partial else 'verification.json'), report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.allow_partial)))

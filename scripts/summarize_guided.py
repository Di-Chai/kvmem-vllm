"""Paired quality and measured-cost report; incomplete arms have no headline score."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from analyze_memory import key, lines
from benchmark_niah import save
from guided_benchmark_protocol import require


def mean(values):
    return statistics.fmean(values) if values else None


def describe(values):
    return dict(count=len(values), mean=mean(values),
                median=statistics.median(values) if values else None)


def measure(directory):
    status = json.loads((directory/'status.json').read_text())
    capture = directory/'server/capture'
    refreshes = lines(capture/'bounded_refresh.jsonl')
    queries = [r for r in refreshes if r.get('selection') == 'query']
    steps = lines(capture/'model_step_latency.jsonl')
    decode = [r for r in steps if r['decode_requests'] and not r['prefill_requests']]
    c8 = [r for r in decode if r['decode_requests'] == 8]
    prefill = [r for r in steps if r['prefill_requests'] and not r['decode_requests']]
    resources = lines(directory/'resources.jsonl')
    records = [json.loads(p.read_text()) for p in sorted((directory/'answers/cases').glob('*/result.json'))]
    outputs = sum(r.get('completion_tokens', r.get('usage', {}).get('completion_tokens', 0)) for r in records)
    seconds = status.get('inference_seconds')
    return dict(cold_prepare_seconds=status.get('cold_prepare_seconds'),
        whole_suite_seconds=seconds, completion_tokens=outputs,
        whole_suite_output_tokens_per_second=outputs/seconds if seconds else None,
        pure_public_prefill_model_tokens_per_second=(
            1000*sum(r['prefill_tokens'] for r in prefill)/sum(r['model_ms'] for r in prefill)
            if prefill else None),
        decode_step_ms=describe([r['step_ms'] for r in decode]),
        common_c8_decode_step_ms=describe([r['step_ms'] for r in c8]),
        query_count=len(queries), generated_query_count=sum(bool(r.get('guided_published_tokens')) for r in queries),
        guided_tokens=sum(r.get('guided_query_tokens', 0) for r in queries),
        private_forward_seconds=sum(r.get('guided_seconds', 0) for r in queries),
        guided_fallbacks=dict(Counter(r['guided_fallback'] for r in queries if r.get('guided_fallback'))),
        min_available_cpu_bytes=min((r['cpu']['MemAvailable'] for r in resources), default=None),
        max_gpu_used_mib=max((float(r['gpu'].split(',')[0]) for r in resources), default=None),
        performance_scope='Descriptive quality-workload costs, not isolated guided overhead; output trajectories and MTP acceptance may differ.',
        mtp_metrics=[r['metrics']['speculative_decoding'] for r in records
                     if 'speculative_decoding' in r.get('metrics', {})])


def scores(directory, suite):
    answers = directory/'answers'
    if suite == 'niah':
        values = {p.parent.name:json.loads(p.read_text())['evaluation']['normalized_exact_numeric_set']
                  for p in (answers/'cases').glob('*/result.json')}
        require(len(values) == 60, 'Incomplete NIAH')
        return dict(score=mean(list(values.values())), metric='exact accuracy'), values
    if suite == 'longmemeval_s':
        summary = json.loads((answers/'scoring-pro/summary.json').read_text())
        require(summary['complete'] and summary['count'] == 102, 'Incomplete LME judgments')
        values = {p.stem:json.loads(p.read_text())['correct'] for p in (answers/'scoring-pro/scores').glob('*.json')}
        return dict(score=summary['accuracy'], metric='official prompt / Pro accuracy',
                    by_type=summary['by_type']), values
    summary = json.loads((answers/'scoring/summary.json').read_text())
    require(summary['complete'] and summary['scored'] == 1116, 'Incomplete MAB judgments')
    special = {r['key']:r['score'] for r in lines(answers/'scoring/special_judgments.jsonl')}
    values = {key(r):special[key(r)] if key(r) in special else r['official_local_metrics']['substring_exact_match']
              for r in lines(answers/'scoring/rows/selected/results.jsonl')}
    return dict(score=summary['three_category_macro'], metric='three-category macro; not four-category official full set',
                by_source=summary.get('by_source')), values


def run(root):
    report = dict(status='completed', suites={},
        baseline='Historical C: V2 multi Query alpha .2 / cumulative attention pressure',
        judge='Both arms use DeepSeek Pro; historical LME Flash scores are not paired controls')
    for suite in ('niah', 'longmemeval_s', 'memoryagentbench'):
        arms, values = {}, {}
        for arm in ('off', 'guided'):
            directory = root/f'{suite}-{arm}'
            require(json.loads((directory/'status.json').read_text())['status'] == 'completed', 'Incomplete inference')
            quality, values[arm] = scores(directory, suite)
            arms[arm] = dict(quality=quality, measurements=measure(directory))
        require(values['off'].keys() == values['guided'].keys(), 'Unpaired cases')
        delta = {k:float(values['guided'][k])-float(v) for k, v in values['off'].items()}
        report['suites'][suite] = dict(arms=arms,
            score_delta=arms['guided']['quality']['score']-arms['off']['quality']['score'],
            paired=dict(count=len(delta), improved=sum(v>0 for v in delta.values()),
                        regressed=sum(v<0 for v in delta.values()), tied=sum(v==0 for v in delta.values())),
            per_case_delta=delta)
    save(root/'comparison.json', report)
    print(json.dumps({suite:dict(delta=value['score_delta'], paired=value['paired'])
                      for suite, value in report['suites'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    run(parser.parse_args().campaign.resolve())

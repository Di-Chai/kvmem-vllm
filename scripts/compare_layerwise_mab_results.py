"""Compare completed official MAB verdicts without invoking or retrying a judge."""
import argparse
import hashlib
import json
from pathlib import Path


REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
HISTORICAL = REPO/'artifacts/layerwise-suite-20261102-r7/analysis/all-methods-comparison-v2.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_scores(values):
    binary = {k: v for k, v in values.items() if not k.startswith('Long_Range_Understanding|')}
    if len(binary) != 1100 or not all(v in (0, 1) for v in binary.values()):
        raise ValueError('Expected 1100 binary QA verdicts; summarization is not pass/fail')
    return binary


def pairwise(before, after):
    if before.keys() != after.keys():
        raise ValueError('Only compare identical scored samples')
    wins = sum(after[k] > v + 1e-12 for k, v in before.items())
    losses = sum(after[k] < v - 1e-12 for k, v in before.items())
    return dict(wins=wins, losses=losses, ties=len(before)-wins-losses)


def run(campaign, scores, output):
    audit = read(campaign/'execution-audit.json')
    if audit['status'] != 'passed' or audit.get('qualification'):
        raise ValueError('Require the completed formal execution audit')
    current = read(scores)
    if not current['complete'] or current['count'] != 1116:
        raise ValueError('Require all 1116 official verdicts')
    historical = read(HISTORICAL)['arms']
    references = {'mean_k_attention_pressure': historical['historical_A']['mab'],
                  'shared_pressure_layerwise_query': historical['layerwise']['mab']}
    rows = {}
    files = {str(scores): sha(scores), str(HISTORICAL): sha(HISTORICAL),
             str(campaign/'execution-audit.json'): sha(campaign/'execution-audit.json')}
    for p in (campaign/'run/answers/cases').glob('*/result.json'):
        row = read(p)
        key = f"{row['split']}|{row['dataset_row']}|{row['question_index']}"
        if row['status'] != 'completed' or key in rows:
            raise ValueError('Duplicate or incomplete answer')
        rows[key] = dict(id=row['id'], source=row['source'],
                         truncated=row['finish_reason'] == 'length',
                         inherited=bool(row.get('inherited', False)))
        files[str(p)] = sha(p)
    if rows.keys() != current['values'].keys():
        raise ValueError('Score identities differ from the completed first answers')
    binary = binary_scores(current['values'])
    comparisons = {}
    for name, before in references.items():
        if before['judge'] != current['judge'] or before['count'] != 1116:
            raise ValueError('Official grading protocol differs')
        comparisons[name] = dict(
            baseline_macro=100*before['three_category_macro'],
            current_macro=100*current['three_category_macro'],
            macro_delta_points=100*(current['three_category_macro']-before['three_category_macro']),
            all_score_pairs=pairwise(before['values'], current['values']),
            binary_pairs=pairwise(binary_scores(before['values']), binary),
            binary_pass_rates=dict(baseline=100*sum(binary_scores(before['values']).values())/1100,
                                   current=100*sum(binary.values())/1100),
            by_source={s: dict(count=v['count'], baseline=100*before['by_source'][s]['score'],
                              current=100*v['score'],
                              delta_points=100*(v['score']-before['by_source'][s]['score']))
                       for s, v in current['by_source'].items()},
            by_category={s: dict(baseline=100*before['by_category'][s], current=100*v,
                                delta_points=100*(v-before['by_category'][s]))
                         for s, v in current['by_category'].items()})
    report = dict(count=1116, official_macro=100*current['three_category_macro'],
                  binary_qa=dict(count=1100, passed=sum(binary.values()),
                                 pass_rate=100*sum(binary.values())/1100),
                  truncated=sum(r['truncated'] for r in rows.values()),
                  truncation_rate=100*sum(r['truncated'] for r in rows.values())/1116,
                  comparisons=comparisons, files=files,
                  limitations=[
                      'Official total is the macro mean of three categories, not 1116 binary answers.',
                      'Binary QA excludes the 16 summarization cases, which use continuous scores.',
                      'Historical Mean-K uses graph collect; layerwise uses graph multi and newer execution paths.',
                      'The shared-pressure/layerwise-Query reference is the closer control for layerwise pressure.',
                      'Eight-owner numerical batch dependence prevents a claim of exact full-trajectory equivalence.'])
    if (campaign/'inference-performance.json').exists():
        report['inference_performance'] = read(campaign/'inference-performance.json')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('count', 'official_macro', 'binary_qa', 'truncated')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'scores', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.campaign.resolve(), args.scores.resolve(), args.output.resolve())

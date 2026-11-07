"""Export explicitly authored, post-hoc source reviews, NOT independent labels."""
import argparse
import csv
import html
import json
from pathlib import Path

from audit import check, load_lines, normalize_boundaries
from observer import digest, save


def strings(value, pointer=''):
    if isinstance(value, str):
        yield pointer, value
    elif isinstance(value, (list, dict)):
        entries = enumerate(value) if isinstance(value, list) else value.items()
        for key, child in entries:
            escaped = str(key).replace('~', '~0').replace('/', '~1')
            yield from strings(child, pointer + '/' + escaped)


def locate(bundle, step, query):
    for message in step['messages']:
        record = json.loads((bundle / message['trace_path']).read_text().splitlines()[message['trace_line'] - 1])
        if query.get('role') and record['original']['role'] != query['role']:
            continue
        for i, block in enumerate(record['original']['content']):
            if query.get('kind') and block['type'] != query['kind']:
                continue
            for pointer, text in strings(block, f'/original/content/{i}'):
                if query.get('field') and not pointer.endswith('/' + query['field']):
                    continue
                needle = query['needle']
                if needle not in text:
                    continue
                start = max(0, text.index(needle) - 40)
                end = min(len(text), start + max(240, len(needle)))
                quote = text[start:end]
                visible = step['current_step']
                return dict(source={**message, 'pointer': pointer, 'character_span': [start, end]},
                            quote=quote, needle=needle,
                            needle_present_in_current_step=(needle in visible
                                or json.dumps(needle, ensure_ascii=False)[1:-1] in visible
                                or any(needle in e['quote'] for e in step['evidence'].values())))
    raise ValueError(f'No raw-source match for Step {step["step"]}: {query}')


def export(bundle, dataset, results, cases, output):
    source = {(s['window_id'], s['step']): s for s in load_lines(dataset)}
    predictions = {(p['window_id'], p['step']): p for p in
                   normalize_boundaries(load_lines(results / 'decisions.jsonl'))}
    specs = json.loads(cases.read_text())
    rows = []
    for spec in specs['cases']:
        wid = spec.get('window_id', next(iter(source))[0])
        matches = [predictions[wid, turn] for turn in spec['steps']]
        citations = [locate(bundle, source[wid, query['step']], query) for query in spec['citations']]
        row = dict(spec, window_id=wid, predictions=matches, verified_raw_citations=citations,
                   review_type='posthoc_model_assisted_source_review', independent_human_label=False)
        rows.append(row)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'cases.json', dict(reference_type=specs['reference_type'],
         case_spec_sha256=digest(cases), counts_are_selected_cases_not_error_rates=True, cases=rows))
    fields = ['case_id', 'issue_type', 'adjudication', 'window_id', 'steps', 'rationale',
              'predicted_stage_ids', 'predicted_activities', 'predicted_outcomes',
              'format_diagnostics', 'verified_raw_citations', 'independent_human_label']
    with (output / 'cases.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields[:6]} | dict(
                predicted_stage_ids=json.dumps([p['stage_id'] for p in row['predictions']]),
                predicted_activities=json.dumps([p['activity'] for p in row['predictions']]),
                predicted_outcomes=json.dumps([p['outcome'] for p in row['predictions']]),
                format_diagnostics=json.dumps([p['diagnostics'] for p in row['predictions']]),
                verified_raw_citations=json.dumps(row['verified_raw_citations'], ensure_ascii=False),
                independent_human_label=False))
    parts = ['<!doctype html><meta charset="utf-8"><title>Source case review</title>',
             '<style>body{font:15px system-ui;margin:30px;max-width:1100px}pre{white-space:pre-wrap}</style>',
             '<h1>Post-hoc model-assisted source review</h1>',
             '<p>Selected cases, not independent ground truth or estimated error rates. '
             'All raw citations are verified; source validity is separate from semantic support. '
             'Later Steps are reviewer-only context, never Observer inputs.</p>']
    for row in rows:
        parts.append('<h2>' + html.escape(row['case_id'] + ': ' + row['issue_type']) + '</h2>')
        parts.append('<p>' + html.escape(row['adjudication'] + ' — ' + row['rationale']) + '</p>')
        for citation in row['verified_raw_citations']:
            parts.append('<pre>' + html.escape(json.dumps(citation, ensure_ascii=False, indent=2)) + '</pre>')
        parts.append('<details><summary>Native predictions / diagnostics</summary><pre>' +
                     html.escape(json.dumps(row['predictions'], ensure_ascii=False, indent=2)) + '</pre></details>')
    (output / 'cases.html').write_text('\n'.join(parts))
    print(json.dumps(dict(cases=len(rows), output=str(output),
                         independent_human_labels=False), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'dataset', 'results', 'cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    export(args.bundle, args.dataset, args.results, args.cases, args.output)


if __name__ == '__main__':
    main()

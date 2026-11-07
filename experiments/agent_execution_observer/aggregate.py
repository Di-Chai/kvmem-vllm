"""Aggregate audited cohorts and paired format probes without new inference."""
import argparse
import copy
import csv
import difflib
import hashlib
import json
from pathlib import Path
import tarfile

from audit import check, diagnostics, distribution, load_lines, normalize_boundaries
from observer import digest, save, validate


LETTER_SUFFIX = (' Return ONLY the single option LETTER (A, B, C, etc.). '
                 'Do not return an option name, a stage ID such as S001, JSON, or an explanation.')


def probe_audit(completed):
    full = completed / 'full-short-results'
    probes = completed / 'format-probes'
    by_step = {p['step']: p for p in load_lines(full / 'decisions.jsonl')}
    summary = json.loads((probes / 'summary.json').read_text())
    check(len(summary) == 6, 'Expected exactly six format probes')
    pairs = []
    for turn in (6, 10, 16):
        before = json.loads((full / 'calls' / (by_step[turn]['call_id'] + '.request.json')).read_text())
        pair = dict(step=turn)
        for variant in ('original', 'explicit_letter'):
            ident = f'step-{turn}-{variant}'
            expected = copy.deepcopy(before)
            expected['questions'] = {'stage_link': expected['questions']['stage_link']}
            if variant == 'explicit_letter':
                expected['questions']['stage_link']['instructions'] += LETTER_SUFFIX
            request = json.loads((probes / (ident + '.request.json')).read_text())
            response = json.loads((probes / (ident + '.response.json')).read_text())
            check(json.dumps(request,ensure_ascii=False) == json.dumps(expected,ensure_ascii=False),
                  f'Unexpected probe state, prompt or option order change: {ident}')
            validate(response, request)
            row = next(r for r in summary if r['step'] == turn and r['variant'] == variant)
            check(row['answer'] == response['answers']['stage_link']
                  and row['diagnostics'] == response['diagnostics']['stage_link']
                  and row['usage'] == response['usage'], 'Probe summary differs from native response')
            pair[variant] = row
        pair['choice_changed'] = pair['original']['answer']['choice'] != pair['explicit_letter']['answer']['choice']
        pairs.append(pair)
    return dict(status='passed', pairs=pairs, independent_validation=False,
                selection='Adaptive diagnostic of three previously observed problematic requests',
                http_calls=len(summary), input_tokens=sum(r['usage']['input_tokens'] for r in summary),
                label_reads=sum(r['usage']['output_tokens'] for r in summary),
                http_seconds=sum(r['seconds'] for r in summary))


def runtime_audit(root, completed, windows):
    with tarfile.open(root / 'pilot-inputs.tar.gz') as archive:
        members = [m for m in archive.getmembers() if m.name.endswith('/observer.py')]
        check(len(members) == 1, 'Unexpected archived baseline code')
        original = archive.extractfile(members[0]).read()
    current = Path(__file__).with_name('observer.py').read_bytes()
    old_hash = hashlib.sha256(original).hexdigest()
    new_hash = hashlib.sha256(current).hexdigest()
    old_run = json.loads((completed / 'full-short-results' / 'run.json').read_text())
    new_run = json.loads((windows / 'run.json').read_text())
    correction = json.loads((completed / 'client-boundary-correction.json').read_text())
    check(old_hash == old_run['observer_sha256'] == correction['original_runtime'], 'v1 runtime hash differs')
    check(new_hash == new_run['observer_sha256'] == correction['window_runtime'], 'v2 runtime hash differs')
    old_line = "            boundary = bool(self.recent and self.recent[-1]['stage_id'] != link)\n"
    new_lines = ("            # An unresolved previous assignment cannot establish a transition.\n"
                 "            previous_id = self.recent[-1]['stage_id'] if self.recent else None\n"
                 "            boundary = previous_id != link if previous_id is not None else None\n")
    check(original.decode().replace(old_line, new_lines) == current.decode(),
          'Runtime difference includes changes beyond the documented boundary correction')
    return dict(status='passed', archive_sha256=digest(root / 'pilot-inputs.tar.gz'),
                original_runtime_sha256=old_hash, window_runtime_sha256=new_hash,
                prompt_change=False, difference=''.join(difflib.unified_diff(
                    original.decode().splitlines(True), current.decode().splitlines(True),
                    fromfile='archived-v1', tofile='window-v2')))


def aggregate(root, completed, windows, analysis, output):
    probes = probe_audit(completed)
    runtime = runtime_audit(root, completed, windows)
    full = load_lines(completed / 'full-short-results' / 'decisions.jsonl')
    win = load_lines(windows / 'decisions.jsonl')
    groups = {'full-short': full, 'windows': win}
    datasets = {name: load_lines(root / design / 'dataset.jsonl')
                for name, design in [('full-short', 'full-short'), ('windows', 'windows')]}
    identities = {name: {(s['trajectory_id'], s['step']) for s in rows}
                  for name, rows in datasets.items()}
    all_predictions = full + win
    costs = {}
    for name, predictions in groups.items():
        costs[name] = dict(steps=len(predictions), http_calls=len(predictions),
            input_tokens=sum(p['usage']['input_tokens'] for p in predictions),
            label_reads=sum(p['usage']['output_tokens'] for p in predictions),
            http_seconds=sum(p['seconds'] for p in predictions),
            latency_seconds=distribution([p['seconds'] for p in predictions]),
            input_tokens_per_step=distribution([p['usage']['input_tokens'] for p in predictions]))
    smoke = json.loads((completed / 'smoke-summary.json').read_text())
    reference_audit = {}
    started = json.loads((completed / 'full-short-results' / 'run.json').read_text())['started_at']
    for name in ('activity', 'boundary'):
        path = root / f'provisional-{name}-reference.json'
        reference = json.loads(path.read_text())
        check(reference['created_at'] < started, 'Reference was not frozen before baseline start')
        reference_audit[name] = dict(sha256=digest(path), declared_created_at=reference['created_at'],
                                    baseline_started_at=started, human_verified=False,
                                    chronology_note='Recorded timestamps, not a cryptographic external timestamp.')
    report = dict(status='completed', human_gold_accuracy=None, human_gold_boundary_f1=None,
        dataset=dict(evaluations=len(all_predictions), unique_raw_steps=len(identities['full-short'] | identities['windows']),
            overlap_raw_steps=len(identities['full-short'] & identities['windows']),
            windows=len({s['window_id'] for s in datasets['windows']}),
            trajectories_in_windows=len({s['trajectory_id'] for s in datasets['windows']}),
            complete_trajectories=1),
        costs=costs, paired_format_probe=probes, runtime_audit=runtime,
        reference_provenance=reference_audit,
        diagnostics_by_cohort={name: diagnostics(rows) for name, rows in groups.items()},
        diagnostics_all_evaluations=diagnostics(all_predictions),
        baseline_http_calls=len(all_predictions),
        baseline_input_tokens=sum(p['usage']['input_tokens'] for p in all_predictions),
        baseline_label_reads=sum(p['usage']['output_tokens'] for p in all_predictions),
        baseline_http_seconds=sum(p['seconds'] for p in all_predictions),
        baseline_latency_seconds=distribution([p['seconds'] for p in all_predictions]),
        smoke=dict(http_calls=len(smoke), input_tokens=sum(r['usage']['input_tokens'] for r in smoke),
            label_reads=sum(r['usage']['output_tokens'] for r in smoke),
            http_seconds=sum(r['seconds'] for r in smoke), cold_start_seconds=smoke[0]['seconds'],
            synthetic_failed_test_detected=False),
        all_observed_http_calls=len(all_predictions)+len(smoke)+probes['http_calls'],
        all_observed_input_tokens=sum(p['usage']['input_tokens'] for p in all_predictions)
            +sum(r['usage']['input_tokens'] for r in smoke)+probes['input_tokens'],
        all_observed_label_reads=sum(p['usage']['output_tokens'] for p in all_predictions)
            +sum(r['usage']['output_tokens'] for r in smoke)+probes['label_reads'],
        overhead_ratio=None, overhead_ratio_reason='Offline only; no concurrent timed main Agent',
        provisional_review=json.loads((analysis/'full-short-review/provisional-review.json').read_text()),
        audits={name: json.loads((analysis / f'{name}-audit-strict/audit.json').read_text()) for name in groups})
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'results.json', report)
    with (output / 'format-probes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step','variant','choice','label_mass',
            'argmax_is_label','seconds','input_tokens','label_reads'])
        writer.writeheader()
        for pair in probes['pairs']:
            for variant in ('original', 'explicit_letter'):
                row = pair[variant]
                writer.writerow(dict(step=pair['step'], variant=variant, choice=row['answer']['choice'],
                    **row['diagnostics'], seconds=row['seconds'], input_tokens=row['usage']['input_tokens'],
                    label_reads=row['usage']['output_tokens']))
    stage_rows = {}
    for name, predictions in groups.items():
        for p in normalize_boundaries(predictions):
            key = name, p['window_id'], p['stage_id']
            stage_rows.setdefault(key, []).append(p)
    with (output / 'stage-assignments.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cohort','window_id','stage_id','steps','activities','outcomes'])
        writer.writeheader()
        for (cohort, wid, sid), rows in stage_rows.items():
            writer.writerow(dict(cohort=cohort, window_id=wid, stage_id=sid,
                steps=json.dumps([p['step'] for p in rows]),
                activities=json.dumps([p['activity'] for p in rows]),
                outcomes=json.dumps([p['outcome'] for p in rows])))
    full_identity = {(p['trajectory_id'],p['step']):p for p in full}
    with (output / 'overlapping-steps.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['trajectory_id','step','window_id',
            'full_activity','window_activity','full_outcome','window_outcome',
            'full_stage_id','window_stage_id','stage_ids_not_comparable_across_memories'])
        writer.writeheader()
        for p in win:
            key=p['trajectory_id'],p['step']
            if key not in full_identity:
                continue
            f=full_identity[key]
            writer.writerow(dict(trajectory_id=key[0],step=key[1],window_id=p['window_id'],
                full_activity=f['activity'],window_activity=p['activity'],full_outcome=f['outcome'],
                window_outcome=p['outcome'],full_stage_id=f['stage_id'],window_stage_id=p['stage_id'],
                stage_ids_not_comparable_across_memories=True))
    print(json.dumps({k: v for k,v in report.items() if k in ('status','dataset','costs',
        'baseline_input_tokens','all_observed_http_calls','all_observed_input_tokens')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','completed','windows','analysis','output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.root,args.completed,args.windows,args.analysis,args.output)


if __name__ == '__main__':
    main()

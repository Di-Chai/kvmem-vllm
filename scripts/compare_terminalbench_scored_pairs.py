"""Audit all first attempts; compare quality only on jointly scored task pairs.

The frozen original comparator remains the authority for run/configuration
validation. This wrapper never retries, rescores, replaces results, or uses its
legacy all-attempt McNemar/GC-quality analyses when a score is unavailable.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import time


SCHEMA = 'terminalbench-scored-pairs-v1'
ANALYSIS_SCOPE = 'common_ordinary_binary_scored_tasks'
EVIDENCE_FILES = ('manifest.json', 'profile.json', 'job.json', 'agent-time-policy.json',
                  'results.json', 'execution-audit.json', 'formal-events/summary.json')
SELECTION_CAVEAT = ('Jointly scored tasks exclude failures of setup, infrastructure, or other execution stages. '
                    'This selected subset can differ systematically from the full task set; its quality difference '
                    'is not an estimate of complete 89-task accuracy or repeatable superiority.')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score_eligibility(row):
    reasons = []
    if row.get('status') != 'scored' or row.get('rewards') is None:
        reasons.append('unscored')
    else:
        reward = row['rewards'].get('reward') if isinstance(row['rewards'], dict) else None
        if type(reward) not in (int, float) or reward not in (0, 1):
            reasons.append('nonbinary_reward')
    if row.get('exception') is not None:
        reasons.append('exception_recorded')
    return dict(eligible=not reasons, reasons=reasons, status=row.get('status'),
                rewards=row.get('rewards'), exception=row.get('exception'))


def quality_analysis(baseline, candidate, exact_mcnemar):
    """Pure analysis shared with the monitor's independent result check."""
    require(set(baseline) == set(candidate), 'Quality task cohorts differ')
    included = []; excluded = []; pairs = dict(both_pass=[], both_fail=[],
                                             candidate_only_pass=[], baseline_only_pass=[])
    for task in sorted(baseline):
        sides = dict(baseline=score_eligibility(baseline[task]), candidate=score_eligibility(candidate[task]))
        if not all(side['eligible'] for side in sides.values()):
            excluded.append(dict(task=task, **sides))
            continue
        included.append(task)
        left, right = baseline[task]['rewards']['reward'] == 1, candidate[task]['rewards']['reward'] == 1
        key = 'both_pass' if left and right else 'candidate_only_pass' if right else 'baseline_only_pass' if left else 'both_fail'
        pairs[key].append(task)
    n = len(included); total = len(baseline)
    left = len(pairs['both_pass']) + len(pairs['baseline_only_pass'])
    right = len(pairs['both_pass']) + len(pairs['candidate_only_pass'])
    return dict(analysis_scope=ANALYSIS_SCOPE, task_count=n, attempted_tasks_per_arm=total,
        coverage_fraction=n/total if total else None, excluded_task_count=len(excluded),
        tasks=included, baseline_passed=left, candidate_passed=right,
        baseline_pass_rate=left/n if n else None, candidate_pass_rate=right/n if n else None,
        accuracy_delta_percentage_points=100*(right-left)/n if n else None,
        paired_task_lists=pairs, paired_counts={key: len(value) for key, value in pairs.items()},
        exact_mcnemar=exact_mcnemar(len(pairs['candidate_only_pass']), len(pairs['baseline_only_pass'])) if n else None,
        status='measured' if n else 'unavailable', selection_bias=SELECTION_CAVEAT), excluded


def raw_outcome_evidence(run):
    output = {}; root = run['root']
    for name, row in run['outcomes'].items():
        relative = 'jobs/formal/' + row['trial'] + '/result.json'
        path = root/relative
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), 'Raw result outside its run')
        raw = read(path); exception = raw.get('exception_info')
        require(exception is None or (isinstance(exception, dict) and isinstance(exception.get('exception_type'), str)
                                     and exception['exception_type']), 'Malformed original exception evidence')
        rewards = (raw.get('verifier_result') or {}).get('rewards')
        error = exception['exception_type'] if exception else None
        status = 'scored' if rewards is not None else 'agent_timeout' if error == 'AgentTimeoutError' else 'error'
        require(rewards is not None or exception is not None, 'Unscored attempt lacks an original exception')
        require(raw.get('trial_name') == row['trial'] and raw.get('task_name', '').removeprefix('terminal-bench/') == name
                and rewards == row.get('rewards') and error == row.get('exception') and status == row['status'],
                'Original raw outcome differs: ' + name)
        require(row.get('passed') is bool(rewards and rewards.get('reward') == 1),
                'Original raw reward/pass mismatch: ' + name)
        event_relative = 'formal-events/' + row['trial'] + '.json'
        require(read(root/event_relative) == raw, 'Raw and lifecycle result differ: ' + name)
        output[name] = dict(trial=row['trial'], path=relative, sha256=sha(path),
            event_path=event_relative, event_sha256=sha(root/event_relative), exception_info=exception)
    return output


def load_comparator(path):
    path = Path(path).resolve()
    require(path.is_file(), 'Frozen comparator is missing')
    spec = importlib.util.spec_from_file_location('_frozen_terminalbench_comparator', path)
    require(spec is not None and spec.loader is not None, 'Cannot load frozen comparator')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def compare(baseline_root, candidate_root, comparator_path=None):
    baseline_root, candidate_root = Path(baseline_root).resolve(), Path(candidate_root).resolve()
    comparator_path = Path(comparator_path or candidate_root.parent/'compare_terminalbench_policies.py').resolve()
    original = load_comparator(comparator_path)
    runs = dict(baseline=original.load_run(baseline_root), candidate=original.load_run(candidate_root))
    validation = original.verify_pair(runs['baseline'], runs['candidate'])
    require(validation.get('passed') is True, 'Frozen comparator validation did not pass')
    raw = {label: raw_outcome_evidence(run) for label, run in runs.items()}
    quality, excluded = quality_analysis(runs['baseline']['outcomes'], runs['candidate']['outcomes'], original.exact_mcnemar)
    sides = {}; operational = {}
    for label, run in runs.items():
        metrics = deepcopy(run['metrics']); outcomes = run['outcomes']
        unscored = [name for name, row in outcomes.items() if row['status'] != 'scored']
        metrics.update(unscored=len(unscored), unscored_tasks=sorted(unscored),
            pass_at_1_scope='Operational successes divided by all 89 first attempts; not scored-only model accuracy')
        sides[label] = dict(root=str(run['root']), policy='lru' if label == 'baseline' else 'thinking redundancy with LRU fallback', **metrics)
        operational[label] = dict(attempts=len(outcomes), passed=metrics['passed'], scored=metrics['scored'],
            unscored=len(unscored), errors=metrics['errors'], task_timeouts=metrics['task_timeouts'],
            success_fraction_all_attempts=metrics['passed']/len(outcomes),
            unscored_tasks=[dict(task=name, outcome=original.outcome_identity(outcomes[name]),
                                 exception_info=raw[label][name]['exception_info']) for name in sorted(unscored)])
    scoring_complete = quality['task_count'] == 89
    return dict(schema=SCHEMA, created_at=time.time(), passed=True, attempts_complete=True,
        scoring_complete=scoring_complete, analysis_scope=ANALYSIS_SCOPE,
        scope='All 89 original attempts retained in each arm; primary quality comparison uses only jointly ordinary binary-scored pairs',
        **sides, operational=dict(scope='First-attempt operational outcomes; unscored attempts remain explicit', **operational),
        quality=quality, excluded_tasks=excluded, validation=validation,
        agent_time_limit=validation['agent_time_limit'], official_timing_comparable=validation['official_timing_comparable'],
        timing_scope=validation['timing_scope'],
        gc_triggered_subsets=dict(status='disabled', reason='The legacy all-attempt GC quality subsets can include unavailable scores; this wrapper does not use them. Per-task and total GC quantities remain in each arm.'),
        evidence_sha256={label: {name: sha(run['root']/name) for name in EVIDENCE_FILES} for label, run in runs.items()},
        raw_outcome_evidence=raw,
        analysis_implementation=dict(wrapper=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                                     comparator=dict(path=str(comparator_path), sha256=sha(comparator_path))),
        caveats=[validation['timing_scope'], SELECTION_CAVEAT,
            'Each task has one attempted trajectory per policy. Sampling and batching can change trajectories; one paired run does not establish repeatable quality superiority.',
            'The full-attempt operational success fraction is reported separately and must not be substituted for jointly-scored quality.',
            'No missing reward is imputed, no existing result is replaced, and no Agent or verifier is rerun.',
            'Exact McNemar is unavailable when no jointly scored pairs exist; a non-significant p-value does not establish equivalence.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--comparator', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Never replace an existing analysis artifact')
    result = compare(args.baseline, args.candidate, args.comparator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(dict(passed=result['passed'], attempts_complete=True,
        scoring_complete=result['scoring_complete'], jointly_scored_tasks=result['quality']['task_count'],
        unscored={label: result[label]['unscored'] for label in ('baseline', 'candidate')}, output=str(args.output))))


if __name__ == '__main__':
    main()

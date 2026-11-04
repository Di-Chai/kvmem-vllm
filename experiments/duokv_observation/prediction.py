"""Leakage-controlled grouped out-of-fold evaluation of five simple rules."""
from bisect import bisect_right
from collections import defaultdict
import hashlib
import math
import random
from core import words
from analysis_stats import mean, ratio, cluster_interval, WINDOWS

BASELINES = ['random', 'lru', 'frequency', 'query_relevance', 'execution_state']
TARGETS = ['explicit', 'semantic', 'high_confidence_dependency']


def stable_uniform(text):
    return int.from_bytes(hashlib.blake2s(text.encode(), digest_size=8).digest(), 'big') / 2**64


def task_folds(index):
    out = {}
    for family in ('deepswe', 'terminalbench'):
        tasks = sorted({it['task_id'] for it in index if it['family'] == family},
                       key=lambda t: hashlib.sha256(t.encode()).hexdigest())
        for n, task in enumerate(tasks):
            out[(family, task)] = n % 5
    return out


def prediction_examples(item, units, edges, events, folds, stride=8, sample_size=12):
    past = [u for u in units if u['main_eligible']]
    uses = defaultdict(lambda: defaultdict(set))
    for e in edges:
        uses[e['source_unit_id']][e['level']].add(e['target_step'])
    uses = {u: {k: sorted(v) for k, v in d.items()} for u, d in uses.items()}
    texts = {u['unit_id']: words(u['text']) for u in past}
    out = []
    final = item['main_steps']
    for t in range(1, final, stride):
        candidates = [u for u in past if u['step'] <= t]
        # Fixed-size, uniform-without-replacement sample based only on IDs and t.
        rng = random.Random(f'{item["trajectory_id"]}:{t}:20261103')
        chosen = rng.sample(candidates, min(sample_size, len(candidates)))
        query_units = [u for u in past if u['step'] == t and u['role'] != 'assistant']
        if not query_units:
            # Available current assistant text only, never turn t+1's query.
            query_units = [u for u in past if u['step'] == t]
        query = set().union(*(texts[u['unit_id']] for u in query_units)) if query_units else set()
        known_events = [e for e in events if e['step'] <= t]
        last_event = max(known_events, key=lambda e: e['step']) if known_events else None
        current_tools=sorted({u.get('tool') for u in past if u['step']==t and u.get('tool')})
        recent_kind=last_event['event_type'] if last_event else None
        phase=('validation' if recent_kind in {'test_success','test_failure','build_success','build_failure'}
               else 'modification' if recent_kind=='code_modification' else 'exploration_or_other')
        for u in chosen:
            by_level = uses.get(u['unit_id'], {})
            observed = sorted({v for level in ('explicit', 'semantic') for v in by_level.get(level, []) if v <= t})
            last = observed[-1] if observed else u['step']
            age = t - last
            overlap = len(texts[u['unit_id']] & query) / max(1, len(texts[u['unit_id']] | query))
            recent_failure = bool(last_event and last_event['event_type'] in {'test_failure', 'build_failure'} and t - last_event['step'] <= 4)
            phase_match=(phase=='modification' and u['subtype']=='code') or (phase=='validation' and u['subtype']=='error_log')
            state = min(1., .35 * overlap + .2 / (1 + age) + .15 * min(len(observed) / 3, 1) +
                        .1 * (u['content_type'] == 'task_instruction') + .1 * (recent_failure and u['subtype'] == 'error_log') + .1*phase_match)
            scores = dict(random=stable_uniform(u['unit_id'] + ':' + str(t)), lru=1 / (1 + age),
                          frequency=min(1., math.log1p(len(observed)) / math.log(21)),
                          query_relevance=overlap, execution_state=state)
            for k in WINDOWS:
                if t + k > final:
                    continue
                labels = {level: int(any(t < v <= t + k for v in by_level.get(level, []))) for level in ('explicit', 'semantic')}
                # No confirmed dependencies in this offline rule-only phase.
                labels['high_confidence_dependency'] = None
                out.append(dict(example_id=f'{u["unit_id"]}@{t}+{k}', family=u['family'], task_id=u['task_id'],
                    trajectory_id=u['trajectory_id'], seed=u['seed'], attempt_source=u['attempt_source'],
                    unit_id=u['unit_id'], checkpoint_step=t, window=k, content_type=u['content_type'],
                    fold=folds[(u['family'], u['task_id'])], source_birth_step=u['step'],
                    features_max_step=t, last_observed_use_step=last, prior_reuse_count=len(observed),
                    recent_event_type=last_event['event_type'] if last_event else None,
                    execution_phase=phase,current_tools=current_tools,task_status='no_independently_verified_completion',
                    candidate_population=len(candidates), inclusion_probability=len(chosen)/len(candidates),
                    query_unit_ids=[x['unit_id'] for x in query_units], labels=labels, scores=scores))
    return out


def average_precision(labels, scores):
    positives = sum(labels)
    if not positives:
        return None
    buckets = defaultdict(lambda: [0, 0])
    for y, s in zip(labels, scores):
        buckets[s][0] += y
        buckets[s][1] += 1
    tp = n = 0
    ap = 0.
    for s, (p, count) in sorted(buckets.items(), reverse=True):
        tp += p
        n += count
        ap += p / positives * tp / n
    return ap


def metrics(rows):
    if not rows:
        return dict(n=0, positives=0, prevalence=None, precision=None, recall=None, f1=None,
                    pr_auc_average_precision=None, brier=None, ece_10_bins=None,
                    status='no_identifiable_labels')
    y = [r['label'] for r in rows]
    pred = [r['predicted'] for r in rows]
    tp = sum(a and b for a,b in zip(y,pred)); pp = sum(pred); positives = sum(y)
    precision = tp / pp if pp else 0.
    recall = tp / positives if positives else None
    f1 = 2*precision*recall/(precision+recall) if recall is not None and precision+recall else (0. if positives else None)
    bins = defaultdict(list)
    for r in rows:
        bins[min(9, int(r['probability'] * 10))].append(r)
    ece = sum(len(v)/len(rows)*abs(mean(r['probability'] for r in v)-mean(r['label'] for r in v)) for v in bins.values())
    return dict(n=len(rows), positives=positives, prevalence=positives/len(rows), predicted_positives=pp,
                true_positives=tp, precision=precision, recall=recall, f1=f1,
                pr_auc_average_precision=average_precision(y, [r['score'] for r in rows]),
                brier=mean((r['probability']-r['label'])**2 for r in rows), ece_10_bins=ece,
                status='ok' if positives else 'no_positive_test_labels')


def evaluate(examples):
    forecasts, summary, calibration, models = [], [], [], []
    groups = defaultdict(list)
    for ex in examples:
        groups[(ex['family'], ex['window'])].append(ex)
    for (family, window), group in sorted(groups.items()):
        for target in TARGETS:
            labeled = [e for e in group if e['labels'][target] is not None]
            for baseline in BASELINES:
                combined = []
                for fold in range(5):
                    train = [e for e in labeled if e['fold'] != fold]
                    test = [e for e in labeled if e['fold'] == fold]
                    if not train or not test:
                        continue
                    train_tasks = sorted({e['task_id'] for e in train})
                    test_tasks = sorted({e['task_id'] for e in test})
                    assert not set(train_tasks) & set(test_tasks)
                    score_counts = defaultdict(lambda: [0,0])
                    bins = [[0,0] for _ in range(10)]
                    for e in train:
                        s, y = e['scores'][baseline], e['labels'][target]
                        score_counts[s][0] += y; score_counts[s][1] += 1
                        b = min(9, int(s*10)); bins[b][0] += y; bins[b][1] += 1
                    prevalence = sum(e['labels'][target] for e in train) / len(train)
                    probs = [(p + 2*prevalence)/(n+2) for p,n in bins]
                    # Threshold chosen on training labels only, tie prefers
                    # higher threshold (fewer selected units).
                    positives = sum(e['labels'][target] for e in train)
                    tp = pp = 0
                    best_f1, threshold = -1., 2.
                    for s, (p,n) in sorted(score_counts.items(), reverse=True):
                        tp += p; pp += n
                        f1 = 2*tp/(pp+positives) if pp+positives else 0.
                        if positives and f1 > best_f1:
                            best_f1, threshold = f1, s
                    models.append(dict(family=family, window=window, target=target, baseline=baseline,
                        held_out_fold=fold, train_tasks=train_tasks, test_tasks=test_tasks,
                        training_examples=len(train), test_examples=len(test), threshold=threshold,
                        training_prevalence=prevalence, calibration_bin_probabilities=probs))
                    for ex in test:
                        score = ex['scores'][baseline]
                        row = dict(example_id=ex['example_id'], family=family, task_id=ex['task_id'],
                            trajectory_id=ex['trajectory_id'], content_type=ex['content_type'],
                            window=window, target=target, baseline=baseline, fold=fold,
                            label=ex['labels'][target], score=score,
                            probability=probs[min(9,int(score*10))], predicted=int(score >= threshold),
                            threshold=threshold)
                        combined.append(row)
                forecasts.extend(combined)
                for typ in ['ALL'] + sorted({e['content_type'] for e in combined}):
                    rows = combined if typ == 'ALL' else [r for r in combined if r['content_type'] == typ]
                    by_task = defaultdict(list)
                    for r in rows:
                        by_task[r['task_id']].append(r)
                    result = dict(family=family, window=window, target=target, baseline=baseline, content_type=typ,
                                  tasks=len(by_task), **metrics(rows))
                    if typ == 'ALL':
                        result.update(cluster_interval([metrics(v)['pr_auc_average_precision'] for v in by_task.values()]))
                    summary.append(result)
                for b in range(10):
                    rr = [r for r in combined if min(9,int(r['probability']*10)) == b]
                    calibration.append(dict(family=family, window=window, target=target, baseline=baseline,
                        probability_bin=b, examples=len(rr), mean_probability=mean(r['probability'] for r in rr),
                        observed_fraction=mean(r['label'] for r in rr)))
    return forecasts, summary, calibration, models

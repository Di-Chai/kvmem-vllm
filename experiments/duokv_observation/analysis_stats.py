"""Descriptive estimators. Unknowns remain null; task is the clustering unit."""
from collections import Counter, defaultdict
from bisect import bisect_left
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics as st

LEVELS = ['explicit', 'semantic', 'potential_dependency']
WINDOWS = [1, 4, 16]
LONG = 16


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def csv_value(v):
    return json.dumps(v, ensure_ascii=False, separators=(',', ':')) if isinstance(v, (list, dict, set)) else v


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(v) for k, v in row.items()})


def ratio(n, d):
    return n / d if d else None


def mean(values):
    vals = [x for x in values if x is not None]
    return st.mean(vals) if vals else None


def cluster_interval(values, seed=20261103, iterations=1000):
    """Equal task weights; percentile bootstrap over tasks, never seeds."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return dict(task_macro_mean=mean(vals), ci_low=None, ci_high=None, cluster_tasks=len(vals))
    rng = random.Random(seed)
    samples = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(iterations))
    return dict(task_macro_mean=mean(vals), ci_low=samples[int(.025 * iterations)],
                ci_high=samples[int(.975 * iterations)], cluster_tasks=len(vals))


def km_curve(observations):
    """(duration, event_observed), simultaneous events precede censor removals."""
    grouped = defaultdict(lambda: [0, 0])
    for duration, event in observations:
        grouped[duration][0 if event else 1] += 1
    risk, survival = len(observations), 1.0
    rows = []
    for duration, (events, censored) in sorted(grouped.items()):
        before = risk
        if risk:
            survival *= 1 - events / risk
        rows.append(dict(distance=duration, at_risk=before, observed_reuse=events,
                         censored=censored, survival=survival, cumulative_reuse=1 - survival))
        risk -= events + censored
    return rows


def curve_at(curve, distance):
    values = [r['cumulative_reuse'] for r in curve if r['distance'] <= distance]
    return values[-1] if values else 0.0


def future_stats(units, edges, index_by_id, include_retry=False, variant='main'):
    by_source = defaultdict(lambda: defaultdict(set))
    for e in edges:
        by_source[e['source_unit_id']][e['level']].add(e['target_turn'] if include_retry else e['target_step'])
    unit_rows, groups, gaps = [], defaultdict(list), []
    for u in units:
        if not (u['content_type'] in {'thinking', 'analysis', 'plan', 'tool_call', 'tool_result', 'terminal_observation', 'execution_status', 'task_instruction', 'assistant_text', 'parser_feedback', 'parser_warning'} if include_retry else u['main_eligible']):
            continue
        it = index_by_id[u['trajectory_id']]
        if variant == 'exclude_fresh_restart' and u['attempt_source'] == 'fresh_restart':
            continue
        if variant == 'exclude_incomplete' and not it['final_saved_response_available']:
            continue
        birth = u['turn_id'] if include_retry else u['step']
        end = it['saved_assistant_turns'] if include_retry else it['main_steps']
        followup = end - birth
        for level in LEVELS:
            uses = sorted(by_source[u['unit_id']][level])
            first = uses[0] - birth if uses else None
            # Dormancy is interrupted by either observed Explicit or Semantic
            # reuse, even when the target event is of only one of those levels.
            observed_uses = sorted(by_source[u['unit_id']]['explicit'] | by_source[u['unit_id']]['semantic']) if level in {'explicit', 'semantic'} else uses
            intervals = []
            for target in uses:
                pos = bisect_left(observed_uses, target)
                intervals.append(target - (observed_uses[pos-1] if pos else birth))
            row = dict(variant=variant, trajectory_id=u['trajectory_id'], task_id=u['task_id'],
                       family=u['family'], seed=u['seed'], unit_id=u['unit_id'],
                       content_type=u['content_type'], subtype=u['subtype'], level=level,
                       birth_step=birth, followup_steps=followup, reuse_frequency=len(uses),
                       first_reuse_distance=first, first_event_observed=bool(uses),
                       duration=first if uses else followup,
                       first_any_observed_reuse_distance=observed_uses[0]-birth if observed_uses else None,
                       max_observed_reuse_gap=max(intervals) if intervals else None,
                       last_use_step=uses[-1] if uses else birth,
                       terminal_censor_reason='trace_end_missing_final_response' if not it['final_saved_response_available'] else 'saved_trace_end_no_success_inference')
            unit_rows.append(row)
            for typ in ('ALL', u['content_type']):
                groups[(u['family'], typ, level)].append(row)
            for distance, target in zip(intervals, uses):
                gaps.append(dict(variant=variant, family=u['family'], task_id=u['task_id'],
                                 trajectory_id=u['trajectory_id'], unit_id=u['unit_id'],
                                 content_type=u['content_type'], subtype=u['subtype'], level=level,
                                 target_step=target, gap_since_last_observed_use=distance,
                                 long_range=distance >= LONG))
    summaries, curves, dormant = [], [], []
    for (family, typ, level), rows in groups.items():
        curve = km_curve([(r['duration'], r['first_event_observed']) for r in rows])
        curves.extend(dict(variant=variant, family=family, content_type=typ, level=level, **r) for r in curve)
        tasks = defaultdict(list)
        for r in rows:
            tasks[r['task_id']].append(r)
        reused = sum(r['first_event_observed'] for r in rows)
        observed_gap_rows = [r for r in rows if r['max_observed_reuse_gap'] is not None]
        for k in WINDOWS:
            # Fixed full followup cohort: no early positives with truncated
            # negatives mixed into the denominator.
            full = [r for r in rows if r['followup_steps'] >= k]
            hits = sum(r['first_reuse_distance'] is not None and r['first_reuse_distance'] <= k for r in full)
            task_rates = []
            for task_rows in tasks.values():
                rr = [r for r in task_rows if r['followup_steps'] >= k]
                if rr:
                    task_rates.append(sum(r['first_reuse_distance'] is not None and r['first_reuse_distance'] <= k for r in rr) / len(rr))
            summaries.append(dict(variant=variant, family=family, content_type=typ, level=level, window=k,
                units=len(rows), trajectories=len({r['trajectory_id'] for r in rows}), tasks=len(tasks),
                observed_reused_units=reused, observed_reuse_ratio=ratio(reused, len(rows)),
                never_observed_reuse_units=len(rows) - reused, never_observed_reuse_ratio=ratio(len(rows) - reused, len(rows)),
                right_censored_first_reuse=len(rows) - reused, zero_followup_units=sum(r['followup_steps'] == 0 for r in rows),
                full_window_units=len(full), excluded_incomplete_window=len(rows) - len(full),
                full_window_hits=hits, future_reuse_probability=ratio(hits, len(full)),
                km_cumulative_reuse=curve_at(curve, k),
                median_first_reuse_distance=st.median(r['first_reuse_distance'] for r in rows if r['first_event_observed']) if reused else None,
                mean_reuse_frequency=mean(r['reuse_frequency'] for r in rows),
                long_range_threshold=LONG, long_range_reused_units=sum(r['max_observed_reuse_gap'] >= LONG for r in observed_gap_rows),
                long_range_ratio_among_reused=ratio(sum(r['max_observed_reuse_gap'] >= LONG for r in observed_gap_rows), reused),
                **cluster_interval(task_rates)))
        for dormant_age in (4, 8, 16):
            eligible = [r for r in rows if r['followup_steps'] >= dormant_age + 4 and
                        (r['first_any_observed_reuse_distance'] is None or r['first_any_observed_reuse_distance'] > dormant_age)]
            hits = sum(r['first_reuse_distance'] is not None and r['first_reuse_distance'] <= dormant_age + 4 for r in eligible)
            dormant.append(dict(variant=variant, family=family, content_type=typ, level=level,
                                dormant_steps=dormant_age, future_window=4, eligible_units=len(eligible),
                                future_reused_units=hits, conditional_reuse_probability=ratio(hits, len(eligible)),
                                definition='No detected reuse since birth through landmark; entire future window observable'))
    return unit_rows, summaries, curves, gaps, dormant


def event_stats(events, relations, units):
    groups = defaultdict(list)
    for r in relations:
        groups[(r['family'], r['event_type'])].append(r)
    out = []
    for (family, typ), rows in groups.items():
        counts = Counter(r['relation'] for r in rows)
        known = len(rows) - counts['uncertain']
        task_did = defaultdict(list)
        for r in rows:
            if r['did'] is not None:
                task_did[r['task_id']].append(r['did'])
        out.append(dict(family=family, event_type=typ,
            observed_events=sum(e['family'] == family and e['event_type'] == typ for e in events),
            assessed_event_source_pairs=len(rows), unique_source_units=len({r['source_unit_id'] for r in rows}),
            fully_superseded=counts['fully_superseded'], partially_superseded=counts['partially_superseded'],
            not_superseded=counts['not_superseded'], uncertain=counts['uncertain'],
            fully_superseded_ratio=counts['fully_superseded'] / len(rows),
            partial_supersession_ratio=counts['partially_superseded'] / len(rows),
            superseded_ratio=(counts['fully_superseded'] + counts['partially_superseded']) / len(rows),
            superseded_ratio_excluding_uncertain=ratio(counts['fully_superseded'] + counts['partially_superseded'], known),
            post_event_reused_pairs=sum(r['post_event_reuse'] for r in rows),
            post_event_reuse_ratio=sum(r['post_event_reuse'] for r in rows) / len(rows),
            lexical_residual_measured_pairs=sum(r['residual_lexical_fraction'] is not None for r in rows),
            mean_residual_lexical_fraction=mean(r['residual_lexical_fraction'] for r in rows),
            semantic_residual_fraction=None, event_to_decay_latency=None,
            fully_observable_window_pairs=sum(r['full_window_eligible'] for r in rows),
            matched_control_pairs=sum(r['control'] is not None for r in rows),
            mean_pre_reuse_rate=mean(r['pre_reuse_steps'] / r['window'] for r in rows if r['control']),
            mean_post_reuse_rate=mean(r['post_reuse_steps'] / r['window'] for r in rows if r['control']),
            **cluster_interval([mean(v) for v in task_did.values()])))
    return out


def subtype_stats(unit_stats):
    groups=defaultdict(list)
    for row in unit_stats:
        groups[(row['family'],row['subtype'],row['level'])].append(row)
    out=[]
    for (family,subtype,level),rows in sorted(groups.items()):
        reused=[r for r in rows if r['first_event_observed']]
        long=sum(r['max_observed_reuse_gap']>=LONG for r in reused)
        for k in WINDOWS:
            complete=[r for r in rows if r['followup_steps']>=k]
            hit=sum(r['first_reuse_distance'] is not None and r['first_reuse_distance']<=k for r in complete)
            out.append(dict(family=family,subtype=subtype,level=level,window=k,units=len(rows),
                observed_reused_units=len(reused),long_range_reused_units=long,
                long_range_ratio_among_reused=ratio(long,len(reused)),
                median_first_reuse_distance=st.median(r['first_reuse_distance'] for r in reused) if reused else None,
                full_window_units=len(complete),full_window_hits=hit,future_reuse_probability=ratio(hit,len(complete)),
                excluded_short_followup=len(rows)-len(complete),
                classification_status='structural_regex_proxy_not_semantic_ground_truth'))
    return out


def cross_seed(units, edges, relations, index):
    used = {e['source_unit_id'] for e in edges if e['level'] in {'explicit', 'semantic'}}
    superseded = {r['source_unit_id'] for r in relations if r['relation'] in {'fully_superseded', 'partially_superseded'}}
    event_kinds = defaultdict(set)
    exposed_kinds = defaultdict(set)
    relation_states=defaultdict(set)
    event_states=defaultdict(lambda:defaultdict(set))
    for r in relations:
        exposed_kinds[r['source_unit_id']].add(r['event_type'])
        relation_states[r['source_unit_id']].add(r['relation'])
        event_states[r['source_unit_id']][r['event_type']].add(r['relation'])
        if r['relation'] in {'fully_superseded', 'partially_superseded'}:
            event_kinds[r['source_unit_id']].add(r['event_type'])
    align = defaultdict(lambda: defaultdict(list))
    for u in units:
        if u['family'] != 'deepswe' or not u['main_eligible']:
            continue
        anchors = list(u['entities'])
        if u['content_type'] == 'task_instruction':
            anchors += ['constraint:' + hashlib.sha256(' '.join(u['text'].split()).encode()).hexdigest()[:20]]
        for anchor in anchors:
            align[(u['task_id'], anchor)][u['seed']].append(u)
    rows, event_rows = [], []
    for (task, anchor), seeds in sorted(align.items()):
        present = sorted(seeds)
        reused = sorted(seed for seed, us in seeds.items() if any(u['unit_id'] in used for u in us))
        supers = sorted(seed for seed, us in seeds.items() if any(u['unit_id'] in superseded for u in us))
        def known_state(states):
            if states & {'fully_superseded','partially_superseded'}:
                return True
            if states and states <= {'not_superseded'}:
                return False
            return None
        states_by_seed={s:known_state(set().union(*(relation_states[u['unit_id']] for u in us))) for s,us in seeds.items()}
        known_seeds=sorted(s for s,state in states_by_seed.items() if state is not None)
        eligible = len(present) == 4
        rows.append(dict(task_id=task, alignment_anchor=anchor, alignment_kind=anchor.split(':')[0],
            equivalence='same_entity_or_exact_constraint_proxy_not_semantic_equivalence',
            present_seeds=present, reused_seeds=reused, superseded_seeds=supers,
            present_seed_count=len(present), reused_seed_count=len(reused),
            absent_seeds_are_unknown=True, four_seed_eligible=eligible,
            stable_demand=eligible and len(reused) == 4,
            trajectory_dependent_demand=eligible and 0 < len(reused) < 4,
            reuse_agreement=ratio(sum((a in reused) == (b in reused) for i, a in enumerate(present) for b in present[i+1:]), len(present) * (len(present)-1) // 2),
            supersession_agreement=ratio(sum(states_by_seed[a]==states_by_seed[b] for i,a in enumerate(known_seeds) for b in known_seeds[i+1:]),len(known_seeds)*(len(known_seeds)-1)//2),
            supersession_known_seeds=known_seeds,supersession_unknown_seeds=sorted(set(present)-set(known_seeds)),
            event_types_by_seed={str(s): sorted(set().union(*(event_kinds[u['unit_id']] for u in us))) for s, us in seeds.items()},
            fresh_restart_seeds=sorted(s for s, us in seeds.items() if any(u['attempt_source'] == 'fresh_restart' for u in us)),
            unit_ids_by_seed={str(s): [u['unit_id'] for u in us] for s, us in seeds.items()}))
        exposed={s:set().union(*(exposed_kinds[u['unit_id']] for u in us)) for s,us in seeds.items()}
        supers_by_seed={s:set().union(*(event_kinds[u['unit_id']] for u in us)) for s,us in seeds.items()}
        for kind in sorted(set().union(*exposed.values())):
            kind_states={s:known_state(set().union(*(event_states[u['unit_id']][kind] for u in us))) for s,us in seeds.items()}
            comparable=sorted(s for s in seeds if kind in exposed[s] and kind_states[s] is not None)
            supers_seeds=sorted(s for s in comparable if kind in supers_by_seed[s])
            pairs=len(comparable)*(len(comparable)-1)//2
            event_rows.append(dict(task_id=task,alignment_anchor=anchor,event_type=kind,
                exposed_seeds=sorted(s for s in seeds if kind in exposed[s]),known_state_seeds=comparable,
                unknown_state_seeds=sorted(s for s in seeds if kind in exposed[s] and kind_states[s] is None),
                superseded_seeds=supers_seeds,comparable_seed_pairs=pairs,
                comparable=len(comparable)>=2,
                event_specific_consistency=ratio(sum((a in supers_seeds)==(b in supers_seeds)
                    for i,a in enumerate(comparable) for b in comparable[i+1:]),pairs),
                exposure_definition='At least one assessed event-source relation for this anchor and event type',
                supersession_rule_status='automated_candidate'))
    taskrows = []
    for task in sorted({it['task_id'] for it in index if it['family'] == 'deepswe'}):
        allrows = [r for r in rows if r['task_id'] == task]
        rr = [r for r in allrows if r['four_seed_eligible']]
        taskrows.append(dict(task_id=task, total_anchor_groups=len(allrows), four_seed_groups=len(rr),
            excluded_missing_seed_groups=len(allrows)-len(rr),
            stable_demand_ratio=ratio(sum(r['stable_demand'] for r in rr), len(rr)),
            trajectory_dependent_demand_ratio=ratio(sum(r['trajectory_dependent_demand'] for r in rr), len(rr)),
            reuse_consistency=mean(r['reuse_agreement'] for r in rr),
            supersession_consistency=mean(r['supersession_agreement'] for r in rr),
            event_specific_supersession_consistency=mean(r['event_specific_consistency'] for r in event_rows if r['task_id']==task and r['comparable']),
            fresh_restart_task=any(r['fresh_restart_seeds'] for r in allrows)))
    return rows, taskrows, event_rows


def rank_correlation(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        rr = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and values[order[j]] == values[order[i]]:
                j += 1
            for n in order[i:j]:
                rr[n] = (i + j - 1) / 2
            i = j
        return rr
    xx, yy = ranks(x), ranks(y)
    mx, my = mean(xx), mean(yy)
    return sum((a-mx)*(b-my) for a,b in zip(xx,yy)) / (sum((a-mx)**2 for a in xx)*sum((b-my)**2 for b in yy))**.5


def kv_analysis(root, units, edges, index_by_id):
    from core import read_json, read_jsonl
    observations = read_json(root / 'analysis/retention/retention-observations.json')
    snapshots = list(read_jsonl(root / 'evidence/selected-capture-snapshots.jsonl'))
    trends = list(csv.DictReader((root / 'analysis/retention/block-trends.csv').open()))
    assert len(snapshots) == 5 and len(trends) == 299
    reused = defaultdict(set)
    for e in edges:
        if e['level'] in {'explicit', 'semantic'}:
            reused[e['source_unit_id']].add(e['target_step'])
    data, correlations = [], []
    for trace in observations['traces']:
        tid = ('deepswe/seed-1001/' if trace['family'] == 'deepswe' else 'terminalbench/') + trace['task_id']
        uu = [u for u in units if u['trajectory_id'] == tid and u['main_eligible']]
        by_key = defaultdict(list)
        for u in uu:
            kind = 'command_request' if trace['family'] == 'terminalbench' and u['content_type'] == 'tool_call' else u['content_type']
            by_key[(u['turn_id'], kind)].append(u)
        block_groups = defaultdict(list)
        for b in trace['blocks']:
            block_groups[(b['birth_turn'], b['kind'])].append(b)
        turn_steps = {u['turn_id']: u['step'] for u in units if u['trajectory_id'] == tid}
        for (birth, kind), blocks in block_groups.items():
            matching = by_key[(birth, kind)]
            for i, snap in enumerate(trace['snapshots']):
                eligible = all(b['observations'][i]['eligible'] for b in blocks)
                total = sum(b['tokens'] for b in blocks)
                kept = sum(b['observations'][i]['selected_tokens'] for b in blocks) if eligible else None
                step = turn_steps.get(snap['turn'])
                full = step is not None and step + 4 <= index_by_id[tid]['main_steps']
                future = sum(any(step < t <= step + 4 for t in reused[u['unit_id']]) for u in matching) if full else None
                data.append(dict(trajectory_id=tid, task_id=trace['task_id'], family=trace['family'],
                    birth_turn=birth, kind=kind, snapshot_turn=snap['turn'], snapshot_step=step,
                    block_ids=[b['id'] for b in blocks], semantic_unit_ids=[u['unit_id'] for u in matching],
                    mapping='turn_and_content_type_aggregate_only', exact_character_to_token_mapping=False,
                    blocks=len(blocks), semantic_units=len(matching), tokens=total, eligible=eligible,
                    selected_fraction=kept / total if eligible else None,
                    overlaps_fixed_sink=any(b['overlaps_fixed_sink'] for b in blocks),
                    future_window=4, full_future_window=full, future_reused_units=future,
                    future_reuse_fraction=ratio(future, len(matching)) if future is not None else None))
        for sink in (False, True):
            rr = [r for r in data if r['trajectory_id'] == tid and r['eligible'] and r['future_reuse_fraction'] is not None and (sink or not r['overlaps_fixed_sink'])]
            correlations.append(dict(trajectory_id=tid, include_sink=sink, matched_aggregate_observations=len(rr),
                snapshots=len(trace['snapshots']), spearman=rank_correlation([r['selected_fraction'] for r in rr], [r['future_reuse_fraction'] for r in rr]),
                p_value=None, independent_sample_size=None,
                caveat='Repeated sparse snapshots; descriptive only, no causal or population inference'))
    return data, correlations

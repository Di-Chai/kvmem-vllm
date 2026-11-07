"""Fresh DeepSWE redundancy census and frozen 24-candidate graph pilot."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'duokv_observation'))
from core import normalize_trace, resolve
from protocol import canonical, digest, graph, history, rankings, observed_failure, scaffold_overlap


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def jsonl(path, rows):
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def run(bundle, output, max_characters):
    output.mkdir(parents=True, exist_ok=False)
    (output/'snapshots').mkdir()
    index = json.loads((bundle/'index.json').read_text())
    source_hashes = {'index.json': digest((bundle/'index.json').read_text())}
    items = [r for r in index if r['family'] == 'deepswe']
    if len(items) != 64:
        raise ValueError('Expected existing 16-task / four-seed DeepSWE bundle')
    census, pools, snapshots, exclusions = [], defaultdict(list), [], []
    removed_scaffolding = 0
    begun = time.perf_counter()

    def add_snapshot(item, prefix, sid, left_censored=False):
        t0 = time.perf_counter(); edges = graph(prefix); build_s = time.perf_counter()-t0
        t0 = time.perf_counter(); scores = rankings(prefix, edges); rank_s = time.perf_counter()-t0
        full, spans = history(prefix)
        evidence = defaultdict(list)
        for e in edges:
            evidence[e['target']].append(e)
        record = dict(snapshot_id=sid, task_id=item['task_id'], trajectory_id=item['trajectory_id'],
            checkpoint_step=max(u['step'] for u in prefix), units=prefix, edges=edges,
            scores=scores, rendered_history=full, body_character_spans=spans,
            source_text_characters=sum(len(u['text']) for u in prefix), rendered_characters=len(full),
            graph_build_seconds=build_s, graph_rank_seconds=rank_s,
            left_censored=left_censored, observed_steps=sorted({u['step'] for u in prefix}),
            attention_association_status='not_measured', physical_mapping_status='not_measured',
            representation='Complete eligible units in specified Steps plus task constraints; framework/tool schemas excluded; original agent prompt not reproduced')
        save(output/'snapshots'/f'{sid}.json', record)
        snapshots.append({k:v for k,v in record.items() if k not in ('units','edges','scores','rendered_history','body_character_spans')})
        score_by = {r['unit_id']: r for r in scores}
        for u in prefix:
            if len(canonical(u['text'])) < 80 or len(u['text']) > 1600:
                continue
            ev = evidence[u['unit_id']]
            if u['step'] >= record['checkpoint_step']:
                continue
            categories = []
            if any(e['relation']=='redundant_with' for e in ev): categories.append('exact_duplicate')
            if any(e['relation']=='lexical_overlap' for e in ev): categories.append('partial_overlap')
            if u['content_type']=='task_instruction': categories.append('user_constraint_control')
            if observed_failure(u): categories.append('error_observation')
            if u['subtype']=='code' and u['content_type']=='tool_result': categories.append('code_observation')
            if u['content_type']=='thinking' and not any(e['relation'] in ('redundant_with','lexical_overlap') for e in ev): categories.append('thinking_without_replacement')
            # Late windows are added only to obtain actual failure controls.
            # They must not change the early-prefix candidate pools post hoc.
            if left_censored:
                categories = [c for c in categories if c == 'error_observation']
            for category in categories:
                pools[category].append(dict(snapshot_id=sid, task_id=item['task_id'], unit=u, evidence=ev,
                                            rank_score=score_by[u['unit_id']]['graph_propagation']))
        print(json.dumps(dict(snapshot=sid, units=len(prefix), checkpoint=record['checkpoint_step'], edges=len(edges))), flush=True)

    for item in items:
        path = bundle/item['trace_path']
        before = digest(path.read_text())
        units, _, records, *_ = normalize_trace(bundle, item)
        raw = dict(records)
        units = [u for u in units if u['main_eligible']]
        filtered = []
        for u in units:
            a, b = u['character_span']
            field = resolve(raw[u['trace_line']], u['field_pointer'])
            if field[a:b] != u['text']:
                raise ValueError('Unit no longer matches original source')
            if scaffold_overlap(field, a, b):
                removed_scaffolding += 1
            else:
                filtered.append(u)
        units = filtered
        if digest(path.read_text()) != before:
            raise ValueError('Source changed during extraction')
        source_hashes[item['trace_path']] = before
        seen_raw, seen_ws = set(), set()
        counts = defaultdict(Counter)
        for u in units:
            key = canonical(u['text'])
            c = counts[u['content_type']]
            c['all_units'] += 1
            c['characters'] += len(u['text'])
            if len(key) < 80:
                continue
            c['eligible_units_80chars'] += 1
            c['earlier_byte_duplicate'] += u['text'] in seen_raw
            c['earlier_whitespace_duplicate'] += key in seen_ws
            seen_raw.add(u['text']); seen_ws.add(key)
        for kind, count in counts.items():
            census.append(dict(trajectory_id=item['trajectory_id'], task_id=item['task_id'],
                               seed=item['seed'], content_type=kind, **count))
        if item['seed'] != 1001:
            continue
        by_step = defaultdict(list)
        for u in units:
            by_step[u['step']].append(u)
        prefix, nchar = [], 0
        for step, group in sorted(by_step.items()):
            # Complete observed Steps, no clipping within units or tool returns.
            size = sum(len(u['text']) for u in group)
            if nchar + size > max_characters:
                break
            prefix.extend(group); nchar += size
        if not prefix or max(u['step'] for u in prefix) < 2:
            exclusions.append(dict(trajectory_id=item['trajectory_id'],
                                   reason='fewer_than_two_complete_steps_within_character_budget'))
        else:
            add_snapshot(item, prefix, item['task_id'])
        failures = [u for u in units if observed_failure(u) and 80 <= len(canonical(u['text'])) <= 1600]
        if failures:
            failure = failures[0]
            checkpoint = failure['step'] + 1
            constraints = [u for u in units if u['content_type']=='task_instruction' and u['step']==0]
            recent, nch = [], sum(len(u['text']) for u in constraints)
            for step in sorted((s for s in by_step if 0 < s <= checkpoint), reverse=True):
                group = by_step[step]
                size = sum(len(u['text']) for u in group)
                if nch + size > max_characters:
                    break
                recent = group + recent; nch += size
            window = constraints + recent
            if failure['unit_id'] in {u['unit_id'] for u in window} and max(u['step'] for u in window) > failure['step']:
                add_snapshot(item, window, item['task_id']+'-failure-window', left_censored=True)
            else:
                exclusions.append(dict(trajectory_id=item['trajectory_id'], reason='first_actual_failure_window_exceeds_character_budget'))
    selected, used, task_counts = [], set(), Counter()
    categories = ['exact_duplicate','partial_overlap','error_observation','code_observation',
                  'thinking_without_replacement','user_constraint_control']
    for category in categories:
        pool = pools[category]
        category_tasks = set()
        for _ in range(4):
            available = [p for p in pool if p['unit']['unit_id'] not in used]
            if not available:
                raise ValueError(f'Insufficient candidates in {category}')
            p = min(available, key=lambda p:(p['task_id'] in category_tasks,
                    task_counts[p['task_id']], -p['rank_score'], p['unit']['unit_id']))
            u = p['unit']; used.add(u['unit_id']); category_tasks.add(p['task_id']); task_counts[p['task_id']] += 1
            selected.append(dict(candidate_id=f'c{len(selected)+1:02}', category=category,
                                 snapshot_id=p['snapshot_id'], task_id=p['task_id'], unit_id=u['unit_id'],
                                 content_type=u['content_type'], subtype=u['subtype'],
                                 source={k:u[k] for k in ('trace_path','trace_line','field_pointer','character_span','text_sha256')},
                                 text=u['text'], relation_evidence=p['evidence'],
                                 text_cluster_sha256=digest(canonical(u['text'])),
                                 semantic_gold=None, human_review_status='pending',
                                 selection_uses_future_after_checkpoint=False))
    jsonl(output/'candidates.jsonl', selected)
    jsonl(output/'census.jsonl', census)
    save(output/'snapshots.json', snapshots)
    save(output/'exclusions.json', exclusions)
    save(output/'source_hashes.json', source_hashes)
    totals = Counter()
    for row in census:
        for key in ('all_units','eligible_units_80chars','earlier_byte_duplicate','earlier_whitespace_duplicate','characters'):
            totals[key] += row.get(key, 0)
    save(output/'summary.json', dict(status='offline_preparation_completed_gpu_pending',
        trajectories=len(items), tasks=len({r['task_id'] for r in items}), totals=totals,
        byte_duplicate_ratio=totals['earlier_byte_duplicate']/totals['eligible_units_80chars'],
        whitespace_duplicate_ratio=totals['earlier_whitespace_duplicate']/totals['eligible_units_80chars'],
        semantic_overlap_ratio=None, supersession_ratio=None, historical_reuse_ratio=None,
        candidate_precision_at_k=None, candidate_recall_at_k=None, false_safe_rate=None,
        metric_status='Semantic accuracy and safe-deletion labels require independent review and actual interventions.',
        snapshots=len(snapshots), candidates=len(selected), category_counts=Counter(r['category'] for r in selected),
        category_distinct_tasks={kind:len({r['task_id'] for r in selected if r['category']==kind}) for kind in categories},
        excluded_framework_reminder_units=removed_scaffolding,
        max_source_characters=max_characters, elapsed_seconds=time.perf_counter()-begun))
    with (output/'human-review.csv').open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id','category','source_text','replacement_complete','critical_information','safe_delete','reviewer','notes'])
        for c in selected:
            writer.writerow([c['candidate_id'],c['category'],c['text'],'','','','',''])
    # Freeze the exact implementation used, separately from other workspace edits.
    (output/'code').mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        (output/'code'/path.name).write_bytes(path.read_bytes())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-characters', type=int, default=60000)
    a = p.parse_args(); run(a.bundle, a.output, a.max_characters)

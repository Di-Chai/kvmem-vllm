"""Reproduce the DuoKV observation study without model inference or writes to input."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from core import VERSION, normalize_trace, read_json, read_jsonl, detect_reuse, verify_unit, digest
from events import observed_events, supersession_relations
from analysis_stats import (write_json, write_jsonl, write_csv, future_stats, event_stats,
                            cross_seed, kv_analysis, cluster_interval, subtype_stats)
from prediction import task_folds, prediction_examples, evaluate


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle, out = args.bundle.resolve(), args.output.resolve()
    if out == bundle or bundle in out.parents or out in bundle.parents:
        parser.error('Input and output directories must be disjoint; input is read-only.')
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'study_config.json'
    if any(out.iterdir()) and not marker.exists():
        parser.error('Nonempty output must contain this study\'s study_config.json.')
    config = dict(version=VERSION, bundle=str(bundle), output=str(out), windows=[1,4,16],
        long_range_threshold=16, prediction_stride=8, prediction_units_per_checkpoint=12,
        random_seed=20261103, generated_date='2026-11-03', uses_llm=False,
        labels_are_ground_truth=False, bundle_manifest_sha256=sha_file(bundle/'MANIFEST.json'))
    write_json(marker, config)
    for folder in ('structured', 'figures', 'code'):
        (out/folder).mkdir(exist_ok=True)
    started = time.monotonic()
    # Run the supplied verifier with bytecode disabled, so input remains read-only.
    result = subprocess.run([sys.executable, '-B', str(bundle/'tools/verify_bundle.py')],
                            capture_output=True, text=True, check=True)
    verification = json.loads(result.stdout)
    write_json(out/'structured/bundle-verification-before.json', verification)
    index = read_json(bundle/'index.json')
    assert len(index) == 150
    assert len({r['task_id'] for r in index if r['family'] == 'deepswe'}) == 16
    assert sum(it['attempt_source'] == 'fresh_restart' for it in index) == 5
    assert sum(not it['final_saved_response_available'] for it in index) == 2
    print('Bundle verification passed; analyzing 150 traces.', flush=True)
    all_units, all_edges, retry_edges, all_events, all_relations, quality_rows, examples = [], [], [], [], [], [], []
    folds = task_folds(index)
    streams = {n: (out/'structured'/f'{n}.jsonl').open('w', encoding='utf-8') for n in
               ['units', 'reuse-evidence', 'retry-reuse-evidence', 'events', 'event-source-relations', 'executor-audit-unmapped']}
    def emit(name, rows):
        for row in rows:
            streams[name].write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
    try:
        for count, item in enumerate(index, 1):
            units, messages, records, steps, calls, quality = normalize_trace(bundle, item)
            cache = {item['trace_path']: dict(records)}
            for u in units:
                verify_unit(bundle, u, cache)
            assert len({u['unit_id'] for u in units}) == len(units)
            item['main_steps'] = max(steps.values())
            edges = detect_reuse(units)
            retries = detect_reuse(units, True) if item['family']=='terminalbench' else [dict(e, includes_retry=True) for e in edges]
            events, diag = observed_events(item, units, messages, records, calls)
            relations = supersession_relations(units, events, edges, item['main_steps'])
            for e in edges:
                assert e['target_step'] > e['source_step']
                assert not next(u for u in units if u['unit_id'] == e['target_unit_id'])['retry']
            examples.extend(prediction_examples(item, units, edges, events, folds))
            for e in events:
                e.pop('_output', None)
            emit('units', units); emit('reuse-evidence', edges); emit('retry-reuse-evidence', retries)
            emit('events', events); emit('event-source-relations', relations)
            emit('executor-audit-unmapped', [dict(r, trajectory_id=item['trajectory_id'], trace_line=n,
                trace_path=item['trace_path']) for n,r in records if r['record_type']=='execution_event'])
            all_units.extend(units); all_edges.extend(edges); retry_edges.extend(retries)
            all_events.extend(events); all_relations.extend(relations)
            quality_rows.append(dict(**item, units=len(units), main_units=sum(u['main_eligible'] for u in units),
                retry_units=sum(u['retry'] for u in units), span_checks_passed=len(units),
                rule_reuse_edges=len(edges), retry_inclusive_edges=len(retries),
                observed_events=len(events), assessed_relations=len(relations), normalization=quality,
                event_diagnostics=diag, confirmed_task_success=None))
            if count % 10 == 0 or count == 1:
                print(f'{count}/150 traces; units={len(all_units):,}; reuse edges={len(all_edges):,}; elapsed={time.monotonic()-started:.1f}s', flush=True)
    finally:
        for f in streams.values():
            f.close()
    by_id = {r['trajectory_id']: r for r in index}
    write_json(out/'structured/trajectory-quality.json', quality_rows)
    write_csv(out/'trajectory-quality.csv', quality_rows)
    write_json(out/'structured/task-folds.json', [dict(family=f, task_id=t, fold=k) for (f,t),k in sorted(folds.items())])
    print('Computing censored demand summaries and sensitivity analyses.', flush=True)
    unit_stats, demand, curves, gaps, dormant = future_stats(all_units, all_edges, by_id)
    write_csv(out/'future-reuse-statistics.csv', demand)
    write_csv(out/'unit-reuse-statistics.csv', unit_stats)
    write_csv(out/'subtype-reuse-statistics.csv', subtype_stats(unit_stats))
    write_csv(out/'future-demand-survival.csv', curves)
    write_csv(out/'reuse-gaps.csv', gaps)
    write_csv(out/'dormant-future-demand.csv', dormant)
    sensitivity = []
    for variant, ed, retry in [('include_parser_retry', retry_edges, True),
                               ('exclude_fresh_restart', all_edges, False),
                               ('exclude_incomplete', all_edges, False)]:
        _, summary, _, _, _ = future_stats(all_units, ed, by_id, retry, variant)
        sensitivity.extend(summary)
    write_csv(out/'sensitivity-statistics.csv', sensitivity)
    long_sensitivity = []
    for family in ('deepswe','terminalbench'):
        for level in ('explicit','semantic','potential_dependency'):
            rr=[r for r in gaps if r['family']==family and r['level']==level]
            for threshold in (8,16,32):
                long_sensitivity.append(dict(family=family, level=level, threshold=threshold,
                    observed_gaps=len(rr), long_gaps=sum(r['gap_since_last_observed_use']>=threshold for r in rr),
                    ratio=sum(r['gap_since_last_observed_use']>=threshold for r in rr)/len(rr) if rr else None))
    write_csv(out/'long-range-sensitivity.csv', long_sensitivity)
    event_summary=event_stats(all_events, all_relations, all_units)
    write_csv(out/'supersession-statistics.csv', event_summary)
    write_csv(out/'event-control-comparison.csv', [{k:r[k] for k in ['relation_id','trajectory_id','task_id','event_type','content_type','event_step','pre_reuse_steps','post_reuse_steps','window','full_window_eligible','control','did']} for r in all_relations])
    # Examples remain trace-linked and explicitly automated candidates.
    evolution=[]
    for label in ('fully_superseded','partially_superseded','not_superseded','uncertain'):
        rr=[r for r in all_relations if r['relation']==label]
        evolution.extend(sorted(rr,key=lambda r:(not r['post_event_reuse'],r['relation_id']))[:12])
    write_jsonl(out/'value-evolution-examples.jsonl', evolution)
    cross, cross_tasks, cross_events=cross_seed(all_units, all_edges, all_relations, index)
    write_csv(out/'cross-seed-demand.csv', cross)
    write_csv(out/'cross-seed-task-statistics.csv', cross_tasks)
    write_csv(out/'cross-seed-supersession-by-event.csv', cross_events)
    write_json(out/'structured/cross-seed-task-bootstrap.json', {key:cluster_interval([t[key] for t in cross_tasks]) for key in
        ['stable_demand_ratio','trajectory_dependent_demand_ratio','reuse_consistency','supersession_consistency','event_specific_supersession_consistency']})
    write_csv(out/'cross-seed-no-restart-tasks.csv', [r for r in cross_tasks if not r['fresh_restart_task']])
    print(f'Evaluating task-grouped prediction baselines on {len(examples):,} sampled windows.', flush=True)
    write_jsonl(out/'structured/prediction-examples.jsonl', examples)
    forecasts, baselines, calibration, models=evaluate(examples)
    write_csv(out/'future-demand-prediction.csv', forecasts)
    write_csv(out/'baseline-comparison.csv', baselines)
    write_csv(out/'prediction-calibration.csv', calibration)
    write_jsonl(out/'structured/prediction-models.jsonl', models)
    kv, corr=kv_analysis(bundle, all_units, all_edges, by_id)
    write_csv(out/'kv-snapshot-correlation.csv', kv)
    write_csv(out/'kv-correlation-summary.csv', corr)
    from reporting import make_reports, make_figures, make_cases, counterfactual_design
    cases=make_cases(bundle, all_units, all_edges, gaps, all_relations)
    write_jsonl(out/'structured/trace-case-studies.jsonl', cases)
    candidates, cf=counterfactual_design(all_units, all_edges, gaps, all_relations)
    write_jsonl(out/'counterfactual-candidates.jsonl', candidates)
    write_json(out/'counterfactual-design.json', cf)
    # Independent final input verification detects accidental input mutations.
    result=subprocess.run([sys.executable,'-B',str(bundle/'tools/verify_bundle.py')],capture_output=True,text=True,check=True)
    write_json(out/'structured/bundle-verification-after.json',json.loads(result.stdout))
    assert sha_file(bundle/'MANIFEST.json')==config['bundle_manifest_sha256']
    audit=dict(unit_span_checks=len(all_units), unique_units=len({u['unit_id'] for u in all_units}),
        positive_distance_edges=len(all_edges), verified_bundle_before_and_after=True,
        task_split_overlap=0, retained_original_input=True, event_counts=dict(Counter(e['event_type'] for e in all_events)),
        relationship_counts=dict(Counter(r['relation'] for r in all_relations)),
        explicit_edges=sum(e['level']=='explicit' for e in all_edges),
        semantic_edges=sum(e['level']=='semantic' for e in all_edges),
        potential_edges=sum(e['level']=='potential_dependency' for e in all_edges),
        human_reviewed_labels=0, causal_deletion_validations=0)
    write_json(out/'structured/analysis-audit.json',audit)
    # Data products precede all plots and prose.
    make_figures(out,demand,curves,gaps,event_summary,cross_tasks,baselines)
    make_reports(out,config,verification,quality_rows,audit,demand,sensitivity,dormant,event_summary,
                 cross_tasks,baselines,corr,cases,candidates)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in {'.py','.txt','.md'} or p.name=='requirements.txt':
            shutil.copy2(p,out/'code'/p.name)
    files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha_file(p))
           for p in sorted(out.rglob('*')) if p.is_file() and p.name!='output-manifest.json' and '__pycache__' not in p.parts]
    write_json(out/'output-manifest.json',dict(config=config,files=files,elapsed_seconds=round(time.monotonic()-started,2)))
    print(json.dumps(dict(status='complete',output=str(out),units=len(all_units),edges=len(all_edges),
        events=len(all_events),relations=len(all_relations),files=len(files),seconds=round(time.monotonic()-started,2))),flush=True)


if __name__=='__main__':
    main()

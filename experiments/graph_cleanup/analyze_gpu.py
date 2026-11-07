"""Audit and summarize recorded interventions; no human labels are inferred."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from protocol import quantile


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def table(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranks(values):
    result = [0.]*len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    begin = 0
    while begin < len(ordered):
        end = begin+1
        while end < len(ordered) and values[ordered[end]] == values[ordered[begin]]:
            end += 1
        for i in ordered[begin:end]:
            result[i] = (begin+end-1)/2+1
        begin = end
    return result


def spearman(x, y):
    if len(x) < 3 or len(x) != len(y):
        return None
    a, b = ranks(x), ranks(y)
    ma, mb = statistics.mean(a), statistics.mean(b)
    da, db = [v-ma for v in a], [v-mb for v in b]
    denominator = math.sqrt(sum(v*v for v in da)*sum(v*v for v in db))
    return sum(u*v for u,v in zip(da,db))/denominator if denominator else None


def prometheus(path):
    result = defaultdict(float)
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        match = re.match(r'([^ {]+)(?:\{[^}]*\})?\s+([-+\d.eE]+)(?:\s|$)', line)
        if match:
            result[match[1]] += float(match[2])
    return result


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cases = read(args.root/'paired-r1/prepared-cases.json')
    judge_status_path=args.root/args.judge_name/'status.json'
    judge_status=read(judge_status_path) if judge_status_path.exists() else {}
    rows, layers, rank_rows, audits = [], [], [], []
    for case in cases:
        c = case['candidate']; cid = c['candidate_id']
        folder = args.root/'paired-r1'/cid
        if not (folder/'reduced-kv.measurements.json').exists():
            continue
        full, reduced = [read(folder/f'{arm}.measurements.json') for arm in ('full','reduced-kv')]
        ftext, rtext = [read(folder/f'{arm}.response.json') for arm in ('full','reduced-kv')]
        assert ftext['token_ids_sha256'] == rtext['token_ids_sha256'] == case['token_ids_sha256']
        assert full['historical_kv_sha256'] == reduced['historical_kv_sha256']
        assert len(full['historical_kv_sha256']) == 16
        expected = list(range(case['query_start'],case['query_end']))
        assert len(reduced['measurements']) == 80
        layer_groups = defaultdict(list)
        this_layers = []
        for key, values in reduced['measurements'].items():
            assert [r['position'] for r in values] == expected
            assert [r['position'] for r in full['measurements'][key]] == expected
            kind = key.split('/')[0]
            number = int(re.search(r'layers\.(\d+)',key)[1])
            changes = [r['relative_l2'] for r in values]
            entry = dict(candidate_id=cid,category=c['category'],kind=kind,layer=number,
                tokens=len(values),mean=statistics.mean(changes),p95=quantile(changes,.95),max=max(changes))
            layers.append(entry); this_layers.append(entry); layer_groups[kind].extend(changes)
        repeated = folder/'reduced-kv-repeat.measurements.json'
        repeat_delta = None
        if repeated.exists():
            again = read(repeated)
            assert again['historical_kv_sha256'] == reduced['historical_kv_sha256']
            assert set(again['measurements']) == set(reduced['measurements'])
            repeat_delta = max(abs(a['relative_l2']-b['relative_l2'])
                for key, values in again['measurements'].items()
                for a,b in zip(values,reduced['measurements'][key]))
        masses = full['measured_attention_mass']
        assert len(masses) == 16
        block_count = len(next(iter(masses.values()))['logical_block_mass'])
        block_mass = [statistics.mean(m['logical_block_mass'][b] for m in masses.values()) for b in range(block_count)]
        candidate_mass = sum(block_mass[b] for b in case['candidate_mapping']['logical_blocks'])
        snapshot = read(args.dataset/'snapshots'/f'{case["snapshot_id"]}.json')
        scores = {s['unit_id']:dict(s) for s in snapshot['scores']}
        unit_map = case['all_unit_mapping']
        for uid, mapping in unit_map.items():
            scores[uid]['attention_mass'] = sum(block_mass[b] for b in mapping['logical_blocks'])
        selected = {}
        for method in ('lru_proxy','redundancy','graph_rules','graph_propagation','attention_mass'):
            ordered = sorted(scores, key=lambda uid:(scores[uid][method] if method=='attention_mass' else -scores[uid][method],uid))
            selected[method] = ordered[:10]
            rank_rows.append(dict(candidate_id=cid,method=method,rank=ordered.index(c['unit_id'])+1,
                universe=len(ordered),score=scores[c['unit_id']][method],
                top10=json.dumps(ordered[:10]),
                overlap_with_graph_top10=None))
        graph_top = set(selected['graph_propagation'])
        for row in rank_rows[-5:]:
            row['overlap_with_graph_top10'] = len(graph_top & set(selected[row['method']]))
        semantic = args.root/args.judge_name/cid/'response.json'
        choices, probabilities, diagnostics = {}, {}, {}
        judge_seconds = None
        if semantic.exists():
            sem = read(semantic)
            if sem.get('success'):
                choices = {k:v['choice'] for k,v in sem['body']['answers'].items()}
                probabilities = {k:v['probabilities'] for k,v in sem['body']['answers'].items()}
                diagnostics = sem['body']['diagnostics']
                judge_seconds = sem['seconds']
        row = dict(candidate_id=cid,category=c['category'],task_id=c['task_id'],
            left_censored=case['left_censored'],prompt_tokens=len(case['input_ids']),
            fixed_query_tokens=len(expected),deleted_blocks=len(case['candidate_mapping']['logical_blocks']),
            candidate_body_tokens=len(case['candidate_mapping']['token_indices']),
            excluded_tokens=32*len(case['candidate_mapping']['logical_blocks']),
            collateral_units=len(case['candidate_mapping']['collateral_unit_ids']),
            excluded_attention_kv_bytes=32*16*4*512*len(case['candidate_mapping']['logical_blocks']),
            physically_freed_bytes=0,attention_mass=candidate_mass,
            attention_mean=statistics.mean(layer_groups['attention_pre_gate_per_head']),
            attention_p95=quantile(layer_groups['attention_pre_gate_per_head'],.95),
            attention_max=max(layer_groups['attention_pre_gate_per_head']),
            transformer_mean=statistics.mean(layer_groups['transformer_residual_sum']),
            transformer_p95=quantile(layer_groups['transformer_residual_sum'],.95),
            transformer_max=max(layer_groups['transformer_residual_sum']),
            final_layer_mean=next(r['mean'] for r in this_layers if r['kind']=='transformer_residual_sum' and r['layer']==63),
            repeat_delta=repeat_delta,full_answer_tokens=ftext['output_tokens'],reduced_answer_tokens=rtext['output_tokens'],
            complete_json=bool(ftext['parsed'] is not None and rtext['parsed'] is not None and ftext['finish_reason']=='stop' and rtext['finish_reason']=='stop'),
            judged_dimensions=len(choices),consistent_dimensions=sum(v=='consistent' for v in choices.values()),
            omission_dimensions=sum(v=='omission' for v in choices.values()),
            contradiction_dimensions=sum(v=='contradiction' for v in choices.values()),
            uncertain_dimensions=sum(v=='uncertain' for v in choices.values()),
            inconsistency_probability=statistics.mean(1-v['consistent'] for v in probabilities.values()) if probabilities else None,
            judge_omission_contradiction_probability=statistics.mean(v['omission']+v['contradiction'] for v in probabilities.values()) if probabilities else None,
            min_label_mass=min(v['label_mass'] for v in diagnostics.values()) if diagnostics else None,
            all_argmax_labels=all(v['argmax_is_label'] for v in diagnostics.values()) if diagnostics else None,
            judge_seconds=judge_seconds)
        rows.append(row)
        audits.append(dict(candidate_id=cid,historical_hashes_equal=True,same_prompt_tokens=True,
            complete_fixed_position_measurements=True,attention_layers=16,transformer_layers=64,
            query_tokens=len(expected),repeat_max_change=repeat_delta,
            logical_blocks=case['candidate_mapping']['logical_blocks'],
            collateral_unit_ids=case['candidate_mapping']['collateral_unit_ids'],
            semantic_choices=choices,semantic_diagnostics=diagnostics))
    table(args.output/'candidates.csv',rows); table(args.output/'layers.csv',layers)
    table(args.output/'ranking.csv',rank_rows); save(args.output/'audit.json',audits)
    review, key, labels = [], [], []
    for case in cases:
        cid=case['candidate']['candidate_id']
        if not any(r['candidate_id']==cid for r in rows):
            continue
        folder=args.root/'paired-r1'/cid
        answers={arm:read(folder/f'{arm}.response.json')['parsed'] for arm in ('full','reduced-kv')}
        order=['full','reduced-kv']
        if hashlib.sha256(('blinded-review-'+cid).encode()).digest()[0]%2:
            order.reverse()
        review.append(dict(review_id=cid,history=case['prompt'],
            answer_A=answers[order[0]],answer_B=answers[order[1]],
            instructions='Using the quoted history as evidence, independently assess factual support and important omissions/contradictions in each answer. Model scores and intervention identity are withheld.'))
        key.append(dict(review_id=cid,answer_A_arm=order[0],answer_B_arm=order[1]))
        for dimension in ('goal','progress','decision','open_issues','constraints','next_action'):
            labels.append(dict(review_id=cid,dimension=dimension,A_supported='',B_supported='',
                A_critical_omission='',B_critical_omission='',A_contradiction='',B_contradiction='',
                source_evidence='',reviewer=''))
    with (args.output/'review-blinded.jsonl').open('w') as f:
        for row in review:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    save(args.output/'review-key.json',key)
    # Preserve any future human entries when regenerating derived analysis.
    review_path=args.output/'human-semantic-review.csv'
    if review_path.exists():
        with review_path.open() as f:
            previous={(r['review_id'],r['dimension']):r for r in csv.DictReader(f)}
        labels=[previous.get((r['review_id'],r['dimension']),r) for r in labels]
    table(review_path,labels)
    from judge_revision import normalize
    identical_fields=[]
    for case in cases:
        cid=case['candidate']['candidate_id'];folder=args.root/'paired-r1'/cid
        old_path=args.root/'judge-r1'/cid/'response.json';new_path=args.root/'judge-r2'/cid/'response.json'
        if not old_path.exists() or not new_path.exists():continue
        a,b=[read(folder/f'{arm}.response.json')['parsed'] for arm in ('full','reduced-kv')]
        old,new=read(old_path),read(new_path)
        if not old['success'] or not new['success']:continue
        for key in a:
            if normalize(a[key])==normalize(b[key]):
                identical_fields.append(dict(candidate_id=cid,dimension=key,normalized_text=normalize(a[key]),
                    original_full=a[key],original_reduced=b[key],initial_choice=old['body']['answers'][key]['choice'],
                    revised_choice=new['body']['answers'][key]['choice']))
    save(args.output/'identical-field-audit.json',dict(
        scope='Mechanical equality of citation-normalized text in the same field, not independent task-safety labels.',
        count=len(identical_fields),fields=identical_fields,
        initial_choices=dict(Counter(r['initial_choice'] for r in identical_fields)),
        revised_choices=dict(Counter(r['revised_choice'] for r in identical_fields))))
    valid = [r for r in rows if r['inconsistency_probability'] is not None]
    correlations = []
    for kind in ('attention_pre_gate_per_head','transformer_residual_sum'):
        for layer in sorted({r['layer'] for r in layers if r['kind']==kind}):
            index = {r['candidate_id']:r for r in layers if r['kind']==kind and r['layer']==layer}
            correlations.append(dict(kind=kind,layer=layer,n=len(valid),
                spearman_inconsistency=spearman([index[r['candidate_id']]['mean'] for r in valid],[r['inconsistency_probability'] for r in valid]),
                spearman_omission_contradiction=spearman([index[r['candidate_id']]['mean'] for r in valid],[r['judge_omission_contradiction_probability'] for r in valid])))
    table(args.output/'correlations.csv',correlations)
    groups = []
    for category in sorted({r['category'] for r in rows}):
        chosen = [r for r in rows if r['category']==category]
        groups.append(dict(category=category,n=len(chosen),
            attention_mean=statistics.mean(r['attention_mean'] for r in chosen),
            transformer_mean=statistics.mean(r['transformer_mean'] for r in chosen),
            final_layer_mean=statistics.mean(r['final_layer_mean'] for r in chosen),
            judged=sum(r['judged_dimensions'] for r in chosen),
            choices=dict(Counter({v:sum(r[v+'_dimensions'] for r in chosen) for v in ('consistent','omission','contradiction','uncertain')}))))
    save(args.output/'groups.json',groups)
    quadrants = defaultdict(list)
    threshold = statistics.median(r['transformer_mean'] for r in valid) if valid else None
    for r in valid:
        quadrant = ('higher' if r['transformer_mean']>=threshold else 'lower')+'_delta__'+('all_consistent' if r['consistent_dimensions']==6 else 'not_all_consistent')
        quadrants[quadrant].append(r['candidate_id'])
    save(args.output/'summary.json',dict(candidates=len(rows),complete_json=sum(r['complete_json'] for r in rows),
        judge_source=args.judge_name,
        judge_controls_passed=judge_status.get('controls_passed'),semantic_metric_validated=False,
        fixed_query_tokens_per_candidate=148,attention_layers=16,transformer_layers=64,
        judged_candidates=len(valid),choices=dict(Counter({v:sum(r[v+'_dimensions'] for r in valid) for v in ('consistent','omission','contradiction','uncertain')})),
        zero_deletion=read(args.root/'paired-r1/zero-deletion-control.json'),
        identical_repeat=read(args.root/'paired-r1/identical-repeat-control.json'),
        repeat_max=max((r['repeat_delta'] for r in rows if r['repeat_delta'] is not None),default=None),
        exploratory_median_delta=threshold,quadrants=dict(quadrants),
        interpretation='Descriptive model-assisted scores, not calibrated safety or human ground truth. A failed control invalidates treating the scores as semantic-loss labels. Median split is visualization only.',
        human_precision=None,human_recall=None,false_safe_rate=None,high_risk_recall=None))
    serving = args.root/'serving-r1/summary.json'
    if serving.exists():
        metrics = []
        for cell in read(serving):
            path = args.root/'serving-r1'/cell['name']
            before, after = [prometheus(path/f'metrics-{part}.txt') for part in ('before','after')]
            delta = {k:after[k]-before.get(k,0) for k in after if k.endswith(('_sum','_count','_total'))}
            # Sum request prefill times can overlap in a concurrent scheduler;
            # report this rate as request-time-normalized, never engine throughput.
            elapsed = delta.get('vllm:request_prefill_time_seconds_sum',0)
            token_sum = delta.get('vllm:prompt_tokens_total',0)
            preemptions = delta.get('vllm:num_preemptions_total')
            requests=[read(p) for p in path.glob('*.json') if p.name!='summary.json']
            successful=[r for r in requests if r.get('success') and r.get('client_ttft_seconds') is not None]
            window=(max(r['started']+r['client_ttft_seconds'] for r in successful)-min(r['started'] for r in successful)) if successful and len(successful)==cell['submitted'] else None
            row = dict(**cell,request_time_normalized_prefill_tps=token_sum/elapsed if elapsed>0 else None,
                all_first_tokens_window_seconds=window,
                effective_input_tokens_per_admission_window_second=sum(r['input_tokens'] for r in successful)/window if window else None,
                preemptions=preemptions)
            metrics.append(row)
            save(args.output/'serving-metric-deltas'/f'{cell["name"]}.json',delta)
        table(args.output/'serving.csv',metrics)
        save(args.output/'serving-summary.json',dict(cells=len(metrics),requests=sum(r['submitted'] for r in metrics),
            completed=sum(r['completed'] for r in metrics),failed=sum(r['failed'] for r in metrics),
            wall_seconds=sum(r['wall_seconds'] for r in metrics),
            caveat='One wave per cell; prefill rate uses summed request prefill time and is not device-level aggregate throughput.'))
    native = args.root/'serving-r1/native-systemone/summary.json'
    if native.exists():
        native_metrics=[]
        for cell in read(native):
            cell=dict(cell)
            cell.setdefault('calibration_trailing_whitespace','')
            path=native.parent/cell['name']
            before,after=[prometheus(path/f'metrics-{part}.txt') for part in ('before','after')]
            native_metrics.append(dict(**cell,
                effective_input_tokens_per_wall_second=cell['aggregate_read_input_tokens']/cell['wall_seconds'],
                preemptions=after.get('vllm:num_preemptions_total',0)-before.get('vllm:num_preemptions_total',0)))
        table(args.output/'native-serving.csv',native_metrics)
        save(args.output/'native-serving-summary.json',dict(cells=len(native_metrics),
            requests=sum(r['submitted'] for r in native_metrics),
            completed=sum(r['completed'] for r in native_metrics),failed=sum(r['failed'] for r in native_metrics),
            wall_seconds=sum(r['wall_seconds'] for r in native_metrics),
            caveat='One native categorical label per request; HTTP response latency, not streaming TTFT.'))
    print(json.dumps(dict(candidates=len(rows),semantic_candidates=len(valid),output=str(args.output))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True); p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--judge-name',default='judge-r2')
    analyze(p.parse_args())

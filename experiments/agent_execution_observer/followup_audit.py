"""Audit new format/goal-scope experiments without invoking inference."""
import argparse
import csv
import html
import json
import math
from pathlib import Path

from audit import diagnostics, distribution, normalize_boundaries
from followup import (CONTROL_STEPS, bounded_goal, check_inputs, diagnostic_summary,
                      explicit_letters, read_jsonl)
from observer import Memory, digest, request_for, save, validate
from observer import resolve
from goal_hint_probe import STEPS as HINT_STEPS, replace_hint


def require(condition,message):
    if not condition:raise ValueError(message)


def same_payload(actual,expected):
    # Dict equality alone would miss option order, which changes native labels.
    require(json.dumps(actual,ensure_ascii=False)==json.dumps(expected,ensure_ascii=False),
            'Model state, question, or option order differs from declared intervention')


def validate_native(response,payload):
    validate(response,payload)
    for qid,answer in response['answers'].items():
        require(answer['type']=='choice','Unexpected response question type')
        require(math.isclose(answer['probabilities'][answer['choice']],max(answer['probabilities'].values()),abs_tol=1e-12),
                'Chosen answer does not maximize native label probability')
        diagnostic=response['diagnostics'][qid]
        require(type(diagnostic['argmax_is_label']) is bool,'Invalid format diagnostic')
        require(0<=diagnostic['label_mass']<=1+1e-6,'Invalid vocabulary label mass')
    require(type(response['usage']['input_tokens']) is int and response['usage']['input_tokens']>0,
            'No actual input token usage')


def summary_records(rows):
    return dict(http_calls=len(rows),input_tokens=sum(x['response']['usage']['input_tokens'] for x in rows),
        label_reads=sum(x['response']['usage']['output_tokens'] for x in rows),
        http_seconds=sum(x['seconds'] for x in rows),
        latency_seconds=distribution([x['seconds'] for x in rows]),diagnostics=diagnostic_summary(rows))


def audit_goal_hint(inputs,source_file,results,bundle):
    source=json.loads(source_file.read_text());hint=source['goal_hint'];src=hint['source']
    path=bundle/src['trace_path']
    require(digest(path)==source['source_trace_sha256'],'Goal source trace changed')
    record=read_jsonl(path)[src['trace_line']-1]
    require(record['turn']==src['step']==11,'Goal source must be the already observed Step11')
    require(record['message_index']==src['message_index'],'Goal source message identity changed')
    raw=resolve(record,src['pointer']);begin,end=src['character_span']
    require(raw[begin:end]==hint['text'],'Goal hint is not the original source quote')
    original11=json.loads((inputs/'baseline-requests/0011.json').read_text())
    require(hint['text'] in original11['state']['current_step'],'Goal hint not observed in original Step11 payload')
    run=json.loads((results/'run.json').read_text())
    require(run['source_sha256']==digest(source_file),'Executed goal source changed')
    require(run['code_sha256']==digest(Path(__file__).with_name('goal_hint_probe.py')),'Goal probe code changed')
    require(json.loads((results/'summary.json').read_text())['status']=='completed','Goal probes incomplete')
    rows=read_jsonl(results/'responses.jsonl')
    require({r['id'] for r in rows}=={f'{n:04}-{v}' for n in HINT_STEPS for v in ('original_hint','prior_task_subject')}
            and len(rows)==12,'Goal representation pair coverage differs')
    by_id={r['id']:r for r in rows};pairs=[]
    for n in HINT_STEPS:
        original=json.loads((inputs/'baseline-requests'/f'{n:04}.json').read_text())
        base=explicit_letters(original);base['questions']={'stage_link':base['questions']['stage_link']}
        pair=dict(step=n)
        for variant in ('original_hint','prior_task_subject'):
            ident=f'{n:04}-{variant}';expected=base if variant=='original_hint' else replace_hint(base,hint)
            payload=json.loads((results/'calls'/(ident+'.request.json')).read_text())
            response=json.loads((results/'calls'/(ident+'.response.json')).read_text())
            same_payload(payload,expected);validate_native(response,payload)
            require(by_id[ident]['response']==response,'Goal probe response metadata changed')
            pair[variant]=dict(choice=response['answers']['stage_link']['choice'],diagnostic=response['diagnostics']['stage_link'])
        pair['choice_changed']=pair['original_hint']['choice']!=pair['prior_task_subject']['choice'];pairs.append(pair)
    return dict(status='passed',exact_requests=12,source_quote_verified=True,source_observed_at_step=11,
                pairs=pairs,cost=summary_records(rows),
                scope='Adaptive goal-representation diagnostic on frozen potentially inconsistent baseline memory; '
                      'not autonomous goal discovery or human-gold segmentation evaluation.')


def audit_followup(inputs,results,baseline,output,activity_reference=None,
                   goal_hint_source=None,goal_hint_results=None,bundle=None):
    manifest=check_inputs(inputs)
    require(json.loads((results/'status.json').read_text())['status']=='completed','Followup is not complete')
    run=json.loads((results/'run.json').read_text())
    require(run['input_manifest_sha256']==digest(inputs/'manifest.json'),'Executed input manifest changed')
    require(run['code_sha256']==digest(Path(__file__).with_name('followup.py')),'Executed followup code changed')
    require(run['observer_sha256']==digest(Path(__file__).with_name('observer.py')),'Executed memory code changed')
    require(digest(baseline/'run.json')==manifest['baseline_run_sha256'],'Original baseline run changed')
    require(digest(baseline/'summary.json')==manifest['baseline_summary_sha256'],'Original baseline summary changed')
    steps=json.loads((inputs/'dataset.json').read_text())
    old=json.loads((inputs/'baseline-decisions.json').read_text())
    require(old==read_jsonl(baseline/'decisions.jsonl'),'Original native decisions changed')
    old_by_step={r['step']:r for r in old};step_by_number={s['step']:s for s in steps}
    originals={s['step']:json.loads((inputs/'baseline-requests'/f'{s["step"]:04}.json').read_text()) for s in steps}
    for row in manifest['requests']:
        n=row['step'];name=old_by_step[n]['call_id']+'.request.json'
        require(digest(baseline/'calls'/name)==row['original_request_sha256'],'Original request bytes changed')
        same_payload(originals[n],json.loads((baseline/'calls'/name).read_text()))
    counts={};all_records={};selected_evidence=0

    def check_call(cohort,record,expected):
        nonlocal selected_evidence
        directory=results/cohort/'calls';ident=record['id']
        payload=json.loads((directory/(ident+'.request.json')).read_text())
        response=json.loads((directory/(ident+'.response.json')).read_text())
        same_payload(payload,expected);validate_native(response,payload)
        require(record['response']==response,'Saved response metadata changed')
        n=record['step'];state=payload['state']
        require(state['current_step_id']==n,'Wrong current Step')
        memory=state['execution_memory']
        require(all(x['step']<n for x in memory['recent_steps']),'Future/current recent memory')
        require(all(x['last_step']<n for x in memory['stages']),'Future/current stage memory')
        for e in state['evidence_candidates'].values():require(e['source']['step']==n,'Cross-Step evidence')
        evidence=response['answers'].get('evidence')
        if evidence and evidence['choice']!='uncertain':
            require(evidence['choice'] in step_by_number[n]['evidence'],'Invented evidence reference')
            selected_evidence+=1
        counts[cohort]=counts.get(cohort,0)+1
        return response

    frozen=read_jsonl(results/'frozen-state/responses.jsonl')
    expected_ids={f'{n:04}-letters' for n in range(1,58)} | {f'{n:04}-original' for n in CONTROL_STEPS}
    require({r['id'] for r in frozen}==expected_ids and len(frozen)==len(expected_ids),'Frozen-state coverage differs')
    for row in frozen:
        source=originals[row['step']]
        expected=source if row['variant']=='fresh_original_control' else explicit_letters(source)
        check_call('frozen-state',row,expected)
    all_records['frozen-state']=frozen

    sequential=read_jsonl(results/'sequential/responses.jsonl')
    decisions=read_jsonl(results/'sequential/decisions.jsonl')
    require([r['step'] for r in sequential]==list(range(1,58)) and len(decisions)==57,'Causal replay coverage differs')
    memory=Memory()
    for step,row,decision in zip(steps,sequential,decisions):
        response=check_call('sequential',row,explicit_letters(request_for(step,memory,'observer')))
        require(memory.apply(step,response)==decision,'Derived stage state differs from native answers')
    require(memory.view()==json.loads((results/'sequential/memory.json').read_text()),'Final causal memory changed')
    all_records['sequential']=sequential

    references=json.loads((inputs/'goal-references.json').read_text())
    goals=read_jsonl(results/'goal-scope/responses.jsonl')
    require(len(goals)==len(references)==7,'Goal-scope coverage differs')
    checks=[]
    for ref,row in zip(references,goals):
        require(row['id']==f'{ref["step"]:04}-{ref["case"]}','Goal case identity differs')
        expected=bounded_goal(originals[ref['step']],ref['case'],ref['goal'])
        response=check_call('goal-scope',row,expected)
        checks.append(dict(**ref,observed=response['answers']['outcome']['choice'],
            satisfies_predeclared_check=response['answers']['outcome']['choice'] in ref['accepted_outcomes'],
            outcome_diagnostic=response['diagnostics']['outcome'],evidence=response['answers']['evidence']['choice']))
    all_records['goal-scope']=goals
    for cohort,records in all_records.items():
        require(not (results/cohort/'failure.json').exists(),'Recorded failure in '+cohort)
        require(len(list((results/cohort/'calls').glob('*.request.json')))==len(records),'Extra/missing requests')
        require(len(list((results/cohort/'calls').glob('*.response.json')))==len(records),'Extra/missing responses')
        summary=json.loads((results/cohort/'summary.json').read_text())
        require(summary['http_calls']==len(records),'Summary call count differs')
        require(summary['input_tokens']==sum(x['response']['usage']['input_tokens'] for x in records),'Summary usage differs')
        require(summary['label_reads']==sum(x['response']['usage']['output_tokens'] for x in records),'Summary label reads differ')
    formatted=[r for r in frozen if r['variant']=='explicit_letters'];controls=[r for r in frozen if r['variant']=='fresh_original_control']
    changes={q:sum(r['response']['answers'][q]['choice']!=old_by_step[r['step']]['native_answers'][q]['choice'] for r in formatted)
             for q in ('activity','stage_link','outcome','evidence')}
    control_changes=[]
    for row in controls:
        n=row['step'];before=old_by_step[n]['native_answers'];after=row['response']['answers']
        control_changes.append(dict(step=n,changed_questions=[q for q in before if before[q]['choice']!=after[q]['choice']],
            max_absolute_probability_change=max(abs(before[q]['probabilities'][label]-after[q]['probabilities'][label])
                for q in before for label in before[q]['probabilities'])))
    report=dict(status='passed',exact_request_reconstructions=sum(counts.values()),counts=counts,
        selected_evidence_references_valid=selected_evidence,semantic_evidence_support_assessed=False,
        original_diagnostics=diagnostics(old),formatted_frozen=summary_records(formatted),
        fresh_original_controls=summary_records(controls),controls_vs_saved_baseline=control_changes,
        frozen_state_changed_choices=changes,sequential=summary_records(sequential),
        original_normalized_boundaries=sum(d['stage_boundary'] is True for d in normalize_boundaries(old)),
        new_sequential_boundaries=sum(d['stage_boundary'] is True for d in decisions),
        new_sequential_unknown_ids=sum(d['stage_id'] is None for d in decisions),
        goal_scope_checks=checks,goal_checks_satisfied=sum(c['satisfies_predeclared_check'] for c in checks),
        human_gold_accuracy=None,human_gold_boundary_f1=None,adaptive_diagnostic_not_independent_validation=True)
    if activity_reference:
        anchors=json.loads(activity_reference.read_text())['labels']
        new={d['step']:d for d in decisions}
        report['provisional_activity_agreement']=dict(anchors=len(anchors),
            original_agree=sum(old_by_step[a['step']]['activity'] in a['accepted_activity'] for a in anchors),
            new_sequential_agree=sum(new[a['step']]['activity'] in a['accepted_activity'] for a in anchors),
            reference_sha256=digest(activity_reference),human_gold=False)
    if goal_hint_source or goal_hint_results or bundle:
        require(all(x is not None for x in (goal_hint_source,goal_hint_results,bundle)),
                'Goal hint audit requires source, results, and source bundle')
        report['goal_representation']=audit_goal_hint(inputs,goal_hint_source,goal_hint_results,bundle)
    output.mkdir(parents=True,exist_ok=False)
    save(output/'audit.json',report)
    with (output/'stage-comparison.csv').open('w') as f:
        fields=['step','old_stage_id','new_stage_id','old_boundary','new_boundary','old_activity','new_activity','old_outcome','new_outcome','new_evidence']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for old_row,new in zip(normalize_boundaries(old),decisions):
            w.writerow(dict(step=new['step'],old_stage_id=old_row['stage_id'],new_stage_id=new['stage_id'],
                old_boundary=old_row['stage_boundary'],new_boundary=new['stage_boundary'],
                old_activity=old_row['activity'],new_activity=new['activity'],old_outcome=old_row['outcome'],
                new_outcome=new['outcome'],new_evidence=json.dumps(new['selected_evidence'],ensure_ascii=False)))
    table=[]
    for before,after in zip(normalize_boundaries(old),decisions):
        evidence=after['selected_evidence'] or {}
        values=[after['step'],before['stage_id'],after['stage_id'],before['stage_boundary'],after['stage_boundary'],
                before['activity'],after['activity'],before['outcome'],after['outcome'],evidence.get('quote','')]
        cells=''.join('<td>'+html.escape(str(v))+'</td>' for v in values)
        table.append('<tr>'+cells+'<td><details><summary>本次可见 Step 与证据坐标</summary><pre>'+
            html.escape(json.dumps(dict(current_step=step_by_number[after['step']]['current_step'],
                                        citation=evidence.get('source')),ensure_ascii=False,indent=2))+'</pre></details></td></tr>')
    goal_rows=[]
    for ref in checks:
        values=[ref['step'],ref['case'],ref['goal'],ref['accepted_outcomes'],ref['observed'],ref['satisfies_predeclared_check']]
        goal_rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    document='''<!doctype html><meta charset="utf-8"><title>System One 格式与目标范围复核</title>
<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;width:100%;margin-bottom:30px}
td,th{border:1px solid #ccc;padding:7px;vertical-align:top}pre{white-space:pre-wrap;max-width:680px}</style>
<h1>System One 新一轮验证</h1><p>原生模型结果。无独立人工 GT，不报告正式准确率或边界 F1。
Stage ID 属于各自运行，两个相同数字不代表相同语义目标。旧未知边界仅在审阅副本中归一化。
目标范围检查提供明确目标，不证明模型自主发现目标。</p>
<h2>明确目标范围检查</h2><table><tr><th>Step</th><th>Case</th><th>提供的目标</th><th>预定义允许结果</th><th>实际</th><th>满足检查</th></tr>'''
    document+=''.join(goal_rows)+'</table>'
    if 'goal_representation' in report:
        hint_rows=[]
        for pair in report['goal_representation']['pairs']:
            values=[pair['step'],pair['original_hint']['choice'],pair['prior_task_subject']['choice'],pair['choice_changed']]
            hint_rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
        document+='''<h2>目标表示配对诊断</h2><p>只替换 S004 的目标摘录，来源是此前 Step11 已观察到的任务主题；
其余旧记忆保留，可能含原有错误归属。这是条件目标辨识检查，不能替代自主目标发现或人工标注。</p>
<table><tr><th>Step</th><th>原目标摘录</th><th>先前任务主题</th><th>选择变化</th></tr>'''+''.join(hint_rows)+'</table>'
    document+='''<h2>原始基线与统一格式的独立连续回放</h2><table><tr>
<th>Step</th><th>旧 ID</th><th>新 ID</th><th>旧边界</th><th>新边界</th><th>旧活动</th><th>新活动</th>
<th>旧状态</th><th>新状态</th><th>新证据</th><th>可展开输入</th></tr>'''+''.join(table)+'</table>'
    (output/'review.html').write_text(document)
    save(output/'output-hashes.json',{str(p.relative_to(output)):digest(p) for p in output.iterdir() if p.is_file()})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('inputs','results','baseline','output'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--activity-reference',type=Path)
    p.add_argument('--goal-hint-source',type=Path);p.add_argument('--goal-hint-results',type=Path)
    p.add_argument('--bundle',type=Path);a=p.parse_args()
    result=audit_followup(a.inputs,a.results,a.baseline,a.output,a.activity_reference,
                         a.goal_hint_source,a.goal_hint_results,a.bundle)
    print(json.dumps(dict(status=result['status'],exact_requests=result['exact_request_reconstructions'],
                         goal_checks_satisfied=result['goal_checks_satisfied']),indent=2))


if __name__=='__main__':main()

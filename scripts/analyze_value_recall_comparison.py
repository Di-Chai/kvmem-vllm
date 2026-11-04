"""Official scores and same-state output errors for the three Value methods."""
import argparse
import csv
from itertools import combinations
from pathlib import Path
import statistics

import recall_comparison_protocol as p
import ruler_protocol as ruler
from run_recall_comparison import phase_rows,read_events,resolve,saved_result,verify


def analyze(root):
    fixture,profile=verify(root)
    p.require(fixture.get('value_recall_reference') and
              (fixture.get('full_value_comparison') or fixture.get('pressure_attention_hybrid')),
              'Expected the independent Value-reference campaign')
    methods=p.campaign_methods(fixture)
    pilot=bool(fixture.get('answer_prefix_pilot'))
    phase='qualification' if pilot else 'formal'
    sources=phase_rows(fixture,phase)
    samples_per_task=fixture['pilot_samples_per_task'] if pilot else 20
    scorer=ruler.load_scorer(resolve(root,fixture['upstream_path']))
    arms={};official={}
    for method in methods:
        rows=[]
        for source in sources:
            case=root/'arms'/method/phase/'cases'/source['id']
            result=saved_result(case,source,method,root)
            p.require(result is not None,'Incomplete Value method; no full-cohort score is published')
            attempt=case/result['attempt'];response=p.read(attempt/'response.json')
            expected=p.score_case(source,response,scorer,starts_in_thinking=fixture['starts_in_thinking'])
            p.require(all(result[k]==v for k,v in expected.items()),'Stored official score changed')
            evidence=p.read(attempt/'evidence.json')
            arm_profile=p.read(root/'profiles'/(method+'.json'))
            p.audit_events(evidence['events'],method,arm_profile,query_span=source['served_query'],
                required_types=('prefill_pressure','query'))
            rows.append(result)
        arms[method]={r['id']:r for r in rows}
        official[method]=ruler.score_suite(sources,
            {(r['task'],r['index']):r['text'] for r in rows},scorer,
            samples_per_task=samples_per_task,starts_in_thinking=fixture['starts_in_thinking'])
    per_task={task:{m:official[m]['per_task'][task]['score'] for m in methods} for task in ruler.TASK_FAMILIES}
    macro={m:statistics.fmean(per_task[t][m] for t in per_task) for m in methods}
    paired={}
    for a,b in combinations(methods,2):
        ds=[arms[a][i]['case_score_unrounded']-arms[b][i]['case_score_unrounded'] for i in arms[a]]
        paired[a+'_vs_'+b]=dict(wins=sum(d>1e-9 for d in ds),losses=sum(d < -1e-9 for d in ds),ties=sum(abs(d)<=1e-9 for d in ds))
    diagnostics=[]
    for path in (root/'arms').glob('*/qualification/server-attempts/*/server/capture/value_recall_diagnostics.jsonl'):
        diagnostics.extend(read_events(path))
    p.require(diagnostics or fixture.get('pressure_attention_hybrid'),'Missing immutable-state diagnostics')
    errors={m:dict(states=len(diagnostics),mean_state_l2=statistics.fmean(
        d['methods'][m]['set_output_error']['mean_query_layer_head_l2'] for d in diagnostics)) for m in methods} if diagnostics else {}
    truncation={m:dict(cases=len(values),truncated=sum(r['finish_reason']=='length' for r in values.values()),
        rate=sum(r['finish_reason']=='length' for r in values.values())/len(values),
        completion_tokens=sum(r['completion_tokens'] for r in values.values()),
        per_task={t:dict(cases=sum(r['task']==t for r in values.values()),
            truncated=sum(r['task']==t and r['finish_reason']=='length' for r in values.values()))
            for t in ruler.TASK_FAMILIES}) for m,values in arms.items()}
    result=dict(cases=len(sources),samples_per_task=samples_per_task,pilot=pilot,methods=list(methods),
        per_task=per_task,macro_scores=macro,truncation=truncation,
        truncation_definition='Response finish_reason=length; supplied assistant prefix is input, not generated output.',
        answer_prefix_policy=fixture.get('answer_prefix_policy'),
        pressure_attention_hybrid=bool(fixture.get('pressure_attention_hybrid')),
        full_reference_successes={m:sum(r['full_reference_recall'] for r in arms[m].values()) for m in methods},
        pairwise=paired,same_state_set_output_errors=errors,
        scope='Scores use fresh independent original inputs. Errors are whole selected sets on shared frozen content-frame Q/K/V states, never sums of per-block deletion errors.',
        warning='Same-state attention output error is a local diagnostic, not an upper bound on task score.')
    p.save(root/'analysis/comparison.json',result)
    with (root/'analysis/per-task.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task',*methods])
        for task,row in per_task.items():w.writerow([task,*[row[m] for m in methods]])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    print(analyze(parser.parse_args().root.resolve())['macro_scores'],flush=True)

"""Compare new first answers with the exact previously stored 26-query baseline."""
from pathlib import Path
import statistics

import recall_comparison_protocol as p
import ruler_protocol
import run_layerwise_query_diagnostic as driver
from run_concurrent_layerwise_ruler import audit_events


def analyze(root,baseline,output):
    driver.verify(root)
    fixture,profile=p.read(root/'fixture.json'),p.read(root/'profile.json')
    old_fixture,old_profile=p.read(baseline/'fixture.json'),p.read(baseline/'profile.json')
    old_rows={r['id']:r for r in old_fixture['rows']}
    if profile['kvmem']!=old_profile['kvmem']:
        raise ValueError('Historical per-request KV controls differ')
    for key in ['model','tokenizer','hf_config_path','execution_mode','prefill_chunk_tokens','max_num_batched_tokens']:
        if profile[key]!=old_profile[key]:raise ValueError('Historical model/runtime controls differ: '+key)
    scorer=ruler_protocol.load_scorer(root/'upstream');cases=[];predictions={a:{} for a in ['baseline','layerwise_pressure']}
    response_ids=set()
    for row in fixture['rows']:
        if row!=old_rows[row['id']]:raise ValueError('Original task/sample identity changed')
        item=dict(id=row['id'],task=row['task'],arms={})
        for arm,parent in [('baseline',baseline),('layerwise_pressure',root)]:
            directory=parent/'formal/sequential/cases'/row['id']
            result,response,request,evidence=(p.read(directory/n) for n in ['result.json','response.json','request.json','evidence.json'])
            if p.sha(directory/'response.json')!=result['response_sha256'] or p.sha(directory/'evidence.json')!=result['evidence_sha256']:
                raise ValueError('A first answer or its execution evidence changed')
            if (request['input_sha256']!=row['served_sha256'] or request['query_span']!=row['served_query'] or
                    request['sampling']!=p.case_sampling(fixture,row) or
                    p.sha(parent/'inputs'/(row['id']+'.i32'))!=row['served_sha256']):
                raise ValueError('Historical input/Query/sampling controls differ')
            score=p.score_case(row,response,scorer,starts_in_thinking=False)
            if result['official_case_score']!=score['official_case_score']:raise ValueError('Official rescore differs')
            if response['response_id'] in response_ids:raise ValueError('A response was reused across arms')
            response_ids.add(response['response_id'])
            audit_events(evidence['events'],evidence['refresh'],response,row,arm=='layerwise_pressure')
            predictions[arm][(row['task'],row['index'])]=response['text']
            item['arms'][arm]=dict(score=score['official_case_score'],truncated=response['finish_reason']=='length',
                seconds=result['seconds'],response_sha256=result['response_sha256'],evidence_sha256=result['evidence_sha256'])
        cases.append(item)
    official={arm:ruler_protocol.score_suite(fixture['rows'],values,scorer,samples_per_task=2,starts_in_thinking=False)
        for arm,values in predictions.items()}
    ds=[r['arms']['layerwise_pressure']['score']-r['arms']['baseline']['score'] for r in cases]
    result=dict(cases=26,baseline_root=str(baseline),new_root=str(root),
        macro_scores={a:v['macro_score'] for a,v in official.items()},
        macro_delta=official['layerwise_pressure']['macro_score']-official['baseline']['macro_score'],
        per_task={task:{a:v['per_task'][task]['score'] for a,v in official.items()} for task in official['baseline']['per_task']},
        paired=dict(wins=sum(d>0 for d in ds),losses=sum(d<0 for d in ds),ties=sum(d==0 for d in ds)),
        truncated={a:sum(c['arms'][a]['truncated'] for c in cases) for a in official},rows=cases,
        new_formal_answers=26,baseline_answers_reused_unchanged=26,formal_answers_resampled=0,
        baseline_workers=old_profile['max_num_seqs'],new_workers=profile['max_num_seqs'],
        input_query_sampling_model_precision_budgets_equal=True,
        official_rescore_verified=True,
        limitations=['Historical baseline is serial eager; new execution is concurrent eager.',
            'Only original first two cases per each of thirteen tasks; no full-benchmark or statistical-stability claim.',
            'Independent pressure observes all sixteen target layers; historical shared pressure aggregates spread3.'])
    output.mkdir(exist_ok=True,parents=True);p.save(output/'comparison.json',result)
    print({k:v for k,v in result.items() if k not in ['rows','per_task','limitations']},flush=True)
    return result

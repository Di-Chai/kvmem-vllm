"""Finite qualification and original-position 13x2 experiment, no inspection timer."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import recall_comparison_protocol as p
import run_layerwise_query_diagnostic as serial

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'artifacts/layerwise-pressure-concurrency-20261103-r1/formal-c4/independent'
MODULES=['original_rope.py','window_bridge.py','mtp_runtime.py','packed_assembly.py',
         'window_assembly.py','action_scoring.py','action_attention.py','layerwise_query.py','layerwise_pressure.py',
         'capture_runtime.py']


def prepare_arm(root,ids,port):
    root.mkdir(parents=True)
    runtime=root/'runtime-v1'
    shutil.copytree(SOURCE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in MODULES:
        shutil.copy2(REPO/'src/vllm_kvmem'/name,runtime/'src/vllm_kvmem'/name)
    for name in ['run_concurrent_layerwise_ruler.py',Path(__file__).name]:
        shutil.copy2(REPO/'scripts'/name,runtime/'scripts'/name)
    # Transplant only this opt-in CLI addition into the earlier frozen launcher.
    serve=runtime/'src/vllm_kvmem/serve.py';text=serve.read_text()
    anchor='    args = parser.parse_args(argv)\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,"    parser.add_argument('--layerwise-original-rope', action='store_true')\n"+anchor)
    local=(REPO/'src/vllm_kvmem/serve.py').read_text()
    start=local.index('    if args.layerwise_original_rope:\n')
    end=local.index('    if args.layerwise_prefill_pressure:\n',start)
    anchor='    plan = build_launch(profile, base, run_dir)\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+local[start:end]);serve.write_text(text)
    fixture=deepcopy(p.read(SOURCE/'fixture.json'))
    fixture['rows']=[r for r in fixture['rows'] if r['id'] in ids]
    if {r['id'] for r in fixture['rows']}!=set(ids):raise ValueError('Unknown original sample')
    fixture.update(cases=len(ids),qualification_ids=[r['id'] for r in fixture['rows']],
        qualification_only=len(ids)!=26,rope_position_policy='original',
        source_fixture=str(SOURCE/'fixture.json'),source_fixture_sha256=p.sha(SOURCE/'fixture.json'),
        diagnostic_scope='Original versus dense RoPE; per-layer accumulated-attention pressure and per-layer Mean-K Query unchanged.',
        diagnostic_limits=['Original positions apply to target and MTP Q/K from the start of the original input.',
            'Physical slots and causal masks remain dense; historical tensors are recomputed independently.',
            'Same four-way eager, NVFP4, input, sampling, prefix, output caps and KV budgets as dense reference.',
            'Qualification answers are separate; formal answers run fresh once. No score-based retries.'])
    profile=p.read(SOURCE/'profile.json');profile.update(port=port,served_model_name='layerwise-original-rope-'+root.name)
    config=p.read(Path(profile['model'])/'config.json');config=config.get('text_config',config)
    if config.get('rope_parameters',{}).get('rope_type') != 'default':
        raise ValueError('Only original default RoPE frequencies are qualified')
    if any(r['served_input_tokens']+fixture['task_max_tokens'][r['task']]+profile['draft_tokens'] >= profile['max_model_len']
           for r in fixture['rows']):
        raise ValueError('Original inputs and output caps exceed the frozen model length')
    fixture['rope_lookup_extension']=dict(original_capacity=config['max_position_embeddings'],
        capacity=profile['max_model_len'],existing_prefix_bytes_preserved=True,
        frequency_and_scaling_unchanged=True)
    (root/'inputs').mkdir()
    for row in fixture['rows']:
        source=SOURCE/'inputs'/(row['id']+'.i32')
        if p.sha(source)!=row['served_sha256']:raise ValueError('Original input hash differs')
        shutil.copy2(source,root/'inputs'/source.name)
    shutil.copytree(SOURCE/'upstream',root/'upstream')
    p.save(root/'fixture.json',fixture);p.save(root/'profile.json',profile)
    p.save(root/'runtime-proof.json',{str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
                                   if f.is_file() and '__pycache__' not in f.parts})
    p.save(root/'prepared.json',dict(at=time.time(),ids=fixture['qualification_ids'],expected_runs=len(ids),
        profile_sha256=p.sha(root/'profile.json'),fixture_sha256=p.sha(root/'fixture.json'),
        runtime_proof_sha256=p.sha(root/'runtime-proof.json')))
    p.save(root/'status.json',dict(status='prepared',expected=len(ids),at=time.time()))
    serial.verify(root)


def prepare(root):
    if root.exists():raise ValueError('Use a new experiment directory')
    root.mkdir(parents=True)
    fixture=p.read(SOURCE/'fixture.json')
    prepare_arm(root/'qualification',{'niah_single_1-0000','niah_single_1-0001','niah_multivalue-0001','cwe-0001'},19042)
    prepare_arm(root/'formal',{r['id'] for r in fixture['rows']},19043)
    shutil.copy2(__file__,root/'run_campaign.py')
    p.save(root/'plan.json',dict(baseline=str(SOURCE),stages=['qualification','formal'],
        qualification_cases=4,formal_cases=26,workers=4,scheduled_inspections_enabled=False,
        completion_gates='Technical correctness and position evidence only; never scores.',
        controller_sha256=p.sha(root/'run_campaign.py'),
        prepared_hashes={name:p.sha(root/name/'prepared.json') for name in ['qualification','formal']}))
    p.save(root/'status.json',dict(status='prepared',scheduled_inspections_enabled=False,at=time.time()))
    print(dict(root=str(root),qualification=4,formal=26),flush=True)


def analyze(root,baseline):
    import ruler_protocol
    from run_concurrent_layerwise_ruler import audit_events
    new=root/'formal';fixture=p.read(new/'fixture.json');old_fixture=p.read(baseline/'fixture.json')
    if fixture['rows']!=old_fixture['rows']:raise ValueError('Comparison cohort changed')
    np,op=p.read(new/'profile.json'),p.read(baseline/'profile.json')
    if {k:v for k,v in np.items() if k not in ['port','served_model_name']} != {k:v for k,v in op.items() if k not in ['port','served_model_name']}:
        raise ValueError('Original versus dense execution controls differ')
    scorer=ruler_protocol.load_scorer(new/'upstream');pred={a:{} for a in ['dense','original']};cases=[]
    for row in fixture['rows']:
        item=dict(id=row['id'],task=row['task'],arms={})
        for arm,parent in [('dense',baseline),('original',new)]:
            folder=parent/'formal/sequential/cases'/row['id']
            result,response,request,evidence=[p.read(folder/n) for n in ['result.json','response.json','request.json','evidence.json']]
            if any(p.sha(folder/file)!=result[key] for file,key in [('response.json','response_sha256'),('evidence.json','evidence_sha256')]):
                raise ValueError('Saved first answer/evidence changed')
            if (p.sha(parent/'inputs'/(row['id']+'.i32'))!=row['served_sha256'] or
                    request['query_span']!=row['served_query'] or request['sampling']!=p.case_sampling(fixture,row)):
                raise ValueError('Original input/sampling changed')
            audit_events(evidence['events'],evidence['refresh'],response,row,True)
            score=p.score_case(row,response,scorer,starts_in_thinking=False)
            if score['official_case_score']!=result['official_case_score']:raise ValueError('Official score changed')
            pred[arm][(row['task'],row['index'])]=response['text']
            item['arms'][arm]=dict(score=score['official_case_score'],truncated=response['finish_reason']=='length',
                seconds=result['seconds'],response_sha256=result['response_sha256'])
        cases.append(item)
    scores={arm:ruler_protocol.score_suite(fixture['rows'],values,scorer,samples_per_task=2,starts_in_thinking=False) for arm,values in pred.items()}
    delta=[x['arms']['original']['score']-x['arms']['dense']['score'] for x in cases]
    output=dict(cases=26,macro_scores={a:s['macro_score'] for a,s in scores.items()},
        per_task={t:{a:s['per_task'][t]['score'] for a,s in scores.items()} for t in scores['dense']['per_task']},
        paired=dict(wins=sum(d>0 for d in delta),losses=sum(d<0 for d in delta),ties=sum(d==0 for d in delta)),
        truncated={a:sum(c['arms'][a]['truncated'] for c in cases) for a in scores},rows=cases,
        same_input_sampling_budgets_precision_and_concurrency=True,official_rescore_verified=True,
        baseline_answers_reused_unchanged=26,formal_answers_resampled=0,
        limitations=['13 tasks with two cases each; no statistical-stability claim.',
            'Same concurrency and configuration; actual dynamic batch membership can differ.'])
    p.save(root/'analysis/comparison.json',output)
    return output


def run(root):
    from run_recall_comparison import termination_guard
    plan=p.read(root/'plan.json')
    if p.sha(Path(__file__))!=plan['controller_sha256']:raise ValueError('Finite controller changed')
    p.save(root/'process.json',dict(pid=__import__('os').getpid(),at=time.time()))
    with termination_guard():
        try:
            for name in plan['stages']:
                arm=root/name
                if p.sha(arm/'prepared.json')!=plan['prepared_hashes'][name]:raise ValueError('Frozen stage changed')
                serial.verify(arm)
                p.save(root/'status.json',dict(status='running',stage=name,active_root=str(arm),scheduled_inspections_enabled=False,at=time.time()))
                p.save(REPO/'artifacts/value-recall-active.json',dict(queue_root=str(root),root=str(arm),campaign=str(arm),
                    status=str(arm/'status.json'),process=str(arm/'process.json'),experiment_type='original-rope-'+name,at=time.time()))
                command=[sys.executable,'-B',str(arm/'runtime-v1/scripts/run_concurrent_layerwise_ruler.py'),'--root',str(arm)]
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                    p.save(root/'child.json',dict(pid=child.pid,root=str(arm),stage=name,at=time.time()))
                    code=child.wait()
                if code or p.read(arm/'status.json')['status']!='completed':raise RuntimeError('Stage failed: '+name)
            result=analyze(root,Path(plan['baseline']))
            p.save(root/'status.json',dict(status='completed',cases=26,macro_scores=result['macro_scores'],
                scheduled_inspections_enabled=False,at=time.time()))
        except BaseException as error:
            p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','analyze']);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':prepare(root)
    elif args.action=='analyze':analyze(root,Path(p.read(root/'plan.json')['baseline']))
    else:
        # The frozen controller lives directly in the experiment directory.
        REPO=Path(p.read(root/'plan.json')['baseline']).parents[3]
        run(root)

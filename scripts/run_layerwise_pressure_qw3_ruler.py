"""Frozen full RULER cohort, native qw3 generation protocol, layerwise pressure.

Only generation/prompt controls are adopted from qw3. The existing 260 questions,
per-case seeds, NVFP4 model and per-owner KV budgets remain those of this repo.
"""
import argparse
from copy import deepcopy
import csv
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import recall_comparison_protocol as p
import ruler_protocol as ruler
import run_layerwise_query_diagnostic as driver


def prepare(root):
    import numpy as np
    from transformers import AutoTokenizer
    from benchmark_ruler import render_case

    repo = Path(__file__).resolve().parents[1]
    baseline = repo/'artifacts/layerwise-meank-ruler260-20261102-r2'
    qualified = repo/'artifacts/layerwise-pressure-concurrency-20261103-r1/formal-c4/independent'
    qw3 = Path('/home/chaidi/qw3/benchmark/attention_multilayer_qa_20261011')
    driver.verify(qualified)
    driver.verify(baseline)
    if root.exists():
        raise ValueError('Use a fresh version; never overwrite answers or a frozen campaign')
    old = p.read(baseline/'fixture.json')
    ruler.validate_cohort(old['rows'], 20)
    expected = dict(enable_thinking=True, thinking_budget=4096, thinking_summary=False,
        max_tokens=8192, temperature=1.0, top_p=.95, top_k=20, min_p=0.0,
        presence_penalty=0.0, repetition_penalty=1.0)
    sources = sorted((qw3/'results/attention/cases/niah').glob('*/request_config.json'))
    if len(sources) != 180:
        raise ValueError('Historical native qw3 evidence differs')
    for path in sources:
        d = p.read(path)
        if any(d.get(k) != v for k, v in expected.items()):
            raise ValueError('Native qw3 generation controls are inconsistent')
    profile = p.read(qualified/'profile.json')
    profile.update(port=19045, served_model_name='layerwise-pressure-qw3-ruler260',
        reasoning_end_str='\n\nConsidering the limited time by the user, I have to give the '
        'solution based on the thinking directly now.\n</think>')
    if (profile['kvmem']['select_budget'], profile['kvmem']['gen_budget'],
        profile['kvmem']['block_tokens'], profile['kvmem']['sink_blocks'],
        profile['kvmem']['recent_blocks'], profile['max_num_seqs']) != (36864, 16384, 32, 41, 0, 4):
        raise ValueError('Qualified budgets or concurrency changed')
    root.mkdir(parents=True)
    p.save(root/'status.json', dict(status='preparing', at=time.time()))
    runtime = root/'runtime-v1'
    shutil.copytree(qualified/'runtime-v1', runtime,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    # Model/runtime inference code stays byte-identical to the qualified dense
    # position implementation. Only the driver and scoring dispatch change.
    for name in ['run_concurrent_layerwise_ruler.py', 'run_layerwise_query_diagnostic.py', 'benchmark_ruler.py',
                 Path(__file__).name]:
        shutil.copy2(repo/'scripts'/name, runtime/'scripts'/name)
    shutil.copytree(baseline/'upstream', root/'upstream')
    (root/'inputs').mkdir()
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    rows = []
    for original in p.execution_rows(old['rows']):
        row = deepcopy(original)
        # This tokenizer's "medium" template option emits no extra system
        # instruction; verify the entire wrapper against native qw3's framing.
        tokens, span, question, rendered = render_case(tokenizer, row['task'], row['input'], reasoning_effort='medium')
        expected_prompt = '<|im_start|>user\n'+row['input']+'<|im_end|>\n<|im_start|>assistant\n<think>\n'
        if question != row['question'] or rendered != expected_prompt:
            raise ValueError('Raw public question or native qw3 chat template changed')
        path = root/'inputs'/(row['id']+'.i32')
        np.asarray(tokens, dtype='<i4').tofile(path)
        for key in ['prefix_proof', 'thinking_source_input_sha256', 'thinking_source_input_tokens']:
            row.pop(key, None)
        row.update(served_query=span, served_input_tokens=len(tokens), served_sha256=p.sha(path),
            rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            prior_served_sha256=original['served_sha256'],
            raw_input_sha256=hashlib.sha256(row['input'].encode()).hexdigest())
        rows.append(row)
        if len(rows) % 26 == 0:
            print(dict(prepared_inputs=len(rows), expected=260), flush=True)
    sampling = {k:expected[k] for k in ['max_tokens','temperature','top_p','top_k','min_p',
                                        'presence_penalty','repetition_penalty']}
    sampling['thinking_token_budget'] = expected['thinking_budget']
    limits = [
        'Historical shared-pressure baseline uses non-thinking, official assistant answer prefix, '
        'short task output caps and an extra system message; this is not a single-variable ablation.',
        'New run is four-owner eager; historical shared-pressure baseline is one-owner eager.',
        'Independent pressure observes sixteen layers; historical shared pressure aggregates spread3.',
        'Native qw3 evidence has 180 different NIAH cases and Q8_0 weights; only its generation '
        'and user-only chat protocol are adopted. Original 260 questions/seeds and NVFP4 stay fixed.',
        'Dense compacted RoPE policy is unchanged from qualified independent-pressure run.',
        'Decode pressure remains accumulated attention; with max output 8192 below Gen 16384, '
        'no post-Query pressure should be needed. Unsupported refresh fails explicitly.',
    ]
    fixture = dict(rows=rows, cases=260, tasks=13, samples_per_task=20, target_length=262144,
        seed=old['seed'], starts_in_thinking=True, sampling=sampling, task_max_tokens=None,
        generation_limit_policy='native_qw3_thinking4096_total8192',
        answer_prefix_policy='none', prompt_protocol='Qwen user-only chat, open <think>, no assistant answer prefill',
        chat_template_reasoning_effort='medium',
        tokenizer=profile['tokenizer'], tokenizer_sha256=p.sha(Path(profile['tokenizer'])/'tokenizer.json'),
        source_fixture=str(baseline/'fixture.json'), source_fixture_sha256=p.sha(baseline/'fixture.json'),
        input_directory='inputs', upstream_path='upstream', upstream=old['upstream'],
        arms=['sequential'], workers=4, layerwise_workers=4, layerwise_prefill_pressure=True,
        layerwise_optimized=True, layerwise_full_diagnostics=False, qualification_only=False,
        methods=['mean_k_sequential_layerwise'], cohort='ruler260',
        diagnostic_scope='Original 13x20 RULER, layerwise accumulated-attention prefill pressure '
        'and sequential layerwise Mean-K Query, native qw3 generation controls.',
        diagnostic_limits=limits, rope_position_policy='compacted',
        fixed_controls=dict(active_kv=36864, gen_kv=16384, block_tokens=32, sink_blocks=41,
            recent_blocks=0, thinking=True, answer_prefix='none',
            prefill_pressure='accumulated_attention_per_layer', decode_pressure='accumulated_attention'))
    p.save(root/'profile.json', profile)
    p.save(root/'fixture.json', fixture)
    proof = {str(path.relative_to(runtime)):p.sha(path) for path in runtime.rglob('*')
             if path.is_file() and '__pycache__' not in path.parts}
    model_proof = {k:v for k,v in proof.items() if k.startswith(('src/', 'csrc/'))}
    previous_proof = p.read(qualified/'runtime-proof.json')
    if any(previous_proof.get(k) != v for k,v in model_proof.items()):
        raise ValueError('Qualified inference implementation changed')
    p.save(root/'runtime-proof.json', proof)
    baseline_bindings = {}
    for row in old['rows']:
        case = baseline/'formal/sequential/cases'/row['id']
        for name in ['request.json', 'response.json', 'result.json', 'evidence.json']:
            path = case/name
            baseline_bindings[str(path)] = p.sha(path)
    p.save(root/'historical-baseline-proof.json', baseline_bindings)
    p.save(root/'alignment-proof.json', dict(native_qw3_root=str(qw3), native_qw3_cases=180,
        request_files={str(path):p.sha(path) for path in sources}, native_generation=expected,
        seed_policy='Keep each original 260-case seed; do not map seeds from a different 180-case cohort.',
        requested_cases=260, inference_source=str(qualified), inference_files_verified=len(model_proof),
        source_runtime_proof_sha256=p.sha(qualified/'runtime-proof.json'),
        profile_changes_from_qualified={k:dict(old=p.read(qualified/'profile.json').get(k),new=v)
            for k,v in profile.items() if p.read(qualified/'profile.json').get(k)!=v},
        kv_controls_identical=True, raw_questions_and_references_identical=True,
        limitations=limits))
    p.save(root/'prepared.json', dict(at=time.time(), expected_runs=260,
        profile_sha256=p.sha(root/'profile.json'), fixture_sha256=p.sha(root/'fixture.json'),
        runtime_proof_sha256=p.sha(root/'runtime-proof.json')))
    p.save(root/'campaign-plan.json', dict(expected=260, baseline_root=str(baseline),
        prepared_sha256=p.sha(root/'prepared.json'),
        historical_baseline_proof_sha256=p.sha(root/'historical-baseline-proof.json'),
        alignment_proof_sha256=p.sha(root/'alignment-proof.json'),
        scheduled_inspections_enabled=False, automatic_retries=False, at=time.time()))
    driver.verify(root)
    p.save(root/'status.json', dict(status='prepared', completed=0, expected=260, at=time.time()))


def analyze(root):
    driver.verify(root)
    plan = p.read(root/'campaign-plan.json')
    baseline = Path(plan['baseline_root'])
    for name in ['historical-baseline-proof', 'alignment-proof']:
        if p.sha(root/(name+'.json')) != plan[name.replace('-', '_')+'_sha256']:
            raise ValueError('Campaign provenance changed')
    for path,digest in p.read(root/'historical-baseline-proof.json').items():
        if p.sha(path) != digest:
            raise ValueError('Historical first answer or evidence changed')
    fixtures = {'shared_pressure_historical':p.read(baseline/'fixture.json'),
                'layerwise_pressure_qw3':p.read(root/'fixture.json')}
    rows_by_arm = {arm:{r['id']:r for r in fixture['rows']} for arm,fixture in fixtures.items()}
    names = list(fixtures)
    if set(rows_by_arm[names[0]]) != set(rows_by_arm[names[1]]):
        raise ValueError('Historical cohort differs')
    scorer = ruler.load_scorer(root/'upstream')
    predictions = {a:{} for a in fixtures}
    cases = []
    for row in fixtures[names[1]]['rows']:
        before = rows_by_arm[names[0]][row['id']]
        for key in ['input', 'outputs', 'seed', 'task', 'index', 'upstream_row_sha256', 'question']:
            if row[key] != before[key]:
                raise ValueError('Original sample identity changed: '+key)
        item = dict(id=row['id'], task=row['task'], arms={})
        for arm,parent in zip(names, [baseline,root]):
            r = rows_by_arm[arm][row['id']]
            case = parent/'formal/sequential/cases'/r['id']
            response, result, request = (p.read(case/name) for name in ['response.json','result.json','request.json'])
            if (p.sha(case/'response.json') != result['response_sha256'] or
                request['sampling'] != p.case_sampling(fixtures[arm],r) or
                request['input_sha256'] != r['served_sha256']):
                raise ValueError('Frozen first answer, request or score changed')
            score = p.score_case(r,response,scorer,starts_in_thinking=fixtures[arm]['starts_in_thinking'])
            if score['official_case_score'] != result['official_case_score']:
                raise ValueError('Official rescore differs')
            predictions[arm][(r['task'],r['index'])] = response['text']
            item['arms'][arm] = dict(score=score['official_case_score'],
                truncated=response['finish_reason']=='length', thinking_complete=score['thinking_complete'],
                response_sha256=result['response_sha256'], completion_tokens=result['completion_tokens'])
        item['delta'] = item['arms'][names[1]]['score']-item['arms'][names[0]]['score']
        cases.append(item)
    scores = {arm:ruler.score_suite(fixtures[arm]['rows'],pred,scorer,samples_per_task=20,
                  starts_in_thinking=fixtures[arm]['starts_in_thinking']) for arm,pred in predictions.items()}
    report = dict(cases=260, macro_scores={a:s['macro_score'] for a,s in scores.items()},
        macro_delta=scores[names[1]]['macro_score']-scores[names[0]]['macro_score'],
        per_task={task:{a:s['per_task'][task]['score'] for a,s in scores.items()} for task in ruler.TASK_FAMILIES},
        paired=dict(wins=sum(c['delta']>0 for c in cases),losses=sum(c['delta']<0 for c in cases),
                    ties=sum(c['delta']==0 for c in cases)),
        truncated={a:sum(c['arms'][a]['truncated'] for c in cases) for a in names},
        thinking_incomplete={a:sum(c['arms'][a]['thinking_complete'] is False for c in cases) for a in names},
        raw_questions_seeds_model_precision_budgets_equal=True, generation_protocol_equal=False,
        historical_only_not_single_variable_ablation=True, baseline_root=str(baseline), new_root=str(root),
        limits=fixtures[names[1]]['diagnostic_limits'], rows=cases)
    p.save(root/'analysis/historical-comparison.json',report)
    with (root/'analysis/historical-per-task.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['task',*names,'delta'])
        for task,values in report['per_task'].items():
            writer.writerow([task,*[values[a] for a in names],values[names[1]]-values[names[0]]])
    return report


def expand(root, source):
    """Preserve an unstarted four-owner version, freeze requested eight owners."""
    driver.verify(source)
    stopped=p.read(source/'user-concurrency-switch.json')
    if stopped['answers_preserved'] or any((source/'formal/sequential/cases').glob('*/response.json')):
        raise ValueError('A concurrency handoff cannot resample existing first answers')
    if root.exists():
        raise ValueError('Eight-owner version must have a fresh directory')
    root.mkdir(parents=True)
    for name in ['inputs','upstream','runtime-v1']:
        shutil.copytree(source/name,root/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,root/'runtime-v1/scripts'/Path(__file__).name)
    fixture=p.read(source/'fixture.json')
    fixture.update(workers=8,layerwise_workers=8)
    fixture['diagnostic_limits']=[s.replace('four-owner','eight-owner') for s in fixture['diagnostic_limits']]
    profile=p.read(source/'profile.json')
    profile.update(max_num_seqs=8,port=19046,served_model_name='layerwise-pressure-qw3-ruler260-c8')
    p.save(root/'fixture.json',fixture);p.save(root/'profile.json',profile)
    for name in ['historical-baseline-proof.json','alignment-proof.json']:
        shutil.copy2(source/name,root/name)
    alignment=p.read(root/'alignment-proof.json')
    alignment['profile_changes_from_qualified']['max_num_seqs']=dict(old=4,new=8)
    for key in ['port','served_model_name']:
        alignment['profile_changes_from_qualified'][key]['new']=profile[key]
    alignment['limitations']=fixture['diagnostic_limits']
    alignment['concurrency_authorization']='Explicit user request for eight concurrent owners; completion refills a slot.'
    p.save(root/'alignment-proof.json',alignment)
    runtime=root/'runtime-v1'
    proof={str(path.relative_to(runtime)):p.sha(path) for path in runtime.rglob('*')
           if path.is_file() and '__pycache__' not in path.parts}
    p.save(root/'runtime-proof.json',proof)
    p.save(root/'prepared.json',dict(at=time.time(),expected_runs=260,
        profile_sha256=p.sha(root/'profile.json'),fixture_sha256=p.sha(root/'fixture.json'),
        runtime_proof_sha256=p.sha(root/'runtime-proof.json')))
    plan=p.read(source/'campaign-plan.json')
    plan.update(prepared_sha256=p.sha(root/'prepared.json'),
        historical_baseline_proof_sha256=p.sha(root/'historical-baseline-proof.json'),
        alignment_proof_sha256=p.sha(root/'alignment-proof.json'),supersedes_root=str(source),
        supersedes_stop_proof_sha256=p.sha(source/'user-concurrency-switch.json'),at=time.time())
    p.save(root/'campaign-plan.json',plan)
    driver.verify(root)
    p.save(root/'status.json',dict(status='prepared',workers=8,completed=0,expected=260,at=time.time()))


def run(root):
    driver.verify(root)
    plan = p.read(root/'campaign-plan.json')
    if p.sha(root/'prepared.json') != plan['prepared_sha256']:
        raise ValueError('Frozen stage binding changed')
    with (root/'campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active = root.parent/'value-recall-active.json'
        if active.exists():
            prior = p.read(active)
            state = Path(prior['status'])
            if Path(prior.get('queue_root','')) != root and p.read(state)['status'] != 'completed':
                predecessor=Path(plan.get('supersedes_root',''))
                if (prior.get('queue_root')!=str(predecessor) or
                    p.sha(predecessor/'user-concurrency-switch.json')!=plan['supersedes_stop_proof_sha256']):
                    raise ValueError('Another campaign is active; leave it untouched')
                for pid in [p.read(predecessor/'process.json')['pid'],p.read(predecessor/'process.json')['server_pid']]:
                    stat=Path('/proc')/str(pid)/'stat'
                    if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0] not in ['Z','X']:
                        raise ValueError('Previous owned process has not stopped')
        p.save(active, dict(queue_root=str(root),root=str(root),campaign=str(root),
            status=str(root/'status.json'),process=str(root/'process.json'),
            experiment_type='layerwise-pressure-qw3-ruler260',at=time.time()))
        p.save(root/'queue-process.json',dict(pid=os.getpid(),at=time.time(),scheduled_inspections_enabled=False))
        command=[sys.executable,'-B',str(root/'runtime-v1/scripts/run_concurrent_layerwise_ruler.py'),
                 '--root',str(root)]
        with (root/'formal.log').open('x') as log:
            child=subprocess.Popen(command,cwd=root/'runtime-v1',stdout=log,stderr=subprocess.STDOUT)
        p.save(root/'child.json',dict(pid=child.pid,command=command,at=time.time()))
        if child.wait()!=0:
            raise RuntimeError('Formal stage failed; preserve first answers and evidence without automatic retries')
        report=analyze(root)
        p.save(root/'queue-status.json',dict(status='completed',cases=260,macro_scores=report['macro_scores'],
            official_rescore_verified=True,scheduled_inspections_enabled=False,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','expand','run','analyze'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='expand':
            if args.source is None:raise ValueError('Eight-owner handoff requires a source')
            expand(root,args.source.resolve())
        else:globals()[args.action](root)
    except BaseException as error:
        if args.action=='run':
            p.save(root/'queue-status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise

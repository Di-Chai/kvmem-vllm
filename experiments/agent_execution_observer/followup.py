"""Independent native System One format, causal replay and goal-scope checks.

Baseline outputs remain read-only. Frozen-state rereads and new sequential
replay are distinct experiments; neither is human-gold segmentation accuracy.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

from observer import Memory, digest, request_for, save, validate


LETTER_SUFFIX = (' Return ONLY the single option LETTER (A, B, C, etc.). '
                 'Do not return an option name, a stage ID such as S001, JSON, or an explanation.')
CONTROL_STEPS = (1, 12, 25, 33, 40, 57)
GOAL_CASES = (
    (6, 'inspect-without-external-obstacle',
     'Continue surveying the existing code and test architecture. Determine whether any observed external '
     'prerequisite currently prevents this survey.', ('completed','in_progress','uncertain'),
     'No external blocking evidence is present in the successful reads; completion granularity is unspecified.'),
    (16, 'create-registry-file',
     'Create the error-class-registry.ts source file on disk. This goal is only file creation, not validation '
     'of the entire feature.', ('completed',), 'The current tool return confirms this exact file was created.'),
    (16, 'validate-whole-feature',
     'Implement and validate the complete requested errorStack feature, including its integration and passing '
     'tests. A single source file being created is insufficient.', ('in_progress','uncertain'),
     'Current and visible past evidence do not establish whole-feature validation.'),
    (33, 'current-test-attempt-passes',
     'The just-executed test-suite attempt must finish with all executed tests passing. Assess this completed '
     'attempt, even if later repairs may remain possible.', ('failed',), 'Current test output reports four failures.'),
    (38, 'current-test-attempt-passes',
     'The just-executed test-suite attempt must finish with all executed tests passing. Assess this completed '
     'attempt, even if later repairs may remain possible.', ('failed',), 'Current test output reports one failure.'),
    (40, 'current-test-attempt-passes',
     'The just-executed test-suite attempt must finish with all executed tests passing. Assess only this attempt, '
     'not final delivery of the entire feature.', ('completed',), 'Current test output reports the tests passed.'),
    (52, 'current-tests-and-build-pass',
     'The just-executed test suite and TypeScript build must both pass. Assess only this verification attempt, '
     'not subsequent commit or final delivery.', ('completed',), 'Current test/build output supports successful checks.'),
)


def explicit_letters(payload):
    result = copy.deepcopy(payload)
    for question in result['questions'].values():
        if question['type'] != 'choice':
            raise ValueError('This format intervention is defined only for choice questions')
        question['instructions'] += LETTER_SUFFIX
    return result


def bounded_goal(payload, name, goal):
    """All independent reads see the same named goal; no answer-to-answer inference."""
    result = explicit_letters(payload)
    result['state']['assessment_target'] = dict(target_id=name, goal=goal,
        scope='Only this explicit assessment target; do not substitute an unrelated remembered stage.')
    result['questions'] = {k: result['questions'][k] for k in ('outcome','evidence')}
    result['questions']['outcome']['instructions'] = (
        'Assess ONLY state.assessment_target. Select its current status using observed execution evidence. '
        'For a completed test attempt whose goal is passing tests, explicit failed tests mean failed, '
        'even if the overall repair task may continue. File creation and whole-feature validation are '
        'different goals. Do not infer success from promises or TaskUpdate alone.' + LETTER_SUFFIX)
    result['questions']['evidence']['instructions'] = (
        'Choose the current-Step excerpt most directly supporting the status of state.assessment_target. '
        'If none supports it choose uncertain. The existence of a citation alone does not prove support.' + LETTER_SUFFIX)
    return result


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare(dataset, baseline, output):
    rows = read_jsonl(dataset)
    old = read_jsonl(baseline/'decisions.jsonl')
    if len(rows)!=57 or len(old)!=57 or len({r['window_id'] for r in rows})!=1:
        raise ValueError('Followup requires the completed 57-Step continuous cohort')
    if [r['step'] for r in rows] != list(range(1,58)):
        raise ValueError('Require a complete causal Step sequence')
    if json.loads((baseline/'summary.json').read_text())['status']!='completed':
        raise ValueError('Baseline is incomplete')
    if digest(dataset)!=json.loads((baseline/'run.json').read_text())['dataset_sha256']:
        raise ValueError('Baseline dataset identity changed')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'dataset.json',rows)
    save(output/'baseline-decisions.json',old)
    identities=[]
    for step,previous in zip(rows,old):
        if (step['step'],step['window_id'])!=(previous['step'],previous['window_id']):
            raise ValueError('Baseline/Step alignment differs')
        name=previous['call_id']+'.request.json'
        payload=json.loads((baseline/'calls'/name).read_text())
        save(output/'baseline-requests'/f'{step["step"]:04}.json',payload)
        identities.append(dict(step=step['step'],original_request_sha256=digest(baseline/'calls'/name),
                               frozen_request_sha256=digest(output/'baseline-requests'/f'{step["step"]:04}.json')))
    save(output/'goal-references.json',[dict(step=step,case=name,goal=goal,accepted_outcomes=list(expected),
         basis=basis,reference_type='predeclared_goal_scope_check_not_human_segmentation_gold')
         for step,name,goal,expected,basis in GOAL_CASES])
    hashes={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    save(output/'manifest.json',dict(status='prepared',prepared_at=time.time(),
        input_files=hashes,source_dataset_sha256=digest(dataset),
        baseline_summary_sha256=digest(baseline/'summary.json'),baseline_run_sha256=digest(baseline/'run.json'),
        requests=identities,control_steps=list(CONTROL_STEPS),format_suffix=LETTER_SUFFIX,
        design=['57 frozen-state explicit-letter rereads; six fresh original-prompt controls.',
                'Separate causal 57-Step explicit-letter replay with its own newly generated memory.',
                'Seven predefined goal-scope checks; references remain outside model input.'],
        adaptive_design=True,independent_human_gold=False,
        limitations=['Saved baseline memory is used only in frozen-state rereads.',
                     'A format-conforming answer does not establish semantic correctness.',
                     'Goal-scope checks supply a goal; they do not demonstrate autonomous goal discovery.',
                     'Existing four independent reads and extractive goal hints remain prototype limitations.']))


def check_inputs(inputs):
    manifest=json.loads((inputs/'manifest.json').read_text())
    for relative,expected in manifest['input_files'].items():
        if digest(inputs/relative)!=expected:
            raise ValueError('Frozen input changed: '+relative)
    return manifest


def diagnostic_summary(responses):
    grouped={}
    for row in responses:
        for qid,d in row['response']['diagnostics'].items():grouped.setdefault(qid,[]).append(d)
    return {qid:dict(reads=len(ds),argmax_not_label=sum(not d['argmax_is_label'] for d in ds),
             mean_label_mass=statistics.mean(d['label_mass'] for d in ds),
             median_label_mass=statistics.median(d['label_mass'] for d in ds)) for qid,ds in grouped.items()}


class Client:
    def __init__(self,base,output):
        self.base=base.rstrip('/');self.output=output;self.records=[]
        self.opener=build_opener(ProxyHandler({}))

    def call(self,ident,payload,**metadata):
        directory=self.output/'calls'
        save(directory/(ident+'.request.json'),payload)
        request=Request(self.base+'/v1/systemone',data=json.dumps(payload).encode(),
                        headers={'Content-Type':'application/json'})
        begin=time.perf_counter()
        try:
            with self.opener.open(request,timeout=240) as f:response=json.load(f)
            seconds=time.perf_counter()-begin
            validate(response,payload)
        except Exception as error:
            save(self.output/'failure.json',dict(id=ident,error=repr(error),metadata=metadata,
                body=error.read().decode() if isinstance(error,HTTPError) else None))
            raise
        save(directory/(ident+'.response.json'),response)
        row=dict(id=ident,seconds=seconds,response=response,**metadata)
        self.records.append(row)
        with (self.output/'responses.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        save(self.output/'progress.json',dict(completed=len(self.records),last_id=ident,last_seconds=seconds))
        print(json.dumps(dict(id=ident,seconds=round(seconds,3),
             choices={k:v['choice'] for k,v in response['answers'].items()})),flush=True)
        return response

    def finish(self,**extra):
        values=sorted(r['seconds'] for r in self.records)
        result=dict(status='completed',http_calls=len(self.records),
            input_tokens=sum(r['response']['usage']['input_tokens'] for r in self.records),
            label_reads=sum(r['response']['usage']['output_tokens'] for r in self.records),
            mean_http_seconds=statistics.mean(values),p95_http_seconds=values[max(0,math.ceil(.95*len(values))-1)],
            diagnostics=diagnostic_summary(self.records),human_gold_accuracy=None,human_gold_boundary_f1=None,**extra)
        save(self.output/'summary.json',result)
        return result


def run(inputs,output,base_url):
    check_inputs(inputs)
    output.mkdir(parents=True,exist_ok=False)
    save(output/'run.json',dict(status='running',started_at=time.time(),input_manifest_sha256=digest(inputs/'manifest.json'),
        code_sha256=digest(__file__),observer_sha256=digest(Path(__file__).with_name('observer.py')),
        base_url=base_url,interface='native /v1/systemone',automatic_retries=False))
    rows=json.loads((inputs/'dataset.json').read_text());previous=json.loads((inputs/'baseline-decisions.json').read_text())
    fixed=Client(base_url,output/'frozen-state')
    for step,old in zip(rows,previous):
        n=step['step'];payload=json.loads((inputs/'baseline-requests'/f'{n:04}.json').read_text())
        if n in CONTROL_STEPS:fixed.call(f'{n:04}-original',payload,step=n,variant='fresh_original_control')
        fixed.call(f'{n:04}-letters',explicit_letters(payload),step=n,variant='explicit_letters',
                   previous_choices={k:v['choice'] for k,v in old['native_answers'].items()})
    fixed.finish(design='Frozen saved state; not sequential new-memory replay.')
    memory=Memory();sequential=Client(base_url,output/'sequential');decisions=[]
    for step in rows:
        payload=explicit_letters(request_for(step,memory,'observer'))
        response=sequential.call(f'{step["step"]:04}',payload,step=step['step'],variant='causal_explicit_letters')
        decision=memory.apply(step,response);decisions.append(decision)
        with (sequential.output/'decisions.jsonl').open('a') as f:f.write(json.dumps(decision,ensure_ascii=False)+'\n')
    sequential.finish(design='Continuous causal replay, newly generated execution memory.',
        stage_boundaries=sum(d['stage_boundary'] is True for d in decisions),
        uncertain_links=sum(d['stage_id'] is None for d in decisions))
    save(sequential.output/'memory.json',memory.view())
    goals=Client(base_url,output/'goal-scope');checks=[]
    for ref in json.loads((inputs/'goal-references.json').read_text()):
        payload=json.loads((inputs/'baseline-requests'/f'{ref["step"]:04}.json').read_text())
        response=goals.call(f'{ref["step"]:04}-{ref["case"]}',bounded_goal(payload,ref['case'],ref['goal']),
                           step=ref['step'],variant='explicit_bound_goal')
        outcome=response['answers']['outcome']['choice']
        checks.append(dict(**ref,observed=outcome,satisfies_predeclared_check=outcome in ref['accepted_outcomes']))
    goals.finish(design='Explicit supplied goal; not autonomous goal discovery.',
        scope_checks=checks,checks_satisfied=sum(c['satisfies_predeclared_check'] for c in checks))
    check_inputs(inputs)
    save(output/'status.json',dict(status='completed',finished_at=time.time(),
        cohorts=['frozen-state','sequential','goal-scope'],model_http_calls=len(fixed.records)+len(sequential.records)+len(goals.records)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--inputs',type=Path,required=True)
    r.add_argument('--output',type=Path,required=True);r.add_argument('--base-url',default='http://127.0.0.1:18993')
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.dataset,a.baseline,a.output)
    else:run(a.inputs,a.output,a.base_url)


if __name__=='__main__':main()

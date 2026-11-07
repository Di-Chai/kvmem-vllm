"""Adaptive, paired goal-representation diagnostic on frozen past states."""
import argparse
import copy
import json
from pathlib import Path

from followup import Client, check_inputs, explicit_letters
from observer import digest, save


STEPS=(13,14,15,16,17,22)


def replace_hint(payload,hint,stage_id='S004'):
    result=copy.deepcopy(payload)
    number=result['state']['current_step_id']
    if hint['source']['step']>=number:raise ValueError('Goal hint is not from the past')
    stage=next(s for s in result['state']['execution_memory']['stages'] if s['stage_id']==stage_id)
    original=stage['goal_hint']
    stage['goal_hint']=copy.deepcopy(hint)
    question=result['questions']['stage_link']
    if json.loads(question['criteria'][stage_id])['goal_hint']!=original:
        raise ValueError('Duplicated goal representations disagree before intervention')
    # The same memory entry occurs in state and native option descriptions.
    question['criteria'][stage_id]=json.dumps(stage,ensure_ascii=False)
    return result


def run(inputs,source_file,output,base_url):
    check_inputs(inputs);source=json.loads(source_file.read_text());hint=source['goal_hint']
    if not source['quote_observed_in_original_step11_payload']:
        raise ValueError('Goal subject was not present in the original observed past payload')
    output.mkdir(parents=True,exist_ok=False)
    save(output/'run.json',dict(source_sha256=digest(source_file),inputs_sha256=digest(inputs/'manifest.json'),
        code_sha256=digest(__file__),steps=list(STEPS),adaptive_diagnostic=True,
        design='Explicit-letter stage_link only; replace S004 goal_hint consistently in state and criteria.',
        scope='Supplied prior declared goal, not autonomous goal extraction or segmentation accuracy.'))
    client=Client(base_url,output)
    for n in STEPS:
        original=json.loads((inputs/'baseline-requests'/f'{n:04}.json').read_text())
        formatted=explicit_letters(original)
        formatted['questions']={'stage_link':formatted['questions']['stage_link']}
        for variant,payload in [('original_hint',formatted),('prior_task_subject',replace_hint(formatted,hint))]:
            client.call(f'{n:04}-{variant}',payload,step=n,variant=variant)
    client.finish(scope='Six paired adaptive diagnostics; all other state/criteria and option order retained.')
    check_inputs(inputs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base-url',default='http://127.0.0.1:18993')
    a=p.parse_args();run(a.inputs,a.source,a.output,a.base_url)


if __name__=='__main__':main()

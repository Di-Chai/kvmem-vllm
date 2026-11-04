"""Freeze a new execution version; never edit a running serial campaign."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time

import recall_comparison_protocol as p
import run_layerwise_query_diagnostic as serial

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'artifacts/layerwise-pressure-balanced26-20261103-r2/shared'


def prepare(root,workers,independent,cases=26):
    if root.exists() or workers not in (2,4,8) or not 1<=cases<=26:
        raise ValueError('Need a fresh directory and explicit supported concurrency/cohort')
    root.mkdir(parents=True)
    runtime=root/'runtime-v1'
    shutil.copytree(SOURCE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['layerwise_pressure.py','layerwise_query.py']:
        shutil.copy2(REPO/'src/vllm_kvmem'/name,runtime/'src/vllm_kvmem'/name)
    for name in ['run_concurrent_layerwise_ruler.py','prepare_layerwise_pressure_concurrency.py','analyze_layerwise_pressure.py']:
        shutil.copy2(REPO/'scripts'/name,runtime/'scripts'/name)
    serve=runtime/'src/vllm_kvmem/serve.py';text=serve.read_text()
    if "parser.add_argument('--layerwise-query-native-batches'" in text:
        raise ValueError('Source CLI already has native batching; audit before patching it twice')
    anchor='    args = parser.parse_args(argv)\n';assert text.count(anchor)==1
    text=text.replace(anchor,"    parser.add_argument('--layerwise-query-native-batches', action='store_true')\n"+anchor)
    anchor="        if profile.get('execution_mode') != 'eager' or profile.get('max_num_seqs') != 1:\n"
    assert text.count(anchor)==1
    text=text.replace(anchor,"        if not args.layerwise_query_native_batches and (profile.get('execution_mode') != 'eager' or profile.get('max_num_seqs') != 1):\n")
    anchor="        plan['environment']['KVMEM_LAYERWISE_QUERY_MODE'] = args.layerwise_query_diagnostic\n"
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+"        if args.layerwise_query_native_batches:\n"
        "            if not args.layerwise_query_optimized:\n"
        "                raise ValueError('Concurrent Query requires optimized bounded assembly')\n"
        "            plan['environment']['KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES'] = '1'\n")
    serve.write_text(text)
    fixture=deepcopy(p.read(SOURCE/'fixture.json'));fixture['rows']=fixture['rows'][:cases]
    fixture.update(cases=cases,qualification_ids=[r['id'] for r in fixture['rows']],
        layerwise_workers=workers,layerwise_prefill_pressure=independent,
        source_fixture=str(SOURCE/'fixture.json'),source_fixture_sha256=p.sha(SOURCE/'fixture.json'),
        diagnostic_scope='Original 13x2 concurrent ablation; per-request K/G, inputs, sampling, prefix and caps unchanged.',
        diagnostic_limits=['Execution changes from serial eager to concurrent eager; no model/precision/budget/sampling changes.',
            'Both formal arms must use the same concurrency; prior serial answers are separate references.',
            'Independent pressure observes all sixteen target layers; the shared control retains original spread3.',
            'No answer-based retries or silent method fallback; each owner has independent maps and accepted score histories.'])
    fixture['fixed_controls']['prefill_pressure']='accumulated_attention_per_layer' if independent else 'accumulated_attention'
    fixture['qualification_only']=cases!=26
    profile=p.read(SOURCE/'profile.json');profile.update(max_num_seqs=workers,port=19030+workers+(1 if independent else 0),
        served_model_name=f'layerwise-pressure-c{workers}-'+('independent' if independent else 'shared'))
    original=p.read(SOURCE/'profile.json')
    diff={k:dict(before=original.get(k),after=profile.get(k)) for k in profile if profile.get(k)!=original.get(k)}
    if set(diff)-{'port','served_model_name','max_num_seqs'}:raise ValueError('Unapproved serving parameter change')
    (root/'inputs').mkdir()
    for row in fixture['rows']:
        source=SOURCE/'inputs'/(row['id']+'.i32')
        if p.sha(source)!=row['served_sha256']:raise ValueError('Original input changed')
        shutil.copy2(source,root/'inputs'/source.name)
    shutil.copytree(SOURCE/'upstream',root/'upstream')
    p.save(root/'fixture.json',fixture);p.save(root/'profile.json',profile)
    p.save(root/'control-parity.json',dict(profile_differences=diff,original_inputs_and_query_coordinates=True,
        original_sampling_thinking_prefix_output_caps=True,original_per_request_kv_budgets=True,
        original_prefill_chunk_and_batch_limit=True,source=str(SOURCE)))
    p.save(root/'runtime-proof.json',{str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
        if f.is_file() and '__pycache__' not in f.parts})
    p.save(root/'prepared.json',dict(at=time.time(),ids=fixture['qualification_ids'],expected_runs=cases,
        profile_sha256=p.sha(root/'profile.json'),fixture_sha256=p.sha(root/'fixture.json'),
        runtime_proof_sha256=p.sha(root/'runtime-proof.json')))
    p.save(root/'status.json',dict(status='prepared',expected=cases,workers=workers,at=time.time()))
    serial.verify(root)
    print(dict(root=str(root),workers=workers,cases=cases,independent=independent),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--workers',type=int,required=True)
    parser.add_argument('--cases',type=int,default=26);parser.add_argument('--independent',action='store_true')
    a=parser.parse_args();prepare(a.root.resolve(),a.workers,a.independent,a.cases)

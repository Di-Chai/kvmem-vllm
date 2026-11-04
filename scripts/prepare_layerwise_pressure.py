"""Freeze a 13x2 paired prefill-pressure ablation against original RULER controls."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time

import recall_comparison_protocol as p
import ruler_protocol
import run_layerwise_query_diagnostic as driver

REPO=Path(__file__).resolve().parents[1]
REFERENCE=REPO/'artifacts/layerwise-meank-ruler260-20261102-r2'


def prepare_arm(root, independent):
    if root.exists():raise ValueError('Preserve prior experiments; require a fresh arm directory')
    root.mkdir(parents=True)
    runtime=root/'runtime-v1'
    shutil.copytree(REFERENCE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['action_attention.py','layerwise_query.py','layerwise_pressure.py','window_assembly.py']:
        shutil.copy2(REPO/'src/vllm_kvmem'/name,runtime/'src/vllm_kvmem'/name)
    for name in ['run_layerwise_query_diagnostic.py','prepare_layerwise_pressure.py','analyze_layerwise_pressure.py']:
        shutil.copy2(REPO/'scripts'/name,runtime/'scripts'/name)
    serve=runtime/'src/vllm_kvmem/serve.py';text=serve.read_text()
    anchor='    args = parser.parse_args(argv)\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,"    parser.add_argument('--layerwise-prefill-pressure', action='store_true')\n"+anchor)
    anchor='    plan = build_launch(profile, base, run_dir)\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+"    if args.layerwise_prefill_pressure:\n"
        "        if args.layerwise_query_diagnostic != 'sequential' or not args.layerwise_query_optimized:\n"
        "            raise ValueError('Layerwise pressure requires optimized sequential Query')\n"
        "        plan['environment']['KVMEM_LAYERWISE_PREFILL_PRESSURE'] = '1'\n")
    serve.write_text(text)
    fixture=deepcopy(p.read(REFERENCE/'fixture.json'))
    fixture['rows']=[r for r in fixture['rows'] if r['index']<2]
    ruler_protocol.validate_cohort(fixture['rows'],2)
    fixture.update(cohort='balanced26',cases=26,samples_per_task=2,pilot_samples_per_task=2,
        arms=['sequential'],qualification_ids=[r['id'] for r in fixture['rows']],
        source_fixture=str(REFERENCE/'fixture.json'),source_fixture_sha256=p.sha(REFERENCE/'fixture.json'),
        layerwise_prefill_pressure=independent,layerwise_optimized=True,layerwise_full_diagnostics=False,
        diagnostic_scope='Original 13x2; final Query causal layerwise Mean-K; only prefill pressure aggregation/selection differs.',
        diagnostic_limits=['Serial eager in both arms; original NVFP4, sampling, prefix, caps and K/G unchanged.',
            'Independent pressure samples all sixteen complete-attention layers at unchanged Query rows, strides and intervals.',
            'Shared pressure retains its original spread3 cross-layer aggregate.',
            'Pressure candidates are each layers own accepted resident history; cold history is not reopened.',
            'Sixteen physical layer windows each obey the same Active budget; logical union may be larger.',
            'One public turn and original short outputs; no decode pressure should be needed. No silent fallback or resampling.'])
    fixture['fixed_controls']['prefill_pressure']='accumulated_attention_per_layer' if independent else 'accumulated_attention'
    profile=p.read(REFERENCE/'profile.json')
    profile.update(port=19028 if independent else 19027,served_model_name='layerwise-pressure-'+('independent' if independent else 'shared'))
    if {k:v for k,v in profile.items() if k not in ['port','served_model_name']} != {k:v for k,v in p.read(REFERENCE/'profile.json').items() if k not in ['port','served_model_name']}:
        raise ValueError('Original serving controls changed')
    (root/'inputs').mkdir()
    for row in fixture['rows']:
        source=REFERENCE/'inputs'/(row['id']+'.i32')
        if p.sha(source)!=row['served_sha256']:raise ValueError('Original input changed')
        shutil.copy2(source,root/'inputs'/source.name)
    shutil.copytree(REFERENCE/'upstream',root/'upstream')
    p.save(root/'profile.json',profile);p.save(root/'fixture.json',fixture)
    p.save(root/'runtime-proof.json',{str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
        if f.is_file() and '__pycache__' not in f.parts})
    p.save(root/'prepared.json',dict(at=time.time(),ids=fixture['qualification_ids'],expected_runs=26,
        profile_sha256=p.sha(root/'profile.json'),fixture_sha256=p.sha(root/'fixture.json'),
        runtime_proof_sha256=p.sha(root/'runtime-proof.json')))
    p.save(root/'status.json',dict(status='prepared',expected=26,at=time.time()))
    driver.verify(root)


def prepare(root):
    if root.exists():raise ValueError('A fresh paired experiment is required')
    root.mkdir(parents=True)
    stages=[]
    for name,independent in [('shared',False),('independent',True)]:
        arm=root/name;prepare_arm(arm,independent)
        stages.append(dict(id=name,root=str(arm),kind='ruler',expected=26,proof_file='prepared.json',
            proof_sha256=p.sha(arm/'prepared.json'),command=[sys.executable,'-B',str(arm/'runtime-v1/scripts/run_layerwise_query_diagnostic.py'),
                'run','--root',str(arm),'--arms','sequential']))
    (root/'runtime-v1').symlink_to(root/'shared/runtime-v1',target_is_directory=True)
    shutil.copy2(root/'shared/runtime-proof.json',root/'runtime-proof.json')
    p.save(root/'queue-plan.json',dict(stages=stages,scheduled_inspections_enabled=False,
        runtime_proof_sha256=p.sha(root/'runtime-proof.json'),
        scope='52 fresh independent runs: shared attention pressure versus per-layer attention pressure; final Query per-layer Mean-K in both.',
        controls=dict(active_kv=36864,gen_kv=16384,block_tokens=32,sink_blocks=41,recent_blocks=0,thinking=False),
        no_answer_resampling=True,reference=str(REFERENCE)))
    p.save(root/'status.json',dict(status='prepared',stages=2,at=time.time()))
    (root/'inspection').mkdir();(root/'inspection/STOP').write_text('Scheduled inspections remain cancelled by user.\n')
    shutil.copy2(REPO/'scripts/run_layerwise_suite_queue.py',root/'run_queue.py')
    shutil.copy2(REPO/'scripts/recall_comparison_protocol.py',root/'recall_comparison_protocol.py')
    p.save(root/'queue-source-proof.json',{n:p.sha(root/n) for n in ['run_queue.py','recall_comparison_protocol.py']})
    print(root,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    prepare(parser.parse_args().root.resolve())

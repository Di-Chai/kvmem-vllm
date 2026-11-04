"""Freeze an independent Pro6000 Value-recall campaign; never edit old runs."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import recall_comparison_protocol as p


def prepare(args):
    repo=Path(__file__).resolve().parents[1];root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    runtime=root/'runtime-v1';campaign=root/'campaign-v1'
    if campaign.exists():raise ValueError('Prepare a fresh campaign')
    micro=p.read(root/'gpu-scoring-efficiency-v2.json')
    p.require(micro['status']=='passed' and 'RTX PRO 6000' in micro['gpu'] and
        micro['speedup']>1 and micro['value_norm_speedup']>1 and all(micro['selected_sets_equal'].values()),
        'Local GPU correctness/efficiency gate has not passed')
    if runtime.exists():
        p.require(args.reuse_built_runtime,'Existing runtime; explicitly reuse its verified build')
        proof=p.read(root/'native-build-proof.json')
        p.verify_sources(runtime,proof['sources']);p.verify_binaries(runtime,proof['binaries'])
    else:
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so','.git','build')
        for relative in ('src','csrc','scripts','third_party/qw3'):
            shutil.copytree(repo/relative,runtime/relative,ignore=ignore)
        for name in ('setup.py','pyproject.toml','CMakeLists.txt','MANIFEST.in'):
            shutil.copy2(repo/name,runtime/name)
        for name in ('LICENSE','third_party/vllm/LICENSE'):
            if (repo/name).exists():
                (runtime/name).parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(repo/name,runtime/name)
    env=dict(os.environ,PYTHONPATH=str(runtime/'src'),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_HOME='/usr/local/cuda')
    if not args.reuse_built_runtime:
        with (root/'native-build.log').open('x') as log:
            subprocess.run([str(args.python),'-B','setup.py','build_ext','--inplace'],
                cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        p.save(root/'native-build-proof.json',dict(sources=p.source_manifest(runtime),binaries=p.binary_manifest(runtime)))
    if args.build_only:
        print(json.dumps(dict(status='built',runtime=str(runtime))),flush=True)
        return
    base=p.read(repo/'artifacts/recall-comparison-20261028-r1/campaign-v9/profile.json')
    base.update(model=str(args.model.resolve()),hf_config_path=str(args.model.resolve()),
        tokenizer=str(args.tokenizer.resolve()),host='127.0.0.1',port=18992,
        gpu_memory_utilization=.72,execution_mode='eager')
    base['recall_experiment'].update(method='exact_mass',include_mandatory_in_softmax=True,
        value_backend='streaming',query_batch=128,block_batch=128)
    if args.pressure_attention_hybrid:
        base['recall_experiment'].update(method='block_value_norm',
            prefill_pressure_method='accumulated_attention', decode_pressure_method='accumulated_attention')
    if args.optimize_value_query:
        p.require(args.pressure_attention_hybrid,'Packed Query campaign keeps cumulative pressure policy')
        base['recall_experiment'].update(value_backend='streaming_packed',block_batch=1024)
    p.require(base['kvmem']['select_budget']==36864 and base['kvmem']['gen_budget']==16384,
              'Original Active/Gen budgets changed')
    config=root/'local-base-profile.json';p.save(config,base)
    source=args.source.resolve()
    command=[str(args.python),'-B',str(runtime/'scripts/run_recall_comparison.py'),'prepare',
        '--root',str(campaign),'--source',str(source),'--config',str(config),'--runtime',str(runtime),
        '--inputs',str(source.parent/'dataset/inputs'),'--upstream',str(source.parent/'upstream'),
        '--tokenizer-source',str(args.tokenizer.resolve()),'--value-reference']
    if args.pressure_attention_hybrid:
        command.append('--pressure-attention-hybrid')
    if args.optimize_value_query:
        command.append('--optimize-value-query')
    with (root/'prepare.log').open('x') as log:
        subprocess.run(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([str(args.python),'-B',str(runtime/'scripts/run_recall_comparison.py'),
        'bind-binaries','--root',str(campaign),'--runtime',str(runtime)],cwd=runtime,env=env,check=True)
    methods=p.VALUE_METHODS[1:] if args.pressure_attention_hybrid else p.VALUE_METHODS
    p.save(root/'preparation.json',dict(status='prepared',runtime=str(runtime),campaign=str(campaign),
        methods=list(methods),cases_per_method=260,formal_runs=260*len(methods),
        pressure_attention_hybrid=args.pressure_attention_hybrid,
        optimize_value_query=args.optimize_value_query,
        model=str(args.model.resolve()),model_config_sha256=p.sha(args.model/'config.json'),
        model_precision='Existing local NVFP4 checkpoint; every requested arm reruns independently',
        microbenchmark_sha256=p.sha(root/'gpu-scoring-efficiency-v2.json'),
        scope='Pro6000 independent campaign, never merged into the prior 4090 five-method scores'))
    print(json.dumps(dict(status='prepared',campaign=str(campaign))),flush=True)


if __name__=='__main__':
    repo=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=repo/'artifacts/value-recall-20261101-r1')
    parser.add_argument('--source',type=Path,default=repo/'artifacts/recall-comparison-20261028-r1/campaign-v9/fixture.json')
    parser.add_argument('--model',type=Path,default=repo/'artifacts/nvfp4-20261013/model')
    parser.add_argument('--tokenizer',type=Path,default=Path('/home/chaidi/qw3/models/Qwen3.8-27B-BF16'))
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--reuse-built-runtime',action='store_true')
    parser.add_argument('--pressure-attention-hybrid',action='store_true')
    parser.add_argument('--optimize-value-query',action='store_true')
    prepare(parser.parse_args())

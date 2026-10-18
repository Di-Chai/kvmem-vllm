"""Freeze and run a controlled 60-case, three-arm 256K NIAH comparison."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import benchmark_block_graph as bench
import benchmark_niah as niah
import qualify_graph_niah as client

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('multi','baseline','legacy')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Missing host available-memory observation')


def profiles(base,port):
    common = deepcopy(base)
    common.update(port=port, served_model_name='kvmem-graph-niah', verify_session_restore=False,
                  max_model_len=524288,max_num_seqs=8,max_num_batched_tokens=4224,
                  long_prefill_token_threshold=1024,gpu_memory_utilization=.72,
                  prefill_chunk_tokens=1024,profiler_config=None)
    common['kvmem'].update(select_budget=36864,gen_budget=16384,repository_blocks=16384,
        cpu_read_cache_bytes=100000000000,cpu_history_cache_bytes_per_request=20000000000,
        session_capacity=8,session_checkpoint_bytes=1,session_capture_terminal=True,
        session_capture_decode=False,session_checkpoint_policy='pm',
        query_history=False,action_pressure=False,guided_query_tokens=0,
        action_attention=True,prefill_attention=True,action_scope='non-thinking',
        action_interval=16,action_layers='spread3',prefill_attention_interval=1024,
        prefill_attention_stride=32,prefill_attention_half_life=8192,
        block_tokens=32,sink_blocks=41,recent_blocks=0,gpu_raw_k_cache_bytes=0,
        repository_gc=False,graph_trace=False,graph_locality_weight=.1,
        graph_attention_weight=.2,graph_attention_topk=4,graph_attention_min_mass=0.,
        graph_cross_message_weight=.25,graph_max_bytes_per_session=33554432,
        graph_max_total_bytes=268435456,rolling_active_window=False,
        offline_agent_policy=False,offline_phase_policy=False,offline_prefill_coalescing=False)
    result = {}
    for arm in ARMS:
        profile = deepcopy(common)
        profile['kvmem'].update(graph_mode='multi' if arm == 'multi' else 'off',
                                action_fraction=.5 if arm == 'legacy' else 0.)
        client.validate_profile(profile,arm)
        result[arm] = profile
    validate_profiles(result)
    return result


def validate_profiles(configs):
    stripped = []
    for arm in ARMS:
        cfg = deepcopy(configs[arm])
        client.validate_profile(cfg,arm)
        cfg['kvmem'].pop('graph_mode')
        cfg['kvmem'].pop('action_fraction')
        stripped.append(cfg)
    if not all(cfg == stripped[0] for cfg in stripped):
        raise ValueError('Non-scoring profile settings differ between arms')


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    common = json.loads(args.profile.read_text())
    configs = profiles(common,args.port)
    for arm,cfg in configs.items():
        bench.write(root/(arm+'.json'),cfg)
    source = args.reference.resolve()
    dataset = root/'dataset'
    cfg = niah.normalized_config(configs['baseline'])
    row_args = SimpleNamespace(reference=source,config=root/'baseline.json',output=dataset,
        lengths=[262144],depths=[10,30,50,70,90],seeds=[1000,2000,3000],max_output=8192,
        workers=8,cpu_headroom_gib=16,resume=False)
    rows = niah.prepare(row_args,cfg)
    assert len(rows) == 60
    original = (source/'run.py').read_text()
    node = next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name == 'score')
    fixture = dict(rows=rows,input_directory=str(dataset/'inputs'),workers=8,
        tokenizer_sha256=sha(Path(common['tokenizer'])/'tokenizer.json'),
        source_cohort_sha256=sha(source/'cohort.json'),source_scorer_sha256=sha(source/'run.py'),
        scorer_source=ast.get_source_segment(original,node),
        cpu_payload_credits_bytes=100000000000-16*1024**3,request_timeout_seconds=3600,
        admission_protocol='Continuous FIFO; at most eight owned jobs; per-owner ordinary retirement before refill.',
        sampling=dict(max_tokens=8192,temperature=1.,top_p=.95,top_k=20,min_p=0,
                      presence_penalty=0,repetition_penalty=1,thinking_token_budget=4096),
        gold_sent_as_separate_model_metadata=False,
        limitations=['All three arms share chronological pressure selection and attention capture.',
                     'This isolates scoring; it does not reproduce the historical attention-pressure deployment.',
                     'No message-boundary hints are invented for the original frozen token prompts.'])
    bench.write(root/'fixture.json',fixture)
    tracked_source_changes = subprocess.check_output(['git','diff','HEAD','--name-only','--','src','csrc','third_party'],cwd=ROOT,text=True)
    if tracked_source_changes.strip():
        raise ValueError('Production source changed after the requested baseline commit')
    git_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for arm in ARMS:
        bench.prepare(SimpleNamespace(output=root/arm,fixture=root/'fixture.json',arm=arm,
            profile=root/(arm+'.json'),qualifier=ROOT/'scripts/qualify_graph_niah.py',
            synchronous=False,port=args.port,client_timeout_seconds=7200))
    runtimes = [json.loads((root/arm/'manifest.json').read_text())['runtime_sha256'] for arm in ARMS]
    assert runtimes[0] == runtimes[1] == runtimes[2], 'Frozen runtime differs across arms'
    frozen = root/'frozen'
    frozen.mkdir()
    for name in ('benchmark_graph_niah.py','benchmark_block_graph.py','benchmark_niah.py',
                 'qualify_graph_niah.py','qualify_block_graph.py'):
        shutil.copyfile(ROOT/'scripts'/name,frozen/name)
    maximum_credit = max(((r['served_input_tokens']+8192+16384+31)//32)*32*17*4*256*2 for r in rows)
    maximum_users = min(8,fixture['cpu_payload_credits_bytes']//maximum_credit)
    if maximum_users != 8:
        raise ValueError('Full CPU history credits do not accommodate the intended eight users')
    manifest = dict(source_git_commit=git_commit,arms=list(ARMS),cases_per_arm=60,workers=8,
        frozen_sha256={str(p.relative_to(root)):sha(p) for p in frozen.iterdir()},
        fixture_sha256=sha(root/'fixture.json'),profile_sha256={a:sha(root/a/'profile.json') for a in ARMS},
        runtime_sha256=runtimes[0],minimum_cpu_available_bytes=8*1024**3,
        minimum_gpu_free_mib=4096,start_cpu_available_bytes=8*maximum_credit+24*1024**3,
        score_only_changed_fields=['kvmem.graph_mode','kvmem.action_fraction'],
        model_identity={str(p):sha(p) for p in Path(common['model']).glob('*.json')},
        created_at=time.time())
    bench.write(root/'manifest.json',manifest)
    bench.write(root/'status.json',dict(status='prepared',at=time.time()))
    print(json.dumps(dict(prepared=str(root),baseline_commit=git_commit,cases=60,arms=list(ARMS),
                         start_memory_required_gib=manifest['start_cpu_available_bytes']/1024**3)),flush=True)


def validate(root):
    m = json.loads((root/'manifest.json').read_text())
    if sha(root/'fixture.json') != m['fixture_sha256']:
        raise ValueError('Shared original-input fixture changed')
    for name,digest in m['frozen_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('Frozen controller changed: '+name)
    configs = {a:json.loads((root/a/'profile.json').read_text()) for a in ARMS}
    validate_profiles(configs)
    for arm in ARMS:
        if sha(root/arm/'profile.json') != m['profile_sha256'][arm]:
            raise ValueError('Frozen profile changed')
        if json.loads((root/arm/'manifest.json').read_text())['runtime_sha256'] != m['runtime_sha256']:
            raise ValueError('Arm runtime identities differ')
    for name,digest in m['model_identity'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Model configuration changed')
    return m


def compare(root):
    manifest = validate(root)
    reports = {a:json.loads((root/a/'qualification/results.json').read_text()) for a in ARMS}
    for arm,r in reports.items():
        assert r['status'] == 'passed' and r['cases'] == 60 and r['all_owned_sessions_released']
        assert r['fixture_sha256'] == manifest['fixture_sha256']
        cleanup = json.loads((root/arm/'cleanup.json').read_text())
        assert not cleanup['remaining_owned_process_groups'] and not cleanup['remaining_gpu_pids']
    indexed = {a:{r['id']:r for r in report['results']} for a,report in reports.items()}
    assert set(indexed['baseline']) == set(indexed['legacy']) == set(indexed['multi'])
    paired = []
    for key in sorted(indexed['baseline']):
        rows = [indexed[a][key] for a in ARMS]
        assert len({r['input_sha256'] for r in rows}) == 1
        assert len({(r['seed'],r['position_percent']) for r in rows}) == 1
        paired.append(dict(id=key,task=rows[0]['task'],coverage={a:indexed[a][key]['evaluation']['normalized_coverage'] for a in ARMS},
                           exact={a:indexed[a][key]['evaluation']['normalized_exact_numeric_set'] for a in ARMS}))
    comparisons = {}
    for reference in ('baseline','legacy'):
        wins = sum(r['exact']['multi'] and not r['exact'][reference] for r in paired)
        losses = sum(r['exact'][reference] and not r['exact']['multi'] for r in paired)
        n = wins+losses
        p = min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
        comparisons[reference] = dict(graph_only_correct=wins,reference_only_correct=losses,
            coverage_difference=statistics.fmean(r['coverage']['multi']-r['coverage'][reference] for r in paired),
            exact_difference=wins-losses,mcnemar_exact_two_sided_p=p,
            interpretation='One fixed-seed campaign; paired descriptive evidence, not repeated-run stability.')
    result = dict(status='completed',source_git_commit=manifest['source_git_commit'],cases=60,
        arms={a:{k:r[k] for k in ('macro_coverage','exact','wall_seconds','cpu_admission_peak','graph_changed_queries','output_tokens_per_wall_s','step_summary')} for a,r in reports.items()},
        graph_vs=comparisons,paired=paired,
        scope='Same model/runtime/full original inputs/K/G/sampling and pressure policy. All arms capture attention; only scoring differs.')
    bench.write(root/'comparison.json',result)
    return result


def run(args):
    root = args.output.resolve()
    manifest = validate(root)
    if json.loads((root/'status.json').read_text())['status'] != 'prepared':
        raise ValueError('Run a fresh prepared attempt; failed evidence is retained')
    if Path(__file__).resolve() != root/'frozen/benchmark_graph_niah.py':
        raise ValueError('Execute the frozen campaign controller')
    environment = dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    bench.write(root/'status.json',dict(status='running',at=time.time()))
    try:
        with bench.cleanup_on_termination() as begin_cleanup:
            for arm in ARMS:
                if cpu_available() < manifest['start_cpu_available_bytes']:
                    raise RuntimeError('Insufficient CPU headroom for full eight-user historical credits')
                command = [sys.executable,'-B',str(root/arm/'runtime/scripts/benchmark_block_graph.py'),'run',
                           '--output',str(root/arm),'--arm',arm,'--fixture',str(root/'fixture.json')]
                process = None
                try:
                    with (root/(arm+'-controller.log')).open('x') as log, (root/(arm+'-resources.jsonl')).open('x') as samples:
                        process = subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        while process.poll() is None:
                            available = cpu_available()
                            sample = subprocess.check_output(['nvidia-smi',
                                '--query-gpu=memory.free,utilization.gpu,power.draw',
                                '--format=csv,noheader,nounits'],text=True).strip().split(',')
                            free,util,power = (float(v.strip()) for v in sample)
                            samples.write(json.dumps(dict(at=time.monotonic(),cpu_available_bytes=available,
                                gpu_free_mib=free,gpu_utilization=util,gpu_power_w=power))+'\n')
                            samples.flush()
                            if available < manifest['minimum_cpu_available_bytes'] or free < manifest['minimum_gpu_free_mib']:
                                raise RuntimeError('Runtime memory guard reached; no history will be dropped')
                            time.sleep(1)
                    if process.returncode:
                        raise RuntimeError('Arm failed: '+arm+'; retain controller/client/server logs')
                finally:
                    if process is not None:
                        remaining = bench.stop_owned_groups((process,))
                        if remaining:
                            raise RuntimeError('Campaign controller process group did not stop')
                bench.write(root/'progress.json',dict(completed_arms=[a for a in ARMS[:ARMS.index(arm)+1]],at=time.time()))
            begin_cleanup()
        result = compare(root)
        bench.write(root/'status.json',dict(status='completed',at=time.time()))
        print(json.dumps({k:v for k,v in result.items() if k != 'paired'}),flush=True)
    except BaseException as error:
        bench.write(root/'status.json',dict(status='failed',error=str(error),at=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','compare'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',type=Path,default=ROOT/'artifacts/block-graph-model-20261018-multi-r3/profile.json')
    parser.add_argument('--reference',type=Path,default=Path('/data/chaidi/qw3-experiments/cacheblend_20260925/official_template_20260927'))
    parser.add_argument('--port',type=int,default=18670)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'compare':
        print(json.dumps(compare(args.output.resolve())))
    else:
        run(args)


if __name__ == '__main__':
    main()

"""Qualification/efficiency-gated local Pro6000 campaign with retained failures."""
import argparse
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import recall_comparison_protocol as p


def diagnostic_gate(campaign):
    from run_recall_comparison import functional_qualified,read_events,verify
    fixture,profile=verify(campaign)
    if fixture.get('pressure_attention_hybrid'):
        if fixture.get('value_query_optimization'):
            paths=list((campaign/'arms/block_value_norm/qualification/server-attempts').glob(
                '*/server/capture/value_query_optimization.jsonl'))
            p.require(paths,'Missing real task Query optimization validation')
            records=[]
            for path in paths:records.extend(read_events(path))
            p.require(records and all(r['passed'] and r['normalization_mask_all'] for r in records),
                      'Real Query IO optimization failed')
            for record in records:
                for method in p.VALUE_METHODS[1:]:
                    row=record['methods'][method]
                    p.require(row['selected_set_equal'] and row['scores_close'] and
                              row['query_and_value_scales_equal'] and row['speedup']>1,
                              'Query optimization changed operands/selection or did not improve time')
            from run_recall_comparison import qualification_gate
            qualification_gate(campaign,fixture,profile)
            result=dict(passed=True,pressure_attention_hybrid=True,value_query_optimization=True,
                real_task_states=len(records),profiles={str(f):p.sha(f) for f in paths},
                speedup={m:statistics.median(r['methods'][m]['speedup'] for r in records) for m in p.VALUE_METHODS[1:]},
                scope='Same real Q/K/V, cold operand caches including archive IO, native selected sets equal; '
                      'both complete hybrid runtime entrypoint tests independently passed.')
            p.save(campaign/'qualification/value-reference-efficiency-gate.json',result)
            p.save(campaign/'qualification/query-optimization-gate.json',result)
            return result
        proof=p.read(campaign.parent/'value-scoring-reuse-proof.json')
        source=Path(proof['source_campaign'])
        gate=source/'qualification/value-reference-efficiency-gate.json'
        p.require(p.sha(gate)==proof['source_gate_sha256'] and p.read(gate)['passed'],
                  'Original Value query-scoring validation changed')
        for relative,digest in proof['query_scoring_sources'].items():
            p.require(p.sha(campaign.parent/'runtime-v1'/relative)==digest and
                      p.sha(Path(proof['source_runtime'])/relative)==digest,
                      'Value Query scorer differs from previously validated implementation')
        for method in p.campaign_methods(fixture):
            arm=p.read(campaign/'profiles'/(method+'.json'))
            p.require(functional_qualified(campaign,method,arm),'Missing hybrid three-entrypoint validation')
        from run_recall_comparison import qualification_gate
        qualification_gate(campaign,fixture,profile)
        result=dict(passed=True,pressure_attention_hybrid=True,
            query_scoring_validation_reused=True,source_campaign=str(source),
            scope='Unchanged Value Query scorers reuse original 15-state mathematical/efficiency validation; '
                  'new hybrid task cases and all three runtime entrypoints are independently qualified.')
        p.save(campaign/'qualification/value-reference-efficiency-gate.json',result)
        return result
    all_states=[]
    for method in p.VALUE_METHODS:
        arm=p.read(campaign/'profiles'/(method+'.json'))
        p.require(functional_qualified(campaign,method,arm),'Missing three-entrypoint qualification')
        states=[]
        for path in (campaign/'arms'/method/'qualification/server-attempts').glob('*/server/capture/value_recall_diagnostics.jsonl'):
            states.extend(read_events(path))
        p.require(len(states)>=len(fixture['qualification_ids']),'Missing small-sample frozen-state diagnostics')
        for state in states:
            p.require(all(state['normalization_mask']),'Frozen-state normalization excluded fixed keeps')
            for value_method in p.VALUE_METHODS[1:]:
                report=state['methods'][value_method]
                p.require(report.get('optimized_selected_set_equal') is True,
                          'Optimized selection differs from reference on an original-input state')
                error=report['set_output_error']['mean_query_layer_head_l2']
                p.require(math.isfinite(error) and error>=0,'Invalid whole-set attention output error')
            p.require(state['query_reduction']=='sum' and state['head_layer_reduction']=='mean',
                      'Reference aggregation differs from ExactMass')
        all_states.extend(states)
    deletion_ratios=[s['reference_scoring_seconds']/s['optimized_scoring_seconds'] for s in all_states]
    norm_ratios=[s['reference_scoring_seconds']/s['optimized_value_norm_seconds'] for s in all_states]
    p.require(statistics.median(deletion_ratios)>1 and statistics.median(norm_ratios)>1,
              'Real-state scoring optimization did not improve median elapsed time')
    result=dict(passed=True,states=len(all_states),
        median_frozen_state_scoring_speedup=dict(block_value_norm=statistics.median(norm_ratios),
            block_deletion_error=statistics.median(deletion_ratios)),
        scope='Actual immutable task Q/K/V states; identical native selected sets, independently recomputed set errors. Not whole-task speedup.')
    p.save(campaign/'qualification/value-reference-efficiency-gate.json',result)
    return result


def wait_for_idle(root):
    stable=0
    while stable<6:
        row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        used,util=map(int,row.split(','));idle=used<1000 and util<=5
        stable=stable+1 if idle else 0
        p.save(root/'queue-status.json',dict(status='waiting-for-idle-gpu',memory_used_mib=used,
            utilization=util,idle_checks=stable,at=time.time()))
        if stable<6:time.sleep(10)


def run(args):
    root=args.root.resolve();runtime=root/'runtime-v1';campaign=root/'campaign-v1'
    if (root/'queue-process.json').exists():raise ValueError('Existing queue evidence; do not start a duplicate controller')
    micro=p.read(root/'gpu-scoring-efficiency-v2.json')
    p.require(micro['status']=='passed' and micro['speedup']>1 and micro['value_norm_speedup']>1,
        'First-version microbenchmark efficiency gate failed')
    p.save(root/'queue-process.json',dict(pid=os.getpid(),runtime=str(runtime),campaign=str(campaign),
        python=str(args.python),at=time.time()))
    env={k:v for k,v in os.environ.items() if not k.startswith('KVMEM_')}
    env.update(PYTHONPATH=str(runtime/'src'),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_HOME='/usr/local/cuda',
        PATH='/usr/local/cuda/bin:'+str(args.python.parent)+':'+os.environ.get('PATH',''))
    def stage(name,arguments):
        p.save(root/'queue-status.json',dict(status='running',stage=name,at=time.time()))
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([str(args.python),'-B',*map(str,arguments)],cwd=runtime,env=env,
                stdout=log,stderr=subprocess.STDOUT)
            p.save(root/'queue-child.json',dict(stage=name,pid=proc.pid,at=time.time()))
            code=proc.wait()
        if code:raise RuntimeError(name+' failed; all answers and failure evidence remain retained')
    wait_for_idle(root)
    common=['--root',campaign,'--runtime',runtime,'--python',args.python]
    script=runtime/'scripts/run_recall_comparison.py'
    stage('qualification',[script,'run',*common,'--phase','qualification','--qualify-entrypoints'])
    diagnostic_gate(campaign)
    stage('formal',[script,'run',*common,'--phase','formal'])
    stage('analysis',[runtime/'scripts/analyze_value_recall_comparison.py','--root',campaign])
    p.save(root/'queue-status.json',dict(status='completed',stage='analysis',at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    args=parser.parse_args()
    try:run(args)
    except BaseException as error:
        p.save(args.root/'queue-status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise

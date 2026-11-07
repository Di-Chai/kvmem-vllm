"""Within-candidate page-count sweep; fixed query only, no semantic claim."""
import argparse
import json
from pathlib import Path
import time

from gpu_study import save


def run(args):
    from vllm import LLM, SamplingParams
    args.output.mkdir(parents=True,exist_ok=False)
    cases=json.loads((args.inference/'prepared-cases.json').read_text())
    # Predeclared largest candidates, one overlap and two code observations.
    cases=[c for c in cases if c['candidate']['candidate_id'] in ('c08','c15','c16')]
    save(args.output/'design.json',dict(candidates=[c['candidate']['candidate_id'] for c in cases],
        counts=[1,2,4,8,'all'],order='Ascending original page position; nested sets.',
        scope='Fixed-query diagnostic only. No independent candidates, no generated-answer comparison.'))
    llm=LLM(model='/root/data/kvmem-vllm-gc/models/Qwen3.8-27B-GPTQ-Int4',
        tokenizer='/root/data/recall-comparison-20261028-r1/assets/tokenizer',
        dtype='bfloat16',quantization='gptq_marlin',max_model_len=34816,
        max_num_seqs=1,max_num_batched_tokens=512,gpu_memory_utilization=.88,
        kv_cache_dtype='fp8_e4m3',enable_prefix_caching=False,enforce_eager=True,
        seed=7231,language_model_only=True,attention_config={'backend':'FLASHINFER'},
        worker_extension_cls='gpu_worker.CleanupWorkerExtension')
    try:
        sampling=SamplingParams(temperature=0,seed=7231,max_tokens=1)
        summaries=[]
        for case in cases:
            cid=case['candidate']['candidate_id'];pages=case['candidate_mapping']['logical_blocks']
            for count in [0,1,2,4,8,len(pages)]:
                config=dict(case_id=cid,arm=f'pages-{count}',mode='masked',deleted_blocks=pages[:count],
                    query_start=case['query_start'],query_end=case['query_end'],reference=count==0,attention_scores=False)
                llm.collective_rpc('cleanup_setup',kwargs={'config':config})
                began=time.perf_counter()
                llm.generate([{'prompt_token_ids':case['input_ids']}],sampling,use_tqdm=False)
                elapsed=time.perf_counter()-began
                result=llm.collective_rpc('cleanup_finish')[0]
                save(args.output/cid/f'pages-{count}.json',result)
                row=dict(candidate_id=cid,removed_blocks=count,elapsed_seconds=elapsed)
                if count==0:
                    row['native_zero_delete_max']=result['relative_to_native_attention_max']
                    if row['native_zero_delete_max']!=0:
                        raise RuntimeError('Dose sweep zero-deletion backend differs from native')
                if count:
                    for kind in ('attention_pre_gate_per_head','transformer_residual_sum'):
                        values=[r['relative_l2'] for k,rs in result['measurements'].items() if k.startswith(kind+'/') for r in rs]
                        row[kind+'_mean']=sum(values)/len(values)
                if count==len(pages):
                    prior=json.loads((args.inference/cid/'reduced-kv.measurements.json').read_text())
                    if prior['historical_kv_sha256']!=result['historical_kv_sha256']:
                        raise RuntimeError('Dose sweep differs in original historical KV')
                    row['full_dose_vs_original_max_delta']=max(abs(a['relative_l2']-b['relative_l2'])
                        for k,rs in result['measurements'].items() for a,b in zip(rs,prior['measurements'][k]))
                summaries.append(row);save(args.output/'summary.json',summaries)
                print(json.dumps(row),flush=True)
        save(args.output/'status.json',dict(status='completed',cases=len(cases),arms=len(summaries)))
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:run(args)
    except BaseException as exc:
        if args.output.exists():save(args.output/'failure.json',dict(error=repr(exc)))
        raise

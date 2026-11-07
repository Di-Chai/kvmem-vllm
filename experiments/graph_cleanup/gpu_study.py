"""RTX 4090 offline paired model run; source traces are never executed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from protocol import PROBES, POLICY, token_mapping

ACTIVE_LLM = None


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(data): return hashlib.sha256(data).hexdigest()


def prepare(args):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    candidates=[json.loads(l) for l in (args.dataset/'candidates.jsonl').read_text().splitlines()]
    output=[]
    for c in candidates:
        s=json.loads((args.dataset/'snapshots'/f'{c["snapshot_id"]}.json').read_text())
        marker='\nEND QUOTED HISTORY\n\nFIXED QUESTIONS:\n'
        history=s['rendered_history']
        body='QUOTED HISTORY:\n'+history+marker+json.dumps(PROBES)
        body+='\nReturn one JSON object with exactly these six keys. Each value must be a concise string of at most 30 words, including supporting unit-ID suffixes (m#:u#) when available. Use unknown if unsupported.'
        rendered=tok.apply_chat_template([dict(role='system',content=POLICY),dict(role='user',content=body)],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoding=tok(rendered,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoding['input_ids']; offsets=encoding['offset_mapping']
        start=rendered.index(history)
        spans={uid:[a+start,b+start] for uid,(a,b) in s['body_character_spans'].items()}
        mapping=token_mapping(offsets,spans,32)
        chosen=mapping[c['unit_id']]
        qchar=rendered.index(marker)+len('\nEND QUOTED HISTORY\n\n')
        qstart=next(i for i,(a,b) in enumerate(offsets) if b>qchar)
        blocks=chosen['logical_blocks']
        assert blocks and (max(blocks)+1)*32<=qstart
        output.append(dict(candidate=c,prompt=rendered,input_ids=ids,
            prompt_sha256=sha(rendered.encode()),token_ids_sha256=sha(json.dumps(ids).encode()),
            query_start=qstart,query_end=len(ids),candidate_mapping=chosen,
            all_unit_mapping=mapping,snapshot_id=s['snapshot_id'],left_censored=s['left_censored']))
    return output


def run(args):
    global ACTIVE_LLM
    args.output.mkdir(parents=True,exist_ok=False)
    cases=prepare(args)
    save(args.output/'prepared-cases.json',cases)
    save(args.output/'input-summary.json',[dict(candidate_id=c['candidate']['candidate_id'],
        prompt_tokens=len(c['input_ids']),query_tokens=c['query_end']-c['query_start'],
        deleted_blocks=c['candidate_mapping']['logical_blocks'],
        collateral=c['candidate_mapping']['collateral_unit_ids']) for c in cases])
    if args.prepare_only:return
    import torch
    import vllm
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    save(args.output/'environment.json',dict(gpu=subprocess.check_output(['nvidia-smi'],text=True),
        python=platform.python_version(),vllm=vllm.__version__,torch=torch.__version__,
        model=args.model,tokenizer=args.tokenizer,
        model_config_sha256=sha((Path(args.model)/'config.json').read_bytes())))
    began=time.perf_counter()
    llm=LLM(model=args.model,tokenizer=args.tokenizer,dtype='bfloat16',quantization='gptq_marlin',
        max_model_len=34816,max_num_seqs=1,max_num_batched_tokens=512,
        gpu_memory_utilization=.88,kv_cache_dtype='fp8_e4m3',
        enable_prefix_caching=False,enforce_eager=True,seed=7231,
        language_model_only=True,attention_config={'backend':'FLASHINFER'},
        worker_extension_cls='gpu_worker.CleanupWorkerExtension')
    ACTIVE_LLM = llm
    save(args.output/'load.json',dict(seconds=time.perf_counter()-began))
    schema=dict(type='object',properties={k:dict(type='string') for k in PROBES},
                required=list(PROBES),additionalProperties=False)
    params=SamplingParams(temperature=0,seed=7231,max_tokens=args.max_tokens,
                         structured_outputs=StructuredOutputsParams(json=schema))
    brief=SamplingParams(temperature=0,seed=7231,max_tokens=1)
    status=dict(phase='running',completed=[])
    save(args.output/'status.json',status)

    def arm(case,name,mode,blocks,reference=False,sampling=None,attention_scores=False):
        cid=case['candidate']['candidate_id']
        cfg=dict(case_id=cid,arm=name,mode=mode,deleted_blocks=blocks,
                 query_start=case['query_start'],query_end=case['query_end'],
                 reference=reference,attention_scores=attention_scores)
        llm.collective_rpc('cleanup_setup',kwargs={'config':cfg})
        t=time.perf_counter()
        result=llm.generate([{'prompt_token_ids':case['input_ids']}],sampling or params,use_tqdm=False)[0]
        elapsed=time.perf_counter()-t
        measure=llm.collective_rpc('cleanup_finish')[0]
        completion=result.outputs[0]
        parsed=None;parse_error=None
        try:parsed=json.loads(completion.text)
        except ValueError as e:parse_error=str(e)
        response=dict(candidate_id=cid,arm=name,elapsed_seconds=elapsed,prompt_tokens=len(result.prompt_token_ids),
            output_tokens=len(completion.token_ids),output_token_ids=list(completion.token_ids),
            text=completion.text,parsed=parsed,parse_error=parse_error,finish_reason=completion.finish_reason,
            prompt_sha256=case['prompt_sha256'],token_ids_sha256=case['token_ids_sha256'],
            metrics=vars(result.metrics) if result.metrics is not None else None)
        save(args.output/cid/f'{name}.response.json',response)
        save(args.output/cid/f'{name}.measurements.json',measure)
        print(json.dumps(dict(candidate=cid,arm=name,seconds=elapsed,output_tokens=response['output_tokens'],
                              parsed=parsed is not None,finish_reason=completion.finish_reason)),flush=True)
        return response,measure

    # Control query at native positions, followed by the same no-deletion
    # diagnostic backend. Measure numerical drift before interpreting eviction.
    case=cases[0]
    arm(case,'control-native','native',[],reference=True,sampling=brief)
    _,control=arm(case,'control-custom-zero-delete','masked',[],sampling=brief)
    values=[r['relative_l2'] for rows in control['measurements'].values() for r in rows]
    save(args.output/'zero-deletion-control.json',dict(max_relative_l2=max(values),
        raw_attention_native_max=control['relative_to_native_attention_max'],
        historical_kv_equal=True,passed=max(values)<.03,
        threshold_note='Diagnostic compatibility gate only; not a deletion safety threshold.'))
    if max(values)>=.03:raise RuntimeError('No-deletion custom/native disagreement exceeds diagnostic compatibility gate')
    arm(case,'control-custom-reference','masked',[],reference=True,sampling=brief)
    _,repeat=arm(case,'control-custom-repeat','masked',[],sampling=brief)
    stable=[r['relative_l2'] for rows in repeat['measurements'].values() for r in rows]
    save(args.output/'identical-repeat-control.json',dict(max_relative_l2=max(stable),passed=max(stable)==0))
    if max(stable)!=0:raise RuntimeError('Identical zero-deletion diagnostic is not deterministic')
    for case in cases[:args.limit]:
        cid=case['candidate']['candidate_id']
        full,full_measure=arm(case,'full','masked',[],reference=True,attention_scores=True)
        if cid=='c01' and args.max_tokens>128:
            archive=llm.collective_rpc('cleanup_snapshot_prefix',kwargs={
                'length':16384,'path':str((args.output/'recovery/attention-kv.pt').resolve())})[0]
            save(args.output/'recovery/archive.json',archive)
        reduced,reduced_measure=arm(case,'reduced-kv','masked',case['candidate_mapping']['logical_blocks'])
        # Repeated identical measurements are a stability control, not new cases.
        if cid in ('c01','c09','c21'):
            arm(case,'reduced-kv-repeat','masked',case['candidate_mapping']['logical_blocks'],sampling=brief)
        status['completed'].append(cid);save(args.output/'status.json',status)
    if args.max_tokens>128:
        prefill=[]
        for length in (1024,4096,8192,16384):
            for repeat in range(3):
                llm.collective_rpc('cleanup_setup',kwargs={'config':dict(case_id='recovery',
                    arm='prefill',mode='native',deleted_blocks=[],query_start=length,query_end=length+1,reference=True)})
                t=time.perf_counter()
                result=llm.generate([{'prompt_token_ids':cases[0]['input_ids'][:length]}],brief,use_tqdm=False)[0]
                elapsed=time.perf_counter()-t
                llm.collective_rpc('cleanup_finish')
                evidence=llm.collective_rpc('cleanup_snapshot_prefix',kwargs={'length':length,
                    'reference_archive':str((args.output/'recovery/attention-kv.pt').resolve())})[0]
                prefill.append(dict(repeat=repeat,prefill_plus_first_token_seconds=elapsed,**evidence))
                save(args.output/'recovery/prefill.json',prefill)
                if not evidence['original_prefix_recomputed_byte_equal']:
                    raise RuntimeError('Full original prefix did not reconstruct the recorded attention KV')
    status['phase']='completed';save(args.output/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='/root/data/kvmem-vllm-gc/models/Qwen3.8-27B-GPTQ-Int4')
    p.add_argument('--tokenizer',default='/root/data/recall-comparison-20261028-r1/assets/tokenizer')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--limit',type=int,default=24)
    p.add_argument('--max-tokens',type=int,default=768)
    a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.output.exists():save(a.output/'failure.json',dict(type=type(e).__name__,error=str(e)))
        raise
    finally:
        if ACTIVE_LLM is not None:
            ACTIVE_LLM.llm_engine.engine_core.shutdown()

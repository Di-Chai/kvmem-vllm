"""Small DFlash2 accepted-statistics/session gate, not an agent speed benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request
import uuid

from qualify_block_graph import (expected_output_samples, native_matches, records,
                                 save, wait_released, wait_terminal_publication)


def run(args):
    profile=json.loads(args.config.read_text());kv=profile['kvmem']
    assert profile['draft_method']=='dflash' and profile['execution_mode']=='full_graph'
    assert not kv['request_planning'] and kv['graph_mode']=='collect'
    assert profile['verify_session_restore'] and kv['session_capture_terminal']
    args.output.mkdir(parents=True,exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(profile['tokenizer'],local_files_only=True)
    encode=lambda s:tokenizer.encode(s,add_special_tokens=False)
    prefix=encode('<|im_start|>system\nYou are a helpful assistant. Thinking is disabled.\n<|im_end|>\n<|im_start|>user\n')
    filler=encode('The storage service keeps immutable history and validates every accepted prefix.\n')
    tail=encode('\nExplain how to restore a checkpoint safely.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    initial=prefix+(filler*((8192-len(prefix)-len(tail)+len(filler)-1)//len(filler)))[:8192-len(prefix)-len(tail)]+tail
    sessions=['dflash-overhead-'+uuid.uuid4().hex for _ in range(2)]
    rows=[];prompts={};failures=[];began=time.monotonic()
    def complete(session,prompt,count=257,release=False):
        body=dict(model=profile['served_model_name'],session_id=session,cache_salt=session,
            prompt=[tokenizer.convert_tokens_to_ids('<|im_end|>')] if release else prompt,
            max_tokens=count,min_tokens=count,ignore_eos=True,temperature=0,
            seed=1927,return_token_ids=True,stream=False)
        if release:body['kv_transfer_params']={'kvmem_release_session':True}
        req=urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        start=time.monotonic()
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=600) as response:
                result=json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from None
        output=result['choices'][0]['token_ids']
        assert len(output)==count==result['usage']['completion_tokens']
        row=dict(session=session,response_id=result['id'],input_tokens=len(prompt),
            output_ids=output,seconds=time.monotonic()-start,release=release)
        if not release:row['expected_output_samples']=expected_output_samples(tokenizer,profile,prompt,output)
        return row
    def client(index):
        prompt=list(initial);session=sessions[index]
        for turn in range(3):
            row=complete(session,prompt);row.update(agent=index,turn=turn)
            rows.append(row);save(args.output/f'owner-{index}-turn-{turn}.json',row)
            prompt+=row['output_ids']+encode('<|im_end|>\n<|im_start|>user\nThe tool verified that the checkpoint bytes match. Explain the next validation step.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            prompts[session]=prompt
    report=None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(client,i) for i in range(2)]:future.result()
        archive=wait_terminal_publication(args.capture,rows)
        actions=records(args.capture/'action_attention.jsonl')
        for row in rows:
            own=[e for e in actions if native_matches(row,e)]
            assert len(own)==1 and own[0]['samples']==row['expected_output_samples']
            assert own[0]['prefill_samples']>0
        restores=records(args.capture/'verified_restores.jsonl')
        assert len(restores)>=4 and all(r['byte_equal'] and r['draft_layers']==5 and r['mtp_attention_layers']==0 for r in restores)
        assert archive['session_restored_tokens']>0 and archive['session_checkpoint_deferrals']==0
        assert sum(r['rejected_tokens'] for r in archive['completed'])>0
        assert all(r['terminal_prefix']['history_end']==row['input_tokens']+len(row['output_ids'])-1
            for row in rows for r in archive['completed'] if native_matches(row,r))
        config=json.loads((args.capture/'action_config.json').read_text())
        assert config['native_decode_commit'] and config['download_pool'] and config['fused_scoring']
        assert max(r['decode_commit_native_calls'] for r in actions)>0
        assert max(r['scoring_backend_counts'].get('fused',0) for r in actions)>0
        for r in actions:
            pool=r['download_pool_total'];assert pool['reuses'] and not pool['exhaustions'] and not pool['quarantined_slots']
        report=dict(status='passed',rows=rows,seconds=time.monotonic()-began,
            accepted_sample_oracle=True,draft_and_target_byte_restores=restores,
            action_config=config,graph_propagation_exercised=False,
            scope='Two-owner three-turn DFlash2 FULL-graph gate on legacy planned-off collect mode. '
                'Synthetic requests, no actual tools, no V2 Query/pressure propagation and no agent throughput claim.')
    except BaseException as error:
        failures.append(repr(error));raise
    finally:
        releases=[]
        for session in sessions:
            try:releases.append(complete(session,prompts.get(session,initial),count=1,release=True))
            except Exception as error:failures.append('release: '+repr(error))
        try:released=wait_released(args.capture,sessions,timeout=60)
        except Exception as error:released=None;failures.append('release check: '+repr(error))
        result=report or dict(status='failed',rows=rows)
        result.update(failures=failures,releases=releases,all_owned_sessions_released=released is not None)
        if failures or released is None:result['status']='failed'
        save(args.output/'results.json',result)
    if result['status']!='passed':raise RuntimeError('DFlash qualification failed; retain attempt')
    print(json.dumps(dict(status='passed',cases=6,byte_restores=len(restores))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','capture','output','fixture'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',required=True)
    run(p.parse_args())

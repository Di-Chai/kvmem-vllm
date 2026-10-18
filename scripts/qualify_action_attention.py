"""Native HTTP multi-session output-attention persistence and Query-selection gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text())
    fixture=json.loads(a.input.read_text());prompt=fixture['input_ids']
    q=fixture.get('query_span',[48033,49997])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg['tokenizer'],local_files_only=True)
    def expected_samples(tokens,prompt_tokens):
        from types import SimpleNamespace
        from vllm_kvmem.action_attention import ActionHistory
        ids=[tok.convert_tokens_to_ids(s) for s in ('<tool_call>','</tool_call>','<think>','</think>','<|im_start|>')]
        state=ActionHistory(SimpleNamespace(action_scope=cfg['kvmem']['action_scope'],
            action_interval=cfg['kvmem']['action_interval']),ids)
        state.begin('oracle',prompt_tokens)
        count=0
        for at in range(0,len(tokens)-1,32):
            chunk=tokens[:-1][at:at+32]
            state.parser.prepare(chunk)
            count+=state.parser.commit(len(chunk)).bit_count()
        return count
    def events(name):
        path=a.capture/name
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    def wait_event(name,response_id):
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            rows=[r for r in events(name) if r['request_id'].startswith(response_id+'-')]
            if rows:return rows[-1]
            time.sleep(.1)
        raise TimeoutError(f'Missing {name} for {response_id}')
    def request(session,tokens,query,count):
        payload={'model':cfg['served_model_name'],'session_id':session,'cache_salt':session,
            'prompt':tokens,'kvmem_query_span':query,'kvmem_reselect':'force',
            'max_tokens':count,'temperature':0,'ignore_eos':True,'return_token_ids':True}
        req=urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/completions",
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with http.open(req,timeout=600) as response:return json.load(response)
    def run(index):
        session='action-test-'+uuid.uuid4().hex
        first=request(session,prompt,q,257)
        (a.output/f'{index}-first.json').write_text(json.dumps(first))
        one=wait_event('action_attention.jsonl',first['id'])
        if cfg['kvmem'].get('prefill_attention'):
            first_query=wait_event('bounded_refresh.jsonl',first['id'])
            assert first_query['selection']=='query' and first_query['attention_source']=='prefill'
            assert first_query['prefill_attention_samples']>0 and first_query['action_history_pages']==0
            assert one['prefill_samples']==first_query['prefill_attention_samples'], 'Query/template replay entered prefill scores'
            pressure=[e for e in events('bounded_refresh.jsonl') if e['request_id'].startswith(first['id']+'-') and e['selection']=='pressure']
            if q[0]>cfg['kvmem']['select_budget']+cfg['kvmem']['gen_budget']:
                assert pressure and all(e['attention_source']=='prefill' and e['prefill_attention_samples']>0 for e in pressure)
        assert one['samples']>0 and one['positive_pages']>0 and one['rounds']==1,one
        generated=first['choices'][0]['token_ids']
        assert len(generated)==257
        assert one['samples']==expected_samples(generated,prompt),one
        # Exact token-prefix reuse, including the terminal token not evaluated
        # until this next turn; no re-tokenization of the generated text.
        header=tok.encode('<|im_end|>\n<|im_start|>user\n',add_special_tokens=False)
        question=tok.encode('Explain in detail how the earlier archive separates historical KV from the active window. '
                            'Continue with a detailed technical design.',add_special_tokens=False)
        footer=tok.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
        prefix=prompt+generated+header
        second=request(session,prefix+question+footer,[len(prefix),len(prefix)+len(question)],129)
        (a.output/f'{index}-second.json').write_text(json.dumps(second))
        two=wait_event('action_attention.jsonl',second['id'])
        assert two['samples']==expected_samples(second['choices'][0]['token_ids'],prefix+question+footer),two
        refresh=wait_event('bounded_refresh.jsonl',second['id'])
        assert two['samples']>0 and two['rounds']==2,two
        assert refresh['action_attention_enabled'] and refresh['action_history_pages']>0,refresh
        assert refresh['action_fraction']==cfg['kvmem']['action_fraction']
        assert refresh['selected_tokens']<=cfg['kvmem']['select_budget']
        return {'client':index,'first':one,'second':two,'query':{k:refresh[k] for k in (
            'request_id','selected_tokens','action_history_pages','action_fraction','action_combine')},
            'first_prompt_tokens':len(prompt),'second_prompt_tokens':len(prefix+question+footer)}
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=a.concurrency) as workers:
        results=list(workers.map(run,range(a.concurrency)))
    report={'status':'passed','concurrency':a.concurrency,'seconds':time.monotonic()-started,'results':results}
    (a.output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

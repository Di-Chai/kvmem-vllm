#!/usr/bin/env python3
"""Below-K public Query learning, regeneration and no-Query continuation gate."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    assert cfg['query_history']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(cfg['tokenizer'],local_files_only=True)
    prefix=tokenizer.apply_chat_template([
        {'role':'system','content':'Remember the evidence and answer concisely.'},
        {'role':'user','content':('The archive records that station Cedar has code 731. '*1000)},
        {'role':'assistant','content':'The archive is available.'}],
        tokenize=True,add_generation_prompt=False,return_dict=False,enable_thinking=False)
    header=tokenizer.encode('<|im_start|>user\n',add_special_tokens=False)
    question=tokenizer.encode('Find the station Cedar code in the earlier archive.',add_special_tokens=False)
    footer=tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    tokens=prefix+header+question+footer
    q=[len(prefix)+len(header),len(prefix)+len(header)+len(question)]
    assert 8192<len(tokens)<cfg['active_context_tokens']
    session='query-relevance-'+uuid.uuid4().hex
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results=[]
    for step in range(3):
        current=tokens if step<2 else tokens+tokenizer.encode(' Continue using the same task evidence.',add_special_tokens=False)
        payload={'model':cfg['served_model_name'],'prompt':current,'max_tokens':1,
            'ignore_eos':True,'temperature':0,'return_token_ids':True,'session_id':session,
            'kvmem_query_span':q,'kvmem_reselect':'force' if step<2 else 'auto'}
        request=urllib.request.Request(f"http://127.0.0.1:{cfg['backend_port']}/v1/completions",
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with http.open(request,timeout=180) as response:
            result=json.load(response)
        assert result['usage']['completion_tokens']==1
        assert result['usage']['prompt_tokens']==len(current)
        results.append(result)
    deadline=time.monotonic()+30
    while True:
        path=args.capture/'query_history.jsonl'
        events=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        own=[r for result in results for r in events if r['request_id'].startswith(result['id']+'-')]
        if len(own)==3:break
        if time.monotonic()>deadline:raise TimeoutError('Missing committed relevance round')
        time.sleep(.1)
    assert [r['round'] for r in own]==[1,1,2], 'Regeneration rewarded the same input twice'
    assert [r['observed'] for r in own]==[True,True,False]
    assert all(r['scored_pages']>0 for r in own)
    assert own[1]['score_sha256']==own[2]['score_sha256'], 'No-Query continuation aged evidence'
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'passed','prompt_tokens':len(tokens),'query_span':q,'history':own}
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

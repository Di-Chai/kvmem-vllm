#!/usr/bin/env python3
"""Real-model complete Query, partial-tail, cross-turn and concurrency gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gateway',default='http://127.0.0.1:18382')
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=1)
    p.add_argument('--first-output-tokens',type=int,default=257)
    p.add_argument('--second-output-tokens',type=int,default=129)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text())
    if not cfg.get('verify_session_restore'):
        raise ValueError('Qualification requires verify_session_restore: true in the running profile')
    assert 1<=a.first_output_tokens<=cfg['gen_budget_tokens']
    assert 1<=a.second_output_tokens<=cfg['gen_budget_tokens']
    def local_path(name):
        path=Path(cfg[name]).expanduser()
        return path if path.is_absolute() else a.config.resolve().parent/path
    users=json.loads(local_path('users_file').read_text())['users'][:a.concurrency]
    assert len(users)==a.concurrency
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(local_path('tokenizer'),local_files_only=True,trust_remote_code=True)
    archive='CPU stores immutable history. GPU attention uses selected historical pages. Tool results are not user questions.\n'
    messages=[{'role':'system','content':'You are a helpful assistant.'},
        {'role':'user','content':archive*2400}, {'role':'assistant','content':'I have received the archive.'}]
    prefix=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,
        enable_thinking=False,return_dict=False)
    assert len(prefix)>cfg['active_context_tokens']
    header=tok.encode('<|im_start|>user\n',add_special_tokens=False)
    footer=tok.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    filler=tok.encode('Review this information carefully and preserve its history. '*500,add_special_tokens=False)[:1950]
    question=tok.encode('Explain why historical KV and the selected active context have different budgets.',add_special_tokens=False)
    q=filler+question
    first=prefix+header+q+footer
    span=[len(prefix)+len(header),len(prefix)+len(header)+len(q)]
    assert span[1]-span[0]>1024 and span[0]%128
    fixture={'input_ids':first,'query_span':span,'sha256':hashlib.sha256(bytes(str(first),'utf8')).hexdigest()}
    (a.output/'input.json').write_text(json.dumps(fixture))
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def read_events(name):
        path=a.capture/name
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    event_start=len(read_events('bounded_refresh.jsonl'))
    state_start=len(read_events('verified_state_restores.jsonl'))
    def call(path,key,body=None,method='GET'):
        r=urllib.request.Request(a.gateway+path,data=None if body is None else json.dumps(body).encode(),
            method=method,headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        try:
            with http.open(r,timeout=cfg['turn_timeout_seconds']+30) as response: return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from None
    def run(index):
        key=users[index]['api_key'];sid='explicit-query-'+uuid.uuid4().hex
        started=time.monotonic()
        first_result=call('/v1/kvmem/sessions/'+sid+'/turn',key,
            {'append_token_ids':first,'max_tokens':a.first_output_tokens,'ignore_eos':True,'waiting_for_tool':True,
             'kvmem_query_span':span,'kvmem_reselect':'force'},'POST')
        assert first_result['history_tokens']==len(first)+a.first_output_tokens
        tool=tok.encode('<|im_end|>\n<|im_start|>user\n<tool_response>CPU history is intact.</tool_response>\n',add_special_tokens=False)
        second_q=q[:1301]
        tail=tool+header+second_q+footer
        q2_begin=first_result['history_tokens']+len(tool)+len(header)
        q2=[q2_begin,q2_begin+len(second_q)]
        second_result=call('/v1/kvmem/sessions/'+sid+'/turn',key,
            {'version':1,'append_token_ids':tail,'max_tokens':a.second_output_tokens,'ignore_eos':True,
             'kvmem_query_span':q2,'kvmem_reselect':'force'},'POST')
        assert second_result['history_tokens']==len(first)+a.first_output_tokens+len(tail)+a.second_output_tokens
        status=call('/v1/kvmem/status',key)
        assert {s['session_id'] for s in status['sessions']}=={sid}, 'Cross-user history exposure'
        row={'client_index':index,'session_id':sid,'first_query':span,'second_query':q2,
             'history_tokens':second_result['history_tokens'],'outputs':[a.first_output_tokens,a.second_output_tokens],
             'seconds':time.monotonic()-started,'first_turn_seconds':first_result['turn_seconds'],
             'second_turn_seconds':second_result['turn_seconds'],'status':status,
             'first_native_response':first_result['native_response'],'second_native_response':second_result['native_response']}
        (a.output/f'client-{index}.json').write_text(json.dumps(row))
        call('/v1/kvmem/sessions/'+sid,key,method='DELETE')
        print(json.dumps({k:row[k] for k in ('client_index','session_id','history_tokens','seconds')}),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool: rows=list(pool.map(run,range(a.concurrency)))
    events=read_events('bounded_refresh.jsonl')[event_start:]
    explicit=[e for e in events if e.get('selection')=='query']
    assert len(explicit)==2*a.concurrency, 'Missing or duplicate complete Query replay'
    assert len({e['request_id'] for e in explicit})==2*a.concurrency
    for row in rows:
        for span,response in ((row['first_query'],row['first_native_response']),
                              (row['second_query'],row['second_native_response'])):
            matches=[e for e in explicit if e['request_id'].startswith(response['id']+'-')]
            assert len(matches)==1 and matches[0]['query_span']==span
            assert matches[0]['query_score_tokens']==span[1]-span[0]
            assert matches[0]['replay_span']==[span[0]//128*128,span[1]]
    resources=json.loads((a.capture/'scheduler_resources.json').read_text())
    # Allow background retirement to acknowledge without counting HTTP completion
    # itself as proof that GPU ownership has drained.
    deadline=time.monotonic()+30
    while (resources['requests'] or resources['owned_blocks']) and time.monotonic()<deadline:
        time.sleep(.2);resources=json.loads((a.capture/'scheduler_resources.json').read_text())
    assert resources['requests']==resources['owned_blocks']==0
    assert resources['sessions']['hits']>=a.concurrency
    states=read_events('verified_state_restores.jsonl')[state_start:]
    assert len(states)==a.concurrency, 'Every second turn must restore a CPU checkpoint'
    assert all(s['recurrent_byte_equal'] and s['recurrent_tensors']==96 for s in states)
    assert all(sum(s['request_id'].startswith(row['second_native_response']['id']+'-') for s in states)==1 for row in rows)
    result={'status':'passed','concurrency':a.concurrency,'input_tokens':len(first),
        'query_tokens':len(q),'query_replays':explicit,'resources':resources,'state_restores':states,
        'clients':[{k:r[k] for k in ('client_index','history_tokens','outputs','seconds','first_turn_seconds','second_turn_seconds')} for r in rows]}
    (a.output/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':result['status'],'concurrency':a.concurrency,'query_tokens':len(q),
        'session_hits':resources['sessions']['hits'],'owned_blocks':resources['owned_blocks']}),flush=True)


if __name__=='__main__': main()

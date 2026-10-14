#!/usr/bin/env python3
"""Real-model P/M, stable tool continuation and mandatory-page qualification."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--prompt-token-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,default=8)
    parser.add_argument('--output-tokens',type=int,default=17)
    parser.add_argument('--pressure-tool-tokens',type=int,default=0)
    parser.add_argument('--oversized-file',action='store_true')
    args=parser.parse_args()
    assert 0<=args.pressure_tool_tokens<=32768
    cfg=json.loads(args.config.read_text())
    assert cfg.get('verify_session_restore') and cfg.get('session_checkpoint_policy')=='pm'
    assert 1<=args.concurrency<=cfg['max_num_seqs'] and 1<=args.output_tokens<=cfg['gen_budget_tokens']
    args.output.mkdir(parents=True,exist_ok=False)
    fixture=json.loads(args.prompt_token_file.read_text());first=fixture['input_ids'];query=fixture['query_span']
    assert len(first)>cfg['active_context_tokens'] and query[0]//128*128>cfg['active_context_tokens']
    def path(key):
        value=Path(cfg[key]).expanduser()
        return value if value.is_absolute() else args.config.resolve().parent/value
    users=json.loads(path('users_file').read_text())['users'][:args.concurrency]
    assert len(users)==args.concurrency
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(path('tokenizer'),local_files_only=True,trust_remote_code=True)
    tool=tokenizer.encode('<|im_end|>\n<|im_start|>user\n<tool_response>History is intact. '+
        'The file remains available for retrieval. '*300+'</tool_response>\n',add_special_tokens=False)[:2048]
    force_tail=tokenizer.encode('Continue the same task after checking the retained evidence.\n',add_special_tokens=False)
    header=tokenizer.encode('<|im_end|>\n<|im_start|>user\n',add_special_tokens=False)
    fresh=tokenizer.encode('Explain how the selected active window preserves required file evidence. '*100,add_special_tokens=False)[:1301]
    footer=tokenizer.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base=f"http://127.0.0.1:{cfg['port']}"
    def call(endpoint,key,body=None,method='GET'):
        req=urllib.request.Request(base+endpoint,method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        try:
            with http.open(req,timeout=cfg['turn_timeout_seconds']+30) as response:return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from None
    def events(name):
        p=args.capture/name
        return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
    def view(sid):
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            rows=json.loads((args.capture/'scheduler_resources.json').read_text()).get('session_views',[])
            matches=[r for r in rows if r['namespace'][1]==sid]
            if len(matches)==1:return matches[0]
            time.sleep(.05)
        raise AssertionError('Native M checkpoint did not become idle')
    refresh_start=len(events('bounded_refresh.jsonl'));state_start=len(events('verified_state_restores.jsonl'))
    pin={'begin':12800,'end':48000 if args.oversized_file else 13312,'kind':'file'}
    def run(index):
        key=users[index]['api_key'];sid='context-policy-'+uuid.uuid4().hex
        endpoint='/v1/kvmem/sessions/'+sid+'/turn'
        started=time.monotonic()
        a=call(endpoint,key,{'append_token_ids':first,'max_tokens':args.output_tokens,'ignore_eos':True,
            'waiting_for_tool':True,'kvmem_query_span':query,'kvmem_reselect':'force',
            'kvmem_mandatory_spans':[pin]},'POST')
        av=view(sid)
        assert av['history']==a['history_tokens']-1,'M must retain every evaluated accepted token'
        b=call(endpoint,key,{'version':1,'append_token_ids':tool,'max_tokens':args.output_tokens,'ignore_eos':True,
            'waiting_for_tool':True,'kvmem_reselect':'off','kvmem_prefill_window':'keep_selected'},'POST')
        bv=view(sid)
        assert b['history_tokens']>cfg['active_context_tokens']+cfg['gen_budget_tokens']
        old_prefix=lambda r:[span for span in r['active_records'] if span[0]<query[0]//128*128]
        assert old_prefix(av)==old_prefix(bv),'Stable tool callback changed selected historical pages'
        c=call(endpoint,key,{'version':2,'append_token_ids':force_tail,'max_tokens':args.output_tokens,
            'ignore_eos':True,'kvmem_query_span':query,'kvmem_reselect':'force'},'POST')
        cv=view(sid)
        # Advance past the current partial block before introducing a fresh Q.
        pad=tokenizer.encode('Read the next question carefully. '*30,add_special_tokens=False)
        live_begin=c['history_tokens']+len(pad)
        qbegin=live_begin+len(header);qend=qbegin+len(fresh)
        qspan=[qbegin,qend]
        d=call(endpoint,key,{'version':3,'append_token_ids':pad+header+fresh+footer,
            'max_tokens':args.output_tokens,'ignore_eos':True,'kvmem_query_span':qspan,
            'kvmem_replay_span':[live_begin,qend],'kvmem_reselect':'force'},'POST')
        dv=view(sid)
        status=call('/v1/kvmem/status',key)
        assert {s['session_id'] for s in status['sessions']}=={sid},'Cross-user session exposure'
        assert d['history_tokens']==len(first)+4*args.output_tokens+len(tool)+len(force_tail)+len(pad)+len(header)+len(fresh)+len(footer)
        rows=[a,b,c,d];views=[av,bv,cv,dv]
        if args.pressure_tool_tokens:
            pressure_tool=(tool*((args.pressure_tool_tokens+len(tool)-1)//len(tool)))[:args.pressure_tool_tokens]
            assert dv['window']+len(pressure_tool)+args.output_tokens>cfg['active_context_tokens']+cfg['gen_budget_tokens']
            e=call(endpoint,key,{'version':4,'append_token_ids':pressure_tool,
                'max_tokens':args.output_tokens,'ignore_eos':True,'kvmem_reselect':'off',
                'kvmem_prefill_window':'keep_selected'},'POST')
            ev=view(sid);rows.append(e);views.append(ev)
            assert e['history_tokens']==d['history_tokens']+len(pressure_tool)+args.output_tokens
        for response,saved in zip(rows,views):
            assert saved['history']==response['history_tokens']-1,'M checkpoint unavailable: inspect checkpoint quota/deferrals'
        for r in rows:assert r['native_response']['usage']['completion_tokens']==args.output_tokens
        result={'client_index':index,'sid':sid,'seconds':time.monotonic()-started,
            'native_ids':[r['native_response']['id'] for r in rows],
            'histories':[r['history_tokens'] for r in rows],
            'turn_seconds':[r['turn_seconds'] for r in rows],
            'm_checkpoints':[r['history'] for r in views],
            'active_windows':[r['window'] for r in views],
            'first_query':query,'fresh_query':qspan,'fresh_replay':[live_begin//128*128,qend]}
        (args.output/f'client-{index}.json').write_text(json.dumps(result,indent=2))
        call('/v1/kvmem/sessions/'+sid,key,method='DELETE')
        print(json.dumps({'client':index,'seconds':result['seconds'],'history':result['histories'][-1]}),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:clients=list(pool.map(run,range(args.concurrency)))
    refresh=events('bounded_refresh.jsonl')[refresh_start:]
    states=events('verified_state_restores.jsonl')[state_start:]
    queries=[r for r in refresh if r['selection']=='query']
    assert len(queries)==3*args.concurrency and len(states)==(3+bool(args.pressure_tool_tokens))*args.concurrency
    for client in clients:
        own=lambda data,i:[r for r in data if r['request_id'].startswith(client['native_ids'][i]+'-')]
        assert not own(refresh,1),'Stable callback performed a selection transition'
        expected=[(1,'M',client['m_checkpoints'][0]),(2,'P',query[0]//128*128),(3,'M',client['m_checkpoints'][2])]
        if args.pressure_tool_tokens:expected.append((4,'M',client['m_checkpoints'][3]))
        for i,kind,history in expected:
            r,=own(states,i)
            assert r['checkpoint_kind']==kind and r['history_tokens']==history
            assert r['recurrent_tensors']==96 and r['recurrent_byte_equal'] and not r['pressure_restore']
        for i,q in ((0,query),(2,query),(3,client['fresh_query'])):
            r,=own(queries,i)
            assert r['query_span']==q and r['query_score_tokens']==q[1]-q[0]
            selected={b//128 for b,n in r['selected_history_ranges']}
            assert set(r['mandatory_fit']['hard_blocks']).issubset(selected)
            assert set(range(100,104)).issubset(selected),'Required file evidence was evicted'
            if i==3:assert r['replay_span']==client['fresh_replay']
        if args.pressure_tool_tokens:
            pressure=own(refresh,4)
            assert pressure and all(r['selection']!='query' for r in pressure)
            for r in pressure:
                selected={b//128 for b,n in r['selected_history_ranges']}
                assert set(r['mandatory_fit']['hard_blocks']).issubset(selected)
        if args.oversized_file:
            for r in own(queries,0):
                assert r['mandatory_fit']['capacity_fitted']
                assert r['mandatory_fit']['retrieval_reserve_blocks']==32
                selected={b//128 for b,n in r['selected_history_ranges']}
                assert set(range(371,375)).issubset(selected)
    resource=json.loads((args.capture/'scheduler_resources.json').read_text())
    assert resource['requests']==resource['owned_blocks']==0
    report={'status':'passed','concurrency':args.concurrency,'input_tokens':len(first),
        'outputs_per_turn':args.output_tokens,'successful_turns':(4+bool(args.pressure_tool_tokens))*args.concurrency,
        'pressure_tool_tokens':args.pressure_tool_tokens,'oversized_file':args.oversized_file,'refreshes':refresh,
        'query_replays':queries,'state_restores':states,'resources':resource,
        'clients':[{k:v for k,v in r.items() if k not in ('sid','native_ids')} for r in clients]}
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':'passed','turns':report['successful_turns'],'restores':len(states),'owned_blocks':resource['owned_blocks']}),flush=True)


if __name__=='__main__':main()

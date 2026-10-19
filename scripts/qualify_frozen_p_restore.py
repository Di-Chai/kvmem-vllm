"""Real-model frozen-P restore qualification on a synthetic two-owner fixture.

This validates fences, state bytes and question independence. It is not an
original MAB score or a performance measurement; verification copies are on.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request
import uuid

from benchmark_client_io import CaptureSlice
from benchmark_memory import digest_tokens, wait_terminal
from benchmark_niah import save
import graph_v2_protocol as protocol
import qualify_block_graph as q


def run(args):
    cfg=json.loads(args.config.read_text());fixture=json.loads(args.fixture.read_text())
    kv=cfg['kvmem'];asynchronous=kv['async_restore']
    q.require(fixture['schema']=='frozen-p-restore-functional-1' and len(fixture['owners'])==2,
        'Use the independent synthetic two-owner fixture')
    q.require(kv['session_freeze_prompt'] and kv['session_checkpoint_policy']=='sealed' and
        not kv['session_capture_terminal'] and not kv['session_capture_decode'] and not kv['query_history'],
        'Independent questions must retain only the original P')
    q.require(cfg['verify_session_restore'] and kv['step_latency_interval']==1,
        'Require state byte oracles and native batch evidence')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def owner_run(owner):
        session='frozen-p-qual-'+uuid.uuid4().hex
        prefix=owner['prefix'];samples=None;results=[]
        for index,question in enumerate(owner['questions']):
            directory=args.output/owner['id']/str(index)
            tokens=prefix+question['suffix'];span=[len(prefix),len(tokens)]
            q.require(digest_tokens(tokens)==question['token_sha256'],'Synthetic input binding changed')
            payload=dict(model=cfg['served_model_name'],prompt=tokens,session_id=session,cache_salt=session,
                return_token_ids=True,temperature=0,seed=17,max_tokens=32,ignore_eos=True,
                thinking_token_budget=0,kvmem_query_span=span,kvmem_replay_span=span,kvmem_reselect='force')
            cursor=CaptureSlice(args.capture/name for name in ('bounded_refresh.jsonl','block_graph.jsonl'))
            save(directory/'request.json',{k:v for k,v in payload.items() if k!='prompt'}|
                dict(prompt_tokens=len(tokens),token_sha256=question['token_sha256']))
            http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request=urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/v1/completions",
                data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with http.open(request,timeout=600) as response:answer=json.load(response)
            save(directory/'response.json',answer)
            q.require(answer['choices'][0]['prompt_token_ids']==tokens and
                answer['usage']['prompt_tokens']==len(tokens) and answer['usage']['completion_tokens']==32,
                'Model input or accepted output accounting differs')
            rid=answer['id'];terminal=wait_terminal(args.capture,rid)
            q.require(terminal['checkpoint_kind']=='P' and terminal['checkpoint_tokens']==len(prefix),
                'Question did not retire to its exact input checkpoint')
            evidence=protocol.audit_case(args.capture,rid,cfg,span,require_pressure=not index,event_reader=cursor.read)
            selected=evidence['query']
            if samples is None:samples=selected['prefill_attention_samples']
            q.require(selected['prefill_attention_samples']==samples and selected['action_history_pages']==0,
                'Previous question leaked input/output statistics')
            pressure=[event for event in cursor.read(args.capture/'bounded_refresh.jsonl')
                if q.native_matches({'response_id':rid},event) and event['selection']=='pressure']
            q.require(not index or not pressure,'Frozen P was recomputed between independent questions')
            record=dict(owner=owner['id'],round=index,response_id=rid,prompt_tokens=len(tokens),
                token_sha256=question['token_sha256'],prefix_sha256=digest_tokens(prefix),
                checkpoint_tokens=terminal['checkpoint_tokens'],prefill_samples=samples,
                output_history_pages=selected['action_history_pages'],evidence=evidence,terminal=terminal)
            save(directory/'result.json',record);results.append(record)
        release=protocol.release_owner(cfg,args.capture,session,tokens[-1])
        save(args.output/owner['id']/'release.json',release)
        return results
    with ThreadPoolExecutor(max_workers=2) as pool:
        groups=list(pool.map(owner_run,fixture['owners']))
    released=q.wait_released(args.capture,[],timeout=180)
    results=[row for group in groups for row in group]
    warm=[row for row in results if row['round']]
    matches=lambda row,events:[event for event in events if q.native_matches(row,event)]
    verified=q.records(args.capture/'verified_restores.jsonl')
    captures=q.records(args.capture/'checkpoint_timing.jsonl')
    restores=q.records(args.capture/'async_restore.jsonl')
    for row in warm:
        oracle=matches(row,verified)
        q.require(len(oracle)==1 and oracle[0]['byte_equal'] and oracle[0]['recurrent_tensors']==96 and
            oracle[0]['working_attention_layers']==17 and oracle[0]['mtp_attention_layers']==1,
            'Real target/MTP/recurrent state byte oracle failed')
        imported=matches(row,restores)
        q.require(len(imported)==int(asynchronous),'Expected restore execution path differs')
    expected_captures=len(groups) if asynchronous else len(results)
    owned_captures=[event for row in results for event in matches(row,captures)]
    q.require(len(owned_captures)==expected_captures and all(not event['terminal'] for event in owned_captures),
        'Frozen P was unexpectedly replaced or captured as M')
    steps=[event for event in q.records(args.capture/'model_step_latency.jsonl') if event['started_at']>=started]
    peak=max((event['decode_requests']+event['prefill_requests'] for event in steps),default=0)
    q.require(peak>=2,'Two client owners did not execute in a real native batch')
    archive=released['archive']
    q.require(not archive.get('disk_read_bytes') and not archive.get('disk_write_bytes'),
        'Unexpected disk KV in CPU-history qualification')
    return dict(status='passed',all_owned_sessions_released=True,async_restore=asynchronous,
        cases=len(results),owners=len(groups),real_restores=len(warm),native_batch_peak=peak,
        target_mtp_recurrent_byte_equal=True,p_checkpoint_captures=len(owned_captures),
        prefill_statistics_preserved=True,output_statistics_excluded=True,
        scope='Synthetic functional inputs; full byte oracles enabled. Not original MAB quality or throughput.',
        final_archive=archive,results=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','capture','output','fixture'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',required=True)
    args=parser.parse_args()
    try:
        result=run(args);save(args.output/'results.json',result)
    except BaseException as error:
        save(args.output/'client-failure.json',dict(error=repr(error),at=time.time()))
        raise
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','final_archive')}),flush=True)


if __name__=='__main__':main()

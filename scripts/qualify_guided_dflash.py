"""Owned two-session guided Query/DFlash state gate against a running server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import uuid

from qualify_block_graph import (build_fixture, edited_turn, expected_output_samples,
    native_matches, next_turn, records, require, save, stream_request,
    wait_released, wait_terminal_publication)


def run(args):
    profile=json.loads(args.config.read_text());kv=profile['kvmem']
    require(profile['draft_method']=='dflash' and profile['draft_tokens']==7 and
        profile['execution_mode']=='full_graph' and profile['verify_session_restore'],
        'Require DFlash7 FULL graphs and byte verification')
    require(0<kv['guided_query_tokens']<=4096 and kv['graph_mode']=='multi' and
        kv['graph_scoring_version']=='v2' and kv['pressure_policy']=='attention',
        'Require guided Query with V2 multi and attention pressure')
    args.output.mkdir(parents=True,exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(profile['tokenizer'],local_files_only=True)
    fixture=build_fixture(tokenizer,prompt_tokens=max(12288,kv['select_budget']+4096),concurrency=2)
    save(args.output/'fixture.json',fixture)
    sessions=['guided-dflash-'+uuid.uuid4().hex for _ in range(2)]
    rows=[];errors=[];report={};started=time.monotonic()

    def client(index):
        a=fixture['agents'][index];prompt=a['initial_tokens']
        query=a['query_span'];spans=a['message_spans'];before=second_spans=None
        for turn in range(3):
            row=stream_request(profile,sessions[index],prompt,257,129,query=query,spans=spans)
            row.update(agent=index,turn=turn,expected_code=a['expected_codes'][turn],
                expected_output_samples=expected_output_samples(tokenizer,profile,prompt,row['output_ids']))
            rows.append(row);save(args.output/f'owner-{index}-turn-{turn}.json',row)
            if turn==0:
                prompt,query,spans,before=next_turn(a,prompt,row['output_ids']);second_spans=spans
            elif turn==1:
                prompt,query,spans=edited_turn(a,before,second_spans)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(client,i) for i in range(2)]:future.result()
        archive=wait_terminal_publication(args.capture,rows)
        refreshes=records(args.capture/'bounded_refresh.jsonl')
        graphs=records(args.capture/'block_graph.jsonl')
        actions=records(args.capture/'action_attention.jsonl')
        rollback=records(args.capture/'verified_guided_rollback.jsonl')
        restores=records(args.capture/'verified_restores.jsonl')
        published=[]
        for row in rows:
            q=[r for r in refreshes if native_matches(row,r) and r.get('selection')=='query']
            require(len(q)==1,'Missing or duplicate Query selection')
            q=q[0]
            require(q['selected_tokens']<=kv['select_budget'] and
                q['guided_query_tokens']<=kv['guided_query_tokens'],'Guided scratch/budget exceeded')
            if q.get('guided_published_tokens',0):
                require(q['guided_natural_termination'] and q['guided_fallback'] is None and
                    q['query_score_tokens']==q['guided_published_tokens'],'Invalid private Q published')
                published.append(q)
            if q['guided_query_tokens']:
                if kv.get('guided_query_backend') == 'cuda_graph':
                    require(q.get('guided_execution_mode')=='cuda_graph' and
                        q.get('guided_graph_replays')==q['guided_query_tokens'] and
                        q.get('guided_reference_token_equal') and q.get('guided_reference_q_equal'),
                        'Private graph missed transitions or differs from the eager token/Q oracle')
                own=[r for r in rollback if native_matches(row,r)]
                require(own and all(r['recurrent_byte_equal'] and r['draft_byte_equal'] and
                    r['draft_layers']>=5 for r in own),'Missing exact private rollback/draft proof')
            g=[r for r in graphs if native_matches(row,r) and r.get('phase')=='query']
            require(g and all(r['backend']=='cpp_v2' and r['mode']=='multi' and
                r['alpha']==kv['graph_alpha'] for r in g),'Guided Query bypassed V2 propagation')
            action=[r for r in actions if native_matches(row,r)]
            require(len(action)==1 and action[0]['samples']==row['expected_output_samples'],
                'Private or rejected rows contaminated accepted output attention')
            if row['turn']:
                own=[r for r in restores if native_matches(row,r)]
                require(own and all(r['byte_equal'] and r['draft_layers']==5 for r in own),
                    'Missing target/GDN/five-layer draft restore proof')
        require(published,'Guided enabled but no private Query was actually published')
        require(archive['session_checkpoint_deferrals']==0,'Checkpoint quota fallback in state gate')
        report=dict(status='passed',rows=rows,guided_published_queries=len(published),
            guided_fallbacks=[q.get('guided_fallback') for q in refreshes if q.get('selection')=='query'],
            rollback=rollback,restores=restores,
            quality=dict(correct=sum(r['expected_code'] in r['text'] for r in rows),total=len(rows)),
            scope='Two owned sessions, Query/tool continuation/suffix edit; state gate, not an agent speed benchmark')
    except BaseException as error:
        errors.append(repr(error));raise
    finally:
        releases=[]
        for session in sessions:
            try:
                releases.append(stream_request(profile,session,[fixture['eos_token_id']],1,1,release=True))
            except Exception as error:errors.append('release: '+repr(error))
        try:released=wait_released(args.capture,sessions,timeout=60)
        except Exception as error:released=None;errors.append('release check: '+repr(error))
        report.update(rows=rows,errors=errors,releases=releases,
            all_owned_sessions_released=released is not None,seconds=time.monotonic()-started)
        if errors or released is None:report['status']='failed'
        save(args.output/'results.json',report)
    require(report['status']=='passed','Guided DFlash gate failed; retain evidence')
    print(json.dumps(dict(status='passed',queries=report['guided_published_queries'],quality=report['quality'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','capture','output'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())

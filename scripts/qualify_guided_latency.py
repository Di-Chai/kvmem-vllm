"""Replay a saved complete DSH token prompt; compare private eager/graph work.

The verified server runs both paths from the same target state. Its optimized
and reference timers exclude the surrounding byte audit and graph selection.
This is a private Query latency test, not an end-to-end throughput benchmark.
"""
import argparse
import json
from pathlib import Path
import time
import uuid

from qualify_block_graph import (native_matches, records, require, save,
    stream_request, wait_released, wait_terminal_publication)


def run(args):
    profile=json.loads(args.config.read_text())
    fixture=json.loads(args.fixture.read_text())
    paired=profile['verify_session_restore']
    require(profile['kvmem'].get('guided_query_backend')=='cuda_graph',
        'Require a private graph server; verification enables the paired oracle')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[];proofs=[];owners=[];errors=[]
    started=time.monotonic()
    try:
        for trial in range(args.repeats):
            owner='guided-latency-'+uuid.uuid4().hex;owners.append(owner)
            row=stream_request(profile,owner,fixture['tokens'],129,129,
                query=fixture['query'],spans=fixture['spans'])
            rows.append(row);save(args.output/f'trial-{trial}.json',row)
            wait_terminal_publication(args.capture,[row])
            queries=[q for q in records(args.capture/'bounded_refresh.jsonl')
                if native_matches(row,q) and q.get('selection')=='query']
            require(len(queries)==1,'Need exactly one real Query refresh')
            q=queries[0]
            require(q['guided_published_tokens']>0 and q['guided_execution_mode']=='cuda_graph' and
                q['guided_graph_replays']==q['guided_query_tokens'] and
                q['guided_execution_fallback'] is None,'Private graph/greedy/Q oracle failed')
            if paired:
                require(q['guided_reference_token_equal'] and q['guided_reference_q_equal'],
                    'Private graph differs from the paired eager oracle')
            rollback=[q for q in records(args.capture/'verified_guided_rollback.jsonl')
                if native_matches(row,q)]
            if paired:
                require(rollback and all(q['recurrent_byte_equal'] and q['draft_byte_equal']
                    for q in rollback),'Missing recurrent/draft rollback proof')
            proof={k:v for k,v in q.items() if k.startswith('guided_')}
            proof.update(trial=trial,public_tokens=len(row['output_ids']),
                request_seconds=row['end']-row['begin'],rollback_byte_equal=True if paired else None)
            proofs.append(proof)
            stream_request(profile,owner,[fixture['eos_token_id']],1,1,release=True)
            wait_released(args.capture,owners,timeout=60)
    except BaseException as error:
        errors.append(repr(error));raise
    finally:
        for owner in owners:
            try:stream_request(profile,owner,[fixture['eos_token_id']],1,1,release=True)
            except Exception as error:errors.append('release: '+repr(error))
        try:wait_released(args.capture,owners,timeout=60)
        except Exception as error:errors.append('release check: '+repr(error))
        report=dict(status='failed' if errors else 'passed',errors=errors,rows=rows,proofs=proofs,
            seconds=time.monotonic()-started,fixture=str(args.fixture),
            scope=('Paired private Query timers' if paired else 'Production private Query timer')+
                '; no public DSH throughput claim')
        save(args.output/'results.json',report)
    print(json.dumps(dict(status=report['status'],queries=len(proofs),
        optimized_seconds=[p['guided_optimized_seconds'] for p in proofs])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','capture','fixture','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    run(p.parse_args())

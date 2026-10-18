"""Exercise the observed hot-tail pressure edge with six live decode peers.

Correctness qualification only: the exact failed prompt supplies content and
geometry; new output IDs remain the current model's own accepted history.
No DeepSWE task is scored or retried by this entry point.
"""
import argparse
import array
import asyncio
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pressure_qualification_shared',ROOT/'scripts/qualify_rolling_window.py')
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)
cap, work, core = shared.cap, shared.work, shared.core


async def workload(client, profile, fixture, directory, capture):
    directory.mkdir()
    recorder = work.Recorder(directory,8)
    histories, publications = {}, []
    original = fixture['tokens']
    sessions = [f'pressure-tail-qualification-{i}' for i in range(8)]
    plan = dict(kvmem_reselect='off',kvmem_prefill_window='keep_selected',kvmem_mandatory_spans=[])
    async def call(i, turn, tokens, count, cleanup=False):
        row = await work.stream_call(client,profile,tokens,sessions[i],count,recorder,
            agent=i,epoch=0,turn=turn,kind='cleanup' if cleanup else 'workload',
            request_id=f'pressure-edge-{i}-{turn}',new_input_tokens=len(tokens),
            query=not cleanup,query_plan=plan if turn==0 else dict(
                kvmem_reselect='auto',kvmem_prefill_window='pressure',kvmem_mandatory_spans=[]))
        if not cleanup:histories[i]=tokens+row['output_ids']
        return row
    async def drain(rows, ids, name):
        state=await work.wait_retirement(capture,timeout=180,not_before=max(r['begin'] for r in rows),
            expected_views={sessions[i]:len(histories[i])-1 for i in ids})
        cap.save(directory/f'{name}-resources.json',state)
        archive=cap.read(capture/'archive.json');publications.extend(archive.get('completed',[]))
        cap.save(directory/f'{name}-archive.json',archive)
        return shared.history_evidence(rows,archive.get('completed',[]))
    peers=[]
    try:
        prime=await asyncio.gather(*(call(i,0,original[:98254],1) for i in range(2)))
        proofs=await drain(prime,range(2),'prime')
        for view in cap.read(capture/'scheduler_resources.json')['session_views']:
            if view['namespace'][0] in sessions[:2]:
                cap.require(view['history']==view['window']==98254,'Priming did not retain the exact old full view')
        peers=[asyncio.create_task(call(i,0,original[:2048],8192)) for i in range(2,8)]
        peer_begin=time.monotonic();deadline=peer_begin+120
        while True:
            steps=core.read_records(capture/'model_step_latency.jsonl')
            if any(r['started_at']>=peer_begin and r.get('decode_requests')==6 and not r.get('prefill_tokens') for r in steps):break
            cap.require(not any(t.done() for t in peers) and time.monotonic()<deadline,'Peers did not establish live decode')
            await asyncio.sleep(.2)
        # Replace the old final output token with the current priming output;
        # preserve the entire original tool suffix and exact request length.
        edge=await asyncio.gather(*(call(i,1,histories[i]+original[98255:],256) for i in range(2)))
        cap.require(all(r['input_tokens']==99826 for r in edge),'Failed-request geometry changed')
        proofs.extend(await drain(edge,range(2),'pressure'))
        resumed=await asyncio.gather(*(call(i,2,histories[i]+original[:32],64) for i in range(2)))
        proofs.extend(await drain(resumed,range(2),'resumed'))
        peer_rows=await asyncio.gather(*peers)
        proofs.extend(await drain(peer_rows,range(2,8),'peers'))
        controls=core.read_records(capture/'refresh_control_allocations.jsonl')
        restores=core.read_records(capture/'verified_restores.jsonl')
        states=core.read_records(capture/'verified_state_restores.jsonl')
        refreshes=core.read_records(capture/'bounded_refresh.jsonl')
        for row in edge:
            matches=[r for r in controls if core.native_matches(row,r['request_id'])]
            cap.require(len(matches)==1 and matches[0]['history']==98272 and
                matches[0]['replay_tokens']==32 and matches[0]['private_pressure_verified'] is True and
                matches[0]['public_peer_count']>=1,'Exact pressure control or peer progress was not exercised')
            selected=[r for r in refreshes if core.native_matches(row,r['request_id'])]
            cap.require(len(selected)==1 and selected[0]['selected_tokens']==65536 and
                selected[0]['selection']=='pressure' and not selected[0]['cancelled'], 'Pressure selection did not complete')
        for row in [*edge,*resumed]:
            target=[r for r in restores if core.native_matches(row,r['request_id'])]
            recurrent=[r for r in states if core.native_matches(row,r['request_id'])]
            cap.require(target and recurrent and all(r.get('byte_equal') is True and
                r.get('mtp_attention_layers',0)>0 for r in target) and
                all(r.get('recurrent_byte_equal') is True for r in recurrent),'Missing target/MTP/recurrent byte oracle')
        cap.require(not(capture/'restore_mismatch.json').exists(),'Restore mismatch')
        cleanup=await asyncio.gather(*(call(i,3,[fixture['eos']],1,True) for i in range(8)))
        await work.wait_retirement(capture,timeout=180,not_before=max(r['begin'] for r in cleanup),
            expected_views={sid:1 for sid in sessions})
        return dict(status='passed',history_proofs=proofs,controls=controls,restore_proofs=restores,
            recurrent_proofs=states,cleanup_completed=True,
            total_stream_output_tokens=sum(len(r['output_ids']) for r in recorder.rows),
            scope='Two exact near-full restored views and six live decode peers; correctness only, not task scores or throughput')
    finally:
        for task in peers:
            if not task.done():task.cancel()
        if peers:await asyncio.gather(*peers,return_exceptions=True)
        cap.save(directory/'traces.json',recorder.rows)
        cap.save(directory/'history_publications.json',publications)
        recorder.progress();recorder.stream.close()


def prepare(a):
    root=a.output.resolve();cap.require(not root.exists(),'Use a fresh qualification output')
    candidate=a.candidate.resolve();profile=cap.read(a.config)
    cap.require(profile['max_model_len']==196608 and profile['max_num_seqs']==8 and
        profile['kvmem']['select_budget']==65536 and profile['kvmem']['gen_budget']==32768 and
        profile['kvmem']['rolling_active_window'] is True,'Preserve the failed deployment geometry')
    profile.update(port=a.port,verify_session_restore=True)
    values=array.array('i');values.frombytes(a.prompt.read_bytes())
    if sys.byteorder!='little':values.byteswap()
    cap.require(len(values)==99826,'Exact failed prompt required')
    from transformers import AutoTokenizer
    eos=AutoTokenizer.from_pretrained(profile['tokenizer'],local_files_only=True).eos_token_id
    root.mkdir()
    for folder in ('src','csrc','scripts'):
        source = ROOT/folder if folder == 'scripts' else candidate/folder
        shutil.copytree(source,root/'frozen'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('qualify_prefill_pressure.py','qualify_rolling_window.py'):
        shutil.copy2(ROOT/'scripts'/name,root/'frozen/scripts'/name)
    shutil.copytree(candidate,root/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    cap.save(root/'profile.json',profile);cap.save(root/'fixture.json',dict(tokens=values.tolist(),eos=eos))
    cap.save(root/'manifest.json',dict(kind='prefill_pressure_control_qualification_v1',at=time.time(),
        source_sha256=cap.hashes(root/'frozen'),runtime_sha256=cap.hashes(root/'runtime'),
        profile_sha256=cap.sha(root/'profile.json'),fixture_sha256=cap.sha(root/'fixture.json'),
        prompt_source=str(a.prompt.resolve()),prompt_sha256=cap.sha(a.prompt),native_source=cap.native_source_identity(),
        idle_seconds=15,budget=dict(minimum_free_bytes=8*1024**3,required_available_bytes=70_000_000_000),
        scope='Qualification of a new two-file runtime revision; prior failed DeepSWE attempts preserved, no task score'))
    cap.save(root/'status.json',dict(status='prepared',at=time.time()))


async def run(a):
    import psutil
    root=a.output.resolve();m=cap.read(root/'manifest.json')
    cap.require(ROOT.resolve()==(root/'frozen').resolve(),'Execute frozen qualification')
    cap.require(cap.hashes(ROOT)==m['source_sha256'] and cap.hashes(root/'runtime')==m['runtime_sha256'] and
        cap.sha(root/'profile.json')==m['profile_sha256'] and cap.sha(root/'fixture.json')==m['fixture_sha256'] and
        cap.native_source_identity()==m['native_source'],'Qualification source or input changed')
    cap.require(cap.read(root/'status.json')['status']=='prepared','No automatic restart')
    p=psutil.Process();cap.save(root/'owner.json',dict(pid=p.pid,create_time=p.create_time(),argv=sys.argv,at=time.time()))
    loop=asyncio.get_running_loop();current=asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,current.cancel)
    shared.workload=workload
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            await shared.run_server(root,m)
        cap.save(root/'status.json',dict(status='passed',at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json',dict(status='failed',at=time.time(),error=repr(error),no_automatic_retry=True))
        raise
    finally:
        for sig in (signal.SIGINT,signal.SIGTERM):loop.remove_signal_handler(sig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate',type=Path);p.add_argument('--config',type=Path);p.add_argument('--prompt',type=Path)
    p.add_argument('--port',type=int,default=18620);p.add_argument('--run-prepared',action='store_true');a=p.parse_args()
    if a.run_prepared:asyncio.run(run(a))
    else:prepare(a)

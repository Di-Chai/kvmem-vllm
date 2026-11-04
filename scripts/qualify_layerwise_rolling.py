"""Frozen local qualification of per-layer rolling output with K16K/G4K.

Reuses two complete original LongMemEval inputs. Fixed-token long outputs test
mechanics only and never enter benchmark scores. No prior answers are replaced.
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import httpx
import run_agent_handoff_queue as q


REPO = Path(__file__).resolve().parents[1]
CHANGED = ('action_attention.py', 'admission.py', 'scheduler.py', 'async_refresh.py', 'layerwise_pressure.py',
           'layerwise_pressure_efficiency.py', 'layerwise_query.py', 'native_attention.py',
           'rolling_verification.py', 'serve.py', 'session_worker.py')


def prepare(root, source, port, batch_tokens=None, max_num_seqs=2):
    from benchmark_memory import source as read_input
    from transformers import AutoTokenizer
    if root.exists():
        raise ValueError('Use a new version; existing evidence is immutable')
    if q.read(source/'status.json')['status'] != 'completed':
        raise ValueError('Require the completed 16K reference')
    root.mkdir(parents=True)
    runtime = root/'runtime'
    shutil.copytree(source/'runtime-v1', runtime, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    paths = ['src/vllm_kvmem/'+name for name in CHANGED if name != 'serve.py']
    patch = subprocess.check_output(['git', 'diff', 'HEAD', '--', *paths], cwd=REPO)
    (root/'implementation.patch').write_bytes(patch)
    subprocess.run(['patch', '-p1', '--batch', '--forward'], cwd=runtime, input=patch, check=True)
    # The old frozen launcher predates original-RoPE help text. Apply only the
    # two new rolling controls, without importing unrelated launcher changes.
    launcher = runtime/'src/vllm_kvmem/serve.py'
    text = launcher.read_text()
    anchor = "    parser.add_argument('--layerwise-buffered-history', action='store_true')\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor+"    parser.add_argument('--layerwise-rolling-decode', action='store_true')\n")
    current = (REPO/'src/vllm_kvmem/serve.py').read_text()
    anchor = "    'query_replay_budget': None,\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor+"    'request_planning_batch_tokens': None,\n")
    anchor = "        cfg['max_num_batched_tokens'] = replay + cfg['max_num_seqs'] * (cfg['draft_tokens'] + 1)\n"
    begin = current.index('        minimum_batch = replay +')
    end = current.index('    if options.offline_prefill_coalescing:', begin)
    assert text.count(anchor) == 1
    text = text.replace(anchor, current[begin:end])
    begin = current.index('    if args.layerwise_rolling_decode:')
    end = current.index('    if args.layerwise_buffered_history:', begin)
    anchor = '    plan = build_launch(profile, base, run_dir)\n'
    assert text.count(anchor) == 1
    launcher.write_text(text.replace(anchor, anchor+current[begin:end]))
    shutil.copy2(REPO/'src/vllm_kvmem/layerwise_decode.py', runtime/'src/vllm_kvmem/layerwise_decode.py')
    # Keep the reference native binaries and all unrelated engine files.
    (root/'controller').mkdir()
    for name in ('qualify_layerwise_rolling.py', 'run_agent_handoff_queue.py',
                 'benchmark_memory.py', 'benchmark_niah.py', 'benchmark_client_io.py'):
        shutil.copy2(REPO/'scripts'/name, root/'controller'/name)
    fixture = q.read(source/'fixture.json')
    profile = q.read(source/'profile.json')
    assert profile['kvmem']['select_budget'] == 16384 and profile['kvmem']['gen_budget'] == 16384
    profile['kvmem'].update(gen_budget=4096, rolling_active_window=True)
    profile.update(port=port, max_num_seqs=max_num_seqs, verify_session_restore=True, query_replay_budget=4096)
    if batch_tokens is not None: profile['request_planning_batch_tokens']=batch_tokens
    q.save(root/'profile.json', profile)
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    token = tokenizer.encode(' x', add_special_tokens=False)[-1]
    assert token not in tokenizer.all_special_ids
    rows = fixture['rows'][:2]
    for i, row in enumerate(rows):
        q.save(root/f'input-{i}.json', read_input(row)['tokens'])
    flags = ['--'+name.replace('_', '-') for name, enabled in fixture['execution_flags'].items() if enabled]
    q.save(root/'manifest.json', dict(source=str(source), rows=rows, flags=flags,
        long_output_tokens=9216, thinking_token_budget=4096, token=token, eos=tokenizer.eos_token_id,
        scope='Two original full inputs, constrained continuous output: mechanical qualification, not task scores or throughput.',
        files={str(p.relative_to(root)):q.sha(p) for directory in (runtime, root/'controller')
               for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts} |
              {p.name:q.sha(p) for p in (root/'profile.json', root/'input-0.json', root/'input-1.json')}))
    q.save(root/'status.json', dict(status='prepared', at=time.time()))


async def workload(root, profile, manifest):
    rows = []
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{profile["port"]}',
                                trust_env=False, timeout=3600) as client:
        async def owner(i):
            original = manifest['rows'][i]
            prompt = q.read(root/f'input-{i}.json')
            namespace = root.name+'-'+str(i)
            payload = dict(model=profile['served_model_name'], prompt=prompt,
                session_id=namespace, cache_salt=namespace, return_token_ids=True,
                max_tokens=manifest['long_output_tokens'], temperature=0, seed=20260722,
                thinking_token_budget=manifest['thinking_token_budget'], ignore_eos=True,
                allowed_token_ids=[manifest['token']], kvmem_reselect='force',
                kvmem_query_span=[original['query_begin'], original['query_end']],
                stream=True, stream_options=dict(include_usage=True))
            path = root/f'owner-{i}'; path.mkdir()
            q.save(path/'request.json', {k:v for k,v in payload.items() if k!='prompt'})
            ids, identity, usage, finish, done = [], None, None, None, False
            started = time.time()
            with (path/'stream.jsonl').open('x') as log:
                async with client.stream('POST', '/v1/completions', json=payload) as response:
                    if response.status_code != 200:
                        raise RuntimeError((await response.aread()).decode())
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'): continue
                        data = line[5:].strip()
                        if data == '[DONE]': done=True; break
                        item = json.loads(data)
                        log.write(json.dumps(dict(at=time.time(), event=item))+'\n'); log.flush()
                        if 'error' in item: raise RuntimeError(str(item))
                        if item.get('id'):
                            if identity not in (None, item['id']): raise RuntimeError('Response ID changed')
                            identity = item['id']
                        if item.get('usage'): usage=item['usage']
                        for choice in item.get('choices', []):
                            accepted = choice.get('token_ids')
                            if not isinstance(accepted, list): raise RuntimeError('Missing accepted token IDs')
                            ids.extend(accepted)
                            finish = choice.get('finish_reason') or finish
                        q.save(path/'progress.json', dict(id=identity, accepted=len(ids), at=time.time()))
            q.save(path/'accepted.json', ids)
            if not done or not usage or usage['completion_tokens'] != len(ids):
                raise RuntimeError('Incomplete stream or accepted token count mismatch')
            if len(ids) != manifest['long_output_tokens'] or finish != 'length':
                raise RuntimeError('Generation stopped at a physical refresh boundary')
            result = dict(id=identity, owner=i, sample_id=original['id'], prompt_tokens=len(prompt),
                          accepted=len(ids), finish=finish, seconds=time.time()-started, usage=usage)
            q.save(path/'result.json', result); rows.append(result)
        outcomes = await asyncio.gather(*(owner(i) for i in range(2)), return_exceptions=True)
        q.save(root/'responses.json', rows)
        for result in outcomes:
            if isinstance(result, BaseException): raise result
    capture = root/'run/server/capture'
    events = [json.loads(line) for line in (capture/'layerwise_query.jsonl').read_text().splitlines()]
    proofs = [json.loads(line) for line in (capture/'verified_rolling_pressure.jsonl').read_text().splitlines()]
    decode = [e for e in events if e.get('event_type') == 'decode_pressure']
    for row in rows:
        owned = [e for e in decode if e['request_id'].startswith(row['id']+'-')]
        checks = [p for p in proofs if p['request_id'].startswith(row['id']+'-') and
                  p['history_end'] >= row['prompt_tokens']]
        if len(owned) < 2 or len(checks) != len(owned):
            raise RuntimeError('Missing multiple verified rolling refreshes')
        if not all(p['target_kv_byte_equal'] and p['recurrent_byte_equal'] and
                   p['private_forward_steps'] == p['replay_tokens'] == 0 for p in checks):
            raise RuntimeError('Rolling byte or recurrent-state proof failed')
        if not all(len(e['layers']) == 16 and e['fallback_reason'] is None and
                   all(layer['charged_tokens'] <= 16384 and layer['query_samples'] > 0 and
                       layer['accepted_score_end'] <= e['accepted_history'] for layer in e['layers']) for e in owned):
            raise RuntimeError('Independent accepted attention or budget audit failed')
    if not any(e['thinking_at_refresh'] for e in decode) or not any(
            not e['thinking_at_refresh'] for e in decode):
        raise RuntimeError('Qualification must refresh both during and after thinking')
    q.save(root/'execution-audit.json', dict(status='passed', requests=2, gen_budget=4096,
        active_budget=16384, accepted_tokens=sum(r['accepted'] for r in rows),
        decode_refreshes=len(decode), refreshes_during_thinking=sum(bool(e['thinking_at_refresh']) for e in decode),
        refreshes_after_thinking=sum(not e['thinking_at_refresh'] for e in decode),
        byte_oracle=True, same_native_response=True, per_layer_accepted_attention=True))


def run(root):
    manifest = q.verify_frozen(root)
    if q.read(root/'status.json')['status'] != 'prepared': raise ValueError('No implicit retry')
    owned = q.Owned(); success = False
    with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if q.gpu_pids(): raise RuntimeError('GPU has another owner; leave it untouched')
        try:
            profile=q.read(root/'profile.json'); q.require_free_port(profile['port'])
            (root/'run').mkdir()
            command=[sys.executable, '-B', '-m', 'vllm_kvmem', 'serve', '--config', str(root/'profile.json'),
                '--run-dir', str(root/'run/server'), '--layerwise-query-diagnostic', 'sequential',
                '--layerwise-query-optimized', '--layerwise-query-native-batches', '--layerwise-rolling-decode',
                *manifest['flags']]
            env=q.environment(root/'runtime')
            env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
            server=owned.start(command, root/'runtime', env, root/'run/server.log', root/'run/process.json')
            q.save(root/'status.json', dict(status='loading', at=time.time()))
            q.ready_http(f'http://127.0.0.1:{profile["port"]}/health', server, owned,
                         timeout=1200, progress_path=root/'startup.json')
            q.save(root/'status.json', dict(status='running', at=time.time()))
            asyncio.run(workload(root, profile, manifest)); success=True
        except BaseException as error:
            q.save(root/'failure.json', dict(error=repr(error), at=time.time())); raise
        finally:
            cleanup=owned.stop(); q.save(root/'cleanup.json', cleanup)
            success=success and not cleanup['remaining_processes'] and not cleanup['remaining_owned_gpu_pids']
            q.save(root/'status.json', dict(status='passed' if success else 'failed', at=time.time()))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--port', type=int, default=19036)
    parser.add_argument('--batch-tokens', type=int)
    parser.add_argument('--max-num-seqs', type=int, default=2)
    args=parser.parse_args()
    if args.action == 'prepare': prepare(args.root.resolve(), args.source.resolve(), args.port,
                                         args.batch_tokens, args.max_num_seqs)
    else: run(args.root.resolve())

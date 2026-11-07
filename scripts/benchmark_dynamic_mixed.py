"""Finite real-document mixed-length admission experiment, with natural EOS.

Uses complete LongBench documents, official prompts/output caps and frozen token
inputs. This measures scheduling and resource lifetimes, not full task accuracy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

import benchmark_dynamic_memory as bench
from stress_request_swaps import internal_id_matches, outstanding

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_completion(payload, ids, usage, reasons, done):
    if (not done or not ids or any(type(v) is not int or v < 0 for v in ids) or
            not usage or usage.get('prompt_tokens') != len(payload['prompt']) or
            usage.get('completion_tokens') != len(ids) or len(reasons) != 1 or
            reasons[0] not in ('stop', 'length') or len(ids) > payload['max_tokens'] or
            (reasons[0] == 'length' and len(ids) != payload['max_tokens'])):
        raise ValueError('Incomplete natural stream or inconsistent public accounting')


def natural_request(url, payload, barrier):
    request = urllib.request.Request(url + '/v1/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
        'X-Request-Id': payload['request_id']})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    barrier.wait(timeout=60)
    start = time.perf_counter()
    ids, texts, reasons, bursts = [], [], [], []
    usage, first, done = None, None, False
    result = dict(started=start, input_digest=bench.input_digest(payload), request_id=payload['request_id'])
    try:
        with opener.open(request, timeout=1200) as response:
            for line in response:
                if not line.startswith(b'data:'): continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                packet = json.loads(data)
                if packet.get('error'): raise ValueError(str(packet['error']))
                if packet['id'] != 'cmpl-' + payload['request_id']:
                    raise ValueError('Response identity changed')
                usage = packet.get('usage') or usage
                for choice in packet['choices']:
                    if choice['index'] != 0: raise ValueError('Unexpected completion choice')
                    tokens = choice.get('token_ids') or []
                    if tokens:
                        at = time.perf_counter() - start
                        if first is None: first = at
                        bursts.append(dict(seconds=at, tokens=len(tokens)))
                    ids.extend(tokens)
                    texts.append(choice.get('text') or '')
                    if choice.get('finish_reason') is not None: reasons.append(choice['finish_reason'])
        validate_completion(payload, ids, usage, reasons, done)
        result['status'] = 'completed'
    except Exception as error:
        result.update(status='failed', error=repr(error))
    finally:
        result.update(seconds=time.perf_counter()-start, ttft_seconds=first,
            output_ids=ids, output_text=''.join(texts), usage=usage,
            stream_done=done, finish_reasons=reasons, bursts=bursts)
    return result


class DetailedSampler(bench.PhaseSampler):
    def __init__(self, capture, output):
        super().__init__(capture, interval=.05, max_samples=20000)
        self.output = Path(output).open('x')
        self.last_at = None

    def observe(self, scheduler, archive):
        super().observe(scheduler, archive)
        if scheduler.get('at') is not None and scheduler['at'] != self.last_at:
            self.last_at = scheduler['at']
            self.output.write(json.dumps(dict(observed_at=time.perf_counter(),
                scheduler=scheduler)) + '\n')
            self.output.flush()

    def stop(self):
        try:
            return super().stop()
        finally:
            self.output.close()


def prepare(root, profile_path):
    from transformers import AutoTokenizer
    if (root/'fixture.json').exists(): raise ValueError('Preserve an existing frozen workload')
    data = root/'data'
    tokenizer = AutoTokenizer.from_pretrained(read(profile_path)['tokenizer'], local_files_only=True)
    templates, caps = read(data/'dataset2prompt.json'), read(data/'dataset2maxlen.json')
    selected = {}
    with zipfile.ZipFile(data/'longbench-data.zip') as archive:
        for dataset in ('qasper', 'hotpotqa', 'gov_report', 'multi_news'):
            rows = [json.loads(line) for line in archive.read('data/'+dataset+'.jsonl').decode().splitlines()]
            chosen = []
            low, high = (1024, 4096) if dataset in ('qasper', 'hotpotqa') else (5000, 12000)
            for index, row in enumerate(rows):
                text = templates[dataset].format(**row)
                ids = tokenizer.apply_chat_template([dict(role='user', content=text)],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
                if not isinstance(ids,list) or any(type(v) is not int for v in ids):
                    raise ValueError('Tokenizer did not return a flat token-ID list')
                if not low <= len(ids) <= high: continue
                # Preserve the complete document. Locate the trailing query by
                # common prefix, including possible tokenizer boundary merging.
                prefix = templates[dataset].split('{context}')[0] + row['context']
                prefix_ids = tokenizer.apply_chat_template([dict(role='user', content=prefix)],
                    tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False)
                common = next((i for i, (a, b) in enumerate(zip(ids, prefix_ids)) if a != b),
                              min(len(ids), len(prefix_ids)))
                begin = common // 32 * 32
                if not 0 < len(ids)-begin <= 2048: raise ValueError('Query suffix exceeds unchanged G')
                chosen.append(dict(dataset=dataset, row_index=index, source_id=row['_id'],
                    input_tokens=len(ids), max_tokens=caps[dataset], query_span=[begin, len(ids)],
                    prompt=ids, answers=row['answers'], source_row_sha256=hashlib.sha256(
                        json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()))
                if len(chosen) == 4: break
            if len(chosen) != 4: raise ValueError('Not enough complete documents in length band: '+dataset)
            selected[dataset] = chosen
    rows = [selected[name][i] for i in range(4) for name in selected]
    for i, row in enumerate(rows): row['owner'] = i
    bench.save(root/'fixture.json', dict(rows=rows, selection='First four rows per dataset in fixed input-length bands; no response/quality selection.',
        generation='Natural EOS, official per-task max_tokens; length finishes are censored.',
        chat_template='Local tokenizer user template, enable_thinking=False; no document cropping/padding.',
        scope='Real-document capacity replay; not a complete LongBench accuracy benchmark.'))
    profile = read(profile_path)
    profile.update(max_num_seqs=16, max_num_batched_tokens=2048,
                   verify_request_swap=False, verify_session_restore=False, verify_session_restore_digest=False)
    profile['kvmem'].update(archive_backend='tiered', nvme_max_bytes=8*1024**3,
        nvme_preallocate=False, gpu_memory_admission='incremental',
        request_checkpoint_bytes=512*1024**2, request_checkpoint_ssd_bytes=8*1024**3)
    if (profile['kvmem']['select_budget'], profile['kvmem']['gen_budget']) != (2048, 2048):
        raise ValueError('This workload qualification retains K/G=2048/2048')
    if any(r['input_tokens']+r['max_tokens'] > profile['max_model_len'] for r in rows):
        raise ValueError('Full input does not fit model context')
    bench.save(root/'profile.json', profile)
    runtime = root/'runtime'
    shutil.copytree(ROOT/'src', runtime/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (runtime/'scripts').mkdir()
    for name in ('benchmark_dynamic_mixed.py', 'benchmark_dynamic_memory.py', 'stress_request_swaps.py'):
        shutil.copy2(ROOT/'scripts'/name, runtime/'scripts'/name)
    bench.save(root/'source.json', {str(p.relative_to(runtime)):sha(p) for p in runtime.rglob('*') if p.is_file()})
    bench.save(root/'input-proof.json', {name:sha(root/name) for name in ('fixture.json','profile.json','data/sources.json')})
    print(json.dumps([dict(dataset=r['dataset'], row=r['row_index'], input=r['input_tokens'], cap=r['max_tokens']) for r in rows]), flush=True)


def analyze_timeline(timeline, responses):
    changes, transitions, seen = [], [], set()
    last = None
    start = min(r['started'] for r in responses)
    for item in timeline:
        snapshot, at = item['scheduler'], item['scheduler']['at']
        if at < start: continue
        owners = snapshot['gpu_capacity']['requests']
        resident = sum(v['owned_pages'] > 0 for v in owners.values())
        swaps = snapshot['request_swaps']
        paused = sum(v['phase'] == 'paused' for v in swaps['requests'].values())
        finished = sum(r['started']+r['seconds'] <= at and r['status']=='completed' for r in responses)
        key = (resident, paused, finished)
        if key != last:
            changes.append(dict(seconds=at-start, resident=resident, paused=paused,
                completed=finished, pending_http=len(responses)-finished,
                owned_pages=snapshot['owned_blocks'], free_pages=snapshot['free_blocks']))
            last = key
        for row in swaps['recent_transitions']:
            identity = (row['step'],row['request_id'],row['action'],row.get('history_end'))
            if identity not in seen:
                seen.add(identity)
                transitions.append(row)
    return dict(changes=changes, transitions=transitions,
        pressure_while_more_than_eight_pending=[r for r in changes if r['paused'] > 0 and r['pending_http'] > 8],
        note='Scheduler publication timestamps are sampled observations; HTTP completion and retirement ACK differ. Residency is not a GPU forward batch size.')


def drain(capture, timeout=180):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        final = dict(scheduler=read(capture/'scheduler_resources.json'),archive=read(capture/'archive.json'))
        worker = read(capture/'archive_resources.json')['request_checkpoints']
        history_busy = any(final['archive'].get(k,0) for k in ('active_requests','retiring_requests',
            'pending_downloads','logical_allocated_bytes','ssd_used_bytes','ssd_file_bytes',
            'ssd_retired_pending_bytes','archive_io_pending_bytes','archive_used_slots'))
        checkpoint_busy = worker['resources']['active'] or any(worker['storage'][k] for k in
            ('cpu_resident_bytes','cpu_promised_bytes','ssd_used_bytes','ssd_pending_write_bytes','logical_bytes','entries','io_pending'))
        if not outstanding(final['scheduler']) and not history_busy and not checkpoint_busy:
            return final,worker
        time.sleep(.05)
    raise TimeoutError('Native pages, history or checkpoint physical ownership did not drain')


def run_arm(root, name, gpu_bytes, port, verify):
    if any(sha(root/key) != value for key,value in read(root/'input-proof.json').items()):
        raise ValueError('Frozen workload changed')
    runtime = root/'runtime'
    if any(sha(runtime/key) != value for key,value in read(root/'source.json').items()):
        raise ValueError('Frozen runtime changed')
    arm = root/name
    arm.mkdir(exist_ok=False)
    profile = read(root/'profile.json')
    profile.update(port=port, kv_cache_memory_bytes=gpu_bytes, verify_request_swap=verify)
    bench.save(arm/'profile.json', profile)
    fixture = read(root/'fixture.json')
    payloads = [dict(model=profile['served_model_name'], prompt=r['prompt'],
        kvmem_query_span=r['query_span'], temperature=0, seed=104729, max_tokens=r['max_tokens'],
        ignore_eos=False, return_token_ids=True, stream=True, stream_options={'include_usage':True},
        request_id=f'mixed16-{name}-{r["owner"]:02d}') for r in fixture['rows']]
    bench.save(arm/'requests.json', payloads)
    env = dict(os.environ, PYTHONPATH=str(runtime/'src'), PYTHONDONTWRITEBYTECODE='1',
               MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', KVMEM_LIFECYCLE_TRACE='1')
    command = [sys.executable,'-B','-m','vllm_kvmem','serve','--config',str(arm/'profile.json'),
               '--run-dir',str(arm/'server')]
    bench.save(arm/'command.json', command)
    report = dict(status='starting', diagnostics=verify, responses=[None]*len(payloads))
    server, sampler = None, None
    base = f'http://127.0.0.1:{port}'
    try:
        with (arm/'server.log').open('w') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        bench.wait_ready(server,base)
        warm = natural_request(base, {**payloads[0], 'request_id':f'mixed16-{name}-warmup', 'max_tokens':16},threading.Barrier(1))
        bench.save(arm/'warmup.json', warm)
        if warm['status'] != 'completed': raise RuntimeError('Warmup failed')
        bench.settle(arm/'server/capture')
        print(json.dumps(dict(arm=name, status='running', requests=len(payloads))),flush=True)
        sampler = DetailedSampler(arm/'server/capture', arm/'timeline.jsonl')
        sampler.start()
        report['status'] = 'running'
        barrier = threading.Barrier(len(payloads))
        with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
            futures = {pool.submit(natural_request,base,body,barrier):i for i,body in enumerate(payloads)}
            for future in as_completed(futures):
                i = futures[future]
                response = future.result()
                response['owner'] = i
                response['dataset'] = fixture['rows'][i]['dataset']
                report['responses'][i] = response
                bench.save(arm/f'response-{i:02d}.json', response)
                bench.save(arm/'results.json', report)
                print(json.dumps(dict(arm=name,owner=i,status=response['status'],dataset=response['dataset'],
                    tokens=len(response['output_ids']),seconds=response['seconds'],finish=response['finish_reasons'])),flush=True)
        if any(r['status'] != 'completed' for r in report['responses']): raise RuntimeError('Public requests failed')
        final,worker = drain(arm/'server/capture')
        if verify and (not worker['verification_enabled'] or worker['verification_failures'] or
                       worker['verification_successes'] != worker['restores']):
            raise RuntimeError('Restore byte verification incomplete')
        report.update(status='completed',summary=bench.summarize(report['responses']),settled=final,worker=worker)
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        try:
            if sampler:
                report['resources'] = sampler.stop()
            bench.save(arm/'results.json',report)
        finally:
            if server:
                if server.poll() is None: os.killpg(server.pid,signal.SIGTERM)
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)
    timeline = [json.loads(line) for line in (arm/'timeline.jsonl').read_text().splitlines()]
    detail = analyze_timeline(timeline,report['responses'])
    steps = [json.loads(line) for line in (arm/'server/capture/model_step_latency.jsonl').read_text().splitlines()]
    measured = [s for s in steps if any(any(internal_id_matches(r['request_id'],g['request_id'])
                 for r in report['responses']) for g in s['batch_geometry'])]
    detail['public_batch_peak'] = max(len(s['batch_geometry']) for s in measured)
    detail['public_batches'] = len(measured)
    detail['per_request_swaps'] = {r['request_id']:{kind:sum(t['action']==kind and
        internal_id_matches(r['request_id'],t['request_id']) for t in detail['transitions'])
        for kind in ('capture','restore','drop')} for r in report['responses']}
    bench.save(arm/'timeline-analysis.json',detail)
    print(json.dumps(dict(arm=name,status=report['status'],summary=report['summary'],
        resident_peak=report['resources']['observed_phase_peaks']['simultaneous_gpu_owners'],
        public_batch_peak=detail['public_batch_peak'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--arm')
    parser.add_argument('--gpu-bytes',type=int)
    parser.add_argument('--port',type=int,default=19218)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(args.root.resolve(),args.profile)
    else: run_arm(args.root.resolve(),args.arm,args.gpu_bytes,args.port,args.verify)

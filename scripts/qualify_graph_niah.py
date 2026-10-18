"""Frozen NIAH quality client for Query, cumulative attention and block graphs.

Only original prompt IDs, public Query coordinates and shared sampling settings
are sent to the model. Gold answers and needle spans are used offline only.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from types import SimpleNamespace
import urllib.error
import urllib.request
import uuid

import numpy as np
from tokenizers import Tokenizer

from vllm_kvmem.offline_queue import OfflineQueue, OfflineTask

import qualify_block_graph as q


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def validate_profile(profile, arm):
    kv = profile['kvmem']
    q.require(arm in ('baseline', 'legacy', 'multi'), 'Unknown NIAH arm')
    q.require(profile['execution_mode'] == 'compiled_graph' and profile['draft_method'] == 'mtp' and
              profile['draft_tokens'] == 4, 'Keep compiled native MTP(k=4)')
    q.require(profile['max_num_seqs'] == 8 and profile['max_num_batched_tokens'] == 16424 and
              profile['query_replay_budget'] == 16384 and profile['prefill_chunk_tokens'] == 1024,
              'Match historical batch 16424 / Query 16384 / chunk 1024 / eight-owner limits')
    q.require(kv['mean_k_backend'] == 'qw3_cuda' and kv['archive_backend'] == 'cpu' and
              kv['request_planning'] and kv['session_capacity'] >= 8, 'Require owned CPU-history MeanK sessions')
    q.require(kv['session_checkpoint_bytes'] == 1 and not kv['session_capture_decode'],
              'Independent offline cases require non-retaining checkpoints')
    q.require(not kv['query_history'] and not kv['action_pressure'] and kv['guided_query_tokens'] == 0 and
              not kv.get('repository_gc', False), 'History/guided/pressure/GC must stay off in all three arms')
    q.require(kv['graph_mode'] == ('multi' if arm == 'multi' else 'off'), 'Graph mode does not match arm')
    q.require(kv['action_fraction'] == (.5 if arm == 'legacy' else 0.), 'Only legacy gets cumulative attention quota')
    q.require(kv['action_attention'] and kv['prefill_attention'] and kv['action_scope'] == 'non-thinking',
              'Keep identical attention capture in all arms')
    q.require(not kv.get('graph_trace', False), 'Detailed graph tracing is excluded from this campaign')
    q.require(kv['block_tokens'] == 32 and kv['select_budget'] == 36864 and kv['gen_budget'] == 16384 and
              kv['sink_blocks'] == 41 and kv['recent_blocks'] == 0, 'Frozen K/G/sink/recent changed')


def audit_launch(profile, capture):
    """Reject a silently normalized launch before submitting any model input."""
    launch = json.loads((capture.parent/'launch.json').read_text())
    argv, environment = launch['vllm_argv'], launch['environment']
    expected = {'--max-num-batched-tokens':str(profile['max_num_batched_tokens']),
                '--max-num-seqs':str(profile['max_num_seqs']),
                '--kv-cache-dtype':'fp8_e4m3'}
    observed = {}
    for name,value in expected.items():
        q.require(argv.count(name) == 1 and argv.index(name)+1 < len(argv),
                  'Missing or ambiguous native launch flag: '+name)
        observed[name] = argv[argv.index(name)+1]
        q.require(observed[name] == value, 'Effective native launch differs: '+name)
    q.require(environment.get('KVMEM_PREFILL_CHUNK_TOKENS') == '1024',
              'Effective per-request prefill chunk differs')
    plan = json.loads(environment['KVMEM_REQUEST_PLAN_CONFIG'])
    q.require(plan['query_replay_budget'] == 16384 and plan['gen_budget'] == 16384 and
              plan['select_budget'] == 36864, 'Effective Query/K/G limits differ')
    q.require(environment.get('KVMEM_FULL_GRAPH') == '1' and '--enforce-eager' not in argv,
              'Native compiled/CUDA Graph execution changed')
    spec_flag = '--speculative-config'
    q.require(argv.count(spec_flag) == 1 and argv.index(spec_flag)+1 < len(argv),
              'Missing or ambiguous native speculation configuration')
    speculative = json.loads(argv[argv.index(spec_flag)+1])
    q.require(speculative['method'] == 'mtp' and speculative['num_speculative_tokens'] == 4,
              'Effective native MTP(k=4) differs')
    return dict(status='passed',scope='Actual saved launch argv/environment before the first case.',
                expected=expected,observed=observed,query_replay_budget=plan['query_replay_budget'],
                prefill_chunk_tokens=int(environment['KVMEM_PREFILL_CHUNK_TOKENS']))


def stream(profile, session, tokens, query, sampling, timeout):
    body = dict(sampling, model=profile['served_model_name'], session_id=session, cache_salt=session,
                prompt=tokens, return_token_ids=True, stream=True, stream_options={'include_usage':True},
                kvmem_query_span=query, kvmem_reselect='force')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
        data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    begin = time.monotonic()
    row = dict(output_ids=[], chunks=[], text='', response_id=None, usage=None, finish_reason=None)
    done = False
    try:
        with opener.open(request, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                line = line[5:].strip()
                if line == b'[DONE]':
                    done = True
                    break
                event = json.loads(line)
                q.require(not event.get('error'), 'Native stream error: '+str(event.get('error')))
                if event.get('id'):
                    q.require(row['response_id'] in (None,event['id']), 'Stream response identity changed')
                    row['response_id'] = event['id']
                row['usage'] = event.get('usage') or row['usage']
                for choice in event.get('choices', []):
                    q.require(choice.get('index',0) == 0, 'Multiple choices are unsupported')
                    ids = choice.get('token_ids') or []
                    if ids:
                        row['output_ids'].extend(ids)
                        row['chunks'].append([time.monotonic()-begin,len(ids)])
                    row['text'] += choice.get('text') or ''
                    row['finish_reason'] = choice.get('finish_reason') or row['finish_reason']
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from None
    row['seconds'] = time.monotonic()-begin
    q.require(done and row['response_id'] and row['usage'] and row['output_ids'], 'Incomplete output stream')
    q.require(row['usage']['prompt_tokens'] == len(tokens) and
              row['usage']['completion_tokens'] == len(row['output_ids']) <= sampling['max_tokens'],
              'Prompt or accepted output accounting mismatch')
    q.require(row['finish_reason'] in ('stop','length'), 'Unexpected finish reason')
    row['ttft_seconds'] = row['chunks'][0][0]
    return row


def wait_retired(capture, response_id, timeout=180):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        path = capture/'archive.json'
        if path.exists():
            try:
                archive = json.loads(path.read_text())
            except json.JSONDecodeError:
                time.sleep(.05)
                continue
            row = next((r for r in archive.get('completed', []) if q.native_matches(
                {'response_id':response_id},r) and r.get('retirement_finished_at') and
                r.get('retirement_ack_ready_at')),None)
            if row is not None:
                return row
        time.sleep(.1)
    raise TimeoutError('Missing fenced CPU history retirement')


def audit_selection(capture, response, row, profile, arm, tokens, tokenizer):
    match = lambda event:q.native_matches(response,event)
    queries = [r for r in q.records(capture/'bounded_refresh.jsonl') if match(r) and r['selection'] == 'query']
    q.require(len(queries) == 1, 'Missing or duplicate public Query selection')
    selected = queries[0]
    q.require(selected['query_span'] == row['served_query'] and selected['mean_k_backend'] == 'qw3_cuda' and
              not selected['query_history_enabled'] and selected['guided_query_tokens'] == 0,
              'Query coordinates/backend/strategy differs from the frozen protocol')
    q.require(selected['selected_tokens'] <= profile['kvmem']['select_budget'], 'Selection budget exceeded')
    q.require(selected['action_attention_enabled'] and selected['action_history_pages'] == 0,
              'Cold session inherited output attention or capture was disabled')
    actions = [r for r in q.records(capture/'action_attention.jsonl') if match(r)]
    q.require(len(actions) == 1, 'Missing terminal attention acceptance evidence')
    action = actions[0]
    parser_tok = SimpleNamespace(convert_tokens_to_ids=tokenizer.token_to_id)
    expected = q.expected_output_samples(parser_tok,profile,tokens,response['output_ids'])
    q.require(action['samples'] == expected, 'Accepted non-thinking parser sample count differs')
    q.require(selected.get('prefill_attention_samples',0) > 0 and
              action['prefill_samples'] == selected['prefill_attention_samples'],
              'Missing prefill observation or Query replay polluted attention samples')
    if arm == 'legacy':
        q.require(selected['attention_source'] == 'prefill', 'Legacy did not use historical prefill attention')
    graph = [r for r in q.records(capture/'block_graph.jsonl') if match(r)]
    if arm == 'multi':
        q.require(len(graph) == 1 and graph[0]['mode'] == 'multi' and
                  graph[0]['observed_sources'] > 0 and graph[0]['edges'] > 0,
                  'Graph scoring was not executed with observed attention edges')
        replay_begin,replay_end = selected['replay_span']
        qb,qe = row['served_query']
        q.require(replay_begin == qb//profile['kvmem']['block_tokens']*profile['kvmem']['block_tokens'] and
                  replay_begin <= qb < qe <= replay_end, 'Unexpected public Query replay alignment')
        q.require(graph[0]['accepted_through'] >= replay_begin, 'Graph accepted watermark is stale')
    else:
        q.require(not graph, 'Graph scoring leaked into a baseline arm')
    needles = [{**n,'selected':covered(n['start'],n['end'],selected['selected_history_ranges'])} for n in row['needles']]
    return dict(query=selected,action=action,graph=graph,needles=needles,
                all_needles_selected=all(n['selected'] for n in needles))


def covered(begin,end,ranges):
    cursor = begin
    for start,count in sorted(ranges):
        if start > cursor:
            break
        cursor = max(cursor,start+count)
        if cursor >= end:
            return True
    return False


def owner_released(state, session, response, terminal):
    """A per-owner scheduler ACK, never a global all-requests-idle fence."""
    owners = state.get('session_owners')
    leases = (state.get('cpu_history_admission') or {}).get('lease_views')
    reservations = state.get('reservations')
    if not all(isinstance(value, list) for value in (owners, leases, reservations)):
        return False
    if state.get('at', 0) < terminal.get('retirement_ack_ready_at', float('inf')):
        return False
    namespace = [session, session]  # This client uses its owner as cache salt.
    if any(row.get('namespace') == namespace for row in owners):
        return False
    if any(row.get('key') == ['session', *namespace] or q.native_matches(
            response, {'request_id':row.get('request_id') or ''}) for row in leases):
        return False
    return not any(q.native_matches(response, {'request_id':rid}) for rid in reservations)


def wait_owner_released(capture, session, response, terminal, timeout=180):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        try:
            state = json.loads((capture/'scheduler_resources.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.1)
            continue
        if owner_released(state,session,response,terminal):
            return dict(method='ordinary_retirement', namespace=[session,session],
                        released=True, retirement_ack_ready_at=terminal['retirement_ack_ready_at'],
                        scheduler_observed_at=state['at'], other_requests=state.get('requests'))
        time.sleep(.1)
    raise TimeoutError('This owner did not release its namespace, CPU history and GPU reservation')


def history_bytes(row, fixture, profile):
    extent = row['served_input_tokens']+fixture['sampling']['max_tokens']+profile['kvmem']['gen_budget']
    return ((extent+31)//32)*32*17*4*256*2


def infer_case(args,fixture,profile,row,tokenizer):
    directory = args.output/'cases'/row['id']
    directory.mkdir(parents=True,exist_ok=False)
    source = Path(fixture['input_directory'])/(row['id']+'.i32')
    q.require(sha(source) == row['served_sha256'], 'Frozen prompt bytes changed')
    tokens = np.fromfile(source,dtype='<i4').tolist()
    q.require(len(tokens) == row['served_input_tokens'] and
              tokenizer.decode(tokens[slice(*row['served_query'])],skip_special_tokens=False) == row['question'],
              'Frozen Query span does not decode to the original question')
    session = 'graph-niah-'+uuid.uuid4().hex
    result = dict(id=row['id'],task=row['task'],seed=row['seed'],position_percent=row['position_percent'],
                  input_tokens=len(tokens),input_sha256=row['served_sha256'],arm=args.arm,
                  cpu_reserved_bytes=history_bytes(row,fixture,profile),session=session)
    print(json.dumps(dict(event='start',id=row['id'],arm=args.arm)),flush=True)
    sampling = {**fixture['sampling'],'seed':row['seed']}
    save(directory/'request.json',dict(sampling,session_id=session,prompt_sha256=row['served_sha256'],
         prompt_tokens=len(tokens),kvmem_query_span=row['served_query']))
    response = stream(profile,session,tokens,row['served_query'],sampling,fixture['request_timeout_seconds'])
    save(directory/'response.json',response)
    return result,response,tokens


def finish_case(args,profile,row,scorer,tokenizer,pending):
    result,response,tokens = pending
    directory = args.output/'cases'/row['id']
    terminal = wait_retired(args.capture,response['response_id'])
    q.require(terminal['history_end'] == len(tokens)+len(response['output_ids'])-1,
              'Full logical history does not match accepted input/output')
    q.require(not terminal.get('checkpoint_tokens'), 'Offline case retained a cross-turn checkpoint')
    release = wait_owner_released(args.capture,result['session'],response,terminal)
    save(directory/'release.json',release)
    evidence = audit_selection(args.capture,response,row,profile,args.arm,tokens,tokenizer)
    save(directory/'evidence.json',dict(evidence,terminal=terminal))
    result.update(status='completed',evaluation=scorer(response['text'],row['expected']),
                  completion_tokens=len(response['output_ids']),seconds=response['seconds'],
                  ttft_seconds=response['ttft_seconds'],all_needles_selected=evidence['all_needles_selected'],
                  graph_added_blocks=sum(r['added_blocks'] for r in evidence['graph']))
    save(directory/'result.json',result)
    print(json.dumps(dict(event='finished',id=row['id'],arm=args.arm,
        coverage=result['evaluation']['normalized_coverage'],exact=result['evaluation']['normalized_exact_numeric_set'])),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','capture','output','fixture'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=('baseline','legacy','multi'),required=True)
    args = parser.parse_args()
    profile,fixture = json.loads(args.config.read_text()),json.loads(args.fixture.read_text())
    validate_profile(profile,args.arm)
    q.require(not args.output.exists(),'Use a fresh client directory')
    args.output.mkdir(parents=True)
    save(args.output/'launch-audit.json',audit_launch(profile,args.capture))
    q.require(sha(Path(profile['tokenizer'])/'tokenizer.json') == fixture['tokenizer_sha256'],'Tokenizer changed')
    tree = ast.parse(fixture['scorer_source'])
    q.require(len(tree.body) == 1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name == 'score',
              'Expected the frozen original score function')
    scope = {'re':re}
    exec(compile(tree,'frozen-original-niah-scorer','exec'),scope)
    tokenizer = Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    queue = OfflineQueue([OfflineTask(row['id'],row,history_bytes(row,fixture,profile))
                          for row in fixture['rows']], max_inflight=fixture['workers'],
                         history_capacity_bytes=fixture['cpu_payload_credits_bytes'])
    observed = []
    def on_result(result):
        state = queue.states[result['id']]
        result.update(cpu_queue_seconds=state['admitted_at']-state['queued_at'],
                      retirement_seconds=state['completed_at']-state['inference_done_at'])
        save(args.output/'cases'/result['id']/'result.json',result)
        observed.append(result)
        save(args.output/'progress.json',dict(completed=len(observed),expected=len(fixture['rows']),
             coverage=statistics.fmean(r['evaluation']['normalized_coverage'] for r in observed),
             exact=sum(r['evaluation']['normalized_exact_numeric_set'] for r in observed)))
    def on_failure(error):
        save(args.output/'client-failure.json',dict(error=type(error).__name__+': '+str(error)))
    def on_state(state):
        save(args.output/'queue.json',state)
        with (args.output/'queue_events.jsonl').open('a') as output:
            output.write(json.dumps(dict(at=time.monotonic(),**{k:v for k,v in state.items() if k!='tasks'}))+'\n')
    def work(task):
        return infer_case(args,fixture,profile,task.payload,tokenizer)
    def finish(task,pending):
        return finish_case(args,profile,task.payload,scope['score'],tokenizer,pending)
    begin = time.monotonic()
    results = queue.run(work,finish,on_result=on_result,on_failure=on_failure,on_state=on_state)
    released = q.wait_released(args.capture,[r['session'] for r in results],timeout=180)
    archive = released['archive']
    q.require(archive.get('disk_read_bytes',0) == archive.get('disk_write_bytes',0) == 0 and
              archive.get('nvme_capacity_bytes',0) == 0,'History used disk storage')
    elapsed = time.monotonic()-begin
    owners = list(queue.states.values())
    refills = sum(any(other['admitted_at'] <= owner['admitted_at'] < other['completed_at']
                      for other in owners[:index])
                  for index,owner in enumerate(owners) if index >= fixture['workers'])
    q.require(queue.peak_inflight == fixture['workers'] and queue.reserved_bytes == queue.inflight == 0,
              'Missing eight-owner admission or leaked offline credits')
    result = dict(status='passed',protocol_passed=True,quality_improvement_established=False,
                  all_owned_sessions_released=True,arm=args.arm,fixture_sha256=sha(args.fixture),
                  cases=len(results),wall_seconds=elapsed,cpu_admission_peak=queue.peak_inflight,
                  cpu_admission_peak_bytes=queue.peak_bytes,queue=queue.snapshot(),
                  refills_with_older_jobs_pending=refills,
                  macro_coverage=statistics.fmean(r['evaluation']['normalized_coverage'] for r in results),
                  exact=sum(r['evaluation']['normalized_exact_numeric_set'] for r in results),
                  graph_changed_queries=sum(r['graph_added_blocks'] > 0 for r in results),
                  output_tokens_per_wall_s=sum(r['completion_tokens'] for r in results)/elapsed,
                  throughput_scope='Whole NIAH campaign with long prefill and short answers, not pure decode.',
                  admission_protocol='Continuous FIFO; at most eight owned jobs; per-owner ordinary retirement before refill.',
                  results=sorted(results,key=lambda r:r['id']),final_archive=archive,
                  step_summary=q.summarize_steps(args.capture,begin,time.monotonic()))
    q.require(len(results) == len(fixture['rows']), 'Incomplete campaign')
    save(args.output/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','final_archive')}),flush=True)


if __name__ == '__main__':
    main()

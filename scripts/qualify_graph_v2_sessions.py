"""Real structured V2 session lifecycle client; never starts a model server.

Use an owned running server with verify_session_restore=true. Long histories
come from Protocol-rendered tool results. Returned assistant IDs are retained
verbatim; live tool results use the actual response tool-use IDs. Tool bodies
are deterministic harness fixtures, never execution of model-proposed commands.
This is a lifecycle and boundary smoke test, not a throughput/quality benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


q = module('graph_v2_session_shared', ROOT/'scripts/qualify_block_graph.py')


def verify_plan(prepared, profile):
    """Adding locality evidence must preserve all existing planner decisions."""
    from vllm_kvmem.request_plan import RequestPlan
    tokens, plan = prepared[:2]
    kv = profile['kvmem']
    baseline = {k:v for k,v in plan.items() if k not in ('kvmem_locality_spans','kvmem_locality_provenance')}
    old, new = [RequestPlan.parse(p,len(tokens),kv['block_tokens'],kv['gen_budget']) for p in (baseline,plan)]
    for field in ('query','replay','mandatory','explicit_mandatory'):
        q.require(getattr(old,field) == getattr(new,field), 'Locality metadata changed ' + field)
    q.require(plan.get('kvmem_locality_spans'), 'Structured producer emitted no message evidence')
    q.require(plan['kvmem_locality_provenance']['source'] == 'deepswe_protocol',
              'Missing authoritative structured producer provenance')
    q.require(len(tokens) + 257 < profile['max_model_len'], 'Unclipped complete prompt exceeds model context')
    return dict(token_ids_sha256=q.digest(tokens), query_replay_mandatory_unchanged=True,
                locality_spans=plan['kvmem_locality_spans'], provenance=plan['kvmem_locality_provenance'])


def payload_for(owner, body):
    return dict(model='graph-v2-session-smoke', max_tokens=257,
        system='You are a protocol validation assistant. Follow the current user request. '
               'Use the available InspectArchive tool when asked. Keep tool arguments short. '
               'Do not reason aloud; use tools directly.',
        tools=[dict(name='InspectArchive',description='Inspect a named frozen archive record.',
                    input_schema=dict(type='object',properties=dict(record=dict(type='string')),
                                      required=['record']))],
        messages=[dict(role='user',content=f'Open the immutable archive for owner {owner}.'),
            dict(role='assistant',content=[dict(type='tool_use',id=f'fixture-open-{owner}',
                                               name='InspectArchive',input=dict(record='initial-fixture'))]),
            dict(role='user',content=[dict(type='tool_result',tool_use_id=f'fixture-open-{owner}',content=body),
                dict(type='text',text='The archive is now loaded. Call InspectArchive twice, for records '
                     'primary and secondary. Return only these tool calls and stop; no explanation.')])])


def build_initial(planner, bridge, profile, owner, target):
    unit = ('Immutable archive entry: historical identity is stable; selected pages keep their logical '
            'positions, and tool results belong to the owner who requested them.\n')
    unit_tokens = len(planner.encode(unit))
    repeats = max(1, target // unit_tokens)
    for _ in range(4):
        payload = payload_for(owner, f'Frozen fixture for owner {owner}.\n' + unit * repeats)
        prepared = planner.prepare(payload, bridge.Trajectory())
        delta = target - len(prepared[0])
        if abs(delta) <= unit_tokens:
            break
        repeats = max(1, repeats + round(delta / unit_tokens))
    q.require(prepared[0] == planner.encode(prepared[2]['prompt']),
              'Structured producer changed original full production token encoding')
    kv = profile['kvmem']
    q.require(len(prepared[0]) > kv['select_budget'] + 3*kv['gen_budget'],
              'Initial structured history is too short for multiple pressure events')
    verify_plan(prepared,profile)
    return payload, prepared


def call(profile, bridge, session, prepared, count, timeout):
    tokens, plan = prepared[:2]
    body = bridge.backend_request(model=profile['served_model_name'],tokens=tokens,
        session=session,max_tokens=count,plan=plan,backend_kind='kvmem')
    # A declared smoke setting keeps real tool calls inside a short quota.
    # The authoritative Protocol template and all production prompt IDs remain unchanged.
    body.update(thinking_token_budget=0, temperature=0, min_tokens=0, stream=False)
    request = urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    try:
        with opener.open(request,timeout=timeout) as response:
            native = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from None
    choice, = native['choices']
    output = choice.get('token_ids')
    q.require(isinstance(output,list) and output and all(type(t) is int and t >= 0 for t in output),
              'Server did not return the actual accepted output token IDs')
    q.require(native['usage']['completion_tokens'] == len(output) <= count,
              'Accepted output count differs from the public response')
    q.require(native['usage']['prompt_tokens'] == len(tokens), 'Public prompt token count changed')
    return native, dict(response_id=native['id'], input_tokens=len(tokens), output_ids=output,
        query_span=plan.get('kvmem_query_span'), input_sha256=q.digest(tokens),
        finish_reason=choice['finish_reason'], seconds=time.monotonic()-started,
        exact_output_prefix_reused=prepared[5], text=choice['text'])


def next_payload(payload, response):
    calls = [part for part in response['content'] if part['type'] == 'tool_use']
    q.require(len(calls) >= 2 and len({c['id'] for c in calls}) == len(calls) and
              all(c['name'] == 'InspectArchive' for c in calls),
              'Model did not return complete real InspectArchive tool calls with distinct IDs within the smoke quota')
    result = copy.deepcopy(payload)
    result['messages'] += [dict(role='assistant',content=response['content']),
        dict(role='user',content=[dict(type='tool_result',tool_use_id=c['id'],
            content=f"Frozen result for {c['input'].get('record','unspecified')}: 历史记录🙂 unchanged. "
                    'The owner and generation checks succeeded.') for c in calls] +
            [dict(type='text',text='Summarize these inspection results in a short paragraph. Do not call tools.')])]
    return result, [c['id'] for c in calls]


def first_difference(left, right):
    return next((i for i,(a,b) in enumerate(zip(left,right)) if a != b), min(len(left),len(right)))


def check_evidence(capture, rows, profile, concurrency):
    archive = q.wait_terminal_publication(capture,rows)
    graphs = q.records(capture/'block_graph.jsonl')
    target = q.records(capture/'verified_restores.jsonl')
    recurrent = q.records(capture/'verified_state_restores.jsonl')
    action = q.records(capture/'action_attention.jsonl')
    proof = []
    for row in rows:
        events = [e for e in graphs if q.native_matches(row,e)]
        query = [e for e in events if e.get('phase') == 'query']
        pressure = [e for e in events if e.get('phase') == 'pressure']
        q.require(query, 'No V2 Query graph event for a structured request')
        for event in events:
            provenance = event.get('locality_provenance',{})
            q.require(provenance.get('source') == 'deepswe_protocol' and provenance.get('exact_spans',0) > 0,
                      'Query/pressure selected without structured locality provenance')
            q.require(provenance == row['metadata_check']['provenance'],
                      'Query/pressure used stale or altered request locality provenance')
            q.require(event.get('scoring_version') == 'v2' and
                      event.get('alpha') == profile['kvmem']['graph_alpha'], 'Scoring policy changed')
            q.require(0 < event.get('selected_tokens',0) <= profile['kvmem']['select_budget'],
                      'Selection exceeded the unchanged K budget')
        completed = [e for e in archive.get('completed',[]) if q.native_matches(row,e)]
        expected = row['input_tokens'] + len(row['output_ids']) - 1
        q.require(completed and completed[-1].get('terminal_prefix',{}).get('history_end') == expected,
                  'Terminal accepted watermark differs from actual consumed output IDs')
        action_rows = [e for e in action if q.native_matches(row,e)]
        q.require(action_rows and action_rows[-1].get('block_graph',{}).get('accepted_through') == expected,
                  'Graph accepted watermark differs from consumed tokens')
        q.require(action_rows[-1].get('samples') == row['expected_output_samples'],
                  'Output sampling differs from replay of actual accepted output IDs')
        item = dict(agent=row['agent'],turn=row['turn'],query=query,pressure=pressure)
        if row['turn']:
            kv = [e for e in target if q.native_matches(row,e)]
            state = [e for e in recurrent if q.native_matches(row,e)]
            q.require(kv and state and all(e.get('byte_equal') and
                      e.get('working_attention_layers',0) > 0 and e.get('mtp_attention_layers',0) > 0 for e in kv) and
                      all(e.get('recurrent_byte_equal') for e in state),
                      'Continuation/edit lacks byte-verified target/MTP/recurrent restoration')
            if row['turn'] == 2:
                q.require(any(e.get('checkpoint_kind') == 'P' for e in state),
                          'Edited Query did not exercise the retained P checkpoint')
            item.update(target_restore=kv,state_restore=state)
        proof.append(item)
    for agent in range(concurrency):
        initial = [e for p in proof if p['agent'] == agent and p['turn'] == 0 for e in p['pressure']]
        q.require(len(initial) >= 2, 'Initial long structured history did not exercise two graph pressure events')
        q.require(all(e.get('seed_source') in ('output','prefill','none') for e in initial),
                  'Pressure used an undeclared seed source')
    q.require(not (capture/'restore_mismatch.json').exists(), 'Restore byte mismatch exists')
    return proof


def load_fixture(args):
    fixture = None
    if getattr(args, 'fixture', None):
        fixture = json.loads(args.fixture.read_text())
        q.require(fixture.get('kind') == 'graph_v2_structured_sessions_v1' and not set(fixture) - {
            'kind','prompt_tokens','output_tokens','concurrency','timeout','initial_token_sha256'}, 'Unsupported structured session fixture')
        for field in ('prompt_tokens','output_tokens','concurrency'):
            q.require(type(fixture.get(field)) is int and fixture[field] > 0, 'Invalid structured fixture ' + field)
            setattr(args,field,fixture[field])
        if 'timeout' in fixture:
            q.require(type(fixture['timeout']) is int and 1 <= fixture['timeout'] <= 86400,
                      'Invalid structured fixture timeout')
            args.timeout = fixture['timeout']
    return fixture


def run(args):
    fixture = load_fixture(args)
    q.require(args.protocol is not None and args.protocol.is_file(), 'Provide a frozen --protocol library')
    profile = json.loads(args.config.read_text())
    kv = profile['kvmem']
    if getattr(args, 'arm', None):
        q.require(args.arm == kv.get('graph_mode'), 'Frozen comparison arm differs from profile graph mode')
    q.require(kv.get('graph_scoring_version') == 'v2' and kv.get('pressure_policy') == 'graph' and
        kv.get('graph_mode') in ('locality','attention','multi'), 'Require an explicit V2 graph-pressure profile')
    q.require(profile.get('verify_session_restore') is True and kv.get('session_checkpoint_policy') == 'pm' and
        kv.get('session_capture_terminal') is True, 'Enable byte-verified P/M session checkpoints')
    q.require(kv.get('action_attention') and kv.get('prefill_attention') and kv.get('action_fraction') == 0 and
        not kv.get('query_history',False) and not kv.get('repository_gc',False), 'Keep V2 attention seed contracts')
    q.require(args.concurrency >= 2 and profile['max_num_seqs'] >= args.concurrency and
        kv['session_capacity'] >= args.concurrency, 'Require two independent concurrent owned sessions')
    q.require(129 <= args.output_tokens <= 257, 'Smoke output quota must be 129..257 tokens; natural EOS is allowed')
    q.require(not args.output.exists(), 'Refuse to overwrite prior qualification evidence')
    args.output.mkdir(parents=True)
    bridge = module('graph_v2_session_bridge',ROOT/'benchmarks/deepswe/claude_bridge.py')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'],local_files_only=True)
    planner = bridge.Planner(bridge.Protocol(args.protocol),tokenizer,kv['gen_budget'])
    fixtures = [build_initial(planner,bridge,profile,i,args.prompt_tokens) for i in range(args.concurrency)]
    if fixture and 'initial_token_sha256' in fixture:
        q.require(fixture['initial_token_sha256'] == [q.digest(p[1][0]) for p in fixtures],
                  'Structured inputs differ from the shared frozen token fingerprints')
    q.save(args.output/'manifest.json',dict(schema_version=2,scope=__doc__,profile=profile,
        fixture_sha256=None if fixture is None else q.digest(fixture),
        protocol=str(args.protocol.resolve()), protocol_sha256=bridge.hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        producer=str(ROOT/'benchmarks/deepswe/claude_bridge.py'),
        requested_prompt_tokens=args.prompt_tokens, actual_prompt_tokens=[len(p[1][0]) for p in fixtures],
        output_quota=args.output_tokens,thinking_token_budget=0,min_tokens=0,temperature=0,
        concurrency=args.concurrency,async_restore=kv.get('async_restore',False),
        fixture_rule='Long historical tool_result; real returned tool IDs; changed final user Query after a shared accepted prefix'))
    if getattr(args, 'prepare_only', False):
        for index, (payload, prepared) in enumerate(fixtures):
            q.save(args.output/f'agent-{index}-initial.json',dict(payload=payload,tokens=prepared[0],plan=prepared[1]))
        q.save(args.output/'results.json',dict(status='prepared',scope='CPU only; no HTTP requests or model inference',
            owners=args.concurrency,prompt_tokens=[len(p[1][0]) for p in fixtures]))
        print(json.dumps(dict(status='prepared',owners=args.concurrency)),flush=True)
        return
    sessions = [f'graph-v2-structured-{i}-{uuid.uuid4().hex}' for i in range(args.concurrency)]
    rows, failures, releases = [], [], []
    result = None
    def client(index):
        payload, prepared = fixtures[index]
        state = bridge.Trajectory()
        branch = prior_prompt = None
        for turn in range(3):
            metadata = verify_plan(prepared,profile)
            q.save(args.output/f'agent-{index}-turn-{turn}-input.json',
                   dict(payload=payload,tokens=prepared[0],plan=prepared[1],metadata_check=metadata))
            native, row = call(profile,bridge,sessions[index],prepared,args.output_tokens,args.timeout)
            response = planner.response(native['choices'][0]['text'],prepared[2]['tools'],payload['model'],
                                        native['usage'],native['choices'][0]['finish_reason'])
            row.update(agent=index,turn=turn,session=sessions[index],metadata_check=metadata,
                expected_output_samples=q.expected_output_samples(tokenizer,profile,prepared[0],row['output_ids']))
            if turn == 2:
                row['changed_suffix_begin'] = first_difference(prior_prompt,prepared[0])
                q.require(row['changed_suffix_begin'] >= len(branch.tokens)+len(branch.output) and
                          row['changed_suffix_begin'] < len(prior_prompt), 'Expected a real edit after the accepted branch prefix')
            rows.append(row)
            q.save(args.output/f'agent-{index}-turn-{turn}-response.json',dict(native=native,response=response,row=row))
            planner.commit(state,prepared,payload,response,row['output_ids'])
            if turn == 0:
                q.require(row['output_ids'][-1] == planner.end, 'Tool answer lacks accepted message EOS; cannot prove exact reuse')
                branch = copy.deepcopy(state)
                payload, calls = next_payload(payload,response)
                row['real_tool_use_ids'] = calls
                prepared = planner.prepare(payload,state)
                q.require(prepared[5] and prepared[0][:len(state.tokens)+len(state.output)] == state.tokens+state.output,
                          'Tool continuation did not preserve real accepted assistant IDs')
            elif turn == 1:
                prior_prompt = list(prepared[0])
                payload = copy.deepcopy(payload)
                payload['messages'][-1]['content'][-1]['text'] = (
                    'Correction: explain why the inspected records remain owned by this session after a suffix edit. '
                    'Use a short paragraph and do not call any tools.')
                state = copy.deepcopy(branch)
                prepared = planner.prepare(payload,state)
                q.require(prepared[5], 'Edited branch lost its verified real output prefix')
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(client,i) for i in range(args.concurrency)]
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    failures.append(f'{type(error).__name__}: {error}')
        q.require(not failures,'; '.join(failures))
        proof = check_evidence(args.capture,rows,profile,args.concurrency)
        result = dict(status='passed',scope='Structured producer, accepted-token and restore lifecycle; no quality/throughput claim',
            evidence=proof,owners=args.concurrency,turns=len(rows))
    except BaseException as error:
        failures.append(f'{type(error).__name__}: {error}')
    finally:
        for session in sessions:
            try:
                releases.append(q.stream_request(profile,session,[planner.end],1,0,release=True,timeout=args.timeout))
            except Exception as error:
                failures.append(f'Release {session}: {type(error).__name__}: {error}')
        released = None
        if len(releases) == len(sessions):
            try:
                released = q.wait_released(args.capture,set(sessions))
            except Exception as error:
                failures.append(f'Retirement: {type(error).__name__}: {error}')
        q.save(args.output/'cleanup.json',dict(releases=releases,state=released,errors=failures))
        q.save(args.output/'traces.json',sorted(rows,key=lambda r:(r['agent'],r['turn'])))
        result = result or dict(status='failed')
        result.update(status='failed' if failures else 'passed',errors=failures,
                      all_owned_sessions_released=released is not None)
        q.save(args.output/'results.json',result)
    q.require(not failures,'; '.join(failures))
    print(json.dumps(dict(status=result['status'],owners=args.concurrency,turns=len(rows))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,help='Versioned structured workload settings shared across sync/async')
    parser.add_argument('--arm',choices=('locality','attention','multi'))
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--prompt-tokens',type=int,default=131072)
    parser.add_argument('--output-tokens',type=int,default=257)
    parser.add_argument('--timeout',type=float,default=1200)
    parser.add_argument('--prepare-only',action='store_true',help='Validate/freeze real structured inputs on CPU without contacting the server')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

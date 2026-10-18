#!/usr/bin/env python3
"""Two owned sessions: stable tool continuation, M->P feedback, guided Query policy.

Prepare once with --config PROFILE --fixture FIXTURE --prepare-fixture, then run
through the qualification controller with the ordinary config/capture/output/
fixture/arm arguments. This client never starts or stops a model server.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

from qualify_block_graph import (build_fixture as graph_fixture, digest, native_matches,
    records, require, save, summarize_steps, wait_released, wait_terminal_publication)


def validate_profile(profile):
    kv = profile['kvmem']
    require(profile.get('execution_mode') == 'compiled_graph' and profile.get('draft_method') == 'mtp'
        and profile.get('draft_tokens') == 4 and profile.get('verify_session_restore'),
        'Alignment qualification requires compiled MTP4 and restore byte verification')
    require(profile.get('max_num_seqs', 0) >= 2 and profile.get('max_model_len', 0) >= 32768,
        'Alignment qualification needs two native slots and 32K logical capacity')
    require(kv.get('select_budget') == kv.get('gen_budget') == 4096 and kv.get('block_tokens') == 32,
        'Keep K4096/G4096/B32 fixed')
    require(kv.get('session_capacity', 0) >= 3 and kv.get('session_checkpoint_policy') == 'pm'
        and kv.get('session_capture_terminal') and kv.get('request_planning'),
        'Owned P/M checkpoints and request planning are required')
    require(kv.get('archive_backend') == 'cpu' and not kv.get('repository_gc', False),
        'Keep complete CPU history without repository GC')
    require(kv.get('action_attention') and kv.get('prefill_attention') and kv.get('query_history')
        and kv.get('action_scope') == 'non-thinking' and not kv.get('action_pressure', False),
        'Enable output/prefill feedback and Query history, without pressure reselection')
    require(kv.get('graph_mode', 'off') in ('off', 'collect') and kv.get('guided_query_tokens', 0) >= 1,
        'This alignment probe supports guided Query with graph off or collection only')
    require(profile.get('query_replay_budget', 0) >= 3072 and kv.get('sink_blocks') == 41,
        'Budget negative control requires replay allowance >=3072 and 41 sink blocks')


def build_fixture(tokenizer):
    base = graph_fixture(tokenizer, prompt_tokens=12288, concurrency=2)
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    fill = encode('Verified tool diagnostic: immutable records remain available; generation handles '
        'and session ownership are unchanged. The next action should preserve the selected historical pages.\n')
    require(fill, 'Empty tool fixture')
    for agent in base['agents']:
        project = f"Project Meridian-{agent['index']:02d}"
        close = encode('<|im_end|>\n')
        tool_header = encode('<|im_end|>\n<|im_start|>tool\n')
        user_header = encode('<|im_start|>user\n')
        question = encode(f'The tool finished checking {project}. Repeat the exact deployment code first, '
            'then write at least 250 words explaining the preserved archive and restoration checks.')
        suffix = close+user_header+question+agent['footer']
        count = 2048-len(tool_header)-len(suffix)
        require(count > 0, 'Tool template exceeds 2048 tokens')
        body = (fill*((count+len(fill)-1)//len(fill)))[:count]
        agent['tool_append_tokens'] = tool_header+body+suffix
        agent['tool_append_spans'] = [dict(begin=len(close), end=len(tool_header)+len(body)+len(close), role='tool'),
            dict(begin=len(tool_header)+len(body)+len(close),
                end=len(tool_header)+len(body)+len(close)+len(user_header)+len(question)+len(close), role='user')]
        reminder = encode(f'Confirm the same deployment code for {project}. Start with the code, then '
            'explain the verified tool evidence and cross-turn context restoration in at least 250 words.')
        agent['force_append_tokens'] = close+user_header+reminder+agent['footer']
        agent['force_append_spans'] = [dict(begin=len(close),
            end=len(close)+len(user_header)+len(reminder)+len(close), role='user')]
        agent['expected_code'] = agent['expected_codes'][0]
    fixture = dict(kind='context_alignment_v1', eos_token_id=base['eos_token_id'], agents=base['agents'])
    fixture['sha256'] = digest(fixture)
    return fixture


def prepare_fixture(profile, path):
    require(not path.exists(), 'Refuse to replace a frozen fixture')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    fixture = build_fixture(tokenizer)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, fixture)
    return fixture


def validate_fixture(fixture):
    require(fixture.get('kind') == 'context_alignment_v1' and len(fixture.get('agents', [])) == 2,
        'Expected two fixed context-alignment users')
    require(fixture.get('sha256') == digest({k:v for k,v in fixture.items() if k != 'sha256'}),
        'Frozen context fixture changed')
    for agent in fixture['agents']:
        require(11000 < len(agent['initial_tokens']) < 13000 and len(agent['tool_append_tokens']) == 2048,
            'Keep the initial 12K input and 2K tool input fixed')
        require(agent['first_input_sha256'] == digest(agent['initial_tokens']), 'First prompt changed')
        require(agent['query_span'][0]//32*32 > 4096, 'First Query must follow a bounded active view')


def append_turn(agent, previous_prompt, output, spans, turn):
    kind = 'tool' if turn == 1 else 'force'
    prefix = list(previous_prompt)+list(output)
    additions = [dict(begin=s['begin']+len(prefix), end=s['end']+len(prefix), role=s['role'])
        for s in agent[kind+'_append_spans']]
    return prefix+agent[kind+'_append_tokens'], [dict(s) for s in spans]+additions


def request_body(profile, session, tokens, *, turn=0, query=None, spans=(), release=False):
    body = dict(model=profile['served_model_name'], session_id=session, cache_salt=session,
        prompt=tokens, max_tokens=1 if release else 257, min_tokens=0 if release else 129,
        temperature=0, seed=1927, ignore_eos=bool(release), return_token_ids=True,
        stream=True, stream_options={'include_usage': True}, kvmem_mandatory_parts='none',
        kvmem_reselect='off' if release or turn == 1 else 'force')
    if release:
        body['kv_transfer_params'] = {'kvmem_release_session': True}
    else:
        body['kvmem_message_spans'] = list(spans)
        if query is not None:
            # The stable callback must retain the old Query-origin P, even
            # though it deliberately performs no new Query scoring.
            body['kvmem_mandatory_spans'] = [dict(begin=query[0], end=query[1], kind='query')]
        if turn == 1:
            body['kvmem_prefill_window'] = 'keep_selected'
        else:
            body['kvmem_query_span'] = query
    return body


def stream_request(profile, session, tokens, **kwargs):
    body = request_body(profile, session, tokens, **kwargs)
    row = dict(begin=time.monotonic(), wall_begin=time.time(), session=session,
        input_tokens=len(tokens), input_sha256=digest(tokens), input_ids=list(tokens),
        query_span=body.get('kvmem_query_span'), output_ids=[], chunks=[], text='',
        usage=None, response_id=None, finish_reason=None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    done = False
    try:
        with opener.open(request, timeout=600) as response:
            for raw in response:
                if not raw.startswith(b'data:'):
                    continue
                raw = raw[5:].strip()
                if raw == b'[DONE]':
                    done = True
                    break
                event = json.loads(raw)
                require(not event.get('error'), f"Native stream failed: {event.get('error')}")
                if event.get('id'):
                    require(row['response_id'] in (None, event['id']), 'Response ID changed')
                    row['response_id'] = event['id']
                row['usage'] = event.get('usage') or row['usage']
                for choice in event.get('choices', []):
                    require(choice.get('index', 0) == 0, 'Multiple choices are unsupported')
                    output = choice.get('token_ids') or []
                    if output:
                        row['chunks'].append([time.monotonic(), len(output)])
                        row['output_ids'].extend(output)
                    row['text'] += choice.get('text') or ''
                    row['finish_reason'] = choice.get('finish_reason') or row['finish_reason']
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from None
    row['end'] = time.monotonic()
    require(done and row['response_id'] and row['usage'] and
        row['usage'].get('completion_tokens') == len(row['output_ids']) and
        body['min_tokens'] <= len(row['output_ids']) <= body['max_tokens'] and
        row['finish_reason'] in ('stop', 'length'), 'Incomplete accepted public token stream')
    row['seconds'] = row['end']-row['begin']
    row['ttft_seconds'] = row['chunks'][0][0]-row['begin'] if row['chunks'] else None
    interval = row['chunks'][-1][0]-row['chunks'][0][0] if len(row['chunks']) > 1 else 0
    row['generation_period_tokens_per_second'] = (
        (len(row['output_ids'])-row['chunks'][0][1])/interval if interval > 0 else None)
    return row


def invalid_force(profile, session, tokens):
    body = request_body(profile, session, tokens[:7168], query=[4096, 7168])
    body.update(max_tokens=4096, stream=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with opener.open(request, timeout=60) as response:
            response.read()
    except urllib.error.HTTPError as error:
        payload = error.read().decode()
        require(error.code == 400, f'Invalid forced Query returned HTTP {error.code}: {payload}')
        require('Complete Query, output reserve and fixed sink/recent bands exceed active capacity '
                '(selected budget 992, required 1312)' in payload,
            f'HTTP400 did not exercise joint active/generation admission: {payload}')
        return dict(status=error.code, error=payload, seconds=time.monotonic()-started,
            input_tokens=7168, query_span=[4096,7168], max_tokens=4096)
    raise AssertionError('Impossible forced Query was admitted instead of HTTP400')


def one(rows, message):
    require(len(rows) == 1, message)
    return rows[0]


def verify_evidence(capture, rows, profile):
    refresh = records(capture/'bounded_refresh.jsonl')
    actions = records(capture/'action_attention.jsonl')
    prefill = records(capture/'prefill_attention.jsonl')
    states = records(capture/'verified_state_restores.jsonl')
    targets = records(capture/'verified_restores.jsonl')
    histories = records(capture/'query_history.jsonl')
    ownership, proofs, guided = set(), [], []
    require(len(rows) == 6 and len({r['response_id'] for r in rows}) == 6, 'Six unique public responses required')
    for agent in range(2):
        turns = sorted((r for r in rows if r['agent'] == agent), key=lambda r:r['turn'])
        require([r['turn'] for r in turns] == [0,1,2], 'Incomplete owned trajectory')
        own = lambda data, row: [e for e in data if native_matches(row, e)]
        action = [one(own(actions, row), 'Missing or duplicate action terminal') for row in turns]
        namespace = {row['terminal'].get('authority_namespace') for row in turns}
        require(len(namespace) == 1 and None not in namespace and not (namespace & ownership),
            'Owned authorities crossed users or changed across continuation')
        ownership.update(namespace)
        p = turns[0]['query_span'][0]//32*32
        for index, row in enumerate(turns):
            terminal = row['terminal']
            end = terminal.get('checkpoint_tokens')
            require(terminal.get('checkpoint_kind') == 'M' and end == terminal.get('history_end') and
                end == terminal.get('terminal_prefix', {}).get('history_end') and
                terminal['terminal_prefix'].get('output_tokens') == len(row['output_ids']) and
                0 <= len(row['input_ids'])+len(row['output_ids'])-end <= 1,
                'Terminal M is not the consumed accepted public prefix')
            if index:
                previous = turns[index-1]['input_ids']+turns[index-1]['output_ids']
                require(row['input_ids'][:len(previous)] == previous, 'Continuation changed a public token prefix')
                oracle = one(own(states, row), 'Each continuation requires one recurrent byte oracle')
                expected = turns[0]['terminal']['checkpoint_tokens'] if index == 1 else p
                require(oracle.get('history_tokens') == expected and oracle.get('checkpoint_kind') == ('M' if index == 1 else 'P')
                    and oracle.get('recurrent_byte_equal') is True and not oracle.get('pressure_restore'),
                    'Continuation restored the wrong M/P state')
                kv = one(own(targets, row), 'Each continuation requires one KV byte oracle')
                require(kv.get('history_tokens') == expected and kv.get('byte_equal') is True and
                    kv.get('working_attention_layers', 0) > 0 and kv.get('mtp_attention_layers', 0) > 0,
                    'Target/MTP KV restore did not match its CPU oracle')
        require(not own(refresh, turns[1]), 'Stable tool continuation changed the selected active view')
        require(action[1].get('prefill_end', 0) > p and action[1].get('prefill_end', 0) > action[0].get('prefill_end', 0)
            and action[1].get('prefill_samples', 0) > action[0].get('prefill_samples', 0),
            'Tool input did not create prefill feedback beyond the original P')
        keys = ('prefill_end', 'prefill_samples', 'prefill_chunks')
        require(all(action[2].get(k) == action[1].get(k) for k in keys) and not own(prefill, turns[2]),
            'M->P replay lost or counted prefill feedback twice')
        query_rows = []
        for index in (0,2):
            query = one([event for event in own(refresh, turns[index]) if event.get('selection') == 'query'],
                'Public Query requires one bounded selection')
            require(query.get('selection') == 'query' and query.get('query_span') == turns[0]['query_span']
                and 0 < query.get('selected_tokens', 0) <= 4096, 'Query identity or active budget changed')
            require(query.get('query_history_enabled') is True and query.get('query_history_pages', 0) > 0,
                'Guided Query did not use historical relevance')
            required = ('guided_published_tokens', 'guided_consumed_tokens', 'guided_captured_tokens',
                'guided_natural_termination', 'guided_fallback')
            require(all(k in query for k in required), 'Missing guided policy diagnostics')
            published = query['guided_published_tokens']
            require(0 <= published <= query['guided_captured_tokens'] <= query['guided_consumed_tokens'] <=
                profile['kvmem']['guided_query_tokens'], 'Guided rows exceed their consumed budget')
            if published:
                require(query['guided_natural_termination'] is True and query['guided_fallback'] is None and
                    query.get('query_score_tokens') == published, 'Unterminated guided Query affected scores')
            else:
                require(bool(query['guided_fallback']) and query.get('query_score_tokens') ==
                    turns[0]['query_span'][1]-turns[0]['query_span'][0], 'Fallback did not score the public Query')
            if index == 2:
                require(query.get('prefill_attention_samples') == action[1]['prefill_samples'] and
                    query.get('prefill_attention_chunks') == action[1]['prefill_chunks'],
                    'Query selection saw P feedback instead of the verified M history')
            query_rows.append(query)
            guided.append(query)
        history = [one(own(histories, row), 'Missing Query-history round') for row in turns]
        require([h.get('observed') for h in history] == [True,False,True] and
            [h.get('round') for h in history] == [1,2,3] and
            history[1].get('score_sha256') == history[0].get('score_sha256'),
            'Stable continuation aged or duplicated Query relevance')
        proofs.append(dict(agent=agent, authority_namespace=next(iter(namespace)), prompt_checkpoint=p,
            prefill_terminals=[{k:a[k] for k in keys} for a in action], queries=query_rows, query_history=history))
    cap = profile['kvmem']['guided_query_tokens']
    published_count = sum(q['guided_published_tokens'] > 0 for q in guided)
    if cap == 1:
        require(published_count == 0 and all(q['guided_fallback'] for q in guided), 'Small-cap probe did not fall back')
    require(not (capture/'restore_mismatch.json').exists(), 'A model restore mismatch was recorded')
    return dict(complete=cap == 1 or published_count > 0, guided_cap=cap,
        guided_published_queries=published_count, guided_fallback_queries=len(guided)-published_count,
        isolated_authorities=len(ownership), trajectories=proofs)


def run(args):
    profile = json.loads(args.config.read_text())
    validate_profile(profile)
    fixture = json.loads(args.fixture.read_text())
    validate_fixture(fixture)
    require(not args.output.exists(), 'Use a new output directory')
    args.output.mkdir(parents=True)
    save(args.output/'profile.json', profile)
    save(args.output/'fixture.json', fixture)
    sessions = [f'context-alignment-{i}-{uuid.uuid4().hex}' for i in range(3)]
    rows, failures, releases, result = [], [], [], None
    started = time.monotonic()

    def client(index):
        agent = fixture['agents'][index]
        prompt, spans = agent['initial_tokens'], agent['message_spans']
        for turn in range(3):
            row = stream_request(profile, sessions[index], prompt, turn=turn,
                query=agent['query_span'], spans=spans)
            row.update(agent=index, turn=turn, expected_code=agent['expected_code'],
                correct=agent['expected_code'] in row['text'])
            archive = wait_terminal_publication(args.capture, [row])
            row['terminal'] = one([e for e in archive['completed'] if native_matches(row, e)],
                'Missing unique owned terminal')
            rows.append(row)
            save(args.output/f'agent-{index}-turn-{turn}.json', row)
            if turn < 2:
                prompt, spans = append_turn(agent, prompt, row['output_ids'], spans, turn+1)

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            clients = [pool.submit(client, index) for index in range(2)]
            negative = pool.submit(invalid_force, profile, sessions[2], fixture['agents'][0]['initial_tokens'])
            for future in clients:
                try:
                    future.result()
                except Exception as error:
                    failures.append(f'{type(error).__name__}: {error}')
            invalid = negative.result()
        ended = time.monotonic()
        require(not failures, '; '.join(failures))
        checks = verify_evidence(args.capture, rows, profile)
        result = dict(status='passed' if checks['complete'] else 'incomplete', arm=args.arm,
            fixture_sha256=fixture['sha256'], concurrency=2, successful_turns=len(rows),
            seconds=ended-started, output_tokens=sum(len(r['output_ids']) for r in rows),
            end_to_end_output_tokens_per_second=sum(len(r['output_ids']) for r in rows)/(ended-started),
            quality=dict(correct=sum(r['correct'] for r in rows), total=len(rows),
                criterion='Expected deployment code in a synthetic technical answer; not a benchmark score'),
            step_metrics=summarize_steps(args.capture, started, ended), alignment_checks=checks,
            invalid_force=invalid, scope='Two three-turn owned sessions; per-request generation-period rates include '
                'interference. Restore verification, telemetry and retirement waits are enabled. '
                'No pure decode rate is inferred from speculative proposals.')
    except BaseException as error:
        failures.append(f'{type(error).__name__}: {error}')
        raise
    finally:
        for session in sessions:
            try:
                releases.append(stream_request(profile, session, [fixture['eos_token_id']], release=True))
            except Exception as error:
                failures.append(f'Release {session}: {type(error).__name__}: {error}')
        released = None
        if len(releases) == len(sessions):
            try:
                released = wait_released(args.capture, set(sessions))
            except Exception as error:
                failures.append(f'Retirement: {type(error).__name__}: {error}')
        save(args.output/'cleanup.json', dict(all_owned_sessions_released=released is not None,
            state=released, releases=releases, errors=failures))
        save(args.output/'traces.json', sorted(rows, key=lambda row:(row['agent'],row['turn'])))
        result = result or dict(status='failed', arm=args.arm)
        result.update(all_owned_sessions_released=released is not None, errors=failures)
        if failures:
            result['status'] = 'failed'
        save(args.output/'results.json', result)
    require(not failures, '; '.join(failures))
    print(json.dumps({k:result[k] for k in ('status','arm','seconds','quality')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arm', default='context-alignment')
    parser.add_argument('--prepare-fixture', action='store_true')
    args = parser.parse_args()
    if args.prepare_fixture:
        fixture = prepare_fixture(json.loads(args.config.read_text()), args.fixture)
        print(json.dumps(dict(fixture=str(args.fixture), sha256=fixture['sha256'])))
    else:
        require(args.capture is not None and args.output is not None, 'Supply --capture and --output')
        run(args)


if __name__ == '__main__':
    main()

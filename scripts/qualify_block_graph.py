"""Small, reproducible block-graph qualification against an owned running server.

Run the same saved --fixture against the six independently launched profiles:
Mean-K, collect, legacy cumulative attention, locality, attention and multi.
This client does not start or stop a GPU server. Its own sessions are explicitly
released even when an assertion fails. Throughput is diagnostic, not a benchmark
claim: byte-verified restores deliberately synchronize model state.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


ARMS = ('baseline', 'collect', 'legacy', 'locality', 'attention', 'multi')
SCORING_MODES = ('locality', 'attention', 'multi')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'),
        sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def records(path):
    if not path.exists():
        return []
    # A live JSONL writer may have an incomplete final line. Never hide an
    # invalid completed record or reinterpret one as empty evidence.
    data = path.read_text()
    return [json.loads(line) for line in data.splitlines() if line.strip()] if data.endswith('\n') else [
        json.loads(line) for line in data.splitlines()[:-1] if line.strip()]


def native_matches(row, event):
    rid = event.get('request_id', '')
    response = row.get('response_id')
    return bool(response and (rid == response or rid.startswith(response + '-')))


def expected_output_samples(tokenizer, profile, prompt, output):
    """Independent parser replay uses only consumed, returned public tokens."""
    from types import SimpleNamespace
    from vllm_kvmem.action_attention import ActionHistory
    markers = [tokenizer.convert_tokens_to_ids(marker) for marker in (
        '<tool_call>', '</tool_call>', '<think>', '</think>', '<|im_start|>')]
    kv = profile['kvmem']
    state = ActionHistory(SimpleNamespace(action_scope=kv['action_scope'],
        action_interval=kv.get('action_interval', 16)), markers)
    state.begin('accepted-token-oracle', prompt)
    count = 0
    for begin in range(0, len(output)-1, 32):
        # The final returned token has not supplied a consumed Q row. It is
        # excluded regardless of whether the request stops at EOS or max_tokens.
        chunk = output[:-1][begin:begin+32]
        state.parser.prepare(chunk)
        count += state.parser.commit(len(chunk)).bit_count()
    return count


def validate_profile(profile, arm, concurrency, output_tokens, minimum_output):
    require(arm in ARMS, 'Unknown comparison arm')
    require(type(concurrency) is int and concurrency >= 2, 'Use at least two concurrent owned sessions')
    kv = profile.get('kvmem', {})
    require(profile.get('draft_method') == 'mtp' and profile.get('draft_tokens') == 4 and
        profile.get('execution_mode') == 'compiled_graph', 'Require compiled native MTP(k=4)')
    require(profile.get('verify_session_restore') is True,
        'Enable verify_session_restore for actual P/M byte evidence')
    require(profile.get('max_num_seqs', 0) >= concurrency and kv.get('session_capacity', 0) >= concurrency,
        'Native and session capacities must accommodate all clients')
    require(kv.get('block_tokens') == 32 and kv.get('request_planning') is True and
        kv.get('session_checkpoint_policy') == 'pm' and kv.get('session_capture_terminal') is True,
        'Require 32-token blocks and planned terminal P/M checkpoints')
    require(kv.get('archive_backend') == 'cpu' and not kv.get('repository_gc', False),
        'Qualification requires complete CPU history without repository GC')
    require(not kv.get('query_history', False) and not kv.get('action_pressure', False) and
        kv.get('guided_query_tokens', 0) == 0, 'Keep history, pressure fusion and guided Query disabled in every arm')
    mode = kv.get('graph_mode', 'off')
    require(mode == (arm if arm in ('collect', *SCORING_MODES) else 'off'),
        'Running graph mode does not match the declared arm')
    fraction = kv.get('action_fraction', 0.25) if kv.get('action_attention', False) else 0
    require(fraction > 0 if arm == 'legacy' else fraction == 0,
        'Only the legacy arm may use cumulative attention fusion')
    require(kv.get('action_attention') is True and kv.get('prefill_attention') is True and
        kv.get('action_scope') == 'non-thinking',
        'Keep identical prefill and non-thinking attention capture in all six arms; zero fusion defines baseline')
    require(type(minimum_output) is int and type(output_tokens) is int and
        129 <= minimum_output <= output_tokens <= kv.get('gen_budget', 0),
        'Use at least 129 accepted output tokens within the unchanged generation budget')


def build_fixture(tokenizer, *, prompt_tokens=12288, concurrency=2):
    """Freeze complete first-turn inputs and tool/question text for every arm."""
    require(type(prompt_tokens) is int and prompt_tokens >= 2048, 'Prompt fixture is too short')
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    filler = encode('The archive uses immutable historical keys. Active pages are rebuilt from selected records. '
        'The storage controller checks generation handles before serving reads. '
        'The service keeps session ownership independent from scheduler batch positions.\n')
    require(bool(filler), 'Tokenizer produced an empty filler')
    repeat = lambda count: (filler * ((count + len(filler)-1)//len(filler)))[:count]
    agents = []
    for index in range(concurrency):
        project = f'Project Meridian-{index:02d}'
        record = f'ledger-{71+index}'
        code = f'ORCHID-{4817+index*13}'
        fallback = f'CEDAR-{7291+index*17}'
        prefix, spans = [], []
        def message(role, content):
            begin = len(prefix)
            prefix.extend(encode(f'<|im_start|>{role}\n'))
            prefix.extend(encode(content) if isinstance(content, str) else content)
            prefix.extend(encode('<|im_end|>\n'))
            spans.append(dict(begin=begin, end=len(prefix), role=role))
        message('system', 'Answer from the archive. Put the exact requested access code on the first line, '
            'then provide a detailed technical explanation. Do not invent a code. Thinking is disabled.')
        dependency = encode(f'Validated access record {record}: the deployment code is {code}; '
            f'the recovery code is {fallback}. Both exact strings are authoritative.\n')
        source = encode(f'{project} depends on access record {record}. When asked about this project, '
            'resolve its deployment or recovery code through that record.\n')
        archive = repeat(prompt_tokens//4) + dependency + repeat(prompt_tokens//3) + source
        archive += repeat(max(0, prompt_tokens-len(prefix)-len(archive)-160))
        message('user', archive)
        message('assistant', 'The archive and its record dependencies are available.')
        archive_prefix = list(prefix)
        header = encode('<|im_start|>user\n')
        message_close = encode('<|im_end|>\n')
        footer = encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
        question = encode(f'What is the deployment code for {project}? Resolve its referenced access record. '
            'Begin with the exact code. Then explain step by step how immutable history, generation handles, '
            'working-set restoration and session isolation preserve the dependency. Write at least 250 words.')
        begin = len(prefix) + len(header)
        tokens = prefix + header + question + footer
        first_spans = spans + [dict(begin=len(prefix), end=begin+len(question)+len(message_close), role='user')]
        tool = encode('<|im_end|>\n<|im_start|>tool\nThe deployment audit is complete. '
            f'Now inspect the same {project} recovery record without changing archive facts.<|im_end|>\n')
        second = encode(f'What is the recovery code for {project}? Begin with the exact recovery code. '
            'Then write at least 250 words explaining the historical dependency and safe restoration procedure.')
        edited = encode(f'Correction to this question: give the deployment code, not the recovery code, '
            f'for {project}. Begin with the exact deployment code and write at least 250 words '
            'explaining why suffix edits must not retain evidence from a later branch.')
        agents.append(dict(index=index, initial_tokens=tokens,
            query_span=[begin, begin+len(question)], message_spans=first_spans,
            archive_prefix_tokens=len(archive_prefix), header=header, footer=footer,
            tool_tokens=tool, tool_span_offset=len(message_close), message_close_tokens=len(message_close),
            second_question=second, edited_question=edited,
            expected_codes=[code, fallback, code], dependency_tokens=dependency,
            source_tokens=source, first_input_sha256=digest(tokens)))
    fixture = dict(kind='block_graph_qualification_v1', agents=agents,
        eos_token_id=tokenizer.eos_token_id, requested_prompt_tokens=prompt_tokens)
    fixture['sha256'] = digest(fixture)
    return fixture


def validate_fixture(fixture, profile, concurrency, output_tokens):
    require(fixture.get('kind') == 'block_graph_qualification_v1', 'Unsupported fixture format')
    require(fixture.get('sha256') == digest({k: v for k, v in fixture.items() if k != 'sha256'}),
        'Fixture hash differs: use the same unmodified fixture across arms')
    require(len(fixture.get('agents', [])) == concurrency, 'Fixture concurrency differs')
    from vllm_kvmem.request_plan import RequestPlan
    kv = profile['kvmem']
    for agent in fixture['agents']:
        tokens = agent['initial_tokens']
        require(len(tokens) > kv['select_budget'] and all(type(t) is int and t >= 0 for t in tokens),
            'Fixture must exceed K and contain valid token IDs')
        require(agent['first_input_sha256'] == digest(tokens), 'First input fingerprint differs')
        require(len(tokens)+3*output_tokens+len(agent['tool_tokens'])+len(agent['second_question'])+
            len(agent['edited_question'])+2*(len(agent['header'])+len(agent['footer'])) < profile['max_model_len'],
            'Fixture exceeds logical history limit')
        RequestPlan.parse(dict(kvmem_query_span=agent['query_span'], kvmem_reselect='force',
            kvmem_message_spans=agent['message_spans'], kvmem_mandatory_parts='none'),
            len(tokens), kv['block_tokens'], kv['gen_budget'])


def next_turn(agent, first, first_output):
    prefix = list(first) + list(first_output)
    spans = [dict(s) for s in agent['message_spans']]
    # Generated assistant tokens and template fragments remain unknown here;
    # never fabricate complete message evidence from a token truncation.
    begin = len(prefix)
    prefix += agent['tool_tokens']
    spans.append(dict(begin=begin+agent['tool_span_offset'], end=len(prefix), role='tool'))
    before_question = list(prefix)
    prefix += agent['header']
    begin = len(prefix)
    prompt = prefix + agent['second_question'] + agent['footer']
    spans.append(dict(begin=len(before_question),
        end=begin+len(agent['second_question'])+agent['message_close_tokens'], role='user'))
    return prompt, [begin, begin+len(agent['second_question'])], spans, before_question


def edited_turn(agent, before_question, second_spans):
    prefix = list(before_question) + agent['header']
    begin = len(prefix)
    prompt = prefix + agent['edited_question'] + agent['footer']
    spans = [dict(s) for s in second_spans[:-1]]
    spans.append(dict(begin=len(before_question),
        end=begin+len(agent['edited_question'])+agent['message_close_tokens'], role='user'))
    return prompt, [begin, begin+len(agent['edited_question'])], spans


def stream_request(profile, session, tokens, count, minimum, *, query=None, spans=(), release=False,
                   timeout=600):
    body = dict(model=profile['served_model_name'], session_id=session, cache_salt=session,
        prompt=tokens, max_tokens=count, min_tokens=minimum, temperature=0, seed=1927,
        ignore_eos=bool(release), return_token_ids=True, stream=True,
        stream_options={'include_usage': True}, kvmem_mandatory_parts='none',
        kvmem_reselect='off' if release else 'force')
    if release:
        body['kv_transfer_params'] = {'kvmem_release_session': True}
    else:
        body.update(kvmem_query_span=query, kvmem_message_spans=spans)
    row = dict(begin=time.monotonic(), wall_begin=time.time(), session=session,
        input_tokens=len(tokens), input_sha256=digest(tokens), query_span=query,
        output_ids=[], chunks=[], text='', usage=None, response_id=None, finish_reason=None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    done = False
    try:
        with opener.open(request, timeout=timeout) as response:
            for raw in response:
                if not raw.startswith(b'data:'):
                    continue
                raw = raw[5:].strip()
                if raw == b'[DONE]':
                    done = True
                    break
                event = json.loads(raw)
                require(not event.get('error'), f'Native stream failed: {event.get("error")}')
                if event.get('id'):
                    require(row['response_id'] in (None, event['id']), 'Response ID changed inside stream')
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
        minimum <= len(row['output_ids']) <= count and row['finish_reason'] in ('stop', 'length'),
        'Incomplete native token stream or output accounting mismatch')
    row['seconds'] = row['end']-row['begin']
    row['ttft_seconds'] = row['chunks'][0][0]-row['begin'] if row['chunks'] else None
    interval = row['chunks'][-1][0]-row['chunks'][0][0] if len(row['chunks']) > 1 else 0
    row['generation_period_tokens_per_second'] = (
        (len(row['output_ids'])-row['chunks'][0][1])/interval if interval > 0 else None)
    return row


def event_evidence(capture, rows, profile):
    refreshes = records(capture / 'bounded_refresh.jsonl')
    target = records(capture / 'verified_restores.jsonl')
    recurrent = records(capture / 'verified_state_restores.jsonl')
    graph = records(capture / 'block_graph.jsonl')
    proofs = []
    for row in rows:
        selected = [e for e in refreshes if native_matches(row, e) and e.get('selection') == 'query']
        require(len(selected) == 1, 'Each public Query must be selected exactly once')
        require(selected[0]['query_span'] == row['query_span'] and
            0 < selected[0]['selected_tokens'] <= profile['kvmem']['select_budget'],
            'Query span or fixed selection budget changed')
        proof = dict(turn=row['turn'], agent=row['agent'], response_id=row['response_id'], query=selected[0],
            graph=[e for e in graph if native_matches(row, e)])
        if row['turn']:
            kv_rows = [e for e in target if native_matches(row, e)]
            state_rows = [e for e in recurrent if native_matches(row, e)]
            require(kv_rows and state_rows and all(e.get('byte_equal') is True and
                e.get('working_attention_layers', 0) > 0 and e.get('mtp_attention_layers', 0) > 0
                for e in kv_rows) and all(e.get('recurrent_byte_equal') is True for e in state_rows),
                'Every continuation/edit requires target, MTP and recurrent byte-equal restore evidence')
            proof.update(target_restore=kv_rows, recurrent_restore=state_rows)
        proofs.append(proof)
    require(not (capture / 'restore_mismatch.json').exists(), 'Model restore mismatch exists')
    return proofs


def assert_real_graph_checks(rows, evidence, action_events, completions, profile):
    """Require real accepted-state evidence; report unobserved cases explicitly.

    Actual rejections and changed selections are group-level observations. A
    run with neither is incomplete for that coverage item, never silently
    reported as proof merely because MTP/graph options were enabled.
    """
    kv = profile['kvmem']
    mode = kv.get('graph_mode', 'off')
    reports, output_by_agent, rejection_total = [], {}, 0
    for row in sorted(rows, key=lambda r: (r['agent'], r['turn'])):
        completed = [e for e in completions if native_matches(row, e)]
        require(completed, 'Missing archive terminal evidence for a public request')
        completion = completed[-1]
        require(type(completion.get('rejected_tokens')) is int and completion['rejected_tokens'] >= 0,
            'Missing speculative rejection accounting')
        rejection_total += completion['rejected_tokens']
        expected_end = row['input_tokens']+len(row['output_ids'])-1
        terminal = completion.get('terminal_prefix')
        require(isinstance(terminal, dict) and terminal.get('history_end') == expected_end,
            'Terminal accepted history differs from consumed public token prefix')
        action = [e for e in action_events if native_matches(row, e)]
        if kv.get('action_attention'):
            require(len(action) == 1, 'Each output requires exactly one attention completion')
            require(action[0].get('samples') == row.get('expected_output_samples'),
                'Accepted-output parser oracle differs: rejected/replayed or missing samples')
        if mode == 'off':
            continue
        query_proof, = [p for p in evidence if (p['agent'], p['turn']) == (row['agent'], row['turn'])]
        query_graphs = query_proof['graph']
        require(len(query_graphs) == 1 and query_graphs[0].get('mode') == mode,
            'Each normal Query requires one matching graph report')
        query_graph = query_graphs[0]
        require(type(query_graph.get('accepted_through')) is int and
            0 <= query_graph['accepted_through'] <= expected_end,
            'Query graph includes an unaccepted future suffix')
        if mode == 'collect':
            require(query_graph.get('selection_unchanged') is True,
                'Collect mode did not retain its legacy selection path')
        stats = action[0].get('block_graph') if action else None
        require(isinstance(stats, dict), 'Missing real graph terminal statistics')
        require(stats.get('accepted_through') == expected_end and
            0 <= stats.get('completed_through', -1) <= expected_end,
            'Graph accepted/completed watermark includes a rejected or absent token')
        require(stats.get('total_bytes', 0) > 0 and stats['total_bytes'] <= kv.get(
            'graph_max_bytes_per_session', 33554432) and stats.get('global_bytes', 0) <= kv.get(
            'graph_max_total_bytes', 268435456), 'Graph metadata exceeded its accounted capacity')
        prior = output_by_agent.setdefault(row['agent'], {})
        prior[row['turn']] = row['expected_output_samples']
        expected_cumulative = prior[0] + (prior[row['turn']] if row['turn'] else 0)
        require(stats.get('output_samples') == expected_cumulative,
            'Graph retained duplicate replay or future edited-branch output samples')
        if mode in ('collect', 'attention', 'multi'):
            require(stats.get('observed_sources', 0) > 0 and stats.get('prefill_samples', 0) > 0 and
                stats.get('output_samples', 0) > 0, 'Graph did not observe both real prefill and output sources')
        reports.append(dict(agent=row['agent'], turn=row['turn'], query=query_graph,
            terminal=stats, accepted_output_oracle=row['expected_output_samples']))
    selection_changed = any(p['query'].get('added_blocks', 0) > 0 for p in reports)
    return dict(mode=mode, accepted_sample_oracle_checked=bool(kv.get('action_attention')),
        graph_terminals=reports, rejected_tokens=rejection_total,
        mtp_rejections_observed=rejection_total > 0,
        selection_changed_observed=selection_changed if mode in SCORING_MODES else None,
        complete=rejection_total > 0 and (mode not in SCORING_MODES or selection_changed),
        limitation='Rejection exclusion follows the independent accepted-token parser oracle plus '
            'terminal graph counts; no proposed token is counted as observed output. A false '
            'mtp_rejections_observed or selection_changed_observed needs another coverage fixture.')


def wait_terminal_publication(capture, rows, timeout=120):
    """HTTP completion is not proof that fenced graph/checkpoint retirement ran."""
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        path = capture / 'archive.json'
        if path.exists():
            archive = json.loads(path.read_text())
            completed = archive.get('completed', [])
            if all(any(native_matches(row, event) for event in completed) for row in rows):
                return archive
        time.sleep(0.1)
    raise TimeoutError('Public streams completed without all fenced archive terminal records')


def compare_collect_baseline(baseline_directory, collect_directory):
    """Only first-turn comparisons have identical inputs across closed-loop arms."""
    baseline = json.loads((baseline_directory / 'results.json').read_text())
    collected = json.loads((collect_directory / 'results.json').read_text())
    require(baseline.get('arm') == 'baseline' and collected.get('arm') == 'collect',
        'Expected baseline then collect result directories')
    require(baseline.get('fixture_sha256') == collected.get('fixture_sha256'),
        'Baseline and collect fixtures differ')
    configs = [json.loads((directory / 'profile.json').read_text())
        for directory in (baseline_directory, collect_directory)]
    for cfg in configs:
        cfg.pop('port', None)
        cfg.pop('served_model_name', None)
        for key in tuple(cfg['kvmem']):
            if key.startswith('graph_'):
                del cfg['kvmem'][key]
    require(configs[0] == configs[1], 'Baseline/collect changed non-graph model or inference settings')
    def selections(result):
        values = {}
        for proof in result['evidence']:
            if proof['turn'] != 0:
                continue
            ranges = proof['query'].get('selected_history_ranges')
            require(isinstance(ranges, list) and bool(ranges), 'Missing exact selected logical ranges')
            values[proof['agent']] = ranges
        return values
    old, new = selections(baseline), selections(collected)
    require(old and old == new, 'Collect changed first-turn selection for identical recorded input')
    return dict(passed=True, clients=len(old), first_turn_only=True,
        selected_history_ranges=old, fixture_sha256=baseline['fixture_sha256'])


def summarize_steps(capture, begin, end):
    steps = [e for e in records(capture / 'model_step_latency.jsonl') if begin <= e['started_at'] <= end]
    pure_prefill = [e for e in steps if e.get('prefill_tokens', 0) > 0 and not e.get('decode_requests', 0)]
    pure_decode = [e for e in steps if e.get('decode_requests', 0) > 0 and not e.get('prefill_tokens', 0)]
    prefill_seconds = sum(e.get('model_ms', 0) for e in pure_prefill)/1000
    return dict(observed_steps=len(steps), pure_prefill_steps=len(pure_prefill), pure_decode_steps=len(pure_decode),
        pure_prefill_model_tokens_per_second=(sum(e['prefill_tokens'] for e in pure_prefill)/prefill_seconds
            if prefill_seconds else None),
        pure_decode_batch_histogram={str(n): sum(e['decode_requests'] == n for e in pure_decode)
            for n in sorted({e['decode_requests'] for e in pure_decode})},
        mixed_steps=sum(e.get('prefill_tokens', 0) > 0 and e.get('decode_requests', 0) > 0 for e in steps),
        scope='Pure-prefill rate divides executed public prefill tokens by corresponding model-step GPU time; '
              'includes replay work and is not end-to-end input throughput. Pure-decode token rate is not '
              'inferred from speculative proposals or request counts.')


def wait_released(capture, sessions, timeout=120):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        path = capture / 'scheduler_resources.json'
        if path.exists():
            state = json.loads(path.read_text())
            views = state.get('session_views')
            if isinstance(views, list) and not any(
                    len(e.get('namespace', [])) == 2 and e['namespace'][1] in sessions for e in views):
                archive_path = capture / 'archive.json'
                if state.get('requests') == 0 and archive_path.exists():
                    archive = json.loads(archive_path.read_text())
                    cleared = ('session_checkpoints', 'session_checkpoint_payload_bytes',
                        'cpu_authority_bytes', 'retiring_requests', 'session_retiring_requests', 'pending_downloads')
                    if 'block_graph_bytes' in archive:
                        cleared += ('block_graph_bytes',)
                    if all(type(archive.get(k)) is int and archive[k] == 0 for k in cleared):
                        return dict(resources=state, archive=archive)
        time.sleep(0.1)
    raise TimeoutError('Owned qualification sessions did not retire after explicit release')


def run(args):
    profile = json.loads(args.config.read_text())
    validate_profile(profile, args.arm, args.concurrency, args.output_tokens, args.min_output_tokens)
    require(not args.output.exists(), 'Use a fresh output directory')
    args.output.mkdir(parents=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    if args.fixture and args.fixture.exists():
        fixture = json.loads(args.fixture.read_text())
    else:
        fixture = build_fixture(tokenizer, prompt_tokens=args.prompt_tokens, concurrency=args.concurrency)
        if args.fixture:
            require(not args.fixture.exists(), 'Refuse to replace a frozen fixture')
            save(args.fixture, fixture)
    validate_fixture(fixture, profile, args.concurrency, args.output_tokens)
    save(args.output / 'fixture.json', fixture)
    save(args.output / 'profile.json', profile)
    sessions = [f'block-graph-{args.arm}-{i}-{uuid.uuid4().hex}' for i in range(args.concurrency)]
    rows, failures, release_rows, result = [], [], [], None
    started = time.monotonic()

    def client(index):
        agent = fixture['agents'][index]
        first = agent['initial_tokens']
        prompt, query, spans = first, agent['query_span'], agent['message_spans']
        for turn in range(3):
            row = stream_request(profile, sessions[index], prompt, args.output_tokens,
                args.min_output_tokens, query=query, spans=spans, timeout=args.timeout)
            row.update(agent=index, turn=turn, expected_code=agent['expected_codes'][turn],
                correct=agent['expected_codes'][turn] in row['text'])
            if profile['kvmem'].get('action_attention'):
                row['expected_output_samples'] = expected_output_samples(tokenizer, profile, prompt, row['output_ids'])
            rows.append(row)
            save(args.output / f'agent-{index}-turn-{turn}.json', row)
            if turn == 0:
                prompt, query, spans, before_question = next_turn(agent, first, row['output_ids'])
            elif turn == 1:
                prompt, query, spans = edited_turn(agent, before_question, spans)

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(client, i) for i in range(args.concurrency)]
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    failures.append(f'{type(error).__name__}: {error}')
        ended = time.monotonic()
        require(not failures, '; '.join(failures))
        archive = wait_terminal_publication(args.capture, rows)
        evidence = event_evidence(args.capture, rows, profile)
        checks = assert_real_graph_checks(rows, evidence, records(args.capture / 'action_attention.jsonl'),
            archive.get('completed', []), profile)
        result = dict(status='passed', arm=args.arm, concurrency=args.concurrency,
            fixture_sha256=fixture['sha256'], seconds=ended-started,
            output_tokens=sum(len(r['output_ids']) for r in rows),
            end_to_end_output_tokens_per_second=sum(len(r['output_ids']) for r in rows)/(ended-started),
            quality=dict(correct=sum(r['correct'] for r in rows), total=len(rows),
                criterion='Exact expected code occurs in the model answer; narrow synthetic dependency task only'),
            step_metrics=summarize_steps(args.capture, started, ended), evidence=evidence, graph_checks=checks,
            scope='Two or more closed-loop token-prefix sessions with forced Query, tool continuation and '
                  'suffix edit. Only first-turn inputs are identical across arms; generated continuations '
                  'intentionally differ. Per-request generation-period rates include scheduling interference '
                  'and are not pure decode kernel throughput. Unobserved coverage is reported explicitly '
                  'by graph_checks.complete.')
    except BaseException as error:
        failures.append(f'{type(error).__name__}: {error}')
        raise
    finally:
        for session in sessions:
            try:
                release_rows.append(stream_request(profile, session, [fixture['eos_token_id']], 1, 0,
                    release=True, timeout=args.timeout))
            except Exception as error:
                failures.append(f'Release {session}: {type(error).__name__}: {error}')
        release_state = None
        if len(release_rows) == len(sessions):
            try:
                release_state = wait_released(args.capture, set(sessions))
            except Exception as error:
                failures.append(f'Retirement: {type(error).__name__}: {error}')
        save(args.output / 'cleanup.json', dict(releases=release_rows,
            all_owned_sessions_released=release_state is not None, state=release_state, errors=failures))
        save(args.output / 'traces.json', sorted(rows, key=lambda r: (r['agent'], r['turn'])))
        if result is not None:
            result.update(status='failed' if failures else 'passed' if result['graph_checks']['complete'] else 'incomplete', errors=failures,
                all_owned_sessions_released=release_state is not None)
            save(args.output / 'results.json', result)
        else:
            save(args.output / 'results.json', dict(status='failed', arm=args.arm, errors=failures))
    require(not failures, '; '.join(failures))
    print(json.dumps({k: result[k] for k in ('status', 'arm', 'seconds', 'output_tokens', 'quality')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, help='Create once and reuse unchanged across all six arms')
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--prompt-tokens', type=int, default=12288)
    parser.add_argument('--output-tokens', type=int, default=257)
    parser.add_argument('--min-output-tokens', type=int, default=129)
    parser.add_argument('--timeout', type=float, default=600)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

"""Fixed structured thinking fixture against an already owned idle server.

This gate is excluded from Terminal-Bench tasks, deadlines, scores and timings.
It requires real accepted-token classification and native redundancy retirement.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def snapshot(capture):
    return {name: read(capture/(name+'.json')) for name in ('progress', 'archive', 'scheduler_resources')}


def clean(value):
    state, archive = value['scheduler_resources'], value['archive']
    scheduler_keys = ('requests', 'owned_blocks', 'refresh_pending', 'connector_pending', 'reservations', 'session_owners')
    archive_keys = ('cpu_authority_bytes', 'cpu_index_bytes', 'cpu_index_owners', 'archive_used_slots',
                    'active_requests', 'retiring_requests', 'session_retiring_requests')
    return bool(set(scheduler_keys).issubset(state) and set(archive_keys).issubset(archive)
        and not any(state[key] for key in scheduler_keys)
        and not any(archive[key] for key in archive_keys)
        and (archive.get('historical_gc') or {}).get('active_scopes') == 0)


def preflight_plan(profile, target_tokens=None):
    """Bound the fixture by this profile and cross its actual GC hysteresis.

    The first prompt triggers GC. The next two observations must together
    refill more than the low/high watermark gap so turn three proves a second
    retirement cycle. Keep the previously qualified small fixture unchanged.
    """
    kv = profile['kvmem']
    gc = kv.get('historical_gc') or {}
    if (gc.get('mode') != 'delete' or gc.get('policy') != 'redundancy' or
            not gc.get('policy_params', {}).get('thinking_only')):
        raise ValueError('Thinking preflight requires the actual thinking-only delete profile')
    blocks, block = kv['repository_blocks'], kv['block_tokens']
    low, high = gc['ready_low'], gc['ready_high']
    if (any(type(v) is not int for v in (blocks, block, low, high)) or
            blocks <= 0 or block <= 0 or not 0 < low < high <= blocks):
        raise ValueError('Preflight requires valid repository capacity and GC watermarks')
    capacity = blocks * block
    pressure = (blocks-low) * block
    gap = (high-low) * block
    feedback = max(3072, ((gap+2048+2*block-1)//(2*block))*block)
    if target_tokens is None:
        target_tokens = max(18432, pressure+2048)
    # Allow a profile-sized initial history, never an arbitrary logical-length
    # prompt. The old 64K ceiling remains sufficient for small-capacity gates.
    upper = max(65536, capacity+2048)
    if type(target_tokens) is not int or not pressure+1024 <= target_tokens <= upper:
        raise ValueError(f'Preflight length must cross history pressure and stay within '
                         f'[{pressure+1024}, {upper}] tokens for this profile')
    # Account for the accepted output and message-template/rounding slack as
    # well as both pressure observations and the final short continuation.
    logical_bound = target_tokens + 2*feedback + 32 + 4*64 + 1024
    logical_limit = profile.get('max_model_len')
    if logical_limit is not None and (type(logical_limit) is not int or
                                     logical_limit < logical_bound):
        raise ValueError('Logical context limit cannot accommodate the four-turn preflight')
    return dict(policy='accepted-thinking-native-redundancy-multiturn-v3',
        target_tokens=target_tokens, turns=4, accepted_tokens_per_turn=64,
        native_requests_per_turn=2, feedback_tokens=[feedback, feedback, 32],
        history_capacity_tokens=capacity, history_pressure_tokens=pressure,
        history_hysteresis_tokens=gap, initial_target_upper_bound=upper,
        logical_tokens_upper_bound=logical_bound,
        continuation='real Planner commit; stable task Query; M restore after repeated pressure')


def fixture(profile, protocol, target_tokens, *, planner=None):
    from transformers import AutoTokenizer
    from benchmarks.deepswe.claude_bridge import Planner, Protocol, Trajectory
    from scripts.dsh_bridge import to_anthropic
    from scripts.dsh_query_epoch import prepare_with_query_provenance
    from scripts.terminalbench_query import TaskQuery
    from vllm_kvmem.serving.thinking_history import thinking_token_layout
    from vllm_kvmem import _native
    if planner is None:
        tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
        planner = Planner(Protocol(protocol), tokenizer, profile['kvmem']['gen_budget'])
    tokenizer = planner.tokenizer
    question = 'Verify repeatedly that adding zero leaves the integer seven unchanged. Keep every intermediate check consistent.'
    unit = 'The integer is seven. Adding zero leaves seven unchanged. The same invariant holds again.\n'
    repeats = max(1, target_tokens//len(planner.encode(unit)))
    source = TaskQuery('thinking-gc-preflight-fixture', 'fixed-invariant', question, question)
    source.proof([dict(role='user', content=question)])
    for _ in range(4):
        body = dict(model=profile['served_model_name'], prompt_cache_key='thinking-gc-preflight-fixture',
            messages=[dict(role='user', content=question),
                dict(role='assistant', reasoning_content=unit*repeats, content='The invariant is unchanged.'),
                dict(role='user', content='The checks are recorded. Continue the same verification.')])
        body['kvmem_query_provenance'] = source.proof(body['messages'])
        prepared, _ = prepare_with_query_provenance(planner, to_anthropic(body), Trajectory(), body, required=True)
        delta = target_tokens-len(prepared[0])
        if abs(delta) <= len(planner.encode(unit)):
            break
        repeats = max(1, repeats+round(delta/len(planner.encode(unit))))
    tokens, plan = prepared[:2]
    plan.update(kvmem_mandatory_parts='none', kvmem_mandatory_spans=[])
    provenance = plan['kvmem_locality_provenance']
    if (tokens != planner.encode(prepared[2]['prompt']) or provenance['source'] != 'deepswe_protocol'
            or provenance['unknown_spans'] or not plan['kvmem_query_span'][1] < 512):
        raise ValueError('Preflight lacks exact structured tokens and an early task Query')
    layout = thinking_token_layout(tokenizer)
    if layout is None:
        raise ValueError('Actual tokenizer cannot classify thinking')
    labels = _native.HistoricalThinkingParser(*layout).feed(tokens)
    if labels.count(2) < target_tokens*0.9:
        raise ValueError('Fixture did not render a long real assistant thinking message')
    return dict(tokens=tokens, plan=plan, messages=body['messages'], wire_proof=body['kvmem_query_provenance'],
        prompt_tokens=len(tokens), expected_thinking_tokens=labels.count(2),
        input_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        purpose='Fixed mechanism gate; excluded from benchmark scores and throughput')


def verify(before, observations):
    names = ('thinking_classified', 'thinking_pure', 'redundancy_retired', 'lru_fallback_retired',
             'lru_representative_retired', 'policy_failures', 'retired', 'cpu_released', 'slots_released')
    baseline = before['archive']['historical_gc']
    counters = [v.get('historical_gc') or {} for row in observations for v in (row['progress'], row['archive'])]
    totals = {name: max([baseline.get(name, 0), *(v.get(name, 0) for v in counters)]) for name in names}
    delta = {name: totals[name]-baseline.get(name, 0) for name in names}
    capture = [row['progress'].get('thinking_capture') or {} for row in observations]
    query_capture = [row['progress'].get('redundancy_q_capture') or {} for row in [before, *observations]]
    overflow = max((row['overflow_packets'] for row in query_capture if 'overflow_packets' in row), default=None)
    accepted = max((row.get('accepted_rows', 0) for row in capture), default=0) - \
        before['progress'].get('thinking_capture', {}).get('accepted_rows', 0)
    supported = any(row.get('enabled') and row.get('layout_supported') for row in capture)
    return dict(passed=bool(supported and accepted > 0 and delta['thinking_pure'] > 0 and
                           delta['thinking_classified'] > 0 and delta['redundancy_retired'] > 0 and
                           totals['policy_failures'] == 0 and overflow == 0),
                accepted_rows=accepted, layout_supported=supported, query_overflow_packets=overflow,
                native_delta=delta, native_totals=totals)


class Conversation:
    """A real bridge trajectory; only actual returned IDs enter commit()."""
    def __init__(self, profile, protocol, target_tokens):
        from transformers import AutoTokenizer
        from benchmarks.deepswe.claude_bridge import Planner, Protocol, Trajectory
        from scripts.terminalbench_query import TaskQuery
        tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
        self.planner = Planner(Protocol(protocol), tokenizer, profile['kvmem']['gen_budget'])
        self.initial = fixture(profile, protocol, target_tokens, planner=self.planner)
        self.body = dict(model=profile['served_model_name'], prompt_cache_key='thinking-gc-preflight-fixture',
                         messages=list(self.initial['messages']))
        question = self.body['messages'][0]['content']
        self.source = TaskQuery(self.body['prompt_cache_key'], 'fixed-invariant', question, question)
        self.source.proof(self.body['messages'][:1])
        self.state = Trajectory()
        self.end = self.planner.end

    def prepare(self):
        from scripts.dsh_bridge import to_anthropic
        from scripts.dsh_query_epoch import prepare_with_query_provenance
        self.payload = to_anthropic(self.body)
        self.body['kvmem_query_provenance'] = self.source.proof(self.body['messages'])
        self.prepared, source = prepare_with_query_provenance(
            self.planner, self.payload, self.state, self.body, required=True)
        tokens, plan = self.prepared[:2]
        plan.update(kvmem_mandatory_parts='none', kvmem_mandatory_spans=[])
        return dict(tokens=tokens, plan=plan, exact_output_prefix_reused=self.prepared[5],
                    wire_proof=self.body['kvmem_query_provenance'], source=source)

    def commit(self, accepted):
        from scripts.dsh_bridge import to_openai
        if len(accepted) != 64 or accepted[-1] != self.end:
            raise ValueError('A preflight turn needs exactly 64 actual accepted IDs ending in EOS')
        raw = self.planner.tokenizer.decode(accepted, skip_special_tokens=False)
        response = self.planner.response(raw, self.payload.get('tools', []), self.body['model'],
            dict(prompt_tokens=len(self.prepared[0]), completion_tokens=len(accepted)), 'stop')
        self.planner.commit(self.state, self.prepared, self.payload, response, list(accepted))
        self.body['messages'].append(to_openai(response)['choices'][0]['message'])
        return response

    def feedback(self, target_tokens):
        # Like Harbor, the trusted tool-loop producer transports its observation
        # as a user message. It does not call TaskQuery.real_user(). No official
        # task or actual user message is changed by this diagnostic fixture.
        unit = 'Tool observation: seven plus zero equals seven; verification succeeded.\n'
        count = max(1, target_tokens // len(self.planner.encode(unit)))
        self.body['messages'].append(dict(role='user', content=unit * count))


def capture_offsets(capture):
    return {name: (capture/name).stat().st_size if (capture/name).exists() else 0
            for name in ('async_restore.jsonl', 'scheduler_refresh.jsonl')}


def capture_evidence(capture, offsets, records):
    result = {name: [] for name in offsets}
    for name, offset in offsets.items():
        path = capture/name
        if not path.exists():
            continue
        with path.open('rb') as stream:
            stream.seek(offset)
            for line in stream:
                if not line.endswith(b'\n'):
                    continue  # A concurrent writer may not have finished its last line.
                value = json.loads(line)
                match = next((r for r in records if value.get('request_id', '').startswith(r['request_id']+'-')), None)
                if match:
                    result[name].append(dict(turn=match['turn'], part=match['part'], **value))
    return result


def historical(row):
    return row['archive'].get('historical_gc') or {}


def verify_multiturn(turns, records, observations, evidence):
    ready = [historical(r['after']).get('active_features') or {} for r in records]
    epochs = {f.get('question_epoch') for f in ready}
    invalid = max((historical(row).get('active_features', {}).get('invalid_questions', 0)
                   for row in observations), default=0)
    identity = {(t['plan'].get('kvmem_query_id'), tuple(t['plan'].get('kvmem_query_span', []))) for t in turns}
    restores = evidence['async_restore.jsonl']
    expected, old_prefix_replays = [], []
    for previous, current in zip(records, records[1:]):
        matching = [r for r in restores if r['request_id'].startswith(current['request_id']+'-')]
        terminal = previous.get('terminal') or {}
        # scheduler_refresh also records normal K/G pressure maintenance.
        # Stable Query continuity forbids replay before the saved M; it does
        # not forbid selecting a new window after the new suffix fills G.
        if terminal.get('checkpoint_kind') == 'M':
            for row in evidence['scheduler_refresh.jsonl']:
                if (row.get('request_id', '').startswith(current['request_id']+'-') and
                        row.get('history_begin', -1) < terminal.get('history_end', 0)):
                    old_prefix_replays.append(row)
        expected.append(dict(turn=current['turn'], part=current['part'],
            checkpoint_kind=terminal.get('checkpoint_kind'), expected_history=terminal.get('history_end'),
            restored_history=[r.get('history_tokens') for r in matching],
            passed=bool(terminal.get('checkpoint_kind') == 'M' and len(matching) == 1 and
                        not matching[0].get('cancelled') and
                        matching[0].get('history_tokens') == terminal.get('history_end'))))
    second_pressure = (historical(turns[-2]['after']).get('redundancy_retired', 0) -
                       historical(turns[0]['after']).get('redundancy_retired', 0)) if len(turns) == 4 else 0
    checks = dict(four_committed_turns=len(turns) == 4 and len(records) == 8,
        exact_prefix_reused=len(turns) == 4 and all(t['exact_output_prefix_reused'] for t in turns[1:]),
        stable_query_source=len(identity) == 1 and all(t['plan'].get('kvmem_query_mode') == 'agent' for t in turns),
        question_ready_reused=bool(ready and all(f.get('question_ready') == 1 for f in ready)),
        question_epoch_stable=len(epochs) == 1 and None not in epochs and 0 not in epochs,
        no_invalid_questions=invalid == 0,
        m_restores_completed=len(expected) == 7 and all(row['passed'] for row in expected),
        renewed_redundancy_pressure=second_pressure > 0,
        no_query_replay_before_checkpoint=not old_prefix_replays)
    return dict(passed=all(checks.values()), checks=checks, question_epochs=sorted(epochs, key=str),
                invalid_questions=invalid, second_pressure_redundancy_retired=second_pressure,
                restores=expected, request_features=ready, old_prefix_replays=old_prefix_replays,
                pressure_refresh_events=len(evidence['scheduler_refresh.jsonl']))


async def exercise(base, profile, protocol, capture, output, target_tokens=None):
    schedule = preflight_plan(profile, target_tokens)
    if output.exists():
        raise ValueError('Use a new preflight evidence directory')
    output.mkdir(parents=True)
    save(output/'preflight-plan.json', schedule)
    conversation = Conversation(profile, protocol, schedule['target_tokens'])
    save(output/'fixture.json', conversation.initial)
    deadline = time.monotonic()+90
    while not clean(snapshot(capture)) and time.monotonic() < deadline:
        await asyncio.sleep(.05)
    before = snapshot(capture)
    save(output/'before.json', before)
    if not clean(before):
        raise ValueError('Preflight requires a drained server with no other history owners')
    offsets = capture_offsets(capture)
    session = 'thinking-gc-preflight-'+uuid.uuid4().hex
    observations, records, turns, release_error, request_error = [], [], [], None, None
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=base, trust_env=False, timeout=900) as client:
        async def call(body, turn, part):
            path = output / f'turn-{turn}-{part}'
            path.mkdir()
            save(path/'request.json', body)
            job = asyncio.create_task(client.post('/v1/completions', json=body))
            try:
                while not job.done():
                    observations.append(snapshot(capture))
                    await asyncio.wait({job}, timeout=.05)
                response = await job
                save(path/'http.json', dict(status=response.status_code, body=response.text))
                response.raise_for_status()
                reply = response.json()
                save(path/'response.json', reply)
                accepted = reply['choices'][0]['token_ids']
                if accepted != body['allowed_token_ids'] * body['max_tokens']:
                    raise ValueError('Preflight output differs from its fixed actual-acceptance contract')
                deadline = time.monotonic()+15
                while True:
                    after = snapshot(capture)
                    terminal = next((r for r in after['archive'].get('completed', [])
                                     if r.get('request_id', '').startswith(reply['id']+'-')), None)
                    if terminal is not None or time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(.05)
                if terminal is None:
                    raise ValueError('Native completion/checkpoint report is missing for this exact request')
                observations.append(after)
                record = dict(turn=turn, part=part, request_id=reply['id'], prompt_tokens=len(body['prompt']),
                    accepted=accepted, terminal=terminal, after=after)
                records.append(record)
                save(path/'native-after.json', after)
                return accepted, after
            finally:
                if not job.done():
                    job.cancel()
                await asyncio.gather(job, return_exceptions=True)

        try:
            for index in range(4):
                if index:
                    conversation.feedback(schedule['feedback_tokens'][index-1])
                prepared = conversation.prepare()
                if index and not prepared['exact_output_prefix_reused']:
                    raise ValueError('Real Planner declined exact accepted-prefix continuation')
                tokens, plan = list(prepared['tokens']), prepared['plan']
                body = dict(model=profile['served_model_name'], session_id=session, cache_salt=session,
                    prompt=tokens, **plan, max_tokens=63, temperature=0, seed=1000,
                    allowed_token_ids=[830], ignore_eos=True, return_token_ids=True)
                accepted, _ = await call(body, index+1, 'thinking')
                # EOS is also genuinely sampled and returned, never appended as
                # invented history. The two public native calls form one bridge
                # turn of 64 actual accepted IDs. The extra restore strengthens
                # coverage of the idle-session transition.
                ending, after = await call(dict(body, prompt=tokens+accepted, max_tokens=1,
                    allowed_token_ids=[conversation.end]), index+1, 'eos')
                accepted += ending
                response = conversation.commit(accepted)
                turns.append(dict(turn=index+1, plan=plan, prompt_tokens=len(tokens), accepted=accepted,
                    exact_output_prefix_reused=prepared['exact_output_prefix_reused'],
                    wire_proof=prepared['wire_proof'], response=response, after=after))
                save(output/'turns.json', turns)
                if index == 0:
                    save(output/'single-turn-result.json', verify(before, observations))
        except Exception as error:
            request_error = repr(error)
        finally:
            save(output/'observations.json', observations)
            save(output/'requests.json', records)
            try:
                release = await client.post('/v1/completions', json=dict(model=profile['served_model_name'],
                    session_id=session, cache_salt=session, prompt=[conversation.end], max_tokens=1,
                    kv_transfer_params=dict(kvmem_release_session=True)))
                save(output/'release-http.json', dict(status=release.status_code, body=release.text))
                release.raise_for_status()
                save(output/'release.json', release.json())
            except Exception as error:
                release_error = repr(error)
            deadline = time.monotonic()+90
            while not clean(snapshot(capture)) and time.monotonic() < deadline:
                await asyncio.sleep(.05)
            after = snapshot(capture)
            save(output/'after-release.json', after)
            try:
                evidence = capture_evidence(capture, offsets, records)
            except Exception as error:
                request_error = request_error or 'Native evidence: ' + repr(error)
                evidence = {name: [] for name in offsets}
            save(output/'native-evidence.json', evidence)
            verdict = verify(before, [*observations, after])
            multi = verify_multiturn(turns, records, observations, evidence)
            verdict.update(clean_release=clean(after) and release_error is None, release_error=release_error,
                           request_error=request_error, seconds=time.perf_counter()-started, multi_turn=multi)
            verdict['passed'] = bool(verdict['passed'] and verdict['clean_release'] and multi['passed'] and request_error is None)
            save(output/'result.json', verdict)
    if not verdict['passed']:
        raise RuntimeError('Thinking GC preflight failed; inspect result.json before starting official tasks')
    print(json.dumps(verdict, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-tokens', type=int, default=None,
                        help='Initial fixture size; defaults to the configured GC pressure plus slack')
    args = parser.parse_args()
    asyncio.run(exercise(args.base, read(args.profile), args.protocol, args.capture, args.output, args.target_tokens))

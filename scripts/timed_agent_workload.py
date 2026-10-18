"""Fixed-window, streaming agent trace workload; no server/GPU management.

The external tool schedule is fixed, while every continuation appends its own
accepted output IDs. Finite, declared epochs end normally before their histories
are released; there is no capacity-driven history truncation. A fixed output
length is the default. Optional natural stopping uses the real bridge sampling
settings and a declared generation ceiling. Both are performance replays of
external tool messages, not on-policy DeepSWE quality evaluations.
"""
import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def validate_agent_plans(fixture, *, concurrency, warmup_turns, epoch_turns, gen_budget,
                         enforce_output_budget=True):
    """Validate frozen external traces without changing IDs, plans or quotas."""
    from vllm_kvmem.request_plan import FIELDS, RequestPlan
    version = fixture.get('schema_version', 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError('Unsupported frozen agent trace schema version')
    for name, value in (('concurrency', concurrency), ('warmup_turns', warmup_turns),
                        ('epoch_turns', epoch_turns)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + ' must be a positive integer')
    if warmup_turns >= epoch_turns:
        raise ValueError('Warmup must leave work in its initial epoch')
    if gen_budget is not None and (type(gen_budget) is not int or gen_budget <= 0):
        raise ValueError('gen_budget must be a positive integer when configured')
    plans = fixture.get('agent_plans')
    if not isinstance(plans, list) or len(plans) < concurrency:
        raise ValueError('Insufficient per-user agent_plans')
    agent_fields = {'initial_tokens', 'query_plan', 'turns', 'source_task', 'source_turn',
                    'initial_tokens_sha256', 'canonical_initial_tokens', 'source_logical_initial_tokens',
                    'skipped_incomplete_candidates'}
    if version == 2:
        agent_fields.add('initial_assistant_begin')
    turn_fields = {'output_tokens', 'tool_tokens', 'tool_wait_seconds', 'source_turn',
                   'source_finish_reason', 'source_logical_prompt_tokens', 'tool_tokens_sha256',
                   'external_message_proof'}
    if version == 2:
        turn_fields.update(('tool_locality_spans', 'tool_locality_provenance', 'assistant_begin'))

    def tokens(value, label, *, nonempty=False):
        if (not isinstance(value, list) or (nonempty and len(value) < 2)
                or any(type(n) is not int or n < 0 for n in value)):
            raise ValueError('Invalid ' + label + ' token IDs')

    for index, plan in enumerate(plans):
        if (not isinstance(plan, dict) or set(plan) - agent_fields or
                not {'initial_tokens', 'query_plan', 'turns'} <= plan.keys()):
            raise ValueError(f'Invalid agent_plans[{index}] schema')
        initial = plan['initial_tokens']
        tokens(initial, 'initial', nonempty=True)
        if 'initial_tokens_sha256' in plan and plan['initial_tokens_sha256'] != digest(initial):
            raise ValueError('Initial token fingerprint differs from frozen trace')
        if 'canonical_initial_tokens' in plan and plan['canonical_initial_tokens'] != len(initial):
            raise ValueError('Canonical initial length differs from token IDs')
        query = plan['query_plan']
        if not isinstance(query, dict) or set(query) - FIELDS:
            raise ValueError('Unknown query_plan fields; HTTP overrides are forbidden')
        if not {'kvmem_query_span', 'kvmem_query_message_range'} & query.keys():
            raise ValueError('External trace requires its explicit initial Query')
        # This is the same semantic validator used by the source Planner. It
        # also checks mandatory spans, replay containment and mode conflicts.
        RequestPlan.parse(query, len(initial), 32, gen_budget or 131072)
        header = plan.get('initial_assistant_begin')
        if header is not None and (type(header) is not int or not 0 <= header < len(initial)):
            raise ValueError('Invalid initial assistant header coordinate')
        turns = plan['turns']
        if not isinstance(turns, list) or len(turns) < epoch_turns:
            raise ValueError('Insufficient complete turns for the declared epoch')
        for turn, row in enumerate(turns):
            if (not isinstance(row, dict) or set(row) - turn_fields or
                    not {'output_tokens', 'tool_tokens', 'tool_wait_seconds'} <= row.keys()):
                raise ValueError('Invalid external turn schema')
            count = row['output_tokens']
            if type(count) is not int or count <= 0:
                raise ValueError('output_tokens must be a positive integer')
            if enforce_output_budget and gen_budget is not None and count > gen_budget:
                raise ValueError('External output length exceeds configured gen_budget')
            tokens(row['tool_tokens'], 'external tool')
            if version == 2:
                from vllm_kvmem.prompt_spans import normalize_spans, normalize_provenance
                normalize_spans(row.get('tool_locality_spans', []), len(row['tool_tokens']))
                if row.get('tool_locality_provenance'):
                    normalize_provenance(row['tool_locality_provenance'])
                header = row.get('assistant_begin')
                if header is not None and (type(header) is not int or not 0 <= header < len(row['tool_tokens'])):
                    raise ValueError('Invalid appended assistant header coordinate')
            wait = row['tool_wait_seconds']
            if type(wait) not in (int, float) or not math.isfinite(wait) or wait < 0:
                raise ValueError('tool_wait_seconds must be finite and nonnegative')
            if turn == 0 and (row['tool_tokens'] or wait != 0):
                raise ValueError('Initial turn must not add tool tokens or a tool wait')
            if 'tool_tokens_sha256' in row and row['tool_tokens_sha256'] != digest(row['tool_tokens']):
                raise ValueError('External tool fingerprint differs from frozen trace')
            if 'source_turn' in row and (type(row['source_turn']) is not int or row['source_turn'] <= 0):
                raise ValueError('source_turn must be a positive integer')
    return plans[:concurrency]


def initial_locality(plan):
    query = plan['query_plan'] if plan is not None else {}
    return dict(locality_spans=list(query.get('kvmem_locality_spans', [])),
                locality_provenance=dict(query.get('kvmem_locality_provenance', {})),
                assistant_begin=plan.get('initial_assistant_begin') if plan is not None else None)


def append_locality(state, turn):
    base = len(state['tokens'])
    state['locality_spans'].extend({**span, 'begin': base + span['begin'], 'end': base + span['end']}
                                  for span in turn.get('tool_locality_spans', []))
    if turn.get('tool_locality_provenance'):
        state['locality_provenance'] = dict(turn['tool_locality_provenance'], history='actual_accepted_ids')
    begin = turn.get('assistant_begin')
    state['assistant_begin'] = None if begin is None else base + begin


def accepted_locality(state, output_ids, eos):
    begin = state.get('assistant_begin')
    if begin is not None and output_ids and output_ids[-1] == eos:
        state['locality_spans'].append(dict(begin=begin, end=len(state['tokens']) + len(output_ids), role='assistant'))


def views_match(state, expected_views):
    """A quiescent snapshot must also describe these completed continuations."""
    for session, history in expected_views.items():
        matching = [view for view in state.get('session_views', [])
                    if view.get('namespace') == [session, session]]
        if len(matching) != 1:
            return False
        view = matching[0]
        window = view.get('window')
        records = view.get('active_records')
        if (view.get('history') != history or type(window) is not int or
                not 0 < window <= history or not isinstance(records, list) or not records):
            return False
        if history == 1 and (window != 1 or records != [[0, 1]]):
            return False
        end = total = 0
        for record in records:
            if (not isinstance(record, list) or len(record) != 2 or
                    any(type(n) is not int for n in record)):
                return False
            start, count = record
            if start < end or count <= 0 or start+count > history:
                return False
            end = start+count
            total += count
        if total != window:
            return False
    return True


async def wait_retirement(capture, timeout=120., *, not_before=0., expected_views=None):
    """Wait for published retirement, retaining revocable idle GPU views."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            state = json.loads((capture / 'scheduler_resources.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            state = None
        if (state and state.get('at', -1) >= not_before
                and views_match(state, expected_views or {}) and all(not state[key] for key in (
                'requests', 'refresh_pending', 'connector_pending',
                'owned_blocks', 'reservations'))):
            cpu = state['cpu_history_admission']
            if not cpu['active'] and not cpu['retiring'] and not cpu['waiting']:
                idle = state.get('active_residency', {}).get('idle_blocks', 0)
                if state['free_blocks'] + idle != state['total_blocks'] - 1:
                    raise RuntimeError('Retired page accounting is not conserved')
                return state
        await asyncio.sleep(.01)
    raise TimeoutError('Timed workload retirement evidence did not become ready')


def summarize(rows, window, phase_events):
    """Use event timestamps, including output from requests finishing after T1."""
    t0, t1 = window['t0_monotonic'], window['t1_monotonic']
    measured = [r for r in rows if r['kind'] == 'workload']
    offered = [r for r in measured if t0 <= r['begin'] < t1]
    count = lambda r: sum(n for t, n in r['timeline'] if t0 <= t < t1)
    effective = sum(map(count, measured))
    cleanup = sum(count(r) for r in rows if r['kind'] == 'cleanup')
    gaps = [b[0] - a[0] for r in measured
            for a, b in zip(r['timeline'], r['timeline'][1:])
            if t0 <= a[0] <= b[0] < t1]
    ttft = []
    censored = []
    for row in offered:
        first = row['timeline'][0][0] if row['timeline'] else None
        if first is not None and first < t1:
            ttft.append(first - row['begin'])
        else:
            censored.append({'request_id': row['request_id'],
                             'at_least_seconds': t1 - row['begin']})
    pending = [dict(request_id=r['request_id'], kind=r['kind'], agent=r['agent'],
                    elapsed_seconds=t1-r['begin'], output_tokens_before_deadline=count(r))
               for r in rows if r['begin'] < t1 and (r.get('end') is None or r['end'] >= t1)]
    phases = {}
    for event in phase_events:
        if event['monotonic'] < t1:
            phases[str(event['agent'])] = {k: event[k] for k in ('phase', 'epoch', 'turn')}
    per_agent = {}
    for row in offered:
        item = per_agent.setdefault(str(row['agent']), {'started_requests': 0,
                                    'completed_requests': 0, 'output_tokens': 0})
        item['started_requests'] += 1
        item['completed_requests'] += int(row.get('end') is not None and row['end'] < t1
                                          and row['status'] == 'completed')
        item['output_tokens'] += count(row)
    return dict(
        status='completed' if all(r['status'] == 'completed' for r in rows) else 'failed',
        scope='Client accepted-token events in [T0,T1); fixed-length synthetic agent replay',
        duration_seconds=t1-t0, output_tokens=effective,
        output_tokens_per_second=effective/(t1-t0),
        cleanup_output_tokens_in_window=cleanup,
        all_stream_output_tokens_in_window=sum(map(count, rows)),
        total_stream_output_tokens=sum(len(r['output_ids']) for r in rows),
        stream_output_tokens_by_kind={kind: sum(len(r['output_ids']) for r in rows if r['kind'] == kind)
                                      for kind in ('warmup', 'workload', 'cleanup')},
        requests_started_in_window=len(offered),
        requests_completed_in_window=sum(r.get('end') is not None and r['end'] < t1
                                        and r['status'] == 'completed' for r in offered),
        offered_new_input_tokens=sum(r['new_input_tokens'] for r in offered),
        offered_input_scope='External new input, not measured kernel prefill or KV restoration',
        output_event_gap_p50_seconds=percentile(gaps, .5),
        output_event_gap_p95_seconds=percentile(gaps, .95),
        output_event_gap_samples=len(gaps),
        output_event_gap_scope='Both events inside window; MTP events may contain several tokens',
        output_gap_right_censored=[dict(request_id=r['request_id'],
                                       at_least_seconds=t1-max(t for t, _ in r['timeline'] if t < t1))
                                  for r in measured if r['begin'] < t1
                                  and (r.get('end') is None or r['end'] >= t1)
                                  and any(t0 <= t < t1 for t, _ in r['timeline'])],
        ttft_observed_p50_seconds=percentile(ttft, .5),
        ttft_observed_p95_seconds=percentile(ttft, .95),
        ttft_observed_count=len(ttft), ttft_right_censored=censored,
        ttft_scope='Only first tokens observed before T1 enter quantiles; censored requests listed separately',
        pending_at_deadline=pending, agent_phases_at_deadline=phases,
        client_waiting_first_token_at_deadline=sum(p['phase'] == 'awaiting_first_token' for p in phases.values()),
        client_queue_scope='Client phase only; native admission/ready queue requires server telemetry',
        agent_progress=per_agent)


class Recorder:
    def __init__(self, directory, agents):
        self.directory = directory
        self.stream = (directory / 'events.jsonl').open('w', buffering=1)
        self.rows = []
        self.phase_events = []
        self.agents = {str(i): {'phase': 'initializing', 'epoch': 0, 'turn': 0} for i in range(agents)}
        self.window = None

    def event(self, event, **fields):
        row = dict(event=event, monotonic=time.monotonic(), wall=time.time(), **fields)
        self.stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
        return row

    def phase(self, agent, phase, epoch, turn):
        row = self.event('agent_phase', agent=agent, phase=phase, epoch=epoch, turn=turn)
        self.phase_events.append(row)
        self.agents[str(agent)] = {k: row[k] for k in ('phase', 'epoch', 'turn', 'monotonic')}
        self.progress()

    def progress(self):
        save(self.directory / 'progress.json', dict(
            at=time.time(), agents=self.agents, window=self.window,
            completed_requests=sum(r['status'] == 'completed' for r in self.rows),
            accepted_output_tokens=sum(len(r['output_ids']) for r in self.rows)))


async def stream_call(client, profile, tokens, session, count, recorder, *,
                      agent, epoch, turn, kind, request_id, new_input_tokens, query=True,
                      query_plan=None, source_metadata=None, generation_mode='fixed',
                      release_session=False):
    if generation_mode not in ('fixed','natural'):
        raise ValueError('Unknown generation mode')
    if type(release_session) is not bool or (release_session and
            (kind != 'cleanup' or query or query_plan is not None or count != 1 or
             len(tokens) != 1 or not session)):
        raise ValueError('Explicit session release requires a one-token cleanup without Query')
    # Namespace cleanup is always the original single forced output token.
    generation_mode = 'fixed' if kind == 'cleanup' else generation_mode
    natural = generation_mode == 'natural'
    if natural and (type(count) is not int or count <= 0):
        raise ValueError('Natural generation ceiling must be a positive integer')
    native = profile.get('native_vllm', False)
    payload = dict(model=profile['served_model_name'], prompt=tokens,
                   cache_salt=f'{session}:epoch:{epoch}' if native else session,
                   max_tokens=count, temperature=0, seed=1000,
                   ignore_eos=True, return_token_ids=True, stream=True,
                   stream_options={'include_usage': True},
                   request_id=request_id)
    if natural:
        # Match benchmarks/deepswe/claude_bridge.backend_request. The installed
        # CompletionRequest accepts thinking_token_budget at this top level.
        payload.update(ignore_eos=False,thinking_token_budget=4096,temperature=.6,
            top_p=.95,top_k=20,min_p=0,presence_penalty=0,repetition_penalty=1,
            skip_special_tokens=False)
    if not native:
        payload.update(session_id=session, kvmem_reselect='auto' if query else 'off')
        if query:
            if query_plan is not None:
                payload.update(copy.deepcopy(query_plan))
            else:
                initial_end = min(len(tokens), 32768)
                payload['kvmem_query_span'] = [max(0, initial_end-128), initial_end-1]
        else:
            payload['kvmem_mandatory_spans'] = []
        if release_session:
            payload['kv_transfer_params'] = {'kvmem_release_session': True}
    row = dict(begin=time.monotonic(), end=None, session=session, agent=agent, epoch=epoch,
               turn=turn, kind=kind, request_id=request_id, input_tokens=len(tokens),
               new_input_tokens=new_input_tokens, input_sha256=digest(tokens), output_ids=[],
               timeline=[], response_id=None, usage=None, status='running',
               query_plan=copy.deepcopy(query_plan) if query and not native else None,
               source_metadata=source_metadata,generation_mode=generation_mode,
               declared_generation_ceiling=count,finish_reason=None,
               explicit_session_release=release_session and not native)
    recorder.rows.append(row)
    recorder.event('request_started', **{k: row[k] for k in (
        'request_id', 'kind', 'agent', 'epoch', 'turn', 'input_tokens', 'input_sha256', 'new_input_tokens')})
    if kind == 'workload':
        recorder.phase(agent, 'awaiting_first_token', epoch, turn)
    done = False
    try:
        async with client.stream('POST', f"http://127.0.0.1:{profile['port']}/v1/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                response_id = event.get('id')
                if response_id is not None:
                    if row['response_id'] is not None and row['response_id'] != response_id:
                        raise RuntimeError('Response identity changed inside a stream')
                    row['response_id'] = response_id
                row['usage'] = event.get('usage') or row['usage']
                chunk = []
                for choice in event.get('choices', []):
                    if choice.get('index', 0) != 0:
                        raise RuntimeError('Unexpected multiple completion choices')
                    reason = choice.get('finish_reason')
                    if reason is not None:
                        if natural and row['finish_reason'] not in (None,reason):
                            raise RuntimeError('Finish reason changed inside a natural stream')
                        row['finish_reason'] = reason
                    chunk.extend(choice.get('token_ids') or [])
                timestamp = time.monotonic()
                if chunk:
                    row['timeline'].append([timestamp, len(chunk)])
                    row['output_ids'].extend(chunk)
                recorder.event('stream_chunk', request_id=request_id, response_id=row['response_id'],
                               kind=kind, agent=agent, epoch=epoch, turn=turn,
                               observed_monotonic=timestamp, token_ids=chunk,
                               accepted_tokens=len(chunk), output_offset=len(row['output_ids']),
                               usage=event.get('usage'))
                if chunk and len(row['timeline']) == 1 and kind == 'workload':
                    recorder.phase(agent, 'streaming', epoch, turn)
        usage = row['usage']
        if natural:
            actual = len(row['output_ids'])
            if (not done or not 0 <= actual <= count or not usage or
                    usage.get('completion_tokens') != actual or row['finish_reason'] not in ('stop','length')):
                raise RuntimeError(f'Incomplete natural stream: {actual}/{count}, done={done}, '
                                   f"finish_reason={row['finish_reason']}")
        elif not done or len(row['output_ids']) != count or not usage or usage['completion_tokens'] != count:
            raise RuntimeError(f"Incomplete fixed-length stream: {len(row['output_ids'])}/{count}, done={done}")
        row['status'] = 'completed'
    except BaseException as error:
        row['status'] = 'failed'
        row['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        row['end'] = time.monotonic()
        row['actual_output_tokens'] = len(row['output_ids'])
        row['output_sha256'] = digest(row['output_ids'])
        recorder.event('request_finished', request_id=request_id, kind=kind,
                       status=row['status'], accepted_tokens=len(row['output_ids']),
                       error=row.get('error'))
        save(recorder.directory / f'{request_id}.json', row)
        recorder.progress()
    return row


async def workload(client, profile, fixture, directory, args):
    """Run fixed work warmup, common T0, independent producers, then normal drain.

    Optional args.capture_directory locates server/capture retirement telemetry.
    An explicit native_vllm profile instead requires an async
    args.native_quiescent(*, stage, not_before) callback. The server owner polls
    fresh native running/waiting gauges to zero and returns its native evidence;
    no KVMem resource or session-view snapshot is synthesized in native mode.
    window.json exposes t0_monotonic/t1_monotonic and t0_wall/t1_wall to samplers.
    The caller owns HTTP timeouts, metrics/GPU sampling and server lifetime.
    """
    settings = {name: getattr(args, name, default) for name, default in dict(
        duration_seconds=300., concurrency=4, warmup_turns=2, output_tokens=1024,
        context_tokens=141870, epoch_turns=8, mode='saturated', run_tag='timed',
        generation_mode='fixed',generation_limit=32000).items()}
    for name in ('concurrency', 'warmup_turns', 'output_tokens', 'context_tokens', 'epoch_turns'):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(name + ' must be a positive integer')
    duration = settings['duration_seconds']
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('duration_seconds must be positive and finite')
    if settings['mode'] not in ('saturated', 'tools'):
        raise ValueError('Unknown timed workload mode')
    if settings['generation_mode'] not in ('fixed','natural'):
        raise ValueError('Unknown generation mode')
    natural = settings['generation_mode'] == 'natural'
    native = profile.get('native_vllm', False)
    if type(native) is not bool:
        raise ValueError('native_vllm must be boolean')
    explicit_release = getattr(args, 'explicit_session_release', False)
    if type(explicit_release) is not bool or (native and explicit_release):
        raise ValueError('Explicit session release is a KVMem-only boolean option')
    if explicit_release:
        settings['explicit_session_release'] = True
    native_quiescent = getattr(args, 'native_quiescent', None)
    if native and not callable(native_quiescent):
        raise ValueError('Native workload requires an async native_quiescent callback')
    if native:
        settings.update(native_vllm=True,
            native_scope='Full growing prompts with native prefix caching; no KVMem session, Query or KV retirement',
            native_cache_salt_scope='Per agent and epoch; no cross-epoch initial-prefix reuse')
    if settings['warmup_turns'] >= settings['epoch_turns']:
        raise ValueError('Warmup must leave work in its initial epoch')
    agents = settings['concurrency']
    gen_budget = profile.get('kvmem', {}).get('gen_budget')
    if natural:
        limit, ceiling = settings['generation_limit'], profile.get('max_model_len')
        if type(limit) is not int or limit <= 0 or (gen_budget is not None and limit > gen_budget):
            raise ValueError('Natural generation limit must be positive and within gen_budget')
        if type(ceiling) is not int or ceiling <= limit:
            raise ValueError('Natural generation requires explicit max_model_len above its generation limit')
        if settings['warmup_turns'] < 2:
            raise ValueError('Natural generation requires at least two complete warmup turns')
        settings.update(max_model_len=ceiling,
            generation_sampling=dict(temperature=.6,top_p=.95,top_k=20,min_p=0,
                presence_penalty=0,repetition_penalty=1,thinking_token_budget=4096,seed=1000,
                ignore_eos=False,skip_special_tokens=False),
            epoch_capacity_policy='End a completed finite epoch before next full prompt plus declared '
                'generation ceiling exceeds max_model_len; no history clipping or limit reduction.',
            generation_scope='Natural-stop performance replay with frozen external tool messages; '
                'own accepted output IDs, not on-policy tool execution or task quality.')
    plans = None
    if 'agent_plans' in fixture:
        plans = validate_agent_plans(fixture, concurrency=agents, warmup_turns=settings['warmup_turns'],
                                    epoch_turns=settings['epoch_turns'], gen_budget=gen_budget,
                                    enforce_output_budget=not natural)
        settings.update(fixture_kind='agent_plans',
                        initial_context_tokens=[len(p['initial_tokens']) for p in plans],
                        per_turn_output_tokens=[[r['output_tokens'] for r in p['turns'][:settings['epoch_turns']]]
                                                for p in plans],
                        scalar_context_and_output_arguments='Unused for agent_plans; original per-user IDs and quotas preserved')
        if natural:
            settings['scalar_context_and_output_arguments'] = (
                'Original initial IDs and external tool messages/waits preserved. Recorded output quotas '
                'are provenance only; generation_limit is the unchanged ceiling for each natural turn.')
    else:
        if settings['context_tokens'] > len(fixture['prompt']) or settings['context_tokens'] < 2:
            raise ValueError('Fixture does not contain the requested initial context')
        sizes = fixture['tool_sizes']
        if not sizes or any(type(n) is not int or n <= 0 or n > len(fixture['tool']) for n in sizes):
            raise ValueError('Invalid fixture tool sizes')
        waits = fixture['waits']
        if len(waits) < agents or any(not waits[i] or any(not math.isfinite(w) or w < 0 for w in waits[i])
                                     for i in range(agents)):
            raise ValueError('Invalid fixture tool waits')
        prefixes = fixture.get('user_prefixes')
        if prefixes is not None and (len(prefixes) < agents or
                len({tuple(p) for p in prefixes[:agents]}) != agents or
                max(map(len, prefixes[:agents])) > min(settings['context_tokens'], min(sizes))):
            raise ValueError('Unique user prefixes must fit the prompt and tool fixtures')
        if not natural and gen_budget is not None and settings['output_tokens'] > gen_budget:
            raise ValueError('Output length exceeds configured gen_budget')
        settings['fixture_kind'] = 'synthetic'
    if natural:
        initial_lengths = [len(plan['initial_tokens']) for plan in plans] if plans is not None else [settings['context_tokens']]*agents
        if any(length+settings['generation_limit'] > settings['max_model_len'] for length in initial_lengths):
            raise ValueError('Initial complete history plus natural generation limit exceeds max_model_len')
    if type(fixture.get('eos')) is not int or fixture['eos'] < 0:
        raise ValueError('Fixture requires a nonnegative tokenizer eos ID for cleanup')
    tag = settings['run_tag']
    if not isinstance(tag, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', tag):
        raise ValueError('run_tag must be a safe nonempty identifier')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'events.jsonl').exists():
        raise FileExistsError('Timed workload output already contains events')
    capture = Path(getattr(args, 'capture_directory', directory.parent / 'server' / 'capture'))
    recorder = Recorder(directory, agents)
    states = []
    stop = asyncio.Event()
    settings['fixture_sha256'] = digest(fixture)
    save(directory / 'workload.json', settings)

    def initial(i):
        if plans is not None:
            return list(plans[i]['initial_tokens'])
        tokens = list(fixture['prompt'][:settings['context_tokens']])
        if prefixes is not None:
            tokens[:len(prefixes[i])] = prefixes[i]
        return tokens

    def tool(i, turn):
        if plans is not None:
            return list(plans[i]['turns'][turn]['tool_tokens'])
        result = list(fixture['tool'][:sizes[(turn-1) % len(sizes)]])
        if prefixes is not None:
            result[:len(prefixes[i])] = prefixes[i]
        return result

    async def request(i, tokens, kind, *, query=True, new_input_tokens=0):
        state = states[i]
        serial = state['serial']
        state['serial'] += 1
        source = plans[i] if plans is not None and query else None
        source_turn = source['turns'][state['turn']] if source is not None else None
        count = (source_turn['output_tokens'] if source_turn is not None else settings['output_tokens']) if query else 1
        if natural and query:
            count = settings['generation_limit']
            if len(tokens)+count > settings['max_model_len']:
                raise ValueError('Natural turn would exceed complete context; no request was sent')
        query_plan = dict(source['query_plan']) if source is not None else None
        if query_plan is not None and (state['locality_spans'] or state['locality_provenance']):
            query_plan['kvmem_locality_spans'] = list(state['locality_spans'])
            if state['locality_provenance']:
                query_plan['kvmem_locality_provenance'] = dict(state['locality_provenance'])
        return await stream_call(client, profile, tokens, state['session'],
            count,
            recorder, agent=i,
            epoch=state['epoch'], turn=state['turn'], kind=kind,
            request_id=f'timed-{tag}-{i}-{serial}-{kind}',
            new_input_tokens=new_input_tokens, query=query,
            query_plan=query_plan,
            source_metadata=dict(source_task=source.get('source_task'),
                                 source_turn=source_turn.get('source_turn')) if source is not None else None,
            generation_mode=settings['generation_mode'] if query else 'fixed',
            **({'release_session': True} if explicit_release and kind == 'cleanup' else {}))

    async def warm(i):
        state = states[i]
        recorder.phase(i, 'warming', state['epoch'], 0)
        await request(i, [fixture['eos']], 'cleanup', query=False)
        for turn in range(settings['warmup_turns']):
            state['turn'] = turn
            if natural:
                next_tokens = len(state['tokens'])+(len(tool(i,turn)) if turn else 0)
                if next_tokens+settings['generation_limit'] > settings['max_model_len']:
                    raise ValueError('Natural warmup cannot complete two full turns within context capacity')
            if turn:
                if plans is not None:
                    append_locality(state, plans[i]['turns'][turn])
                state['tokens'].extend(tool(i, turn))
            row = await request(i, state['tokens'], 'warmup')
            accepted_locality(state, row['output_ids'], fixture['eos'])
            state['tokens'].extend(row['output_ids'])
        state['turn'] += 1
        recorder.phase(i, 'warmup_barrier', state['epoch'], state['turn'])

    async def agent(i):
        state = states[i]
        deadline = recorder.window['t1_monotonic']
        try:
            while not stop.is_set() and time.monotonic() < deadline:
                epoch_full = state['turn'] == settings['epoch_turns']
                next_prompt_tokens = None
                capacity_full = False
                if natural and not epoch_full:
                    next_prompt_tokens = len(state['tokens'])+(len(tool(i,state['turn'])) if state['turn'] else 0)
                    capacity_full = next_prompt_tokens+settings['generation_limit'] > settings['max_model_len']
                if epoch_full or capacity_full:
                    if natural:
                        recorder.event('epoch_ended',agent=i,epoch=state['epoch'],completed_turns=state['turn'],
                            reason='context_capacity' if capacity_full else 'declared_turn_limit',
                            retained_history_tokens=len(state['tokens']),next_prompt_tokens=next_prompt_tokens,
                            generation_limit=settings['generation_limit'],max_model_len=settings['max_model_len'])
                    recorder.phase(i, 'cleanup', state['epoch'], state['turn'])
                    await request(i, [fixture['eos']], 'cleanup', query=False)
                    state['epoch'] += 1
                    state['turn'] = 0
                    state['tokens'] = initial(i)
                    state.update(initial_locality(plans[i] if plans is not None else None))
                    if stop.is_set() or time.monotonic() >= deadline:
                        break
                turn = state['turn']
                if turn:
                    delay = (plans[i]['turns'][turn]['tool_wait_seconds'] if plans is not None
                             else waits[i][(turn-1) % len(waits[i])]) if settings['mode'] == 'tools' else 0.
                    recorder.phase(i, 'tool_wait', state['epoch'], turn)
                    if delay:
                        try:
                            await asyncio.wait_for(stop.wait(), timeout=min(delay, max(0., deadline-time.monotonic())))
                        except asyncio.TimeoutError:
                            pass
                    if stop.is_set() or time.monotonic() >= deadline:
                        break
                    addition = tool(i, turn)
                    if plans is not None:
                        append_locality(state, plans[i]['turns'][turn])
                    state['tokens'].extend(addition)
                    added = len(addition)
                else:
                    added = len(state['tokens'])
                recorder.phase(i, 'ready', state['epoch'], turn)
                row = await request(i, state['tokens'], 'workload', new_input_tokens=added)
                accepted_locality(state, row['output_ids'], fixture['eos'])
                state['tokens'].extend(row['output_ids'])
                state['turn'] += 1
        except BaseException:
            stop.set()
            raise
        finally:
            recorder.phase(i, 'stopped', state['epoch'], state['turn'])

    try:
        for i in range(agents):
            states.append(dict(session=f'timed-{tag}-{i}', serial=0, epoch=0, turn=0, tokens=initial(i),
                               **initial_locality(plans[i] if plans is not None else None)))
        warm_results = await asyncio.gather(*(warm(i) for i in range(agents)), return_exceptions=True)
        failures = [r for r in warm_results if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
        if native:
            save(directory / 'native-quiescence-before.json', await native_quiescent(
                stage='before', not_before=max(r['begin'] for r in recorder.rows)))
        else:
            save(directory / 'resources-before.json', await wait_retirement(
                capture, not_before=max(r['begin'] for r in recorder.rows),
                expected_views={s['session']: len(s['tokens'])-1 for s in states}))
        started = time.monotonic()
        wall = time.time()
        recorder.window = dict(begin=started, end=started+duration,
                               t0_monotonic=started, t1_monotonic=started+duration,
                               t0_wall=wall, t1_wall=wall+duration,
                               duration_seconds=duration, interval='[T0,T1)',
                               warmup_turns=settings['warmup_turns'], agents=agents)
        save(directory / 'window.json', recorder.window)
        recorder.event('measurement_started', window=recorder.window)
        results = await asyncio.gather(*(agent(i) for i in range(agents)), return_exceptions=True)
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
        # All requests offered before T1 have finished. No cancellation is used
        # to improve a score; final namespace cleanup is outside the window.
        recorder.event('drain_finished')
        cleanup = await asyncio.gather(*(request(i, [fixture['eos']], 'cleanup', query=False)
                                        for i in range(agents)), return_exceptions=True)
        failures = [r for r in cleanup if isinstance(r, BaseException)]
        if failures:
            raise failures[0]
        if native:
            save(directory / 'native-quiescence-after.json', await native_quiescent(
                stage='after', not_before=max(r['begin'] for r in recorder.rows)))
        else:
            save(directory / 'resources-after.json', await wait_retirement(
                capture, not_before=max(r['begin'] for r in recorder.rows),
                expected_views={} if explicit_release else {s['session']: 1 for s in states}))
        summary = summarize(recorder.rows, recorder.window, recorder.phase_events)
        summary.update(status='completed', drained=True, mode=settings['mode'],
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       finished_wall=time.time(), drain_seconds=max(0., time.monotonic()-recorder.window['t1_monotonic']))
        if natural:
            summary.update(scope=settings['generation_scope'],generation_mode='natural',
                generation_limit=settings['generation_limit'],epoch_capacity_policy=settings['epoch_capacity_policy'])
        if native:
            summary.update(native_vllm=True, native_scope=settings['native_scope'],
                           native_cache_salt_scope=settings['native_cache_salt_scope'])
        save(directory / 'results.json', summary)
        return summary
    except BaseException as error:
        failure = dict(status='failed', error=f'{type(error).__name__}: {error}',
                       window=recorder.window, at=time.time())
        save(directory / 'results.json', failure)
        raise
    finally:
        save(directory / 'traces.json', recorder.rows)
        recorder.progress()
        recorder.stream.close()

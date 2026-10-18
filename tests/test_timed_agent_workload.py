import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    'timed_agent_workload', Path(__file__).parents[1] / 'scripts/timed_agent_workload.py')
timed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timed)


def fixture(agents=2):
    return dict(prompt=list(range(100, 300)), tool=list(range(800, 820)), eos=2,
                tool_sizes=[3, 5], waits=[[0.] for _ in range(agents)],
                user_prefixes=[[1000+i] for i in range(agents)],
                old_recorded_output_ids=[999999])


def arguments(**changes):
    values = dict(duration_seconds=.05, concurrency=2, warmup_turns=2,
                  output_tokens=2, context_tokens=150, epoch_turns=4,
                  mode='saturated', run_tag='test')
    values.update(changes)
    return SimpleNamespace(**values)


class Response:
    def __init__(self, payload, output, delays, failure=False):
        self.payload, self.output, self.delays, self.failure = payload, output, delays, failure

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        if self.failure:
            yield 'data: ' + json.dumps({'error': {'message': 'injected stream failure'}})
            return
        chunks = [self.output[:1], self.output[1:]] if len(self.output) > 1 else [self.output]
        for i, chunk in enumerate(chunks):
            await asyncio.sleep(self.delays[min(i, len(self.delays)-1)])
            yield 'data: ' + json.dumps(dict(id=self.payload['request_id'],
                choices=[dict(index=0, token_ids=chunk)]))
        yield 'data: ' + json.dumps(dict(id=self.payload['request_id'], choices=[],
            usage=dict(completion_tokens=len(self.output))))
        yield 'data: [DONE]'


class Stream:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *args):
        pass


class Client:
    def __init__(self, delays=None, fail=False):
        self.payloads = []
        self.delays = delays or (lambda payload: [.001])
        self.fail = fail

    def stream(self, method, url, *, json):
        payload = copy.deepcopy(json)
        self.payloads.append(payload)
        base = 100000 + len(self.payloads)*10
        output = list(range(base, base+payload['max_tokens']))
        failure = self.fail and payload['request_id'].endswith('-workload')
        return Stream(Response(payload, output, self.delays(payload), failure))


def install_retirement(monkeypatch):
    calls = []

    async def retirement(capture, **kwargs):
        calls.append(dict(at=time.monotonic(), expected_views=kwargs.get('expected_views')))
        return {'retired': True}

    monkeypatch.setattr(timed, 'wait_retirement', retirement)
    return calls


def run(client, tmp_path, args, data=None):
    return asyncio.run(timed.workload(client, {'port': 1, 'served_model_name': 'fake'},
                                     data or fixture(args.concurrency), tmp_path, args))


def row(request_id, begin, end, timeline, *, kind='workload', agent=0):
    return dict(request_id=request_id, begin=begin, end=end, timeline=timeline,
                output_ids=[0]*sum(n for _, n in timeline), kind=kind,
                status='completed', agent=agent, new_input_tokens=7)


@pytest.mark.parametrize('native', [False, True])
def test_explicit_release_transport_is_accounted_and_never_sent_to_native(tmp_path,native):
    client = Client(lambda payload: [0])
    recorder = timed.Recorder(tmp_path,1)
    try:
        result = asyncio.run(timed.stream_call(client,
            dict(port=1,served_model_name='model',native_vllm=native),[42],'session',1,
            recorder,agent=0,epoch=0,turn=0,kind='cleanup',request_id='close',
            new_input_tokens=1,query=False,release_session=True))
        payload, = client.payloads
        assert payload['max_tokens'] == 1 and result['output_ids']
        assert result['explicit_session_release'] is (not native)
        if native:
            assert 'kv_transfer_params' not in payload and 'session_id' not in payload
        else:
            assert payload['kv_transfer_params'] == {'kvmem_release_session':True}
            assert payload['session_id'] == payload['cache_salt'] == 'session'
            assert 'kvmem_query_span' not in payload
    finally:
        recorder.stream.close()


@pytest.mark.parametrize('change', [dict(query=True),dict(count=2),dict(tokens=[1,2]),
    dict(kind='workload'),dict(query_plan={}),dict(release_session='true')])
def test_release_cannot_silently_replace_a_normal_request(tmp_path,change):
    client = Client(lambda payload: [0])
    recorder = timed.Recorder(tmp_path,1)
    args=dict(tokens=[42],session='session',count=1,agent=0,epoch=0,turn=0,
        kind='cleanup',request_id='invalid-close',new_input_tokens=1,query=False,release_session=True)
    args.update(change)
    try:
        with pytest.raises(ValueError,match='release'):
            asyncio.run(timed.stream_call(client,dict(port=1,served_model_name='model'),recorder=recorder,**args))
        assert client.payloads == [] and recorder.rows == []
    finally:
        recorder.stream.close()


def test_window_counts_partial_responses_and_excludes_cleanup_and_right_boundary():
    rows = [row('cross', 9., 21., [[9., 1], [10., 2], [19., 3], [20., 4], [21., 5]]),
            row('first-token-after', 18., 22., [[20., 2]], agent=1),
            row('cleanup', 12., 13., [[12.5, 1]], kind='cleanup'),
            row('finished', 13., 14., [[13.5, 7]])]
    phases = [dict(monotonic=10., agent=0, phase='streaming', epoch=0, turn=1),
              dict(monotonic=21., agent=0, phase='stopped', epoch=0, turn=2)]
    result = timed.summarize(rows, dict(t0_monotonic=10., t1_monotonic=20.), phases)
    assert result['output_tokens'] == 12
    assert result['output_tokens_per_second'] == 1.2
    assert result['cleanup_output_tokens_in_window'] == 1
    assert result['all_stream_output_tokens_in_window'] == 13
    assert {r['request_id'] for r in result['pending_at_deadline']} == {'cross', 'first-token-after'}
    assert result['ttft_right_censored'] == [dict(request_id='first-token-after', at_least_seconds=2.)]
    assert result['agent_phases_at_deadline']['0']['phase'] == 'streaming'
    assert result['output_event_gap_samples'] == 1
    assert result['output_event_gap_p95_seconds'] == 9.


def test_same_work_warmup_barrier_and_each_turn_uses_own_accepted_output(tmp_path, monkeypatch):
    retired = install_retirement(monkeypatch)
    client = Client(lambda p: [.002 if p['session_id'].endswith('-1') else .001])
    args = arguments()
    data = fixture()
    result = run(client, tmp_path, args, data)
    window = json.loads((tmp_path/'window.json').read_text())
    rows = json.loads((tmp_path/'traces.json').read_text())
    warm = [r for r in rows if r['kind'] == 'warmup']
    assert [sum(r['agent'] == i for r in warm) for i in range(2)] == [2, 2]
    assert max(r['end'] for r in warm) <= retired[0]['at'] <= window['begin']
    assert retired[0]['expected_views'] == {'timed-test-0': 156, 'timed-test-1': 156}
    assert retired[1]['expected_views'] == {'timed-test-0': 1, 'timed-test-1': 1}
    assert window['begin'] == window['t0_monotonic']
    assert window['end'] == window['t1_monotonic']
    assert min(r['begin'] for r in rows if r['kind'] == 'workload') >= window['begin']
    by_id = {r['request_id']: r for r in rows}
    previous = {}
    for payload in client.payloads:
        current = by_id[payload['request_id']]
        if current['kind'] == 'cleanup':
            assert payload['kvmem_reselect'] == 'off'
            continue
        key = current['agent'], current['epoch']
        if current['turn']:
            last_payload, last = previous[key]
            addition = data['tool'][:data['tool_sizes'][(current['turn']-1) % 2]]
            addition[:1] = data['user_prefixes'][current['agent']]
            assert payload['prompt'] == last_payload['prompt'] + last['output_ids'] + addition
        assert 999999 not in payload['prompt']
        assert current['input_sha256'] == timed.digest(payload['prompt'])
        previous[key] = payload, current
    assert result['status'] == 'completed' and result['output_tokens'] > 0
    assert len(retired) == 2


def test_agents_refill_independently_without_a_round_barrier(tmp_path, monkeypatch):
    install_retirement(monkeypatch)
    client = Client(lambda p: [.025 if p['session_id'].endswith('-1') else .002])
    run(client, tmp_path, arguments(duration_seconds=.12, warmup_turns=1, epoch_turns=2))
    rows = json.loads((tmp_path/'traces.json').read_text())
    slow = next(r for r in rows if r['kind'] == 'workload' and r['agent'] == 1)
    refill = next(r for r in rows if r['kind'] == 'workload' and r['agent'] == 0 and r['epoch'] == 1)
    assert refill['begin'] < slow['end']
    assert refill['input_tokens'] == 150
    clean = next(r for r in rows if r['kind'] == 'cleanup' and r['agent'] == 0 and r['turn'] == 2)
    assert clean['end'] <= refill['begin']
    assert {p['session_id'] for p in client.payloads} == {'timed-test-0', 'timed-test-1'}


def test_stream_events_are_saved_before_response_finishes_and_drain_is_not_timed(tmp_path, monkeypatch):
    install_retirement(monkeypatch)
    client = Client(lambda p: [.001, .09] if p['request_id'].endswith('-workload') else [.001])

    async def exercise():
        task = asyncio.create_task(timed.workload(client, {'port': 1, 'served_model_name': 'fake'},
            fixture(1), tmp_path, arguments(concurrency=1, duration_seconds=.035)))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if (tmp_path/'events.jsonl').exists():
                events = [json.loads(s) for s in (tmp_path/'events.jsonl').read_text().splitlines()]
                chunks = [e for e in events if e['event'] == 'stream_chunk'
                          and e['kind'] == 'workload' and e['accepted_tokens']]
                if chunks:
                    assert not task.done()
                    assert not (tmp_path / (chunks[0]['request_id']+'.json')).exists()
                    break
            await asyncio.sleep(.001)
        else:
            pytest.fail('No incremental stream event was published')
        return await task

    result = asyncio.run(exercise())
    assert result['output_tokens'] == 1
    assert result['stream_output_tokens_by_kind']['workload'] == 2
    assert result['requests_completed_in_window'] == 0
    assert len(result['pending_at_deadline']) == 1
    assert result['drain_seconds'] > .04
    assert result['status'] == 'completed'


def test_long_tool_wait_stops_at_deadline_without_sending_new_work(tmp_path, monkeypatch):
    install_retirement(monkeypatch)
    data = fixture(1)
    data['waits'] = [[10.]]
    begin = time.monotonic()
    result = run(Client(), tmp_path, arguments(concurrency=1, mode='tools', duration_seconds=.03), data)
    assert time.monotonic()-begin < 1.
    assert result['output_tokens'] == 0
    assert result['requests_started_in_window'] == 0
    assert result['agent_phases_at_deadline']['0']['phase'] == 'tool_wait'


def test_failed_stream_preserves_failure_and_does_not_retry(tmp_path, monkeypatch):
    install_retirement(monkeypatch)
    client = Client(fail=True)
    with pytest.raises(RuntimeError, match='injected stream failure'):
        run(client, tmp_path, arguments(concurrency=1))
    result = json.loads((tmp_path/'results.json').read_text())
    assert result['status'] == 'failed'
    measured = [p for p in client.payloads if p['request_id'].endswith('-workload')]
    assert len(measured) == 1
    traces = json.loads((tmp_path/'traces.json').read_text())
    assert traces[-1]['status'] == 'failed'


def test_retirement_rejects_old_snapshot_and_waits_for_retiring_owner(tmp_path):
    state = dict(at=1., requests=0, refresh_pending=0, connector_pending=0,
                 owned_blocks=0, reservations={}, free_blocks=9, total_blocks=10,
                 cpu_history_admission=dict(active=0, retiring=0, waiting=0))
    path = tmp_path/'scheduler_resources.json'
    path.write_text(json.dumps(state))

    async def exercise():
        with pytest.raises(TimeoutError):
            await timed.wait_retirement(tmp_path, timeout=.015, not_before=2.)
        state['at'] = 3.
        state['cpu_history_admission']['retiring'] = 1
        path.write_text(json.dumps(state))
        with pytest.raises(TimeoutError):
            await timed.wait_retirement(tmp_path, timeout=.015, not_before=2.)
        state['cpu_history_admission']['retiring'] = 0
        path.write_text(json.dumps(state))
        assert (await timed.wait_retirement(tmp_path, timeout=.015, not_before=2.))['at'] == 3.

    asyncio.run(exercise())


def test_quiescent_snapshot_with_previous_view_cannot_open_measurement_window(tmp_path):
    view = dict(namespace=['agent', 'agent'], history=1, window=1, active_records=[[0, 1]])
    state = dict(at=3., requests=0, refresh_pending=0, connector_pending=0,
                 owned_blocks=0, reservations={}, free_blocks=9, total_blocks=10,
                 cpu_history_admission=dict(active=0, retiring=0, waiting=0),
                 session_views=[view])
    path = tmp_path/'scheduler_resources.json'
    path.write_text(json.dumps(state))

    async def exercise():
        # The clock and global idle counters are already acceptable. Only the
        # exact completed view can distinguish this earlier cleanup snapshot.
        with pytest.raises(TimeoutError):
            await timed.wait_retirement(tmp_path, timeout=.015, not_before=2.,
                                        expected_views={'agent': 120})
        view.update(history=120, window=8, active_records=[[2, 4], [100, 4]])
        path.write_text(json.dumps(state))
        actual = await timed.wait_retirement(tmp_path, timeout=.015, not_before=2.,
                                             expected_views={'agent': 120})
        assert actual['session_views'][0]['window'] == 8  # Sparse active view is allowed.

    asyncio.run(exercise())


@pytest.mark.parametrize('change', [
    dict(namespace=['agent', 'other']),
    dict(window=9),
    dict(active_records=[[2, 4], [4, 4]]),
    dict(active_records=[[2, 4], [119, 4]]),
])
def test_retired_view_requires_namespace_and_valid_sparse_active_records(change):
    view = dict(namespace=['agent', 'agent'], history=120, window=8, active_records=[[2, 4], [100, 4]])
    view.update(change)
    assert not timed.views_match({'session_views': [view]}, {'agent': 120})


def test_cleanup_requires_exact_single_token_view_and_no_duplicate_namespace():
    view = dict(namespace=['agent', 'agent'], history=1, window=1, active_records=[[0, 1]])
    assert timed.views_match({'session_views': [view]}, {'agent': 1})
    assert not timed.views_match({'session_views': [view, view]}, {'agent': 1})
    view['active_records'] = [[1, 1]]
    assert not timed.views_match({'session_views': [view]}, {'agent': 1})


def external_fixture(agents=2):
    plans = []
    for i in range(agents):
        plans.append(dict(source_task=f'source-{i}', source_turn=60+i,
            initial_tokens=list(range(1000*(i+1), 1000*(i+1)+150+25*i)),
            query_plan=dict(kvmem_query_span=[12, 18], kvmem_replay_span=[10, 20],
                            kvmem_reselect='auto', kvmem_prefill_window='pressure',
                            kvmem_mandatory_spans=[dict(begin=0, end=2, kind='system'),
                                                   dict(begin=12, end=18, kind='query')]),
            turns=[dict(output_tokens=count+i, tool_tokens=[] if turn == 0 else [8000+i]*(3+i),
                        tool_wait_seconds=0. if turn == 0 else 10., source_turn=60+i+turn)
                   for turn, count in enumerate((1, 3, 2, 4))]))
    return dict(agent_plans=plans, eos=2)


@pytest.mark.parametrize('explicit_release',[False,True])
def test_external_plans_preserve_user_inputs_round_quotas_plan_fields_and_own_output(tmp_path, monkeypatch, explicit_release):
    retired = install_retirement(monkeypatch)
    client = Client()
    data = external_fixture()
    original = copy.deepcopy(data)
    # Scalar synthetic settings must never replace per-user trace lengths or
    # quotas. The 10-second trace waits are ignored only in saturated mode.
    result = run(client, tmp_path, arguments(context_tokens=2, output_tokens=999,
                 duration_seconds=.06,explicit_session_release=explicit_release), data)
    assert data == original
    assert retired[0]['expected_views'] == {'timed-test-0': 156, 'timed-test-1': 184}
    assert retired[-1]['expected_views'] == ({} if explicit_release else {'timed-test-0':1,'timed-test-1':1})
    rows = json.loads((tmp_path/'traces.json').read_text())
    by_id = {r['request_id']: r for r in rows}
    previous = {}
    for payload in client.payloads:
        row = by_id[payload['request_id']]
        if row['kind'] == 'cleanup':
            assert row['explicit_session_release'] is explicit_release
            assert payload.get('kv_transfer_params') == ({'kvmem_release_session':True} if explicit_release else None)
            assert payload['kvmem_mandatory_spans'] == []
            assert payload['kvmem_reselect'] == 'off'
            assert 'kvmem_query_span' not in payload and 'kvmem_replay_span' not in payload
            continue
        assert 'kv_transfer_params' not in payload
        source = data['agent_plans'][row['agent']]
        turn = source['turns'][row['turn']]
        assert payload['max_tokens'] == turn['output_tokens']
        assert all(payload[key] == value for key, value in source['query_plan'].items())
        assert row['source_metadata'] == dict(source_task=source['source_task'], source_turn=turn['source_turn'])
        key = row['agent'], row['epoch']
        if row['turn'] == 0:
            assert payload['prompt'] == source['initial_tokens']
        else:
            old_payload, old_row = previous[key]
            assert payload['prompt'] == old_payload['prompt'] + old_row['output_ids'] + turn['tool_tokens']
        previous[key] = payload, row
    assert {r['input_tokens'] for r in rows if r['kind'] == 'warmup' and r['turn'] == 0} == {150, 175}
    assert [sum(r['kind'] == 'warmup' and r['agent'] == i for r in rows) for i in range(2)] == [2, 2]
    assert result['output_tokens'] > 0


def test_external_tools_wait_respects_deadline_after_fixed_work_warmup(tmp_path, monkeypatch):
    install_retirement(monkeypatch)
    result = run(Client(), tmp_path, arguments(concurrency=1, mode='tools', duration_seconds=.025),
                 external_fixture(1))
    assert result['requests_started_in_window'] == 0
    assert result['agent_phases_at_deadline']['0']['phase'] == 'tool_wait'
    assert result['stream_output_tokens_by_kind']['warmup'] == 4


@pytest.mark.parametrize('change, reason', [
    ('override', 'Unknown query_plan'),
    ('query-outside', 'Query span'),
    ('mandatory-outside', 'mandatory span'),
    ('missing-query', 'explicit initial Query'),
    ('negative-token', 'token IDs'),
    ('boolean-token', 'token IDs'),
    ('boolean-count', 'positive integer'),
    ('too-large', 'gen_budget'),
    ('infinite-wait', 'finite and nonnegative'),
    ('negative-wait', 'finite and nonnegative'),
    ('boolean-wait', 'finite and nonnegative'),
    ('initial-tool', 'Initial turn'),
    ('few-turns', 'Insufficient complete turns'),
    ('source-answer', 'turn schema'),
])
def test_external_plan_validation_rejects_overrides_and_invalid_work(change, reason):
    data = external_fixture(1)
    source = data['agent_plans'][0]
    if change == 'override': source['query_plan']['max_tokens'] = 1
    if change == 'query-outside': source['query_plan']['kvmem_query_span'] = [149, 151]
    if change == 'mandatory-outside': source['query_plan']['kvmem_mandatory_spans'][0]['end'] = 151
    if change == 'missing-query': source['query_plan'] = {'kvmem_reselect': 'auto'}
    if change == 'negative-token': source['initial_tokens'][0] = -1
    if change == 'boolean-token': source['turns'][1]['tool_tokens'][0] = True
    if change == 'boolean-count': source['turns'][1]['output_tokens'] = True
    if change == 'too-large': source['turns'][1]['output_tokens'] = 131073
    if change == 'infinite-wait': source['turns'][1]['tool_wait_seconds'] = float('inf')
    if change == 'negative-wait': source['turns'][1]['tool_wait_seconds'] = -.1
    if change == 'boolean-wait': source['turns'][1]['tool_wait_seconds'] = True
    if change == 'initial-tool': source['turns'][0]['tool_tokens'] = [2]
    if change == 'few-turns': source['turns'].pop()
    if change == 'source-answer': source['turns'][1]['output_ids'] = [99]
    with pytest.raises(ValueError, match=reason):
        timed.validate_agent_plans(data, concurrency=1, warmup_turns=2, epoch_turns=4, gen_budget=131072)


def test_external_plan_validation_checks_all_user_counts_and_source_lengths():
    data = external_fixture(1)
    with pytest.raises(ValueError, match='Insufficient per-user'):
        timed.validate_agent_plans(data, concurrency=2, warmup_turns=2, epoch_turns=4, gen_budget=131072)
    data['agent_plans'][0]['turns'][0]['output_tokens'] = 32000
    before = copy.deepcopy(data)
    accepted = timed.validate_agent_plans(data, concurrency=1, warmup_turns=2, epoch_turns=4, gen_budget=131072)
    assert accepted[0]['turns'][0]['output_tokens'] == 32000
    assert data == before


def test_external_workload_rejects_over_budget_before_sending_http(tmp_path, monkeypatch):
    install_retirement(monkeypatch)
    data = external_fixture(1)
    data['agent_plans'][0]['turns'][0]['output_tokens'] = 131073
    client = Client()
    with pytest.raises(ValueError, match='gen_budget'):
        asyncio.run(timed.workload(client, dict(port=1, served_model_name='fake',
            kvmem=dict(gen_budget=131072)), data, tmp_path, arguments(concurrency=1)))
    assert client.payloads == []


def test_native_external_replay_preserves_full_prompts_quotas_and_epoch_cache_isolation(tmp_path, monkeypatch):
    # This is a transport/epoch-isolation test. Drive its observation clock
    # by completed fake requests, not the host's 131K-token JSON/hash speed.
    clock=[1.0]
    monkeypatch.setattr(timed,'time',SimpleNamespace(monotonic=lambda:clock[0],time=time.time))
    def unexpected_retirement(*args, **kwargs):
        pytest.fail('Native replay must never consume KVMem retirement evidence')

    monkeypatch.setattr(timed, 'wait_retirement', unexpected_retirement)
    quiescence = []

    async def native_quiescent(*, stage, not_before):
        evidence = dict(stage=stage, not_before=not_before, observed_at=clock[0],
                        native_metrics={'vllm:num_requests_running': 0,
                                        'vllm:num_requests_waiting': 0})
        quiescence.append(evidence)
        return evidence

    args = arguments(concurrency=1, duration_seconds=.075, native_quiescent=native_quiescent)
    data = external_fixture(1)
    # This exceeds the KVMem selection budget. The native client must submit
    # every ID, including the historical suffix, rather than select/truncate.
    data['agent_plans'][0]['initial_tokens'] = list(range(131100))
    original = copy.deepcopy(data)
    def advance(payload):
        clock[0]+=.01
        return [0]
    client = Client(advance)
    result = asyncio.run(timed.workload(client,
        dict(port=1, served_model_name='native', native_vllm=True), data, tmp_path, args))
    rows = json.loads((tmp_path/'traces.json').read_text())
    by_id = {r['request_id']: r for r in rows}
    previous, salts = {}, {}
    for payload in client.payloads:
        current = by_id[payload['request_id']]
        assert 'session_id' not in payload
        assert not any(key.startswith('kvmem') for key in payload)
        assert payload['cache_salt'] == f"{current['session']}:epoch:{current['epoch']}"
        salts.setdefault(current['epoch'], set()).add(payload['cache_salt'])
        assert current['query_plan'] is None
        if current['kind'] == 'cleanup':
            assert payload['prompt'] == [data['eos']]
            continue
        source = data['agent_plans'][0]
        turn = source['turns'][current['turn']]
        assert payload['max_tokens'] == turn['output_tokens']
        assert len(current['output_ids']) == turn['output_tokens']
        if current['turn'] == 0:
            assert payload['prompt'] == source['initial_tokens']
        else:
            before_payload, before_row = previous[current['epoch']]
            assert payload['prompt'] == before_payload['prompt'] + before_row['output_ids'] + turn['tool_tokens']
        assert len(payload['prompt']) >= 131100
        assert current['input_sha256'] == timed.digest(payload['prompt'])
        previous[current['epoch']] = payload, current
    assert data == original
    assert len(salts) >= 2
    assert all(len(values) == 1 for values in salts.values())
    assert len({next(iter(values)) for values in salts.values()}) == len(salts)
    assert [r['stage'] for r in quiescence] == ['before', 'after']
    warmup = [r for r in rows if r['kind'] == 'warmup']
    window = json.loads((tmp_path/'window.json').read_text())
    assert max(r['end'] for r in warmup) <= quiescence[0]['observed_at'] <= window['begin']
    assert quiescence[0]['not_before'] == max(r['begin'] for r in rows if r['kind'] != 'workload' and
                                           r['begin'] < window['begin'])
    assert quiescence[1]['not_before'] == max(r['begin'] for r in rows)
    for stage in ('before', 'after'):
        assert json.loads((tmp_path/f'native-quiescence-{stage}.json').read_text()) == quiescence[stage == 'after']
        assert not (tmp_path/f'resources-{stage}.json').exists()
    settings = json.loads((tmp_path/'workload.json').read_text())
    assert settings['native_vllm'] and 'no cross-epoch' in settings['native_cache_salt_scope']
    assert result['native_vllm'] and 'Full growing prompts' in result['native_scope']
    assert result['status'] == 'completed' and result['output_tokens'] > 0


@pytest.mark.parametrize('stage', ['before', 'after'])
def test_native_quiescence_failure_is_not_replaced_with_kvmem_evidence(tmp_path, stage):
    async def native_quiescent(*, stage: str, not_before: float):
        if stage == failing_stage:
            raise TimeoutError('Native engine gauges did not quiesce')
        return dict(stage=stage, native_running=0, native_waiting=0)

    failing_stage = stage
    client = Client()
    args = arguments(concurrency=1, duration_seconds=.015, native_quiescent=native_quiescent)
    with pytest.raises(TimeoutError, match='Native engine gauges'):
        asyncio.run(timed.workload(client, dict(port=1, served_model_name='native', native_vllm=True),
                                  fixture(1), tmp_path, args))
    result = json.loads((tmp_path/'results.json').read_text())
    assert result['status'] == 'failed'
    assert not (tmp_path/f'native-quiescence-{stage}.json').exists()
    if stage == 'before':
        assert not (tmp_path/'window.json').exists()
        assert not any(p['request_id'].endswith('-workload') for p in client.payloads)
    assert not (tmp_path/'resources-before.json').exists()
    assert not (tmp_path/'resources-after.json').exists()


@pytest.mark.parametrize('profile, error', [
    (dict(native_vllm=True), 'native_quiescent'),
    (dict(native_vllm=1), 'native_vllm must be boolean'),
])
def test_native_mode_requires_explicit_valid_quiescence_callback_before_http(tmp_path, profile, error):
    client = Client()
    with pytest.raises(ValueError, match=error):
        asyncio.run(timed.workload(client, dict(port=1, served_model_name='native', **profile),
                                  fixture(1), tmp_path, arguments(concurrency=1)))
    assert client.payloads == []

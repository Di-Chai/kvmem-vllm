"""Host tests for the live HTTP cancellation qualification harness."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest


spec = importlib.util.spec_from_file_location('stress_request_swaps',
    Path(__file__).resolve().parents[1] / 'scripts/stress_request_swaps.py')
stress = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stress
spec.loader.exec_module(stress)


def snapshot():
    return dict(at=1, requests=0, owned_blocks=0, refresh_pending=0, connector_pending=0,
        reservations=[], publication_blocked=[], free_blocks=31, total_blocks=32,
        request_swaps=dict(requests={}, inflight=None, pressure={}, resources=dict(active=0,
            resources=dict(checkpoint_bytes=dict(live=0, promised=0)))),
        cpu_history_admission=dict(leases=0, active=0, retiring=0, reserved_bytes=0))


@pytest.mark.parametrize('internal,expected', [
    ('cmpl-test-0', True), ('cmpl-test-0-abcdef01', True),
    ('cmpl-test-0-abcdef012', False), ('cmpl-test-0-nope', False),
    ('cmpl-test-extra-0-abcdef01', False), ('cmpl-test-1-abcdef01', False)])
def test_mapping_matches_the_whole_native_prompt_request(internal, expected):
    assert stress.internal_id_matches('test', internal) is expected


@pytest.mark.parametrize('finished', ['completed', 'finish_reason', 'cancel_requested', 'not_open'])
def test_never_closes_an_already_finished_stream(finished):
    state = stress.Stream('test', opened=True)
    if finished == 'not_open': state.opened = False
    elif finished == 'finish_reason': state.finish_reason = 'length'
    else: setattr(state, finished, True)
    report = snapshot()
    report['request_swaps']['requests']['cmpl-test-0-abcdef01'] = {'phase': 'paused'}
    assert stress.cancellation_candidate(report, [state]) is None


def test_only_exact_live_migration_phases_qualify():
    state = stress.Stream('test', opened=True)
    report = snapshot()
    for phase in ['capturing', 'paused', 'restoring', 'restored', 'dropping', 'done']:
        report['request_swaps']['requests'] = {'cmpl-test-0-abcdef01': {'phase': phase}}
        assert bool(stress.cancellation_candidate(report, [state])) == (phase in stress.PHASES)


@pytest.mark.parametrize('group,name', [
    (None, 'requests'), (None, 'owned_blocks'), (None, 'connector_pending'),
    ('checkpoint', 'live'), ('checkpoint', 'promised'),
    ('history', 'retiring'), ('history', 'leases'), ('history', 'reserved_bytes')])
def test_reclamation_requires_all_actual_ownership_and_retirement_to_drain(group, name):
    report = snapshot()
    assert not stress.outstanding(report)
    if group == 'checkpoint': report['request_swaps']['resources']['resources']['checkpoint_bytes'][name] = 1
    elif group == 'history': report['cpu_history_admission'][name] = 1
    else: report[name] = 1
    assert stress.outstanding(report)


def test_terminal_trace_proves_server_aborted_after_connection_close(tmp_path):
    path = tmp_path / 'scheduler_finished.jsonl'
    good = dict(request_id='cmpl-test-0-abcdef01', status='FINISHED_ABORTED', at=20)
    path.write_text(json.dumps(good) + '\n')
    assert stress.aborted_terminal(path, good['request_id'], after=19) == good
    with pytest.raises(ValueError): stress.aborted_terminal(path, good['request_id'], after=21)
    path.write_text(json.dumps({**good, 'status': 'FINISHED_LENGTH_CAPPED'}) + '\n')
    with pytest.raises(ValueError): stress.aborted_terminal(path, good['request_id'], after=19)


def test_cancelling_live_http_reader_closes_the_actual_response_stream():
    async def run():
        class LiveStream(httpx.AsyncByteStream):
            closed = False
            async def __aiter__(self):
                packet = dict(id='cmpl-test', choices=[dict(index=0, token_ids=[7], finish_reason=None)])
                yield ('data: ' + json.dumps(packet) + '\n\n').encode()
                await asyncio.Event().wait()
            async def aclose(self): self.closed = True
        live = LiveStream()
        def transport(request):
            assert request.headers['X-Request-Id'] == 'test'
            assert json.loads(request.content)['request_id'] == 'test'
            return httpx.Response(200, stream=live)
        state = stress.Stream('test')
        events = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            task = asyncio.create_task(stress.stream_request(client, 'http://test', {}, state,
                lambda kind, **data: events.append((kind, data))))
            while not state.token_ids: await asyncio.sleep(0)
            state.cancel_requested = True
            task.cancel()
            await task
        assert live.closed and state.cancelled and not state.completed
        assert state.response_id == 'cmpl-test' and state.token_ids == [7]
        assert [kind for kind, _ in events] == ['http_open', 'sse', 'http_closed']
    asyncio.run(run())


def test_wait_drained_does_not_accept_stale_idle_snapshot(tmp_path):
    async def run():
        path = tmp_path / 'snapshot.json'
        path.write_text(json.dumps(snapshot()))
        events = []
        evidence = type('Evidence', (), {'emit': lambda _, kind, **data: events.append(data)})()
        task = asyncio.create_task(stress.wait_drained(path, evidence, after=2))
        await asyncio.sleep(0.02)
        assert not task.done()
        path.write_text(json.dumps({**snapshot(), 'at': 3}))
        result = await asyncio.wait_for(task, 1)
        assert result['at'] == 3 and len(events) == 1
    asyncio.run(run())

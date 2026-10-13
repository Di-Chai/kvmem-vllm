"""Real loopback HTTP + real admission middleware, synthetic inference only."""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
from pathlib import Path
import socket
import sys

import pytest

from vllm_kvmem.multi_user import MultiUserMiddleware


scripts = Path(__file__).parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('admission_gate', scripts / 'qualify_admission.py')
gate = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(scripts))
try:
    spec.loader.exec_module(gate)
finally:
    sys.path.remove(str(scripts))


@asynccontextmanager
async def service(tmp_path, *, reject_peer=False, queue_capacity=64, initial_completed=0):
    uvicorn = pytest.importorskip('uvicorn')
    users = [{'id': 'one', 'api_key': 'private-one-key'}, {'id': 'two', 'api_key': 'private-two-key'}]
    users_file = tmp_path / 'users.json'
    users_file.write_text(json.dumps({'users': users}))
    capture = tmp_path / 'capture'
    capture.mkdir()
    active = 0
    completed = [{'request_id': f'secret-native-id-{index}', 'terminal_prefix': {
        'status': 'FINISHED_ABORTED' if index in (44, 45) else 'FINISHED_STOPPED'}}
        for index in range(initial_completed)]

    def snapshot():
        archive = {
            'active_requests': active, 'retiring_requests': 0, 'pending_downloads': 0,
            'completed': completed[-256:] if initial_completed else completed}
        if initial_completed:
            archive['completed_requests'] = len(completed)
        (capture / 'archive.json').write_text(json.dumps(archive))
        (capture / 'scheduler_resources.json').write_text(json.dumps({
            'requests': active, 'owned_blocks': active, 'reservations': 0,
            'refresh_pending': 0, 'free_blocks': 99 - active, 'total_blocks': 100}))

    snapshot()

    async def native(scope, receive, send):
        nonlocal active
        raw = b''
        while True:
            message = await receive()
            raw += message.get('body', b'')
            if not message.get('more_body'):
                break
        payload = json.loads(raw)
        if reject_peer and scope['state']['kvmem_user_id'] == 'two':
            await send({'type': 'http.response.start', 'status': 503, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'not available'})
            return
        active += 1
        snapshot()
        status = 'FINISHED_STOPPED'

        async def emit(event, more=True):
            text = event if isinstance(event, str) else json.dumps(event)
            await send({'type': 'http.response.body', 'body': ('data: ' + text + '\n\n').encode(),
                        'more_body': more})

        try:
            await send({'type': 'http.response.start', 'status': 200,
                        'headers': [(b'content-type', b'text/event-stream')]})
            await emit({'choices': [{'delta': {'content': 'OK'}, 'finish_reason': None}]})
            if payload.get('ignore_eos'):
                await asyncio.sleep(120)
            await asyncio.sleep(.02)
            await emit({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
            await emit('[DONE]', more=False)
        except asyncio.CancelledError:
            status = 'FINISHED_ABORTED'
            raise
        finally:
            active -= 1
            completed.append({'request_id': 'secret-native-id-do-not-store',
                              'terminal_prefix': {'status': status}})
            snapshot()

    app = MultiUserMiddleware(native, {'users_file': str(users_file), 'max_inflight': 8,
                                     'max_inflight_per_user': 2, 'max_queue': queue_capacity,
                                     'queue_timeout_seconds': 120})
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen(256)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='off', access_log=False))
    server_task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(.005)
    args = gate.parse_args(['--api-base', f'http://127.0.0.1:{port}/v1', '--users-file', str(users_file),
                           '--capture', str(capture), '--output', str(tmp_path / 'report'),
                           '--scenario-timeout', '5'])
    try:
        yield args, users, app
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, 3)
        sock.close()


@pytest.mark.parametrize('initial_completed', [0, 300])
def test_real_http_overload_fairness_cancellation_recovery_and_native_settle(tmp_path, initial_completed):
    async def run():
        async with service(tmp_path, initial_completed=initial_completed) as (args, users, app):
            report = await gate.run_gate(args, users)
            assert report['status'] == 'passed', report
            assert report['fanout'] == 68
            assert report['retry_after_rejections'] >= 1
            assert report['peak_active_content_streams'] == 2
            assert report['all_owned_requests_closed']
            assert gate.native_idle(report['phases']['after'])
            assert report['phases']['peer']['outcome'] == 'success'
            assert report['phases']['peer']['min_other_user_active_content_streams'] == 2
            assert report['phases']['recovery_attempts'][-1]['outcome'] == 'success'
            assert report['native_aborted_delta'] >= 2
            assert report['native_completed_delta'] >= 4
            assert report['phases']['before']['completion_count_source'] == (
                'cumulative' if initial_completed else 'legacy_history_length')
            if initial_completed:
                assert report['phases']['before']['completed_retained'] == 256
                assert report['phases']['after']['completed_retained'] == 256
            assert report['wall_seconds'] < 5
            assert gate.native_idle(report['phases']['after'])
            assert app._admission.total == app._admission.queued == 0
            serialized = json.dumps(report)
            assert 'secret-native-id' not in serialized
            assert all(user['api_key'] not in serialized for user in users)
    asyncio.run(run())


def test_failure_still_closes_active_and_queued_requests(tmp_path):
    async def run():
        async with service(tmp_path, reject_peer=True) as (args, users, app):
            report = await gate.run_gate(args, users)
            assert report['status'] == 'failed'
            assert report['error'] == 'peer_did_not_succeed_while_other_user_saturated'
            assert report['all_owned_requests_closed']
            # HTTP cancellation can arrive just after client sockets close.
            deadline = asyncio.get_running_loop().time() + 1
            while app._admission.total or app._admission.queued:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(.01)
            assert report['wall_seconds'] < 5
    asyncio.run(run())


def test_stale_idle_capture_cannot_prove_post_cancellation_settle(tmp_path):
    capture = tmp_path
    (capture / 'archive.json').write_text(json.dumps({'active_requests': 0, 'retiring_requests': 0,
        'pending_downloads': 0, 'completed': []}))
    (capture / 'scheduler_resources.json').write_text(json.dumps({'requests': 0, 'owned_blocks': 0,
        'reservations': 0, 'refresh_pending': 0, 'free_blocks': 9, 'total_blocks': 10}))
    before = gate.native_snapshot(capture)
    assert gate.native_idle(before)

    async def run():
        with pytest.raises(gate.GateFailure, match='did_not_settle'):
            await gate.settle(capture, gate.time.monotonic() + .1, after=before)
    asyncio.run(run())


def test_missing_overload_times_out_and_cancels_waiting_streams(tmp_path):
    async def run():
        async with service(tmp_path, queue_capacity=128) as (args, users, app):
            args.scenario_timeout = 1
            report = await gate.run_gate(args, users)
            assert report['status'] == 'failed'
            assert report['error'] == 'phase_timeout'
            assert report['retry_after_rejections'] == 0
            assert report['all_owned_requests_closed']
            assert report['wall_seconds'] < 1
            assert gate.native_idle(report['phases']['after'])
            assert app._admission.total == app._admission.queued == 0
    asyncio.run(run())


def test_admission_arguments_require_overload_and_bounded_scenario(tmp_path):
    common = ['--users-file', str(tmp_path / 'users'), '--capture', str(tmp_path),
              '--output', str(tmp_path / 'result')]
    for extra in (['--fanout', '66'], ['--scenario-timeout', '31'], ['--scenario-timeout', 'nan']):
        with pytest.raises(SystemExit):
            gate.parse_args(common + extra)


def test_completion_delta_counts_new_aborts_when_old_aborts_are_evicted():
    before = {'completed_count': 300, 'completed_retained': 256,
              'aborted_count': 2, 'aborted_completion_ordinals': [45, 46]}
    after = {'completed_count': 304, 'completed_retained': 256,
             'aborted_count': 2, 'aborted_completion_ordinals': [302, 303]}
    assert gate.completion_deltas(before, after) == (4, 2)
    assert gate.completion_deltas(before, before) == (0, 0)
    with pytest.raises(gate.GateFailure, match='backwards'):
        gate.completion_deltas(after, before)
    with pytest.raises(gate.GateFailure, match='no_longer_retained'):
        gate.completion_deltas({'completed_count': 0}, after)

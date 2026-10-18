import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sqlite3
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load('profile_restore_diagnostic')
target = load('nsys_diagnostic_target')


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / 'diagnostic'
    scripts = root / 'frozen/scripts'
    scripts.mkdir(parents=True)
    hashes = {}
    for name in ('profile_restore_diagnostic.py', 'nsys_diagnostic_target.py', 'diagnose_restore_stages.py'):
        path = scripts / name
        path.write_text('# frozen test script\n')
        hashes['scripts/' + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(source_sha256=hashes, profile_sha256={'P1': 'original'}, fixture_sha256='same-fixture',
                    workload={'duration_seconds': 300})
    write(root / 'manifest.json', manifest)
    write(root / 'status.json', {'status': 'prepared'})
    monkeypatch.setattr(profile, 'ROOT', root / 'frozen')
    monkeypatch.setattr(target, 'ROOT', root / 'frozen')
    monkeypatch.setattr(profile.diagnostic, 'validate', lambda _: (manifest, {}, {}))
    monkeypatch.setattr(profile.diagnostic, 'LOCK_PATH', str(tmp_path / 'gpu.lock'))
    return root, root / 'nsys', manifest


def claim_for(root, output):
    return dict(nonce='unique-nonce', session='unique-session', capture_output=str(output),
                manifest_sha256=profile.diagnostic.sha256(root / 'manifest.json'))


def target_row(root, output, claim):
    return dict(pid=42, create_time=123., nonce=claim['nonce'],
        diagnostic_output=str(root), capture_output=str(output), manifest_sha256=claim['manifest_sha256'],
        parents=[{'pid': 1, 'create_time': 1.}])


def test_profile_is_bounded_delayed_and_never_kills_the_harness(prepared):
    root, output, _ = prepared
    command = profile.profile_command(Path('/nsys'), root, output, claim_for(root, output), 45.)
    assert '--start-later=true' in command and '--duration=45.0' in command
    assert '--kill=none' in command and '--wait=all' in command
    assert '--trace=cuda,nvtx,osrt' in command and not any('python-gil' in v for v in command)
    assert '--stats=false' in command and '--export=none' in command
    assert str(root / 'frozen/scripts/nsys_diagnostic_target.py') in command
    assert '--trace-fork-before-exec=true' not in command


@pytest.mark.parametrize('change', ['already-running', 'claimed', 'previous-arm', 'outside', 'duration', 'offset'])
def test_only_new_unchanged_prepared_diagnostics_are_accepted(prepared, change):
    root, output, _ = prepared
    seconds, offset = 45., 30.
    if change == 'already-running':
        write(root / 'status.json', {'status': 'running'})
    elif change == 'claimed':
        write(root / 'nsys-claim.json', {})
    elif change == 'previous-arm':
        (root / 'pair-00-P1').mkdir()
    elif change == 'outside':
        output = root.parent / 'elsewhere'
    elif change == 'duration':
        seconds = 300.
    else:
        offset = 280.
    with pytest.raises(ValueError):
        profile.validate_prepared(root, output, seconds, offset)


def test_reparented_target_is_accepted_using_nonce_identity_and_exact_frozen_command(prepared, monkeypatch):
    root, output, _ = prepared
    claim = claim_for(root, output)
    row = target_row(root, output, claim)
    command = ['/python', str(root / 'frozen/scripts/diagnose_restore_stages.py'),
               '--run-prepared', '--output', str(root)]
    process = SimpleNamespace(cmdline=lambda: command)
    monkeypatch.setattr(profile, 'live', lambda _: process)
    assert profile.proven_target(row, root, output, claim) is row
    command[-1] = str(root.parent / 'another')
    with pytest.raises(ValueError, match='PID no longer'):
        profile.proven_target(row, root, output, claim)


def test_recycled_pid_is_not_signalled(prepared, monkeypatch):
    root, output, _ = prepared
    claim = claim_for(root, output)
    process = SimpleNamespace(create_time=lambda: 999., status=lambda: 'running',
                              send_signal=lambda _: pytest.fail('Recycled PID signalled'))
    monkeypatch.setattr(profile.psutil, 'Process', lambda _: process)
    assert not profile.signal_target(target_row(root, output, claim), root, output, claim)


def test_finished_compiler_child_does_not_abort_server_tracking(monkeypatch):
    owners = {10: dict(pid=10, create_time=100., server_group=10),
              11: dict(pid=11, create_time=101., server_group=10)}
    def vanished():
        raise profile.psutil.NoSuchProcess(11)
    process = SimpleNamespace(children=lambda recursive: vanished())
    monkeypatch.setattr(profile, 'live', lambda owner: process if owner['pid'] == 11 else None)
    profile.track_server(dict(pid=1, create_time=1.), {}, owners)
    assert set(owners) == {10, 11}  # Preserve known identities for owned cleanup.


def test_cancel_signals_only_the_already_verified_process_object(prepared, monkeypatch):
    root, output, _ = prepared
    claim = claim_for(root, output)
    sent = []
    process = SimpleNamespace(send_signal=sent.append)
    monkeypatch.setattr(profile, 'live', lambda _: process)
    monkeypatch.setattr(profile, 'proven_target', lambda *a: a[0])
    monkeypatch.setattr(profile.psutil, 'Process', lambda *a: pytest.fail('Do not recreate identity before signal'))
    assert profile.signal_target(target_row(root, output, claim), root, output, claim)
    assert sent == [signal.SIGTERM]


@pytest.mark.parametrize('gpu', [dict(pids=[12], memory_mib=0, utilization=0),
    dict(pids=[], memory_mib=512, utilization=0), dict(pids=[], memory_mib=0, utilization=5)])
def test_busy_gpu_is_rejected_without_queueing(monkeypatch, gpu):
    monkeypatch.setattr(profile.q, 'gpu_state', lambda: gpu)
    with pytest.raises(RuntimeError, match='does not queue'):
        profile.require_idle()


def test_helper_cannot_exec_without_an_ack_and_rechecks_claim_before_exec(prepared, monkeypatch):
    root, output, _ = prepared
    output.mkdir()
    claim = claim_for(root, output)
    write(root / 'nsys-claim.json', claim)
    clock = [0.]
    monkeypatch.setattr(target.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(target.time, 'sleep', lambda _: clock.__setitem__(0, clock[0]+31))
    monkeypatch.setattr(target.psutil, 'Process', lambda: SimpleNamespace(pid=42, create_time=lambda: 123., parents=lambda: []))
    monkeypatch.setattr(target.os, 'execv', lambda *a: pytest.fail('No ACK must prevent model startup'))
    args = SimpleNamespace(diagnostic_output=root, capture_output=output, nonce=claim['nonce'])
    with pytest.raises(TimeoutError, match='not started'):
        target.run(args)
    row = target.read(output / 'target.json')
    assert row['pid'] == 42 and row['create_time'] == 123. and row['nonce'] == claim['nonce']


def test_helper_execs_frozen_diagnostic_without_changing_pid_after_ack(prepared, monkeypatch):
    root, output, _ = prepared
    output.mkdir()
    claim = claim_for(root, output)
    write(root / 'nsys-claim.json', claim)
    write(output / 'allow-exec.json', dict(nonce=claim['nonce'], pid=42, create_time=123.))
    monkeypatch.setattr(target.psutil, 'Process', lambda: SimpleNamespace(pid=42, create_time=lambda: 123., parents=lambda: []))
    monkeypatch.setattr(target.os, 'chdir', lambda path: None)
    calls = []

    class ExecReached(Exception):
        pass

    def execute(executable, command):
        calls.append(command)
        raise ExecReached()

    monkeypatch.setattr(target.os, 'execv', execute)
    with pytest.raises(ExecReached):
        target.run(SimpleNamespace(diagnostic_output=root, capture_output=output, nonce=claim['nonce']))
    assert calls == [[target.sys.executable, str(root / 'frozen/scripts/diagnose_restore_stages.py'),
                      '--run-prepared', '--output', str(root)]]
    assert target.read(output / 'target.json')['pid'] == 42


@pytest.mark.parametrize('table', ['KERNEL', 'GRAPH_TRACE'])
def test_report_requires_a_complete_database_with_real_device_activity(tmp_path, table):
    path = tmp_path / 'report.sqlite'
    table = 'CUPTI_ACTIVITY_KIND_' + table
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE ' + table + '(start INTEGER,end INTEGER,globalPid INTEGER)')
    with pytest.raises(RuntimeError, match='no CUDA'):
        profile.validate_export(path, [42])
    with sqlite3.connect(path) as connection:
        connection.execute('INSERT INTO ' + table + ' VALUES (10,20,?)', (42 << 24,))
    result = profile.validate_export(path, [42])
    assert result['sqlite_integrity'] == 'ok'
    assert result['device_activity_extent_ns'][table] == (10, 20)
    assert result['verified_engine_pids'] == [42]
    assert 'conservative busy envelope' in result['coverage_scope']


@pytest.mark.parametrize('activity', ['host_api_only', 'wrong_pid', 'missing_pid', 'zero_duration', 'memcpy_only'])
def test_export_must_include_actual_engine_device_compute(tmp_path, activity):
    path = tmp_path / 'report.sqlite'
    table = ('RUNTIME' if activity == 'host_api_only' else
             'MEMCPY' if activity == 'memcpy_only' else 'KERNEL')
    with sqlite3.connect(path) as connection:
        if activity == 'missing_pid':
            connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER)')
            connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10,20)')
        else:
            connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_' + table +
                               '(start INTEGER,end INTEGER,globalPid INTEGER)')
            connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_' + table + ' VALUES (10,?,?)',
                (10 if activity == 'zero_duration' else 20, (999 if activity == 'wrong_pid' else 42) << 24))
    with pytest.raises(RuntimeError, match='no CUDA|Missing CUDA'):
        profile.validate_export(path, [42])


def test_engine_proof_reuses_only_guard_samples_inside_capture_window(prepared):
    root, _, _ = prepared
    directory = root / 'pair-00-P1'
    directory.mkdir()
    path = directory / 'gpu_samples.jsonl'
    path.write_text('\n'.join(json.dumps(dict(at=at, pids=pids)) for at, pids in
        [(1., [999]), (10., [42]), (11., [42]), (20., [888])]) + '\n')
    owned = {42: dict(pid=42, create_time=123., server_group=41)}
    result = profile.observed_engines(root, owned, 10., 15.)
    assert result['samples'] == 2 and result['identities'] == [owned[42]]
    with pytest.raises(RuntimeError, match='not proven'):
        profile.observed_engines(root, owned, 1., 15.)
    with pytest.raises(RuntimeError, match='No engine'):
        profile.observed_engines(root, owned, 30., 45.)


@pytest.mark.parametrize('fail_harness', [False, True])
def test_collector_exit_does_not_end_target_tracking_or_allow_early_export(prepared, monkeypatch, fail_harness):
    root, output, _ = prepared
    state = dict(tick=0, launched=False, exported=False, signals=[])
    original_read = profile.read
    # Scope the simulated clocks/IO to the wrapper. Patching the shared time or
    # asyncio modules changes the real test event loop and executor shutdown.
    monkeypatch.setattr(profile, 'time', SimpleNamespace(
        monotonic=lambda: 100.+state['tick'], time=profile.time.time))
    monkeypatch.setattr(profile.q, 'gpu_state', lambda: dict(pids=[], memory_mib=0, utilization=0))
    monkeypatch.setattr(profile, 'identity', lambda p: dict(pid=p.pid, create_time=123.))
    monkeypatch.setattr(profile, 'psutil', SimpleNamespace(
        Process=lambda pid=None: SimpleNamespace(pid=pid or 999)))
    monkeypatch.setattr(profile, 'proven_target', lambda row, *a: row)
    monkeypatch.setattr(profile, 'live', lambda row: SimpleNamespace(pid=42) if row and state['tick']<4 else None)
    monkeypatch.setattr(profile, 'track_server', lambda *a: None)
    monkeypatch.setattr(profile, 'observed_engines', lambda *a: dict(identities=[dict(pid=42)]))
    monkeypatch.setattr(profile, 'owned_clean', lambda *a: state['tick'] >= 4)
    monkeypatch.setattr(profile, 'signal_target', lambda *a: state['signals'].append('harness') or True)

    async def sleep(_):
        state['tick'] += 1
        assert state['tick'] <= 10, 'Lifecycle mock failed to make bounded progress'
        await asyncio.sleep(0)

    async def to_thread(function, *args, **kwargs):
        # All subprocess functions below are in-memory fakes, so real executor
        # scheduling is unrelated to this collector/target lifecycle regression.
        return function(*args, **kwargs)

    monkeypatch.setattr(profile, 'asyncio', SimpleNamespace(
        Event=asyncio.Event, get_running_loop=asyncio.get_running_loop,
        CancelledError=asyncio.CancelledError, sleep=sleep, to_thread=to_thread))

    def read(path):
        if path == root / 'status.json' and state['launched']:
            if fail_harness and state['tick'] >= 1:
                return dict(status='failed', error='fixture failure')
            return dict(status='diagnostic_completed' if state['tick'] >= 4 else 'running')
        if path == root / 'pair-00-P1/saturated/window.json' and state['launched']:
            return dict(begin=70., end=370.)
        return original_read(path)

    monkeypatch.setattr(profile, 'read', read)

    class Collector:
        pid = 888
        returncode = 0

        def __init__(self, command, **kwargs):
            state['launched'] = True
            claim = original_read(root / 'nsys-claim.json')
            write(output / 'target.json', target_row(root, output, claim))
            (output / 'timeline.nsys-rep').write_bytes(b'fake collector report')

        def poll(self):
            return 0 if state['tick'] >= 1 else None

    def tool(command, **kwargs):
        if command[1] == 'export':
            assert state['tick'] >= 4, 'Export ran before target cleanup'
            state['exported'] = True
            with sqlite3.connect(output / 'timeline.sqlite') as connection:
                connection.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER)')
                connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1,2,?)', (42 << 24,))
        return SimpleNamespace(returncode=0, stdout='')

    monkeypatch.setattr(profile, 'subprocess', SimpleNamespace(Popen=Collector, run=tool,
        check_output=lambda *a, **kw: 'nsys CPU fixture', STDOUT=profile.subprocess.STDOUT))
    args = SimpleNamespace(diagnostic_output=root, output=output, nsys=Path('/nsys'),
                           duration_seconds=45., offset_seconds=30.)
    if fail_harness:
        with pytest.raises(RuntimeError, match='harness failed'):
            asyncio.run(profile.run(args))
        assert state['signals'] == ['harness'] and not state['exported']
    else:
        asyncio.run(profile.run(args))
        assert state['exported'] and not state['signals']
        result = original_read(output / 'result.json')
        exited = next(row for row in result['events'] if row['event'] == 'collector_exited')
        assert exited['target_alive'] and result['status'] == 'completed_diagnostic'
    with pytest.raises(ValueError, match='fresh unclaimed'):
        profile.validate_prepared(root, output, 45., 30.)

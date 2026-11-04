"""CPU-only checks for console ownership, configuration and bounded output."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from vllm_kvmem.webui import manager as module
from vllm_kvmem.webui.manager import EngineManager, _OwnedProcess


pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux console')


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.03)
    assert predicate(), 'Timed out waiting for process state'


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


@pytest.fixture
def manager(tmp_path):
    instance = EngineManager(tmp_path / 'state')
    yield instance
    instance.close()


def local_profile(manager, tmp_path):
    value = manager.draft()
    model, tokenizer = tmp_path / 'model', tmp_path / 'tokenizer'
    model.mkdir(exist_ok=True)
    tokenizer.mkdir(exist_ok=True)
    (model / 'config.json').write_text('{"model_type":"qwen3_5"}')
    value.update(model=str(model), hf_config_path=str(model), tokenizer=str(tokenizer))
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        value['port'] = probe.getsockname()[1]
    return value


class FakeProcess:
    pid = 123456

    def __init__(self, argv, log, **kwargs):
        self.argv, self.log = argv, log
        self.code = None
        self.closed = False

    def poll(self):
        return self.code

    def close(self):
        self.closed = True
        self.code = 0 if self.code is None else self.code
        return self.code


def test_starter_is_packaged_and_drafts_do_not_require_model_files(manager):
    initial = manager.draft()
    assert initial['draft_method'] == 'mtp'
    assert initial['max_model_len'] == 16384
    initial['model'] = 'not-downloaded-yet'
    saved = manager.save(initial)
    assert Path(saved['model']).is_absolute()
    assert manager.draft() == saved
    initial['max_model_len'] = 1
    assert manager.draft()['max_model_len'] == 16384
    with pytest.raises(ValueError, match='Missing local model'):
        manager.preview(saved)


def test_saved_profile_reloads_and_invalid_save_preserves_it(manager):
    manager.save(manager.draft())
    before = manager._draft_path.read_bytes()
    with pytest.raises(ValueError, match='unknown fields'):
        manager.save({'unexpected': True})
    with pytest.raises(ValueError, match='finite JSON'):
        manager.save({'gpu_memory_utilization': float('nan')})
    with pytest.raises(ValueError, match='64 KiB'):
        manager.save({'model': 'x' * 65536})
    assert manager._draft_path.read_bytes() == before
    other = EngineManager(manager.state_dir)
    try:
        assert other.draft() == manager.draft()
        assert manager._draft_path.stat().st_mode & 0o077 == 0
    finally:
        other.close()


def test_explicit_profile_resolves_relative_paths_without_overwriting_source(tmp_path):
    source = tmp_path / 'input.json'
    source.write_text('{"model":"weights","tokenizer":"tok","hf_config_path":"weights"}')
    original = source.read_text()
    instance = EngineManager(tmp_path / 'state', source)
    try:
        assert instance.draft()['model'] == str(tmp_path / 'weights')
        instance.save(instance.draft())
        assert source.read_text() == original
    finally:
        instance.close()


def test_preview_uses_real_launch_validation_without_creating_run_directory(manager, tmp_path):
    profile = local_profile(manager, tmp_path)
    plan = manager.preview(profile)
    assert plan['profile']['model'] == profile['model']
    assert plan['profile']['max_num_batched_tokens'] == 2052
    assert plan['vllm_argv'][0] == 'serve'
    assert not (manager.state_dir / 'runs').exists()
    assert manager.status()['state'] == 'stopped'


@pytest.mark.parametrize('host', ['192.168.1.100', 'example.com', '127.0.0.2'])
def test_console_rejects_hosts_its_chat_bridge_cannot_connect_to(manager, tmp_path, host):
    profile = local_profile(manager, tmp_path)
    profile['host'] = host
    with pytest.raises(ValueError, match='loopback-accessible'):
        manager.preview(profile)
    with pytest.raises(ValueError, match='loopback-accessible'):
        manager.start(profile)


@pytest.mark.parametrize(('host', 'address'), [
    ('127.0.0.1', '127.0.0.1'), ('localhost', '127.0.0.1'),
    ('0.0.0.0', '127.0.0.1'), ('::', '[::1]'), ('::1', '[::1]'),
])
def test_native_url_uses_local_interface_for_wildcard_bind(host, address):
    assert EngineManager._engine_url({'host': host, 'port': 8000}) == f'http://{address}:8000'


def test_native_connection_lock_serializes_shutdown(manager):
    stopping = threading.Event()
    stopped = threading.Event()
    def close():
        stopping.set()
        manager.close()
        stopped.set()
    with manager.lock:
        thread = threading.Thread(target=close)
        thread.start()
        assert stopping.wait(1)
        assert not stopped.wait(.05)
    assert stopped.wait(1)
    thread.join(timeout=1)
    with pytest.raises(ValueError, match='closed'):
        manager.start(manager.draft())


def test_start_owns_only_new_process_and_distinct_runs(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_OwnedProcess', FakeProcess)
    monkeypatch.setattr(manager, '_health_ready', lambda *_: False)
    profile = local_profile(manager, tmp_path)
    first = manager.start(profile)
    process = manager._process
    run_dir = manager.run_dir
    assert first['state'] == 'loading'
    assert not run_dir.exists()
    assert process.argv[:4] == [sys.executable, '-m', 'vllm_kvmem', 'serve']
    assert process.argv[-1] == str(run_dir)
    saved_path = Path(process.argv[process.argv.index('--config') + 1])
    assert json.loads(saved_path.read_text()) == manager.active_profile
    with pytest.raises(ValueError, match='already running'):
        manager.start(profile)
    changed = dict(profile, max_model_len=32768)
    manager.save(changed)
    assert manager.active_profile['max_model_len'] == 16384
    manager.stop()
    assert process.closed
    assert manager.active_profile is None
    assert manager.base_url is None
    second = manager.start(profile)
    assert second['instance'] != first['instance']
    assert manager.run_dir != run_dir


def test_existing_listener_is_never_adopted(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_OwnedProcess', lambda *_args, **_kwargs: pytest.fail('must not launch'))
    profile = local_profile(manager, tmp_path)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', profile['port']))
        listener.listen()
        with pytest.raises(ValueError, match='existing service'):
            manager.start(profile)
        assert listener.getsockname()[1] == profile['port']
    assert manager.status()['state'] == 'stopped'


def test_health_requires_native_model_identity_and_tracks_process_failure(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_OwnedProcess', FakeProcess)
    ready = threading.Event()
    monkeypatch.setattr(manager, '_health_ready', lambda *_: ready.is_set())
    manager.start(local_profile(manager, tmp_path))
    ready.set()
    wait_until(lambda: manager.status()['state'] == 'ready')
    manager._process.code = 9
    state = manager.status()
    assert state['state'] == 'failed'
    assert state['exit_code'] == 9
    assert manager.base_url is None
    assert state['error']


def test_health_needs_health_and_matching_v1_models(manager, monkeypatch):
    class Response:
        status = 200
        def __init__(self, body=b''):
            self.body = body
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self, _limit):
            return self.body
    calls = []
    payload = {'data': [{'id': 'other'}]}
    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            return Response(json.dumps(payload).encode())
    monkeypatch.setattr(module.urllib.request, 'build_opener', lambda *_: Opener())
    assert not manager._health_ready('http://127.0.0.1:8000', 'kvmem')
    payload['data'] = [{'id': 'kvmem'}]
    assert manager._health_ready('http://127.0.0.1:8000', 'kvmem')
    assert calls == ['http://127.0.0.1:8000/health', 'http://127.0.0.1:8000/v1/models'] * 2


def test_logs_bound_bytes_lines_and_preserve_complete_records(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_OwnedProcess', FakeProcess)
    monkeypatch.setattr(manager, '_health_ready', lambda *_: False)
    manager.start(local_profile(manager, tmp_path))
    manager._log.write(('行' * 100 + '\n').encode() * 3000 + b'partial')
    manager._log.flush()
    snapshot = manager.logs()
    assert snapshot['protocol'] == 1
    assert snapshot['pending'] and snapshot['truncated']
    assert snapshot['text'].endswith('\n')
    assert '\ufffd' not in snapshot['text']
    assert len(snapshot['text'].encode()) <= module.LOG_LIMIT
    assert len(snapshot['text'].splitlines()) <= module.LOG_LINES
    assert manager.logs(snapshot['revision'])['text'] is None
    manager.stop()
    assert manager.logs()['text'].endswith('partial')
    assert manager.logs()['revision'] != snapshot['revision']


def test_logs_enforce_line_limit_below_byte_limit(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_OwnedProcess', FakeProcess)
    monkeypatch.setattr(manager, '_health_ready', lambda *_: False)
    manager.start(local_profile(manager, tmp_path))
    manager._log.write(b'line\n' * 2200)
    manager._log.flush()
    snapshot = manager.logs()
    assert snapshot['truncated']
    assert len(snapshot['text'].splitlines()) == 2000


def test_logs_bound_replacement_characters_and_redact_inherited_keys(tmp_path, monkeypatch):
    monkeypatch.setenv('VLLM_API_KEY', 'first-key,second-key')
    monkeypatch.setattr(module, '_OwnedProcess', FakeProcess)
    instance = EngineManager(tmp_path / 'state')
    monkeypatch.setattr(instance, '_health_ready', lambda *_: False)
    try:
        instance.start(local_profile(instance, tmp_path))
        instance._log.write((b'\xff' * 200 + b' first-key second-key\n') * 1000)
        instance._log.flush()
        snapshot = instance.logs()
        assert snapshot['truncated']
        assert len(snapshot['text'].encode()) <= module.LOG_LIMIT
        assert 'first-key' not in snapshot['text'] and 'second-key' not in snapshot['text']
        assert '<redacted>' in snapshot['text']
        assert instance.backend_headers == {'Authorization': 'Bearer first-key'}
        assert 'first-key' not in json.dumps(instance.status())
    finally:
        instance.close()


def test_real_owned_group_cleanup_leaves_unrelated_process_running(tmp_path):
    marker = tmp_path / 'child.pid'
    command = ('import subprocess,sys,time; from pathlib import Path; '
               'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
               f'Path({str(marker)!r}).write_text(str(p.pid)); time.sleep(60)')
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    owned = None
    try:
        with (tmp_path / 'log').open('wb') as log:
            owned = _OwnedProcess([sys.executable, '-c', command], log)
            wait_until(marker.exists)
            child_pid = int(marker.read_text())
            assert alive(owned.pid) and alive(child_pid)
            owned.close()
            wait_until(lambda: not alive(child_pid))
            assert not alive(owned.pid)
            assert unrelated.poll() is None
    finally:
        if owned:
            owned.close()
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_controller_death_closes_control_pipe_and_cleans_engine(tmp_path):
    marker = tmp_path / 'engine.pid'
    command = ('import os,time; from pathlib import Path; '
               f'Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)')
    controller_code = ('import sys,time; from vllm_kvmem.webui.manager import _OwnedProcess; '
                       f'log=open({str(tmp_path / "log")!r},"wb"); '
                       f'p=_OwnedProcess([sys.executable,"-c",{command!r}],log); time.sleep(60)')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    controller = subprocess.Popen([sys.executable, '-c', controller_code], env=env)
    try:
        wait_until(marker.exists)
        pid = int(marker.read_text())
        assert alive(pid)
        controller.kill()
        controller.wait(timeout=5)
        wait_until(lambda: not alive(pid))
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)


def test_engine_leader_exit_still_cleans_owned_descendants(tmp_path):
    marker = tmp_path / 'child.pid'
    command = ('import subprocess,sys; from pathlib import Path; '
               'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
               f'Path({str(marker)!r}).write_text(str(p.pid))')
    with (tmp_path / 'log').open('wb') as log:
        owned = _OwnedProcess([sys.executable, '-c', command], log)
        try:
            wait_until(marker.exists)
            child_pid = int(marker.read_text())
            wait_until(lambda: owned.poll() is not None)
            wait_until(lambda: not alive(child_pid))
            assert owned.poll() == 0
        finally:
            owned.close()

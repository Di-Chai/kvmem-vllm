"""Local launch ownership and bounded logs for the Linux web console.

The controller never discovers or adopts an existing engine. A separate small
supervisor retains the engine process-group leader until group cleanup begins;
closing its control pipe also cleans up workers if the controller disappears.
"""
from __future__ import annotations

import atexit
import copy
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid


PROFILE_LIMIT = 64 * 1024
LOG_LIMIT = 256 * 1024
LOG_LINES = 2000


def _profile(value):
    if not isinstance(value, dict):
        raise ValueError('The serving profile must be a JSON object')
    try:
        raw = json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError('The serving profile must contain finite JSON values') from error
    if len(raw.encode('utf-8')) > PROFILE_LIMIT:
        raise ValueError('The serving profile exceeds 64 KiB')
    from ...serving.serve import DEFAULTS, PATHS
    if set(value) - set(DEFAULTS) - set(PATHS) - {'kvmem', 'multi_user', 'linear_backend_per_quant'}:
        raise ValueError('The serving profile contains unknown fields')
    if 'kvmem' in value and not isinstance(value['kvmem'], dict):
        raise ValueError('kvmem must be an object')
    return json.loads(raw)


def _read_profile(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(PROFILE_LIMIT + 1)
    if len(raw) > PROFILE_LIMIT:
        raise ValueError('The serving profile exceeds 64 KiB')
    try:
        return _profile(json.loads(raw))
    except (UnicodeError, RecursionError) as error:
        raise ValueError('The serving profile must be valid UTF-8 JSON') from error


def _absolute_paths(profile, base):
    from ...serving.serve import PATHS
    value = _profile(profile)
    for key in PATHS:
        if isinstance(value.get(key), str) and value[key]:
            value[key] = str((base / Path(value[key]).expanduser()).resolve())
    for key, field in (('multi_user', 'users_file'), ('profiler_config', 'torch_profiler_dir')):
        nested = value.get(key)
        if isinstance(nested, dict) and isinstance(nested.get(field), str) and nested[field]:
            nested[field] = str((base / Path(nested[field]).expanduser()).resolve())
    return value


def _atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _supervise(control_fd, status_fd, argv):
    """Internal child entry: no package imports and no model work here."""
    stop = False
    child = None

    def request_stop(*_):
        nonlocal stop
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)
    try:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                 close_fds=True, start_new_session=True)
        os.write(status_fd, json.dumps({'pid': child.pid}).encode())
        os.close(status_fd)
        status_fd = None
        while not stop:
            if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                break
            if select.select([control_fd], [], [], .1)[0] and not os.read(control_fd, 1):
                break
        _signal_group(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                break
            time.sleep(.05)
        # The unreaped group leader prevents its PID being reused before this
        # final descendant cleanup, even if the leader exited on its own.
        _signal_group(child.pid, signal.SIGKILL)
        code = child.wait(timeout=5)
        return code if code >= 0 else 128 - code
    except BaseException as error:
        if status_fd is not None:
            try:
                os.write(status_fd, json.dumps({'error': str(error)}).encode())
            except OSError:
                pass
        if child is not None:
            _signal_group(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        print('KVMem console supervisor: ' + str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        os.close(control_fd)
        if status_fd is not None:
            os.close(status_fd)


class _OwnedProcess:
    def __init__(self, argv, log, *, env=None):
        read_fd, self.control_fd = os.pipe()
        status_read, status_write = os.pipe()
        self.supervisor = None
        try:
            self.supervisor = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--supervise',
                 str(read_fd), str(status_write), *argv],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env,
                close_fds=True, pass_fds=(read_fd, status_write), start_new_session=True)
        finally:
            os.close(read_fd)
            os.close(status_write)
            if self.supervisor is None:
                os.close(self.control_fd)
                self.control_fd = None
                os.close(status_read)
        try:
            if not select.select([status_read], [], [], 10)[0]:
                raise RuntimeError('Engine supervisor did not start')
            status = json.loads(os.read(status_read, 8192))
            if 'error' in status:
                raise RuntimeError(status['error'])
            self.pid = int(status['pid'])
        except BaseException:
            self.close()
            raise
        finally:
            os.close(status_read)

    def poll(self):
        return self.supervisor.poll()

    def close(self):
        if self.control_fd is not None:
            os.close(self.control_fd)
            self.control_fd = None
        if self.supervisor is not None:
            try:
                return self.supervisor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.supervisor.terminate()
                return self.supervisor.wait(timeout=15)


class EngineManager:
    """One explicitly owned native vLLM engine, independent of editable drafts."""

    def __init__(self, state_dir, config_path=None):
        if not sys.platform.startswith('linux'):
            raise ValueError('This web console currently supports Linux only')
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._draft_path = self.state_dir / 'profile.json'
        source = (Path(config_path).expanduser().resolve() if config_path is not None
                  else self._draft_path if self._draft_path.is_file()
                  else Path(__file__).parent / 'profiles' / 'serve-qwen-mtp.json')
        self._base = source.parent if config_path is not None else Path.cwd()
        self._draft = _absolute_paths(_read_profile(source), source.parent)
        self._lock = threading.RLock()
        self._process = None
        self._log = None
        self._log_path = None
        self._state = 'stopped'
        self._instance = uuid.uuid4().hex
        self._started_at = None
        self._error = None
        self._exit_code = None
        self._active_profile = None
        self._url = None
        self._log_model = None
        self._run_dir = None
        self._monitor_stop = threading.Event()
        self._monitor_thread = None
        self._closed = False
        self._keys = tuple(key.strip() for key in os.environ.get('VLLM_API_KEY', '').split(',') if key.strip())
        self._key = self._keys[0] if self._keys else ''
        atexit.register(self.close)

    @property
    def lock(self):
        """Serialize native connections with engine stop/restart transitions."""
        return self._lock

    @property
    def active_profile(self):
        with self._lock:
            return copy.deepcopy(self._active_profile)

    @property
    def base_url(self):
        with self._lock:
            return self._url if self._state in ('loading', 'ready') else None

    @property
    def backend_headers(self):
        return {'Authorization': 'Bearer ' + self._key} if self._key else {}

    @property
    def run_dir(self):
        with self._lock:
            return self._run_dir

    def draft(self):
        with self._lock:
            return copy.deepcopy(self._draft)

    def save(self, profile):
        value = _absolute_paths(profile, self._base)
        with self._lock:
            _atomic_json(self._draft_path, value)
            self._draft = value
            return copy.deepcopy(value)

    def _plan(self, profile, run_dir):
        from ...serving.serve import build_launch
        value = _absolute_paths(profile, self._base)
        plan = build_launch(value, self._base, run_dir)
        if plan['profile']['host'] not in ('127.0.0.1', 'localhost', '0.0.0.0', '::1', '::'):
            raise ValueError('The console requires a loopback-accessible engine; use 127.0.0.1 or 0.0.0.0')
        return plan

    def preview(self, profile):
        return self._plan(profile, self.state_dir / 'runs' / ('preview-' + uuid.uuid4().hex))

    @staticmethod
    def _engine_url(profile):
        host = profile['host']
        host = '127.0.0.1' if host in ('0.0.0.0', 'localhost') else '::1' if host == '::' else host
        return f'http://{("[" + host + "]") if ":" in host else host}:{profile["port"]}'

    @staticmethod
    def _check_port(profile):
        host = profile['host']
        with socket.socket(socket.AF_INET6 if ':' in host else socket.AF_INET) as probe:
            try:
                probe.bind((host, profile['port']))
            except OSError as error:
                raise ValueError('The inference address is unavailable; an existing service will not be adopted or stopped') from error

    def start(self, profile):
        with self._lock:
            if self._closed:
                raise ValueError('The engine manager is closed')
            self._refresh_exit()
            if self._process is not None:
                raise ValueError('An owned engine is already running; stop it before starting another')
            session = uuid.uuid4().hex
            run_dir = self.state_dir / 'runs' / session
            plan = self._plan(profile, run_dir)
            resolved = plan['profile']
            if resolved.get('multi_user'):
                raise ValueError('The first web console supports direct serve profiles; multi_user requires its own gateway')
            self._check_port(resolved)
            # serve itself creates run_dir exclusively. Keep launcher inputs in
            # a different directory so its fresh-directory guard is preserved.
            launch_dir = self.state_dir / 'launches' / session
            launch_dir.mkdir(parents=True, mode=0o700)
            config = launch_dir / 'profile.json'
            _atomic_json(config, resolved)
            log_path = launch_dir / 'engine.log'
            fd = os.open(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            log = os.fdopen(fd, 'wb')
            argv = [sys.executable, '-m', 'vllm_kvmem', 'serve', '--config', str(config), '--run-dir', str(run_dir)]
            self._instance, self._started_at = session, int(time.time() * 1000)
            self._log_path, self._log_model = log_path, Path(resolved['model']).name
            self._error, self._exit_code = None, None
            self._active_profile, self._url = None, None
            try:
                process = _OwnedProcess(argv, log, env=dict(os.environ))
            except BaseException as error:
                log.close()
                self._state, self._error = 'failed', str(error)
                raise
            self._process, self._log = process, log
            self._active_profile = copy.deepcopy(resolved)
            self._url, self._run_dir = self._engine_url(resolved), run_dir
            self._state = 'loading'
            self._monitor_stop = threading.Event()
            self._monitor_thread = threading.Thread(target=self._monitor, args=(session, self._monitor_stop),
                                                    name='kvmem-webui-health', daemon=True)
            self._monitor_thread.start()
            return self.status()

    def _refresh_exit(self):
        if self._process is None:
            return
        code = self._process.poll()
        if code is None:
            return
        self._process.close()
        self._process = None
        if self._log:
            self._log.close()
            self._log = None
        self._exit_code = code
        self._state = 'exited' if code == 0 else 'failed'
        self._error = None if code == 0 else f'Engine exited with code {code}; see the engine log'
        self._monitor_stop.set()

    def _health_ready(self, url, model):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(urllib.request.Request(url + '/health', headers=self.backend_headers), timeout=1) as response:
                if response.status != 200:
                    return False
            with opener.open(urllib.request.Request(url + '/v1/models', headers=self.backend_headers), timeout=1) as response:
                data = response.read(65537)
                if len(data) > 65536:
                    return False
                payload = json.loads(data)
                return any(item.get('id') == model for item in payload.get('data', []) if isinstance(item, dict))
        except (OSError, ValueError, TypeError, AttributeError, urllib.error.URLError):
            return False

    def _monitor(self, session, stop):
        while not stop.is_set():
            with self._lock:
                if session != self._instance:
                    return
                self._refresh_exit()
                if self._process is None:
                    return
                url, model = self._url, self._active_profile['served_model_name']
            ready = self._health_ready(url, model)
            with self._lock:
                if session != self._instance or stop.is_set():
                    return
                self._refresh_exit()
                if self._process is not None:
                    self._state = 'ready' if ready else 'loading'
            stop.wait(1)

    def status(self):
        with self._lock:
            self._refresh_exit()
            profile = copy.deepcopy(self._active_profile)
            return {'state': self._state, 'instance': self._instance, 'started_at_ms': self._started_at,
                    'error': self._error, 'exit_code': self._exit_code,
                    'pid': self._process.pid if self._process is not None else None,
                    'base_url': self.base_url, 'url': self.base_url,
                    'active_profile': profile,
                    'active_config': {'parameters': {'Model': profile['model'], 'Backend': 'vllm'}} if profile else None}

    def stop(self):
        with self._lock:
            self._monitor_stop.set()
            if self._process is not None:
                self._process.close()
                self._process = None
            if self._log is not None:
                self._log.close()
                self._log = None
            self._state, self._error = 'stopped', None
            self._exit_code = None
            self._active_profile, self._url = None, None
            return self.status()

    def close(self):
        with self._lock:
            self._closed = True
            self.stop()
        atexit.unregister(self.close)

    def logs(self, revision=None):
        with self._lock:
            self._refresh_exit()
            running = self._process is not None
            result = {'protocol': 1, 'source': 'file', 'backend': 'vllm', 'session': self._instance,
                      'model': self._log_model, 'started_at': self._started_at, 'started_at_ms': self._started_at,
                      'running': running, 'exit_code': self._exit_code, 'available': False,
                      'limit_bytes': LOG_LIMIT, 'bytes': 0, 'updated_at': None,
                      'truncated': False, 'pending': False, 'text': '', 'revision': self._instance + ':missing'}
            if self._log_path is None or not self._log_path.is_file():
                return result
            with self._log_path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                size = stat.st_size
                stamp = f'{self._instance}:{stat.st_ino}:{size}:{stat.st_mtime_ns}:{running}'
                result.update(available=True, bytes=size, updated_at=stat.st_mtime_ns // 1_000_000,
                              revision=stamp, truncated=size > LOG_LIMIT)
                if size and running:
                    stream.seek(size - 1)
                    result['pending'] = stream.read(1) not in (b'\r', b'\n')
                if revision == stamp:
                    result['text'] = None
                    return result
                offset = max(0, size - LOG_LIMIT)
                stream.seek(offset)
                data = stream.read(size - offset)
            if offset:
                first = min((index for index in (data.find(b'\r'), data.find(b'\n')) if index >= 0), default=-1)
                data = data[first + 1:] if first >= 0 else b''
            if running and data and data[-1:] not in (b'\r', b'\n'):
                data = data[:max(data.rfind(b'\r'), data.rfind(b'\n')) + 1]
            lines = data.decode('utf-8', errors='replace').splitlines(keepends=True)
            result['truncated'] = result['truncated'] or len(lines) > LOG_LINES
            lines = lines[-LOG_LINES:]
            for index, line in enumerate(lines):
                for key in sorted(self._keys, key=len, reverse=True):
                    line = line.replace(key, '<redacted>')
                lines[index] = line
            sizes = [len(line.encode('utf-8')) for line in lines]
            total, first = sum(sizes), 0
            # Invalid UTF-8 is rendered as replacement characters and can grow
            # beyond the raw byte limit. Keep the rendered snapshot bounded too.
            while total > LOG_LIMIT:
                total -= sizes[first]
                first += 1
                result['truncated'] = True
            text = ''.join(lines[first:])
            result['text'] = text
            return result


if __name__ == '__main__':
    if len(sys.argv) < 5 or sys.argv[1] != '--supervise':
        raise SystemExit('This internal supervisor requires a controller pipe and engine command')
    raise SystemExit(_supervise(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]))

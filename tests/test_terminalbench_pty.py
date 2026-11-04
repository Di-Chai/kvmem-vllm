"""The recorder must drain output while a pasted input stream is backpressured."""
import errno
import os
import signal
import socket
import threading
import time

import pytest

from scripts.terminalbench_pty import nonblocking, pump


@pytest.mark.parametrize('short_writes', [False, True])
def test_duplex_backpressure_preserves_bytes_and_recording(monkeypatch, short_writes):
    master, child = socket.socketpair()
    for sock in (master, child):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2048)
    input_r, input_w = os.pipe()
    output_r, output_w = os.pipe()
    signal_r, signal_w = os.pipe()
    body = bytes(range(256))*2048
    suffix = b'final output after child input completes\n'*2048
    captured, input_recording, output_recording = [], [], []
    done, timed_out = threading.Event(), []
    original_write = os.write
    write_count = 0

    def write(fd, data):
        nonlocal write_count
        if fd in (master.fileno(), output_w):
            write_count += 1
            if write_count % 7 == 0:
                raise BlockingIOError(errno.EAGAIN, 'forced backpressure')
            return original_write(fd, data[:71])
        return original_write(fd, data)

    if short_writes:
        monkeypatch.setattr(os, 'write', write)

    def producer():
        remaining = body
        while remaining:
            remaining = remaining[original_write(input_w, remaining):]
        os.close(input_w)

    def echo():
        # The child fills its output before reading any keyboard input. The
        # original all-input-write loop deadlocks under this duplex pressure.
        child.sendall(b'initial output\n'*4096)
        received = bytearray()
        while len(received) < len(body):
            data = child.recv(1024)
            assert data
            received.extend(data)
            child.sendall(data)
        assert bytes(received) == body
        child.sendall(suffix)
        child.shutdown(socket.SHUT_WR)

    def consumer():
        while True:
            data = os.read(output_r, 2048)
            if not data:
                return
            captured.append(data)
            time.sleep(.0001)

    def watchdog():
        if not done.wait(8):
            timed_out.append(True)
            original_write(signal_w, bytes([signal.SIGTERM]))

    workers = [threading.Thread(target=f, daemon=True) for f in (producer, echo, consumer, watchdog)]
    for thread in workers:
        thread.start()
    try:
        pump(master.fileno(), [input_r, input_r], output_w, signal_r,
             lambda data: (input_recording.append(data), data)[1],
             output_recording.append, lambda sig: sig == signal.SIGTERM,
             high_water=4096)
        done.set()
        assert os.get_blocking(master.fileno())
        assert os.get_blocking(output_w) and os.get_blocking(input_r)
        os.close(output_w)
        for thread in workers:
            thread.join(1)
        assert not timed_out
        assert not any(t.is_alive() for t in workers)
        expected = b'initial output\n'*4096+body+suffix
        assert b''.join(captured) == expected
        assert b''.join(output_recording) == expected
        assert b''.join(input_recording) == body
    finally:
        done.set()
        master.close()
        child.close()
        for fd in (input_r, input_w, output_r, output_w, signal_r, signal_w):
            try:
                os.close(fd)
            except OSError:
                pass


def test_nonblocking_restores_original_flags_on_exception():
    r, w = os.pipe()
    os.set_blocking(r, False)
    try:
        with pytest.raises(RuntimeError), nonblocking([r, w, w]):
            assert not os.get_blocking(r) and not os.get_blocking(w)
            raise RuntimeError('injected')
        assert not os.get_blocking(r) and os.get_blocking(w)
    finally:
        os.close(r)
        os.close(w)


def test_invalid_buffer_limit_rejected():
    with pytest.raises(ValueError, match='Positive'):
        pump(None, [], None, None, None, None, None, high_water=0)


def test_legacy_adapter_preserves_append_offset_and_stdin_opt_in(monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts import terminalbench_pty as adapter
    calls=[]
    term=SimpleNamespace(get_size=lambda:(80,24),raw=object())
    monkeypatch.setitem(sys.modules,'asciinema',SimpleNamespace(term=term))
    monkeypatch.setattr(adapter,'_record',lambda *args,**kwargs:calls.append((args,kwargs)))
    writer=object();environment={'SHELL':'bash'}
    adapter.record_legacy(['bash'],writer,environment,False,11.5)
    adapter.record_legacy(['bash'],writer,environment,True,23.0)
    assert calls[0][0][:3]==(['bash'],environment,writer)
    assert calls[0][1]==dict(time_offset=11.5,rec_stdin=False)
    assert calls[1][1]==dict(time_offset=23.0,rec_stdin=True)


def test_legacy_signal_pipe_restores_handlers_and_closes_descriptors():
    from scripts.terminalbench_pty import signal_pipe
    old=signal.getsignal(signal.SIGUSR1)
    with signal_pipe([signal.SIGUSR1]) as reader:
        os.kill(os.getpid(),signal.SIGUSR1)
        assert os.read(reader,1)==bytes([signal.SIGUSR1])
    assert signal.getsignal(signal.SIGUSR1)==old
    with pytest.raises(OSError):os.fstat(reader)


def asciinema_api(monkeypatch, version):
    """Minimal public contracts from the four explicitly supported releases."""
    import sys
    from types import ModuleType
    modules = {}
    for name in ('asciinema', 'asciinema.term', 'asciinema.recorder', 'asciinema.pty_',
                 'asciinema.tty_', 'asciinema.asciicast', 'asciinema.asciicast.v2',
                 'asciinema.asciicast.raw', 'asciinema.__main__'):
        module = ModuleType(name); module.__path__ = []; modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if '.' in name:
            parent, attr = name.rsplit('.', 1)
            setattr(modules[parent], attr, module)
    package = modules['asciinema']; package.__version__ = version
    modules['asciinema.term'].get_size = lambda: (80, 24)
    modules['asciinema.term'].raw = lambda fd: None
    modules['asciinema.pty_'].SignalFD = lambda signals: None
    modules['asciinema.tty_'].raw = lambda fd: None
    class ByteWriter:
        def write_stdout(self, data): pass
        def write_stdin(self, data): pass
    modules['asciinema.asciicast.v2'].async_writer = ByteWriter
    modules['asciinema.asciicast.raw'].writer = ByteWriter
    def old_pty(command, writer, env=os.environ, rec_stdin=False):
        raise AssertionError('Stock blocking transport must not run')
    def old_record(path, *, command=None, append=False, rec_stdin=False, command_env=None,
                   writer=None, record=old_pty):
        return record(['sh', '-c', command], writer, command_env, rec_stdin)
    package.record_asciicast = old_record
    if version == '2.1.0':
        def recording(path, record=old_pty): return record
    else:
        def recording(path, record_=old_pty): return record_
    modules['asciinema.recorder'].record = recording
    return modules


def test_2_0_bytes_writer_keeps_upstream_clock_stdin_and_shell_arguments(monkeypatch):
    from scripts import terminalbench_pty as adapter
    modules = asciinema_api(monkeypatch, '2.0.2')
    # This release has no recorder module; import of that module was the bug.
    import sys
    monkeypatch.delitem(sys.modules, 'asciinema.recorder')
    del modules['asciinema'].recorder
    output = []; calls = []
    class UpstreamWriter:
        # A one-argument writer owns its clock/append offset, as in 2.0.2.
        def write_stdout(self, data): output.append(('output', 17.5, data))
        def write_stdin(self, data): output.append(('input', 17.5, data))
    def transport(command, env, writer, size, notify, keys, signal_context, raw_context, **kwargs):
        calls.append((command, env, kwargs, keys))
        writer.write_stdout(999, b'output bytes')
        if kwargs['rec_stdin']:
            writer.write_stdin(999, b'input bytes')
    monkeypatch.setattr(adapter, '_record', transport)
    adapter.install_duplex()
    command = "printf '%s' 'spaces; $(literal) `literal`'"
    environment = {'SHELL': '/bin/sh', 'TERM': 'xterm'}
    for stdin in (False, True):
        modules['asciinema'].record_asciicast('out.cast', command=command, append=True,
            rec_stdin=stdin, command_env=environment, writer=UpstreamWriter())
    assert output == [('output', 17.5, b'output bytes'), ('output', 17.5, b'output bytes'),
                      ('input', 17.5, b'input bytes')]
    assert calls == [(['sh', '-c', command], environment, {'rec_stdin': False}, {}),
                     (['sh', '-c', command], environment, {'rec_stdin': True}, {})]


@pytest.mark.parametrize('version', ['2.1.0', '2.2.0', '2.4.0'])
def test_existing_versions_keep_their_original_recording_callbacks(monkeypatch, version):
    from scripts import terminalbench_pty as adapter
    modules = asciinema_api(monkeypatch, version)
    adapter.install_duplex()
    recorder = modules['asciinema.recorder'].record
    expected = adapter.record_legacy if version == '2.1.0' else adapter.record
    assert recorder('out.cast') is expected
    parameter = 'record' if version == '2.1.0' else 'record_'
    with pytest.raises(ValueError, match='Unexpected recorder override'):
        recorder('out.cast', **{parameter: lambda: None})
    with pytest.raises(ValueError, match='Unexpected recorder override'):
        recorder('out.cast', lambda: None)


@pytest.mark.parametrize('defect', ['version', 'entry', 'hook', 'pty_signature', 'cast_writer',
                                  'raw_writer', 'terminal_api'])
def test_version_probe_rejects_incompatible_recording_api_before_cli(monkeypatch, defect):
    from scripts import terminalbench_pty as adapter
    modules = asciinema_api(monkeypatch, '2.0.2'); package = modules['asciinema']
    cli_called = []
    modules['asciinema.__main__'].main = lambda: cli_called.append(True)
    if defect == 'version': package.__version__ = '2.0.3'
    elif defect == 'entry': package.record_asciicast = None
    elif defect == 'hook': package.record_asciicast = lambda path: None
    elif defect == 'pty_signature':
        def recording(path, record=lambda command: None): pass
        package.record_asciicast = recording
    elif defect in ('cast_writer', 'raw_writer'):
        class TimestampWriter:
            def write_stdout(self, timestamp, data): pass
            def write_stdin(self, timestamp, data): pass
        if defect == 'cast_writer': modules['asciinema.asciicast.v2'].async_writer = TimestampWriter
        else: modules['asciinema.asciicast.raw'].writer = TimestampWriter
    else: modules['asciinema.term'].raw = None
    with pytest.raises(RuntimeError): adapter.main()
    assert not cli_called

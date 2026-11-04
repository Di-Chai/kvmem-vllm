"""Opt-in asciinema 2.x duplex transport, uploaded only to new trial containers.

The stock recorder writes all pending keyboard input while it is also the sole
reader of the child's echoed output. A large paste can fill both PTYs and
deadlock that loop. Use nonblocking writes and bounded queues in both directions.
Keep the upstream CLI, cast writer, shell, terminal settings and key bindings.
"""
from contextlib import contextmanager
import errno
import os
import pty
import select
import signal
import struct
import termios
import time


@contextmanager
def nonblocking(fds):
    modes = {fd: os.get_blocking(fd) for fd in set(fds)}
    try:
        for fd in modes:
            os.set_blocking(fd, False)
        yield
    finally:
        for fd, mode in modes.items():
            os.set_blocking(fd, mode)


def pump(master, inputs, output, signal_fd, on_input, on_output, on_signal,
         high_water=65536):
    """Forward bytes in order, retrying short writes/EAGAIN without busy waits.

    ``on_input`` may consume an upstream recorder control key; other bytes are
    unchanged. Stop reading a full direction while continuing to drain the
    opposite direction. Each pending buffer is at most ``high_water`` bytes.
    """
    if high_water < 1:
        raise ValueError('Positive buffer bound required')
    inputs = set(inputs)  # stdin and /dev/tty can be the same descriptor.
    to_child, to_terminal = bytearray(), bytearray()
    master_open = True
    with nonblocking([master, output, *inputs]):
        while master_open or to_terminal:
            readers = [signal_fd]
            if master_open and len(to_terminal) < high_water:
                readers.append(master)
            if master_open and len(to_child) < high_water:
                readers.extend(inputs)
            writers = ([output] if to_terminal else [])
            if master_open and to_child:
                writers.append(master)
            try:
                ready, writable, _ = select.select(readers, writers, [])
            except InterruptedError:
                continue
            # Drain output before accepting more input. Never block the reader
            # of child output behind a write to that same child's input queue.
            for fd, buffer in ((output, to_terminal), (master, to_child)):
                if fd not in writable:
                    continue
                try:
                    count = os.write(fd, buffer)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError as error:
                    if fd != master or error.errno != errno.EIO:
                        raise
                    master_open = False
                    inputs.clear()
                    to_child.clear()
                    continue
                if count == 0:
                    raise OSError('Zero-length PTY write')
                del buffer[:count]
            if master in ready and master_open:
                try:
                    data = os.read(master, min(4096, high_water-len(to_terminal)))
                except (BlockingIOError, InterruptedError):
                    data = None
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    data = b''  # Linux PTY EOF after child exits.
                if data == b'':
                    master_open = False
                    inputs.clear()
                    to_child.clear()
                elif data:
                    to_terminal.extend(data)
                    on_output(data)
            for fd in sorted(inputs.intersection(ready)):
                remaining = high_water-len(to_child)
                if remaining <= 0:
                    break
                try:
                    data = os.read(fd, min(1024, remaining))
                except (BlockingIOError, InterruptedError):
                    continue
                if not data:
                    inputs.discard(fd)
                else:
                    payload = on_input(data)
                    if len(payload) > len(data):
                        raise ValueError('Recorder input filter expanded bytes')
                    to_child.extend(payload)
            if signal_fd in ready:
                data = os.read(signal_fd, 1024)
                for number in data:
                    if on_signal(number):
                        return


@contextmanager
def signal_pipe(signals):
    """The 2.1 recorder exposes no SignalFD; restore its process signal state."""
    reader, writer = os.pipe()
    os.set_blocking(writer, False)
    previous = signal.set_wakeup_fd(writer)
    handlers = {}
    try:
        for number in signals:
            handlers[number] = signal.signal(number, lambda *_: None)
        yield reader
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
        signal.set_wakeup_fd(previous)
        os.close(reader)
        os.close(writer)


def _record(command, env, writer, get_tty_size, notify, key_bindings,
            signal_context, raw_context, tty_stdin_fd=0, tty_stdout_fd=1,
            time_offset=0, rec_stdin=True):
    started = time.time()-time_offset
    paused = None
    prefix_mode = False
    prefix_key, pause_key = key_bindings.get('prefix'), key_bindings.get('pause')

    def resize():
        import fcntl
        cols, rows = get_tty_size()
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))

    def received(data):
        nonlocal started, paused, prefix_mode
        if not prefix_mode and prefix_key and data == prefix_key:
            prefix_mode = True
            return b''
        if prefix_mode or (not prefix_key and data == pause_key):
            prefix_mode = False
            if data == pause_key:
                if paused is not None:
                    started += time.time()-paused
                    paused = None
                    notify('Resumed recording')
                else:
                    paused = time.time()
                    notify('Paused recording')
            return b''
        if rec_stdin and paused is None:
            writer.write_stdin(time.time()-started, data)
        return data

    def emitted(data):
        if paused is None:
            writer.write_stdout(time.time()-started, data)

    def signalled(number):
        if number == signal.SIGWINCH:
            resize()
        # SIGCHLD wakes the pump; read the final child output through PTY EOF.
        return number in (signal.SIGHUP, signal.SIGTERM, signal.SIGQUIT)

    pid, master = pty.fork()
    if pid == pty.CHILD:
        os.execvpe(command[0], command, env)
    signals = [signal.SIGCHLD, signal.SIGHUP, signal.SIGTERM,
               signal.SIGQUIT, signal.SIGWINCH]
    inputs = [tty_stdin_fd]
    if not os.isatty(0):
        inputs.append(0)
    try:
        resize()
        with signal_context(signals) as signal_fd, raw_context(tty_stdin_fd):
            pump(master, inputs, tty_stdout_fd, signal_fd, received, emitted, signalled)
    finally:
        os.close(master)
        os.waitpid(pid, 0)


def record(command, env, writer, get_tty_size, notify, key_bindings,
           tty_stdin_fd=0, tty_stdout_fd=1):
    from asciinema.pty_ import SignalFD
    from asciinema.tty_ import raw
    return _record(command, env, writer, get_tty_size, notify, key_bindings,
                   SignalFD, raw, tty_stdin_fd, tty_stdout_fd)


def record_legacy(command, writer, env=os.environ, rec_stdin=False, time_offset=0,
                  notifier=None, key_bindings=None):
    # Ubuntu 22.04 ships 2.1: its writer, append offset and input opt-in belong
    # to the original recorder; adapt only the PTY callback signature.
    from asciinema import term
    notify = notifier.notify if notifier is not None else lambda _: None
    return _record(command, env, writer, term.get_size, notify, key_bindings or {},
                   signal_pipe, term.raw, time_offset=time_offset, rec_stdin=rec_stdin)


def record_2_0(command, writer, env=os.environ, rec_stdin=False):
    # In 2.0.2 the async/raw writer takes bytes only. Its own clock also owns
    # append offsets; forwarding the newer transport timestamp would either
    # raise TypeError or apply that offset twice. Keep the upstream writer.
    from asciinema import term

    class ByteWriter:
        def write_stdout(self, _elapsed, data):
            writer.write_stdout(data)

        def write_stdin(self, _elapsed, data):
            writer.write_stdin(data)

    return _record(command, env, ByteWriter(), term.get_size, lambda _: None, {},
                   signal_pipe, term.raw, rec_stdin=rec_stdin)


def install_duplex():
    """Qualify the real recording API even when the CLI only asks --version."""
    import inspect
    import asciinema
    version = getattr(asciinema, '__version__', None)
    if version not in ('2.0.2', '2.1.0', '2.2.0', '2.4.0'):
        raise RuntimeError('Unqualified asciinema version: '+str(version))
    if version in ('2.0.2', '2.1.0'):
        from asciinema import term
        dependencies = (term.get_size, term.raw)
    else:
        from asciinema.pty_ import SignalFD
        from asciinema.tty_ import raw
        dependencies = (SignalFD, raw)
    if not all(callable(value) for value in dependencies):
        raise RuntimeError('Asciinema terminal API is unavailable')
    if version == '2.0.2':
        target, attribute, parameter, callback = asciinema, 'record_asciicast', 'record', record_2_0
        # Both public CLI formats use a bytes-only writer in this release.
        from asciinema.asciicast.v2 import async_writer
        from asciinema.asciicast.raw import writer as raw_writer
        try:
            for writer_type in (async_writer, raw_writer):
                for method in ('write_stdout', 'write_stdin'):
                    inspect.signature(getattr(writer_type, method)).bind(None, b'')
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError('Asciinema 2.0.2 bytes-only writer API changed') from error
    else:
        from asciinema import recorder
        target, attribute = recorder, 'record'
        parameter, callback = ('record', record_legacy) if version == '2.1.0' else ('record_', record)
    original = getattr(target, attribute, None)
    if not callable(original):
        raise RuntimeError('Asciinema recording entry point is unavailable: '+attribute)
    signature = inspect.signature(original)
    hook = signature.parameters.get(parameter)
    if (hook is None or hook.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            or not callable(hook.default)):
        raise RuntimeError('Asciinema recorder callback signature changed: '+str(inspect.signature(original)))
    if version == '2.0.2':
        try:
            inspect.signature(hook.default).bind(['sh'], object(), {}, False)
        except (TypeError, ValueError) as error:
            raise RuntimeError('Asciinema 2.0.2 PTY callback API changed') from error

    def duplex_record(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        if 'record_' in bound.arguments or 'record' in bound.arguments:
            raise ValueError('Unexpected recorder override')
        return original(*args, **kwargs, **{parameter: callback})

    setattr(target, attribute, duplex_record)


def main():
    install_duplex()
    from asciinema.__main__ import main as cli
    return cli()


if __name__ == '__main__':
    raise SystemExit(main())

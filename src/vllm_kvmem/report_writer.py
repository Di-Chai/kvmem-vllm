"""Opt-in KVMEM_LAZY_ARCHIVE_REPORT: keep report serialization off engine steps.

archive.json carries the bounded completion history (256 requests when
serving, about 0.7 MB) and was re-serialized with the pure-Python indenting
encoder two to three times per finished request, on the engine thread, in
steps where the GPU is waiting. With the flag:

* while any request is active or retiring, archive.json and the progress.json
  written with it are refreshed at most once per LAZY_REPORT_INTERVAL seconds;
* the report written when the runtime becomes idle, and the one written at
  shutdown, are never skipped, so the file on disk after a run is complete;
* the engine thread only snapshots counters; one writer thread serializes the
  latest snapshot (compact JSON, same keys and values) and replaces the file
  atomically.

Flag off keeps the synchronous, indented writes unchanged.
"""
import json
import os
from threading import Condition, Thread

LAZY_REPORT_INTERVAL = 1.0


def lazy_archive_report():
    value = os.getenv('KVMEM_LAZY_ARCHIVE_REPORT', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAZY_ARCHIVE_REPORT must be 0 or 1')
    return value == '1'


def write_report(output, name, report, *, indent=2):
    temporary = output / f'{name}.tmp'
    temporary.write_text(json.dumps(report, indent=indent))
    temporary.replace(output / f'{name}.json')


class LatestReportWriter:
    """One daemon thread; a newer snapshot of a report replaces an unwritten one."""

    def __init__(self, output):
        self.output = output
        self._condition = Condition()
        self._pending = {}
        self._writing = False
        self._closed = False
        self._error = None
        self.submitted = self.written = self.replaced = 0
        self._thread = Thread(target=self._run, name='kvmem-report-writer', daemon=True)
        self._thread.start()

    def submit(self, name, report):
        with self._condition:
            if self._closed:
                raise RuntimeError('Report writer is closed')
            self.replaced += name in self._pending
            self._pending.pop(name, None)       # newest snapshot goes last
            self._pending[name] = report
            self.submitted += 1
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending:
                    return
                name = next(iter(self._pending))
                report = self._pending.pop(name)
                self._writing = True
            try:
                write_report(self.output, name, report, indent=None)
            except BaseException as error:      # reported by flush on the engine thread
                with self._condition:
                    self._error = self._error or error
            finally:
                with self._condition:
                    self._writing = False
                    self.written += 1
                    self._condition.notify_all()

    def flush(self, timeout=30.0):
        with self._condition:
            if not self._condition.wait_for(lambda: not self._pending and not self._writing, timeout):
                raise TimeoutError('Deferred report writer did not drain')
            error, self._error = self._error, None
        if error is not None:
            raise error

    def close(self, timeout=30.0):
        try:
            self.flush(timeout)
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
            self._thread.join(timeout)

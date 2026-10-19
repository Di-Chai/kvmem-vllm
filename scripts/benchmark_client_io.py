"""Bounded request-local capture reads and compact incremental progress.

These helpers change client bookkeeping only. Full case results remain the
authoritative artifacts; callers retain their retirement and validation barriers.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from threading import RLock

VERSION = 'incremental-client-1'


class CompactResults:
    """Load prior results once; subsequent summaries never reread case files."""
    def __init__(self, rows, cases, fields):
        self.order = tuple(row['id'] for row in rows)
        if len(set(self.order)) != len(self.order):
            raise ValueError('Duplicate progress case identity')
        self.allowed = frozenset(self.order)
        self.fields = frozenset(fields) | {'id', 'status'}
        self._records = {}
        self._lock = RLock()
        for ident in self.order:
            path = Path(cases)/ident/'result.json'
            if path.exists():
                record = json.loads(path.read_text())
                if record.get('id') != ident:
                    raise ValueError('Persisted result identity differs from its case')
                self.put(record)

    def put(self, record):
        ident = record['id']
        if ident not in self.allowed or record.get('status') not in ('completed', 'error'):
            raise ValueError('Invalid compact result identity/status')
        compact = deepcopy({key:value for key,value in record.items() if key in self.fields})
        with self._lock:
            self._records[ident] = compact

    def snapshot(self):
        with self._lock:
            return deepcopy([self._records[ident] for ident in self.order if ident in self._records])


@dataclass(frozen=True)
class _CaptureMark:
    device: int
    inode: int
    offset: int
    anchor: bytes
    at_line_boundary: bool


class CaptureSlice:
    """Capture a log boundary before HTTP dispatch; parse its new rows once.

    A preexisting partial final line belongs to earlier traffic and is skipped.
    Trailing partial lines remain unparsed, as in the native graph audit reader.
    Rotation/truncation/rewrite is an error, never a whole-file fallback.
    """
    def __init__(self, paths):
        self._marks, self._cache = {}, {}
        self.bytes_read = self.records_read = 0
        for value in paths:
            path = Path(value)
            try:
                with path.open('rb') as stream:
                    stat = os.fstat(stream.fileno())
                    stream.seek(max(0,stat.st_size-64))
                    anchor = stream.read(stat.st_size-stream.tell())
                    self._marks[path] = _CaptureMark(stat.st_dev,stat.st_ino,stat.st_size,
                        anchor,not anchor or anchor.endswith(b'\n'))
            except FileNotFoundError:
                self._marks[path] = None

    def read(self, value):
        path = Path(value)
        if path not in self._marks:
            raise ValueError('Capture path was not declared before dispatch')
        if path in self._cache:
            return self._cache[path]
        mark = self._marks[path]
        try:
            with path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                if mark is not None:
                    if (stat.st_dev,stat.st_ino)!=(mark.device,mark.inode) or stat.st_size<mark.offset:
                        raise RuntimeError('Capture log rotated or truncated during request')
                    stream.seek(mark.offset-len(mark.anchor))
                    if stream.read(len(mark.anchor)) != mark.anchor:
                        raise RuntimeError('Capture prefix changed during request')
                    stream.seek(mark.offset)
                data = stream.read()
        except FileNotFoundError:
            if mark is not None:
                raise RuntimeError('Capture log disappeared during request') from None
            data = b''
        self.bytes_read += len(data)
        if mark is not None and not mark.at_line_boundary:
            _,separator,data = data.partition(b'\n')
            if not separator:
                data = b''
        data = data[:data.rfind(b'\n')+1]
        records = tuple(json.loads(line) for line in data.splitlines() if line.strip())
        self.records_read += len(records)
        self._cache[path] = records
        return records

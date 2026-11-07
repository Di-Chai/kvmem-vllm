"""Diagnostic-only cyclic-GC ranges, without changing collection policy."""
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time


class GCPhaseProbe:
    def __init__(self, scopes):
        self.scopes = scopes
        self.rows = []
        self.current = None
        self.saved = 0
        gc.callbacks.append(self.collection)

    def collection(self, phase, info):
        if phase == 'start':
            if not self.scopes.enabled():
                return
            generation = info['generation']
            self.current = dict(generation=generation, tid=threading.get_native_id(),
                                start_ns=time.perf_counter_ns(),
                                cpu_start_ns=time.thread_time_ns(),
                                automatic_collection=gc.isenabled())
            if generation == 2:
                # Scalar stack evidence only: retaining frame objects would
                # itself keep request cycles alive and distort this diagnosis.
                frame = sys._getframe(1)
                stack = []
                while frame is not None and len(stack) < 32:
                    stack.append(dict(file=frame.f_code.co_filename,
                                      function=frame.f_code.co_name,
                                      line=frame.f_lineno))
                    frame = frame.f_back
                self.current['caller_stack'] = stack
            label = 'python.gc.generation' + str(generation)
            self.scopes.labels.add(label)
            self.scopes.push('prefill_phase::' + label)
        elif phase == 'stop' and self.current is not None:
            row, self.current = self.current, None
            self.scopes.pop()
            row.update(end_ns=time.perf_counter_ns(),
                       cpu_ns=time.thread_time_ns()-row.pop('cpu_start_ns'),
                       collected=info['collected'], uncollectable=info['uncollectable'])
            self.rows.append(row)

    def save(self, output):
        # Installation runs at every worker boundary. Keep serialization and
        # synchronous file writes outside the active diagnostic window.
        if self.scopes.enabled() or not output or self.saved == len(self.rows):
            return
        snapshot = list(self.rows)
        path = Path(output) / ('python-gc-' + str(os.getpid()) + '.json')
        path.write_text(json.dumps(dict(diagnostic_only=True, policy_unchanged=True,
                                       rows=snapshot), indent=2) + '\n')
        self.saved = len(snapshot)

    def close(self):
        gc.callbacks.remove(self.collection)

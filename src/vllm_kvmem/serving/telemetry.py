"""Sample owned GPU process memory without synchronizing inference CUDA work.

NVML CLI observations are sampled maxima, not exact instantaneous peaks. RSS is
summed across API/GPU processes and may double-count shared host pages. Foreign
GPU work or sampling errors prevent an exclusive-performance qualification.
"""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
from threading import Event, Thread
import time


def gpu_process_sample(group_id):
    output = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5).stdout
    owned, foreign, unknown = [], [], []
    for row in csv.reader(io.StringIO(output)):
        if not row:
            continue
        pid, memory = int(row[0]), float(row[1])
        if memory < 0 or not memory < float('inf'):
            raise ValueError('Invalid GPU process memory observation')
        entry = {'pid': pid, 'gpu_memory_mib': memory}
        try:
            (owned if os.getpgid(pid) == group_id else foreign).append(entry)
        except ProcessLookupError:
            unknown.append(entry)
    host = []
    for pid in {group_id, *(item['pid'] for item in owned)}:
        try:
            status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
            host.append({'pid': pid, 'rss_kib': int(status['VmRSS'].split()[0]),
                         'high_water_rss_kib': int(status['VmHWM'].split()[0])})
        except (FileNotFoundError, KeyError, ProcessLookupError):
            continue
    return {'at': time.perf_counter(), 'owned': owned, 'foreign': foreign,
            'unknown': unknown, 'host': host}


class ResourceMonitor:
    def __init__(self, path, group_id, interval=2.0):
        if not 0.25 <= interval <= 60:
            raise ValueError('Resource sampling interval must be between 0.25 and 60 seconds')
        self.path, self.group_id, self.interval = Path(path), group_id, interval
        self.stop_event = Event()
        self.samples = self.errors = 0
        self.gpu_samples = 0
        self.foreign_pids, self.unknown_pids = set(), set()
        self.max_gpu_mib = self.max_sum_rss_kib = 0
        self.last_error = None
        self.thread = Thread(target=self._run, name='kvmem-resource-monitor', daemon=True)

    def _observe(self, sample):
        self.samples += 1
        self.gpu_samples += bool(sample['owned'])
        self.foreign_pids.update(x['pid'] for x in sample['foreign'])
        self.unknown_pids.update(x['pid'] for x in sample['unknown'])
        self.max_gpu_mib = max(self.max_gpu_mib, sum(x['gpu_memory_mib'] for x in sample['owned']))
        self.max_sum_rss_kib = max(self.max_sum_rss_kib, sum(x['rss_kib'] for x in sample['host']))

    def _run(self):
        try:
            with self.path.open('x') as output:
                while not self.stop_event.is_set():
                    try:
                        sample = gpu_process_sample(self.group_id)
                        self._observe(sample)
                    except Exception as error:
                        self.errors += 1
                        self.last_error = repr(error)
                        sample = {'at': time.perf_counter(), 'error': self.last_error}
                    output.write(json.dumps(sample) + '\n')
                    output.flush()
                    self.stop_event.wait(self.interval)
        except Exception as error:
            self.errors += 1
            self.last_error = repr(error)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        complete = not self.thread.is_alive()
        return {'scope': 'sampled owned API/GPU processes over server lifetime; startup included',
            'interval_seconds': self.interval, 'samples': self.samples, 'gpu_samples': self.gpu_samples,
            'sampling_errors': self.errors, 'last_error': self.last_error, 'monitor_stopped': complete,
            'sampled_max_owned_gpu_memory_mib': self.max_gpu_mib if self.gpu_samples else None,
            'sampled_max_sum_process_rss_kib': self.max_sum_rss_kib if self.samples else None,
            'foreign_gpu_pids': sorted(self.foreign_pids), 'unknown_gpu_pids': sorted(self.unknown_pids),
            'exclusive_gpu_observed': bool(complete and self.gpu_samples and not self.errors
                                          and not self.foreign_pids and not self.unknown_pids)}

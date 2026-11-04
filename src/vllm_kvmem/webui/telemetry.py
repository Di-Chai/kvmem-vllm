"""Bounded, read-only telemetry for the local console.

Engine-wide Prometheus counters and console request metadata have different
scopes. Missing measurements stay unknown; configured budgets are not usage.
"""
import copy
import csv
import io
import math
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request


def parse_metrics(text):
    """Collect selected native gauges, summing disjoint worker/model series."""
    names = {'vllm:num_requests_running', 'vllm:num_requests_waiting',
             'vllm:gpu_cache_usage_perc', 'vllm:kv_cache_usage_perc',
             'vllm:prompt_tokens_total', 'vllm:generation_tokens_total'}
    values, ratios = {}, {}
    for line in text.splitlines():
        match = re.fullmatch(r'([^\s{]+)(?:\{.*\})?\s+([^\s]+)(?:\s+\d+)?', line)
        if not match or match[1] not in names:
            continue
        try:
            value = float(match[2])
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            if match[1].endswith('_perc'):
                ratios.setdefault(match[1], []).append(value)
            else:
                values[match[1]] = values.get(match[1], 0) + value
    # Without each pool's capacity, multiple utilization ratios cannot be
    # combined correctly. First-version single-pool readings remain exact.
    values.update({name: samples[0] for name, samples in ratios.items() if len(samples) == 1})
    return values


def _number(value, factor=1):
    try:
        result = float(value) * factor
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def _nvidia(query):
    try:
        result = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=2)
        return list(csv.reader(io.StringIO(result.stdout))) if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        return []


def _owned_pids(pid):
    if not pid:
        return set()
    parents = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                parents[int(entry.name)] = int(fields[1])
            except (OSError, ValueError, IndexError):
                pass
    owned = {pid}
    while True:
        children = {child for child, parent in parents.items() if parent in owned}
        if children <= owned:
            return owned
        owned |= children


class Telemetry:
    def __init__(self, manager, chat):
        self.manager, self.chat = manager, chat
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._cpu = None
        self._hardware = {}
        self._devices = None
        self._metrics = {}
        self._metrics_at = None
        self._instance = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name='webui-telemetry')
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=8)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.sample()
            except (OSError, ValueError, RuntimeError):
                # A failed sample expires naturally; it must not stop sampling.
                pass
            self._stop.wait(1)

    def sample(self):
        hardware = {'sampled_at_ms': time.time() * 1000, 'gpus': [],
                    'cpu_percent': None, 'ram_total_bytes': None,
                    'ram_used_bytes': None, 'ram_available_bytes': None}
        try:
            memory = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                key, value = line.split(':', 1)
                memory[key] = int(value.strip().split()[0]) * 1024
            total, available = memory['MemTotal'], memory['MemAvailable']
            hardware.update(ram_total_bytes=total, ram_available_bytes=available,
                            ram_used_bytes=total - available)
            counters = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
            current = (sum(counters), counters[3] + counters[4])
            if self._cpu and current[0] > self._cpu[0]:
                hardware['cpu_percent'] = max(0, min(100, 100 *
                    (1 - (current[1] - self._cpu[1]) / (current[0] - self._cpu[0]))))
            self._cpu = current
        except (OSError, ValueError, KeyError, IndexError):
            pass
        gpu_query = '--query-gpu=uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw'
        for row in _nvidia(gpu_query):
            if len(row) != 8:
                continue
            uuid, pci, name, total, used, load, temperature, power = [x.strip() for x in row]
            hardware['gpus'].append(dict(uuid=uuid, pci_bus_id=pci, name=name,
                total_bytes=_number(total, 1024 ** 2), used_bytes=_number(used, 1024 ** 2),
                utilization_percent=_number(load), temperature_c=_number(temperature), power_w=_number(power)))
        status = self.manager.status()
        devices = None
        if status['state'] in ('loading', 'ready'):
            owned = _owned_pids(status.get('pid'))
            bound = set()
            for row in _nvidia('--query-compute-apps=pid,gpu_uuid'):
                if len(row) == 2 and row[0].strip().isdigit() and int(row[0]) in owned:
                    bound.add(row[1].strip())
            if bound:
                devices = [{k: g[k] for k in ('uuid', 'pci_bus_id', 'name')}
                           for g in hardware['gpus'] if g['uuid'] in bound]
        metrics, sampled = {}, None
        base_url = status.get('base_url')
        if status['state'] == 'ready' and base_url:
            try:
                request = urllib.request.Request(base_url + '/metrics',
                    headers=getattr(self.manager, 'backend_headers', {}))
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=2) as response:
                    data = response.read(4 * 1024 * 1024 + 1)
                if len(data) <= 4 * 1024 * 1024:
                    metrics, sampled = parse_metrics(data.decode('utf-8')), time.time() * 1000
            except (OSError, UnicodeError):
                pass
        # Discard measurements crossing a service restart.
        latest = self.manager.status()
        if latest.get('instance') != status.get('instance') or latest['state'] != status['state']:
            return
        with self._lock:
            self._hardware, self._devices = hardware, devices
            self._metrics, self._metrics_at = metrics, sampled
            self._instance = status.get('instance')

    def hardware(self):
        with self._lock:
            return copy.deepcopy(self._hardware)

    def snapshot(self):
        status = self.manager.status()
        records = self.chat.snapshot()
        with self._lock:
            hardware = copy.deepcopy(self._hardware)
            fresh = (status['state'] == 'ready' and self._instance == status.get('instance')
                     and self._metrics_at and time.time() * 1000 - self._metrics_at < 5000)
            metrics = dict(self._metrics) if fresh else {}
            sampled = self._metrics_at if fresh else None
        now = time.time() * 1000
        rows = {'active': [], 'recent': []}
        for group in rows:
            for record in records.get(group, []):
                if record.get('instance') not in (None, status.get('instance')):
                    continue
                row = dict(record.get('metrics') or {})
                for key in ('id', 'started_at_ms', 'finish_reason', 'prompt_tokens',
                            'generated_tokens', 'elapsed_ms', 'limit'):
                    if key in record:
                        row[key] = record[key]
                state = record.get('state', record.get('status', 'processing'))
                row['status'] = {'thinking': 'generating', 'answering': 'generating',
                                 'cancelled': 'disconnected', 'prefill': 'reading'}.get(state, state)
                row['phase'] = state
                if 'elapsed_ms' not in row and row.get('started_at_ms'):
                    row['elapsed_ms'] = record.get('finished_at_ms', now) - row['started_at_ms']
                rows[group].append(row)
            rows[group].sort(key=lambda row: row.get('started_at_ms', 0), reverse=True)
        return dict(schema_version=1, backend='vllm', server_instance=status.get('instance'),
                    sampled_at_ms=now, state=status['state'], hardware=hardware,
                    active=rows['active'], recent=rows['recent'], request_scope='console',
                    queued_count=metrics.get('vllm:num_requests_waiting'),
                    native_metrics={'scope': 'engine', 'sampled_at_ms': sampled, 'values': metrics},
                    kvmem={})

    def props(self):
        status = self.manager.status()
        profile = status.get('active_profile') or {}
        memory = profile.get('kvmem') or {}
        with self._lock:
            fresh = (status['state'] in ('loading', 'ready') and self._instance == status.get('instance')
                     and time.time() * 1000 - self._hardware.get('sampled_at_ms', 0) < 5000)
            devices = copy.deepcopy(self._devices) if fresh else None
        execution = {'mode': 'single'}
        if devices:
            execution['monitor_devices'] = devices
        return {'model_name': profile.get('model', ''),
                'model_alias': profile.get('served_model_name', 'kvmem'),
                'default_generation_settings': {'n_ctx': profile.get('max_model_len')},
                'execution': execution, 'backend': 'vllm',
                'kvmem': {'select_budget': memory.get('select_budget'),
                          'gen_budget': memory.get('gen_budget')},
                'profile': profile}

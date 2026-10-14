"""vLLM's supported connector metadata channel for working-set transactions.

No prefix hits are invented. The worker runtime owns capture/storage/assembly;
the connector carries commands and completion acknowledgments to the scheduler.
"""
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import distribution
import os

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorWorkerMetadata, SupportsHMA,
    KVConnectorRole,
)

from .window_manager import RefreshTicket
from .terminal import TerminalPrefix
from .archive_admission import ArchiveCapacity
from .session_control import SessionCommands

RETIREMENT_CONNECTOR_SHA256 = 'b89a4333f1e552d86e1815674b59e9f1335f0f49c0a7b46082439076f62c6a81'


@dataclass(frozen=True)
class RefreshCommand:
    request_id: str
    command_id: int
    groups: tuple[tuple[int, RefreshTicket], ...]
    query_span: tuple[int, int] | None = None
    selection: str = 'chunk'
    mandatory_spans: tuple = ()


@dataclass
class ControlMetadata(KVConnectorMetadata):
    commands: list[RefreshCommand] = field(default_factory=list)
    cancelled: set[str] = field(default_factory=set)
    terminal_prefixes: dict[str, TerminalPrefix] = field(default_factory=dict)
    sessions: SessionCommands = field(default_factory=SessionCommands)


@dataclass
class ControlCompletions(KVConnectorWorkerMetadata):
    completed: list[tuple[str, int]] = field(default_factory=list)
    selected_windows: dict[str, int] = field(default_factory=dict)
    archive_sequence: int | None = None
    archive_capacity: dict[str, ArchiveCapacity] = field(default_factory=dict)
    sessions: list = field(default_factory=list)

    def aggregate(self, other):
        raise RuntimeError("Working-set control currently requires a single worker rank")


class KVMemConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("Working-set control currently qualifies TP=1")
        self.commands = []
        self.pending = {}
        self.cancelled = set()
        self.cancel_updates = set()
        self.terminal_prefixes = {}
        self.bounded_target = os.getenv('KVMEM_BOUNDED_TARGET') == '1'
        self.worker_jobs = {}
        self.worker_completions = []
        self.worker_selected_windows = {}
        self.runtime = None
        self.scheduler_completion = None
        options = vllm_config.kv_transfer_config.kv_connector_extra_config.get('kvmem', {})
        self.async_retirement = options.get('async_archive_retirement', False)
        repository_gc = options.get('repository_gc', False)
        if type(self.async_retirement) is not bool or type(repository_gc) is not bool:
            raise ValueError('Retirement and repository GC options must be boolean')
        if (self.async_retirement or repository_gc) and os.getenv('KVMEM_BOUNDED_TARGET') != '1':
            raise ValueError('Retirement/GC requires bounded authority admission')
        if self.async_retirement or repository_gc:
            source = distribution('vllm').locate_file('vllm/v1/worker/gpu/kv_connector.py')
            if hashlib.sha256(source.read_bytes()).hexdigest() != RETIREMENT_CONNECTOR_SHA256:
                raise ValueError('Unqualified native idle-worker retirement completion hook')
        self.retirement_done = None
        self.archive_snapshot = None
        self.scheduler_archive_update = None
        self.archive_sequence = 0
        self.sessions = None  # Scheduler ledger; worker lifecycle is separate.
        self.session_worker = None
        if (self.bounded_target and role == KVConnectorRole.SCHEDULER
                and kv_cache_config is not None and os.getenv('KVMEM_RUNNER_PROBE_DIR')):
            from .cache_manifest import write_mean_k_geometry
            write_mean_k_geometry(vllm_config, kv_cache_config, os.environ['KVMEM_RUNNER_PROBE_DIR'])

    def queue(self, command):
        if command.request_id in self.pending:
            raise RuntimeError("Duplicate active working-set transaction")
        self.pending[command.request_id] = command.command_id
        self.commands.append(command)

    def build_connector_meta(self, scheduler_output):
        result = ControlMetadata(self.commands, self.cancel_updates, self.terminal_prefixes)
        self.commands, self.cancel_updates = [], set()
        self.terminal_prefixes = {}
        if self.sessions is not None:
            result.sessions = self.sessions.take_commands()
        return result

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, ControlMetadata):
            raise TypeError("Unexpected KVMem control metadata")
        self.cancelled.update(metadata.cancelled)
        if metadata.sessions.evictions:
            if self.session_worker is None:
                raise RuntimeError('Session eviction received without its worker')
            self.session_worker.evict(metadata.sessions.evictions)
        for command in metadata.commands:
            if self.runtime is None:
                raise RuntimeError("Working-set command received without a worker runtime")
            if command.request_id in self.worker_jobs:
                raise RuntimeError("Duplicate worker transaction")
            self.worker_jobs[command.request_id] = (command, self.runtime(command))

    def get_finished(self, finished_req_ids):
        for request_id, (command, job) in list(self.worker_jobs.items()):
            if job.query():
                self.worker_completions.append((request_id, command.command_id))
                if getattr(job, "selected_tokens", None) is not None:
                    self.worker_selected_windows[request_id] = job.selected_tokens
                del self.worker_jobs[request_id]
        done = set()
        for request_id in self.cancelled:
            if request_id in self.worker_jobs:
                continue
            if self.async_retirement:
                if self.retirement_done is None:
                    raise RuntimeError('Missing authority retirement completion handler')
                if not self.retirement_done(request_id):
                    continue
            done.add(request_id)
        self.cancelled.difference_update(done)
        return done or None, None

    def build_connector_worker_meta(self):
        capacity = self.archive_snapshot() if self.archive_snapshot is not None else None
        notices = self.session_worker.notices() if self.session_worker is not None else []
        if not self.worker_completions and capacity is None and not notices:
            return None
        result = ControlCompletions(self.worker_completions, self.worker_selected_windows)
        result.sessions = notices
        if capacity is not None:
            self.archive_sequence += 1
            result.archive_sequence = self.archive_sequence
            result.archive_capacity = capacity
        self.worker_completions = []
        self.worker_selected_windows = {}
        return result

    def update_connector_output(self, connector_output):
        metadata = connector_output.kv_connector_worker_meta
        if metadata is not None:
            if not isinstance(metadata, ControlCompletions):
                raise TypeError("Unexpected worker completion metadata")
            for notice in metadata.sessions:
                if self.sessions is None:
                    raise RuntimeError('Unexpected session worker completion')
                self.sessions.consume(notice)
            if metadata.archive_sequence is not None:
                if self.scheduler_archive_update is None:
                    raise RuntimeError('Unexpected archive capacity feedback')
                self.scheduler_archive_update(metadata.archive_sequence, metadata.archive_capacity)
            for request_id, command_id in metadata.completed:
                if self.pending.get(request_id) != command_id:
                    raise ValueError("Stale or foreign worker completion")
                if self.scheduler_completion is None:
                    raise RuntimeError("Missing scheduler completion handler")
                if request_id in metadata.selected_windows:
                    self.scheduler_completion(request_id, command_id,
                                              selected_tokens=metadata.selected_windows[request_id])
                else:
                    self.scheduler_completion(request_id, command_id)
                del self.pending[request_id]
        self.cancelled.difference_update(connector_output.finished_sending or ())

    def request_finished_all_groups(self, request, block_ids):
        if self.sessions is not None:
            self.sessions.finished(request)
        if self.bounded_target:
            if request.request_id in self.terminal_prefixes:
                raise RuntimeError('Duplicate terminal request boundary')
            self.terminal_prefixes[request.request_id] = TerminalPrefix.from_request(request)
        if request.request_id not in self.pending and not self.async_retirement:
            return False, None
        self.cancelled.add(request.request_id)
        self.cancel_updates.add(request.request_id)
        return True, None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if self.sessions is not None:
            return self.sessions.lookup(request, num_computed_tokens), False
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if self.sessions is not None:
            self.sessions.allocated(request, num_external_tokens)
            return
        if num_external_tokens:
            raise ValueError("KVMem reselection is not an external contiguous prefix hit")

    def wait_for_layer_load(self, layer_name):
        pass  # Pending requests are suspended; only ready requests run forward.

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass  # The raw capture adapter saves pre-RoPE authorities, not rotated KV.

    def wait_for_save(self):
        pass

    def has_pending_push_work(self):
        return bool(self.pending or self.cancelled or
                    (self.sessions is not None and self.sessions.has_pending_work()))

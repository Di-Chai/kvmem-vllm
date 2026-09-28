"""vLLM's supported connector metadata channel for working-set transactions.

No prefix hits are invented. The worker runtime owns capture/storage/assembly;
the connector carries commands and completion acknowledgments to the scheduler.
"""
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorWorkerMetadata, SupportsHMA,
)

from .window_manager import RefreshTicket


@dataclass(frozen=True)
class RefreshCommand:
    request_id: str
    command_id: int
    groups: tuple[tuple[int, RefreshTicket], ...]


@dataclass
class ControlMetadata(KVConnectorMetadata):
    commands: list[RefreshCommand] = field(default_factory=list)
    cancelled: set[str] = field(default_factory=set)


@dataclass
class ControlCompletions(KVConnectorWorkerMetadata):
    completed: list[tuple[str, int]] = field(default_factory=list)
    selected_windows: dict[str, int] = field(default_factory=dict)

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
        self.worker_jobs = {}
        self.worker_completions = []
        self.worker_selected_windows = {}
        self.runtime = None
        self.scheduler_completion = None

    def queue(self, command):
        if command.request_id in self.pending:
            raise RuntimeError("Duplicate active working-set transaction")
        self.pending[command.request_id] = command.command_id
        self.commands.append(command)

    def build_connector_meta(self, scheduler_output):
        result = ControlMetadata(self.commands, self.cancel_updates)
        self.commands, self.cancel_updates = [], set()
        return result

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, ControlMetadata):
            raise TypeError("Unexpected KVMem control metadata")
        self.cancelled.update(metadata.cancelled)
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
        done = {request_id for request_id in self.cancelled if request_id not in self.worker_jobs}
        self.cancelled.difference_update(done)
        return done or None, None

    def build_connector_worker_meta(self):
        if not self.worker_completions:
            return None
        result = ControlCompletions(self.worker_completions, self.worker_selected_windows)
        self.worker_completions = []
        self.worker_selected_windows = {}
        return result

    def update_connector_output(self, connector_output):
        metadata = connector_output.kv_connector_worker_meta
        if metadata is not None:
            if not isinstance(metadata, ControlCompletions):
                raise TypeError("Unexpected worker completion metadata")
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
        if request.request_id not in self.pending:
            return False, None
        self.cancelled.add(request.request_id)
        self.cancel_updates.add(request.request_id)
        return True, None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            raise ValueError("KVMem reselection is not an external contiguous prefix hit")

    def wait_for_layer_load(self, layer_name):
        pass  # Pending requests are suspended; only ready requests run forward.

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass  # The raw capture adapter saves pre-RoPE authorities, not rotated KV.

    def wait_for_save(self):
        pass

    def has_pending_push_work(self):
        return bool(self.pending or self.cancelled)

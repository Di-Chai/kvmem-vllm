"""Validated plugin settings carried by vLLM's public connector extra config."""
from dataclasses import dataclass, field

from .config import KVMemConfig


@dataclass(frozen=True)
class RuntimeOptions:
    memory: KVMemConfig = field(default_factory=KVMemConfig)
    cpu_read_cache_bytes: int = 0
    packed_assembly: bool = False
    verify_archive_on_finish: bool = True
    direct_pinned_reads: bool = False

    def __post_init__(self):
        if type(self.cpu_read_cache_bytes) is not int or self.cpu_read_cache_bytes < 0:
            raise ValueError("CPU read-cache capacity must be a nonnegative byte count")
        if any(type(value) is not bool for value in
               (self.packed_assembly, self.verify_archive_on_finish, self.direct_pinned_reads)):
            raise ValueError("Assembly and archive verification settings must be boolean")
        if self.direct_pinned_reads and not self.packed_assembly:
            raise ValueError("Direct pinned reads require packed assembly")

    @classmethod
    def from_vllm(cls, config):
        transfer = config.kv_transfer_config
        extra = transfer.kv_connector_extra_config if transfer is not None else {}
        if not isinstance(extra, dict):
            raise ValueError("Connector extra config must be an object")
        payload = extra.get('kvmem', {})
        if not isinstance(payload, dict):
            raise ValueError("kvmem connector settings must be an object")
        payload = dict(payload)
        cache_bytes = payload.pop('cpu_read_cache_bytes', 0)
        packed = payload.pop('packed_assembly', False)
        verify = payload.pop('verify_archive_on_finish', True)
        direct = payload.pop('direct_pinned_reads', False)
        try:
            memory = KVMemConfig(**payload)
        except TypeError as error:
            raise ValueError(f"Unknown KVMem setting: {error}") from error
        if memory.kv_dtype != config.cache_config.cache_dtype:
            raise ValueError("KVMem source codec differs from the engine KV cache dtype")
        return cls(memory, cache_bytes, packed, verify, direct)

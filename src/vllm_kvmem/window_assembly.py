"""Bounded immutable-source assembly into native vLLM FP8 working pages.

The scheduler must hold the request's physical pages through the returned CUDA
fence. Only after it completes may it trim the old allocation or resume execution.
This first bridge batches native archive reads and H2D copies, with synchronous
CPU I/O between chunks; overlapping scheduler-visible preparation is pending.
"""
from dataclasses import dataclass, field

import time

import torch

from .assembly import assemble_fp8
from .codec import native_vllm_fp8_views
from .memory import Phase
from .transfer import PackedTransfers


@dataclass
class LayerWindow:
    key: torch.Tensor
    value: torch.Tensor
    cos_sin: torch.Tensor
    key_scale: torch.Tensor
    kernel_page_table: torch.Tensor


@dataclass
class AssemblyResult:
    event: torch.cuda.Event
    owners: tuple
    rebuilt_tokens: int
    skipped_tokens: int
    chunks: int
    host_prepare_seconds: float = 0.0
    uploaded_bytes: int = 0
    read_stats: dict = field(default_factory=dict)

    def query(self):
        return self.event.query()


class NativeWindowAssembler:
    def __init__(self, layers: tuple[LayerWindow, ...], block_tokens: int, chunk_blocks=32, packed=False):
        if not layers or min(block_tokens, chunk_blocks) <= 0:
            raise ValueError("Invalid native assembly geometry")
        device = layers[0].key.device
        heads, dim = layers[0].key.shape[-2:]
        checked_tables = set()
        for layer in layers:
            if (layer.key.device != device or layer.key.shape != layer.value.shape or
                    layer.key.shape[-2:] != (heads, dim) or layer.kernel_page_table.ndim != 1 or
                    layer.kernel_page_table.device != device):
                raise ValueError("Native layers disagree on cache geometry or ownership")
            if layer.key.dtype != torch.float8_e4m3fn or layer.value.dtype != torch.float8_e4m3fn:
                raise ValueError("Native assembly requires FP8 e4m3 cache views")
            signature = (layer.kernel_page_table.data_ptr(), layer.key.shape[0])
            if signature not in checked_tables:
                if (not layer.kernel_page_table.numel() or
                        torch.any((layer.kernel_page_table <= 0) | (layer.kernel_page_table >= layer.key.shape[0])).item()):
                    raise ValueError("Native page table contains null or out-of-arena pages")
                checked_tables.add(signature)
        self.layers = layers
        self.block_tokens = block_tokens
        self.chunk_blocks = chunk_blocks
        self.packed = packed
        self.device = device
        self.heads, self.dim = heads, dim
        self.transfers = PackedTransfers(len(layers) * heads * dim * block_tokens * chunk_blocks * 3,
                                         2, device)

    @classmethod
    def from_runner(cls, runner, request_id, layer_names, block_tokens=128, chunk_blocks=32, packed=False):
        modules = dict(runner.model.named_modules())
        groups = {name: group_id for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups)
                  for name in group.layer_names}
        index = runner.req_states.req_id_to_index[request_id]
        tables = {}
        layers = []
        for name in layer_names:
            module = modules[name]
            attn = module.attn
            if attn.kv_cache_dtype != "fp8_e4m3":
                raise ValueError("Native assembly currently qualifies FP8 e4m3")
            group_id = groups[attn.layer_name]
            if group_id not in tables:
                count = int(runner.block_tables.num_blocks.np[group_id, index])
                tables[group_id] = runner.block_tables.block_tables[group_id].gpu[index, :count].clone()
            key, value = native_vllm_fp8_views(attn.kv_cache, module.num_kv_heads, module.head_dim)
            layers.append(LayerWindow(key, value, module.rotary_emb.cos_sin_cache,
                                      attn._k_scale, tables[group_id]))
        return cls(tuple(layers), block_tokens, chunk_blocks, packed=packed)

    def assemble(self, authority, plan):
        try:
            return self._assemble(authority, plan)
        except BaseException:
            if authority.memory.phase is Phase.ASSEMBLING and authority.memory._plan is plan:
                # Even an I/O error after an earlier chunk must not permit the
                # half-rebuilt cache to commit, or staging/leases to be recycled
                # while submitted copies and kernels still use them.
                stream = torch.cuda.current_stream(self.device)
                for transfer in self.transfers.pending.values():
                    stream.wait_event(transfer.event)
                failed = torch.cuda.Event()
                failed.record(stream)
                authority.memory.fail_transaction(AssemblyResult(
                    failed, (self, plan, authority), 0, 0, 0))
            raise

    def _assemble(self, authority, plan):
        if authority.memory.phase is not Phase.ASSEMBLING or authority.memory._plan is not plan:
            raise ValueError("Native assembly requires an active original-core selection")
        if authority.shape != (len(self.layers), self.block_tokens, self.heads, self.dim):
            raise ValueError("Authority codec does not match native working pages")
        by_slot = {record.handle.slot: record for record in authority.records.values()}
        remaps = list(plan.remaps)
        # Validate the entire destination before changing any GPU cache bytes.
        for remap in remaps:
            record = by_slot.get(remap.block_id)
            if record is None or record.count != remap.n_tokens or remap.to_base < 0:
                raise ValueError("Selected generation/extent no longer matches authority")
            for layer in self.layers:
                if remap.to_base + remap.n_tokens > layer.kernel_page_table.numel() * layer.key.shape[1]:
                    raise ValueError("Selected working window exceeds native allocated pages")
                if remap.to_base + remap.n_tokens > layer.cos_sin.shape[0]:
                    raise ValueError("Selected position exceeds the model RoPE table")
        pending = [remap for remap in remaps if not remap.skip]
        rebuilt, chunks, owners = 0, 0, [self, plan, authority]
        host_prepare_seconds, uploaded_bytes = 0.0, 0
        read_stats = {}
        for start in range(0, len(pending), self.chunk_blocks):
            chunk = pending[start:start+self.chunk_blocks]
            handles = [by_slot[remap.block_id].handle for remap in chunk]
            prepare_start = time.perf_counter()
            if self.packed:
                from .packed_assembly import read_packed_records
                if any(remap.n_tokens != self.block_tokens for remap in chunk):
                    raise ValueError("Packed window assembly requires complete selected records")
                payload, stats = read_packed_records(authority, handles)
                for key, value in stats.items():
                    read_stats[key] = read_stats.get(key, 0) + value
                count = len(chunk) * self.block_tokens
            else:
                payloads = authority.read_records(handles)
                raw = torch.cat([raw for raw, _ in payloads], dim=1).contiguous()
                values = torch.cat([value for _, value in payloads], dim=1).contiguous()
                payload = torch.cat((raw.view(torch.uint8).flatten(), values.flatten()))
                count = raw.shape[1]
            host_prepare_seconds += time.perf_counter() - prepare_start
            uploaded_bytes += payload.numel()
            gpu = torch.empty(payload.numel(), dtype=torch.uint8, device=self.device)
            transfer = self.transfers.upload(payload, gpu)
            if transfer is None:
                # Only bounded staging is reused, after its own DMA completes.
                next(iter(self.transfers.pending.values())).event.synchronize()
                transfer = self.transfers.upload(payload, gpu)
            if transfer is None:
                raise RuntimeError("Native assembly staging did not make progress")
            torch.cuda.current_stream(self.device).wait_event(transfer.event)
            if self.packed:
                from .packed_assembly import assemble_packed_layer
                bases = torch.tensor([remap.to_base for remap in chunk], device=self.device, dtype=torch.int64)
                for i, layer in enumerate(self.layers):
                    assemble_packed_layer(gpu, bases, layer, layer_index=i, layers=len(self.layers),
                                          block_tokens=self.block_tokens, raw_dtype=authority.raw_dtype)
            else:
                elements = raw.numel()
                raw_gpu = gpu[:elements*2].view(authority.raw_dtype).reshape(raw.shape)
                values_gpu = gpu[elements*2:].view(torch.float8_e4m3fn).reshape(values.shape)
                positions = torch.cat([torch.arange(remap.to_base, remap.to_base+remap.n_tokens,
                                                    device=self.device, dtype=torch.int64) for remap in chunk])
                for i, layer in enumerate(self.layers):
                    page_tokens = layer.key.shape[1]
                    slots = layer.kernel_page_table[positions // page_tokens].to(torch.int64) * page_tokens + positions % page_tokens
                    assemble_fp8(raw_gpu[i], values_gpu[i], positions, slots, layer.cos_sin,
                                 layer.key, layer.value, layer.key_scale)
            rebuilt += count
            chunks += 1
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.device))
        return AssemblyResult(done, tuple(owners), rebuilt,
                              sum(remap.n_tokens for remap in remaps if remap.skip), chunks,
                              host_prepare_seconds, uploaded_bytes, read_stats)

    def close(self):
        return self.transfers.close()

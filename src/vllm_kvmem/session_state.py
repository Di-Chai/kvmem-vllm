"""Portable GDN and native DFlash context for sealed prompt checkpoints.

Only quiescent, non-final prefill boundaries are supported. Target attention KV
is reconstructed separately from WindowCheckpoint's immutable authority. Capture
offloads synchronously to bounded CPU storage; no old native page stays owned by
the returned snapshot. The session owner must verify exact prompt identity and
hold destination pages until the restore fence completes before admitting work.
"""
from dataclasses import dataclass

import torch

from .native_state import NativeRecurrentState


@dataclass(frozen=True)
class PromptModelState:
    owner: object
    position: int
    draft_begin: int
    recurrent: dict[tuple[str, str], torch.Tensor]
    draft: dict[str, torch.Tensor]

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (*self.recurrent.values(), *self.draft.values()))


class PromptStateAdapter:
    def __init__(self, runner):
        from .draft_contract import DraftContract
        from .options import RuntimeOptions
        self.runner = runner
        self.recurrent = NativeRecurrentState(runner)
        self.contract = None
        if runner.speculative_config is not None:
            self.contract = DraftContract.from_vllm(runner.vllm_config,
                RuntimeOptions.from_vllm(runner.vllm_config).memory)
            self.contract.validate_native_groups(runner)
        self._identity = object()  # Snapshots never cross workers/model loads.
        options = RuntimeOptions.from_vllm(runner.vllm_config)
        self.capture_decode = options.session_capture_decode or options.session_checkpoint_policy=='pm'

    def _prefill_index(self, request_id, position, *, terminal=False):
        index = self.runner.req_states.req_id_to_index[request_id]
        states = self.runner.req_states
        if (type(position) is not int or position <= 0 or
                (not terminal and (int(states.num_computed_tokens_np[index]) != position or
                    (int(states.prefill_len.np[index]) <= position and not getattr(self, 'capture_decode', False))))):
            raise ValueError('Model checkpoint requires the exact non-final prefill boundary')
        return index

    def _recurrent_views(self, request_id):
        # borrow validates generation, null pages and the native state layout.
        bundle = self.recurrent.borrow([request_id])
        result = {}
        for gid, layers in self.recurrent.groups.items():
            row = bundle.page_rows[request_id, gid][0]
            for name, (conv, ssm) in layers.items():
                result[name, 'conv'] = conv[row]
                result[name, 'ssm'] = ssm[row]
        return result

    def _draft_views(self, request_id, begin, end):
        if self.contract is not None and self.contract.method == 'mtp':
            if begin != end:
                raise ValueError('MTP working KV belongs to the immutable window checkpoint')
            # MTP raw-K and FP8-V are part of target window assembly. Native
            # autoregressive proposal overwrites its temporary hidden inputs
            # with the next target batch; there is no extra persistent state.
            self.contract.validate_native_groups(self.runner)
            return {}
        if self.contract is None:
            if begin != end:
                raise ValueError('Target-only snapshot cannot contain draft history')
            return {}
        groups = self.contract.validate_native_groups(self.runner)
        index = self.runner.req_states.req_id_to_index[request_id]
        modules = {getattr(m, 'layer_name', None): m
                   for m in self.runner.speculator.model.modules()
                   if getattr(m, 'layer_name', None) is not None}
        result = {}
        for gid in groups:
            block_size = self.runner.block_tables.kernel_block_sizes[gid]
            count = int(self.runner.block_tables.num_blocks.np[gid, index])
            table = self.runner.block_tables.block_tables[gid].gpu[index, :count]
            first, stop = begin // block_size, (end + block_size - 1) // block_size
            if not 0 <= first < stop <= count:
                raise ValueError('Draft checkpoint context is outside its native page table')
            pages = table[first:stop].cpu().tolist()
            if any(p <= 0 for p in pages) or len(set(pages)) != len(pages):
                raise ValueError('Draft checkpoint has null or aliased context pages')
            logical = torch.arange(begin, end, device=table.device, dtype=torch.int64)
            physical, offset = table[logical // block_size].long(), logical % block_size
            for name in self.runner.kv_cache_config.kv_cache_groups[gid].layer_names:
                layer = modules.get(name)
                cache = getattr(layer, 'kv_cache', None)
                if (cache is None or cache.dtype != torch.bfloat16 or cache.ndim != 4 or
                        cache.shape[1] != layer.num_kv_heads or
                        cache.shape[2] != block_size or cache.shape[3] != 2 * layer.head_size or
                        cache.stride(-1) != 1 or cache.device != table.device or
                        max(pages) >= cache.shape[0]):
                    raise ValueError('Unqualified native draft BHN(2D) cache view')
                result[name] = (cache.transpose(1, 2), physical, offset)
        return result

    def capture(self, request_id, position, *, max_bytes, terminal=False):
        index = self._prefill_index(request_id, position, terminal=terminal)
        counter = self.runner.model_state.num_accepted_tokens_gpu
        if terminal or int(counter[index].item()) != 1:
            if not terminal and not getattr(self, 'capture_decode', False):
                raise ValueError('Speculative terminal state requires canonicalization or replay')
            runtime = self.runner._kvmem_capture_runtime
            steps = runtime.last_steps.get(request_id, [])
            if not steps or steps[-1]['history_begin'] + steps[-1]['accepted'] != position:
                raise ValueError('Decode snapshot is missing its exact accepted sampling extent')
            step = steps[-1]
            if terminal and runtime.authorities[request_id].history_end != position:
                raise ValueError('Terminal state differs from the accepted KV authority')
            event = self.recurrent.canonicalize_accepted(request_id,
                retained_tokens=step['accepted'], query_tokens=step['scheduled'],
                speculative=step['history_begin'] >= int(self.runner.req_states.prefill_len.np[index]))
            if event is not None:
                event.synchronize()
        recurrent = self._recurrent_views(request_id)
        begin = (max(0, position - self.contract.sliding_window + 1)
                 if self.contract and self.contract.method == 'dflash' else position)
        draft = self._draft_views(request_id, begin, position)
        size = sum(t.numel() * t.element_size() for t in recurrent.values())
        size += sum((position - begin) * cache.shape[2] * cache.shape[3] * cache.element_size()
                    for cache, _, _ in draft.values())
        if type(max_bytes) is not int or max_bytes <= 0 or size > max_bytes:
            raise BufferError('Prompt model checkpoint exceeds its byte reservation')
        # Blocking copies establish a stable CPU-owned checkpoint before the
        # next target forward can overwrite the borrowed GDN/draft source views.
        tensors = list(recurrent.values()) + [cache[pages, offsets] for cache, pages, offsets in draft.values()]
        # One transfer/synchronization for all GDN layers and draft pages. Each
        # typed view owns a disjoint range of this CPU checkpoint buffer.
        packed = torch.cat([t.contiguous().view(torch.uint8).flatten() for t in tensors])
        if packed.device.type == 'cuda':
            # A persistent checkpoint owns this bounded pinned buffer. Pageable
            # snapshots otherwise add a host staging copy at every sealed page.
            payload = torch.empty(packed.numel(),dtype=torch.uint8,device='cpu',pin_memory=True)
            payload.copy_(packed,non_blocking=True)
            completed = torch.cuda.Event()
            completed.record(torch.cuda.current_stream(packed.device))
            completed.synchronize()
        else:
            payload = packed.to('cpu',copy=True)
        copies, offset = [], 0
        for tensor in tensors:
            nbytes = tensor.numel() * tensor.element_size()
            copies.append(payload[offset:offset+nbytes].view(tensor.dtype).reshape(tensor.shape))
            offset += nbytes
        saved_recurrent = dict(zip(recurrent, copies[:len(recurrent)]))
        saved_draft = dict(zip(draft, copies[len(recurrent):]))
        snapshot = PromptModelState(self._identity, position, begin, saved_recurrent, saved_draft)
        if snapshot.bytes != size:
            raise RuntimeError('Checkpoint byte reservation disagrees with materialized state')
        return snapshot

    def restore(self, snapshot, request_id):
        if not isinstance(snapshot, PromptModelState) or snapshot.owner is not self._identity:
            raise ValueError('Prompt state belongs to another worker/model load')
        index = self._prefill_index(request_id, snapshot.position)
        recurrent = self._recurrent_views(request_id)
        expected_begin = (max(0, snapshot.position - self.contract.sliding_window + 1)
                          if self.contract and self.contract.method == 'dflash' else snapshot.position)
        if snapshot.draft_begin != expected_begin:
            raise ValueError('Checkpoint draft context differs from the model window')
        draft = self._draft_views(request_id, snapshot.draft_begin, snapshot.position)
        if recurrent.keys() != snapshot.recurrent.keys() or draft.keys() != snapshot.draft.keys():
            raise ValueError('Incomplete prompt model checkpoint')
        # Validate every tensor before the first native state/page write.
        for name, destination in recurrent.items():
            source = snapshot.recurrent[name]
            if source.device.type != 'cpu' or source.shape != destination.shape or source.dtype != destination.dtype:
                raise ValueError('Recurrent checkpoint schema changed')
        for name, (cache, _, _) in draft.items():
            source = snapshot.draft[name]
            shape = (snapshot.position - snapshot.draft_begin, cache.shape[2], cache.shape[3])
            if source.device.type != 'cpu' or tuple(source.shape) != shape or source.dtype != cache.dtype:
                raise ValueError('Draft checkpoint schema changed')
        sources = [snapshot.recurrent[name] for name in recurrent] + [snapshot.draft[name] for name in draft]
        payload = torch.cat([t.contiguous().view(torch.uint8).flatten() for t in sources]).to(
            self.runner.model_state.num_accepted_tokens_gpu.device)
        uploaded, offset = [], 0
        for source in sources:
            nbytes = source.numel() * source.element_size()
            uploaded.append(payload[offset:offset+nbytes].view(source.dtype).reshape(source.shape))
            offset += nbytes
        for (name, destination), source in zip(recurrent.items(), uploaded[:len(recurrent)]):
            destination.copy_(source)
        for (name, (cache, pages, offsets)), source in zip(draft.items(), uploaded[len(recurrent):]):
            cache[pages, offsets] = source
        counter = self.runner.model_state.num_accepted_tokens_gpu
        counter[index].fill_(1)
        if counter.device.type == 'cuda':
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(counter.device))
            return event
        return None

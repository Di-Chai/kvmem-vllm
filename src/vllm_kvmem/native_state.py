"""Borrow exact native Qwen GDN state pages for fenced query replay.

No reconstructed or approximated recurrent state: both convolution and SSM
tensors, every allocated speculation row, and the native acceptance counter are
included. Prefix-cached/aligned and context-parallel layouts need separate adapters.
"""
from dataclasses import dataclass

import torch


@dataclass
class StateBundle:
    tensors: dict[str, torch.Tensor]
    request_indices: dict[str, int]
    request_generations: dict[str, int]
    page_rows: dict[tuple[str, int], tuple[int, ...]]

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors.values())


class NativeRecurrentState:
    def __init__(self, runner):
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
        from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
        from .integration import check_engine_version
        check_engine_version()
        config = runner.vllm_config
        if config.cache_config.mamba_cache_mode != "none" or config.cache_config.enable_prefix_caching:
            raise ValueError("Native GDN replay currently requires non-prefix-cached 'none' state layout")
        parallel = config.parallel_config
        if any(getattr(parallel, name, 1) != 1 for name in (
                "tensor_parallel_size", "pipeline_parallel_size", "decode_context_parallel_size",
                "prefill_context_parallel_size")):
            raise ValueError("Native GDN replay currently requires a single worker rank")
        if getattr(runner.model_state, "recoverssm", None) is not None:
            raise ValueError("RecoverSSM state is not part of the qualified replay schema")
        self.runner = runner
        self.groups = {}
        modules = config.compilation_config.static_forward_context
        for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            spec = spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            if not isinstance(spec, MambaSpec):
                continue
            layers = {}
            for name in group.layer_names:
                module = modules[name]
                if not isinstance(module, QwenGatedDeltaNetAttention):
                    raise ValueError(f"Unqualified recurrent-state layer: {name}")
                if not isinstance(module.kv_cache, (tuple, list)) or len(module.kv_cache) != 2:
                    raise ValueError("Expected native convolution and recurrent cache tensors")
                layers[name] = module.kv_cache
            if runner.block_tables.blocks_per_kv_block[group_id] != 1:
                raise ValueError("Recurrent state rows must not use attention page expansion")
            self.groups[group_id] = layers
        if not self.groups:
            raise ValueError("No native Qwen GDN state groups found")

    def _rows(self, request_id, group_id):
        index = self.runner.req_states.req_id_to_index[request_id]
        count = int(self.runner.block_tables.num_blocks.np[group_id, index])
        if count <= 0:
            raise ValueError("Request has no allocated native recurrent state")
        # Called only at a quiescent model boundary, after native staged table writes.
        rows = tuple(self.runner.block_tables.block_tables[group_id].gpu[index, :count].cpu().tolist())
        if any(row <= 0 for row in rows) or len(set(rows)) != count:
            raise ValueError("Null or duplicate recurrent state page")
        return index, rows

    def borrow(self, request_ids):
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Duplicate request in a recurrent checkpoint")
        tensors, indices, generations, pages = {}, {}, {}, {}
        used = set()
        for request_id in request_ids:
            generations[request_id] = self.runner._kvmem_request_generations[request_id]
            for group_id, layers in self.groups.items():
                index, rows = self._rows(request_id, group_id)
                indices[request_id] = index
                pages[request_id, group_id] = rows
                for row in rows:
                    if (group_id, row) in used:
                        raise ValueError("Different requests share a writable recurrent state page")
                    used.add((group_id, row))
                    for layer_name, caches in layers.items():
                        for kind, cache in zip(("conv", "ssm"), caches):
                            if row >= cache.shape[0]:
                                raise ValueError("Native recurrent row exceeds its cache arena")
                            tensors[f"{request_id}/{layer_name}/{kind}/{row}"] = cache[row]
            counters = self.runner.model_state.num_accepted_tokens_gpu
            tensors[f"{request_id}/num_accepted"] = counters[index:index+1]
        return StateBundle(tensors, indices, generations, pages)

    def validate(self, bundle):
        for (request_id, group_id), rows in bundle.page_rows.items():
            if self.runner._kvmem_request_generations.get(request_id) != bundle.request_generations[request_id]:
                raise ValueError("Request incarnation changed since checkpoint")
            if self.runner.req_states.req_id_to_index.get(request_id) != bundle.request_indices[request_id]:
                raise ValueError("Request state slot was freed or reused since checkpoint")
            _, current = self._rows(request_id, group_id)
            if current != rows:
                raise ValueError("Native recurrent page table changed since checkpoint")

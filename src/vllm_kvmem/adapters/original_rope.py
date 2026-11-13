"""Opt-in original-history RoPE; physical cache slots remain dense."""
import os
from types import MethodType

import torch
from ..core import settings as _settings


def enabled():
    return _settings.get('KVMEM_LAYERWISE_ORIGINAL_ROPE') == '1'


def record_positions(records, device):
    values = [p for r in records for p in range(r.history_begin, r.history_begin+r.count)]
    return torch.tensor(values, dtype=torch.int64, device=device)


def extend_rope_table(rope, capacity):
    """Extend default-RoPE lookup only; preserve every existing cache byte."""
    cache = rope.cos_sin_cache
    if cache.shape[0] >= capacity:
        return
    with torch.device(cache.device):
        freq = rope._compute_inv_freq(rope.base)
        positions = torch.arange(cache.shape[0], capacity, dtype=torch.float32)
        phase = torch.einsum('i,j->ij', positions, freq)
        tail = torch.cat((phase.cos(),phase.sin()),dim=-1).to(cache.dtype)
    rope.cos_sin_cache = torch.cat((cache,tail),dim=0)
    rope.max_position_embeddings = max(rope.max_position_embeddings, capacity)
    if getattr(rope,'cos_sin_cache_bf16',None) is not None:
        rope.cos_sin_cache_bf16 = rope.cos_sin_cache.to(torch.bfloat16)


class MTPPositions:
    """Rotate draft Q/K at logical positions without changing draft slot writes.

    Native MTP advances its dense input positions to address compact KV pages.
    This adapter adds each owner's history offset only at the RoPE projection.
    It does not alter native sampling counters, causal masks or acceptance.
    """
    def __init__(self, runner, modules):
        if not runner.model_config.enforce_eager:
            raise ValueError('Original-position MTP experiment requires eager execution')
        self.runner = runner
        self.prefill = self.decode = None
        for module in modules:
            extend_rope_table(module.rotary_emb, runner.model_config.max_model_len)
            original = module._project_qkv_gate
            owner = self
            def project(layer, qkv, positions, _original=original):
                offsets = owner.prefill if layer._kvmem_mtp_prefix_active else owner.decode
                if offsets is None:
                    return _original(qkv, positions)
                n = positions.shape[-1]
                if n > offsets.numel():
                    raise RuntimeError('MTP RoPE owner map is smaller than the native batch')
                return _original(qkv, positions + offsets[:n])
            module._project_qkv_gate = MethodType(project, module)

    def begin(self, batch):
        if self.prefill is not None:
            raise RuntimeError('Overlapping original-position MTP owners')
        self.prefill = torch.zeros(batch.num_tokens_after_padding, dtype=torch.int64,
                                   device=self.runner.device)
        self.decode = torch.zeros(self.runner.max_num_reqs, dtype=torch.int64,
                                  device=self.runner.device)
        offsets = getattr(self.runner._kvmem_window_adapter, 'offsets', {})
        for i, rid in enumerate(batch.req_ids):
            begin, end = map(int, batch.query_start_loc_np[i:i+2])
            offset = offsets.get(rid, 0)
            self.prefill[begin:end] = offset
            self.decode[i] = offset

    def end(self):
        self.prefill = self.decode = None

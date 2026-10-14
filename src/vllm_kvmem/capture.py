"""Instance-local Qwen gated-attention adapter; native batched projections remain.

Only the fused norm/RoPE kernel is extended to return its pre-RoPE intermediates.
Weight quantization, attention and recurrent execution stay owned by vLLM.
The sink must copy or retain captured tensors through asynchronous completion.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import Callable

import torch

from ._fused_capture import fused_qk_rmsnorm_rope_gate


@dataclass
class CapturedLayer:
    name: str
    positions: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


_sink: ContextVar[Callable[[CapturedLayer], None] | None] = ContextVar("kvmem_capture", default=None)


@contextmanager
def capture_to(sink: Callable[[CapturedLayer], None]):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def install_qwen_capture(model) -> list[str]:
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, Qwen3NextAttention)]
    if not layers:
        raise ValueError("No compatible Qwen gated attention layers found")
    # Validate all layers before changing any instance.
    for name, layer in layers:
        if not layer.use_fused_qk_norm_rope_gate:
            raise ValueError(f"Capture adapter not qualified for unfused layer {name}")
        if hasattr(layer, "_kvmem_original_projection"):
            raise ValueError(f"Capture already installed on {name}")
    for name, layer in layers:
        layer._kvmem_original_projection = layer._project_qkv_gate
        layer._kvmem_capture_name = name
        layer._project_qkv_gate = MethodType(_project_with_capture, layer)
    return [name for name, _ in layers]


def _project_with_capture(self, qkv, positions):
    buffers = getattr(self, '_kvmem_capture_buffers', None)
    # Compiled forwards and graph replays write stable tensors. ContextVar
    # lookup/callbacks are reserved for the diagnostic eager-only adapter.
    sink = _sink.get() if buffers is None else None
    capture_gate = getattr(self, '_kvmem_capture_gate', None)
    if (capture_gate is None and getattr(self, '_kvmem_mtp_prefix_only', False)
            and not getattr(self, '_kvmem_mtp_prefix_active', False)):
        return self._kvmem_original_projection(qkv, positions)
    if sink is None and buffers is None:
        return self._kvmem_original_projection(qkv, positions)
    q_gate, key, value = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
    if positions.ndim == 2 and not getattr(self.rotary_emb, "mrope_section", None):
        positions = positions[0]
    q, k, gate, raw_q, raw_k = fused_qk_rmsnorm_rope_gate(
        q_gate, key, self.q_norm.weight, self.k_norm.weight,
        self.rotary_emb.cos_sin_cache, positions, self.q_norm.variance_epsilon,
        self.num_heads, self.num_kv_heads, self.head_dim, self.rotary_emb.rotary_dim,
        norm_beta=1.0,
        capture_enabled=capture_gate,
        value_input=value if buffers is not None else None,
        value_output=(buffers.value[:qkv.shape[0]].view(qkv.shape[0], -1)
                      if buffers is not None else None),
        mrope_section=(getattr(self.rotary_emb, "mrope_section", None)
                       if positions.ndim == 2 else None),
        raw_q_output=(buffers.query[:qkv.shape[0]].view(qkv.shape[0], -1)
                      if buffers is not None else None),
        raw_k_output=(buffers.key[:qkv.shape[0]].view(qkv.shape[0], -1)
                      if buffers is not None else None),
    )
    raw_value = value.view(-1, self.num_kv_heads, self.head_dim)
    if buffers is not None:
        raw_value = buffers.value[:qkv.shape[0]]
    if sink is not None:
        sink(CapturedLayer(self._kvmem_capture_name, positions,
                          raw_q.view(-1, self.num_heads, self.head_dim),
                          raw_k.view(-1, self.num_kv_heads, self.head_dim), raw_value))
    return q, k, value, gate


def install_persistent_capture(model, names, max_tokens, positions):
    """Stable outputs written by every graph replay, before sampling's D2H copy.

    ContextVar callbacks cannot execute during CUDA graph replay. These buffers
    are owned by the runner for its lifetime; the native single-batch contract
    ensures sampling packs their contents before the next forward overwrites them.
    """
    modules = dict(model.named_modules())
    captured = []
    for name in names:
        layer = modules[name]
        weight = layer.q_norm.weight
        def allocate(heads):
            return torch.empty((max_tokens, heads, layer.head_dim),
                               device=weight.device, dtype=torch.bfloat16)
        record = CapturedLayer(name, positions, allocate(layer.num_heads),
                               allocate(layer.num_kv_heads), allocate(layer.num_kv_heads))
        layer._kvmem_capture_buffers = record
        captured.append(record)
    return captured

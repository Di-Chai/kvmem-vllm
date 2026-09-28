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
    sink = _sink.get()
    if sink is None:
        return self._kvmem_original_projection(qkv, positions)
    q_gate, key, value = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
    if positions.ndim == 2 and not getattr(self.rotary_emb, "mrope_section", None):
        positions = positions[0]
    q, k, gate, raw_q, raw_k = fused_qk_rmsnorm_rope_gate(
        q_gate, key, self.q_norm.weight, self.k_norm.weight,
        self.rotary_emb.cos_sin_cache, positions, self.q_norm.variance_epsilon,
        self.num_heads, self.num_kv_heads, self.head_dim, self.rotary_emb.rotary_dim,
        norm_beta=1.0,
        mrope_section=(getattr(self.rotary_emb, "mrope_section", None)
                       if positions.ndim == 2 else None),
    )
    sink(CapturedLayer(self._kvmem_capture_name, positions,
                      raw_q.view(-1, self.num_heads, self.head_dim),
                      raw_k.view(-1, self.num_kv_heads, self.head_dim),
                      value.view(-1, self.num_kv_heads, self.head_dim)))
    return q, k, value, gate

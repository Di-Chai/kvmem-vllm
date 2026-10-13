"""Use native quantized projections for DFlash's context KV on a pinned source.

The upstream fused BF16 path slices raw qkv weights and calls F.linear, which
does not represent transposed FP8 weights/scales. Each native projection applies
its own codec; the target and draft KV layout remain unchanged.
"""
from types import MethodType

import torch


def install_quantized_context(runner):
    config = runner.speculative_config
    if config is None or config.quantization != 'fp8':
        return
    from .draft_contract import verify_draft_sources
    verify_draft_sources()
    model = runner.speculator.model.model
    if config.method != 'dflash' or not model.layers:
        raise ValueError('FP8 context projection requires the pinned DFlash2 model')

    def project_context(self, context_states, num_ctx, num_layers, num_kv_heads, head_dim):
        if len(self.layers) != num_layers:
            raise ValueError('Draft context projection differs from its native layer geometry')
        normalized = self.hidden_norm(context_states)
        keys, values = [], []
        for layer in self.layers:
            attn = layer.self_attn
            result = attn.qkv_proj(normalized)
            projected = result[0] if isinstance(result, tuple) else result
            key = projected[:, attn.q_size:attn.q_size + attn.kv_size]
            value = projected[:, attn.q_size + attn.kv_size:]
            keys.append(key.reshape(num_ctx, num_kv_heads, head_dim))
            values.append(value.reshape(num_ctx, num_kv_heads, head_dim))
        return torch.stack(keys), torch.stack(values)

    model._project_context_kv = MethodType(project_context, model)

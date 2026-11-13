"""Ordered opaque norm/RoPE projection with runner-owned capture targets.

The large raw Q/K/V buffers are state of the Attention instance in vLLM's
forward context, just like its KV cache. They are not functionalized graph
inputs. ORDERED effects preserve writes even when outputs are unused; actual
Q/K/gate outputs additionally order the following attention computation.

Only the existing single-batch, same producer-stream execution is qualified.
CUDA graphs retain the installed tensor addresses: replacing a model, capture
buffer or gate requires a new installation AND graph capture, never reusing an
old graph with a new owner. CPU dispatch checks are not replay-time guards.
"""
from dataclasses import dataclass
import weakref

import torch

from ._fused_capture import fused_qk_rmsnorm_rope_gate
from .capture_compile import CAPTURE_OPAQUE_ABI


@dataclass
class OpaqueCaptureStats:
    source_fingerprint: str
    layers_installed: int = 0
    host_dispatch_count: int = 0

    def snapshot(self):
        return {'enabled': True, 'source_fingerprint': self.source_fingerprint,
                'abi': CAPTURE_OPAQUE_ABI, 'layers_installed': self.layers_installed,
                'host_dispatch_count': self.host_dispatch_count,
                'graph_replays_excluded': True}


def _tensor_signature(tensor):
    if tensor is None:
        return None
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device)


def _structure(layer):
    return (layer.q_size, layer.kv_size, layer.num_heads, layer.num_kv_heads,
            layer.head_dim, layer.rotary_emb.rotary_dim, layer.q_norm.variance_epsilon,
            tuple(getattr(layer.rotary_emb, 'mrope_section', None) or ()))


@dataclass(frozen=True)
class _ProjectionBinding:
    owner: object
    attention: object
    buffers: object
    capture_gate: object
    signature: tuple
    structure: tuple
    stats: OpaqueCaptureStats

    def resolve(self, attention):
        layer = self.owner()
        if (layer is None or self.attention() is not attention or layer.attn is not attention or
                getattr(layer, '_kvmem_capture_buffers', None) is not self.buffers):
            raise RuntimeError('Opaque capture projection owner was replaced')
        current = tuple(_tensor_signature(value) for value in (
            self.buffers.query, self.buffers.key, self.buffers.value,
            getattr(layer, '_kvmem_capture_gate', None)))
        if current != self.signature or _structure(layer) != self.structure:
            raise RuntimeError('Opaque capture projection targets or geometry changed; recapture required')
        return layer


def _make_binding(layer, stats, *, require_cuda=True):
    buffers = getattr(layer, '_kvmem_capture_buffers', None)
    if buffers is None:
        raise ValueError('Opaque capture projection requires persistent capture buffers')
    tensors = (buffers.query, buffers.key, buffers.value)
    capacity = tensors[0].shape[0] if tensors[0].ndim == 3 else 0
    device = layer.q_norm.weight.device
    if require_cuda and device.type != 'cuda':
        raise ValueError('Opaque capture projection requires CUDA capture buffers')
    if (layer.q_size != layer.num_heads * layer.head_dim or
            layer.kv_size != layer.num_kv_heads * layer.head_dim):
        raise ValueError('Opaque capture projection head geometry is inconsistent')
    for tensor, heads in zip(tensors, (layer.num_heads, layer.num_kv_heads, layer.num_kv_heads)):
        if (capacity < 1 or tuple(tensor.shape) != (capacity, heads, layer.head_dim) or
                tensor.dtype != torch.bfloat16 or tensor.device != device or
                not tensor.is_contiguous()):
            raise ValueError('Opaque capture projection requires stable contiguous BF16 buffers')
    # They are write-only independent targets. Alias would also change the old
    # kernel's semantics and cannot be repaired after launching the opaque op.
    intervals = sorted((tensor.data_ptr(), tensor.data_ptr() + tensor.numel()*tensor.element_size())
                       for tensor in tensors)
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError('Opaque capture projection capture buffers must not alias')
    capture_gate = getattr(layer, '_kvmem_capture_gate', None)
    if capture_gate is not None and (capture_gate.dtype != torch.int32 or
            capture_gate.numel() != 1 or capture_gate.device != device):
        raise ValueError('Opaque capture projection requires a device int32 gate')
    if getattr(layer, '_kvmem_mtp_prefix_only', False) and capture_gate is None:
        raise ValueError('Opaque MTP capture requires a dynamic device gate')
    return _ProjectionBinding(weakref.ref(layer), weakref.ref(layer.attn), buffers, capture_gate,
        tuple(_tensor_signature(value) for value in (*tensors, capture_gate)), _structure(layer), stats)


def _forward_context():
    from vllm.forward_context import get_forward_context
    return get_forward_context()


def _project_impl(qkv, positions, layer_name, q_size, kv_size):
    context = _forward_context()
    attention = context.no_compile_layers.get(layer_name)
    binding = getattr(attention, '_kvmem_opaque_projection_binding', None)
    if binding is None:
        raise RuntimeError(f'Opaque capture projection has no bound owner: {layer_name}')
    layer = binding.resolve(attention)
    if (q_size != layer.q_size or kv_size != layer.kv_size or qkv.ndim != 2 or
            qkv.shape[1] != 2*q_size + 2*kv_size or
            qkv.shape[0] > binding.buffers.query.shape[0] or
            qkv.device != binding.buffers.query.device or qkv.dtype != torch.bfloat16):
        raise ValueError('Opaque capture projection input differs from installed geometry/capacity')
    q_gate, key, value = qkv.split([q_size*2, kv_size, kv_size], dim=-1)
    if positions.ndim == 2 and not getattr(layer.rotary_emb, 'mrope_section', None):
        positions = positions[0]
    tokens = qkv.shape[0]
    # Keep the original kernel and every arithmetic/cast argument unchanged.
    q, k, gate, _, _ = fused_qk_rmsnorm_rope_gate(
        q_gate, key, layer.q_norm.weight, layer.k_norm.weight,
        layer.rotary_emb.cos_sin_cache, positions, layer.q_norm.variance_epsilon,
        layer.num_heads, layer.num_kv_heads, layer.head_dim, layer.rotary_emb.rotary_dim,
        norm_beta=1.0, capture_enabled=binding.capture_gate,
        value_input=value,
        value_output=binding.buffers.value[:tokens].view(tokens, -1),
        mrope_section=(getattr(layer.rotary_emb, 'mrope_section', None)
                       if positions.ndim == 2 else None),
        raw_q_output=binding.buffers.query[:tokens].view(tokens, -1),
        raw_k_output=binding.buffers.key[:tokens].view(tokens, -1))
    binding.stats.host_dispatch_count += 1
    return q, k, gate


def _project_fake(qkv, positions, layer_name, q_size, kv_size):
    # No owner/context lookup, buffer mutation or CUDA allocation in tracing.
    return (qkv.new_empty((qkv.shape[0], q_size)),
            qkv.new_empty((qkv.shape[0], kv_size)),
            qkv.new_empty((qkv.shape[0], q_size)))


# Keep the registration/effect handles alive, not a process-global owner map.
_LIBRARY = torch.library.Library('kvmem', 'FRAGMENT')
_LIBRARY.define('capture_projection(Tensor qkv, Tensor positions, str layer_name, '
                'int q_size, int kv_size) -> (Tensor, Tensor, Tensor)')
_LIBRARY.impl('capture_projection', _project_impl, 'CUDA')
_LIBRARY._register_fake('capture_projection', _project_fake)
from torch._library.effects import EffectType
torch.library._register_effectful_op(torch.ops.kvmem.capture_projection.default,
                                    EffectType.ORDERED, lib=_LIBRARY)


def install_opaque_capture(runner, source_fingerprint):
    """Bind all target/MTP owners before any compile or graph capture occurs."""
    modules = dict(runner.model.named_modules())
    layers = [modules[name] for name in runner._kvmem_layers]
    layers.extend(getattr(runner, '_kvmem_mtp_modules', {}).values())
    static = runner.vllm_config.compilation_config.static_forward_context
    stats = OpaqueCaptureStats(source_fingerprint)
    prepared, names = [], set()
    # All validation precedes mutation; a partial installation is not usable.
    for layer in layers:
        attention = layer.attn
        name = attention.layer_name
        if name in names or static.get(name) is not attention:
            raise ValueError('Opaque capture projection requires unique forward-context owners')
        if (hasattr(attention, '_kvmem_opaque_projection_binding') or
                getattr(layer, '_kvmem_opaque_projection', False)):
            raise ValueError('Opaque capture projection is already installed')
        names.add(name)
        prepared.append((layer, attention, _make_binding(layer, stats)))
    if not prepared:
        raise ValueError('Opaque capture projection has no target layers')
    for layer, attention, binding in prepared:
        attention._kvmem_opaque_projection_binding = binding
        layer._kvmem_opaque_projection = True
    stats.layers_installed = len(prepared)
    return stats

"""Explicit text-only Qwen3.8 Flash Next B12X adapter.

Main raw K/V use the ordinary KVMem codec. Selector projections are BF16
*bytes*, carried in auxiliary archive planes after target and MTP layers.
They share accepted-prefix publication, storage leases, GC and retirement with
the main payload, but are excluded from scoring and never quantized/rotated.
The padded planes intentionally trade CPU space for the existing atomic record
format. They must not be interpreted as additional attention layers.
"""
from dataclasses import replace
from types import MethodType, SimpleNamespace
import os

import torch
from ..core import settings as _settings

QSA_MODULE = 'vllm.models.qwen4_exp.nvidia.b12x_qsa'


def is_b12x_qsa(module):
    cls = type(module)
    return cls.__module__ == QSA_MODULE and cls.__name__ == 'Qwen4ExpQSAAttention'


def attention_owner(module):
    return module if is_b12x_qsa(module) else module.attn


def validate_qsa_config(config, options):
    if getattr(getattr(config.model_config, 'hf_text_config', None), 'model_type', None) != 'qwen4_exp_text':
        return False
    compilation = getattr(config, 'compilation_config', None)
    # Full CUDA graphs capture the persistent raw projection copies, including
    # the MTP prefix-only branch. Torch compilation has a different capture
    # contract and remains unqualified for QSA.
    full_graph = (_settings.get('KVMEM_FULL_GRAPH') == '1' and compilation is not None
                  and compilation.mode == 0
                  and getattr(compilation.cudagraph_mode, 'name', None) == 'FULL')
    if ((not config.model_config.enforce_eager and not full_graph) or config.scheduler_config.async_scheduling or
            config.max_concurrent_batches != 1 or config.cache_config.enable_prefix_caching or
            config.cache_config.mamba_cache_mode != 'none' or
            config.cache_config.cache_dtype != 'fp8_e4m3' or
            (config.model_config.multimodal_config is not None and
             not config.model_config.multimodal_config.language_model_only)):
        raise ValueError('Flash Next KVMem requires eager or full_graph text-only synchronous FP8 E4M3, APC-off, Mamba none')
    if any(getattr(config.parallel_config, key, 1) != 1 for key in (
            'tensor_parallel_size', 'pipeline_parallel_size',
            'decode_context_parallel_size', 'prefill_context_parallel_size')):
        raise ValueError('Flash Next KVMem requires one GPU rank')
    # Each of these has additional state/ownership or compiled capture contracts.
    unsupported = ('session_capacity', 'async_refresh', 'async_restore',
                   'rolling_active_window', 'capture_fused_pack', 'capture_opaque_projection',
                   'guided_query_tokens', 'guided_reserve', 'action_pressure',
                   'prefill_capture_ahead', 'prefill_publication_pipeline',
                   'offline_prefill_coalescing', 'prefill_physical_coalescing',
                   'assembly_fused_layers', 'packed_assembly', 'direct_pinned_reads',
                   'reuse_resident_v', 'gpu_raw_k_cache_bytes')
    enabled = [key for key in unsupported if getattr(options, key, False)]
    if (getattr(options, 'action_native_scores', 'off') != 'off' or
            options.memory.action_attention):
        enabled.append('native/action attention scoring')
    if enabled:
        raise ValueError('Flash Next QSA adapter does not yet qualify: ' + ', '.join(enabled))
    if options.memory.block_tokens % 4:
        raise ValueError('QSA archive records must align to four-token compression groups')
    return True


def install_qsa_projection(layer, name):
    from .qsa_contract import verify_qsa_sources
    verify_qsa_sources()
    if (layer.dcp_size != 1 or layer.num_projected_kv_heads != layer.num_kv_heads or
            layer.kv_cache_dtype != 'fp8_e4m3' or layer.compress_ratio != 4 or
            layer.index_head_dim != 128 or layer.overlap_input_projections):
        raise ValueError('QSA capture requires B12X FP8, ratio 4, one rank and projection overlap disabled')
    layer._kvmem_capture_name = name
    layer._kvmem_original_projection = layer._project_qkv_gate
    # Eager native projection is preserved exactly, including its rounding.
    def project(self, qkv, positions):
        from .capture import CapturedLayer, _sink
        sink = _sink.get()
        buffers = getattr(self, '_kvmem_capture_buffers', None)
        active = not getattr(self, '_kvmem_mtp_prefix_only', False) or self._kvmem_mtp_prefix_active
        if not active or (sink is None and buffers is None):
            return self._kvmem_original_projection(qkv, positions)
        qg, k, v = qkv.split((self.q_size * 2, self.kv_size, self.kv_size), dim=-1)
        q, gate = qg.unflatten(-1, (self.num_heads, self.head_dim * 2)).chunk(2, -1)
        raw_q = self.q_norm(q)
        raw_k = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        query, key = self.rotary_emb(positions, raw_q.flatten(-2), raw_k.flatten(-2))
        value = v.view(-1, self.num_kv_heads, self.head_dim)
        record = CapturedLayer(self._kvmem_capture_name, positions, raw_q, raw_k, value)
        if buffers is not None:
            n = qkv.shape[0]
            buffers.query[:n].copy_(raw_q)
            buffers.key[:n].copy_(raw_k)
            buffers.value[:n].copy_(value)
        if sink is not None:
            sink(record)
        return query, key, v, gate.flatten(-2)
    layer._project_qkv_gate = MethodType(project, layer)
    original = layer.indexer.project
    layer._kvmem_index_raw = torch.empty(layer.max_tokens, layer.index_head_dim,
        dtype=torch.bfloat16, device=layer.device)
    def index_project(self, hidden_states):
        query, raw = original(hidden_states)
        if not getattr(layer, '_kvmem_mtp_prefix_only', False) or layer._kvmem_mtp_prefix_active:
            layer._kvmem_index_raw[:raw.shape[0]].copy_(raw)
        return query, raw
    layer.indexer.project = MethodType(index_project, layer.indexer)


def install_qsa_mtp_positions(runner):
    """Feed text MRoPE from the same dense frame as the target projection.

    The vendor's DefaultModelState.get_model_positions reads its RopeState,
    ignoring InputBatch.positions. Replacing the draft batch alone therefore
    leaves MTP at logical-history positions after a KVMem selection. Native
    prepare_prefill_inputs broadcasts this 1D text vector to all MRoPE axes.
    Profiling/capture without a public pending batch keeps the native path.
    """
    proposer = runner.speculator
    original = proposer._target_model_positions
    def positions(self, input_batch, is_profile):
        window = getattr(runner, '_kvmem_window_batch', None)
        if getattr(runner, '_kvmem_pending_capture', None) is not None and window is not None:
            if input_batch.positions is not window.positions:
                raise ValueError('QSA MTP positions must use the owned dense target frame')
            return input_batch.positions
        return original(input_batch, is_profile)
    proposer._target_model_positions = MethodType(positions, proposer)


def install_qsa_archive(runner):
    modules = dict(runner.model.named_modules())
    modules.update(getattr(runner, '_kvmem_mtp_modules', {}))
    names = list(getattr(runner, '_kvmem_archive_layers', runner._kvmem_layers))
    qsa = [(name, modules[name]) for name in names if is_b12x_qsa(modules[name])]
    if not qsa:
        return
    if len(qsa) != len(names):
        raise ValueError('Mixed QSA and ordinary attention archive is not qualified')
    runner._kvmem_qsa_layers = dict(qsa)
    auxiliary, records = {}, []
    from .capture import CapturedLayer
    for name, layer in qsa:
        aux_name = '__qsa_index__.' + name
        width = layer.num_kv_heads * layer.head_dim
        if layer.index_head_dim * 2 > width:
            raise ValueError('BF16 selector projection exceeds auxiliary archive plane')
        zeros = torch.zeros(layer.max_tokens, layer.num_kv_heads, layer.head_dim,
            dtype=torch.bfloat16, device=layer.device)
        records.append(CapturedLayer(aux_name, runner.input_buffers.positions,
            zeros, zeros, zeros, layer._kvmem_index_raw))
        auxiliary[aux_name] = SimpleNamespace(num_kv_heads=layer.num_kv_heads,
            head_dim=layer.head_dim, num_heads=layer.num_kv_heads,
            attn=SimpleNamespace(kv_cache_dtype='fp8_e4m3',
                _v_scale=torch.ones((), device=layer.device)))
    runner._kvmem_aux_modules = auxiliary
    runner._kvmem_qsa_aux_captures = records
    runner._kvmem_archive_layers = names + list(auxiliary)
    runner._kvmem_qsa_pending_rings = []


def append_qsa_captures(runner, captured):
    auxiliary = getattr(runner, '_kvmem_qsa_aux_captures', ())
    if not auxiliary:
        return captured
    return [*captured, *(replace(c, positions=captured[0].positions) for c in auxiliary)]


def write_auxiliary_bytes(values, captured, rows=None, tokens=None):
    """Overwrite only explicitly declared auxiliary planes, before the D2H fence."""
    for index, record in enumerate(captured):
        source = getattr(record, 'archive_value_bytes', None)
        if source is None:
            continue
        source = source[:tokens] if rows is None else source.index_select(0, rows)
        payload = source.contiguous().view(torch.uint8).reshape(source.shape[0], -1)
        destination = values[index].view(torch.uint8).reshape(source.shape[0], -1)
        if payload.shape[1] > destination.shape[1]:
            raise ValueError('Auxiliary projection exceeds its archive payload')
        destination.zero_()
        destination[:, :payload.shape[1]].copy_(payload)


def finish_qsa_refresh(runner):
    """Publish rebuilt rings after provisional-query state has been rolled back."""
    pending = getattr(runner, '_kvmem_qsa_pending_rings', ())
    for layer, index, positions, raw, selected in pending:
        ring = positions % layer.raw_ring_capacity
        layer._raw_k_ring[index].zero_()
        layer._raw_logical_positions[index].fill_(-1)
        layer._raw_rope_positions[index].fill_(-1)
        layer._raw_k_ring[index, ring] = raw
        layer._raw_logical_positions[index, ring] = positions
        layer._raw_rope_positions[index, ring] = positions[:, None].expand(-1, layer.position_axes)
        accepted = runner.model_state.qsa_committed_num_accepted_tokens_gpu[index]
        layer._raw_interval_start_positions[index].copy_(selected - accepted)
        layer.set_skip_topk(False)
    if pending:
        # Metadata contains copies made before rollback; force fresh staging.
        from vllm.forward_context import get_forward_context
        context = get_forward_context()
        for key in tuple(context.additional_kwargs):
            if key.startswith('qwen38_qsa'):
                context.additional_kwargs.pop(key)
        runner._kvmem_qsa_pending_rings.clear()


def borrow_qsa_state(runner, request_id, index):
    """Snapshot request-local QSA state alongside GDN before a private forward."""
    result = {}
    for name, module in getattr(runner, '_kvmem_qsa_layers', {}).items():
        for kind, tensor in enumerate(module.get_recurrent_checkpoint_tensors()):
            result[f'{request_id}/{name}/qsa/{kind}'] = tensor[index:index+1]
    if result:
        for kind, tensor in enumerate(runner.model_state.get_recurrent_checkpoint_tensors()):
            result[f'{request_id}/qsa_control/{kind}'] = tensor[index:index+1]
    return result

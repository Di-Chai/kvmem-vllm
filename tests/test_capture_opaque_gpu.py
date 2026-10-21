"""Real CUDA/Inductor qualification, independent of full vLLM piecewise tests."""
import os
from types import SimpleNamespace as NS

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='explicit GPU opt-in required')


class Owner:
    pass


def fixture(tokens, *, capacity=None, mrope=False, draft=False):
    import torch
    from vllm.forward_context import ForwardContext
    from vllm_kvmem.capture import CapturedLayer
    from vllm_kvmem.capture_compile import capture_compile_fingerprint
    from vllm_kvmem.capture_opaque import install_opaque_capture
    torch.manual_seed(769)
    capacity = max(tokens+7, 64) if capacity is None else capacity
    heads, kvheads, dim, rotary = 24, 4, 256, 64
    weights = [torch.randn(dim, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
    angles = torch.randn(max(32768, tokens+32), rotary//2, device='cuda')
    trig = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    layers, contexts = [], []
    for enabled in (False, True):
        layer, attn = Owner(), Owner()
        attn.layer_name = 'fixture.attn'
        layer.attn = attn
        layer.q_size, layer.kv_size = heads*dim, kvheads*dim
        layer.num_heads, layer.num_kv_heads, layer.head_dim = heads, kvheads, dim
        layer.q_norm = NS(weight=weights[0], variance_epsilon=1e-6)
        layer.k_norm = NS(weight=weights[1])
        layer.rotary_emb = NS(rotary_dim=rotary, cos_sin_cache=trig,
                             mrope_section=[11, 11, 10] if mrope else None)
        layer._kvmem_capture_name = 'fixture'
        layer._kvmem_capture_buffers = CapturedLayer('fixture', None,
            torch.full((capacity, heads, dim), -3, dtype=torch.bfloat16, device='cuda'),
            torch.full((capacity, kvheads, dim), -5, dtype=torch.bfloat16, device='cuda'),
            torch.full((capacity, kvheads, dim), -7, dtype=torch.bfloat16, device='cuda'))
        if draft:
            layer._kvmem_mtp_prefix_only = True
            layer._kvmem_capture_gate = torch.ones((), dtype=torch.int32, device='cuda')
        context = ForwardContext(no_compile_layers={attn.layer_name: attn},
                                 attn_metadata={}, slot_mapping={})
        if enabled:
            runner = NS(model=NS(named_modules=lambda: [('fixture', layer)]),
                _kvmem_layers=['fixture'], _kvmem_mtp_modules={}, vllm_config=NS(
                    compilation_config=NS(static_forward_context=context.no_compile_layers)))
            layer.stats = install_opaque_capture(runner, capture_compile_fingerprint())
        layers.append(layer)
        contexts.append(context)
    x = torch.randn(tokens, heads*dim*2 + kvheads*dim*2, dtype=torch.bfloat16, device='cuda')
    positions = torch.arange(tokens, device='cuda')+8
    if mrope:
        positions = torch.stack((positions, positions+1, positions+2))
    return NS(layers=layers, contexts=contexts, x=x, positions=positions, capacity=capacity)


def raw(layer):
    value = layer._kvmem_capture_buffers
    return (value.query, value.key, value.value)


def same_bits(left, right):
    import torch
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def functions(data, compiled):
    import torch
    from vllm_kvmem.capture import _project_with_capture
    result = []
    for layer in data.layers:
        def forward(x, positions, owner=layer):
            return _project_with_capture(owner, x, positions)
        result.append(torch.compile(forward, fullgraph=True) if compiled else forward)
    return result


def compare(data, forwards):
    from vllm.forward_context import override_forward_context
    outputs = []
    for context, forward in zip(data.contexts, forwards):
        with override_forward_context(context):
            outputs.append(forward(data.x, data.positions))
    assert all(same_bits(left, right) for left, right in zip(*outputs))
    assert all(same_bits(left, right) for left, right in zip(*(raw(layer) for layer in data.layers)))
    return outputs


@pytest.mark.parametrize('tokens,mrope,draft', [(1, False, False), (37, True, True),
    (2048, True, False), (14336, True, False)])
def test_opaque_projection_exact_outputs_full_capture_and_padding(tokens, mrope, draft):
    import torch
    data = fixture(tokens, capacity=16424, mrope=mrope, draft=draft)
    pointers = tuple(t.data_ptr() for layer in data.layers for t in raw(layer))
    compare(data, functions(data, compiled=False))
    if tokens >= 2048:
        compare(data, functions(data, compiled=True))
    assert pointers == tuple(t.data_ptr() for layer in data.layers for t in raw(layer))
    for layer in data.layers:
        for tensor, sentinel in zip(raw(layer), (-3, -5, -7)):
            assert torch.all(tensor[tokens:] == sentinel)
    torch.cuda.synchronize()
    torch._dynamo.reset()


def test_compiled_graph_replay_gate_positions_query_shape_and_owner_isolation():
    import torch
    from vllm.forward_context import override_forward_context
    data = fixture(37, mrope=True, draft=True)
    forwards = functions(data, compiled=True)
    compare(data, forwards)
    graphs, outputs = [], []
    for context, forward in zip(data.contexts, forwards):
        graph = torch.cuda.CUDAGraph()
        with override_forward_context(context), torch.cuda.graph(graph):
            outputs.append(forward(data.x, data.positions))
        graphs.append(graph)
    # No Python dispatch during replay, nor lookup in whatever owner context is
    # current at replay time: each graph keeps its original fixed addresses.
    host_dispatches = data.layers[1].stats.host_dispatch_count
    prefix = [tensor.clone() for tensor in raw(data.layers[1])]
    for layer in data.layers:
        layer._kvmem_capture_gate.zero_()
    data.x.normal_()
    data.positions.add_(2)
    for graph in graphs:
        graph.replay()
    assert all(same_bits(left, right) for left, right in zip(*outputs))
    assert all(same_bits(left, right) for left, right in zip(prefix, raw(data.layers[1])))
    for layer in data.layers:
        layer._kvmem_capture_gate.fill_(1)
    for graph in graphs:
        graph.replay()
    assert all(same_bits(left, right) for left, right in zip(*outputs))
    assert all(same_bits(left, right) for left, right in zip(*(raw(layer) for layer in data.layers)))
    assert not same_bits(prefix[0], raw(data.layers[1])[0])
    assert data.layers[1].stats.host_dispatch_count == host_dispatches
    # Only the candidate graph runs: the independent same-name owner must stay
    # untouched. A fresh private Query size can subsequently use the same bank.
    old = [tensor.clone() for tensor in raw(data.layers[0])]
    data.x.normal_()
    graphs[1].replay()
    assert all(same_bits(left, right) for left, right in zip(old, raw(data.layers[0])))
    compare(data, forwards)
    data.x, data.positions = data.x[:3], data.positions[:, :3]
    compare(data, forwards)
    torch.cuda.synchronize()
    torch._dynamo.reset()


def test_actual_compiled_dispatch_removes_capture_clones_and_copybacks():
    import torch
    from vllm.forward_context import override_forward_context
    data = fixture(2048, capacity=16424, mrope=True)
    forwards = functions(data, compiled=True)
    compare(data, forwards)
    names = []
    for context, forward in zip(data.contexts, forwards):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            with override_forward_context(context):
                forward(data.x, data.positions)
            torch.cuda.synchronize()
        names.append([event.name for event in profile.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA])
    old, new = names
    assert sum('fused_qk_rmsnorm_rope_gate_kernel' in name for name in old) == 1, old
    assert len(old) > 1 and any('clone' in name or 'copy' in name or 'triton_poi_fused' in name for name in old), old
    assert len(new) == 1 and 'fused_qk_rmsnorm_rope_gate_kernel' in new[0], new
    assert all(same_bits(left, right) for left, right in zip(*(raw(layer) for layer in data.layers)))
    output = os.getenv('KVMEM_OPAQUE_KERNEL_PROOF')
    if output:
        import json
        from pathlib import Path
        Path(output).write_text(json.dumps({'old_cuda_kernels': old, 'opaque_cuda_kernels': new,
            'stats': data.layers[1].stats.snapshot(), 'byte_exact': True}, indent=2)+'\n')
    torch._dynamo.reset()


def test_real_cuda_capacity_context_rejection_does_not_write():
    import torch
    from vllm.forward_context import override_forward_context
    from vllm_kvmem.capture_opaque import _project_impl
    data = fixture(3, capacity=4)
    before = [tensor.clone() for tensor in raw(data.layers[1])]
    with override_forward_context(data.contexts[1]):
        with pytest.raises(ValueError, match='geometry/capacity'):
            _project_impl(data.x.repeat(2, 1), data.positions.repeat(2), 'fixture.attn', 6144, 1024)
    with override_forward_context(data.contexts[0]):
        with pytest.raises(RuntimeError, match='no bound owner'):
            _project_impl(data.x, data.positions, 'fixture.attn', 6144, 1024)
    assert all(same_bits(left, right) for left, right in zip(before, raw(data.layers[1])))
    torch.cuda.synchronize()

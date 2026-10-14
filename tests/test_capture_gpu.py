import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("KVMEM_RUN_GPU_TESTS") != "1", reason="explicit GPU opt-in required")


@pytest.mark.parametrize("tokens,heads,kvheads,dim,rotary", [(1, 24, 4, 256, 64), (37, 24, 4, 256, 64), (17, 8, 2, 128, 128)])
def test_fused_capture_preserves_upstream_outputs(tokens, heads, kvheads, dim, rotary):
    import torch
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate as upstream
    from vllm_kvmem._fused_capture import fused_qk_rmsnorm_rope_gate as captured
    generator = torch.Generator(device="cuda").manual_seed(764)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    # Match native non-contiguous QKV split views.
    fused = rand(tokens, heads * 2 * dim + kvheads * 2 * dim)
    qgate, key, value = fused.split([heads * 2 * dim, kvheads * dim, kvheads * dim], -1)
    qw, kw = rand(dim), rand(dim)
    angles = torch.randn(512, rotary//2, generator=generator, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    positions = torch.arange(tokens, device="cuda") + 8
    args = (qgate, key, qw, kw, cache, positions, 1e-6, heads, kvheads, dim, rotary)
    expected = upstream(*args, norm_beta=1.0)
    actual = captured(*args, norm_beta=1.0)
    for before, after in zip(expected, actual[:3]):
        assert torch.equal(before, after)
    raw = key.reshape(tokens, kvheads, dim).float()
    oracle = (raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-6)
              * (kw.float() + 1)).bfloat16()
    torch.testing.assert_close(actual[4].view_as(oracle), oracle, rtol=0.008, atol=0.016)


def test_content_outputs_are_updated_by_graph_replay_without_python_callback():
    import torch
    from vllm_kvmem._fused_capture import fused_qk_rmsnorm_rope_gate
    tokens, heads, kvheads, dim, rotary = 8, 24, 4, 256, 64
    q = torch.randn(tokens, heads * 2 * dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(tokens, kvheads * dim, device='cuda', dtype=torch.bfloat16)
    weight = torch.zeros(dim, device='cuda', dtype=torch.bfloat16)
    angles = torch.randn(512, rotary // 2, device='cuda')
    cache = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    positions = torch.arange(tokens, device='cuda')
    raw_q = torch.empty(tokens, heads * dim, device='cuda', dtype=torch.bfloat16)
    raw_k = torch.empty_like(k)
    args = (q, k, weight, weight, cache, positions, 1e-6, heads, kvheads, dim, rotary)
    def forward():
        return fused_qk_rmsnorm_rope_gate(*args, norm_beta=1,
            raw_q_output=raw_q, raw_k_output=raw_k)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    original = raw_k.clone()
    k.copy_(torch.randn_like(k)); q.copy_(torch.randn_like(q))
    graph.replay()
    assert not torch.equal(original, raw_k)
    expected = fused_qk_rmsnorm_rope_gate(*args, norm_beta=1)
    assert torch.equal(raw_q, expected[3])
    assert torch.equal(raw_k, expected[4])


def test_native_compile_preserves_mutated_capture_buffers_across_replays():
    import torch
    from vllm_kvmem._fused_capture import fused_qk_rmsnorm_rope_gate
    heads, kvheads, dim, rotary = 24, 4, 256, 64
    q = torch.randn(8, heads * 2 * dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(8, kvheads * dim, device='cuda', dtype=torch.bfloat16)
    weight = torch.zeros(dim, device='cuda', dtype=torch.bfloat16)
    angles = torch.randn(512, rotary // 2, device='cuda')
    cache = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    positions = torch.arange(8, device='cuda')
    raw_q = torch.empty(8, heads * dim, device='cuda', dtype=torch.bfloat16)
    raw_k = torch.empty_like(k)
    def forward(q, k, positions):
        return fused_qk_rmsnorm_rope_gate(q, k, weight, weight, cache,
            positions, 1e-6, heads, kvheads, dim, rotary, norm_beta=1,
            raw_q_output=raw_q, raw_k_output=raw_k)[:3]
    compiled = torch.compile(forward, fullgraph=True)
    compiled(q, k, positions)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = compiled(q, k, positions)
    first = raw_k.clone()
    for _ in range(2):
        q.copy_(torch.randn_like(q)); k.copy_(torch.randn_like(k))
        positions.add_(1)
        graph.replay()
        expected = fused_qk_rmsnorm_rope_gate(q, k, weight, weight, cache,
            positions, 1e-6, heads, kvheads, dim, rotary, norm_beta=1)
        for actual, oracle in zip(result, expected[:3]):
            assert torch.equal(actual, oracle)
        assert torch.equal(raw_q, expected[3])
        assert torch.equal(raw_k, expected[4])
        assert not torch.equal(raw_k, first)


def test_compiled_device_gate_protects_all_prefix_captures_from_later_drafts():
    import torch
    from vllm_kvmem._fused_capture import fused_qk_rmsnorm_rope_gate
    q = torch.randn(4, 24 * 2 * 256, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(4, 4 * 256, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    weight = torch.zeros(256, device='cuda', dtype=torch.bfloat16)
    angles = torch.randn(512, 32, device='cuda')
    cache = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    pos = torch.arange(4, device='cuda')
    raw_q = torch.empty(4, 24 * 256, device='cuda', dtype=torch.bfloat16)
    raw_k, raw_v = torch.empty_like(k), torch.empty_like(v)
    gate = torch.ones((), dtype=torch.int32, device='cuda')
    def forward(q, k, v, gate):
        return fused_qk_rmsnorm_rope_gate(q, k, weight, weight, cache, pos,
            1e-6, 24, 4, 256, 64, norm_beta=1, raw_q_output=raw_q,
            raw_k_output=raw_k, value_input=v, value_output=raw_v,
            capture_enabled=gate)[:3]
    compiled = torch.compile(forward, fullgraph=True)
    compiled(q, k, v, gate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = compiled(q, k, v, gate)
    prefix = [x.clone() for x in (raw_q, raw_k, raw_v)]
    gate.zero_()
    q.normal_(); k.normal_(); v.normal_()
    graph.replay()
    for actual, before in zip((raw_q, raw_k, raw_v), prefix):
        assert torch.equal(actual, before)
    oracle = fused_qk_rmsnorm_rope_gate(q, k, weight, weight, cache, pos,
                                     1e-6, 24, 4, 256, 64, norm_beta=1)
    for actual, expected in zip(result, oracle[:3]):
        assert torch.equal(actual, expected)
    gate.fill_(1)
    graph.replay()
    assert torch.equal(raw_q, oracle[3]) and torch.equal(raw_k, oracle[4])
    assert torch.equal(raw_v, v)

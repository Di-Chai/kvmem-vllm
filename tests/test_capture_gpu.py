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

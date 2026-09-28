"""Canonical accepted state versus independently evaluated native prefixes."""
import os
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')
pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='Explicit GPU qualification only')


@pytest.mark.parametrize('dim_first', [False, True])
@pytest.mark.parametrize('retained', [1, 3, 8])
def test_canonical_state_matches_native_prefix_conv_and_gdn(dim_first, retained):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
    from vllm_kvmem.native_state import NativeRecurrentState, verify_state_sources
    verify_state_sources()
    torch.manual_seed(913)
    query, width, features, blocks = 8, 4, 128, 11
    heads, key_dim, value_dim = 2, 64, 32
    length = width - 1 + query - 1
    conv = torch.randn((blocks, features, length) if dim_first else
                       (blocks, length, features), device='cuda', dtype=torch.bfloat16)
    ssm = torch.randn(blocks, heads, value_dim, key_dim, device='cuda', dtype=torch.float32) * .01
    ref_conv, ref_ssm = conv.clone(), ssm.clone()
    untouched_conv, untouched_ssm = conv[9:].clone(), ssm[9:].clone()
    rows = torch.arange(1, query+1, device='cuda', dtype=torch.int32).reshape(1, -1)
    starts = torch.tensor([0, query], device='cuda', dtype=torch.int32)
    one = torch.ones(1, device='cuda', dtype=torch.int32)
    x = torch.randn(query, features, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(features, width, device='cuda', dtype=torch.bfloat16)
    def conv_view(cache): return cache if dim_first else cache.transpose(1, 2)
    causal_conv1d_update(x.clone(), conv_view(conv), weights, activation='silu',
                        conv_state_indices=rows[:, 0], num_accepted_tokens=one,
                        query_start_loc=starts, max_query_len=query)
    for i in range(retained):
        causal_conv1d_update(x[i:i+1].clone(), conv_view(ref_conv), weights,
                            activation='silu', conv_state_indices=rows[:, 0])
    q = torch.randn(1, query, heads, key_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, query, heads, value_dim, device='cuda', dtype=torch.bfloat16)
    a, b = [torch.randn(query, heads, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    A = torch.zeros(heads, device='cuda')
    bias = torch.zeros_like(A)
    fused_sigmoid_gating_delta_rule_update(A, a, b, bias, q, k, v,
        initial_state=ssm, cu_seqlens=starts, ssm_state_indices=rows,
        num_accepted_tokens=one, use_qk_l2norm_in_kernel=True)
    # The native fused update writes one state-index column per input token,
    # including when num_accepted_tokens is absent. Ordinary reference decode
    # therefore uses one token per call, rather than passing a multi-token query
    # with a one-column state table (which is not a supported native layout).
    single = torch.tensor([0, 1], device='cuda', dtype=torch.int32)
    for i in range(retained):
        fused_sigmoid_gating_delta_rule_update(A, a[i:i+1], b[i:i+1], bias,
            q[:, i:i+1].contiguous(), k[:, i:i+1].contiguous(), v[:, i:i+1].contiguous(),
            initial_state=ref_ssm, cu_seqlens=single,
            ssm_state_indices=rows[:, 0], use_qk_l2norm_in_kernel=True)
    owner = NativeRecurrentState.__new__(NativeRecurrentState)
    owner.conv_dim_first, owner.conv_widths = dim_first, {'gdn': width}
    owner.groups = {0: {'gdn': (conv, ssm)}}
    owner.runner = NS(req_states=NS(req_id_to_index={'a': 0}), _kvmem_request_generations={'a': 1},
        model_state=NS(num_accepted_tokens_gpu=torch.tensor([query], device='cuda', dtype=torch.int32)),
        block_tables=NS(num_blocks=NS(np=torch.tensor([[query]]).numpy()),
                        block_tables=[NS(gpu=rows)]))
    event = owner.canonicalize_accepted('a', retained_tokens=retained, query_tokens=query, speculative=True)
    event.synchronize()
    assert owner.runner.model_state.num_accepted_tokens_gpu.item() == 1
    torch.testing.assert_close(conv_view(conv)[1, :, :width-1],
                               conv_view(ref_conv)[1, :, :width-1], rtol=0, atol=0)
    torch.testing.assert_close(ssm[1], ref_ssm[1], rtol=1e-5, atol=1e-5)
    probe = torch.randn(1, features, device='cuda', dtype=torch.bfloat16)
    actual = causal_conv1d_update(probe.clone(), conv_view(conv), weights, activation='silu',
                                  conv_state_indices=rows[:, 0])
    expected = causal_conv1d_update(probe.clone(), conv_view(ref_conv), weights, activation='silu',
                                    conv_state_indices=rows[:, 0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(conv[9:], untouched_conv) and torch.equal(ssm[9:], untouched_ssm)

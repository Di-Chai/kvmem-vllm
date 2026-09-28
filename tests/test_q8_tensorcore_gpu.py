import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("KVMEM_RUN_GPU_TESTS") != "1", reason="explicit GPU opt-in required")


def test_activation_quantization_matches_cuda_roundf_contract():
    import torch
    from vllm_kvmem.q8_tensorcore import quantize_activations

    x = torch.zeros(3, 64, device="cuda", dtype=torch.float32)
    x[0, :8] = torch.tensor([127., -127., .5, -.5, 1.5, -1.5, 2.5, -2.5], device="cuda")
    x[1].copy_(torch.linspace(-.03, .021, 64, device="cuda"))
    q, d = quantize_activations(x)
    grouped = x.view(3, 2, 32)
    scale = grouped.abs().amax(-1) / 127
    ratio = grouped / torch.where(scale == 0, 1., scale)[..., None]
    expected = (ratio.abs() + .5).floor() * ratio.sign()
    assert torch.equal(q.view_as(grouped), expected.to(torch.int8))
    assert torch.equal(d, scale.half())
    assert q[0, :8].tolist() == [127, -127, 1, -1, 2, -2, 3, -3]


def test_activation_bytes_match_pinned_fast_math_cuda_via_identity_weights():
    # Identity Q8 weights expose native Q8_1 values without adding a test-only
    # CUDA quantizer that could repeat our implementation's rounding mistake.
    import numpy as np
    import torch
    from vllm_gguf_plugin import ops
    from vllm_kvmem.q8_tensorcore import quantize_activations

    rng = np.random.default_rng(4919)
    x = torch.from_numpy(rng.standard_normal((256, 512), dtype=np.float32)).cuda().bfloat16()
    identity = np.zeros((512, 16, 34), dtype=np.uint8)
    identity[..., :2] = np.array([1.], dtype=np.float16).view(np.uint8)
    for i in range(512):
        identity[i, i // 32, 2 + i % 32] = 1
    w = torch.from_numpy(identity.reshape(512, -1)).cuda()
    native = ops.ggml_mul_mat_a8(w, x.float(), 8, 512)
    q, d = quantize_activations(x)
    assert torch.equal(native, q.float() * d.repeat_interleave(32, 1))


@pytest.mark.parametrize("rows,outputs,inputs,split", [(1, 17, 64, 1), (8, 129, 160, 4), (32, 256, 512, 8), (65, 128, 1024, 2)])
def test_grouped_integer_dot_preserves_q8_scales(rows, outputs, inputs, split):
    import numpy as np
    import torch
    import gguf
    from vllm_gguf_plugin import ops
    from vllm_kvmem.q8_tensorcore import q8_tensorcore_linear, quantize_activations

    rng = np.random.default_rng(1717)
    weights = rng.normal(size=(outputs, inputs)).astype(np.float32) / np.sqrt(inputs)
    packed = gguf.quants.quantize(weights, gguf.GGMLQuantizationType.Q8_0)
    w = torch.from_numpy(packed).cuda()
    saved = w.clone()
    x = torch.from_numpy(rng.normal(size=(rows, inputs)).astype(np.float32)).cuda().bfloat16()
    actual = q8_tensorcore_linear(x, w, split_k=split)
    aq, ad = quantize_activations(x)
    blocks = packed.reshape(outputs, inputs // 32, 34)
    wd = torch.from_numpy(blocks[..., :2].copy().view(np.float16).squeeze(-1)).cuda().double()
    wq = torch.from_numpy(blocks[..., 2:].copy().view(np.int8)).cuda().double()
    dots = torch.einsum('mgk,ngk->mgn', aq.view(rows, inputs // 32, 32).double(), wq)
    reference = (dots * ad.double()[..., None] * wd.T[None]).sum(1).bfloat16()
    assert torch.equal(w, saved)
    torch.testing.assert_close(actual, reference, rtol=.008, atol=1e-5)
    # Pinned plugin pads activation rows to 512; original weights stay unpadded.
    native = ops.ggml_mul_mat_a8(w, x, int(gguf.GGMLQuantizationType.Q8_0), outputs)
    torch.testing.assert_close(actual, native, rtol=.008, atol=1e-5)
    # F32 output removes BF16 rounding boundaries and tests FP32 accumulation
    # against the same independent FP64 group-dot reference.
    actual_float = q8_tensorcore_linear(x.float(), w, split_k=split)
    reference_float = (dots * ad.double()[..., None] * wd.T[None]).sum(1).float()
    torch.testing.assert_close(actual_float, reference_float, rtol=2e-5, atol=2e-6)

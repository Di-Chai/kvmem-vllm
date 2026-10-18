from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')


def test_context_projection_uses_codec_forward_instead_of_transposed_weight_storage():
    from vllm_kvmem.draft_quantization import install_quantized_context
    torch.manual_seed(73)
    weights = [torch.randn(8, 5) for _ in range(2)]
    calls = []
    class Projection:
        def __init__(self, index):
            self.index = index
            self.weight = weights[index].t()  # Native packed/transposed storage
        def __call__(self, value):
            calls.append(self.index)
            return torch.nn.functional.linear(value, weights[self.index]), None
    model = NS(hidden_norm=lambda x: x,
        layers=[NS(self_attn=NS(qkv_proj=Projection(i), q_size=4, kv_size=2)) for i in range(2)])
    runner = NS(speculative_config=NS(method='dflash', quantization='fp8'),
                speculator=NS(model=NS(model=model)))
    install_quantized_context(runner)
    inputs = torch.randn(3, 5)
    keys, values = model._project_context_kv(inputs, 3, 2, 1, 2)
    independent = [torch.nn.functional.linear(inputs, w) for w in weights]
    assert calls == [0, 1]
    torch.testing.assert_close(keys, torch.stack([x[:, 4:6].reshape(3, 1, 2) for x in independent]))
    torch.testing.assert_close(values, torch.stack([x[:, 6:8].reshape(3, 1, 2) for x in independent]))
def test_mtp_inherited_fp8_does_not_enter_dflash_patch(monkeypatch):
    from types import SimpleNamespace
    from vllm_kvmem.draft_quantization import install_quantized_context
    import vllm_kvmem.draft_contract as contract

    def forbidden():
        raise AssertionError('Native MTP must not run DFlash verification')

    monkeypatch.setattr(contract, 'verify_draft_sources', forbidden)
    # No DFlash model is provided: it must not even be accessed.
    runner = SimpleNamespace(speculative_config=SimpleNamespace(method='mtp', quantization='fp8'))
    install_quantized_context(runner)

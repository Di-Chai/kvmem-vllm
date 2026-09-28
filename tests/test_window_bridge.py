from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from vllm_kvmem.window_batch import WindowBatch, contextual_metadata
from vllm_kvmem.metadata import WindowLengths
from vllm_kvmem.window_bridge import install_window_bridge


def test_instance_bridge_preserves_recurrent_inputs_and_restores_context():
    batch = SimpleNamespace(positions=torch.tensor([100, 200]))
    window_positions = torch.tensor([4, 5])
    lens = torch.tensor([5, 6], dtype=torch.int32)
    window = WindowBatch(window_positions, {0: WindowLengths(lens, lens, 6, window_positions)})

    class Native:
        def prepare_attn(self, received):
            assert received is batch
            assert received.positions.tolist() == [100, 200]
            return contextual_metadata(None)

        def prepare_inputs(self, received, state):
            return {"positions": received.positions.expand(3, -1), "marker": state}

    native = Native()
    install_window_bridge(native, lambda _: window)
    metadata = native.prepare_attn(batch)
    assert metadata.get_extra_common_attn_kwargs(0, 2)["positions"] is window_positions
    assert metadata.get_extra_common_attn_kwargs(1, 2) == {}
    result = native.prepare_inputs(batch, "unchanged")
    assert result["positions"].tolist() == [[4, 5]] * 3
    assert result["marker"] == "unchanged"
    assert contextual_metadata(None) is None
    with pytest.raises(ValueError, match="already"):
        install_window_bridge(native, lambda _: window)

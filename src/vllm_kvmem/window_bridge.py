"""Instance-scoped bridge to native model-state metadata and position inputs."""
from types import MethodType

from .window_batch import attention_windows


def install_window_bridge(model_state, current_window):
    """The runner supplies its current batch; native recurrent metadata is intact."""
    if hasattr(model_state, "_kvmem_original_prepare_attn"):
        raise ValueError("Window bridge already installed")
    model_state._kvmem_original_prepare_attn = model_state.prepare_attn
    model_state._kvmem_original_prepare_inputs = model_state.prepare_inputs

    def prepare_attn(self, input_batch, *args, **kwargs):
        window = current_window(input_batch)
        with attention_windows(window.groups if window is not None else None):
            return self._kvmem_original_prepare_attn(input_batch, *args, **kwargs)

    def prepare_inputs(self, input_batch, req_states):
        result = self._kvmem_original_prepare_inputs(input_batch, req_states)
        window = current_window(input_batch)
        if window is None:
            return result
        # Initial support is text-only. MRoPE's axes coincide for text tokens;
        # multimodal position maps need a separate qualified model adapter.
        original_positions = result.get("positions")
        positions = window.positions
        if original_positions is not None and original_positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(original_positions.shape[0], -1)
        return {**result, "positions": positions}

    model_state.prepare_attn = MethodType(prepare_attn, model_state)
    model_state.prepare_inputs = MethodType(prepare_inputs, model_state)

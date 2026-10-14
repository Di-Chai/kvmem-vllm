"""FULL decode graphs with original KVMem authority and explicit refresh batches.

No semantic transaction is captured in a graph. Refresh commands dispatch an
ordinary uncaptured control batch (with compiled model forward when enabled),
so MeanK query/probe, GDN rollback, KV assembly and
connector acknowledgments retain the original ownership order.
"""
from types import MethodType

from .window_batch import WindowBatchAdapter
from .cache_groups import full_attention_groups
from .options import RuntimeOptions


def initialize_graph_window(runner):
    cfg = RuntimeOptions.from_vllm(runner.vllm_config).memory
    groups = full_attention_groups(runner.kv_cache_config)
    adapter = WindowBatchAdapter(groups, cfg.select_budget + cfg.gen_budget)
    adapter.reserve_graph_buffers(runner.max_num_tokens, runner.max_num_reqs, runner.device)
    runner._kvmem_window_adapter = adapter
    for record in runner._kvmem_graph_captures:
        record.positions = adapter.buffers[0]
    manager = runner.cudagraph_manager
    if not hasattr(manager, '_kvmem_original_fullgraph'):
        manager._kvmem_original_fullgraph = manager.run_fullgraph
        def run_fullgraph(self, descriptor):
            if (runner._kvmem_probe_output / 'armed').is_file():
                runner._kvmem_graph_decode_calls = getattr(runner, '_kvmem_graph_decode_calls', 0) + 1
            return self._kvmem_original_fullgraph(descriptor)
        manager.run_fullgraph = MethodType(run_fullgraph, manager)
    state = runner.model_state
    # Native graph capture builds inputs outside prepare_inputs(). Use the same
    # stable dense-position buffer in both capture and subsequent execution.
    if not hasattr(state, '_kvmem_original_dummy_inputs'):
        state._kvmem_original_dummy_inputs = state.prepare_dummy_inputs
        def prepare_dummy_inputs(self, num_reqs, num_tokens):
            result = self._kvmem_original_dummy_inputs(num_reqs, num_tokens)
            positions = runner._kvmem_window_adapter.buffers[0][:num_tokens]
            original = result.get('positions')
            if original is not None and original.ndim == 2:
                positions = positions.unsqueeze(0).expand(original.shape[0], -1)
            return {**result, 'positions': positions}
        state.prepare_dummy_inputs = MethodType(prepare_dummy_inputs, state)


def needs_refresh_batch(scheduler_output):
    from .connector import ControlMetadata
    control = scheduler_output.kv_connector_metadata
    return isinstance(control, ControlMetadata) and bool(control.commands)

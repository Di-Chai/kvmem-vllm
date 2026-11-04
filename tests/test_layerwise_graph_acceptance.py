"""Graph publication must commit/abort every layer with the shared history."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.layerwise_pressure import LayerwisePrefillPressure
from vllm_kvmem.block_graph import BlockGraph
from test_graph_capture import action, history, prefill_packet


@pytest.fixture(params=[False, True])
def case(tmp_path, request):
    runtime, shared = action(tmp_path), history()
    controller = LayerwisePrefillPressure.__new__(LayerwisePrefillPressure)
    controller.names = ('layer0', 'layer1')
    controller.options = shared.options
    controller.histories = {}
    controller.buffered_history = request.param
    controller._history('r', shared.options)
    runtime.runner = NS(_kvmem_layerwise_pressure=controller)
    LayerwisePrefillPressure.install_acceptance()
    packet = prefill_packet(shared, 0, 8, [7], [0, 4], [[.2, .8]])
    pending = packet.action_inputs['r']
    pending.layerwise_pressure = (controller, 'r', [[0, 4], [0, 40]],
                                 torch.tensor([[[.2, .8]], [[.7, .3]]]))
    graph = BlockGraph('layerwise-transaction', 4)
    yield runtime, shared, controller, packet, graph
    graph.close()


def snapshots(shared, controller):
    return shared.prefill.snapshot(), [h.snapshot() for h in controller.histories['r']]


def test_graph_reservation_keeps_histories_unpublished_until_commit(case):
    runtime, shared, controller, packet, graph = case
    before = snapshots(shared, controller)
    prepared = runtime.prepare_accept(packet, 'r', 8, graph=graph, history_begin=0)
    assert snapshots(shared, controller) == before
    assert prepared.graph_update is graph._pending_update
    prepared.commit()
    assert shared.prefill.snapshot() == controller.histories['r'][0].snapshot()
    assert set(controller.histories['r'][1].scores()) == {0, 40}
    assert controller.histories['r'][1].scores() != controller.histories['r'][0].scores()
    after = snapshots(shared, controller)
    with pytest.raises(RuntimeError): prepared.commit()
    assert snapshots(shared, controller) == after


@pytest.mark.parametrize('fault', ['abort', 'capacity', 'invalid_layer', 'partial'])
def test_unaccepted_or_failed_graph_transaction_never_commits_layer_feedback(case, fault):
    runtime, shared, controller, packet, graph = case
    before = snapshots(shared, controller)
    accepted = 7 if fault == 'partial' else 8
    if fault == 'capacity': graph.max_bytes_per_session = graph.bytes
    if fault == 'invalid_layer': packet.action_inputs['r'].layerwise_pressure[3][1, 0, 0] = float('nan')
    if fault in ('capacity', 'invalid_layer'):
        with pytest.raises((RuntimeError, ValueError, BufferError)):
            runtime.prepare_accept(packet, 'r', accepted, graph=graph, history_begin=0)
    else:
        prepared = runtime.prepare_accept(packet, 'r', accepted, graph=graph, history_begin=0)
        if fault == 'abort': prepared.abort()
        else: prepared.commit()
    assert snapshots(shared, controller) == before

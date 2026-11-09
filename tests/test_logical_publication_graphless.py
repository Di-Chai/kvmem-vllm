"""prefill_physical_graphless: physical coalescing with graph_mode=off (for historical GC)."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.authority import append_authority_batch
from vllm_kvmem.logical_publication import NO_GRAPH, LogicalPublicationCursor
from test_logical_publication import advance, confirm, fixture


def state(env):
    return env.authority.history_end, env.state.prefill.snapshot()


@pytest.mark.parametrize('native_history', [False, True])
@pytest.mark.parametrize('retained', [0, 2049, 4096])
def test_graphless_children_equal_the_graph_off_accept_path(tmp_path, retained, native_history):
    with fixture(tmp_path/'new', graphless=True, native_history=native_history) as new, \
            fixture(tmp_path/'old', graphless=True, native_history=native_history) as old:
        assert new.cursor.graph is NO_GRAPH and new.authority.block_graph is None
        # Other tests may have installed the process-wide layerwise acceptance
        # hook on ActionRuntime; it reads runtime.runner (always set in serving).
        old.action.runner = getattr(old.action, 'runner', NS())
        confirm(new, retained)
        for index, (span, part, count) in enumerate(zip(old.group.spans, old.parts,
                                                        new.group.accepted_counts(retained))):
            # The uncoalesced graph-off path: plain accept, then the archive append.
            child = NS(action_inputs={'r': part.action_packet(old.state)})
            old.action.accept(child, 'r', count)
            assert append_authority_batch([(old.authority, old.raw[:, index*2048:(index+1)*2048],
                old.values[:, index*2048:(index+1)*2048], dict(history_begin=span.history_begin,
                local_rope_begin=span.local_begin, accepted=count))]) == 1
            receipt = advance(new)
            assert state(new) == state(old)
            assert (receipt.logical_index, receipt.accepted, receipt.owner_done) == (index, count, index == 1)
        assert new.cursor.done
        check_bytes(new)


def check_bytes(env):
    # A function scope: native read rights must be gone before teardown retires.
    for record in env.authority.records.values():
        raw, values = env.authority.read_record(record.handle)
        begin, end = record.history_begin, record.history_begin+record.count
        assert torch.equal(raw.view(torch.uint8), env.raw[:, begin:end].view(torch.uint8))
        assert torch.equal(values, env.values[:, begin:end])


def test_a_missing_graph_is_still_refused_without_the_opt_in(tmp_path):
    with fixture(tmp_path, graphless=True) as env:
        with pytest.raises(ValueError, match='graph transactions'):
            LogicalPublicationCursor(env.group, env.packet, env.authority, env.action, env.parts,
                                     owner_epoch=env.group.owner_epoch, source_generation=9)


def test_a_graph_installed_mid_publication_is_refused(tmp_path):
    with fixture(tmp_path, graphless=True) as env:
        confirm(env)
        env.authority.block_graph = object()
        with pytest.raises(RuntimeError, match='owner or physical source generation'):
            advance(env)
        env.authority.block_graph = None


def test_option_is_declared_and_defaults_off():
    import dataclasses
    from vllm_kvmem.options import RuntimeOptions
    field = {f.name: f for f in dataclasses.fields(RuntimeOptions)}['prefill_physical_graphless']
    assert field.default is False

"""Graph attention accepts exactly the same public consumed rows as legacy."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.action_attention import ActionHistory, ActionPacket, ActionRuntime
from vllm_kvmem.block_graph import BlockGraph


def options(**changes):
    values = dict(graph_mode='collect', action_scope='non-thinking', action_interval=1,
                  action_decay=.5, prefill_attention=True, native_prefill_commit=True,
                  prefill_attention_half_life=1024, prefill_attention_interval=4,
                  prefill_attention_stride=1)
    values.update(changes)
    return NS(**values)


def action(tmp_path):
    runtime = ActionRuntime.__new__(ActionRuntime)
    runtime.runtime = NS(output=tmp_path)
    return runtime


def history():
    result = ActionHistory(options(), [10, 11, 12, 13, 14])
    result.begin('first', [14])
    return result


def output_packet(state, tokens, begin, targets, masses):
    mask = state.parser.prepare(tokens)
    rows = [i for i in range(len(tokens)) if mask >> i & 1]
    pending = ActionPacket(state, rows, targets, torch.tensor(masses), None,
        source_positions=tuple(begin+i for i in rows), history_begin=begin,
        candidate_count=len(tokens))
    return NS(action_inputs={'r':pending})


def prefill_packet(state, begin, count, rows, targets, masses, *, starts=None):
    starts = list(range(0, len(targets)*4, 4)) if starts is None else starts
    pending = ActionPacket(state,rows,targets,torch.tensor(masses),None,
        (begin+count,count,starts,begin),tuple(begin+i for i in rows),begin,count)
    return NS(action_inputs={'r':pending})


def commit(runtime, graph, packet, begin, accepted):
    prepared = runtime.prepare_accept(packet,'r',accepted,graph=graph,history_begin=begin)
    prepared.commit()


def test_output_uses_consumed_positions_and_one_mask_across_responses(tmp_path):
    runtime, state = action(tmp_path), history()
    graph = BlockGraph(('tenant','session'),4,initial_position=4)
    # Row 3 is speculative and discarded. The accepted consumed row is 6,
    # not the predicted token at 7 (nor its dense active position).
    packet = output_packet(state,[12,13,20,21],4,[0,4],[[.25,.75],[1.,0.]])
    commit(runtime,graph,packet,4,3)
    assert state.samples == 1 and state.parser.state == (False,False,1)
    assert graph.accepted_through == 7 and graph.partial.samples == 1
    assert graph.rows == ()
    state.begin('next-request-after-tool',[14])
    commit(runtime,graph,output_packet(state,[22],7,[0,4],[[.75,.25]]),7,1)
    assert state.samples == 1  # Legacy response accumulator resets as before.
    assert graph.partial is None
    assert graph.rows[0].source == 4 and graph.rows[0].samples == 2
    assert graph.rows[0].output_samples == 2
    assert graph.rows[0].targets == (0,) and graph.rows[0].weights == (.5,)
    graph.close()


def test_graph_invalid_row_and_capacity_failure_leave_parser_and_legacy_untouched(tmp_path):
    runtime, state = action(tmp_path), history()
    graph = BlockGraph('s',4,initial_position=4)
    packet = output_packet(state,[20],4,[0,4],[[1.,-1.]])
    original = state.parser.state,dict(state.mass),state.samples,graph.binding
    with pytest.raises(RuntimeError,match='mass'):
        runtime.prepare_accept(packet,'r',1,graph=graph,history_begin=4)
    assert (state.parser.state,state.mass,state.samples,graph.binding) == original
    packet.action_inputs['r'].masses = torch.tensor([[.5,.5]])
    limit = graph.max_bytes_per_session
    graph.max_bytes_per_session = graph.bytes
    with pytest.raises(BufferError,match='capacity'):
        runtime.prepare_accept(packet,'r',1,graph=graph,history_begin=4)
    assert (state.parser.state,state.mass,state.samples,graph.binding) == original
    assert 'r' in packet.action_inputs
    graph.max_bytes_per_session = limit
    prepared = runtime.prepare_accept(packet,'r',1,graph=graph,history_begin=4)
    assert (state.parser.state,state.mass,state.samples,graph.binding) == original
    prepared.abort()
    assert graph.bytes == graph.live_bytes
    commit(runtime,graph,packet,4,1)
    assert state.samples == 1 and graph.partial.samples == 1
    graph.close()


def test_prefill_multiple_source_blocks_duplicate_segments_and_fullchunk_only(tmp_path):
    runtime, state = action(tmp_path), history()
    graph = BlockGraph('s',4)
    packet = prefill_packet(state,0,12,[3,7,11],[0,4,4,8],
        [[1.,0.,0.,0.],[.5,.25,.25,0.],[.25,.25,.25,.25]], starts=[0,4,6,8])
    commit(runtime,graph,packet,0,12)
    assert graph.accepted_through == 12 and graph.completed_through == 12
    assert [row.source for row in graph.rows] == [0,4,8]
    assert graph.rows[2].targets == (4,0)
    assert graph.rows[2].weights == (.5,.25)
    assert all(row.prefill_samples == 1 for row in graph.rows)
    prior = state.prefill.snapshot()
    packet = prefill_packet(state,12,4,[0,3],[0,12],[[.8,.2],[.9,.1]],starts=[0,12])
    commit(runtime,graph,packet,12,2)
    assert graph.accepted_through == 14 and graph.partial is None
    assert state.prefill.snapshot() == prior
    graph.close()


def test_prefill_graph_reservation_failure_does_not_publish_legacy(tmp_path):
    runtime, state = action(tmp_path), history()
    graph = BlockGraph('s',4)
    packet = prefill_packet(state,0,4,[3],[0],[[1.]])
    before = state.prefill.snapshot(),graph.binding
    graph.max_bytes_per_session = graph.bytes
    with pytest.raises(BufferError,match='capacity'):
        runtime.prepare_accept(packet,'r',4,graph=graph,history_begin=0)
    assert (state.prefill.snapshot(),graph.binding) == before
    assert 'r' in packet.action_inputs and graph.bytes == graph.live_bytes
    graph.close()


def test_no_packet_and_thinking_only_still_complete_source(tmp_path):
    runtime, state = action(tmp_path), history()
    graph = BlockGraph('s',4,initial_position=4)
    commit(runtime,graph,output_packet(state,[20],4,[0,4],[[.6,.4]]),4,1)
    assert graph.partial.samples == 1
    commit(runtime,graph,output_packet(state,[12,21],5,[],[]),5,2)
    assert graph.accepted_through == 7 and graph.partial.samples == 1
    commit(runtime,graph,NS(action_inputs={}),7,1)
    assert graph.accepted_through == 8 and graph.partial is None
    assert graph.rows[0].samples == 1 and graph.rows[0].targets == (0,)
    graph.close()


def test_repeated_prefix_new_request_and_public_query_replay_do_not_add_samples(tmp_path):
    runtime, state = action(tmp_path), history()
    graph = BlockGraph('s',4)
    commit(runtime,graph,prefill_packet(state,0,8,[3,7],[0,4],[[1.,0.],[.5,.5]]),0,8)
    saved = graph.rows,graph.binding
    state.begin('different-native-request',[14])
    commit(runtime,graph,prefill_packet(state,0,8,[3,7],[0,4],[[1.,0.],[.5,.5]]),0,8)
    assert (graph.rows,graph.binding) == saved
    state.prefill_replay = (8,12)
    commit(runtime,graph,prefill_packet(state,8,4,[3],[0,8],[[.8,.2]]),8,4)
    assert graph.accepted_through == 12 and graph.rows == saved[0]
    graph.close()


def test_wrong_predicted_source_position_rejected_before_parser_commit(tmp_path):
    runtime, state = action(tmp_path), history()
    graph = BlockGraph('s',4,initial_position=4)
    packet = output_packet(state,[20],4,[0,4],[[.5,.5]])
    packet.action_inputs['r'].source_positions = (5,)  # Predicted token, invalid.
    before = state.parser.state
    with pytest.raises(ValueError,match='consumed Q'):
        runtime.prepare_accept(packet,'r',1,graph=graph,history_begin=4)
    assert state.parser.state == before and graph.accepted_through == 4
    graph.close()


def test_collect_preserves_legacy_double_arithmetic_and_single_native_commit(tmp_path):
    runtime, state = action(tmp_path), history()
    baseline = ActionHistory(options(graph_mode='off'),[10,11,12,13,14])
    baseline.begin('first',[14])
    graph = BlockGraph('s',4,initial_position=4)
    tokens = [20,21,22]
    masses = torch.tensor([[.1,.9],[.2,.8],[.3,.7]],dtype=torch.float32)
    packet = output_packet(state,tokens,4,[0,4],masses.tolist())
    pending = packet.action_inputs['r']
    baseline.parser.prepare(tokens)
    baseline.accept(3,[0,1,2],[0,4],masses.tolist())
    commit(runtime,graph,packet,4,3)
    assert state.mass == baseline.mass and state.samples == baseline.samples
    assert all(type(value) is float for value in state.mass.values())
    assert state.parser.state == baseline.parser.state
    # An accidental second parser.commit would have returned an empty mask
    # and lost all graph samples on the first commit.
    assert graph.partial.samples == 3
    before = graph.binding,state.mass.copy(),state.samples
    commit(runtime,graph,NS(action_inputs={'r':pending}),4,3)
    assert (graph.binding,state.mass,state.samples) == before
    graph.close()


def test_graph_failure_precedes_query_and_parser_in_publication(tmp_path):
    from test_request_publication import fixture,packet,close
    runtime,a,b,archive = fixture(tmp_path)
    graph = a.block_graph = BlockGraph('a',4,initial_position=4)
    action_runtime = runtime.action = action(tmp_path)
    state = a.action_history = history()
    shared = packet(runtime,[('a',4,1)])
    shared.action_inputs = output_packet(state,[20],4,[0,4],[[.5,.5]]).action_inputs
    shared.action_inputs['a'] = shared.action_inputs.pop('r')
    query_calls = []
    runtime._accept_query_input = lambda *args: query_calls.append(args)
    graph.max_bytes_per_session = graph.bytes
    before = state.parser.state,state.mass.copy(),graph.binding
    with pytest.raises(BufferError,match='capacity'):
        runtime._publish_claim(shared,[0],False)
    assert not query_calls
    assert (state.parser.state,state.mass,graph.binding) == before
    assert 'a' in shared.action_inputs and not shared.published
    # Capacity preflight now precedes raw archive and semantic acceptance.
    assert runtime.accepted['a'] == a.history_end == 4
    runtime.queue.clear();runtime.downloads.pending.clear()
    graph.close()
    close(runtime,(a,b),archive)


def test_publication_without_action_runtime_advances_only_retained_prefix(tmp_path):
    from test_request_publication import fixture,packet,close
    from vllm_kvmem.terminal import TerminalPrefix
    from test_terminal_prefix import request
    runtime,a,b,archive = fixture(tmp_path)
    a.block_graph = BlockGraph('a',4,initial_position=4)
    b.block_graph = BlockGraph('b',4,initial_position=4)
    shared = packet(runtime,[('a',4,8),('b',4,4)],rejected=(2,0))
    runtime.terminal_prefixes['a'] = TerminalPrefix.from_request(request(10,0,8,4))
    assert runtime.drain_requests({'a'},wait=True)
    assert a.block_graph.accepted_through == 7  # 2 rejected + 3 after EOS.
    assert b.block_graph.accepted_through == 4
    assert runtime.drain_requests({'b'},wait=True)
    assert b.block_graph.accepted_through == 8
    assert shared.published == {0,1}
    a.block_graph.close();b.block_graph.close()
    close(runtime,(a,b),archive)

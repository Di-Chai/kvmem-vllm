"""Bounded N-child score/publication semantics with the real CPU consumers."""
import json
import weakref
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.action_attention import ActionPacket
from vllm_kvmem.logical_action import project_logical_scores
from vllm_kvmem.terminal import TerminalPrefix
from test_logical_action import fixture as score_fixture, state, consumer, publish
from test_logical_action_runtime import runtime as action_fixture
from test_logical_capture_publication import runtime_fixture
from test_logical_publication import fixture as cursor_fixture, confirm, advance, snapshot


BOUNDARIES = [(n, accepted) for n in (3,4,7,8)
    for accepted in sorted({max(0,min(n*2048,i*2048+d))
                            for i in range(n+1) for d in (-1,0,1)})]


@pytest.mark.parametrize('chunks,retained', BOUNDARIES)
def test_each_boundary_preserves_original_scores_exposure_decay_and_graph(tmp_path,chunks,retained):
    group,plans,rows,masses = score_fixture(chunks=chunks)
    parts = project_logical_scores(group,plans,masses,physical_rows=rows)
    old,new = state(True),state(True)
    old_action,old_graph = consumer(tmp_path)
    new_action,new_graph = consumer(tmp_path)
    try:
        for index,(plan,part) in enumerate(zip(plans,parts)):
            span=plan.span
            original=ActionPacket(old,list(plan.rows),list(plan.history),
                masses[index*64:(index+1)*64,:len(plan.history)].copy(),None,
                (span.history_end,span.count,plan.starts,span.local_begin),
                tuple(span.history_begin+r for r in plan.rows),span.history_begin,span.count)
            accepted=max(0,min(2048,retained-index*2048))
            assert publish(old_action,old_graph,original,accepted)==publish(
                new_action,new_graph,part.action_packet(new),accepted)
        assert new.prefill.samples==64+64*(retained//2048)
        assert new.prefill.chunks==4+retained//2048
        assert new_graph.accepted_through==8192+retained
    finally:
        old_graph.close();new_graph.close()


@pytest.mark.parametrize('chunks,retained', BOUNDARIES)
def test_runtime_all_acceptance_boundaries_keep_one_physical_receipt(tmp_path,chunks,retained):
    with runtime_fixture(tmp_path,chunks=chunks) as env:
        runtime=env.runtime
        runtime.begin_step({'r'},{'r':TerminalPrefix(retained,env.group.count,0,'FINISHED_ABORTED')})
        assert runtime.drain_requests({'r'},wait=True)
        assert env.authority.history_end==env.graph.accepted_through==retained
        assert runtime.accepted['r']==retained and runtime.terminal_discarded['r']==env.group.count-retained
        assert runtime.batched_append_calls==chunks and env.state.prefill.samples==64*(retained//2048)
        assert runtime.last_steps['r'][-1]['physical_step']==dict(history_begin=0,
            scheduled=env.group.count,accepted=retained,rejected=0,terminal_discarded=env.group.count-retained)
        assert sum('physical_step' in row for row in runtime.last_steps['r'])==1
        assert len(runtime.released)==1 and not runtime.queue and not runtime.downloads.pending
        row=json.loads((tmp_path/'logical_prefill_publication.jsonl').read_text())
        assert len(row['spans'])==chunks and row['retained']==retained


@pytest.mark.parametrize('chunks',[3,4,7,8])
@pytest.mark.parametrize('retry',[False,True])
def test_actual_capture_all_layers_keep_rows_columns_and_original_fallback_widths(monkeypatch,chunks,retry):
    action,batch,captured,current,calls=action_fixture(monkeypatch,chunks=chunks,retry=retry,pooled_metadata=True)
    before=current.prefill.snapshot()
    action.prepare(batch,captured[0].positions)
    result=action.capture(captured,batch)
    parts=result.logical_slices['r']
    assert len(parts)==chunks and [p.masses.shape for p in parts]==[(64,2+64*(i+1)) for i in range(chunks)]
    assert all(p.source_owner is result['r'].masses for p in parts)
    assert current.prefill.snapshot()==before
    assert action.logical_prefill_stats['planned_logical_chunks']==chunks
    assert action.logical_prefill_stats['projected_logical_chunks']==chunks
    assert action.logical_prefill_stats['sampled_rows']==chunks*64
    assert calls.count('reference score')==(chunks*3 if retry else 0)
    assert action.download_pool.snapshot()['exhaustions']==0
    if retry: assert action.download_pool.snapshot()['reuses']==1


@pytest.mark.parametrize('chunks',[3,4,7,8])
@pytest.mark.parametrize('limit',['config','native_rows','bytes','metadata'])
def test_admission_checks_config_and_capacity_before_arm_or_state_change(monkeypatch,chunks,limit):
    action,batch,captured,current,calls=action_fixture(monkeypatch,chunks=chunks)
    if limit=='config':action.options.prefill_physical_max_chunks=chunks-1
    elif limit=='native_rows':action.native.max_rows=chunks*64-1
    elif limit=='bytes':action.download_pool.max_bytes=1
    else:action.metadata_upload=NS(capacity=1)
    before=current.prefill.snapshot()
    with pytest.raises(RuntimeError):action.prepare(batch,captured[0].positions)
    assert 'arm' not in calls and current.prefill.snapshot()==before
    assert not getattr(action,'prefill_plan_frames',{})


@pytest.mark.parametrize('chunks',[3,4,7,8])
def test_late_cancel_after_capacity_retry_keeps_committed_children_and_clips_suffix(tmp_path,monkeypatch,chunks):
    with runtime_fixture(tmp_path,chunks=chunks) as env:
        runtime=env.runtime;stop=(chunks-1)*2048;original=env.authority._prepare_append;blocked=[True]
        def prepare(*args,**kwargs):
            return None if kwargs['history_begin']==stop and blocked[0] else original(*args,**kwargs)
        monkeypatch.setattr(env.authority,'_prepare_append',prepare)
        runtime.begin_step();assert not runtime.drain_requests({'r'},wait=False)
        assert runtime.accepted['r']==stop and env.packet.logical_cursor.next_index==chunks-1
        assert not runtime.released and env.packet.download.owners
        runtime.runner._kvmem_logical_prefill_step=9
        runtime.begin_step({'r'},{'r':TerminalPrefix(stop+1,env.group.count,0,'FINISHED_ABORTED')})
        blocked[0]=False
        assert runtime.drain_requests({'r'},wait=True)
        assert runtime.accepted['r']==stop+1 and runtime.batched_append_calls==chunks
        assert env.state.prefill.samples==64*(chunks-1)
        assert runtime.last_steps['r'][-1]['physical_step']['accepted']==stop+1
        assert len(runtime.released)==1


@pytest.mark.parametrize('chunks',[3,4,7,8])
@pytest.mark.parametrize('failure',['graph_budget','after_append'])
def test_late_child_failure_poison_retains_prefix_and_whole_source(tmp_path,monkeypatch,chunks,failure):
    with runtime_fixture(tmp_path,chunks=chunks) as env:
        runtime=env.runtime;stop=(chunks-1)*2048
        if failure=='graph_budget':
            original=env.action.prepare_accept
            def prepare(*args,**kwargs):
                if kwargs['history_begin']==stop:env.graph.max_bytes_per_session=env.graph.bytes
                return original(*args,**kwargs)
            monkeypatch.setattr(env.action,'prepare_accept',prepare)
            error=BufferError
        else:
            original=env.authority._commit_prepared_append
            def commit(prepared):
                last=env.authority.history_end==stop
                original(prepared)
                if last:raise OSError('last append committed before failure')
            monkeypatch.setattr(env.authority,'_commit_prepared_append',commit)
            error=OSError
        runtime.begin_step()
        with pytest.raises(error) as caught:runtime.drain_requests({'r'},wait=True)
        assert runtime._publication_error is caught.value and env.packet.logical_cursor.error is caught.value
        assert runtime.accepted['r']==stop and env.packet.logical_cursor.next_index==chunks-1
        assert env.authority.history_end==(stop if failure=='graph_budget' else env.group.count)
        assert not runtime.released and env.packet.download.owners and env.packet.logical_payload is not None
        before=snapshot(env)
        with pytest.raises(error) as repeated:runtime.drain_requests({'r'},wait=True)
        assert repeated.value is caught.value and snapshot(env)==before
        assert env.graph._pending_update is None


@pytest.mark.parametrize('chunks',[3,4,7,8])
def test_dynamic_confirmation_and_explicit_lease_release_after_last_ack(tmp_path,chunks):
    with cursor_fixture(tmp_path,chunks=chunks) as env:
        confirm(env)  # No hardcoded4096 accepted-prefix default.
        owner=weakref.ref(env.packet.download.owners[0]);del env.packet
        for index in range(chunks):
            receipt=env.cursor.advance(env.raw,env.values,owner_epoch=env.group.owner_epoch)
            assert receipt.logical_index==index and receipt.owner_done==(index==chunks-1)
            assert owner() is not None
            with pytest.raises(RuntimeError,match='unpublished'):env.cursor.release()
            env.cursor.ack(receipt)
        env.cursor.release()
        assert owner() is None


@pytest.mark.parametrize('chunks',[3,4,7,8])
def test_every_score_slice_and_exported_numpy_view_retain_the_single_pool_lease(chunks):
    from test_action_download_pool import Backend
    from vllm_kvmem.action_download_pool import ActionDownloadPool
    group,plans,rows,source=score_fixture(chunks=chunks)
    # Exactly one source allocation; never enlarge slot count or rely on GC.
    pool=ActionDownloadPool(max(1<<20,1 << (source.nbytes-1).bit_length()),1,backend=Backend())
    host,event=pool.copy(torch.from_numpy(source));event.synchronize();pointer=host.data_ptr()
    parts=project_logical_scores(group,plans,host,physical_rows=rows,event=event)
    children=[p.action_packet(state(True)) for p in parts];exported=children[-1].masses.numpy()
    del parts,host,event
    while children:
        del children[0]
        with pytest.raises(BufferError):pool.copy(torch.from_numpy(source))
    del exported
    host,event=pool.copy(torch.from_numpy(source))
    assert host.data_ptr()==pointer and pool.snapshot()['reuses']==1


@pytest.mark.parametrize('chunks',[3,4,7,8])
def test_actual_group_can_be_shorter_than_configured_limit(monkeypatch,chunks):
    action,batch,captured,current,_=action_fixture(monkeypatch,chunks=chunks)
    action.options.prefill_physical_max_chunks=8
    action.prepare(batch,captured[0].positions)
    result=action.capture(captured,batch)
    assert len(result.logical_slices['r'])==chunks


@pytest.mark.parametrize('chunks,retained',BOUNDARIES)
def test_physical_extent_consumers_keep_full_group_and_partial_checkpoint_rule(monkeypatch,chunks,retained):
    from test_native_publication_step import row
    from test_session_state import fixture
    from vllm_kvmem.publication_step import native_publication_step
    from vllm_kvmem.native_state import canonicalize_before_non_speculative_batch
    logical=row(retained,chunks=chunks)
    physical=native_publication_step(logical)
    assert physical['scheduled']==chunks*2048 and physical['accepted']==retained
    adapter,conv,ssm,_,_=fixture(draft=False)
    runner=adapter.runner
    adapter.recurrent.conv_dim_first=False
    adapter.recurrent.conv_widths={'gdn':4}
    runner.speculative_config=object()
    runner.model_state.num_accepted_tokens_gpu[0]=3
    runner._kvmem_capture_runtime=NS(drain_requests=lambda *_a,**_kw:True,last_steps={'a':[logical]})
    monkeypatch.setattr('vllm_kvmem.native_state.NativeRecurrentState',lambda _:adapter.recurrent)
    batch=NS(num_reqs=1,req_ids=['a'],num_draft_tokens_per_req=[0],is_prefilling_np=[False],
             num_computed_tokens_np=[8192+retained])
    before_conv,before_ssm=conv.clone(),ssm.clone()
    if retained != chunks*2048:
        with pytest.raises(ValueError,match='checkpoint replay'):
            canonicalize_before_non_speculative_batch(runner,batch)
        assert runner.model_state.num_accepted_tokens_gpu[0]==3
    else:
        canonicalize_before_non_speculative_batch(runner,batch)
        assert runner.model_state.num_accepted_tokens_gpu[0]==1
    assert torch.equal(conv,before_conv) and torch.equal(ssm,before_ssm)

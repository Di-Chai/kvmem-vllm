"""Actual CaptureRuntime FIFO/confirmation integration with CPU archive graphs."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace as NS
import json

import pytest
import torch

from vllm_kvmem.capture_runtime import CaptureRuntime
from vllm_kvmem.terminal import TerminalPrefix
from test_logical_publication import fixture, snapshot


@contextmanager
def runtime_fixture(tmp_path, *, chunks=2):
    with fixture(tmp_path, chunks=chunks) as env:
        runtime = CaptureRuntime.__new__(CaptureRuntime)
        runtime.raw_dtype = env.raw.dtype
        runtime.options = NS(async_cpu_scatter=False,batched_cpu_append=True,
            early_capture_slot_release=False,prefill_publication_pipeline=False,
            prefill_physical_coalescing=True,prefill_physical_max_chunks=chunks)
        runtime.runner = NS(vllm_config=NS(max_concurrent_batches=4),_kvmem_logical_prefill_step=8)
        runtime.output = tmp_path
        runtime.config,runtime.archive = env.authority.memory.config,env.archive
        runtime.layers,runtime.heads,runtime.dim = [None],1,2
        runtime.row_elements,runtime.resident_records = 2,256
        runtime.authorities = {'r':env.authority}
        runtime.accepted,runtime.rejected = {'r':0},{'r':0}
        runtime.last_steps = {'r':[]}
        runtime.terminal_prefixes,runtime.terminal_discarded = {},{}
        runtime.batches,runtime._progress_batch = 0,0
        runtime.completed,runtime.retiring = [],{}
        runtime._published_capacity = {}
        runtime.action = env.action
        env.action._prefill_identity = lambda rid,state: env.group.owner_epoch
        runtime.queue = [env.packet]
        header = torch.tensor([0,env.group.count,0,0,0],dtype=torch.int64)
        data = torch.cat((header.view(torch.uint8),env.raw.view(torch.uint8).flatten(),env.values.flatten()))
        env.packet.download = NS(payload=data,slot=0,query=lambda:True,owners=(object(),),
            event=NS(synchronize=lambda:pytest.fail('Ready CPU source waited for CUDA')))
        env.packet.logical_group,env.packet.logical_slices = env.group,env.parts
        runtime.downloads = NS(buffers=[object(),object()],pending={0:env.packet.download})
        runtime.released = []
        def release(download):
            assert env.packet.logical_cursor.done
            assert env.packet.logical_cursor.phase == 'released'
            assert env.packet.logical_payload is None and not env.packet.logical_slices
            assert env.packet.logical_cursor.slices is None
            del runtime.downloads.pending[download.slot]
            runtime.released.append(download)
            download.owners = ()
            return True
        runtime.downloads.release = release
        runtime._snapshot_capacity('r',env.authority)
        env.runtime = runtime
        yield env


@pytest.mark.parametrize('retained',[0,1,2048,2049,4095,4096])
def test_actual_runtime_preserves_logical_stats_and_releases_only_complete_owner(tmp_path,retained):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        assert not runtime.drain_requests({'r'},wait=True)
        assert env.authority.history_end == 0 and not runtime.released
        if retained == 4096:
            runtime.begin_step()
        else:
            runtime.begin_step({'r'},{'r':TerminalPrefix(retained,4096,0,'FINISHED_ABORTED')})
        assert runtime.drain_requests({'r'},wait=True)
        assert env.authority.history_end == retained and env.graph.accepted_through == retained
        assert runtime.accepted == {'r':retained} and runtime.rejected == {'r':0}
        assert runtime.terminal_discarded == {'r':4096-retained}
        assert [(row['history_begin'],row['scheduled'],row['accepted']) for row in runtime.last_steps['r']] == [
            (0,2048,min(retained,2048)),(2048,2048,max(0,retained-2048))]
        assert 'physical_step' not in runtime.last_steps['r'][0]
        assert runtime.last_steps['r'][-1]['physical_step'] == dict(history_begin=0,
            scheduled=4096,accepted=retained,rejected=0,terminal_discarded=4096-retained)
        assert runtime.batched_append_calls == runtime.batched_append_requests == 2
        metrics = next(iter(runtime._publication_metrics.values()))
        assert metrics['logical_prefill_commits'] == 2
        assert metrics['logical_prefill_physical_owners'] == 1
        assert runtime._published_capacity['r'].history_end == retained
        assert runtime.publication_ready({'r'}) and not runtime.queue
        assert runtime.released == [env.packet.download] and not runtime.downloads.pending
        records = [json.loads(line) for line in (tmp_path/'logical_prefill_publication.jsonl').read_text().splitlines()]
        assert records == [dict(request_id='r',scheduler_step=env.group.scheduler_step,
            confirmation_step=8,history_begin=0,local_begin=0,count=4096,retained=retained,
            source_generation=env.packet.source_generation,spans=[dict(
                history_begin=part.plan.span.history_begin,local_begin=part.plan.span.local_begin,
                count=2048,sampled_rows=len(part.plan.rows),segments=len(part.plan.starts),
                sampled_row_indices=list(part.plan.rows)) for part in env.parts])]
        assert runtime.drain_requests({'r'},wait=True)
        assert len((tmp_path/'logical_prefill_publication.jsonl').read_text().splitlines()) == 1


def test_second_child_capacity_retry_keeps_first_counts_and_same_physical_lease(tmp_path,monkeypatch):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        original = env.authority._prepare_append
        blocked = [True]
        def prepare(*args,**kwargs):
            if kwargs['history_begin'] == 2048 and blocked[0]:
                return None
            return original(*args,**kwargs)
        monkeypatch.setattr(env.authority,'_prepare_append',prepare)
        runtime.begin_step()
        assert not runtime.drain_requests({'r'},wait=False)
        cursor = env.packet.logical_cursor
        raw,values = env.packet.logical_payload
        assert cursor.next_index == 1 and not cursor.done
        assert env.authority.history_end == runtime.accepted['r'] == 2048
        assert env.graph._pending_update is None and len(runtime.last_steps['r']) == 1
        assert env.packet.download.owners and not runtime.released
        assert not (tmp_path/'logical_prefill_publication.jsonl').exists()
        assert not env.packet.published and not env.packet.consuming
        blocked[0] = False
        assert runtime.drain_requests({'r'},wait=False)
        assert runtime.accepted['r'] == 4096 and len(runtime.last_steps['r']) == 2
        assert cursor.next_index == 2 and cursor.done and env.packet.logical_payload is None
        assert len((tmp_path/'logical_prefill_publication.jsonl').read_text().splitlines()) == 1


def test_final_audit_write_failure_quarantines_completed_payload_before_ack(tmp_path):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        (tmp_path/'logical_prefill_publication.jsonl').mkdir()
        runtime.begin_step()
        with pytest.raises(IsADirectoryError) as caught:
            runtime.drain_requests({'r'},wait=True)
        assert runtime._publication_error is caught.value
        cursor = env.packet.logical_cursor
        assert cursor.phase == 'failed' and cursor.error is caught.value
        assert cursor.next_index == 2 and not cursor.done
        assert env.authority.history_end == runtime.accepted['r'] == 4096
        assert not runtime.released and not env.packet.published
        assert env.packet.logical_payload is not None and env.packet.download.owners
        with pytest.raises(IsADirectoryError) as repeated:
            runtime.drain_requests({'r'},wait=True)
        assert repeated.value is caught.value and len(runtime.last_steps['r']) == 2


def test_waiting_retry_only_waits_for_this_owner_then_advances_second_child(tmp_path,monkeypatch):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        original = env.authority._prepare_append
        blocked,waits = [True],[]
        def prepare(*args,**kwargs):
            return None if kwargs['history_begin']==2048 and blocked[0] else original(*args,**kwargs)
        def progress(ids):
            waits.append(ids)
            blocked[0] = False
        monkeypatch.setattr(env.authority,'_prepare_append',prepare)
        runtime._wait_archive_progress = progress
        runtime.begin_step()
        assert runtime.drain_requests({'r'},wait=True)
        assert waits == [{'r'}] and runtime.batched_append_calls == 2


def test_second_graph_failure_quarantines_slot_and_never_replays_first_child(tmp_path,monkeypatch):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        original = env.action.prepare_accept
        error = BufferError('second graph budget')
        def prepare(*args,**kwargs):
            if kwargs['history_begin'] == 2048:
                raise error
            return original(*args,**kwargs)
        monkeypatch.setattr(env.action,'prepare_accept',prepare)
        runtime.begin_step()
        with pytest.raises(BufferError) as caught:
            runtime.drain_requests({'r'},wait=True)
        assert caught.value is runtime._publication_error is error
        assert env.packet.logical_cursor.error is error
        assert env.authority.history_end == runtime.accepted['r'] == 2048
        assert len(runtime.last_steps['r']) == 1 and not runtime.released
        assert env.packet.download.owners and env.packet.logical_payload is not None
        before = snapshot(env)
        with pytest.raises(BufferError) as repeated:
            runtime.drain_requests({'r'},wait=True)
        assert repeated.value is error and snapshot(env) == before


def test_later_terminal_receipt_trims_only_unpublished_second_child(tmp_path,monkeypatch):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        original = env.authority._prepare_append
        blocked = [True]
        def prepare(*args,**kwargs):
            return None if kwargs['history_begin']==2048 and blocked[0] else original(*args,**kwargs)
        monkeypatch.setattr(env.authority,'_prepare_append',prepare)
        runtime.begin_step()
        assert not runtime.drain_requests({'r'},wait=False)
        runtime.runner._kvmem_logical_prefill_step = 9
        runtime.begin_step({'r'},{'r':TerminalPrefix(2049,4096,0,'FINISHED_ABORTED')})
        blocked[0] = False
        assert runtime.drain_requests({'r'},wait=True)
        assert runtime.accepted['r'] == 2049 and runtime.terminal_discarded['r'] == 2047
        assert [row['accepted'] for row in runtime.last_steps['r']] == [2048,1]


@pytest.mark.parametrize('kind',['header','rejection','query','generation','missing_receipt','stale_step'])
def test_invalid_physical_metadata_never_commits(tmp_path,kind):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        if kind == 'stale_step':
            runtime.runner._kvmem_logical_prefill_step = 7
            with pytest.raises(RuntimeError,match='follow'):
                runtime.begin_step()
            assert not env.packet.confirmed
            return
        runtime.begin_step()
        if kind == 'header': env.packet.download.payload[:40].view(torch.int64)[2] = 32
        elif kind == 'rejection': env.packet.download.payload[:40].view(torch.int64)[4] = 1
        elif kind == 'query': env.packet.query_inputs['r'] = object()
        elif kind == 'generation': env.action._prefill_identity = lambda *args: ('stale',)
        elif kind == 'missing_receipt': env.packet.logical_confirmation = None
        with pytest.raises((ValueError,RuntimeError)):
            runtime.drain_requests({'r'},wait=True)
        assert env.authority.history_end == 0 and runtime.accepted['r'] == 0
        assert not runtime.released and env.packet.download.owners


def test_bookkeeping_failure_after_child_commit_is_fatal_and_preserves_payload(tmp_path):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        runtime.begin_step()
        def fail(*args): raise OSError('capacity snapshot failure')
        runtime._snapshot_capacity = fail
        with pytest.raises(OSError,match='capacity snapshot'):
            runtime.drain_requests({'r'},wait=True)
        assert env.authority.history_end == 2048 and env.packet.logical_cursor.next_index == 1
        assert env.packet.logical_cursor.phase == 'failed'
        assert not runtime.released and env.packet.logical_payload is not None


@pytest.mark.parametrize('raises',[False,True])
def test_source_release_failure_cannot_publish_owner_or_ack_ready(tmp_path,raises):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        runtime.begin_step()
        def fail(download):
            if raises:
                raise OSError('pool release failed')
            return False
        runtime.downloads.release = fail
        with pytest.raises((OSError,RuntimeError)):
            runtime.drain_requests({'r'},wait=True)
        assert runtime._publication_error is not None
        assert not env.packet.published and env.packet in runtime.queue
        assert env.packet.download.owners and not runtime.released
        with pytest.raises((OSError,RuntimeError)):
            runtime.publication_ready({'r'})


def test_terminal_arriving_inside_final_native_append_is_checked_before_ack(tmp_path,monkeypatch):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        runtime.begin_step()
        original = env.authority._commit_prepared_append
        def commit(prepared):
            if env.authority.history_end == 2048:
                # Simulate begin_step while the second native append has not
                # yet made its new history prefix visible to _install_terminals.
                runtime.runner._kvmem_logical_prefill_step = 9
                runtime.begin_step({'r'},{'r':TerminalPrefix(2049,4096,0,'FINISHED_ABORTED')})
            return original(prepared)
        monkeypatch.setattr(env.authority,'_commit_prepared_append',commit)
        with pytest.raises(ValueError,match='retract'):
            runtime.drain_requests({'r'},wait=True)
        assert env.authority.history_end == 4096  # Irreversible work is fail-stop.
        assert runtime.accepted['r'] == 2048 and len(runtime.last_steps['r']) == 1
        assert env.packet.logical_cursor.phase == 'failed' and not env.packet.published
        assert not runtime.released and env.packet.download.owners


def test_explicit_capture_metadata_checks_plan_and_preserves_plain_4k_reference(tmp_path):
    with runtime_fixture(tmp_path) as env:
        runtime = env.runtime
        batch = NS(num_reqs=1,req_ids=('r',),is_prefilling_np=(True,),
                   num_computed_tokens_np=(0,),query_start_loc_np=(0,4096))
        assert runtime._logical_capture_inputs({},batch) == (None,())
        result = NS(logical_groups={'r':env.group},logical_slices={'r':env.parts})
        runtime.runner._kvmem_logical_prefill_step = 7
        assert runtime._logical_capture_inputs(result,batch) == (env.group,env.parts)
        runtime.runner._kvmem_logical_prefill_step = 8
        with pytest.raises(RuntimeError,match='immutable'):
            runtime._logical_capture_inputs(result,batch)
        runtime.runner._kvmem_logical_prefill_step = 7
        runtime.runner._kvmem_sessions = NS(bindings={'r':NS(tokens=range(16384),query_span=(2000,2100))})
        with pytest.raises(RuntimeError,match='Query seed'):
            runtime._logical_capture_inputs(result,batch)


@pytest.mark.parametrize('kind',['synchronize','release'])
def test_invalid_logical_plan_and_private_dma_cleanup_failure_quarantines_runtime(tmp_path,monkeypatch,kind):
    from test_request_publication import fixture as capture_fixture,close
    from test_capture_dma_isolation import configure
    from test_prefill_capture_ahead import inputs,prepare
    runtime,a,b,archive = capture_fixture(tmp_path)
    configure(runtime)
    captures,batch,rejected = inputs(runtime,[('a',4,4,True)])
    prepare(runtime,batch,[])
    failure = ValueError('invalid logical capture plan')
    def invalid(*args):
        raise failure
    runtime._logical_capture_inputs = invalid
    original = runtime.downloads.download
    def download(payload):
        result = original(payload)
        def synchronize():
            if kind == 'synchronize':
                raise OSError('own private DMA failed')
        result.event.synchronize = synchronize
        return result
    runtime.downloads.download = download
    if kind == 'release':
        runtime.downloads.release = lambda download: False
    try:
        with pytest.raises(ValueError) as caught:
            runtime.submit(captures,batch,rejected)
        assert caught.value is failure and runtime._publication_error is failure
        assert any('cleanup also failed' in note for note in failure.__notes__)
        assert len(runtime.downloads.pending) == 1 and not runtime.queue
        assert (a.history_end,b.history_end) == (4,4)
        with pytest.raises(ValueError):
            runtime.publication_ready({'a'})
    finally:
        close(runtime,(a,b),archive)

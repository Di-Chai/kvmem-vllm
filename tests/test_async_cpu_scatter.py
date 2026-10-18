from threading import Event, current_thread
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.capture_runtime import CaptureRuntime, CapturedPacket
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.terminal import TerminalPrefix
from test_authority import setup,payload,drain
from test_terminal_prefix import request


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('later_packet', [False, True])
def test_background_scatter_finishes_authorized_packet_before_terminal_clip(tmp_path,monkeypatch, batched, later_packet):
    a,archive=setup(tmp_path/'a')
    b,_=setup(tmp_path/'b','b',archive)
    for owner in (a,b):
        assert owner.append(*payload(0,4),history_begin=0,local_rope_begin=0)
        drain(owner)
    runtime=CaptureRuntime.__new__(CaptureRuntime)
    runtime.raw_dtype=torch.bfloat16  # Explicit legacy packet fixture.
    runtime.options=RuntimeOptions(memory=a.memory.config,async_cpu_scatter=True,cpu_scatter_min_tokens=1,batched_cpu_append=batched)
    runtime.config,runtime.archive=a.memory.config,archive
    runtime.runner=NS(vllm_config=NS(max_concurrent_batches=1))
    runtime.layers,runtime.heads,runtime.dim=[None,None],1,2
    runtime.row_elements,runtime.resident_records=4,4
    runtime.authorities={'a':a,'b':b}
    runtime.accepted,runtime.rejected={'a':4,'b':4},{'a':0,'b':0}
    runtime.last_steps={'a':[],'b':[]}
    runtime.terminal_prefixes,runtime.terminal_discarded={},{}
    runtime.batches,runtime._progress_batch,runtime.completed=2,0,[]
    runtime._write_archive_report=lambda:None
    released=[]
    runtime.downloads=NS(release=lambda packet:released.append(packet) or True)
    def packet(begin):
        raw=torch.cat((payload(begin,4)[0],payload(begin+100,4)[0]),dim=1)
        header=torch.tensor([0,4,8,begin,begin,begin,begin,0,0],dtype=torch.int64)
        data=torch.cat((header.view(torch.uint8),raw.contiguous().view(torch.uint8).flatten(),raw.to(torch.uint8).flatten()))
        download=NS(payload=data,query=lambda:True)
        return CapturedPacket(('a','b'),8,download)
    initial=packet(4)
    runtime.queue=[initial]
    started,proceed=Event(),Event()
    append=a.memory.begin_append
    threads=[]
    def delayed(*args,**kwargs):
        spans=append(*args,**kwargs)
        threads.append(current_thread().name)
        started.set()
        assert proceed.wait(5)
        return spans
    monkeypatch.setattr(a.memory,'begin_append',delayed)
    try:
        assert runtime.start_cpu_scatter() and started.wait(5)
        assert a.history_end==4 and not released
        entered=Event()
        def foreground():
            entered.set()
            return runtime.drain(wait=True)
        with ThreadPoolExecutor(max_workers=1) as reader:
            pending=reader.submit(foreground)
            assert entered.wait(5) and not pending.done()
            proceed.set()
            assert pending.result(5)
        # Capacity feedback sees the fully published prefix after the fence.
        assert runtime.archive_capacity()['a'].history_end==8
        assert b.history_end==8 and not runtime.queue and released==[initial.download]
        assert all(name.startswith('kvmem-scatter') for name in threads)
        latest=packet(8)
        runtime.queue.append(latest)
        if later_packet:
            runtime.queue.append(packet(12))
        runtime.finish({'a'},{'a':TerminalPrefix.from_request(request(12,0,10,6))})
        assert runtime.completed[0]['history_end']==9
        assert runtime.completed[0]['terminal_discarded_tokens']==(7 if later_packet else 3)
        assert runtime.authorities=={'b':b} and b.history_end==8
        assert runtime.publication_ready({'a'}) and not runtime.publication_ready({'b'})
        assert released==[initial.download]  # The shared slot still owns b.
        assert runtime.drain_requests({'b'},wait=True)
        assert b.history_end==(16 if later_packet else 12)
        assert runtime.cpu_scatter_batches==1 and len(released)==(3 if later_packet else 2)
        drain(b)
        assert b.close() and archive.close()
    finally:
        proceed.set()
        executor=getattr(runtime,'_cpu_scatter_executor',None)
        if executor is not None:executor.shutdown(wait=True)


def test_worker_failure_is_fatal_to_every_publication_join():
    runtime=CaptureRuntime.__new__(CaptureRuntime)
    runtime.options=RuntimeOptions(async_cpu_scatter=True,cpu_scatter_min_tokens=1)
    runtime.queue=[NS(tokens=1)]
    def fail(**kwargs):raise OSError('scatter failure')
    runtime.drain=fail
    try:
        assert runtime.start_cpu_scatter()
        for _ in range(2):
            with pytest.raises(OSError,match='scatter failure'):runtime.join_cpu_scatter()
        assert runtime._cpu_scatter_future is not None
    finally:
        runtime._cpu_scatter_executor.shutdown(wait=True)


def test_small_decode_packet_is_deferred_without_releasing_dma_or_launching_a_worker():
    runtime=CaptureRuntime.__new__(CaptureRuntime)
    runtime.options=RuntimeOptions(async_cpu_scatter=True,cpu_scatter_min_tokens=128)
    packet=NS(tokens=32)
    runtime.queue=[packet]
    assert not runtime.start_cpu_scatter()
    assert runtime.queue==[packet] and not hasattr(runtime,'_cpu_scatter_executor')


def test_ready_large_packet_reaches_worker_instead_of_foreground_drain():
    runtime=CaptureRuntime.__new__(CaptureRuntime)
    runtime.options=RuntimeOptions(async_cpu_scatter=True,cpu_scatter_min_tokens=128)
    runtime.queue=[NS(tokens=2048)]
    entered, release=Event(), Event()
    threads=[]
    def drain(**kwargs):
        threads.append(current_thread().name)
        entered.set()
        assert release.wait(5)
        runtime.queue.clear()
    runtime.drain=drain
    try:
        runtime.advance_publication({'decode-peer':5}, NS(jobs={}))
        assert entered.wait(5)
        # The caller has already returned while the CPU archive worker waits.
        assert not runtime._cpu_scatter_future.done()
        assert all(name.startswith('kvmem-scatter') for name in threads)
    finally:
        release.set()
        runtime.join_cpu_scatter()
        runtime._cpu_scatter_executor.shutdown(wait=True)


@pytest.mark.parametrize('tokens', [5, 2048])
def test_zero_token_iteration_publishes_packet_without_later_submit(tokens):
    runtime=CaptureRuntime.__new__(CaptureRuntime)
    runtime.options=RuntimeOptions(async_cpu_scatter=True,cpu_scatter_min_tokens=128)
    runtime.queue=[CapturedPacket(('a',),tokens,None,request_tokens=(tokens,))]
    runtime.drain=lambda **kwargs:runtime.queue.clear()
    runtime.drain_requests=lambda request_ids,**kwargs:runtime.queue.clear() if request_ids else None
    runtime.advance_publication({}, NS(jobs={}))
    runtime.join_cpu_scatter()
    assert not runtime.queue
    # A large packet progresses off-thread even without a later submit.
    executor=getattr(runtime,'_cpu_scatter_executor',None)
    if executor is not None:executor.shutdown(wait=True)
    assert (executor is not None)==(tokens>=128)


def test_maintenance_reads_authority_after_foreground_publication():
    runtime=CaptureRuntime.__new__(CaptureRuntime)
    runtime.options=RuntimeOptions(async_cpu_scatter=True)
    runtime.queue=[]
    events=[]
    runtime.drain_requests=lambda request_ids,**kwargs:events.append(('published',request_ids))
    refresh=NS(jobs={'restore':object()}, pump=lambda peers:events.append(('maintenance',peers)))
    runtime.advance_publication({'ready':5}, refresh)
    assert events==[('published',set()),('maintenance',('ready',))]

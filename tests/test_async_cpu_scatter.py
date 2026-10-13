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


def test_background_scatter_finishes_authorized_packet_before_terminal_clip(tmp_path,monkeypatch):
    a,archive=setup(tmp_path/'a')
    b,_=setup(tmp_path/'b','b',archive)
    for owner in (a,b):
        assert owner.append(*payload(0,4),history_begin=0,local_rope_begin=0)
        drain(owner)
    runtime=CaptureRuntime.__new__(CaptureRuntime)
    runtime.options=RuntimeOptions(memory=a.memory.config,async_cpu_scatter=True,cpu_scatter_min_tokens=1)
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
        # Capacity feedback must join before reading the newly published prefix.
        assert runtime.archive_capacity()['a'].history_end==8
        assert b.history_end==8 and not runtime.queue and released==[initial.download]
        assert all(name.startswith('kvmem-scatter') for name in threads)
        latest=packet(8)
        runtime.queue.append(latest)
        runtime.finish({'a'},{'a':TerminalPrefix.from_request(request(12,0,10,6))})
        assert runtime.completed[0]['history_end']==9
        assert runtime.completed[0]['terminal_discarded_tokens']==3
        assert runtime.authorities=={'b':b} and b.history_end==12
        assert runtime.cpu_scatter_batches==1 and len(released)==2
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

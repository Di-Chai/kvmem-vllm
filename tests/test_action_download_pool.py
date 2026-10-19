import gc
import numpy as np
import pytest
import torch

from vllm_kvmem.action_download_pool import ActionDownloadPool


class Event:
    def __init__(self):self.ready=False;self.waits=0
    def query(self):return self.ready
    def synchronize(self):self.waits+=1;self.ready=True


class Backend:
    def __init__(self):self.events=[];self.fail=False
    def allocate(self,size):return torch.empty(size,dtype=torch.uint8)
    def event(self):
        result=Event();self.events.append(result);return result
    def record(self,event,source):
        if self.fail:raise RuntimeError('event failure')
        event.ready=False


@pytest.mark.parametrize('alias',['torch_view','numpy','numpy_view','detached','event'])
def test_all_cpu_and_event_references_must_end_before_reuse(alias):
    backend=Backend();pool=ActionDownloadPool(256,1,backend=backend)
    source=torch.arange(16,dtype=torch.float32)
    host,event=pool.copy(source);event.synchronize()
    pointer=host.data_ptr()
    retained={'torch_view':lambda:host[::2], 'numpy':host.numpy,
              'numpy_view':lambda:host.numpy()[::2], 'detached':host.detach,
              'event':lambda:event}[alias]()
    del host,event;gc.collect()
    with pytest.raises(BufferError):pool.copy(source+1)
    del retained;gc.collect()
    host,event=pool.copy(source+1)
    assert host.data_ptr()==pointer and torch.equal(host,source+1)
    assert pool.snapshot()['reuses']==1 and backend.events[0].waits==1


def test_dma_completion_without_cpu_aliases_and_no_batch_wait():
    backend=Backend();pool=ActionDownloadPool(256,1,backend=backend)
    host,event=pool.copy(torch.ones(8));del host,event;gc.collect()
    with pytest.raises(BufferError):pool.copy(torch.zeros(8))
    assert backend.events[0].waits==0
    backend.events[0].ready=True
    host,event=pool.copy(torch.zeros(8))
    assert not host.any() and pool.snapshot()['event_count']==1


def test_bounded_idle_eviction_and_oversize_do_not_expand_pool():
    backend=Backend();pool=ActionDownloadPool(512,2,backend=backend)
    host,event=pool.copy(torch.ones(1));event.synchronize();del host,event;gc.collect()
    host,event=pool.copy(torch.ones(128));assert host.numel()==128
    assert pool.snapshot()['allocated_pinned_bytes']==512
    with pytest.raises(BufferError):pool.copy(torch.ones(129))
    assert pool.snapshot()['allocated_pinned_bytes']==512


def test_fence_failure_quarantines_storage_even_after_cpu_release():
    backend=Backend();backend.fail=True;pool=ActionDownloadPool(256,1,backend=backend)
    with pytest.raises(RuntimeError,match='event failure'):pool.copy(torch.ones(1))
    gc.collect();backend.fail=False
    with pytest.raises(BufferError):pool.copy(torch.ones(1))
    assert pool.snapshot()['quarantined_slots']==1 and not backend.events[0].waits


def test_owner_data_remains_unchanged_while_other_owner_downloads():
    backend=Backend();pool=ActionDownloadPool(512,2,backend=backend)
    a,ea=pool.copy(torch.ones(16));ea.synchronize()
    b,eb=pool.copy(torch.full((16,),2.));eb.synchronize()
    assert torch.equal(a,torch.ones(16)) and torch.equal(b,torch.full((16,),2.))
    assert a.data_ptr()!=b.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_real_pinned_pool_after_graph_replay_preserves_retained_numpy():
    pool=ActionDownloadPool(512,2)
    source=torch.ones(16,device='cuda')
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):source.add_(1)
    graph.replay()
    a,ea=pool.copy(source);ea.synchronize();old=a.numpy();assert a.is_pinned()
    graph.replay()
    b,eb=pool.copy(source);eb.synchronize()
    assert np.all(old==2) and torch.all(b==3)
    pointer=a.data_ptr();del a,ea,old;gc.collect()
    c,ec=pool.copy(source);ec.synchronize()
    assert c.data_ptr()==pointer and pool.snapshot()['event_count']==2

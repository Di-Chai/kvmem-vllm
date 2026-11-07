"""Borrowed pages retain original validation and one transaction's ownership."""
from types import SimpleNamespace as NS

import pytest
import torch
from vllm_kvmem.window_assembly import LayerWindow, NativeWindowAssembler


def parent_view():
    cache = torch.empty((5,32,2,2,64),dtype=torch.float8_e4m3fn)
    layer = LayerWindow(cache[:,:,:,0],cache[:,:,:,1],torch.zeros(128,32),
                        torch.tensor(1.),torch.tensor([1,3,4]))
    parent = NativeWindowAssembler((layer,),32,32,raw_dtype=torch.float8_e4m3fn)
    runner = NS(_kvmem_capture_runtime=NS(options=NS(reuse_resident_v=False)))
    parent._runner_binding = runner,'r'
    parent.staging.claim(parent)
    return parent,runner,layer


def test_borrow_reuses_exact_views_without_gpu_value_validation(monkeypatch):
    parent,runner,layer = parent_view()
    monkeypatch.setattr(torch,'any',lambda *a:pytest.fail('Unexpected page-table value read'))
    child = NativeWindowAssembler.from_pressure_parent(runner,'r',parent,0,32,256)
    assert child.layers == (layer,) and child._validated_parent is parent
    assert child.staging.slot_bytes == 32*256*2*64*2
    assert child.staging is runner._kvmem_pressure_assembly_staging
    assert child.staging is not parent.staging and child._shared_staging
    assert child.close()
    assert parent.staging.owner is parent
    parent.staging.release(parent)
    assert parent.close()


@pytest.mark.parametrize('fault',['runner','request','closed','released','lease','layer',
                                  'table','mutated_table','block','index','nested'])
def test_borrow_rejects_changed_transaction_or_allocation(fault):
    parent,runner,layer = parent_view();request='r';block=32;index=0
    if fault=='runner':runner=NS()
    if fault=='request':request='other'
    if fault=='closed':parent.closed=True
    if fault=='released':parent.staging.release(parent)
    if fault=='lease':parent._restore_lease=object()
    if fault=='layer':parent.layers=(LayerWindow(layer.key,layer.value,layer.cos_sin,layer.key_scale,layer.kernel_page_table.clone()),)
    if fault=='table':layer.kernel_page_table=layer.kernel_page_table.clone()
    if fault=='mutated_table':layer.kernel_page_table[0]=0
    if fault=='block':block=64
    if fault=='index':index=1
    if fault=='nested':parent._validated_parent=object()
    with pytest.raises(ValueError):NativeWindowAssembler.from_pressure_parent(runner,request,parent,index,block,256)

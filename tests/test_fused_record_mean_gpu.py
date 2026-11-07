"""Byte-exact GPU capture/archive round trip, including recycled staging slots."""
import os
import pytest
import torch

from test_fused_record_mean import make_owner
from vllm_kvmem.authority import append_authority_batch

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',reason='Exclusive GPU byte check')


@pytest.mark.parametrize('dtype',[torch.float8_e4m3fn,torch.bfloat16])
def test_full_geometry_capture_archive_and_reload_bytes(dtype):
    generator=torch.Generator(device='cuda').manual_seed(851)
    raw=torch.randn((17,256,4,256),device='cuda',generator=generator).to(dtype)
    values=torch.randint(0,256,raw.shape,device='cuda',dtype=torch.uint8,generator=generator)
    host_raw=torch.empty(raw.shape,dtype=dtype,pin_memory=True)
    host_v=torch.empty(values.shape,dtype=torch.uint8,pin_memory=True)
    host_raw.copy_(raw,non_blocking=True);host_v.copy_(values,non_blocking=True)
    torch.cuda.synchronize()
    indices=[]
    for enabled in (False,True):
        a,archive=make_owner(dtype,enabled,resident=4,layers=17,heads=4,dim=256,block=32)
        try:
            for start in range(0,256,128):
                assert append_authority_batch([(a,host_raw[:,start:start+128],host_v[:,start:start+128],
                    dict(history_begin=start,local_rope_begin=start))])==1
                a.publish_sealed(lambda:archive.wait_for_capacity(1))
            restored=[a.read_record(r.handle) for r in a.records.values()]
            restored_raw=torch.cat([r for r,v in restored],dim=1).cuda()
            restored_v=torch.cat([v for r,v in restored],dim=1).cuda()
            assert torch.equal(restored_raw.view(torch.uint8),raw.view(torch.uint8))
            assert torch.equal(restored_v,values)
            indices.append(torch.stack([r.mean_key for r in a.records.values()]))
            if enabled:assert a.fused_mean_stats==dict(records=8,batches=2)
        finally:
            assert a.close() and archive.close()
    assert torch.equal(indices[0].view(torch.uint8),indices[1].view(torch.uint8))

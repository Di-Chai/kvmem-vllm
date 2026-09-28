import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='Explicit GPU qualification only')


def test_portable_prompt_state_copies_exact_cuda_bytes_to_new_native_pages():
    from test_session_state import fixture
    adapter, conv, ssm, cache, _ = fixture(device='cuda')
    snapshot = adapter.capture('a', 8, max_bytes=4096)
    before = conv.clone(), ssm.clone(), cache.clone()
    adapter.runner.req_states.req_id_to_index['new'] = 2
    adapter.runner._kvmem_request_generations['new'] = 3
    conv[1:3].fill_(-17); ssm[1:3].fill_(-17); cache[1].fill_(-17)
    completion = adapter.restore(snapshot, 'new')
    assert completion is not None
    completion.synchronize()
    assert torch.equal(conv[5], before[0][1]) and torch.equal(ssm[5], before[1][1])
    assert torch.equal(cache[8], before[2][1])
    assert torch.equal(conv[3:5], before[0][3:5]) and torch.equal(ssm[3:5], before[1][3:5])
    assert torch.equal(cache[2], before[2][2])

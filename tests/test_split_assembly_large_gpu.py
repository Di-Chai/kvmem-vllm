import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')
pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='Explicit GPU qualification only')


def test_resident_v_snapshot_above_signed_int32_byte_range():
    """Exercise the real 128K/17-layer layout, including its >2 GiB tail."""
    from vllm_kvmem.split_assembly import gather_resident_values
    records, layers, tokens, heads, dim = 4096, 17, 32, 4, 256
    source = torch.arange(records * tokens * heads * dim, device='cuda',
                          dtype=torch.int32).remainder_(251).to(torch.uint8)
    source = source.view(records, tokens, heads, dim)
    # Distinct logical and physical orders catch page-table/gather mistakes.
    pages = torch.arange(records - 1, -1, -1, device='cuda', dtype=torch.int32)
    logical = torch.arange(records, device='cuda', dtype=torch.int64).roll(173)
    bases = logical * tokens
    native = [SimpleNamespace(value=source, kernel_page_table=pages)] * layers
    result = gather_resident_values(native, bases, tokens)
    torch.cuda.synchronize()
    assert result.numel() == 2281701376 and result.numel() > 2**31
    expected = source.index_select(0, pages[logical].long())
    for layer in range(layers):
        assert torch.equal(result[:, layer], expected), layer

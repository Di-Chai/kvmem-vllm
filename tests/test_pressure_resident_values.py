from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving.pressure_resident_values import resident_bases


def records(*starts):
    return [NS(history_begin=s,count=32,archived=True) for s in starts]


def test_sources_follow_each_layers_original_dense_order_not_global_block_order():
    mapping=[(-1,-1,64,32,0),(-1,-1,0,32,32),(-1,-1,128,32,64)]
    assert resident_bases(records(0,64,128),mapping,96)==[32,0,64]


@pytest.mark.parametrize('mapping,extent',[
    ([(-1,-1,0,32,0),(-1,-1,0,32,32)],64),
    ([(-1,-1,0,32,0),(-1,-1,32,32,64)],96),
    ([(-1,-1,0,32,0)],64),
    ([(-1,-1,0,32,-1)],32),
    ([(-1,-1,0,33,0)],33),
])
def test_invalid_source_geometry_is_rejected_before_gpu_access(mapping,extent):
    with pytest.raises(ValueError):resident_bases(records(0),mapping,extent)


def test_partial_nonresident_and_unpublished_sources_keep_original_reader():
    partial=NS(history_begin=32,count=17,archived=False)
    assert resident_bases([*records(0),partial],[(-1,-1,0,32,0),(-1,-1,32,17,32)],49) is None
    assert resident_bases(records(64),[(-1,-1,0,32,0)],32) is None
    unpublished=records(0);unpublished[0].archived=False
    assert resident_bases(unpublished,[(-1,-1,0,32,0)],32) is None


def test_duplicate_selection_cannot_alias_a_snapshot_index():
    with pytest.raises(ValueError,match='Repeated'):
        resident_bases(records(0,0),[(-1,-1,0,32,0)],32)

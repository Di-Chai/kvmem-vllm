"""CPU checks for complete-plan validation before native cache mutation."""
import random
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')
from vllm_kvmem.window_assembly import (NativeWindowAssembler, _validate_destinations,
                                      _validate_resident_sources)


def layer(capacity=16, rope=16):
    return NS(kernel_page_table=torch.empty(capacity // 4, dtype=torch.int32),
              key=torch.empty((1, 4, 1, 4)), cos_sin=torch.empty((rope, 4)))


def remap(slot, count, begin):
    return NS(block_id=slot, n_tokens=count, to_base=begin)


def original(remaps, records, layers):
    for r in remaps:
        record = records.get(r.block_id)
        if record is None or record.count != r.n_tokens or r.to_base < 0:
            raise ValueError('Selected generation/extent no longer matches authority')
        for value in layers:
            if r.to_base + r.n_tokens > value.kernel_page_table.numel() * value.key.shape[1]:
                raise ValueError('Selected working window exceeds native allocated pages')
            if r.to_base + r.n_tokens > value.cos_sin.shape[0]:
                raise ValueError('Selected position exceeds the model RoPE table')


def outcome(call, *args):
    try:
        call(*args)
    except ValueError as error:
        return str(error)
    return None


def test_partial_tail_and_exact_smallest_layer_bound_are_allowed():
    rows = [remap(9, 4, 0), remap(2, 3, 4)]
    records = {9: NS(count=4), 2: NS(count=3)}
    _validate_destinations(rows, records, (layer(16, 16), layer(8, 7)))
    with pytest.raises(ValueError, match='RoPE table'):
        _validate_destinations(rows, records, (layer(16, 16), layer(8, 6)))
    with pytest.raises(ValueError, match='allocated pages'):
        _validate_destinations(rows, records, (layer(16, 16), layer(4, 16)))


def test_validation_recomputes_limits_for_each_new_assembly():
    rows, records = [remap(1, 4, 4)], {1: NS(count=4)}
    layers = [layer(8, 16)]
    _validate_destinations(rows, records, layers)
    layers[0].kernel_page_table = torch.empty(1, dtype=torch.int32)
    with pytest.raises(ValueError, match='allocated pages'):
        _validate_destinations(rows, records, layers)


@pytest.mark.parametrize('rows,records', [([remap(1, 4, 0)], {}),
    ([remap(1, 4, 0)], {1: NS(count=3)}), ([remap(1, 4, -1)], {1: NS(count=4)})])
def test_identity_errors_precede_any_layer_inspection(rows, records):
    with pytest.raises(ValueError, match='generation/extent'):
        _validate_destinations(rows, records, [object()])


@pytest.mark.parametrize('layers,expected', [((layer(16, 4), layer(4, 16)), 'RoPE table'),
    ((layer(4, 16), layer(16, 4)), 'allocated pages')])
def test_failure_keeps_original_layer_and_check_order(layers, expected):
    with pytest.raises(ValueError, match=expected):
        _validate_destinations([remap(1, 8, 0)], {1: NS(count=8)}, layers)


def test_empty_plan_inspects_no_layers_and_empty_constructor_still_rejects():
    _validate_destinations([], {}, [object()])
    with pytest.raises(ValueError, match='Invalid native assembly geometry'):
        NativeWindowAssembler((), 4)


def test_reduced_checks_match_original_for_nonuniform_frames():
    rng = random.Random(731)
    for _ in range(500):
        layers = tuple(layer(4*rng.randint(1, 8), rng.randint(1, 32))
                       for _ in range(rng.randint(1, 5)))
        rows = [remap(i, rng.randint(1, 4), rng.randint(-1, 28))
                for i in range(rng.randint(0, 8))]
        records = {r.block_id: NS(count=r.n_tokens) for r in rows}
        if rows and rng.randrange(4) == 0:
            records.pop(rows[-1].block_id)
        assert outcome(_validate_destinations, rows, records, layers) == outcome(
            original, rows, records, layers)


def resident_layer(capacity, *, page_tokens=4, device='cpu'):
    return NS(kernel_page_table=torch.empty(capacity // page_tokens, dtype=torch.int32, device=device),
              value=torch.empty((1, page_tokens, 1, 4), device=device))


def original_resident(remaps, layers):
    for row in remaps:
        if row.from_base < 0 or any(row.from_base+row.n_tokens >
                value.kernel_page_table.numel()*value.value.shape[1] for value in layers):
            raise ValueError('Resident V source exceeds retained native pages')


def test_resident_uses_smallest_capacity_and_allows_exact_extent_without_tensor_reads():
    layers = [resident_layer(32, device='meta'), resident_layer(16, page_tokens=8, device='meta')]
    # Meta tensors have no readable contents; only their CPU-side geometry is used.
    _validate_resident_sources([NS(from_base=8,n_tokens=8)],layers)
    with pytest.raises(ValueError,match='Resident V source'):
        _validate_resident_sources([NS(from_base=9,n_tokens=8)],layers)
    layers[1].kernel_page_table = torch.empty(1,dtype=torch.int32,device='meta')
    with pytest.raises(ValueError,match='Resident V source'):
        _validate_resident_sources([NS(from_base=8,n_tokens=8)],layers)


def test_resident_empty_and_negative_source_keep_original_short_circuit():
    _validate_resident_sources([], [object()])
    _validate_resident_sources([NS(from_base=100,n_tokens=32)], [])
    with pytest.raises(ValueError,match='Resident V source'):
        _validate_resident_sources([NS(from_base=-1,n_tokens=1)], [object()])
    with pytest.raises(ValueError,match='Resident V source'):
        _validate_resident_sources([NS(from_base=-1,n_tokens=1)], [])
    with pytest.raises(ValueError,match='Invalid native assembly geometry'):
        NativeWindowAssembler((),32,packed=True,direct_pinned_reads=True,reuse_resident_v=True)


def test_resident_layer_metadata_is_read_once_not_per_page():
    class Pages:
        def __init__(self): self.calls=0
        def numel(self): self.calls+=1; return 4096
    tables=[Pages() for _ in range(17)]
    layers=[NS(kernel_page_table=t,value=NS(shape=(1,16,4,256))) for t in tables]
    _validate_resident_sources([NS(from_base=i*32,n_tokens=32) for i in range(931)],layers)
    assert [t.calls for t in tables] == [1]*17


def test_resident_hoist_matches_original_with_nonuniform_layers_and_partial_extents():
    rng=random.Random(58013)
    for _ in range(300):
        layers=[resident_layer(4*rng.randint(0,12)) for _ in range(rng.randint(0,6))]
        rows=[NS(from_base=rng.randint(-1,50),n_tokens=rng.randint(1,8))
              for _ in range(rng.randint(0,8))]
        assert outcome(_validate_resident_sources,rows,layers) == outcome(original_resident,rows,layers)


def test_split_checks_all_resident_sources_before_first_snapshot():
    assembler=NativeWindowAssembler.__new__(NativeWindowAssembler)
    assembler.layers=[resident_layer(16),resident_layer(8)]
    assembler.raw_cache=None; assembler.reuse_resident_v=True
    assembler.read_prefetch=False
    reached=[]
    assembler._upload_metadata=lambda values: reached.append(values)
    authority=NS(raw_record_bytes=32,value_record_bytes=32,memory=NS(request_id='owner'))
    rows=[NS(block_id=i,working_k_resident=True,from_base=begin,n_tokens=4)
          for i,begin in enumerate((0,8))]
    records={i:NS(handle=NS(slot=i,generation=1)) for i in range(2)}
    with pytest.raises(ValueError,match='Resident V source'):
        assembler._assemble_split(authority,object(),rows,rows,records)
    assert reached == []


@pytest.mark.parametrize('packed', [False,True])
def test_packed_and_generic_paths_keep_existing_whole_plan_capacity_guard(packed):
    from vllm_kvmem.memory import Phase
    assembler=NativeWindowAssembler.__new__(NativeWindowAssembler)
    assembler.layers=[layer(16,16),layer(8,16)]
    assembler.block_tokens=4;assembler.heads=1;assembler.dim=4
    assembler.raw_dtype=torch.bfloat16;assembler.raw_cache=None
    assembler.packed=packed;assembler.reuse_resident_v=False
    reached=[];assembler._assemble_generic=lambda *args: reached.append('generic')
    row=remap(1,4,8);row.skip=False
    plan=NS(remaps=[row])
    authority=NS(memory=NS(phase=Phase.ASSEMBLING,_plan=plan),shape=(2,4,1,4),
                 raw_dtype=torch.bfloat16,records={1:NS(handle=NS(slot=1),count=4)})
    with pytest.raises(ValueError,match='allocated pages'):
        assembler._assemble(authority,plan)
    assert reached == []

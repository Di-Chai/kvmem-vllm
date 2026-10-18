"""Small real-native correctness checks; no benchmark timing or GPU work."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


path=Path(__file__).resolve().parents[1]/'scripts/benchmark_sealed_mean_append.py'
spec=importlib.util.spec_from_file_location('sealed_mean_append_micro',path)
micro=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=micro
spec.loader.exec_module(micro)


def test_representative_geometry_is_explicit_and_bounded():
    g=micro.Geometry();g.validate();plan=g.allocation_plan()
    assert (g.target_layers,g.mtp_layers,g.heads,g.dim,g.block,g.tokens,g.owners)==(16,1,4,256,32,2048,8)
    assert plan['retained_delta_bytes']==32*1024**2
    assert plan['one_widened_block_bytes']==2*1024**2
    assert plan['conservative_tensor_payload_peak_bytes']<2*1024**3
    with pytest.raises(ValueError,match='2 GiB'):
        micro.Geometry(tokens=4096).validate()


def test_real_batch_archive_and_mean_index_exclude_rejected_and_mtp_rows():
    g=micro.Geometry(target_layers=2,mtp_layers=1,heads=1,dim=64,block=8,tokens=16,owners=2)
    inputs=micro.sources(g)
    first=micro.run_append(g,inputs,measure=False)
    assert first['append_host_seconds'] is None
    proof=first['proof']
    assert proof['records_checked']==4 and proof['history_tokens_per_owner']==16
    assert proof['archived_raw_value_byte_equal'] and proof['mean_index_bit_equal']
    assert proof['poisoned_prior_sum_overwritten'] and proof['rejected_rows_excluded']
    # Change only excluded rows and the MTP layer. Accepted raw bytes change
    # because MTP is archived, while the target MeanK index remains bit-equal.
    for raw,values in inputs:
        raw.view(torch.uint8)[2,:,:,:]=56
        raw.view(torch.uint8)[:,16:,:,:]=126
        values[:,16:,:,:]=255
    second=micro.run_append(g,inputs,measure=False)['proof']
    assert second['sha256']['mean_index']==proof['sha256']['mean_index']
    assert second['sha256']['raw']!=proof['sha256']['raw']
    assert second['sha256']['values']==proof['sha256']['values']


def test_partial_sum_is_required_until_real_seal_recomputes_in_token_order():
    proof=micro.partial_contract()
    assert proof['partial_sum_required'] and proof['partial_index_bit_equal']
    assert proof['completed_seal_matches_token_order'] and proof['mean_index_bit_equal']

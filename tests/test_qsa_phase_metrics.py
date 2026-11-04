import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('benchmark_qsa_phases',
    Path(__file__).parents[1]/'scripts/benchmark_qsa_phases.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prefill_is_a_shared_wall_interval_not_sum_of_concurrent_request_durations():
    rows=[dict(started=0,first=t,input_tokens=100,output_tokens=1,token_ids_verified=True)
          for t in (8,9,10)]
    result=module.phase_summary(rows)
    assert result['effective_prefill_tokens_per_second']==30
    assert result['prefill_wall_seconds']==10
    assert 'decode_steady_aggregate_tokens_per_second' not in result


def test_decode_excludes_prefill_first_chunk_and_finished_stream_tail():
    rows=[dict(started=0,first=first,last=last,input_tokens=100,output_tokens=101,
               token_ids_verified=True,chunks=[dict(at=t,tokens=n) for t,n in
                   [(first,1),(8,40),(9,4),(11,5),(12,50),(last,1)]])
          for first,last in [(2,14),(3,15),(6,16)]]
    # Common interval is (8,12], with uneven accepted MTP chunk sizes.
    result=module.phase_summary(rows)
    assert result['decode_steady_seconds']==4
    assert result['decode_steady_tokens_by_request']==[59]*3
    assert result['decode_steady_aggregate_tokens_per_second']==177/4
    rows[0]['token_ids_verified']=False
    with pytest.raises(ValueError,match='verified accepted'):
        module.phase_summary(rows)

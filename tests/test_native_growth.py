import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_native_growth import CHUNKS, phase


def rows():
    result, total = [], 0
    for i, count in enumerate(CHUNKS, 1):
        total += count
        result.append(dict(sequence=i, call_index=i, previous_sequence=i-1,
            previous_call_index=i-1, previous_end_to_model_ms=2.0, model_ms=10.0,
            decode_requests=0, batch_geometry=[dict(request_id='same', prefill=True,
            prefill_terminal=total==65536, query_tokens=count, active_seq_len=total,
            logical_seq_len=total, offset=0)]))
    return result


def test_actual_append_interval_excludes_prefix_and_leading_gap():
    value = phase(rows(), 32768, 65536)
    assert value['seconds'] == .046
    assert value['new_tokens'] == 32768
    assert value['includes_terminal_sampling']
    assert len(value['shapes']) == 4
    assert phase(rows(), 32768, 49152)['seconds'] == .022
    assert not phase(rows(), 32768, 49152)['includes_terminal_sampling']


@pytest.mark.parametrize('corruption', ['missing','link','owner','context','negative','nan','decode','private'])
def test_phase_rejects_invalid_timing(corruption):
    data = copy.deepcopy(rows())
    if corruption == 'missing':
        data.pop(5)
    elif corruption == 'link':
        data[5]['previous_call_index'] = 0
    elif corruption == 'owner':
        data[5]['batch_geometry'][0]['request_id'] = 'other'
    elif corruption == 'context':
        data[5]['batch_geometry'][0]['logical_seq_len'] += 1
    elif corruption == 'negative':
        data[5]['model_ms'] = -1
    elif corruption == 'nan':
        data[5]['previous_end_to_model_ms'] = float('nan')
    elif corruption == 'decode':
        data[5]['decode_requests'] = 1
    else:
        data[5]['batch_geometry'][0]['offset'] = 32
    with pytest.raises(ValueError):
        phase(data,32768,65536)


def test_phase_boundary_must_be_real_and_complete():
    with pytest.raises(ValueError, match='crosses'):
        phase(rows(), 32769, 65536)
    with pytest.raises(ValueError, match='incomplete'):
        phase(rows(),32768,65537)

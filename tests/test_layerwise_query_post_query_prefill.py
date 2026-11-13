"""A prefill pressure refresh after the final Query is legal in rolling mode.

dsh c8 suite crash (2026-11-11 01:30): an agent Query marks the latest user
message; a cold (unrestored) 213K prompt had ~50K tokens of that turn's tool
history after the Query span, so the window filled again during prefill and
the scheduler issued a prefill-stage pressure refresh. The one-turn diagnostic
guard only exempted decode-stage rolling refreshes and killed the engine.
"""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving.layerwise_query import LayerwiseQuery


def controller(rolling):
    calls = []
    q = LayerwiseQuery.__new__(LayerwiseQuery)
    q.finished = {'r'}
    q.pressure = NS(rolling_decode=rolling, prepare=lambda refresh, job: calls.append(job.command.pressure_stage))
    return q, calls


def job(selection, stage):
    return NS(command=NS(request_id='r', selection=selection, pressure_stage=stage))


@pytest.mark.parametrize('stage', ['prefill', 'decode'])
def test_rolling_pressure_after_the_final_query_reaches_the_layer_pressure_path(stage):
    q, calls = controller(True)
    q.prepare(None, job('pressure', stage), ())
    assert calls == [stage]


@pytest.mark.parametrize('rolling,selection,stage', [
    (False, 'pressure', 'prefill'), (False, 'pressure', 'decode'),
    (True, 'query', 'prefill'), (True, 'query', None)])
def test_other_refreshes_after_the_final_query_still_fail(rolling, selection, stage):
    q, calls = controller(rolling)
    with pytest.raises(RuntimeError, match='one-turn diagnostic'):
        q.prepare(None, job(selection, stage), ())
    assert calls == []

import json
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from vllm_kvmem.capture_runtime import CaptureRuntime
from vllm_kvmem.diagnostics import CompletionHistory, SERVING_COMPLETION_LIMIT, completion_counts
from vllm_kvmem.session_control import SessionBinding
from vllm_kvmem.session_worker import SessionWorker


def test_serving_history_releases_old_reports_but_keeps_cumulative_counts():
    class Report(dict):
        pass
    history = CompletionHistory(serving=True)
    first = Report(request_id='old')
    released = weakref.ref(first)
    history.append(first)
    del first
    for index in range(SERVING_COMPLETION_LIMIT * 4):
        history.append({'request_id': str(index)})
    assert released() is None
    assert len(history) == SERVING_COMPLETION_LIMIT
    assert history[-1]['request_id'] == str(SERVING_COMPLETION_LIMIT * 4 - 1)
    assert completion_counts(history) == {
        'completed_requests': SERVING_COMPLETION_LIMIT * 4 + 1,
        'completed_requests_retained': SERVING_COMPLETION_LIMIT,
        'completed_requests_dropped': SERVING_COMPLETION_LIMIT * 3 + 1,
        'completion_history_limit': SERVING_COMPLETION_LIMIT,
    }


@pytest.mark.parametrize('serving', [False, True])
@pytest.mark.parametrize('retirement', ['synchronous', 'asynchronous', 'checkpoint'])
def test_runtime_retirements_bound_serving_reports_and_preserve_qualification(
        tmp_path, monkeypatch, serving, retirement):
    import vllm_kvmem.capture_runtime as capture
    archive = NS(tier=NS(used_slots=0), capacity_bytes=0, preallocated=False, io_usage=lambda: {})
    layer = NS(num_kv_heads=1, head_dim=2,
               attn=NS(kv_cache_dtype='fp8_e4m3', _v_scale=torch.tensor(1.0)))
    runner = NS(_kvmem_serving=serving, _kvmem_layers=('layer',),
        _kvmem_prepared_archive=archive, model=NS(named_modules=lambda: [('layer', layer)]),
        device=torch.device('cpu'), decode_query_len=1,
        vllm_config=NS(kv_transfer_config=None, cache_config=NS(cache_dtype='fp8_e4m3'),
            max_concurrent_batches=1, scheduler_config=NS(max_num_batched_tokens=128, max_num_seqs=4)))
    monkeypatch.setattr(capture, 'PackedDownloads', lambda *args: NS(pending=[]))
    runtime = CaptureRuntime(runner, tmp_path)
    if retirement == 'asynchronous':
        runtime.retirement = NS(poll=lambda _: {'history_end': 1})
    if retirement == 'checkpoint':
        worker = SessionWorker.__new__(SessionWorker)
        worker.runner = NS(_kvmem_capture_runtime=runtime)
        worker.pool = NS(idle=lambda _: True,prompts={})
        worker.bindings, worker.retiring, worker.completed = {}, {}, []
    # Exercise all production append sites without writing every intermediate
    # snapshot to disk; the final report below uses the real serialization path.
    monkeypatch.setattr(runtime, '_write_archive_report', lambda: None)
    count = SERVING_COMPLETION_LIMIT + 17
    for index in range(count):
        request_id = str(index)
        if retirement == 'synchronous':
            runtime.authorities[request_id] = NS(history_end=1, records={}, writes=[],
                gc_reclaimed_records=0, stop_gc=lambda: None, flush=lambda: True,
                poll=lambda: None, close=lambda: True)
            runtime.accepted[request_id] = 1
            runtime.rejected[request_id] = 0
            runtime.last_steps[request_id] = []
            runtime.finish({request_id})
        elif retirement == 'asynchronous':
            runtime.retiring[request_id] = {'request_id': request_id}
            assert runtime.retirement_done(request_id)
        else:
            worker.bindings[request_id] = SessionBinding(('tenant', 'session'), 1, request_id, (1,))
            worker.retiring[request_id] = (NS(position=1, kind='P', rewind_started=False,
                window=NS(window_tokens=1,records=((0,1,0,1,0),)),
                authority=NS(gc=None, memory=NS(repository=NS(ready_slots=2)))), {'request_id': request_id})
            assert worker.retirement_done(request_id)
            worker.completed.clear()  # Connector consumes notices after each engine step.
        assert len(runtime.completed) <= (SERVING_COMPLETION_LIMIT if serving else count)
    assert not runtime.authorities and not runtime.accepted and not runtime.rejected
    assert not runtime.last_steps and not runtime.retiring
    if retirement == 'checkpoint':
        assert not worker.bindings and not worker.retiring
    CaptureRuntime._write_archive_report(runtime)
    archive_report = json.loads((tmp_path / 'archive.json').read_text())
    progress = json.loads((tmp_path / 'progress.json').read_text())
    retained = SERVING_COMPLETION_LIMIT if serving else count
    assert len(archive_report['completed']) == retained
    assert archive_report['completed'][-1]['request_id'] == str(count - 1)
    assert archive_report['completed'][0]['request_id'] == str(count - retained)
    for report in (archive_report, progress):
        assert report['completed_requests'] == count
        assert report['completed_requests_retained'] == retained
        assert report['completed_requests_dropped'] == count - retained
        assert report['completion_history_limit'] == (SERVING_COMPLETION_LIMIT if serving else None)

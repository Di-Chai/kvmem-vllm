from types import SimpleNamespace

from vllm_kvmem.step_timing import StepTiming


class Event:
    def __init__(self): self.records = 0
    def record(self): self.records += 1
    def query(self): return True
    def elapsed_time(self, end): return 10.


def batch(ids=('p', 'd'), histories=(16384, 65536), scheduled=(2048, 5),
          prompts=(100000, 65536), prefilling=(True, False)):
    return SimpleNamespace(req_ids=list(ids), num_computed_tokens_np=histories,
        num_scheduled_tokens=scheduled, prefill_len_np=prompts,
        num_computed_prefill_tokens_np=tuple(min(h, p) for h, p in zip(histories, prompts)),
        positions=tuple(position for h, n in zip(histories, scheduled) for position in range(h, h+n)),
        is_prefilling_np=prefilling)


def metadata(b, **changes):
    return {rid: dict(history=h, prompt_tokens=p, scheduled_tokens=n,
        atomic_range=None, prepared_replay=False, authorization_id=None) | changes
        for rid, h, p, n in zip(b.req_ids, b.num_computed_tokens_np,
                              b.prefill_len_np, b.num_scheduled_tokens)}


def timer():
    return StepTiming(1, None, event_factory=Event, clock=lambda: 10.)


def test_classification_uses_public_input_not_mtp_padding_or_restored_history():
    b = batch(histories=(99999, 65536), scheduled=(5, 5))
    t = timer(); t.begin(metadata(b)); t.model_begin(b, {'p': 50000})
    t.classify(b)  # Native sample calls this again; retain cost tags/buckets.
    t.end(); row, = t.poll()
    assert row['prefill_tokens'] == row['ordinary_prefill_tokens'] == 1
    assert row['decode_requests'] == 1 and row['prefill_requests'] == 1
    assert row['ordinary_prefill_only'] is True
    assert row['prefill_context_bucket'] == 32768
    assert 'cost_classification' not in row


def test_atomic_overlap_and_prepared_replay_never_train_ordinary_costs():
    b = batch()
    for change in ({'atomic_range': (18000, 19000)}, {'prepared_replay': True}):
        t = timer(); source = metadata(b); source['p'].update(change)
        t.begin(source); t.model_begin(b); t.classify(b); t.end()
        row, = t.poll()
        assert row['ordinary_prefill_only'] is False
        assert row['atomic_prefill_tokens'] == 2048
    # Historical Query spans must not classify an appended tool suffix as Query.
    t = timer(); t.begin(metadata(b, atomic_range=(100, 200)))
    t.model_begin(b); t.end()
    assert t.poll()[0]['ordinary_prefill_only'] is True


def test_missing_or_stale_request_metadata_is_unknown_and_preserves_receipt():
    b = batch()
    for source in ({}, {'p': metadata(b)['p']},
                   metadata(b, history=0, authorization_id=7)):
        t = timer(); t.begin(source); t.model_begin(b); t.end()
        row, = t.poll()
        assert row['ordinary_prefill_only'] is False
        if source and source['p'].get('authorization_id'):
            assert row['unknown_prefill_tokens'] == 2048
            assert row['coalescing_authorization_id'] == 7


def test_public_prepare_replaces_private_query_classification_and_sample_keeps_it():
    public = batch(scheduled=(4096, 5))
    t = timer(); t.begin(metadata(public, authorization_id=23))
    private = batch(ids=('query-copy',), histories=(19008,), scheduled=(4014,),
                    prompts=(23024,), prefilling=(True,))
    t.model_begin(private)
    assert t.active['ordinary_prefill_only'] is False
    t.model_begin(public); t.classify(public)
    assert t.active['model'].records == 2
    t.end(); row, = t.poll()
    assert row['ordinary_prefill_only'] is True
    assert row['prefill_tokens'] == 4096 and row['decode_requests'] == 1
    assert row['coalescing_authorization_id'] == 23


def test_runner_passes_metadata_to_timing_and_public_prepare_wins(tmp_path):
    from test_work_counters import _runner
    from vllm_kvmem.connector import ControlMetadata
    runner = _runner(tmp_path, None)
    public = batch()
    timing = runner._kvmem_step_timing = timer()
    (tmp_path / 'armed').touch()
    output = SimpleNamespace(total_num_scheduled_tokens=2053,
        input_batch=public, finished_req_ids=set(),
        kv_connector_metadata=ControlMetadata(cost_classification=metadata(public)))
    runner.execute_model(output)
    assert timing.active['ordinary_prefill_only'] is True
    assert timing.active['prefill_tokens'] == 2048
    timing.end()

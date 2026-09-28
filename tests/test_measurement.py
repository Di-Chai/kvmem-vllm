import pytest

from vllm_kvmem.measurement import StreamTrace, summarize_group


def trace(start, first, last, end, tokens=3):
    return StreamTrace(start, end, [(first, [1]), (last, list(range(2, tokens + 1)))],
                       finish_reason="stop", usage={"completion_tokens": tokens}, done=True)


def test_concurrent_decode_uses_one_group_window_and_retains_late_prefill_interference():
    # Overlapping durations (4+3) must not become a fictitious seven-second wall.
    result = summarize_group([trace(0, 2, 6, 7), trace(0, 4, 7, 8)])
    assert result["group_arrival_decode_seconds"] == 5
    assert result["aggregate_arrival_decode_tokens_per_second"] == 4 / 5
    assert result["cold_service_tokens_per_second"] == 6 / 8
    assert result["requests"][1]["ttft_seconds"] == 4


def test_single_speculative_burst_is_not_infinite_decode_throughput():
    item = StreamTrace(0, 4, [(3, list(range(8)))], finish_reason="length",
                       usage={"completion_tokens": 8}, done=True)
    result = summarize_group([item])
    assert result["aggregate_arrival_decode_tokens_per_second"] is None
    assert result["requests"][0]["arrival_decode_tokens_per_second"] is None
    assert result["requests"][0]["censored"]


def test_truncated_or_miscounted_stream_cannot_pass_measurement():
    item = trace(0, 1, 2, 3)
    item.done = False
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_group([item])
    item.done = True
    item.usage["completion_tokens"] = 4
    with pytest.raises(ValueError, match="accounting"):
        summarize_group([item])


def test_sse_preserves_bursts_empty_text_tokens_and_trailing_usage(monkeypatch, tmp_path):
    import io
    import json
    from vllm_kvmem import measurement
    wire = ('data: {"id":"r","choices":[\n'
            'data: {"index":0,"text":"hello","token_ids":[10,11]}]}\n\n'
            'data: {"id":"r","choices":[{"index":0,"text":"","token_ids":[12],"finish_reason":"stop"}]}\n\n'
            'data: {"id":"r","choices":[],"usage":{"completion_tokens":3}}\n\n'
            'data: [DONE]\n\n').encode()

    def open_response(request, timeout):
        payload = json.loads(request.data)
        assert payload["stream"] and payload["return_token_ids"]
        assert payload["stream_options"]["include_usage"]
        return io.BytesIO(wire)

    clock = iter([0, 2, 3, 3.5, 4, 4.1])
    monkeypatch.setattr(measurement.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(measurement.time, "perf_counter", lambda: next(clock))
    record = tmp_path / "events.jsonl"
    result = measurement.stream_completion("http://localhost/v1/completions", {"prompt": [1]}, trace_path=record)
    assert result.response_id == "r" and result.token_ids == [10, 11, 12]
    assert result.text == "hello" and result.bursts == [(2, [10, 11]), (3, [12])]
    assert result.finished == 4.1 and result.done
    assert len(record.read_text().splitlines()) == 5

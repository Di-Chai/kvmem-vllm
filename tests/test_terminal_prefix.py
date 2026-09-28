from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.terminal import TerminalPrefix


def request(computed, inflight, total, outputs, status='FINISHED_STOPPED'):
    return NS(request_id='a', num_computed_tokens=computed, num_in_flight_tokens=inflight,
              num_tokens=total, num_output_tokens=outputs, status=NS(name=status),
              is_finished=lambda: True)


@pytest.mark.parametrize('values,expected', [
    ((110, 0, 108, 8), 107),  # Accepted speculation past EOS.
    ((107, 0, 108, 8), 107),  # Ordinary decode, final sampled row not executed.
    ((1024, 128, 2048, 0, 'FINISHED_ABORTED'), 896),  # Cancel during prefill.
    ((118, 8, 111, 11, 'FINISHED_ABORTED'), 110),
    ((0, 0, 2048, 0, 'FINISHED_IGNORED'), 0),
])
def test_native_scheduler_terminal_counter_rules(values, expected):
    assert TerminalPrefix.from_request(request(*values)).history_end == expected


def test_invalid_terminal_state_fails_without_guessing():
    with pytest.raises(ValueError, match='consistent'):
        TerminalPrefix.from_request(request(0, 1, 5, 0))
    r = request(4, 0, 5, 1)
    r.is_finished = lambda: False
    with pytest.raises(ValueError, match='finished'):
        TerminalPrefix.from_request(r)


def test_connector_carries_terminal_prefix_even_without_delayed_page_release(monkeypatch):
    pytest.importorskip('vllm')
    from test_connector import pair
    monkeypatch.setenv('KVMEM_BOUNDED_TARGET', '1')
    scheduler, _ = pair()
    assert scheduler.request_finished_all_groups(request(110, 0, 108, 8), ()) == (False, None)
    metadata = scheduler.build_connector_meta(None)
    assert metadata.terminal_prefixes['a'].history_end == 107
    assert not scheduler.build_connector_meta(None).terminal_prefixes


def test_terminal_packet_is_clipped_before_archive_and_preserves_batch_neighbour(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('vllm')
    from test_authority import setup, payload, drain
    from vllm_kvmem.capture_runtime import CaptureRuntime, CapturedPacket
    from vllm_kvmem.options import RuntimeOptions
    a, archive = setup(tmp_path / 'archive')
    b, _ = setup(tmp_path / 'unused', 'b', archive)
    for owner in (a, b):
        assert owner.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
        drain(owner)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options = RuntimeOptions(memory=a.memory.config)
    runtime.config, runtime.archive = a.memory.config, archive
    runtime.runner = NS(vllm_config=NS(max_concurrent_batches=1))
    runtime.layers, runtime.heads, runtime.dim = [None, None], 1, 2
    runtime.row_elements, runtime.resident_records = 4, 4
    runtime.authorities = {'a': a, 'b': b}
    runtime.accepted, runtime.rejected = {'a': 4, 'b': 4}, {'a': 0, 'b': 0}
    runtime.last_steps = {'a': [], 'b': []}
    runtime.terminal_prefixes, runtime.terminal_discarded = {}, {}
    runtime.batches, runtime._progress_batch, runtime.completed = 2, 0, []
    runtime._write_archive_report = lambda: None
    raw = torch.cat((payload(4, 8)[0], payload(104, 8)[0]), dim=1)
    values = raw.to(torch.uint8)
    header = torch.tensor([0, 8, 16, 4, 4, 4, 4, 2, 0], dtype=torch.int64)
    data = torch.cat((header.view(torch.uint8), raw.contiguous().view(torch.uint8).flatten(), values.flatten()))
    download = NS(payload=data, query=lambda: True)
    runtime.queue = [CapturedPacket(('a', 'b'), 16, download)]
    released = []
    runtime.downloads = NS(release=lambda packet: released.append(packet) or True)
    runtime.finish({'a'}, {'a': TerminalPrefix.from_request(request(10, 0, 8, 4))})
    record = runtime.completed[0]
    assert record['history_end'] == record['accepted_tokens'] == 7
    assert record['terminal_discarded_tokens'] == 3 and record['rejected_tokens'] == 2
    assert record['last_steps'][-1]['history_begin'] + record['last_steps'][-1]['accepted'] == 7
    assert record['sealed_records_verified'] == 1
    assert runtime.authorities == {'b': b} and released == [download]
    assert b.history_end == 12 and runtime.accepted['b'] == 12
    assert b.read_record(next(iter(b.records.values())).handle)[0].shape[1] == 4
    drain(b)
    assert b.close() and archive.close()

from types import SimpleNamespace
import random

import pytest
import torch

from vllm_kvmem.action_attention import ActionHistory, action_metadata


def batch(device='cpu'):
    return SimpleNamespace(num_reqs=3, is_prefilling_np=[False, True, False],
        query_start_loc_np=[0, 3, 7, 9],
        input_ids=torch.tensor([10, 20, 11, 30, 31, 32, 33, 40, 41], device=device),
        positions=torch.tensor([100, 101, 102, 500, 501, 502, 503, 900, 901], device=device))


@pytest.mark.parametrize('prefill', [False, True])
def test_metadata_uses_exact_gpu_coordinates_and_keeps_owned_snapshot(prefill):
    b = batch()
    local = b.positions - 64
    packet = action_metadata(b, local, prefill)
    chosen = list(range(9)) if prefill else [0, 1, 2, 7, 8]
    expected = torch.stack((b.input_ids, b.positions, local))[:, chosen].clone()
    b.input_ids.fill_(-1)
    b.positions.fill_(-1)
    local.fill_(-1)
    torch.testing.assert_close(packet.host, expected)
    assert packet.batch is b
    assert packet.indices == ((0, 1, 2) if prefill else (0, 2))


def test_no_prefill_feedback_needs_no_metadata_packet():
    b = batch()
    b.is_prefilling_np = [True] * 3
    assert action_metadata(b, b.positions, False) is None


def test_reverse_scope_scan_matches_original_complete_prompt_scan():
    rng = random.Random(718)
    options = SimpleNamespace(action_scope='non-thinking', action_interval=1, action_decay=.5)
    for _ in range(200):
        prompt = rng.choices([10, 11, 12, 13, 14, 99], k=rng.randrange(0, 500))
        expected = False
        for token in prompt:
            if token == 14:
                expected = False
            elif token == 12:
                expected = True
            elif token == 13:
                expected = False
        history = ActionHistory(options, [10, 11, 12, 13, 14])
        history.begin('request', prompt)
        assert history.parser.state[0] == expected


def test_latest_message_header_does_not_read_old_history():
    class LongPrompt:
        def __reversed__(self):
            yield 99
            yield 14
            raise AssertionError('Action initialization scanned beyond its final scope reset')
    options = SimpleNamespace(action_scope='non-thinking', action_interval=1, action_decay=.5)
    history = ActionHistory(options, [10, 11, 12, 13, 14])
    history.begin('request', LongPrompt())
    assert history.parser.state[0] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA metadata fence qualification')
def test_metadata_fence_does_not_wait_for_later_model_work():
    b = batch('cuda')
    local = b.positions - 64
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        packet = action_metadata(b, local, True)
        torch.cuda._sleep(250_000_000)
        later = torch.cuda.Event()
        later.record()
    try:
        packet.event.synchronize()
        assert not later.query(), 'Reading Action metadata waited for later model kernels'
        assert packet.host[0].tolist() == [10, 20, 11, 30, 31, 32, 33, 40, 41]
        assert packet.host[2].tolist() == [36, 37, 38, 436, 437, 438, 439, 836, 837]
    finally:
        later.synchronize()

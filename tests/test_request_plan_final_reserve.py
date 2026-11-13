"""Opt-in: reserve the final Query replay only in the window where it starts."""
import pytest

from vllm_kvmem.serving.request_plan import RequestPlan

CAP, BT = 36864 + 4096, 32


def plan():
    # Query at 400000..400710, replayed from 399616 (block aligned).
    return RequestPlan(query=(399616+64, 400326), replay=(399616, 400326), block_tokens=BT, gen_budget=4096)


def test_default_reserves_the_query_replay_in_every_window(monkeypatch):
    monkeypatch.delenv('KVMEM_QUERY_RESERVE_FINAL_WINDOW', raising=False)
    p = plan(); replay = p.replay_end - p.replay_begin
    reserved = (CAP - replay - BT) // BT * BT
    assert p.prefill_boundary(100000, CAP, offset=100000-36864) == reserved
    assert p.prefill_boundary(100000, CAP) == reserved


def test_final_window_only(monkeypatch):
    monkeypatch.setenv('KVMEM_QUERY_RESERVE_FINAL_WINDOW', '1')
    p = plan(); replay = p.replay_end - p.replay_begin
    full, reserved = CAP - 2*BT, (CAP - replay - BT) // BT * BT
    # Early window: the Query cannot start here, pressure comes first.
    assert p.prefill_boundary(100000, CAP, offset=100000-36864) == full
    # Final window (Query start inside the full boundary): original reserve.
    offset = p.replay_begin - (full - 1024)
    assert p.prefill_boundary(p.replay_begin - 2048, CAP, offset=offset) == reserved
    # Callers without a window offset keep the original reserve.
    assert p.prefill_boundary(100000, CAP) == reserved


@pytest.mark.parametrize('start_gap', range(0, 6000, 160))
def test_query_always_fits_when_reached(monkeypatch, start_gap):
    """Walk prefill with pressure at each boundary; the Query must still fit."""
    monkeypatch.setenv('KVMEM_QUERY_RESERVE_FINAL_WINDOW', '1')
    p = plan(); K = 36864
    history = p.replay_begin - 3 * 4096 - start_gap
    history -= history % BT
    offset = history - K                      # window just refreshed to K
    while history < p.replay_begin:
        boundary = p.prefill_boundary(history, CAP, offset=offset)
        step = min(p.replay_begin - history, boundary - (history - offset))
        if step <= 0:                         # pressure: refill window to K
            offset = history - K
            continue
        history += step
    local = history - offset
    assert local + (p.replay_end - p.replay_begin) + BT <= CAP

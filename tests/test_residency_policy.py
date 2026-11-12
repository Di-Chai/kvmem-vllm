"""Opt-in active_resident_victim=expected_wait: revoke the session expected to stay away longest."""
import pytest

from vllm_kvmem.active_residency import ActiveResidency
from vllm_kvmem.residency_policy import ReturnGapModel
from test_active_residency import save
from test_window_manager import make_manager


def test_return_gap_model_learns_gaps_and_session_factors():
    now = [0.0]
    model = ReturnGapModel(clock=lambda: now[0])
    for gap, ns in [(0.2, 'quick'), (0.3, 'quick'), (30.0, 'slow'), (40.0, 'slow'), (0.25, 'quick')]:
        model.finished(ns)
        now[0] += gap
        assert model.arrival(ns) == pytest.approx(gap)
    assert model.arrival('never-finished') is None and model.observations == 5
    # longer elapsed -> longer expected remaining (heavy tail), and slow sessions rank higher
    assert model.expected_remaining('quick', 1.0) > 0
    assert model.expected_remaining('slow', 1.0) > model.expected_remaining('quick', 1.0)
    assert model.expected_remaining('quick', 100.0) == pytest.approx(100.0 * 0.25)   # beyond samples, clipped factor
    assert model.snapshot()['samples'] == 5


def test_expected_wait_revokes_the_longest_expected_absence_first():
    manager, pool = make_manager()
    now = [10.0]
    model = ReturnGapModel(clock=lambda: now[0])
    for ns, gap in [(('', 'quick'), 0.1), (('', 'slow'), 50.0)]:
        model.finished(ns)
        now[0] += gap
        model.arrival(ns)
    resident = ActiveResidency({0: manager}, pool, 0, 1.0, pressure_only=True, clock=lambda: now[0],
                               victim='expected_wait', wait_model=model)
    slow = save(resident, manager, 'a', ('', 'slow'))        # retained first
    quick = save(resident, manager, 'b', ('', 'quick'))
    resident.make_room(pool.get_num_free_blocks() + 1)
    # FIFO would revoke the older 'slow' view too; expected-wait keeps 'quick' because
    # its session returns fast, while the 'slow' session is expected to stay away.
    assert list(resident.entries) == [quick.namespace]
    assert resident.snapshot()['victim'] == 'expected_wait'


def test_expected_wait_differs_from_fifo_when_the_older_view_returns_sooner():
    manager, pool = make_manager()
    now = [10.0]
    model = ReturnGapModel(clock=lambda: now[0])
    for ns, gap in [(('', 'quick'), 0.1), (('', 'slow'), 50.0)]:
        model.finished(ns)
        now[0] += gap
        model.arrival(ns)
    resident = ActiveResidency({0: manager}, pool, 0, 1.0, pressure_only=True, clock=lambda: now[0],
                               victim='expected_wait', wait_model=model)
    quick = save(resident, manager, 'a', ('', 'quick'))       # older, but returns quickly
    slow = save(resident, manager, 'b', ('', 'slow'))
    resident.make_room(pool.get_num_free_blocks() + 1)
    assert list(resident.entries) == [quick.namespace]


@pytest.mark.parametrize('victim,model', [('expected_wait', None), ('fifo', ReturnGapModel()), ('lru', None)])
def test_victim_configuration_is_validated(victim, model):
    manager, pool = make_manager()
    with pytest.raises(ValueError):
        ActiveResidency({0: manager}, pool, 0, 1.0, pressure_only=True, victim=victim, wait_model=model)


def test_option_accepts_only_known_victims():
    from vllm_kvmem.options import RuntimeOptions
    import inspect
    source = inspect.getsource(RuntimeOptions)
    assert "active_resident_victim must be 'fifo' or 'expected_wait'" in source

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('analyze_pressure_profile')


def test_concurrent_refreshes_are_not_added_as_wall_time(profile):
    observed = profile.held_concurrency([dict(start=1, end=4), dict(start=2, end=3)])
    assert observed == {1: 2, 2: 1}
    assert sum(observed.values()) == 3


def test_refresh_stages_partition_owner_hold(profile):
    stages = ['suspended', 'assembly_queued', 'assembly_started', 'selection_prepared',
              'assembly_submitted', 'ready']
    events = [dict(request_id='r', command_id=1, stage=name, at=float(i)) for i,name in enumerate(stages)]
    events[3].update(publication_seconds=.2, selection_seconds=.8)
    report = dict(request_id='r', command_id=1, selection='pressure',
                  assembly_uploaded_bytes=1024, assembly_seconds=1)
    row, = profile.refresh_intervals(events, [report])
    assert sum(row['stages_seconds'].values()) == row['seconds'] == 5
    assert row['selection_seconds'] == .8
    with pytest.raises(ValueError, match='lacks'):
        profile.refresh_intervals(events[:-1], [report])
    with pytest.raises(ValueError, match='Duplicate'):
        profile.refresh_intervals(events + events[:1], [report])
    events[2]['at'] = 99
    with pytest.raises(ValueError, match='ordered'):
        profile.refresh_intervals(events, [report])
def test_full_prefill_window_includes_admission_after_earlier_decode(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from compare_gen_budget_runs import full_prefill_window
    rows = [dict(wall_time=t, completed=dict(public_prefill_tokens=p, public_non_prefill_tokens=d))
            for t, p, d in [(0, 0, 0), (10, 0, 0), (20, 700, 0), (30, 700, 20),
                            (40, 800, 30), (50, 800, 40)]]
    result = full_prefill_window(rows)
    assert result['executed_tokens'] == 800
    assert result['seconds'] == 30
    assert result['tokens_per_second'] == 800 / 30

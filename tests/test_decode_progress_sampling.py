import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('decode_sample', Path(__file__).parents[1]/'scripts/sample_decode_progress.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample(at, history=100, prefill=3, refresh=0, owners=('a', 'b')):
    return {'at': at, 'prefill': prefill, 'decode': 10, 'prefill_raced': False,
            'progress': {'accepted_history': {k: history for k in owners}, 'retiring_requests': 0},
            'scheduler': {'requests': 2, 'refresh_pending': refresh, 'connector_pending': 0}}


def test_stalls_are_counted_and_prefill_crossover_is_excluded():
    old, a, b = sample(0), sample(1), sample(2)
    interval = module.steady_interval(old, a, b, 2)
    assert interval['seconds'] == 1 and interval['accepted_input_tokens'] == 0
    assert module.steady_interval(old, a, sample(2, history=130), 2)['accepted_input_tokens'] == 60
    assert module.steady_interval(sample(0, prefill=2), a, b, 2) is None
    assert module.steady_interval(old, a, sample(2, prefill=4), 2) is None


def test_owner_changes_refresh_rewind_and_invalid_reads_are_not_pure_decode():
    old, a = sample(0), sample(1)
    for b in (sample(2, owners=('a', 'c')), sample(2, refresh=1), sample(2, history=99),
              {**sample(2), 'prefill_raced': True}, {'at': 2, 'error': 'read failure'}):
        assert module.steady_interval(old, a, b, 2) is None

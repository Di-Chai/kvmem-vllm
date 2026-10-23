"""CPU-only execution of the synthetic producer against real native GC."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from vllm_kvmem import _native as n
from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from test_historical_authority import append, close, payload

_NAME = 'historical_gc_throughput_fixture'
spec = importlib.util.spec_from_file_location(_NAME,
    Path(__file__).resolve().parents[1]/'diagnostics'/(_NAME+'.py'))
fixture = importlib.util.module_from_spec(spec)
sys.modules[_NAME] = fixture
spec.loader.exec_module(fixture)


class Runner:
    pass


class Clock:
    def __init__(self):
        self.now = 10.
    def __call__(self):
        self.now += .001
        return self.now


def setup(tmp_path, mode):
    config = KVMemConfig(block_tokens=4, select_budget=16, gen_budget=32,
        repository_blocks=32, sink_blocks=1, recent_blocks=1)
    archive = MemoryArchive(str(tmp_path), 48, 32, native_payloads=True)
    gc = n.HistoricalGcConfig()
    gc.mode, gc.background, gc.policy = mode, False, 'statistical'
    gc.ready_low, gc.ready_high = 1, 2
    gc.scan_blocks, gc.delete_blocks = 32, 8
    worker = archive.enable_history_gc(gc)
    authority = RequestAuthority('a', config, archive, 2, 1, 2, resident_records=16)
    append(authority)
    authority.poll()
    runner = Runner()
    runner._kvmem_capture_runtime = SimpleNamespace(archive=archive)
    runner._kvmem_sessions = SimpleNamespace(bindings={'a':
        SimpleNamespace(namespace=('salt', 'session'), history=0, tokens=())})
    plan = fixture.Plan(0, 40, 4, 4, 12, 16, 20, 24)
    controller = fixture.Controller(Clock(), sample_stride=4)
    controller.arm(runner, ('salt', 'session'), plan)
    return authority, archive, worker, controller, runner


def advance(authority, worker, controller, count=32):
    for _ in range(count):
        worker.advance()
        authority.poll()
        controller.poll(authority)
    # CPU policy pulses do not execute model steps. Sample the resulting
    # counters explicitly; accepted-token pacing is covered separately below.
    controller._sample(authority)


@pytest.mark.parametrize('mode,expected', [(n.HistoricalGcMode.OFF, 0),
                                         (n.HistoricalGcMode.DELETE, 2)])
def test_same_producer_runs_in_off_and_delete_with_real_cpu_release(tmp_path, mode, expected):
    a, archive, worker, controller, runner = setup(tmp_path, mode)
    try:
        controller.poll(a)
        advance(a, worker, controller)
        state = controller.snapshot()
        assert state['error'] is None
        assert state['flushed'] and state['observations_published'] == 3
        assert state['complete'] and len(state['candidate_handles']) == 2
        assert state['source'] == fixture.SOURCE and state['semantic_measurement'] is False
        assert state['last_sample']['counters'] == dict(retired=expected, cpu_released=expected)
        assert a.gc_reclaimed_records == expected
        assert archive.io_usage()['cpu_authority_bytes'] == (8-expected)*48
        # Reclamation changes CPU history; it does not refresh the live window.
        assert len(a.memory._store.working_records()) == 8
        result_slot, result_generation = state['result_handles'][0]
        assert a.memory.repository.pin(n.Handle(result_slot, result_generation))
        events = [r for r in state['events'] if r['kind'] in fixture.COUNTERS]
        assert len(events) == (2 if expected else 0)
        assert all(r['interval_begin'] < r['interval_end'] for r in events)
        # Off and delete both keep sampling after successful publication.
        assert state['poll_calls'] == 33
    finally:
        close(a, archive)


def test_fixture_cannot_bypass_existing_selection_or_result_guards(tmp_path):
    a, archive, worker, controller, runner = setup(tmp_path, n.HistoricalGcMode.DELETE)
    hold = None
    try:
        candidates = [r.handle for r in a.records.values() if 4 <= r.history_begin < 12]
        hold = a.gc.scope.protect_selection(candidates)
        assert hold is not None
        controller.poll(a)
        advance(a, worker, controller)
        assert controller.snapshot()['flushed']
        assert a.gc.scope.stats['retired'] == 0
        assert all(a.memory.repository.pin(h) for h in candidates)
        hold = None
        advance(a, worker, controller)
        assert a.gc.scope.stats['retired'] == a.gc.scope.stats['cpu_released'] == 2
        witness = next(r.handle for r in a.records.values() if r.history_begin == 16)
        assert a.gc.scope.publish_evidence(witness, 100, True, 10, 10, 0., None)
        advance(a, worker, controller)
        assert a.memory.repository.pin(witness)
        assert a.gc.scope.stats['retired'] == 2
    finally:
        hold = None
        close(a, archive)


def test_foreign_namespace_gets_no_evidence_and_restore_fails_closed(tmp_path):
    a, archive, worker, controller, runner = setup(tmp_path, n.HistoricalGcMode.OFF)
    try:
        binding = runner._kvmem_sessions.bindings['a']
        binding.namespace = ('salt', 'foreign')
        controller.poll(a)
        assert controller.snapshot()['poll_calls'] == 0
        assert a.gc.evidence is None
        binding.namespace = ('salt', 'session')
        binding.history = 4
        controller.poll(a)
        assert 'cold request' in controller.snapshot()['error']
        assert a.gc.evidence is None
    finally:
        close(a, archive)


def test_unpublished_source_waits_for_ordinary_archival(tmp_path):
    a, archive, worker, controller, runner = setup(tmp_path, n.HistoricalGcMode.OFF)
    record = next(r for r in a.records.values() if r.history_begin == 4)
    try:
        record.archived = False
        controller.poll(a)
        assert controller.snapshot()['error'] is None
        assert controller.snapshot()['observations_published'] == 0
        assert a.gc.evidence is None
        record.archived = True
        controller.poll(a)
        assert controller.snapshot()['flushed']
    finally:
        record.archived = True
        record = None
        close(a, archive)


def test_standard_plan_is_generated_only_and_has_a_sealed_result_before_trigger():
    plan = fixture.Plan.standard(256, 512)
    assert (plan.source_begin, plan.source_end, plan.result_begin, plan.result_end) == (288, 416, 416, 448)
    assert plan.result_end + plan.block_tokens <= plan.prompt_tokens + plan.trigger_decode_tokens
    for prompt, decode in ((255, 512), (256, 511)):
        with pytest.raises(ValueError):
            fixture.Plan.standard(prompt, decode)
    with pytest.raises(ValueError):
        fixture.Plan(0, 40, 4, 0, 12, 16, 20, 24)


def test_arm_rejects_native_attention_and_non_statistical_worker(tmp_path):
    a, archive, worker, controller, runner = setup(tmp_path, n.HistoricalGcMode.OFF)
    try:
        runner._kvmem_native_attention = object()
        with pytest.raises(ValueError, match='SM12'):
            controller.arm(runner, ('salt', 'session'), controller.plan)
        runner._kvmem_native_attention = None
        archive._history_policy = 'lru'
        with pytest.raises(ValueError, match='statistical'):
            controller.arm(runner, ('salt', 'session'), controller.plan)
    finally:
        close(a, archive)


def test_release_observation_needs_later_accepted_decode_progress(tmp_path):
    a, archive, worker, controller, runner = setup(tmp_path, n.HistoricalGcMode.DELETE)
    try:
        controller.poll(a)
        advance(a, worker, controller)
        events = [r for r in controller.snapshot()['events'] if r['kind'] in fixture.COUNTERS]
        assert len(events) == 2
        assert all('following_decode_at' not in r for r in events)
        assert a.append(*payload(4), history_begin=32, local_rope_begin=32)
        for job, _ in a.writes:
            job.result(5)
        a.poll()
        controller.poll(a)
        events = [r for r in controller.snapshot()['events'] if r['kind'] in fixture.COUNTERS]
        assert all(r['accepted_history'] == 32 and r['following_accepted_history'] == 36
                   and r['interval_end'] < r['following_decode_at'] for r in events)
    finally:
        close(a, archive)


def test_opt_in_install_is_idempotent_and_preserves_original_poll_order(monkeypatch):
    assert fixture._controller is None
    observed = []
    monkeypatch.setattr(RequestAuthority, 'poll', lambda authority: observed.append('original') or 7)
    original = RequestAuthority.poll
    monkeypatch.setattr(fixture, '_controller', None)
    try:
        fixture.install()
        wrapper = RequestAuthority.poll
        monkeypatch.setattr(fixture._controller, 'poll', lambda authority: observed.append('fixture'))
        fixture.install()
        assert RequestAuthority.poll is wrapper
        assert RequestAuthority.poll(object()) == 7
        assert observed == ['original', 'fixture']
    finally:
        RequestAuthority.poll = original


def test_native_counter_sampling_is_paced_by_accepted_tokens(tmp_path, monkeypatch):
    a, archive, worker, controller, runner = setup(tmp_path, n.HistoricalGcMode.OFF)
    try:
        controller.sample_stride = 16
        controller.poll(a)  # Initial snapshot and publication at history 32.
        calls = []
        original = controller._sample
        def sample(authority):
            calls.append(authority.history_end)
            original(authority)
        monkeypatch.setattr(controller, '_sample', sample)
        for _ in range(100):
            controller.poll(a)
        assert not calls
        assert a.append(*payload(4), history_begin=32, local_rope_begin=32)
        for job, _ in a.writes:
            job.result(5)
        a.poll()
        controller.poll(a)
        assert not calls
        assert a.append(*payload(12), history_begin=36, local_rope_begin=36)
        for job, _ in a.writes:
            job.result(5)
        a.poll()
        controller.poll(a)
        assert calls == [48]
    finally:
        close(a, archive)


def test_arm_installs_in_worker_even_if_module_was_cached_before_enable(monkeypatch):
    calls = []
    controller = SimpleNamespace(arm=lambda runner, namespace, plan:
        dict(namespace=namespace, prompt=plan.prompt_tokens))
    monkeypatch.setattr(fixture, '_controller', None)
    def install():
        calls.append('installed')
        fixture._controller = controller
    monkeypatch.setattr(fixture, 'install', install)
    extension = fixture.Extension()
    extension.model_runner = object()
    monkeypatch.delenv(fixture.ENABLED, raising=False)
    with pytest.raises(RuntimeError, match='Explicit'):
        extension.historical_gc_fixture_arm(['salt', 'session'], 256, 512)
    assert not calls
    monkeypatch.setenv(fixture.ENABLED, '1')
    value = extension.historical_gc_fixture_arm(['salt', 'session'], 256, 512)
    assert calls == ['installed'] and value['prompt'] == 256

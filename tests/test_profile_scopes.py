from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')

from vllm_kvmem import profile_scopes as profiling


@pytest.mark.parametrize('kind', [None, 'cuda', 'proton'])
def test_disabled_install_leaves_callables_untouched_and_skips_stage_imports(monkeypatch, kind):
    call = lambda value: value
    runner = NS(vllm_config=NS(profiler_config=NS(profiler=kind)), execute_model=call)
    monkeypatch.setattr(profiling, '_owned_stages', lambda: pytest.fail('Disabled profiler imported stages'))
    profiling.install_profile_scopes(runner)
    assert runner.execute_model is call
    assert set(vars(runner)) == {'vllm_config', 'execute_model'}


def test_instance_scopes_preserve_binding_results_and_idempotence(monkeypatch):
    monkeypatch.setattr(profiling, '_owned_stages', lambda: ())
    class Graph:
        def run_fullgraph(self, value, *, extra):
            return value, extra
    class Draft:
        def precompute_and_store_context_kv(self, value):
            return value
    class Speculator:
        def __init__(self):
            self.model, self.query_cudagraph_manager = Draft(), Graph()
        def propose(self, value):
            value = self.model.precompute_and_store_context_kv(value)
            return self.query_cudagraph_manager.run_fullgraph(value, extra=self)
    class Runner:
        vllm_config = NS(profiler_config=NS(profiler='torch'))
        def __init__(self):
            self.cudagraph_manager, self.speculator = Graph(), Speculator()
        def execute_model(self, value):
            self.prepare_attn(value)
            return self.cudagraph_manager.run_fullgraph(value, extra=self)
        def prepare_attn(self, value):
            return value
        def sample(self, value):
            return value
        def sample_tokens(self, value):
            return self.speculator.propose(self.sample(value))
    original = Runner.execute_model
    runner, untouched = Runner(), Runner()
    profiling.install_profile_scopes(runner)
    first_wrapper = runner.execute_model
    profiling.install_profile_scopes(runner)
    assert runner.execute_model is first_wrapper
    assert Runner.execute_model is original and 'execute_model' not in vars(untouched)
    sentinel = object()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        assert runner.execute_model(sentinel) == (sentinel, runner)
        assert runner.sample_tokens(sentinel) == (sentinel, runner.speculator)
    names = [event.name for event in trace.events() if event.name.startswith('kvmem::')]
    assert sorted(names) == sorted('kvmem::' + stage for stage in (
        'runner.execute', 'runner.prepare_attn', 'target.graph', 'runner.sample_tokens',
        'runner.sample', 'draft.propose', 'draft.context_kv', 'draft.graph'))


def test_owned_class_scopes_keep_nested_work_and_propagate_original_exception(monkeypatch):
    failure = ValueError('original failure')
    class Authority:
        def append(self, value, *, fail=False):
            return self.flush(value, fail=fail)
        def flush(self, value, *, fail=False):
            if fail:
                raise failure
            return value
    monkeypatch.setattr(profiling, '_owned_stages', lambda: (
        (Authority, 'append', 'authority.append'), (Authority, 'flush', 'authority.flush')))
    runner = NS(vllm_config=NS(profiler_config=NS(profiler='torch')))
    profiling.install_profile_scopes(runner)
    first_wrapper = Authority.append
    profiling.install_profile_scopes(runner)
    assert Authority.append is first_wrapper
    authority, sentinel = Authority(), object()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        assert authority.append(sentinel) is sentinel
        with pytest.raises(ValueError) as caught:
            authority.append(sentinel, fail=True)
    assert caught.value is failure
    events = [event for event in trace.events() if event.name.startswith('kvmem::')]
    assert [event.name for event in events].count('kvmem::authority.append') == 2
    assert [event.name for event in events].count('kvmem::authority.flush') == 2
    for event in events:
        if event.name == 'kvmem::authority.flush':
            assert event.cpu_parent.name == 'kvmem::authority.append'


def test_late_graph_managers_and_counter_wrapper_do_not_duplicate_scopes(monkeypatch):
    monkeypatch.setattr(profiling, '_owned_stages', lambda: ())
    class Graph:
        def run_fullgraph(self, value):
            return value
    target = Graph()
    speculator = NS(query_cudagraph_manager=None)
    runner = NS(vllm_config=NS(profiler_config=NS(profiler='torch')),
                cudagraph_manager=target, speculator=speculator)
    profiling.install_profile_scopes(runner)
    # Mirror graph_runtime.initialize_graph_window's retained replay method.
    target._kvmem_original_fullgraph = target.run_fullgraph
    target.run_fullgraph = lambda value: target._kvmem_original_fullgraph(value)
    speculator.query_cudagraph_manager = Graph()
    profiling.install_profile_scopes(runner)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        assert target.run_fullgraph(4) == 4
        assert speculator.query_cudagraph_manager.run_fullgraph(5) == 5
    names = [event.name for event in trace.events() if event.name.startswith('kvmem::')]
    assert names == ['kvmem::target.graph', 'kvmem::draft.graph']

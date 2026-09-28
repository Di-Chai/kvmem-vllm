from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")


@pytest.mark.parametrize('failed', [False, True])
def test_runner_shutdown_drains_retirement_before_releasing_native_gpu_state(failed):
    from vllm_kvmem.runner import make_probe_runner
    calls = []
    class Base:
        def shutdown(self):
            calls.append('native')
    def retire():
        calls.append('retire')
        if failed:
            raise OSError('retirement failure')
    runner = make_probe_runner(Base)()
    runner._kvmem_capture_runtime = SimpleNamespace(shutdown_retirement=retire)
    if failed:
        with pytest.raises(OSError, match='retirement'):
            runner.shutdown()
    else:
        runner.shutdown()
    assert calls == ['retire', 'native']


def test_unarmed_native_warmup_does_not_access_temporarily_disabled_connector(monkeypatch, tmp_path):
    from vllm_kvmem.runner import make_probe_runner
    import vllm.distributed.kv_transfer as transfer

    class Base:
        def execute_model(self, *args, **kwargs):
            return "native"

    runner = make_probe_runner(Base)()
    runner._kvmem_bounded_target = True
    runner._kvmem_probe_output = tmp_path
    runner._kvmem_probe_done = True
    runner._kvmem_continuous = False
    runner._kvmem_query_refresh = lambda command: command
    batch = SimpleNamespace(total_num_scheduled_tokens=1, finished_req_ids=set())

    def disabled():
        raise AssertionError("Native warmup disables the global connector agent")

    monkeypatch.setattr(transfer, "get_kv_transfer_group", disabled)
    assert runner.execute_model(batch) == "native"
    connector = SimpleNamespace(runtime=None)
    monkeypatch.setattr(transfer, "get_kv_transfer_group", lambda: connector)
    (tmp_path / "armed").touch()
    assert runner.execute_model(batch) == "native"
    assert connector.runtime is runner._kvmem_query_refresh


def test_startup_reservation_closes_even_without_a_first_request():
    from vllm_kvmem.runner import make_probe_runner
    closed = []
    class Base:
        def shutdown(self): closed.append('native')
    runner = make_probe_runner(Base)()
    runner._kvmem_prepared_archive = SimpleNamespace(close=lambda: closed.append('archive') or True)
    runner.shutdown()
    assert closed == ['archive', 'native']


def test_prompt_session_controls_start_after_native_warmup(tmp_path):
    from vllm_kvmem.runner import make_probe_runner
    from vllm_kvmem.connector import ControlMetadata
    from vllm_kvmem.session_control import SessionCommands

    class Base:
        def execute_model(self, *args, **kwargs):
            return 'native'

    received = []
    runner = make_probe_runner(Base)()
    runner._kvmem_probe_output = tmp_path
    runner._kvmem_probe_done = True
    runner._kvmem_continuous = False
    runner._kvmem_sessions = SimpleNamespace(receive=received.append)
    batch = SimpleNamespace(total_num_scheduled_tokens=1, finished_req_ids=set(),
                            kv_connector_metadata=None)
    assert runner.execute_model(batch) == 'native'
    assert not received
    (tmp_path / 'armed').touch()
    with pytest.raises(RuntimeError, match='connector metadata'):
        runner.execute_model(batch)
    batch.kv_connector_metadata = ControlMetadata()
    assert runner.execute_model(batch) == 'native'
    assert len(received) == 1 and isinstance(received[0], SessionCommands)


def test_serving_arms_at_first_native_metadata_before_the_forward(tmp_path):
    from vllm_kvmem.runner import make_probe_runner
    from vllm_kvmem.connector import ControlMetadata

    armed_at_forward = []
    received = []
    class Base:
        def execute_model(self, *args, **kwargs):
            armed_at_forward.append((tmp_path/'armed').is_file())
            return 'native'

    runner = make_probe_runner(Base)()
    runner._kvmem_serving = True
    runner._kvmem_probe_output = tmp_path
    runner._kvmem_probe_done = True
    runner._kvmem_continuous = False
    runner._kvmem_sessions = SimpleNamespace(receive=received.append)
    batch = SimpleNamespace(total_num_scheduled_tokens=1, finished_req_ids=set(), kv_connector_metadata=None)
    assert runner.execute_model(batch) == 'native'
    assert armed_at_forward == [False] and not received
    batch.kv_connector_metadata = ControlMetadata()
    assert runner.execute_model(batch) == 'native'
    assert armed_at_forward == [False, True] and len(received) == 1

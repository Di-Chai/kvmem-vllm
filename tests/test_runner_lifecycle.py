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

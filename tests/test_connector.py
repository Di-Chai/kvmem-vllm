"""Control-plane lifecycle tests against vLLM's real connector API types."""
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.outputs import KVConnectorOutput

from vllm_kvmem.connector import KVMemConnector, RefreshCommand, ControlCompletions


class Event:
    done = False

    def query(self):
        return self.done


def pair():
    config = SimpleNamespace(
        kv_transfer_config=KVTransferConfig(kv_connector="KVMemConnector", kv_role="kv_both"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1))
    scheduler = KVMemConnector(config, KVConnectorRole.SCHEDULER, None)
    worker = KVMemConnector(config, KVConnectorRole.WORKER, None)
    return scheduler, worker


def send(scheduler, worker):
    worker.bind_connector_metadata(scheduler.build_connector_meta(None))
    worker.start_load_kv(None)
    worker.clear_connector_metadata()


def poll(scheduler, worker):
    sending, receiving = worker.get_finished(set())
    output = KVConnectorOutput(finished_sending=sending, finished_recving=receiving,
                               kv_connector_worker_meta=worker.build_connector_worker_meta())
    scheduler.update_connector_output(output)
    return output


def test_cancel_waits_for_worker_completion_and_commit_precedes_page_release():
    scheduler, worker = pair()
    event = Event()
    worker.runtime = lambda command: event
    committed = []
    scheduler.scheduler_completion = lambda req, command: committed.append((req, command))
    scheduler.queue(RefreshCommand("a", 17, ()))
    send(scheduler, worker)
    assert poll(scheduler, worker).is_empty()
    assert scheduler.request_finished_all_groups(SimpleNamespace(request_id="a"), ([],)) == (True, None)
    send(scheduler, worker)
    assert poll(scheduler, worker).is_empty()
    assert scheduler.has_pending_push_work()
    event.done = True
    output = poll(scheduler, worker)
    assert committed == [("a", 17)]
    assert output.finished_sending == {"a"}
    assert not scheduler.has_pending_push_work()
    assert poll(scheduler, worker).is_empty()


def test_cancel_before_dispatch_retains_pages_until_admitted_job_finishes():
    scheduler, worker = pair()
    event = Event()
    worker.runtime = lambda command: event
    scheduler.scheduler_completion = lambda *_: None
    scheduler.queue(RefreshCommand("a", 23, ()))
    assert scheduler.request_finished_all_groups(SimpleNamespace(request_id="a"), ()) == (True, None)
    send(scheduler, worker)
    assert poll(scheduler, worker).finished_sending is None
    event.done = True
    assert poll(scheduler, worker).finished_sending == {"a"}


def test_out_of_order_completions_isolate_requests_and_reject_stale_ack():
    scheduler, worker = pair()
    events = {key: Event() for key in ("a", "b")}
    worker.runtime = lambda command: events[command.request_id]
    committed = []
    scheduler.scheduler_completion = lambda req, command: committed.append((req, command))
    scheduler.queue(RefreshCommand("a", 17, ()))
    scheduler.queue(RefreshCommand("b", 18, ()))
    with pytest.raises(RuntimeError, match="Duplicate"):
        scheduler.queue(RefreshCommand("a", 19, ()))
    send(scheduler, worker)
    events["b"].done = True
    poll(scheduler, worker)
    assert committed == [("b", 18)]
    assert scheduler.pending == {"a": 17}
    with pytest.raises(ValueError, match="Stale"):
        scheduler.update_connector_output(KVConnectorOutput(
            kv_connector_worker_meta=ControlCompletions([("a", 999)])))
    assert scheduler.pending == {"a": 17}
    events["a"].done = True
    poll(scheduler, worker)
    assert committed == [("b", 18), ("a", 17)]


def test_ordinary_finish_needs_no_fake_prefix_hit_or_delayed_free():
    scheduler, _ = pair()
    request = SimpleNamespace(request_id="a")
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert scheduler.request_finished_all_groups(request, ()) == (False, None)
    assert not scheduler.has_pending_push_work()
    with pytest.raises(ValueError, match="prefix"):
        scheduler.update_state_after_alloc(request, None, 1)

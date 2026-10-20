"""Native idle pressure uses the same record envelope as scheduler admission."""
from types import SimpleNamespace as NS

from vllm_kvmem import _native as n
from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.prompt_checkpoints import CheckpointPhase
from vllm_kvmem.session_control import HistoryCapacityRequest
from vllm_kvmem.session_worker import SessionWorker
from test_historical_authority import payload, eventually


def test_retained_budget_subtracts_the_index_only_partial_tail_before_native_pressure(tmp_path):
    config = KVMemConfig(block_tokens=4,select_budget=16,gen_budget=16,
        repository_blocks=16,sink_blocks=0,recent_blocks=0)
    archive = MemoryArchive(str(tmp_path),48,16,native_payloads=True)
    settings = n.HistoricalGcConfig()
    settings.mode,settings.background = n.HistoricalGcMode.DELETE,False
    settings.ready_low,settings.ready_high = 1,2
    native = archive.enable_history_gc(settings)
    authority = RequestAuthority('a',config,archive,2,1,2)
    namespace = ('tenant','s')
    worker = SessionWorker.__new__(SessionWorker)
    worker.identities = {namespace:1}
    checkpoint = NS(namespace=namespace,phase=CheckpointPhase.IDLE,authority=authority,window=NS(records=()))
    worker.pool = NS(entries={namespace:checkpoint},prompts={})
    try:
        assert authority.append(*payload(9),history_begin=0,local_rope_begin=0)
        for job,_ in authority.writes:
            job.result(5)
        authority.poll()
        assert archive.namespace_allocated_records('a') == 2 and authority._resident_count == 1
        assert worker._history_rows(checkpoint) == (3,1)
        worker._receive_history_capacity([HistoryCapacityRequest(namespace,1,retained_records=2)])
        def reclaimed():
            native.advance()
            authority.poll()
            return archive.namespace_allocated_records('a') == 1 and not authority.gc.pending_index
        eventually(reclaimed)
        assert authority._resident_count == 1
        assert authority.gc.scope.stats['retired'] == 1
        assert archive.namespace_allocated_records('a')+authority._resident_count == 2
        assert worker._history_rows(checkpoint) == (2,1)
    finally:
        eventually(authority.close)
        eventually(archive.close)

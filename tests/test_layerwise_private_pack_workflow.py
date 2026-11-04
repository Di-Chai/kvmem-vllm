from copy import deepcopy

import pytest

from run_layerwise_matched_performance import archive_backend_audit


def profile():
    return dict(kvmem=dict(cpu_direct_record_seal=True,
                          cpu_seal_serial_allocate=True))


def counters():
    return dict(seal_records=33, seal_direct_records=33,
                seal_payload_bytes=33000, seal_direct_bytes=33000,
                seal_record_copy_seconds=0., archive_io_pending_bytes=0,
                pending_downloads=0, cpu_seal_serial_allocate=True,
                cpu_seal_pack_workers=1, cpu_seal_serial_copy_workers=8,
                seal_serial_allocate_batches_completed=3, seal_batches=3,
                seal_pack_parallel_batches_completed=2,
                seal_pack_parallel_shards_completed=16)


def test_private_backend_proof_includes_small_serial_tail_and_real_parallel_work():
    cfg, stats = profile(), counters()
    before = deepcopy((cfg, stats))
    actual = archive_backend_audit(cfg, stats)
    assert actual['private_pack'] == dict(batches=3, parallel_batches=2,
                                          parallel_shards=16, copy_workers=8)
    assert actual['direct_seal']['records'] == 33
    assert (cfg, stats) == before


@pytest.mark.parametrize('field,value', [
    ('cpu_seal_serial_allocate', False), ('cpu_seal_pack_workers', 8),
    ('cpu_seal_serial_copy_workers', 4),
    ('seal_serial_allocate_batches_completed', 2),
    ('seal_serial_allocate_batches_completed', 4),
    ('seal_batches', 0), ('seal_pack_parallel_batches_completed', 0),
    ('seal_pack_parallel_batches_completed', 4),
    ('seal_pack_parallel_shards_completed', 3),
    ('seal_pack_parallel_shards_completed', 17),
])
def test_flag_cannot_hide_partial_work_or_changed_thread_budget(field, value):
    stats = counters()
    stats[field] = value
    with pytest.raises(ValueError, match='Private CPU packing'):
        archive_backend_audit(profile(), stats)


@pytest.mark.parametrize('field,value', [
    ('cpu_direct_record_seal', False), ('cpu_seal_pack_workers', 8),
])
def test_private_packing_requires_original_direct_records_and_pack_policy(field, value):
    cfg = profile()
    cfg['kvmem'][field] = value
    with pytest.raises(ValueError, match='Private CPU packing'):
        archive_backend_audit(cfg, counters())


def test_memory_option_whitelist_keeps_all_algorithm_controls_frozen():
    from run_layerwise_memory import EXECUTION_KV_KEYS
    assert 'cpu_seal_serial_allocate' in EXECUTION_KV_KEYS
    assert not {'select_budget', 'gen_budget', 'block_tokens', 'sink_blocks',
                'recent_blocks', 'action_layers', 'query_score_mode'} & EXECUTION_KV_KEYS

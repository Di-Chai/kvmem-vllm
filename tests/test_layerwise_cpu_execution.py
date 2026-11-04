from copy import deepcopy

import pytest

from run_layerwise_cpu_execution import archive_backend
from run_layerwise_matched_performance import archive_backend_audit


def snapshot():
    return dict(cpu_seal_hash_workers=8,seal_hash_parallel_batches_completed=12,
                seal_hash_parallel_shards_completed=96,archive_io_pending_bytes=0,pending_downloads=0)


def test_parallel_backend_acceptance_requires_actual_completed_work():
    assert archive_backend(snapshot())['passed']
    assert archive_backend_audit({'kvmem':{'cpu_seal_hash_workers':8}},snapshot())['passed']
    assert archive_backend_audit({'kvmem':{}},{}) is None


@pytest.mark.parametrize('field,value',[
    ('cpu_seal_hash_workers',1),('seal_hash_parallel_batches_completed',0),
    ('seal_hash_parallel_shards_completed',12),('archive_io_pending_bytes',1),('pending_downloads',1)])
def test_requested_flag_is_not_proof_that_parallel_hashing_executed(field,value):
    stats=snapshot();stats[field]=value
    with pytest.raises(ValueError,match='execute or drain'):archive_backend(stats)
    with pytest.raises(ValueError,match='execute or drain'):
        archive_backend_audit({'kvmem':{'cpu_seal_hash_workers':8}},stats)


def region_snapshot():
    return dict(snapshot(),cpu_region_gather_parallel_min_bytes=32*1024**2,
                cpu_region_gather_inline_calls=10,cpu_region_gather_inline_bytes=80*1024**2,
                cpu_region_gather_parallel_calls=0,cpu_region_gather_parallel_bytes=0)


def test_region_acceptance_needs_real_native_inline_work_and_unchanged_hash_policy():
    report=archive_backend_audit({'kvmem':{'cpu_seal_hash_workers':8}},region_snapshot(),32*1024**2)
    assert report['workers']==8 and report['region_gather']['inline_calls']==10
    assert archive_backend_audit({'kvmem':{}},region_snapshot(),32*1024**2)['region_gather']['inline_calls']==10


@pytest.mark.parametrize('field,value',[
    ('cpu_region_gather_parallel_min_bytes',8*1024**2),('cpu_region_gather_inline_calls',0),
    ('cpu_region_gather_inline_bytes',0),('archive_io_pending_bytes',1),('pending_downloads',1)])
def test_region_flag_cannot_replace_actual_execution_or_drain(field,value):
    stats=region_snapshot();stats[field]=value
    with pytest.raises(ValueError,match='execute or drain'):
        archive_backend_audit({'kvmem':{'cpu_seal_hash_workers':8}},stats,32*1024**2)


def test_new_region_mode_rejects_old_archive_telemetry():
    with pytest.raises(KeyError):
        archive_backend_audit({'kvmem':{'cpu_seal_hash_workers':8}},snapshot(),32*1024**2)


def direct_snapshot():
    return dict(region_snapshot(),seal_records=16,seal_direct_records=16,
                seal_payload_bytes=16384,seal_direct_bytes=16384,seal_record_copy_seconds=0.)


def test_direct_publication_requires_all_seals_to_avoid_second_copy():
    from run_layerwise_direct_seal import archive_backend as direct_backend
    profile={'kvmem':{'cpu_direct_record_seal':True,'cpu_seal_hash_workers':8}}
    report=direct_backend(profile,direct_snapshot(),{'cpu_region_gather_min_bytes':32*1024**2})
    assert report['direct_seal']['records']==16
    assert report['direct_seal']['second_record_copy_seconds']==0
    assert report['region_gather']['inline_calls']==10 and report['workers']==8


@pytest.mark.parametrize('field,value',[
    ('seal_direct_records',0),('seal_records',17),('seal_payload_bytes',0),
    ('seal_direct_bytes',16383),('seal_record_copy_seconds',.01)])
def test_requested_direct_publication_cannot_hide_an_unused_or_partial_backend(field,value):
    stats=direct_snapshot();stats[field]=value
    with pytest.raises(ValueError,match='Direct CPU publication'):
        archive_backend_audit({'kvmem':{'cpu_direct_record_seal':True}},stats)


def test_whole_record_control_does_not_require_unexecuted_region_work():
    stats=direct_snapshot()
    stats['cpu_region_gather_inline_calls']=stats['cpu_region_gather_inline_bytes']=0
    profile={'kvmem':{'cpu_direct_record_seal':True,'cpu_seal_hash_workers':8}}
    report=archive_backend_audit(profile,stats,32*1024**2,require_region_execution=False)
    assert report['region_gather']['execution_required'] is False
    assert report['direct_seal']['records']==16
    with pytest.raises(ValueError,match='execute or drain'):
        archive_backend_audit(profile,stats,32*1024**2)

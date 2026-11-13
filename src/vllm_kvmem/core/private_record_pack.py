"""Opt-in serial allocation/parallel copy on an archive's existing bounded pool."""


def _copy_private_range(archive, batch, begin, end):
    archive._gather_worker.active = True
    try:
        return batch.copy_range(begin, end)
    finally:
        batch = None
        archive._gather_worker.active = False


def serial_allocate_pack(archive, views, *, workers=None):
    from vllm_kvmem import _native
    batch = None
    jobs, error = [], None
    try:
        if getattr(archive._gather_worker, 'active', False):
            raise RuntimeError('CPU gather worker cannot recursively pack')
        count = len(views)//2
        # write_from_views already reserves 2x source+destination and record
        # slots before dispatch. The private API never exceeds that payload.
        batch = _native.PrivateRecordPack(views, archive.record_bytes,
                                         count*archive.record_bytes)
        views = None
        workers = min(archive.cpu_seal_pack_workers if workers is None else workers, count)
        if workers <= 1:
            batch.copy_range(0, count)
        else:
            with archive._gather_control:
                if archive._gather_unusable is not None:
                    raise RuntimeError('CPU gather pool unavailable: '+archive._gather_unusable)
                try:
                    for index in range(workers):
                        jobs.append(archive._gather_executor.submit(_copy_private_range,
                            archive, batch, count*index//workers, count*(index+1)//workers))
                except BaseException as failure:
                    error = failure
                    archive._gather_unusable = 'private-pack submission failed'
            if error is not None:
                # A submit may enqueue without returning its Future. Drain the
                # poisoned pool before aborting private outputs/source exports.
                archive._stop_gather_pool()
            for job in jobs:
                try:
                    job.result()
                except BaseException as failure:
                    error = error or failure
            if error is not None:
                raise error
        records = batch.finish()
        if workers > 1:
            with archive._lock:
                archive._seal_pack_parallel_batches_completed += 1
                archive._seal_pack_parallel_shards_completed += workers
        if archive.cpu_seal_serial_allocate:
            with archive._lock:
                archive._seal_serial_allocate_batches_completed += 1
        return records
    finally:
        if batch is not None and not batch.abort():
            raise RuntimeError('Private pack still has active writers after worker drain')
        views = batch = None
        jobs.clear()


def install(archive):
    from types import MethodType
    if not archive.cpu_direct_record_seal or archive._native_payload_arena is not None:
        raise ValueError('Private pack requires direct immutable CPU bytes')
    archive._pack_records = MethodType(serial_allocate_pack, archive)
    return archive

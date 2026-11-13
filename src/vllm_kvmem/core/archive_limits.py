"""Shared logical disk geometry; CPU residency never reserves a whole turn."""


def tiered_capacity_records(options, record_bytes, max_requests):
    """Keep scheduler and worker quota rounding identical, before opening files.

    The first tiered mode guarantees backing for all accepted history on SSD.
    This is a logical quota, not upfront disk allocation or CPU residency.
    Per-scope repository descriptor limits remain independent.
    """
    if any(type(value) is not int or value <= 0 for value in
           (record_bytes, max_requests, options.nvme_max_bytes)):
        raise ValueError('Tiered archive requires positive record, owner and SSD geometry')
    capacity = min(options.memory.repository_blocks * (max_requests + options.session_capacity),
                   options.nvme_max_bytes // record_bytes)
    if capacity <= 0:
        raise ValueError('Tiered SSD quota cannot fit one complete archive record')
    return capacity

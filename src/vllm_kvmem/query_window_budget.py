"""Reserve the shared Query-window cache outside admitted history bytes."""


def split_cpu_history_budget(options):
    """Return history and cache limits; scheduler and archive must share this.

    The overlap cache is an evictable process-wide allocation, not an extra
    historical record. Reserving it once before history admission keeps both
    resources inside the configured CPU limit without changing record leases.
    """
    total = options.cpu_read_cache_bytes
    if getattr(options, 'query_score_mode', 'mean_k') not in ('overlap_16_8', 'key_fps_4'):
        return total, 0
    reserved = getattr(options, 'query_window_cache_bytes', 2147483648)
    if (type(total) is not int or type(reserved) is not int or
            reserved <= 0 or total <= reserved):
        raise ValueError('Overlap Query cache requires a positive reservation '
                         'smaller than the configured CPU memory budget')
    return total - reserved, reserved

"""Shared worker/admission reservation for opt-in redundancy features."""

# At most four BF16 transfer slots, one temporary FP32 conversion,
# and headroom for the paired host/device publication staging. This is reserved
# once per archive, not once per request or historical record. The public
# query_capture_bytes setting scales this reserve by eight; 64 MiB is default.
REDUNDANCY_Q_STAGING_BYTES = 512 * 1024**2
# Accepted input IDs/positions, paired transfer slots and bounded packet
# references. Capture startup checks its maximum against this shared reserve.
THINKING_STAGING_BYTES = 32 * 1024**2


def redundancy_policy_metadata_bytes(capacity, params):
    """Conservative native carry/pool/score workspace bound per owner."""
    pools = min(capacity, params['candidate_blocks']) + min(capacity, params['reference_blocks'])
    per_slot = 192 if params.get('lru_fallback', False) else 64
    if params.get('thinking_only', False):
        per_slot += 512  # Native provenance plus bounded Python record state.
    return per_slot * capacity + 512 * pools + 65536


def split_historical_feature_budget(options, history_bytes, max_scopes):
    """Subtract feature limits after the shared retrieval cache reservation.

    Session owners can outlive running requests, so their maximum number must
    be the same in the scheduler and the archive worker. Native feature/query
    caps include payloads retained by in-flight policy snapshots.
    """
    settings = getattr(options, 'historical_gc', None)
    if settings is None or settings.policy != 'redundancy':
        return history_bytes, 0
    if type(max_scopes) is not int or max_scopes <= 0:
        raise ValueError('Redundancy feature reservation requires positive max_scopes')
    params = settings.policy_params
    metadata = redundancy_policy_metadata_bytes(options.memory.repository_blocks, params)
    reserved = max_scopes * (params['feature_cache_bytes'] + params['query_cache_bytes'] + metadata)
    reserved += 8 * params.get('query_capture_bytes', REDUNDANCY_Q_STAGING_BYTES // 8)
    if params.get('thinking_only', False):
        reserved += THINKING_STAGING_BYTES
    if type(history_bytes) is not int or history_bytes <= reserved:
        raise ValueError('CPU history budget cannot fit redundancy feature/query '
                         'reservations plus historical records')
    return history_bytes - reserved, reserved

"""Archive feature flags read by core and serving alike."""
from . import settings as _settings


def layerwise_gc():
    """Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: prepared pressure readers on a CPU
    archive with deleting historical GC (native payload arena). The reader holds
    one native read right per source (``HistoricalPayloadRead``) for its whole
    plan, so a concurrent retirement cannot free or reuse those bytes before
    the reader closes; a record retired before preparation raises KeyError."""
    value = _settings.get('KVMEM_LAYERWISE_HISTORICAL_GC', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAYERWISE_HISTORICAL_GC must be 0 or 1')
    return value == '1'


def native_reader():
    value = _settings.get('KVMEM_PRESSURE_NATIVE_READER', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_NATIVE_READER must be 0 or 1')
    from .. import _native
    return value == '1' and callable(getattr(_native, 'RecordSources', None))

"""Bounded immutable original-Query Q for guided-refresh fallback.

Only accepted public input publishes a seed. The tensor stays on the CPU with
its authority, and checkpoint restoration discards seeds from a future branch.
"""
from dataclasses import dataclass
import weakref


MAX_OWNER_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 256 * 1024**2


@dataclass(frozen=True)
class QuerySeed:
    end: int
    tensor: object

    @property
    def bytes(self):
        return self.tensor.numel()*self.tensor.element_size()


def publish_seed(runtime, authority, queries, end):
    if queries is None:
        return
    owners = getattr(runtime, '_guided_seed_owners', None)
    if owners is None:
        owners = runtime._guided_seed_owners = weakref.WeakSet()
    size = queries.numel()*queries.element_size()
    total = sum(getattr(owner, 'guided_seed').bytes for owner in owners
                if owner is not authority and getattr(owner, 'guided_seed', None) is not None)
    if size > MAX_OWNER_BYTES or total+size > MAX_TOTAL_BYTES:
        authority.guided_seed = None
        authority.guided_seed_fallback = 'original-query-memory-limit'
        return
    tensor = queries.detach().to('cpu', copy=True).contiguous()
    authority.guided_seed = QuerySeed(end, tensor)
    authority.guided_seed_fallback = None
    owners.add(authority)


def original_queries(authority, history, device):
    seed = getattr(authority, 'guided_seed', None)
    if seed is None or seed.end > history:
        return None
    return seed.tensor.to(device)

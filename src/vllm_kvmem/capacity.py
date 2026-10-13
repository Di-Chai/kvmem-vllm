"""Reserve each admitted target request's future native GPU-page growth.

The native allocator still owns every physical block. Reservations are credits,
not extra allocations: trimming a window returns physical pages while retaining
that request's right to grow again. The optional qualified draft contract adds
native SWA and speculative GDN credits.
Execution remains synchronous, APC-off, TP=1.
"""
import hashlib
import inspect
from pathlib import Path

from .window_manager import WorkingSetManager

KV_MANAGER_SHA256 = "58800535eda854580d84e1025bf32fe01b2a43ce0bba501b4f3d125e4f2c8afa"


class ReservedKVCacheManager:
    def __init__(self, native, vllm_config, *, draft_contract=None, max_authorities=None):
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager, SlidingWindowManager
        from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs

        if hashlib.sha256(Path(inspect.getfile(KVCacheManager)).read_bytes()).hexdigest() != KV_MANAGER_SHA256:
            raise ValueError("Unqualified native page reservation/free hooks")
        if ((vllm_config.speculative_config is not None) != (draft_contract is not None) or
                vllm_config.cache_config.enable_prefix_caching or
                vllm_config.cache_config.mamba_cache_mode != "none" or
                vllm_config.max_concurrent_batches != 1):
            raise ValueError("Capacity reservation requires synchronous APC-off execution and an explicit draft contract")
        self.native = native
        if max_authorities is not None and (type(max_authorities) is not int or max_authorities <= 0):
            raise ValueError('Authority admission requires a positive request count')
        self.max_authorities = max_authorities
        self.authority_rejections = 0
        self.lookahead = draft_contract.query_tokens if draft_contract is not None else 0
        self.groups = tuple(native.coordinator.single_type_managers)
        self.static_blocks = {}
        for index, manager in enumerate(self.groups):
            if manager.block_pool is not native.block_pool:
                raise ValueError("Capacity reservation requires the native shared page pool")
            if isinstance(manager, WorkingSetManager):
                continue
            spec = manager.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.first_spec
            if type(manager) is SlidingWindowManager and draft_contract is not None:
                if type(spec) is not SlidingWindowSpec or spec.sliding_window != draft_contract.sliding_window:
                    raise ValueError("Unqualified draft sliding-window reservation")
                # Native recycling happens before allocation. Include an extra
                # drafter lookahead beyond the target in-flight token budget.
                cap = spec.max_admission_blocks_per_request(
                    vllm_config.max_in_flight_tokens + self.lookahead, native.max_model_len)
                self.static_blocks[index] = cap
                continue
            if (type(manager) is not MambaManager or type(spec) is not MambaSpec or
                    spec.num_speculative_blocks != (self.lookahead - 1 if self.lookahead else 0) or
                    spec.num_prefill_checkpoint_blocks or spec.mamba_cache_mode != "none" or
                    spec.block_size < native.max_model_len):
                raise ValueError("Unqualified recurrent-state reservation geometry")
            self.static_blocks[index] = spec.max_num_blocks_per_req(vllm_config, native.max_model_len)
        self.reservations = {}
        self._step_remaining = None
        self._step_total = 0
        self.rejections = 0
        if self._demand(native.max_model_len) > native.block_pool.get_num_free_blocks():
            raise ValueError("GPU page pool cannot fit one worst-case KVMem request")

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _demand(self, tokens):
        return sum(self.static_blocks[index] if index in self.static_blocks else
                   (min(tokens, manager.capacity) + manager.block_size - 1) // manager.block_size
                   for index, manager in enumerate(self.groups))

    def _owned(self, request_id):
        # APC is disabled: live pages are exclusive and null pages own no credit.
        return sum(not block.is_null for manager in self.groups
                   for block in manager.req_to_blocks.get(request_id, ()))

    def _remaining(self, request_id, demand):
        remaining = demand - self._owned(request_id)
        if remaining < 0:
            raise RuntimeError("Native request allocation exceeds its admitted lifetime bound")
        return remaining

    def begin_step(self):
        """Reconcile trims/imports once, before native synchronous scheduling.

        During this step native allocation changes only its current request;
        free and deferred-free operations below update the same credit ledger.
        Acknowledged window trims happen between scheduling steps. Standalone
        callers retain the conservative scan path outside this explicit scope.
        """
        if self._step_remaining is not None:
            raise RuntimeError('Capacity credit step already open')
        remaining = {key:self._remaining(key, quota)
                     for key, (_, quota) in self.reservations.items()}
        self._step_remaining, self._step_total = remaining, sum(remaining.values())

    def end_step(self):
        self._step_remaining, self._step_total = None, 0

    def _update_credit(self, request_id, demand):
        if self._step_remaining is not None:
            remaining = self._remaining(request_id, demand)
            self._step_total += remaining - self._step_remaining.get(request_id, 0)
            self._step_remaining[request_id] = remaining

    def _forget_credit(self, request_id):
        if self._step_remaining is not None:
            self._step_total -= self._step_remaining.pop(request_id, 0)

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        request_id = request.request_id
        existing = self.reservations.get(request_id)
        if existing is not None and existing[0] is not request:
            raise ValueError("Request identity reused before releasing capacity")
        if existing is None and self.max_authorities is not None and len(self.reservations) >= self.max_authorities:
            self.rejections += 1
            self.authority_rejections += 1
            return None
        # A terminal speculative step may verify a full block and draft another
        # before the scheduler observes EOS/max_tokens. Retain those credits.
        tokens = min(self.native.max_model_len,
                     request.num_prompt_tokens + request.max_tokens + 2 * self.lookahead)
        demand = self._demand(tokens)
        if existing is not None and existing[1] != demand:
            raise ValueError("Request lifetime bound changed after admission")
        others = (self._step_total - self._step_remaining.get(request_id, 0)
                  if self._step_remaining is not None else
                  sum(self._remaining(key, quota) for key, (_, quota) in self.reservations.items()
                      if key != request_id))
        extra = kwargs.get("reserved_blocks", 0)
        # Preserve any additional native connector constraint conservatively.
        # This connector normally contributes zero extra credits.
        reserved = others + extra
        if existing is None and self._remaining(request_id, demand) + reserved > self.block_pool.get_num_free_blocks():
            self.rejections += 1
            return None
        result = self.native.allocate_slots(request, num_new_tokens, *args,
                                             **{**kwargs, "reserved_blocks": reserved})
        if result is not None:
            self.reservations[request_id] = (request, demand)
        if result is not None or existing is not None:
            if self._step_remaining is not None:
                self._update_credit(request_id, demand)
            elif result is not None:
                self._remaining(request_id, demand)
        return result

    def free(self, request):
        self.native.free(request)
        self._forget_credit(request.request_id)
        self.reservations.pop(request.request_id, None)

    def pop_blocks_for_free(self, request):
        blocks = self.native.pop_blocks_for_free(request)
        self._forget_credit(request.request_id)
        # Deferred physical blocks remain unavailable in the native pool. The
        # completed request no longer needs credits for future growth.
        self.reservations.pop(request.request_id, None)
        return blocks

"""Native operation-sized GPU allocation backed by an explicit pause owner.

The native shared pool remains the physical allocator. This adapter removes
only future lifetime growth credits; original K/G extents, guided probes,
pressure refresh controls, recurrent/speculative rows and restore permits stay
in the existing allocation path. A synchronous scheduler must preflight each
complete operation and skip a refused owner rather than invoke recomputation.
"""
from .capacity import ReservedKVCacheManager
from .window_manager import WorkingSetManager


class IncrementalKVCacheManager(ReservedKVCacheManager):
    """Allocate exact atomic operations, with no future G credit between steps.

    ``pause_controller`` must implement ``is_suspended(request_id)`` and
    ``on_pressure(request, required_pages, available_pages)``. The latter is a
    notification only: it must not free pages or pause already scheduled rows
    synchronously inside this callback. The scheduler consumes pressure at a
    safe boundary, and the controller acknowledges checkpoint/restore fences.

    Merely constructing this object is insufficient scheduler integration:
    ``can_schedule`` must gate the final unsplit operation in the native running
    loop, and ordinary live-request preemption must remain forbidden.
    """
    def __init__(self, native, vllm_config, *, pause_controller,
                 draft_contract=None, max_authorities=None):
        if (pause_controller is None or
                not callable(getattr(pause_controller, 'is_suspended', None)) or
                not callable(getattr(pause_controller, 'on_pressure', None))):
            raise ValueError('Incremental capacity requires an explicit fenced pause controller')
        if draft_contract is not None and draft_contract.method != 'mtp':
            raise ValueError('Incremental capacity supports target-only or native MTP')
        if getattr(getattr(vllm_config, 'scheduler_config', None), 'async_scheduling', False):
            raise ValueError('Incremental capacity requires synchronous scheduling')
        super().__init__(native, vllm_config, draft_contract=draft_contract,
                         max_authorities=max_authorities)
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager
        if any(not isinstance(manager, WorkingSetManager) and type(manager) is not MambaManager
               for manager in self.groups):
            raise ValueError('Incremental capacity requires full-attention and non-recycling state groups')
        self.pause_controller = pause_controller
        self.pressure_requests = {}
        self.operation_deferrals = 0

    def _remaining(self, request_id, demand):
        # Keep the existing lifetime *ceiling* check for malformed allocations.
        # It is no longer a reservation for work which has not been scheduled.
        super()._remaining(request_id, demand)
        return 0

    def admission_preview(self, request):
        if (request.request_id not in self.reservations and self.max_authorities is not None
                and len(self.reservations) >= self.max_authorities):
            return 'retirement_or_active_limit'
        if self.pause_controller.is_suspended(request.request_id):
            return 'request_suspended'
        # No operation shape is available here. The final scheduler shape and
        # native allocation decide; max_tokens is deliberately not predicted.
        return None

    def _operation_pages(self, request, num_new_tokens, kwargs):
        from vllm.v1.request import RequestStatus
        counts = (num_new_tokens, kwargs.get('num_new_computed_tokens', 0),
                  kwargs.get('num_external_computed_tokens', 0),
                  kwargs.get('num_lookahead_tokens', 0), kwargs.get('reserved_blocks', 0))
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError('Operation token/page extents must be nonnegative integers')
        if not num_new_tokens and not kwargs.get('num_external_computed_tokens', 0):
            raise ValueError('An operation must compute or import tokens')
        if kwargs.get('num_encoder_tokens', 0):
            raise ValueError('Incremental capacity does not support encoder groups')
        computed = kwargs.get('new_computed_blocks')
        blocks = computed.blocks if computed is not None else self.native.empty_kv_cache_blocks.blocks
        if kwargs.get('num_new_computed_tokens', 0) or any(blocks):
            raise ValueError('Incremental capacity does not accept ordinary prefix hits')
        local = request.num_computed_tokens
        external = kwargs.get('num_external_computed_tokens', 0)
        total = min(local + external, self.native.max_model_len)
        main_end = total + num_new_tokens
        end = min(main_end + kwargs.get('num_lookahead_tokens', 0), self.native.max_model_len)
        needed = self.native.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id, num_tokens=end, new_computed_blocks=blocks,
            num_encoder_tokens=0, total_computed_tokens=local+external,
            num_local_computed_tokens=local, num_tokens_main_model=main_end)
        # Preserve the same native waiting-queue watermark. Native allocation
        # checks this again; no synthetic prefix hit or loaned null page exists.
        if (kwargs.get('has_scheduled_reqs', True) and
                request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)):
            needed += self.native.watermark_blocks
        return needed + kwargs.get('reserved_blocks', 0)

    def _pressure(self, request, required, available):
        record = dict(required_pages=required, available_pages=available,
                      history_end=request.num_computed_tokens)
        previous = self.pressure_requests.get(request.request_id)
        self.pressure_requests[request.request_id] = record
        self.operation_deferrals += 1
        if previous != record:
            self.pause_controller.on_pressure(request, required, available)

    def _pause_reserve_options(self, request, options):
        callback = getattr(self.pause_controller, 'reserved_pages', None)
        reserve = callback(request.request_id) if callback is not None else 0
        if type(reserve) is not int or reserve < 0:
            raise ValueError('Invalid pending restore page reservation')
        return {**options, 'reserved_blocks': options.get('reserved_blocks', 0) + reserve}

    def can_schedule(self, request, num_new_tokens, *, notify=True, **kwargs):
        """Read-only physical preview of the final whole operation.

        Returning False lets ``_reserve_prefill_lookahead`` return zero so the
        native loop continues with other owners. This does not reserve pages;
        invoke it while constructing the same synchronous batch, and retain
        the final allocation check. It never truncates Query/private work.
        """
        if self.pause_controller.is_suspended(request.request_id):
            return False
        if (request.request_id not in self.reservations and self.max_authorities is not None
                and len(self.reservations) >= self.max_authorities):
            return False
        options = self._guided_allocation_options(request, num_new_tokens, kwargs)
        options, _ = self._prefill_allocation_options(request, num_new_tokens, (), options,
                                                     self.reservations.get(request.request_id))
        options = self._pause_reserve_options(request, options)
        required = self._operation_pages(request, num_new_tokens, options)
        available = self.block_pool.get_num_free_blocks()
        residency = self.active_residency
        if residency is not None:
            # Idle leases can be revoked; staged imports already disappeared
            # from entries and are subtracted by WorkingSetManager's predictor.
            available += len({block.block_id for view in residency.entries.values()
                              for block in view.blocks if not block.is_null and block.ref_cnt == 1})
        if required > available:
            if notify:
                self._pressure(request, required, available)
            return False
        return True

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        if args:
            raise ValueError('Incremental allocation requires named native options')
        if self.pause_controller.is_suspended(request.request_id):
            return None
        # Keep the native step allocation and all recurrent/speculative groups;
        # only its prompt-wide admission hint is replaced by operation credit.
        return super().allocate_slots(request, num_new_tokens,
                                      **{**kwargs, 'full_sequence_must_fit': False})

    def _native_allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        kwargs = self._pause_reserve_options(request, kwargs)
        required = self._operation_pages(request, num_new_tokens, kwargs)
        if self.active_residency is not None:
            self.active_residency.make_room(required)
        available = self.block_pool.get_num_free_blocks()
        if required > available:
            self._pressure(request, required, available)
            return None
        result = super()._native_allocate_slots(request, num_new_tokens, *args, **kwargs)
        if result is None:
            self._pressure(request, required, self.block_pool.get_num_free_blocks())
        else:
            self.pressure_requests.pop(request.request_id, None)
            observer = getattr(self.pause_controller, 'on_allocation', None)
            if observer is not None:
                observer(request, request.num_computed_tokens + kwargs.get('num_external_computed_tokens', 0))
        return result

    def free(self, request):
        super().free(request)
        self.pressure_requests.pop(request.request_id, None)

    def pop_blocks_for_free(self, request):
        result = super().pop_blocks_for_free(request)
        self.pressure_requests.pop(request.request_id, None)
        return result

    def snapshot(self):
        result = super().snapshot()
        result['mode'] = 'incremental'
        result['lifetime_ceiling_pages'] = result.pop('lifetime_quota_pages')
        result['future_reserved_pages'] = 0
        callback = getattr(self.pause_controller, 'reserved_pages', None)
        restore_credit = callback('') if callback is not None else 0
        result['restore_progress_reserved_pages'] = restore_credit
        result['free_after_growth_reserve_pages'] = result['physical_free_pages'] - restore_credit
        for request_id, row in result['requests'].items():
            row['lifetime_ceiling_pages'] = row.pop('lifetime_quota_pages')
            row['future_reserved_pages'] = 0
            row['suspended'] = self.pause_controller.is_suspended(request_id)
        result['operation_deferrals'] = self.operation_deferrals
        result['pressure_requests'] = {key: dict(value) for key, value in self.pressure_requests.items()}
        return result

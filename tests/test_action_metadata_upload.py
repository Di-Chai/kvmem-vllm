"""CPU lifetime and accepted-prefix checks; CUDA numerical qualification is separate."""
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.action_metadata_upload import PackedActionMetadata, typed_views
from vllm_kvmem.options import RuntimeOptions
from test_options import config


class Event:
    def __init__(self, *, ready=False, failure=None):
        self.ready, self.failure, self.queries = ready, failure, 0

    def query(self):
        self.queries += 1
        if self.failure is not None:
            raise self.failure
        return self.ready

    def synchronize(self):
        pytest.fail('The staging pool must never synchronize')


class Backend:
    def __init__(self):
        self.capture = False
        self.allocations = self.uploads = 0
        self.events = []
        self.order = []
        self.fence_error = self.copy_error = None
        self.current = NS(device=torch.device('cuda:0'))

    def capturing(self):
        return self.capture

    def stream(self, device):
        return self.current

    def allocate(self, size, device):
        self.allocations += 1
        return torch.empty(size, dtype=torch.uint8), torch.empty(size, dtype=torch.uint8)

    def upload(self, device, host):
        self.uploads += 1
        self.order.append('upload')
        device.copy_(host)
        if self.copy_error is not None:
            raise self.copy_error

    def fence(self, stream):
        assert stream is self.current
        self.order.append('fence')
        if self.fence_error is not None:
            raise self.fence_error
        event = Event()
        self.events.append(event)
        return event


def pool(max_bytes=256, max_leases=128):
    backend = Backend()
    return PackedActionMetadata(max_bytes=max_bytes, max_leases=max_leases, backend=backend), backend


def upload(p, **kw):
    return p.lease([0], [4], [0], 0, 4, 'cuda:0', **kw)


def test_four_fields_exact_integer_types_alignment_and_owned_source():
    p, backend = pool(512)
    starts, counts, rows = [0, 7, 12], [7, 5, 4], [0, 3]
    with p.lease(starts, counts, rows, 2**31+9, 2**53+1, 'cuda:0') as lease:
        expected = ([0, 7, 12], [7, 5, 4], [2**31+9, 2**31+12], [2**53+1, 2**53+4])
        assert tuple(x.dtype for x in lease.views) == (torch.int32, torch.int32, torch.int64, torch.int64)
        assert tuple(x.tolist() for x in lease.views) == expected
        assert all(x.data_ptr() % 8 == 0 for x in lease.views[2:])
        starts[0] = counts[0] = rows[0] = -1
        assert tuple(x.tolist() for x in typed_views(lease.host, 3, 2)) == expected
        assert lease.host.data_ptr() != lease.device.data_ptr()
        assert p.pending[lease.offset] is lease
    assert backend.uploads == backend.allocations == 1
    assert p.snapshot()['arena_host_bytes'] + p.snapshot()['arena_gpu_bytes'] == 512
    assert p.snapshot()['pending_payload_bytes'] == 2*(8*3+16*2)


def test_busy_credit_is_not_reused_and_completed_owner_can_retire_out_of_order():
    p, backend = pool(96)  # Two 24-byte slices, each charged on CPU and GPU.
    with upload(p) as first:
        pass
    with upload(p) as second:
        pass
    assert first.offset != second.offset
    with upload(p) as unavailable:
        assert unavailable is None
    assert backend.uploads == 2 and len(p.pending) == 2
    second.event.ready = True
    with upload(p) as third:
        assert third.offset == second.offset
        assert first.offset in p.pending and first.event.queries > 0
    assert second.released and not first.released
    first.event.ready = third.event.ready = True
    p.poll()
    p.poll()
    assert not p.pending and p.free == [(0, 48)]
    assert p.snapshot()['completed'] == 3 and backend.allocations == 1


@pytest.mark.parametrize('error', [RuntimeError('consumer failed'), asyncio.CancelledError()])
def test_unpublished_packet_or_cancel_retains_lease_until_all_enqueued_consumers(error):
    p, backend = pool()
    with pytest.raises(type(error)):
        with upload(p) as lease:
            backend.order.extend(['layer1', 'layer2', 'layer3', 'mass_d2h'])
            raise error  # Packet has not been created/published.
    assert backend.order == ['upload', 'layer1', 'layer2', 'layer3', 'mass_d2h', 'fence']
    assert p.pending[lease.offset] is lease and not lease.released
    p.poll()
    assert not lease.released
    lease.event.ready = True
    p.poll()
    assert lease.released


def test_normal_completion_reuses_original_mass_fence_instead_of_upload_fence():
    p, backend = pool()
    original = Event()
    with upload(p) as lease:
        backend.order.extend(['score_layer1', 'score_layer2', 'score_layer3', 'original_mass_event'])
        lease.complete(original)
    assert not backend.events and p.pending[lease.offset].event is original
    original.ready = True
    p.poll()
    assert lease.released


def test_copy_error_after_possible_enqueue_keeps_storage_and_fences():
    p, backend = pool()
    backend.copy_error = RuntimeError('copy may already be submitted')
    with pytest.raises(RuntimeError, match='copy may'):
        with upload(p):
            pytest.fail('copy error must propagate')
    assert backend.order == ['upload', 'fence'] and len(p.pending) == 1
    backend.events[0].ready = True
    p.poll()
    assert not p.pending


def test_fence_record_failure_quarantines_unrecorded_event_and_prevents_new_work():
    p, backend = pool()
    backend.fence_error = RuntimeError('event record failed')
    with pytest.raises(RuntimeError, match='event record failed'):
        with upload(p) as lease:
            pass
    assert lease.event is None and p.pending[lease.offset] is lease
    assert p.snapshot()['quarantined'] and p.snapshot()['fence_failures'] == 1
    for operation in (p.poll, lambda: upload(p).__enter__()):
        with pytest.raises(RuntimeError, match='quarantined'):
            operation()
    assert backend.uploads == 1 and p.snapshot()['pending_payload_bytes'] == 48


def test_event_query_error_is_fatal_and_retains_every_uncompleted_owner():
    p, backend = pool()
    with upload(p) as first:
        pass
    with upload(p) as second:
        pass
    first.event.failure = RuntimeError('asynchronous CUDA failure')
    second.event.ready = True
    with pytest.raises(RuntimeError, match='asynchronous CUDA'):
        p.poll()
    assert len(p.pending) == 2 and not first.released and not second.released
    with pytest.raises(RuntimeError, match='quarantined'):
        p.poll()
    assert p.snapshot()['query_failures'] == 1


def test_pack_overflow_before_enqueue_returns_credit_without_creating_event():
    p, backend = pool()
    with pytest.raises((RuntimeError, OverflowError)):
        with p.lease([2**31], [1], [0], 0, 0, 'cuda:0'):
            pytest.fail('int32 overflow must not silently wrap')
    assert not p.pending and backend.uploads == 0 and not backend.events
    assert p.free == [(0, p.capacity)]


def test_graph_capture_skips_allocation_and_event_poll_and_non_cuda_falls_back():
    p, backend = pool()
    with upload(p) as first:
        pass
    first.event.ready = True
    backend.capture = True
    with upload(p) as value:
        assert value is None
    p.poll()
    assert first.event.queries == 0 and not first.released
    with p.lease([0], [1], [0], 0, 0, 'cpu') as value:
        assert value is None
    assert backend.uploads == backend.allocations == 1
    assert p.snapshot()['fallbacks'] == {'graph_capture': 1, 'non_cuda': 1}


@pytest.mark.parametrize('max_bytes,max_leases,reason', [(32, 128, 'oversize'), (256, 1, 'lease_limit')])
def test_budget_failure_uses_legacy_fallback_without_loss_or_sync(max_bytes, max_leases, reason):
    p, backend = pool(max_bytes, max_leases)
    if reason == 'lease_limit':
        with upload(p):
            pass
    before = backend.uploads
    with upload(p) as lease:
        assert lease is None
    assert backend.uploads == before and p.snapshot()['fallbacks'][reason] == 1
    assert p.snapshot()['arena_host_bytes'] + p.snapshot()['arena_gpu_bytes'] <= max_bytes


def test_profile_counts_packing_single_transfer_and_fallback_without_changing_values():
    stages = []
    class Profile:
        @contextmanager
        def stage(self, name, **fields):
            stages.append((name, fields))
            yield
    p, backend = pool(48)
    with upload(p, profile=Profile()):
        pass
    with upload(p, profile=Profile()) as value:
        assert value is None
    assert [x[0] for x in stages] == ['metadata.packed_arena_allocate', 'metadata.packed_cpu',
        'metadata.packed_h2d', 'metadata.packed_fallback']
    assert stages[2][1]['transfer_bytes'] == 24
    assert p.snapshot()['uploaded_bytes'] == 24 and p.snapshot()['submissions'] == 1


def test_fragmented_credit_falls_back_then_merges_only_completed_slices():
    p, backend = pool(144)  # Three 24-byte slices.
    leases = []
    for _ in range(3):
        with upload(p) as lease:
            leases.append(lease)
    leases[0].event.ready = leases[2].event.ready = True
    p.poll()
    # 48 total free bytes, but no contiguous 40-byte owned range.
    with p.lease([0], [4], [0, 1], 0, 4, 'cuda:0') as unavailable:
        assert unavailable is None
    assert p.snapshot()['fallbacks'] == {'byte_credit': 1}
    leases[1].event.ready = True
    with p.lease([0], [4], [0, 1], 0, 4, 'cuda:0') as lease:
        assert lease is not None and lease.offset == 0 and lease.size == 40
    assert backend.allocations == 1


def test_other_device_cannot_reuse_arena_and_other_thread_cannot_poll():
    from concurrent.futures import ThreadPoolExecutor
    p, backend = pool()
    with upload(p) as lease:
        pass
    backend.current = NS(device=torch.device('cuda:1'))
    with p.lease([0], [4], [0], 0, 4, 'cuda:1') as value:
        assert value is None
    assert p.snapshot()['fallbacks'] == {'different_device': 1}
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match='engine-thread owned'):
            executor.submit(p.poll).result()
    assert not lease.released and backend.allocations == 1


def test_options_are_off_by_default_and_explicit_enable_requires_action_scoring():
    assert RuntimeOptions.from_vllm(config({})).async_action_metadata is False
    values = {'async_action_metadata': True, 'action_attention': True,
              'session_capacity': 2, 'session_checkpoint_bytes': 4096,
              'async_archive_retirement': True}
    assert RuntimeOptions.from_vllm(config({'kvmem': values})).async_action_metadata is True
    assert values['async_action_metadata'] is True
    with pytest.raises(ValueError, match='requires action attention'):
        RuntimeOptions.from_vllm(config({'kvmem': {'async_action_metadata': True}}))
    with pytest.raises(ValueError, match='boolean'):
        RuntimeOptions.from_vllm(config({'kvmem': {'async_action_metadata': 1}}))


def test_three_layer_actual_capture_preserves_metadata_packets_and_accepted_prefix(tmp_path, monkeypatch):
    """Exercise both production upload branches; only GPU storage/kernels are CPU fakes."""
    from vllm_kvmem.action_attention import ActionRuntime
    import vllm_kvmem.action_attention as action
    import vllm_kvmem.action_scoring as scoring
    empty = torch.empty
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: empty(*a, **{n: v for n, v in k.items() if n != 'pin_memory'}))
    original_events = []
    class MassEvent(Event):
        def record(self):
            original_events.append(self)
        def synchronize(self):
            self.ready = True
    monkeypatch.setattr(torch.cuda, 'Event', MassEvent)
    monkeypatch.setattr(action, 'native_vllm_fp8_views', lambda *_: (None, None))
    launches = []
    def score(q, pos, key, table, trig, ks, starts, counts, length, **kwargs):
        launches.append((q.tolist(), pos.tolist(), starts.tolist(), counts.tolist(), length))
        # Distinct layer/row values expose row selection, layer sum and prefix truncation.
        return (q.float().sum((1, 2))[:, None]/100 * (starts.float()[None, :]+1)).softmax(-1)
    monkeypatch.setattr(scoring, 'page_attention', score)
    opts = NS(prefill_attention=True, prefill_attention_interval=1, prefill_attention_stride=1,
        prefill_attention_half_life=128, native_prefill_commit=False,
        memory=NS(select_budget=16, gen_budget=8, block_tokens=4),
        action_scope='non-thinking', action_interval=1, action_decay=.5)
    outcomes = []
    for enabled in (False, True):
        p, backend = pool(1024)
        class CPUAdapter:
            def lease(self, starts, counts, rows, begin, local_begin, device, **kw):
                return p.lease(starts, counts, rows, begin, local_begin, 'cuda:0', **kw)
        owners = {rid: NS(memory=NS(_store=NS(working_records=lambda: [(0, 1, 0, 4, 0)]))) for rid in ('a', 'b')}
        r = ActionRuntime.__new__(ActionRuntime)
        r.options = opts
        r.profile = None
        r.metadata_upload = CPUAdapter() if enabled else None
        r.runtime = NS(options=opts, config=NS(block_tokens=4), output=tmp_path,
            drain_requests=lambda ids, wait: True, ensure_authority=owners.__getitem__)
        r.runner = NS(req_states=NS(req_id_to_index={'a': 0, 'b': 1}),
            _kvmem_sessions=NS(bindings={'b': NS(tokens=[14, 13])}),
            block_tables=NS(block_tables=[NS(gpu=torch.tensor([[0, 1], [0, 1]]))]))
        r.layers = [(i, NS(attn=NS(layer_name=f'L{i}', kv_cache=None, _k_scale=None,
            _q_scale=None, impl=NS(scale=.5)), num_kv_heads=1, head_dim=4,
            rotary_emb=NS(cos_sin_cache=None))) for i in range(3)]
        r.groups = {f'L{i}': 0 for i in range(3)}
        r.q_dtypes = r.prefill_q_dtypes = {f'L{i}': torch.bfloat16 for i in range(3)}
        r.ids = [10, 11, 12, 13, 14]
        r.metadata_early_batches = 0
        r.metadata = None
        r.candidate_samples = r.packets = 0
        r.capture_seconds = 0.
        b = NS(num_reqs=2, req_ids=['a', 'b'], is_prefilling_np=[True, False],
            query_start_loc_np=[0, 2, 4], input_ids=torch.tensor([20, 21, 22, 23]),
            positions=torch.tensor([8, 9, 8, 9]))
        captured = [NS(query=(torch.arange(32).reshape(4, 2, 4)+i).to(torch.bfloat16),
                       positions=torch.tensor([4, 5, 4, 5])) for i in range(3)]
        r.prepare(b, captured[0].positions)
        packets = r.capture(captured, b)
        saved = [(rid, v.rows, v.history, v.masses.tolist(), v.prefill) for rid, v in packets.items()]
        assert backend.uploads == (2 if enabled else 0)
        if enabled:
            assert len(p.pending) == 2 and all(v.event in original_events for v in p.pending.values())
            assert not backend.events  # Normal route reuses exact original mass events.
        for rid in ('a', 'b'):
            r.accept(NS(action_inputs=packets), rid, 2 if rid == 'a' else 1)
        assert owners['a'].action_history.prefill.samples == 2
        assert owners['b'].action_history.samples == 1  # Reject second speculative row.
        outcomes.append((saved, owners['a'].action_history.prefill.snapshot(),
                         owners['b'].action_history.mass, owners['b'].action_history.samples))
        p.poll()
        assert not p.pending
    assert outcomes[0] == outcomes[1]
    assert launches[:6] == launches[6:] and len(launches) == 12

"""Refresh attribution reuses step timing without enabling Action CUDA profiling."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem.async_refresh import AsyncRefresh, RefreshJob
from vllm_kvmem.config import KVMemConfig


@pytest.mark.parametrize('action_profile,step_timing', [(False, False), (False, True),
                                                       (True, False), (True, True)])
def test_step_timing_exposes_existing_refresh_diagnostics_without_extra_work(
        monkeypatch, tmp_path, action_profile, step_timing):
    from vllm_kvmem import prefill_attention, window_assembly

    cfg = KVMemConfig(block_tokens=4, select_budget=12, gen_budget=12,
                      repository_blocks=32, sink_blocks=0, recent_blocks=0)
    manager = AsyncRefresh.__new__(AsyncRefresh)
    manager.options = SimpleNamespace(memory=cfg, pressure_policy='attention',
        packed_assembly=True, direct_pinned_reads=True, mean_k_backend='qw3_cuda',
        action_profiling=action_profile)
    manager.runner = SimpleNamespace(device='cpu', _kvmem_probe_output=tmp_path,
        _kvmem_step_timing=object() if step_timing else None,
        _kvmem_capture_runtime=SimpleNamespace(names=('layer',)))
    manager.maintenance = None
    calls, events = [], []
    manager.stream = SimpleNamespace(wait_event=lambda event: calls.append('ready'))
    manager.emit = lambda job, stage, **fields: events.append((stage, fields))
    authority = SimpleNamespace(records={}, publish_sealed=lambda wait: calls.append('publish'),
        memory=SimpleNamespace(_store=SimpleNamespace(working_records=lambda: ())))
    command = SimpleNamespace(request_id='owner', command_id=1,
        groups=((0, SimpleNamespace(history_end=64, window_tokens=12)),),
        selection='pressure', mandatory_spans=(), query_span=None, input_tokens=(1,))
    job = RefreshJob(command, authority=authority, selected_tokens=12)
    plan = object()
    assembly = SimpleNamespace(host_prepare_seconds=.1, uploaded_bytes=100,
        raw_cache_hit_tokens=0, resident_v_reused_tokens=8, read_stats={'cache_copy_seconds': .03},
        chunks=1, rebuilt_tokens=8, skipped_tokens=4)
    def select(*args, **kwargs):
        calls.append('select')
        return plan
    def assemble(*args):
        calls.append('assemble')
        return assembly
    assembler = SimpleNamespace(assemble=assemble)
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *args: nullcontext())
    monkeypatch.setattr(prefill_attention, 'select_pressure', select)
    monkeypatch.setattr(prefill_attention, 'selection_report', lambda *args, **kwargs: {})
    monkeypatch.setattr(window_assembly.NativeWindowAssembler, 'from_runner',
                        lambda *args, **kwargs: assembler)

    result = manager._assemble(job, None, object())
    assert result[:3] == (assembler, assembly, plan)
    assert calls == ['ready', 'publish', 'select', 'assemble']
    enabled = action_profile or step_timing
    selection_events = [fields for stage, fields in events if stage == 'selection_prepared']
    assert len(selection_events) == int(enabled)
    if enabled:
        assert selection_events[0]['publication_seconds'] >= 0
        assert selection_events[0]['selection_seconds'] >= 0
    manager.report(job, assembly, result[3], result[4])
    report = json.loads((tmp_path/'bounded_refresh.jsonl').read_text())
    assert ('assembly_read_stats' in report) == enabled
    if enabled:
        assert report['assembly_read_stats'] == assembly.read_stats
        assert report['assembly_chunks'] == 1
        assert report['assembly_rebuilt_tokens'] == 8
        assert report['assembly_skipped_tokens'] == 4
    assert manager.options.action_profiling is action_profile

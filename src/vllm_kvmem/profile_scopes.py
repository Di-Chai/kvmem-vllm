"""Opt-in CPU scopes for native execution and KVMem's context lifecycle.

These ranges annotate launches and host work; they do not synchronize CUDA or
measure GPU duration themselves. Native objects are wrapped per runner instance.
Project-owned classes are wrapped once in the dedicated profiler worker process.
Ordinary serving never installs these wrappers.
"""
from functools import wraps


def _wrapped_for(method, stage):
    return getattr(method, '_kvmem_profile_stage', None) == stage


def _wrap(owner, method_name, stage):
    if owner is None:
        return
    original = getattr(owner, method_name, None)
    if not callable(original) or _wrapped_for(original, stage):
        return
    from torch.profiler import record_function

    @wraps(original)
    def scoped(*args, **kwargs):
        with record_function('kvmem::' + stage):
            return original(*args, **kwargs)

    scoped._kvmem_profile_stage = stage
    setattr(owner, method_name, scoped)


def _owned_stages():
    # Import only in an explicitly enabled profiler worker, after model loading.
    from .authority import RequestAuthority
    from . import authority as authority_module
    from .capture_runtime import CaptureRuntime
    from .cached_storage import CachedArchive
    from .native_state import NativeRecurrentState
    from .session_state import PromptStateAdapter
    from .state import StateCheckpoints
    from .storage import AsyncArchive
    from .transfer import PackedDownloads, PackedTransfers
    from .window_assembly import NativeWindowAssembler
    from .window_batch import WindowBatchAdapter
    return (
        (CaptureRuntime, 'submit', 'capture.submit'),
        (CaptureRuntime, 'drain', 'capture.drain'),
        (CaptureRuntime, 'finish', 'capture.finish'),
        (CaptureRuntime, 'join_cpu_scatter', 'capture.join_cpu_scatter'),
        (CaptureRuntime, '_write_progress', 'capture.progress'),
        (RequestAuthority, 'append', 'authority.append'),
        (authority_module, 'append_authority_batch', 'authority.append_batch'),
        (RequestAuthority, '_prepare_append', 'authority.prepare_append'),
        (RequestAuthority, '_commit_prepared_append', 'authority.commit_append'),
        (RequestAuthority, 'flush', 'authority.flush'),
        (RequestAuthority, 'select_for_queries', 'authority.select'),
        (WindowBatchAdapter, 'prepare', 'window.prepare'),
        (NativeWindowAssembler, 'assemble', 'window.assemble'),
        (NativeRecurrentState, 'borrow', 'recurrent.borrow'),
        (NativeRecurrentState, 'canonicalize_accepted', 'recurrent.canonicalize'),
        (StateCheckpoints, 'capture', 'recurrent.checkpoint'),
        (StateCheckpoints, 'restore', 'recurrent.rollback'),
        (PromptStateAdapter, 'capture', 'session.capture'),
        (PromptStateAdapter, 'restore', 'session.restore'),
        (PackedDownloads, 'download', 'transfer.download'),
        (PackedTransfers, 'upload_from_reader', 'transfer.upload'),
        (AsyncArchive, 'write', 'archive.write'),
        (CachedArchive, '_cache_write', 'archive.cache_write'),
        (CachedArchive, 'read', 'archive.read'),
    )


def install_profile_scopes(runner):
    config = getattr(getattr(runner, 'vllm_config', None), 'profiler_config', None)
    if getattr(config, 'profiler', None) != 'torch':
        return
    for method, stage in (
        ('execute_model', 'runner.execute'),
        ('sample_tokens', 'runner.sample_tokens'),
        ('prepare_attn', 'runner.prepare_attn'),
        ('sample', 'runner.sample'),
    ):
        _wrap(runner, method, stage)
    manager = getattr(runner, 'cudagraph_manager', None)
    # initialize_graph_window preserves the previous replay method underneath
    # its counter wrapper. If that method is already scoped, a second install
    # must not introduce a duplicate nested target.graph range.
    if not _wrapped_for(getattr(manager, '_kvmem_original_fullgraph', None), 'target.graph'):
        _wrap(manager, 'run_fullgraph', 'target.graph')
    speculator = getattr(runner, 'speculator', None)
    _wrap(speculator, 'propose', 'draft.propose')
    _wrap(getattr(speculator, 'model', None), 'precompute_and_store_context_kv', 'draft.context_kv')
    _wrap(getattr(speculator, 'query_cudagraph_manager', None), 'run_fullgraph', 'draft.graph')
    _wrap(getattr(speculator, 'prefill_cudagraph_manager', None), 'run_fullgraph', 'mtp.prefill_graph')
    _wrap(getattr(speculator, 'decode_cudagraph_manager', None), 'run_fullgraph', 'mtp.decode_graph')
    for owner, method, stage in _owned_stages():
        _wrap(owner, method, stage)

"""Native Qwen MTP with target-selected, lockstep immutable draft KV.

Only the accepted prefix of the native MTP prefill is archived. Subsequent
speculative draft steps use the original projection and cannot overwrite that
prefix capture. Target MeanK scores alone select both target and MTP raw KV;
assembly re-RoPEs both into the same dense working frame, as in qw3.
"""
import torch
from dataclasses import replace
from types import MethodType

from .capture import install_qwen_capture, install_persistent_capture


def install_mtp_runtime(runner):
    if runner.speculative_config is None or runner.speculative_config.method != 'mtp':
        return
    proposer = runner.speculator
    from .draft_contract import verify_mtp_sources
    verify_mtp_sources()
    names = install_qwen_capture(proposer.model)
    modules = dict(proposer.model.named_modules())
    mapped = {'__mtp__.' + name: modules[name] for name in names}
    if len(mapped) != 1:
        raise ValueError('Bounded Qwen MTP requires one native full-attention layer')
    for name, module in mapped.items():
        module._kvmem_capture_name = name
        module._kvmem_mtp_prefix_only = True
        module._kvmem_mtp_prefix_active = False
        if getattr(runner, '_kvmem_native_compilation', False):
            # Native compiled forwards omit Python guards on later calls. A
            # device gate survives both prefill and independent draft replays.
            module._kvmem_capture_gate = torch.zeros((), dtype=torch.int32, device=runner.device)
    runner._kvmem_mtp_modules = mapped
    runner._kvmem_archive_layers = [*runner._kvmem_layers, *mapped]
    # Stable prefix outputs survive the independent single-token draft graphs.
    captures = install_persistent_capture(proposer.model, names, runner.max_num_tokens,
                                          proposer.input_buffers.positions)
    for record, name in zip(captures, mapped):
        record.name = name
    runner._kvmem_mtp_captures = captures
    original_prefill = proposer._prefill

    def prefill(self, *args, **kwargs):
        for module in mapped.values():
            module._kvmem_mtp_prefix_active = True
            gate = getattr(module, '_kvmem_capture_gate', None)
            if gate is not None:
                gate.fill_(1)
        try:
            return original_prefill(*args, **kwargs)
        finally:
            for module in mapped.values():
                module._kvmem_mtp_prefix_active = False
                gate = getattr(module, '_kvmem_capture_gate', None)
                if gate is not None:
                    gate.zero_()

    proposer._prefill = MethodType(prefill, proposer)
    original_propose = proposer.propose

    def propose(self, input_batch, *args, **kwargs):
        pending = getattr(runner, '_kvmem_pending_capture', None)
        window = getattr(runner, '_kvmem_window_batch', None)
        draft_batch = input_batch
        if pending is not None and window is not None:
            # Preserve logical scheduler/SSM history, while native MTP RoPE and
            # its metadata builders consume the committed dense KV window.
            lengths = next(iter(window.groups.values()))
            draft_batch = replace(input_batch, positions=window.positions,
                seq_lens=lengths.seq_lens, seq_lens_cpu_upper_bound=lengths.cpu_upper_bound)
        result = original_propose(draft_batch, *args, **kwargs)
        if pending is not None:
            captured, original_batch = pending
            if original_batch is not input_batch:
                raise ValueError('MTP publication must match the accepted target batch')
            rejected = runner._kvmem_mtp_rejected
            runner._kvmem_capture_runtime.submit([*captured, *captures], original_batch, rejected)
            runner._kvmem_pending_capture = None
            runner._kvmem_mtp_rejected = None
        return result

    proposer.propose = MethodType(propose, proposer)

"""Expensive, opt-in byte oracle for accepted-history pressure maintenance.

This runs only with the existing session qualification switch. It deliberately
copies state to CPU and must never be used for performance measurements.
"""
import json

import torch


class RollingPressureOracle:
    def __init__(self, runner, job):
        from .native_state import NativeRecurrentState
        self.runner = runner
        self.adapter = NativeRecurrentState(runner)
        self.bundle = self.adapter.borrow([job.command.request_id])
        self.before = {key: tensor.detach().cpu().clone()
                       for key, tensor in self.bundle.tensors.items()}
        runtime = runner._kvmem_capture_runtime
        steps = runtime.last_steps.get(job.command.request_id, ())
        self.last_step = dict(steps[-1]) if steps else None
        if self.last_step is None or self.last_step['history_begin'] + self.last_step['accepted'] != job.history:
            raise RuntimeError('Rolling pressure oracle requires the exact accepted-prefix publication')
        session = runner._kvmem_sessions
        if session is None or job.authority.raw_dtype != torch.float8_e4m3fn:
            raise RuntimeError('Rolling pressure byte oracle requires FP8 CPU history sessions')
        self.old_tokens = job.history - runner._kvmem_window_adapter.offsets.get(job.command.request_id, 0)
        self.old_records = {r[0]: r for r in job.authority.memory._store.working_records()}
        self.old_native = session._target_bytes(job.command.request_id, self.old_tokens).clone()
        self.width = runtime.heads * runtime.dim

    def verify(self, job):
        self.adapter.validate(self.bundle)
        if any(not torch.equal(tensor.detach().cpu(), self.before[key])
               for key, tensor in self.bundle.tensors.items()):
            raise RuntimeError('Rolling pressure changed native recurrent or acceptance state')
        target_bytes = None
        stable_tokens = 0
        if not job.cancelled:
            session = self.runner._kvmem_sessions
            if session is None or job.authority.raw_dtype != torch.float8_e4m3fn:
                raise RuntimeError('Rolling pressure byte oracle requires FP8 CPU history sessions')
            expected = session._rebuilt_target_bytes(job.command.request_id, job.authority)
            # Original forward K was rotated before its FP8 cache conversion.
            # Unmoved rows can retain those bytes; rerotating archived FP8 raw K
            # is intentionally lossy. Compare stable rows to their independent
            # pre-maintenance native snapshot and moved rows to the CPU rebuild.
            rebuilt = expected.reshape(2*len(self.runner._kvmem_capture_runtime.names),
                                       job.selected_tokens, self.width)
            old = self.old_native.reshape(rebuilt.shape[0], self.old_tokens, self.width)
            plan = job.result[2]
            for remap in plan.remaps:
                if not remap.skip:
                    continue
                record = self.old_records.get(remap.block_id)
                if (record is None or not remap.working_k_resident or remap.from_base != remap.to_base or
                        (record[3],record[4]) != (remap.n_tokens,remap.from_base) or
                        not 0 <= remap.to_base < remap.to_base+remap.n_tokens <= job.selected_tokens or
                        remap.from_base+remap.n_tokens > self.old_tokens):
                    raise RuntimeError('Rolling oracle found an invalid stable native KV remap')
                begin,end = remap.to_base,remap.to_base+remap.n_tokens
                rebuilt[:,begin:end].copy_(old[:,begin:end])
                stable_tokens += remap.n_tokens
            actual = session._target_bytes(job.command.request_id, job.selected_tokens)
            if not torch.equal(actual, expected):
                raise RuntimeError('Rolling pressure target/MTP KV differs from immutable FP8 history')
            target_bytes = actual.numel()
        row = dict(request_id=job.command.request_id, history_end=job.history,
            selected_tokens=job.selected_tokens, cancelled=job.cancelled,
            replay_tokens=job.command.groups[0][1].replay_tokens,
            recurrent_byte_equal=True, recurrent_tensors=len(self.before),
            target_kv_byte_equal=target_bytes is not None, target_kv_bytes=target_bytes,
            oracle_kind='native_stable_rows_plus_fp8_rebuild', native_stable_tokens=stable_tokens,
            attention_layers=len(self.runner._kvmem_capture_runtime.names),
            mtp_attention_layers=len(getattr(self.runner, '_kvmem_mtp_modules', {})),
            previous_accepted_step=self.last_step, private_forward_steps=job.steps)
        with (self.runner._kvmem_probe_output/'verified_rolling_pressure.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

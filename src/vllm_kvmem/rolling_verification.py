"""Expensive, opt-in byte oracle for accepted-history pressure maintenance.

This runs only with the existing session qualification switch. It deliberately
copies state to CPU and must never be used for performance measurements.
"""
import json

import torch

from .publication_step import native_publication_step


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
        self.last_step = dict(native_publication_step(steps[-1])) if steps else None
        if self.last_step is None or self.last_step['history_begin'] + self.last_step['accepted'] != job.history:
            raise RuntimeError('Rolling pressure oracle requires the exact accepted-prefix publication')
        session = runner._kvmem_sessions
        if session is None or job.authority.raw_dtype != torch.float8_e4m3fn:
            raise RuntimeError('Rolling pressure byte oracle requires FP8 CPU history sessions')
        self.old_tokens = job.history - runner._kvmem_window_adapter.offsets.get(job.command.request_id, 0)
        self.old_records = {r[0]: r for r in job.authority.memory._store.working_records()}
        self.old_native = session._target_bytes(job.command.request_id, self.old_tokens).clone()
        self.width = runtime.heads * runtime.dim
        self.draft = []
        contract = getattr(runner, '_kvmem_draft_contract', None)
        if contract is not None and contract.method == 'dflash':
            from .guided_query import snapshot_guided_draft
            self.draft = snapshot_guided_draft(runner, runner._kvmem_last_batch, contract)

    def verify(self, job):
        self.adapter.validate(self.bundle)
        if any(not torch.equal(tensor.detach().contiguous().view(torch.uint8).cpu(),
                               self.before[key].contiguous().view(torch.uint8))
               for key, tensor in self.bundle.tensors.items()):
            raise RuntimeError('Rolling pressure changed native recurrent or acceptance state')
        for request_id, name, cache, pages, offsets, expected in self.draft:
            if not torch.equal(cache[pages, offsets].contiguous().view(torch.uint8).cpu(), expected):
                raise RuntimeError('Rolling pressure changed committed draft KV: '+request_id+'/'+name)
        target_bytes = None
        stable_tokens = 0
        if not job.cancelled:
            session = self.runner._kvmem_sessions
            if session is None or job.authority.raw_dtype != torch.float8_e4m3fn:
                raise RuntimeError('Rolling pressure byte oracle requires FP8 CPU history sessions')
            layered = getattr(job, 'layerwise_pressure_plan', None)
            groups = None
            if layered is not None:
                from .layerwise_decode import record_map
                maps = [record_map(records) for records in layered[0]]
                maps += [maps[0]] * (len(self.runner._kvmem_capture_runtime.names) - len(maps))
                groups = tuple((frame, (index,)) for index, frame in enumerate(maps))
            expected = session._rebuilt_target_bytes(job.command.request_id, job.authority,
                **(dict(record_groups=groups) if groups is not None else {}))
            # Original forward K was rotated before its FP8 cache conversion.
            # Unmoved rows can retain those bytes; rerotating archived FP8 raw K
            # is intentionally lossy. Compare stable rows to their independent
            # pre-maintenance native snapshot and moved rows to the CPU rebuild.
            rebuilt = expected.reshape(2*len(self.runner._kvmem_capture_runtime.names),
                                       job.selected_tokens, self.width)
            old = self.old_native.reshape(rebuilt.shape[0], self.old_tokens, self.width)
            plan = job.result[2]
            for remap in plan.remaps:
                if layered is not None and job.command.pressure_stage == 'decode':
                    continue  # Rolling layered assembly rebuilds every layer.
                if not remap.skip:
                    continue
                record = self.old_records.get(remap.block_id)
                if (record is None or not remap.working_k_resident or remap.from_base != remap.to_base or
                        (record[3],record[4]) != (remap.n_tokens,remap.from_base) or
                        not 0 <= remap.to_base < remap.to_base+remap.n_tokens <= job.selected_tokens or
                        remap.from_base+remap.n_tokens > self.old_tokens):
                    raise RuntimeError('Rolling oracle found an invalid stable native KV remap')
                begin,end = remap.to_base,remap.to_base+remap.n_tokens
                if layered is not None:
                    # Prefill single assembly skips only layer zero and MTP.
                    n = len(layered[0])
                    rebuilt[:2,begin:end].copy_(old[:2,begin:end])
                    rebuilt[2*n:,begin:end].copy_(old[2*n:,begin:end])
                else:
                    rebuilt[:,begin:end].copy_(old[:,begin:end])
                stable_tokens += remap.n_tokens
            actual = session._target_bytes(job.command.request_id, job.selected_tokens)
            if not torch.equal(actual, expected):
                self._record_kv_mismatch(job, actual, expected, stable_tokens)
                raise RuntimeError('Rolling pressure target/MTP KV differs from immutable FP8 history')
            target_bytes = actual.numel()
        row = dict(request_id=job.command.request_id, history_end=job.history,
            selected_tokens=job.selected_tokens, cancelled=job.cancelled,
            replay_tokens=job.command.groups[0][1].replay_tokens,
            recurrent_byte_equal=True, recurrent_tensors=len(self.before),
            target_kv_byte_equal=target_bytes is not None, target_kv_bytes=target_bytes,
            oracle_kind='native_stable_rows_plus_fp8_rebuild', native_stable_tokens=stable_tokens,
            attention_layers=len(self.runner._kvmem_capture_runtime.names),
            draft_byte_equal=True, draft_layers=len(self.draft),
            mtp_attention_layers=len(getattr(self.runner, '_kvmem_mtp_modules', {})),
            previous_accepted_step=self.last_step, private_forward_steps=job.steps)
        if getattr(job, 'layerwise_pressure_plan', None) is not None:
            row['layerwise_selected_starts'] = [[r.history_begin for r in records]
                                               for records in job.layerwise_pressure_plan[0]]
        with (self.runner._kvmem_probe_output/'verified_rolling_pressure.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def _record_kv_mismatch(self, job, actual, expected, stable_tokens):
        """Keep bounded failure evidence without weakening the byte oracle."""
        names = self.runner._kvmem_capture_runtime.names
        shape = (2 * len(names), job.selected_tokens, self.width)
        actual, expected = actual.reshape(shape), expected.reshape(shape)
        old = self.old_native.reshape(shape[0], self.old_tokens, self.width)
        remaps = list(job.result[2].remaps)
        failures = []
        for index in range(shape[0]):
            different = actual[index] != expected[index]
            count = int(different.sum())
            if not count:
                continue
            # Inspect one small row at a time: do not materialize coordinates
            # for every corrupt byte in a full working window.
            tokens = different.any(-1).nonzero().flatten()
            examples = []
            for token in tokens[:4].tolist():
                columns = different[token].nonzero().flatten()[:4].tolist()
                remap = next((r for r in remaps if r.to_base <= token < r.to_base + r.n_tokens), None)
                source = (remap.from_base + token - remap.to_base
                          if remap is not None and remap.from_base >= 0 else None)
                examples.append(dict(token=token, columns=columns,
                    actual=actual[index, token, columns].tolist(),
                    expected=expected[index, token, columns].tolist(),
                    old_native=old[index, source, columns].tolist()
                        if source is not None and source < self.old_tokens else None,
                    remap={key: getattr(remap, key) for key in
                        ('block_id', 'skip', 'working_k_resident', 'from_base', 'to_base', 'n_tokens')}
                        if remap is not None else None))
            failures.append(dict(layer=names[index // 2], kind='key' if index % 2 == 0 else 'value',
                mismatch_bytes=count, mismatch_tokens=tokens.numel(), examples=examples))
        row = dict(request_id=job.command.request_id, history_end=job.history,
            selected_tokens=job.selected_tokens, old_tokens=self.old_tokens,
            stable_tokens=stable_tokens, previous_accepted_step=self.last_step,
            remap_count=len(remaps), remap_token_counts=sorted({r.n_tokens for r in remaps}),
            failures=failures)
        with (self.runner._kvmem_probe_output/'rolling_pressure_mismatch.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

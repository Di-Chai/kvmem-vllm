"""Worker-side prompt checkpoint capture, import and fenced retirement.

Import runs after native request/table initialization and block zeroing, but
before attention metadata is built. Native sampling and draft proposal retain
their normal per-request lifecycle. CPU checkpoint ownership is bounded separately
from the shared authority arena; the runner must reserve room for idle authorities.
"""
import hashlib
import os
import time

import torch

from .authority import _Ready
from .prompt_checkpoints import PromptCheckpointPool
from .session_control import SessionNotice
from .session_state import PromptStateAdapter


def verify_preserved_authority(authority, timeout=180):
    """Verify all still-retained sealed records without stopping live GC.

    Pin the complete foreground snapshot before reading. GC tombstones can race
    initial pin acquisition, so discard that attempt and retry after draining.
    Holding those leases makes checksum iteration immutable without changing the
    original background policy or counting collected records as corrupt misses.
    """
    deadline = time.monotonic() + timeout
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('Preserved authority verification did not drain')
        return value
    while not authority.flush():
        for job, _ in authority.writes:
            job.result(timeout=remaining())
        if not authority.archive.wait_for_capacity(remaining()):
            raise TimeoutError('Preserved authority publication timed out')
    for job, _ in authority.writes:
        job.result(timeout=remaining())
    while True:
        remaining()
        authority.poll()
        records = list(authority.records.values())
        leases = [authority.memory.repository.pin(record.handle) for record in records]
        if all(leases):
            break
        for lease in leases:
            lease.reset()
    try:
        archived = sum(record.archived for record in records)
        if archived + authority.gc_reclaimed_records != authority.history_end // authority.shape[1]:
            raise RuntimeError('Preserved authority has unpublished sealed records')
        verified = 0
        for record in records:
            if not record.archived:
                continue
            while True:
                try:
                    raw, values = authority.read_record(record.handle)
                    break
                except BlockingIOError:
                    if not authority.archive.wait_for_capacity(remaining()):
                        raise TimeoutError('Preserved authority checksum read timed out')
            digest = hashlib.sha256(raw.view(torch.uint8).numpy().tobytes() + values.numpy().tobytes()).hexdigest()
            if digest != record.digest:
                raise RuntimeError('Preserved authority checksum differs from accepted capture')
            verified += 1
        return verified
    finally:
        for lease in leases:
            lease.reset()


class SessionWorker:
    def __init__(self, runner, options):
        self.runner, self.options = runner, options
        self.pool = PromptCheckpointPool(options.session_capacity, options.session_checkpoint_bytes)
        self.bindings = {}
        self.identities = {}
        self.imports = set()
        self.captures = {}
        self.retiring = {}
        self.evictions = {}
        self.completed = []
        self.model_adapter = None
        self.terminal_captures = {}
        self.restore_oracles = {}
        self.checkpoint_deferrals = 0
        self.restored_tokens = 0

    @property
    def runtime(self):
        return self.runner._kvmem_capture_runtime

    def receive(self, commands):
        for binding in commands.bindings:
            if binding.request_id in self.bindings:
                raise ValueError('Duplicate worker session binding')
            old = self.identities.get(binding.namespace)
            if old is not None and old != binding.generation:
                raise ValueError('Session namespace reused before worker reclamation')
            if bool(binding.history) != (binding.namespace in self.pool.entries):
                raise ValueError('Worker checkpoint and scheduler import disagree')
            self.identities[binding.namespace] = binding.generation
            self.bindings[binding.request_id] = binding
            if binding.history:
                self.imports.add(binding.request_id)
        for capture in commands.captures:
            binding = self.bindings.get(capture.request_id)
            if binding is None or binding.generation != capture.generation or capture.request_id in self.captures:
                raise ValueError('Stale or duplicate worker checkpoint command')
            self.captures[capture.request_id] = capture
        for capture in getattr(commands, 'terminals', ()):
            binding = self.bindings.get(capture.request_id)
            if binding is None or binding.generation != capture.generation or capture.request_id in self.terminal_captures:
                raise ValueError('Stale or duplicate terminal checkpoint command')
            self.terminal_captures[capture.request_id] = capture

    def before_attention(self, batch):
        selected = self.imports.intersection(batch.req_ids) | (self.captures.keys() & set(batch.req_ids))
        if not selected:
            return
        if self.runtime is None:
            raise RuntimeError('Prompt checkpoint has no accepted authority runtime')
        self.runtime.drain(wait=True)
        if self.model_adapter is None:
            self.model_adapter = PromptStateAdapter(self.runner)
        for request_id in batch.req_ids:
            if request_id in self.imports:
                self._restore(request_id)
                self.imports.remove(request_id)
            command = self.captures.pop(request_id, None)
            if command is None:
                continue
            binding = self.bindings[request_id]
            authority = self.runtime.authorities[request_id]
            if authority.history_end != command.position:
                raise ValueError('Prompt capture differs from accepted archive history')
            try:
                self.pool.capture(binding.namespace, request_id, authority,
                    command.tokens or binding.tokens, self.model_adapter,
                    replace=getattr(self.options, 'session_capture_decode', False))
            except BufferError:
                # Cache admission failure preserves ordinary native execution.
                self.checkpoint_deferrals += 1
                if getattr(self.options, 'session_capture_decode', False):
                    self.completed.append(SessionNotice('capture_deferred', binding.namespace,
                        binding.generation, request_id, command.position))
            else:
                if getattr(self.options, 'session_capture_decode', False):
                    self.completed.append(SessionNotice('captured', binding.namespace,
                        binding.generation, request_id, command.position))

    def _target_bytes(self, request_id, tokens):
        """Opt-in real-model restoration oracle; unused in ordinary serving."""
        from .window_assembly import NativeWindowAssembler
        assembler = NativeWindowAssembler.from_runner(self.runner, request_id, self.runtime.names,
                                                       self.options.memory.block_tokens)
        try:
            positions = torch.arange(tokens, device=self.runner.device)
            rows = []
            for layer in assembler.layers:
                block = layer.key.shape[1]
                pages = layer.kernel_page_table[positions // block].long()
                offsets = positions % block
                rows.extend((layer.key[pages, offsets].view(torch.uint8).flatten(),
                             layer.value[pages, offsets].view(torch.uint8).flatten()))
            return torch.cat(rows).cpu()
        finally:
            assembler.close()

    def _restore(self, request_id):
        from .cache_groups import full_attention_groups
        from .window_assembly import NativeWindowAssembler
        from .window_batch import WindowBatchAdapter
        binding = self.bindings[request_id]
        checkpoint = self.pool.claim(binding.namespace, request_id, binding.tokens)
        if (checkpoint is None or checkpoint.position != binding.history or
                checkpoint.window.window_tokens != binding.window):
            raise ValueError('Worker cannot restore the advertised exact prompt prefix')
        if request_id in self.runtime.authorities:
            raise ValueError('Native import already owns a different authority')
        authority = checkpoint.authority
        cfg = self.options.memory
        assembler = NativeWindowAssembler.from_runner(self.runner, request_id, self.runtime.names,
            cfg.block_tokens, packed=self.options.packed_assembly,
            direct_pinned_reads=self.options.direct_pinned_reads)
        plan = self.pool.prepare_claim_window(checkpoint)
        try:
            assembly = assembler.assemble(authority, plan)
            assembly.event.synchronize()
            authority.memory.attach_completion(assembly)
            assert authority.memory.commit_selection()
            event = self.model_adapter.restore(checkpoint.model, request_id)
            if event is not None:
                event.synchronize()
            oracle = getattr(self, 'restore_oracles', {}).pop(binding.namespace, None)
            if oracle is not None:
                actual = self._target_bytes(request_id, checkpoint.window.window_tokens)
                if not torch.equal(actual, oracle):
                    raise RuntimeError('Restored target KV differs from the native terminal window')
                recurrent = self.model_adapter._recurrent_views(request_id)
                draft = self.model_adapter._draft_views(request_id, checkpoint.model.draft_begin,
                                                       checkpoint.position)
                if (any(not torch.equal(t.cpu(), checkpoint.model.recurrent[name]) for name, t in recurrent.items())
                        or any(not torch.equal(cache[pages, offsets].cpu(), checkpoint.model.draft[name])
                               for name, (cache, pages, offsets) in draft.items())):
                    raise RuntimeError('Restored GDN or draft state differs from its terminal checkpoint')
                import json
                with (self.runner._kvmem_probe_output / 'verified_restores.jsonl').open('a') as stream:
                    stream.write(json.dumps({'request_id': request_id, 'history_tokens': binding.history,
                        'working_tokens': checkpoint.window.window_tokens, 'target_kv_bytes': oracle.numel(),
                        'working_attention_layers': len(self.runtime.names),
                        'mtp_attention_layers': len(getattr(self.runner, '_kvmem_mtp_modules', {})),
                        'recurrent_tensors': len(recurrent), 'draft_layers': len(draft), 'byte_equal': True}) + '\n')
            restored = self.pool.finish_claim(checkpoint, event or _Ready())
            assert restored is authority
        except BaseException:
            # Import failure is fatal to this worker. Retain all state/page
            # owners through completed device work before normal shutdown cleanup.
            if self.runner.device.type == 'cuda':
                torch.cuda.synchronize(self.runner.device)
            self.pool.evict(checkpoint, _Ready())
            raise
        finally:
            assembler.close()
        if self.runner._kvmem_window_adapter is None:
            self.runner._kvmem_window_adapter = WindowBatchAdapter(
                full_attention_groups(self.runner.kv_cache_config), cfg.select_budget + cfg.gen_budget)
        self.runner._kvmem_window_adapter.commit(request_id, binding.history, binding.window)
        self.runtime.authorities[request_id] = authority
        self.runtime.accepted[request_id] = binding.history
        self.runtime.rejected[request_id] = 0
        self.runtime.last_steps[request_id] = []
        if authority.gc is not None:
            authority.memory._store.gc_begin_step()
        self.restored_tokens += binding.history
        self.completed.append(SessionNotice('restored', binding.namespace, binding.generation,
                                            request_id, binding.history, binding.window))

    def detach(self, request_id, authority, report):
        binding = self.bindings.get(request_id)
        if binding is None:
            return False
        command = getattr(self, 'terminal_captures', {}).pop(request_id, None)
        if command is not None:
            if command.position != authority.history_end:
                raise ValueError('Terminal checkpoint differs from the accepted history')
            if self.model_adapter is None:
                self.model_adapter = PromptStateAdapter(self.runner)
            try:
                self.pool.capture(binding.namespace, request_id, authority, command.tokens,
                                  self.model_adapter, replace=True, terminal=True)
            except BufferError:
                self.checkpoint_deferrals += 1
                kind = 'capture_deferred'
            else:
                kind = 'captured'
            self.completed.append(SessionNotice(kind, binding.namespace, binding.generation,
                                                request_id, command.position))
        checkpoint = self.pool.entries.get(binding.namespace)
        if checkpoint is None:
            return False  # Ordinary retirement will emit an empty checkpoint acknowledgment.
        if checkpoint.request_id != request_id or checkpoint.authority is not authority:
            raise ValueError('Terminal request does not own the retained prompt snapshot')
        if command is not None and kind == 'captured' and os.getenv('KVMEM_VERIFY_SESSION_RESTORE') == '1':
            self.restore_oracles[binding.namespace] = self._target_bytes(request_id, checkpoint.window.window_tokens)
        verified = verify_preserved_authority(authority) if self.options.verify_archive_on_finish else 0
        self.pool.retire(checkpoint, _Ready())  # Capture downloads and CPU state copies have drained.
        self.retiring[request_id] = (checkpoint, {**report, 'history_end': authority.history_end,
            'checkpoint_tokens': checkpoint.position, 'sealed_records_verified': verified,
            'retirement_strategy': 'terminal_checkpoint' if command is not None else 'checkpoint_rewind',
            'authority_namespace': authority.memory.request_id})
        return True

    def retirement_done(self, request_id):
        pending = self.retiring.get(request_id)
        if pending is None:
            return None
        checkpoint, report = pending
        if not self.pool.idle(checkpoint):
            return False
        binding = self.bindings.pop(request_id)
        self.completed.append(SessionNotice('retired', binding.namespace, binding.generation,
            request_id, checkpoint.position, checkpoint.window.window_tokens,
            checkpoint.authority.memory.repository.ready_slots))
        self.runtime.completed.append({**report, 'retirement_ack_ready_at': time.perf_counter()})
        del self.retiring[request_id]
        self.runtime._write_archive_report()
        return True

    def ordinary_retired(self, request_id):
        binding = self.bindings.pop(request_id, None)
        if binding is not None:
            self.completed.append(SessionNotice('retired', binding.namespace, binding.generation, request_id))
            del self.identities[binding.namespace]

    def evict(self, commands):
        for command in commands:
            if self.identities.get(command.namespace) != command.generation or command.namespace in self.evictions:
                raise ValueError('Stale or duplicate session eviction')
            checkpoint = self.pool.entries[command.namespace]
            self.evictions[command.namespace] = command
            self.pool.evict(checkpoint)
        self._poll_evictions()

    def _poll_evictions(self):
        changed = False
        for namespace, command in list(self.evictions.items()):
            checkpoint = self.pool.entries.get(namespace)
            if checkpoint is not None and not self.pool.poll_eviction(checkpoint):
                continue
            self.completed.append(SessionNotice('evicted', namespace, command.generation, ''))
            del self.evictions[namespace]
            del self.identities[namespace]
            getattr(self, 'restore_oracles', {}).pop(namespace, None)
            changed = True
        if changed and self.runtime is not None:
            self.runtime._write_archive_report()

    def notices(self):
        self._poll_evictions()
        result, self.completed = self.completed, []
        return result

    def close(self):
        from .prompt_checkpoints import CheckpointPhase
        for checkpoint in list(self.pool.entries.values()):
            if checkpoint.phase is CheckpointPhase.LIVE:
                self.pool.drop_live(checkpoint)  # Runtime still owns active authority.
            elif checkpoint.phase is CheckpointPhase.CLOSING:
                self.pool.poll_eviction(checkpoint)
            else:
                self.pool.evict(checkpoint, _Ready() if checkpoint.phase is CheckpointPhase.CLAIMED else None)
        if self.pool.entries:
            return False
        self.bindings.clear()
        self.identities.clear()
        self.imports.clear()
        self.captures.clear()
        getattr(self, 'terminal_captures', {}).clear()
        getattr(self, 'restore_oracles', {}).clear()
        self.retiring.clear()
        self.evictions.clear()
        return True

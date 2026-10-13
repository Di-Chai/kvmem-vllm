"""Version-isolated runner extension. Probe mode does not enable KVMem serving.

The base runner continues all scheduling, projections, attention and sampling.
This stage verifies actual target-layer capture on a real native batch.
"""
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from .cache_groups import full_attention_groups
from .capture import capture_to, install_qwen_capture


def make_probe_runner(base):
    class KVMemProbeRunner(base):
        def load_model(self, *args, **kwargs):
            super().load_model(*args, **kwargs)
            from .draft_quantization import install_quantized_context
            install_quantized_context(self)
            self._kvmem_graph = os.getenv('KVMEM_FULL_GRAPH') == '1'
            if not self.model_config.enforce_eager and not self._kvmem_graph:
                raise ValueError("Runner capture probe currently requires enforce_eager")
            if self._kvmem_graph:
                from vllm.config import CUDAGraphMode
                if (self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL or
                        int(self.vllm_config.compilation_config.mode) != 0 or
                        self.vllm_config.max_concurrent_batches != 1):
                    raise ValueError('KVMem graph capture requires FULL, compilation NONE, synchronous batches')
            if self.vllm_config.parallel_config.tensor_parallel_size != 1:
                raise ValueError("Probe currently qualifies single-rank execution only")
            self._kvmem_layers = install_qwen_capture(self.model)
            self._kvmem_serving = os.getenv('KVMEM_SERVING') == '1'
            self._kvmem_probe_done = self._kvmem_serving
            self._kvmem_probe_output = Path(os.environ["KVMEM_RUNNER_PROBE_DIR"])
            self._kvmem_scheduled = os.getenv("KVMEM_SCHEDULED_REBUILD_PROBE") == "1"
            self._kvmem_bounded_target = os.getenv("KVMEM_BOUNDED_TARGET") == "1"
            from .options import RuntimeOptions
            options = RuntimeOptions.from_vllm(self.vllm_config)
            self._kvmem_repository_gc = options.repository_gc
            self._kvmem_prepared_archive = None
            if options.nvme_preallocate or options.nvme_max_bytes or options.archive_io_max_bytes:
                from .storage import make_shared_archive
                modules = dict(self.model.named_modules())
                layers = [modules[name] for name in self._kvmem_layers]
                geometry = {(layer.num_kv_heads, layer.head_dim) for layer in layers}
                if len(geometry) != 1:
                    raise ValueError('NVMe startup reservation requires uniform target KV geometry')
                heads, dim = geometry.pop()
                record_bytes = len(layers) * heads * dim * options.memory.block_tokens * 3
                self._kvmem_prepared_archive = make_shared_archive(options,
                    self._kvmem_probe_output / 'continuous_archive', record_bytes,
                    self.vllm_config.scheduler_config.max_num_seqs if self._kvmem_bounded_target else 1,
                    max_query_tokens=self.vllm_config.scheduler_config.max_num_batched_tokens)
            self._kvmem_saved_capture = None
            self._kvmem_window_adapter = None
            self._kvmem_window_batch = None
            self._kvmem_continuous = os.getenv("KVMEM_ARCHIVE_PROBE") == "1"
            self._kvmem_capture_runtime = None
            self._kvmem_pending_capture = None
            self._kvmem_sessions = None
            if options.session_capacity:
                if not self._kvmem_bounded_target or not self._kvmem_continuous:
                    raise ValueError('Prompt sessions require bounded continuous authority capture')
                from .session_worker import SessionWorker
                self._kvmem_sessions = SessionWorker(self, options)
            self._kvmem_request_generations = {}
            self._kvmem_next_generation = 0
            if self._kvmem_bounded_target:
                if any(os.getenv(flag) == "1" for flag in (
                        "KVMEM_REPLAY_PROBE", "KVMEM_COMPRESSION_PROBE", "KVMEM_REBUILD_PROBE", "KVMEM_SCHEDULED_REBUILD_PROBE")):
                    raise ValueError("Bounded target execution requires its own scheduled refresh lifecycle")
                from .bounded_runtime import install_bounded_target
                install_bounded_target(self)
            if os.getenv("KVMEM_REPLAY_PROBE") == "1":
                from .replay_probe import install_replay_probe
                install_replay_probe(self)
            if os.getenv("KVMEM_COMPRESSION_PROBE") == "1":
                if os.getenv("KVMEM_REPLAY_PROBE") == "1" or not self._kvmem_continuous:
                    raise ValueError("Compression probe requires its own replay and continuous authority")
                from .compression_probe import install_compression_probe
                install_compression_probe(self)
            if os.getenv("KVMEM_WINDOW_METADATA") == "1":
                from .window_bridge import install_window_bridge
                def current_window(batch):
                    if batch is getattr(self, '_kvmem_last_batch', None):
                        return self._kvmem_window_batch
                    if self._kvmem_graph and self._kvmem_window_adapter is not None:
                        return self._kvmem_window_adapter.capture_window(batch)
                    return None
                install_window_bridge(self.model_state, current_window)
            if os.getenv("KVMEM_REBUILD_PROBE") == "1" and self.speculative_config is not None:
                raise ValueError("Rebuild probe needs target-only qualification before draft lifecycle integration")
            from .profile_scopes import install_profile_scopes
            install_profile_scopes(self)

        def add_requests(self, scheduler_output):
            result = super().add_requests(scheduler_output)
            for request in scheduler_output.scheduled_new_reqs:
                self._kvmem_next_generation += 1
                self._kvmem_request_generations[request.req_id] = self._kvmem_next_generation
            return result

        def initialize_kv_cache(self, *args, **kwargs):
            result = super().initialize_kv_cache(*args, **kwargs)
            if getattr(self, '_kvmem_graph', False):
                from .graph_runtime import initialize_graph_window
                initialize_graph_window(self)
            from .profile_scopes import install_profile_scopes
            install_profile_scopes(self)
            return result

        def finish_requests(self, scheduler_output):
            result = super().finish_requests(scheduler_output)
            for request_id in scheduler_output.finished_req_ids:
                self._kvmem_request_generations.pop(request_id, None)
            return result

        def shutdown(self):
            try:
                runtime = getattr(self, '_kvmem_capture_runtime', None)
                if runtime is not None:
                    runtime.shutdown_retirement()
                else:
                    archive = getattr(self, '_kvmem_prepared_archive', None)
                    if archive is not None and not archive.close():
                        raise RuntimeError('Unused startup archive retained I/O ownership')
            finally:
                super().shutdown()

        def _kvmem_refresh(self, command):
            from .refresh_probe import rebuild_native_batch
            captured, request_id = self._kvmem_saved_capture
            if (command.request_id != request_id or
                    self._kvmem_last_batch.req_ids != [request_id]):
                raise ValueError("Scheduled probe refers to a different request")
            if any(ticket.history_end != self._kvmem_last_batch.num_tokens
                   or ticket.window_tokens != ticket.history_end
                   for _, ticket in command.groups):
                raise ValueError("Scheduled probe only qualifies initial identity refresh")
            report = rebuild_native_batch(self, captured, self._kvmem_probe_output)
            if not report["all_byte_equal"]:
                raise RuntimeError("Scheduled refresh changed native KV bytes")
            from .options import RuntimeOptions
            from .window_batch import WindowBatchAdapter
            if os.getenv("KVMEM_WINDOW_METADATA") != "1":
                raise RuntimeError("Scheduled refresh requires the explicit window metadata hook")
            if self._kvmem_window_adapter is None:
                config = RuntimeOptions.from_vllm(self.vllm_config).memory
                self._kvmem_window_adapter = WindowBatchAdapter(
                    tuple(group for group, _ in command.groups), config.select_budget + config.gen_budget)
            extents = {(ticket.history_end, ticket.window_tokens) for _, ticket in command.groups}
            if len(extents) != 1:
                raise ValueError("All full-attention groups must commit the same dense window")
            history_end, window_tokens = extents.pop()
            self._kvmem_window_adapter.commit(command.request_id, history_end, window_tokens)
            report["command_id"] = command.command_id
            report["request_id"] = command.request_id
            path = self._kvmem_probe_output / "capture.json"
            capture_report = json.loads(path.read_text())
            capture_report["scheduled_rebuild"] = report
            path.write_text(json.dumps(capture_report, indent=2))
            self._kvmem_saved_capture = None
            event = torch.cuda.Event()
            event.record()
            return event

        def prepare_attn(self, input_batch):
            sessions = getattr(self, '_kvmem_sessions', None)
            if sessions is not None:
                sessions.before_attention(input_batch)
            if getattr(self, '_kvmem_bounded_target', False):
                from .native_state import canonicalize_before_non_speculative_batch
                canonicalize_before_non_speculative_batch(self, input_batch)
            self._kvmem_last_batch = input_batch
            self._kvmem_window_batch = None
            if self._kvmem_window_adapter is not None:
                window = self._kvmem_window_adapter.prepare(input_batch)
                self._kvmem_window_batch = window
                if window is not None:
                    active_groups = set(full_attention_groups(self.kv_cache_config))
                    if active_groups != set(window.groups):
                        raise ValueError("Window mapping must cover exactly the full-attention cache groups")
                    return super().prepare_attn(replace(input_batch, positions=window.positions))
            return super().prepare_attn(input_batch)

        def sample(self, hidden_states, input_batch, grammar_output):
            result = super().sample(hidden_states, input_batch, grammar_output)
            pending = getattr(self, "_kvmem_pending_capture", None)
            if pending is not None:
                captured, original_batch = pending
                if input_batch is not original_batch:
                    raise ValueError("Continuous authority requires the native unsharded target batch")
                self._kvmem_capture_runtime.submit(captured, input_batch, result[2])
                self._kvmem_pending_capture = None
            return result

        def execute_model(self, scheduler_output, intermediate_tensors=None,
                          dummy_run=False, skip_attn_for_dummy_run=False,
                          is_profile=False, context_len=0, valid_dummy_state_slots=False):
            kwargs = dict(intermediate_tensors=intermediate_tensors, dummy_run=dummy_run,
                          skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                          is_profile=is_profile, context_len=context_len,
                          valid_dummy_state_slots=valid_dummy_state_slots)
            runtime = getattr(self, "_kvmem_capture_runtime", None)
            sessions = getattr(self, '_kvmem_sessions', None)
            if not dummy_run and getattr(self, '_kvmem_serving', False):
                from .connector import ControlMetadata
                if isinstance(scheduler_output.kv_connector_metadata, ControlMetadata):
                    # Native warmup disables the connector and has no metadata.
                    # Arm at the first real scheduler message, before any client
                    # forward; API readiness polling must not create a capture gap.
                    if not (self._kvmem_probe_output / 'armed').is_file():
                        (self._kvmem_probe_output / 'armed').touch()
            # V2 kernel warmup invokes ordinary execute_model with synthetic
            # requests and no connector metadata. Session control starts at the
            # same explicit post-warmup boundary as accepted authority capture.
            if (not dummy_run and sessions is not None
                    and (self._kvmem_probe_output / "armed").is_file()):
                from .connector import ControlMetadata
                control = scheduler_output.kv_connector_metadata
                if not isinstance(control, ControlMetadata):
                    raise RuntimeError('Prompt sessions require native connector metadata')
                sessions.receive(control.sessions)
            if not dummy_run and runtime is not None:
                terminal = None
                if getattr(self, '_kvmem_bounded_target', False):
                    from .connector import ControlMetadata
                    control = scheduler_output.kv_connector_metadata
                    if not isinstance(control, ControlMetadata):
                        raise RuntimeError('Bounded capture requires scheduler control metadata')
                    terminal = control.terminal_prefixes
                # Apply EOS/cancellation truncation before draining the latest
                # accepted sample packet into the immutable authority.
                runtime.finish(scheduler_output.finished_req_ids, terminal)
                runtime.drain(wait=False)
            if not dummy_run and getattr(self, "_kvmem_window_adapter", None) is not None:
                for request_id in scheduler_output.finished_req_ids:
                    self._kvmem_window_adapter.free(request_id)
            if (getattr(self, "_kvmem_scheduled", False) and not dummy_run
                    and (self._kvmem_probe_output / "armed").is_file()):
                from vllm.distributed.kv_transfer import get_kv_transfer_group
                get_kv_transfer_group().runtime = self._kvmem_refresh
            if (getattr(self, "_kvmem_bounded_target", False) and not dummy_run
                    and (self._kvmem_probe_output / "armed").is_file()):
                from vllm.distributed.kv_transfer import get_kv_transfer_group
                connector = get_kv_transfer_group()
                connector.session_worker = sessions
                connector.runtime = self._kvmem_query_refresh
                connector.retirement_done = (runtime.retirement_done if runtime is not None else lambda _: True)
                if getattr(self, '_kvmem_repository_gc', False):
                    connector.archive_snapshot = lambda: (
                        self._kvmem_capture_runtime.archive_capacity()
                        if self._kvmem_capture_runtime is not None else {})
            if (dummy_run or
                    not scheduler_output.total_num_scheduled_tokens or
                    (getattr(self, "_kvmem_probe_done", True) and not self._kvmem_continuous) or
                    not (self._kvmem_probe_output / "armed").is_file()):
                return super().execute_model(scheduler_output, **kwargs)
            captured = []
            if getattr(self, '_kvmem_graph', False):
                from .graph_runtime import needs_refresh_batch
                if needs_refresh_batch(scheduler_output):
                    # Native dispatch uses need_eager for this flag. The real
                    # batch still executes all GDN groups and normal sampling.
                    kwargs['is_profile'] = True
            with capture_to(captured.append):
                result = super().execute_model(scheduler_output, **kwargs)
            if getattr(self, '_kvmem_graph', False):
                captured = self._kvmem_graph_captures
            if not captured:
                raise RuntimeError("Real native batch executed without any KVMem captures")
            if self._kvmem_continuous:
                if self._kvmem_pending_capture is not None:
                    raise RuntimeError("Previous target capture never reached native sampling")
                if self._kvmem_capture_runtime is None:
                    from .capture_runtime import CaptureRuntime
                    self._kvmem_capture_runtime = CaptureRuntime(self, self._kvmem_probe_output)
                self._kvmem_pending_capture = (captured, self._kvmem_last_batch)
            if self._kvmem_probe_done:
                return result
            self._kvmem_probe_done = True
            if self._kvmem_scheduled:
                if self._kvmem_last_batch.num_reqs != 1:
                    raise ValueError("First scheduled probe must be a single request")
                self._kvmem_saved_capture = (captured, self._kvmem_last_batch.req_ids[0])
            output = self._kvmem_probe_output
            output.mkdir(parents=True, exist_ok=True)
            rebuild = None
            if os.getenv("KVMEM_REBUILD_PROBE") == "1":
                from .refresh_probe import rebuild_native_batch
                rebuild = rebuild_native_batch(self, captured, output)
            # Diagnostic-only synchronized export, once after the first real batch.
            payload = {c.name: {"positions": c.positions.cpu(), "q": c.query.cpu(),
                                "k": c.key.cpu(), "v": c.value.cpu()}
                       for c in captured}
            torch.save(payload, output / "target_capture.pt")
            report = {"scope": "native target capture; refresh qualification is recorded separately",
                      "layers_expected": self._kvmem_layers,
                      "layers_captured": list(payload),
                      "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                      "requests": scheduler_output.num_scheduled_tokens,
                      "rebuild": rebuild,
                      "finite": all(torch.isfinite(t).all().item() for c in payload.values() for t in c.values())}
            (output / "capture.json").write_text(json.dumps(report, indent=2))
            if report["layers_captured"] != self._kvmem_layers or not report["finite"]:
                raise RuntimeError("Incomplete or non-finite native target capture")
            if rebuild is not None and not rebuild["all_byte_equal"]:
                raise RuntimeError("Native-page archive/rebuild differs from original cache bytes")
            return result

    return KVMemProbeRunner

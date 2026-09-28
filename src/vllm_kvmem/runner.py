"""Version-isolated runner extension. Probe mode does not enable KVMem serving.

The base runner continues all scheduling, projections, attention and sampling.
This stage verifies actual target-layer capture on a real native batch.
"""
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

from .capture import capture_to, install_qwen_capture


def make_probe_runner(base):
    class KVMemProbeRunner(base):
        def load_model(self, *args, **kwargs):
            super().load_model(*args, **kwargs)
            if not self.model_config.enforce_eager:
                raise ValueError("Runner capture probe currently requires enforce_eager")
            if self.vllm_config.parallel_config.tensor_parallel_size != 1:
                raise ValueError("Probe currently qualifies single-rank execution only")
            self._kvmem_layers = install_qwen_capture(self.model)
            self._kvmem_probe_done = False
            self._kvmem_probe_output = Path(os.environ["KVMEM_RUNNER_PROBE_DIR"])
            self._kvmem_scheduled = os.getenv("KVMEM_SCHEDULED_REBUILD_PROBE") == "1"
            self._kvmem_saved_capture = None
            self._kvmem_window_adapter = None
            self._kvmem_window_batch = None
            if os.getenv("KVMEM_WINDOW_METADATA") == "1":
                from .window_bridge import install_window_bridge
                install_window_bridge(self.model_state, lambda batch:
                    self._kvmem_window_batch if batch is getattr(self, "_kvmem_last_batch", None) else None)
            if os.getenv("KVMEM_REBUILD_PROBE") == "1" and self.speculative_config is not None:
                raise ValueError("Rebuild probe needs target-only qualification before draft lifecycle integration")

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
            from .config import KVMemConfig
            from .window_batch import WindowBatchAdapter
            if os.getenv("KVMEM_WINDOW_METADATA") != "1":
                raise RuntimeError("Scheduled refresh requires the explicit window metadata hook")
            if self._kvmem_window_adapter is None:
                config = KVMemConfig()
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
            self._kvmem_last_batch = input_batch
            self._kvmem_window_batch = None
            if self._kvmem_window_adapter is not None:
                window = self._kvmem_window_adapter.prepare(input_batch)
                self._kvmem_window_batch = window
                if window is not None:
                    active_groups = {i for i, enabled in enumerate(self.block_tables._slot_mapping_enabled) if enabled}
                    if active_groups != set(window.groups):
                        raise ValueError("Window mapping must cover exactly the slot-mapped attention groups")
                    return super().prepare_attn(replace(input_batch, positions=window.positions))
            return super().prepare_attn(input_batch)

        def execute_model(self, scheduler_output, intermediate_tensors=None,
                          dummy_run=False, skip_attn_for_dummy_run=False,
                          is_profile=False, context_len=0, valid_dummy_state_slots=False):
            kwargs = dict(intermediate_tensors=intermediate_tensors, dummy_run=dummy_run,
                          skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                          is_profile=is_profile, context_len=context_len,
                          valid_dummy_state_slots=valid_dummy_state_slots)
            if not dummy_run and getattr(self, "_kvmem_window_adapter", None) is not None:
                for request_id in scheduler_output.finished_req_ids:
                    self._kvmem_window_adapter.free(request_id)
            if (getattr(self, "_kvmem_scheduled", False) and not dummy_run
                    and (self._kvmem_probe_output / "armed").is_file()):
                from vllm.distributed.kv_transfer import get_kv_transfer_group
                get_kv_transfer_group().runtime = self._kvmem_refresh
            if (dummy_run or
                    not scheduler_output.total_num_scheduled_tokens or
                    getattr(self, "_kvmem_probe_done", True) or
                    not (self._kvmem_probe_output / "armed").is_file()):
                return super().execute_model(scheduler_output, **kwargs)
            captured = []
            with capture_to(captured.append):
                result = super().execute_model(scheduler_output, **kwargs)
            if not captured:
                raise RuntimeError("Real native batch executed without any KVMem captures")
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
            report = {"scope": "native target capture probe; no reselection enabled",
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

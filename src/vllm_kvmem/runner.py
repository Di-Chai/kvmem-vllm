"""Version-isolated runner extension. Probe mode does not enable KVMem serving.

The base runner continues all scheduling, projections, attention and sampling.
This stage verifies actual target-layer capture on a real native batch.
"""
import json
import os
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
            if os.getenv("KVMEM_REBUILD_PROBE") == "1" and self.speculative_config is not None:
                raise ValueError("Rebuild probe needs target-only qualification before draft lifecycle integration")

        def prepare_attn(self, input_batch):
            self._kvmem_last_batch = input_batch
            return super().prepare_attn(input_batch)

        def execute_model(self, scheduler_output, intermediate_tensors=None,
                          dummy_run=False, skip_attn_for_dummy_run=False,
                          is_profile=False, context_len=0, valid_dummy_state_slots=False):
            kwargs = dict(intermediate_tensors=intermediate_tensors, dummy_run=dummy_run,
                          skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                          is_profile=is_profile, context_len=context_len,
                          valid_dummy_state_slots=valid_dummy_state_slots)
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

"""Native recurrent checkpoint and two-pass forward qualification.

The probe pass does not sample or publish KV history. Restore all requests in the
batch before the real pass, so unselected neighbours cannot advance their SSM
state twice. A future semantic refresh inserts selection/assembly between passes.
"""
import json
from types import MethodType

import torch

from .capture import capture_to
from .native_state import NativeRecurrentState
from .state import StateCheckpoints


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, (tuple, list)):
        return [_clone(v) for v in value]
    raise TypeError(f"Unqualified native model output {type(value)}")


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
                and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
    return isinstance(b, (tuple, list)) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))


def install_replay_probe(runner):
    model = runner.model
    original = model.forward
    reports = []
    seen = set()
    state_adapter = None

    def forward(self, *args, **kwargs):
        nonlocal state_adapter
        batch = getattr(runner, "_kvmem_last_batch", None)
        if (batch is None or not (runner._kvmem_probe_output / "armed").is_file()
                or not batch.num_reqs):
            return original(*args, **kwargs)
        if batch.num_reqs == 4:
            case = "four_requests"
        elif batch.num_reqs == 1 and batch.num_computed_tokens_np[0] == 0:
            case = "initial_prefill"
        elif batch.num_reqs == 1 and not batch.is_prefilling_np[0]:
            case = "decode"
        else:
            return original(*args, **kwargs)
        if case in seen:
            return original(*args, **kwargs)
        if state_adapter is None:
            state_adapter = NativeRecurrentState(runner)
        bundle = state_adapter.borrow(batch.req_ids)
        # Bound qualification storage explicitly; failure does not silently omit
        # speculation rows or another request's recurrent state.
        if bundle.bytes * 2 > 14 * 1024**3:
            raise BufferError("Native replay checkpoint exceeds 14 GiB qualification budget")
        snapshots = StateCheckpoints(f"replay-{case}", max(1, bundle.bytes * 2), capacity=2)
        before = snapshots.capture(0, bundle.tensors)
        probe_capture = []
        with capture_to(probe_capture.append):
            probed = _clone(original(*args, **kwargs))
        after = snapshots.capture(1, bundle.tensors)
        state_adapter.validate(bundle)
        snapshots.restore(before, bundle.tensors)
        result = original(*args, **kwargs)
        output_equal = _equal(probed, result)
        state_equal = all(_equal(after.tensors[name], tensor) for name, tensor in bundle.tensors.items())
        event = torch.cuda.Event()
        event.record()
        event.synchronize()
        assert snapshots.release(0) and snapshots.release(1)
        reports.append({"case": case, "requests": list(batch.req_ids),
            "history_begins": batch.num_computed_tokens_np.tolist(),
            "query_lengths": batch.num_scheduled_tokens.tolist(),
            "state_tensors": len(bundle.tensors), "state_bytes": bundle.bytes,
            "recurrent_layers": sum(len(layers) for layers in state_adapter.groups.values()),
            "recurrent_pages": sum(len(rows) for rows in bundle.page_rows.values()),
            "captured_attention_layers": len(probe_capture),
            "output_byte_equal": output_equal, "state_byte_equal": state_equal})
        runner._kvmem_probe_output.mkdir(parents=True, exist_ok=True)
        (runner._kvmem_probe_output / "replay.json").write_text(json.dumps({
            "scope": "native recurrent restore and identity query replay; no compressed refresh",
            "cases": reports}, indent=2))
        if not output_equal or not state_equal:
            raise RuntimeError("Native recurrent restore/replay changed model output or state bytes")
        seen.add(case)
        return result

    model.forward = MethodType(forward, model)

"""Explicit diagnostic only: first-pressure immutable prefix and frozen scores.

Never imported by ordinary serving. No change to operands, selection or fences.
The extra reads/second preview make this unsuitable for throughput measurement.
"""
import hashlib
import json
from pathlib import Path


def install():
    from .layerwise_pressure import LayerwisePrefillPressure
    from .capture_runtime import CaptureRuntime
    if getattr(LayerwisePrefillPressure, '_repro_probe', False):
        return
    original = LayerwisePrefillPressure.prepare
    submit = CaptureRuntime.submit

    def snapshot_submit(runtime, captured, batch, rejected):
        # Observe the original GPU projection buffers BEFORE authority packing
        # and archival. Thus an upstream divergence cannot be blamed on the
        # later pressure assembler or archive region-copy path.
        import torch
        seen = getattr(runtime, '_repro_model_seen', None)
        if seen is None:
            seen = runtime._repro_model_seen = set()
        for index, rid in enumerate(batch.req_ids):
            if rid in seen or int(batch.num_computed_tokens_np[index]) != 0:
                continue
            lo, hi = map(int, batch.query_start_loc_np[index:index+2])
            if hi-lo < 96:
                raise RuntimeError('Initial batch is too short for frozen common-prefix diagnostic')
            root = Path(runtime.runner._kvmem_probe_output)/'first_model_prefix'
            root.mkdir(exist_ok=True)
            path = root/(rid+'.pt')
            if path.exists():raise RuntimeError('Never replace a first model-prefix snapshot')
            keys = torch.stack([r.key[lo:lo+96].clone() for r in captured[:16]]).cpu()
            values = torch.stack([r.value[lo:lo+96].clone() for r in captured[:16]]).cpu()
            torch.save(dict(raw=keys,values=values),path)
            with (root/(rid+'.json')).open('x') as f:
                json.dump(dict(request_id=rid,batch_requests=list(batch.req_ids),
                    batch_index=index,public_rows=hi-lo,prefix_tokens=96,
                    layer_names=[r.name for r in captured[:16]],
                    scope='Original BF16 GPU capture before authority packing/archive/first pressure.'),f)
            seen.add(rid)
        return submit(runtime,captured,batch,rejected)

    def prepare(controller, refresh, job):
        result = original(controller, refresh, job)
        rid = job.command.request_id
        seen = getattr(controller, '_repro_seen', None)
        if seen is None:
            seen = controller._repro_seen = set()
        if rid in seen or job.command.pressure_stage != 'prefill':
            return result
        seen.add(rid)
        import torch
        chosen, evidence = job.layerwise_pressure_plan
        first_selection = [[r.history_begin for r in layer] for layer in chosen]
        histories = controller.histories[rid]
        scores = [h.scores() for h in histories]
        original(controller, refresh, job)
        second_selection = [[r.history_begin for r in layer] for layer in job.layerwise_pressure_plan[0]]
        repeated = (first_selection == second_selection and scores == [h.scores() for h in histories])
        if not repeated:
            raise RuntimeError('Identical pressure state produced different score/selection')
        records = sorted(job.authority.records.values(), key=lambda r:r.history_begin)[:3]
        if [r.history_begin for r in records] != [0,32,64] or any(r.count!=32 for r in records):
            raise RuntimeError('First immutable prefix records are missing')
        operands = job.authority.read_records([r.handle for r in records])
        raw = torch.stack([k.view(torch.uint8) for k,v in operands]).cpu()
        values = torch.stack([v.view(torch.uint8) for k,v in operands]).cpu()
        root = Path(controller.runner._kvmem_probe_output)/'first_pressure_repro'
        root.mkdir(exist_ok=True)
        path = root/(rid+'.pt')
        if path.exists():
            raise RuntimeError('Never replace a first-pressure snapshot')
        torch.save(dict(raw=raw,values=values),path)
        info = dict(request_id=rid,accepted_history=job.history,prefix_tokens=96,
            original_layer_names=list(refresh.runtime.names),
            scores=scores,selection=first_selection,
            repeated_fixed_state_equal=repeated,
            samples=[dict(end=h.end,samples=h.samples,chunks=h.chunks) for h in histories],
            payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scope='Immutable FP8 archive operands before first pressure assembly; repeated selector preview '
                  'uses identical scores and does not replay the model. Diagnostic, not performance.')
        with (root/(rid+'.json')).open('x') as stream:
            json.dump(info,stream)
        return result

    LayerwisePrefillPressure.prepare = prepare
    LayerwisePrefillPressure._repro_probe = True
    CaptureRuntime.submit = snapshot_submit

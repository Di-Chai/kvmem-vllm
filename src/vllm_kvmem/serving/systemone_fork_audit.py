"""Audit one changed observer against every untouched owner at identical layout.

Snapshots live on CPU so four-way qualification fits the same 4090 budget.
The actual post-forward state is restored before foreground sampling/MTP.
Ordinary latency runs execute none of the snapshots or counterfactual forwards.
"""
import time
import torch
from ..adapters.capture import capture_to
from .native_state import NativeRecurrentState
from .systemone_worker import tail_views


class HostTailSnapshot:
    def __init__(self, views):
        self.rows = [(tensor, pages, offsets, tensor[pages, offsets].cpu())
                     for tensor, pages, offsets in views]

    def restore(self):
        for tensor, pages, offsets, saved in self.rows:
            tensor[pages, offsets] = saved.to(tensor.device)

    def verify(self):
        return all(torch.equal(tensor[pages, offsets].cpu(), saved)
                   for tensor, pages, offsets, saved in self.rows)


def clone_output(output):
    if isinstance(output, torch.Tensor):
        return output.clone()
    if isinstance(output, (tuple, list)):
        return type(output)(clone_output(value) for value in output)
    raise RuntimeError('Unsupported observer audit model output')


def equal_output(a, b, begin, end):
    if isinstance(a, torch.Tensor):
        return torch.equal(a[begin:end].contiguous().view(torch.uint8),
                           b[begin:end].contiguous().view(torch.uint8))
    return all(equal_output(x, y, begin, end) for x, y in zip(a, b))


def equal_bytes(a, b):
    return torch.equal(a.cpu().contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def install(fork):
    runner = fork.runner
    original = runner.model.forward

    def forward(*args, **kwargs):
        frame = fork.frame
        if frame is None:
            return original(*args, **kwargs)
        lanes = frame['lanes']
        if not any(lane.active['audit'] and
                   lane.active.get('audited_epoch') != int(lane.active['detached']) for lane in lanes):
            return original(*args, **kwargs)
        if args:
            raise RuntimeError('Fork audit requires named native model inputs')
        public = frame['public']
        parent = public.req_ids[0]
        children = [lane.active['child'] for lane in lanes]
        owners = [parent, *children]
        states = NativeRecurrentState(runner)
        bundle = states.borrow(owners)  # Proves every owner's recurrent rows disjoint.
        views = bundle.tensors
        before = {name: value.cpu() for name, value in views.items()}
        parent_window = int(public.num_computed_tokens_np[0]) - runner._kvmem_window_adapter.offsets.get(parent, 0)
        tails = {parent: tail_views(runner, parent, parent_window, public.num_tokens)}
        slices = {parent: (0, public.num_tokens)}
        for lane in lanes:
            job, piece = lane.active, lane.frame
            tails[job['child']] = tail_views(runner, job['child'], job['window'] + job['consumed'], piece['private_count'])
            slices[job['child']] = (piece['begin'], piece['last_token_index'] + 1)
        before_tail = HostTailSnapshot([view for tail in tails.values() for view in tail])
        inputs = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
        result = clone_output(original(*args, **kwargs))
        after = {name: value.cpu() for name, value in views.items()}
        after_tail = {owner: HostTailSnapshot(tail) for owner, tail in tails.items()}

        def restore(snapshot, tail):
            states.validate(bundle)
            for name, value in views.items():
                value.copy_(snapshot[name])
            for item in tail:
                item.restore()

        try:
            for lane in lanes:
                started = time.perf_counter()
                job = lane.active
                changed = job['child']
                begin, end = slices[changed]
                alternate_inputs = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
                poison = torch.zeros(end-begin, device=runner.device, dtype=torch.int64)
                if alternate_inputs.get('input_ids') is not None:
                    alternate_inputs['input_ids'][begin:end].zero_()
                if alternate_inputs.get('inputs_embeds') is not None:
                    alternate_inputs['inputs_embeds'][begin:end] = runner.model.embed_input_ids(poison)
                restore(before, [before_tail])
                with capture_to(lambda _: None):
                    alternate = runner._kvmem_private_forward(**alternate_inputs)
                checks = {}
                for owner in owners:
                    if owner == changed:
                        continue
                    checks[owner] = dict(
                        hidden_byte_equal=equal_output(result, alternate, *slices[owner]),
                        recurrent_byte_equal=all(equal_bytes(value, after[name]) for name, value in views.items() if name.startswith(owner + '/')),
                        kv_tail_byte_equal=after_tail[owner].verify())
                changed_state = any(not equal_bytes(value, after[name]) for name, value in views.items() if name.startswith(changed + '/'))
                report = dict(changed_owner=changed, unchanged_owners=checks,
                    parent_hidden_byte_equal=checks[parent]['hidden_byte_equal'],
                    parent_recurrent_byte_equal=checks[parent]['recurrent_byte_equal'],
                    parent_kv_tail_byte_equal=checks[parent]['kv_tail_byte_equal'],
                    other_observers_byte_equal=all(all(checks[owner].values()) for owner in children if owner != changed),
                    private_state_changed=changed_state, concurrent_branches=len(lanes),
                    after_cow=job['detached'], elapsed_ms=(time.perf_counter()-started)*1000)
                fork.emit('counterfactual', operation=job['operation'], batch_id=frame['batch_id'], **report)
                if not changed_state or not all(all(check.values()) for check in checks.values()):
                    raise RuntimeError('Observer content changed another owner at identical batch layout')
                job.setdefault('counterfactual_checks', []).append(report)
                job['audited_epoch'] = int(job['detached'])
        finally:
            restore(after, after_tail.values())
            torch.cuda.current_stream(runner.device).synchronize()
        return result

    runner.model.forward = forward

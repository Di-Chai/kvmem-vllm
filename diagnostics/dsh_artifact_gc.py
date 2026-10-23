"""Opt-in DSH experiment: reclaim exact file bodies backed by verified artifacts.

No attention observations or model scores are invented. The experiment profile
requires one real file-equality observation, disables the cold-count gate (0),
and accepts unknown/high importance (1). Default statistical settings stay 3/3.
Only an owned loopback development server loads this worker extension.
"""
import copy
import functools
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import weakref

ENABLED = 'KVMEM_DSH_ARTIFACT_GC'
_controller = None


def token_digest(ids):
    return hashlib.sha256(struct.pack('<' + 'I'*len(ids), *ids)).hexdigest()


def file_body_tokens(ids, raw, content, token_bytes):
    """Map only complete accepted tokens inside the unique literal file body."""
    body = content.encode()
    marker = b'<parameter=content>\n' + body
    data = raw.encode()
    if not body or data.count(marker) != 1:
        raise ValueError('File body has no unique literal parameter boundary')
    begin = data.index(marker) + len(b'<parameter=content>\n')
    end = begin + len(body)
    parts = [token_bytes(i) for i in ids]
    accepted = b''.join(parts)
    if not accepted.startswith(data):
        raise ValueError('Native text differs from actual accepted token bytes')
    offset, covered = 0, []
    for i, part in enumerate(parts):
        stop = offset + len(part)
        if begin <= offset and stop <= end:
            covered.append(i)
        offset = stop
    if not covered or covered != list(range(covered[0], covered[-1]+1)):
        raise ValueError('File body has no contiguous complete-token range')
    return covered[0], covered[-1]+1


def verified_candidate(root, output, owner, state, prepared, body, token_bytes):
    """Consume the real previous write/result pair; fail closed on ambiguity."""
    if not state.turns or not prepared[5]:
        return None
    directory = output/owner/f'{state.turns:05d}'
    previous = json.loads((directory/'response.json').read_text())
    calls = previous['choices'][0]['message'].get('tool_calls', [])
    if len(calls) != 1 or calls[0]['function']['name'] != 'write':
        return None
    call = calls[0]
    args = json.loads(call['function']['arguments'])
    if set(args) != {'file_path', 'content'}:
        return None
    path = Path(args['file_path'])
    root = root.resolve(strict=True)
    # Keep a strict workspace boundary, including all symlink ancestors.
    if not path.is_absolute() or path.is_symlink():
        return None
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.stat().st_size > 1048576:
        return None
    content = args['content']
    if not isinstance(content, str) or resolved.read_bytes() != content.encode():
        return None
    expected = f'<path>{path}</path>\n<type>file</type>\n<content>\nCreated file\n</content>'
    results = [m for m in body['messages'] if m.get('role') == 'tool' and m.get('tool_call_id') == call['id']]
    if len(results) != 1 or results[0].get('content') != expected:
        return None
    tokens, _, rendered, _, _, _, locations = prepared
    if tokens[:len(state.tokens)+len(state.output)] != state.tokens + state.output:
        raise ValueError('Artifact evidence lost the exact accepted prefix')
    native = json.loads((directory/'native.json').read_text())
    if native['token_ids'] != state.output:
        raise ValueError('Artifact response differs from accepted token ledger')
    begin, end = file_body_tokens(state.output, native['text'], content, token_bytes)
    witnesses = [s for s in rendered['spans'] if s['role'] == 'tool' and
        rendered['messages'][s['index']].get('tool_call_id') == call['id']]
    if len(witnesses) != 1:
        return None
    s = witnesses[0]
    rb = locations[(s['index'], s['content_begin']-s['begin'])]
    re = locations[(s['index'], s['content_end']-s['begin'])]
    begin += len(state.tokens)
    end += len(state.tokens)
    if (end//32 - (begin+31)//32) < 1 or (re//32 - (rb+31)//32) < 1:
        return None
    return dict(span_id=f'file-body-{state.turns}', namespace=[owner, owner],
        prompt_tokens=len(tokens), prefix_sha256=token_digest(tokens),
        source_begin=begin, source_end=end, result_begin=rb, result_end=re,
        relative_path=str(resolved.relative_to(root)), artifact_sha256=hashlib.sha256(content.encode()).hexdigest(),
        artifact_bytes=len(content.encode()), source='verified_dsh_write_artifact',
        importance=1.0, cold=False, observations=1)


class Controller:
    def __init__(self):
        self.runner = None
        self.states = []

    def arm(self, runner, plan):
        if len(self.states) >= 64:
            raise ValueError('Artifact experiment span limit exceeded')
        required = ('source_begin', 'source_end', 'result_begin', 'result_end', 'prompt_tokens')
        if any(type(plan.get(k)) is not int for k in required):
            raise ValueError('Artifact coordinates must be integers')
        if not 0 <= plan['source_begin'] < plan['source_end'] <= plan['result_begin'] < plan['result_end'] <= plan['prompt_tokens']:
            raise ValueError('Artifact ranges are not ordered accepted history')
        if len(plan.get('namespace', [])) != 2 or any(not isinstance(v, str) or not v for v in plan['namespace']):
            raise ValueError('Artifact namespace is required')
        if plan.get('source') != 'verified_dsh_write_artifact' or plan.get('importance') != 1 or plan.get('cold') is not False or plan.get('observations') != 1:
            raise ValueError('Artifact observations cannot impersonate attention statistics')
        identity = plan['namespace'], plan['span_id']
        if any((s['plan']['namespace'], s['plan']['span_id']) == identity for s in self.states):
            raise ValueError('Duplicate artifact evidence')
        self.runner = weakref.ref(runner)
        self.states.append(dict(plan=plan, phase='armed', error=None, events=[],
            registered=False, observed=False, complete=False, flushed=False,
            prefix_checked=False, candidate_handles=[], result_handles=[], last_sample=None))
        return self.snapshot()

    def snapshot(self):
        return copy.deepcopy(self.states)

    def poll(self, authority):
        runner = self.runner() if self.runner is not None else None
        sessions = getattr(runner, '_kvmem_sessions', None)
        binding = getattr(sessions, 'bindings', {}).get(authority.memory.request_id)
        if binding is None:
            return
        for state in self.states:
            p = state['plan']
            if state['error'] is not None or list(binding.namespace) != p['namespace']:
                continue
            try:
                if not state['prefix_checked']:
                    if len(binding.tokens) != p['prompt_tokens']:
                        continue
                    if token_digest(binding.tokens) != p['prefix_sha256']:
                        raise ValueError('Artifact plan does not match the actual worker prefix')
                    state['prefix_checked'] = True
                    state['generation'] = binding.generation
                if binding.generation != state['generation']:
                    raise ValueError('Artifact evidence crossed a session generation')
                if authority.history_end < p['result_end'] + 32:
                    continue
                previous = state['last_sample']
                if previous is None or authority.history_end - previous['accepted_history'] >= 16:
                    stats = dict(authority.gc.scope.stats)
                    sample = dict(at=time.perf_counter(), accepted_history=authority.history_end,
                        retired=stats['retired'], cpu_released=stats['cpu_released'])
                    if stats['policy_failures']:
                        raise RuntimeError('Native policy failure')
                    for event in state['events']:
                        if event['kind'] in ('retired', 'cpu_released') and 'following_accepted_history' not in event and sample['accepted_history'] > event['accepted_history']:
                            event['following_accepted_history'] = sample['accepted_history']
                            event['following_at'] = sample['at']
                    if previous is not None:
                        for key in ('retired', 'cpu_released'):
                            if sample[key] > previous[key]:
                                state['events'].append(dict(kind=key, before=previous[key], after=sample[key], at=sample['at'], accepted_history=sample['accepted_history']))
                    state['last_sample'] = sample
                if state['flushed']:
                    state['remaining_source_records'] = sum((s, g) in authority.records for s, g in state['candidate_handles'])
                    if previous is None or authority.history_end - previous['accepted_history'] >= 16:
                        state['window_source_records'] = sum(p['source_begin'] <= row[2] and row[2]+row[3] <= p['source_end']
                            for row in authority.memory._store.working_records())
                        if state.get('last_window_request') != binding.request_id:
                            state['events'].append(dict(kind='request_window', request_id=binding.request_id,
                                accepted_history=authority.history_end, source_records=state['window_source_records']))
                            state['last_window_request'] = binding.request_id
                    continue
                ledger = authority.gc.evidence
                epoch = tuple(p['namespace'])
                if ledger is None:
                    ledger = authority.gc.open_evidence(authority.records, epoch=epoch)
                if ledger.closed or ledger.epoch != epoch:
                    raise RuntimeError('Artifact ledger crossed a branch reset')
                if not state['registered']:
                    sources = [r for r in authority.records.values() if r.archived and r.count == 32 and
                        p['source_begin'] <= r.history_begin and r.history_begin+32 <= p['source_end']]
                    results = [r for r in authority.records.values() if r.archived and r.count == 32 and
                        p['result_begin'] <= r.history_begin and r.history_begin+32 <= p['result_end']]
                    expected = p['source_end']//32-(p['source_begin']+31)//32
                    if len(sources) != expected or not results:
                        continue
                    if not ledger.register_span(p['span_id'], p['source_begin'], p['source_end'], kind='tool_detail', epoch=epoch):
                        continue
                    state['candidate_handles'] = [[r.handle.slot, r.handle.generation] for r in sources]
                    state['result_handles'] = [[r.handle.slot, r.handle.generation] for r in results]
                    state['registered'] = True
                    state['events'].append(dict(kind='registered', at=time.perf_counter(), accepted_history=authority.history_end, candidate_blocks=len(sources)))
                if not state['observed']:
                    if not ledger.observe(p['span_id'], epoch=epoch, observation_id=0, importance=1.0, cold=False):
                        continue
                    state['observed'] = True
                if not state['complete']:
                    results = [authority.records[tuple(h)] for h in state['result_handles']]
                    if not ledger.complete_span(p['span_id'], results, epoch=epoch):
                        continue
                    state['complete'] = True
                if ledger.flush(max_blocks=32):
                    state['flushed'], state['phase'] = True, 'observing'
                    state['events'].append(dict(kind='published', at=time.perf_counter(), accepted_history=authority.history_end))
            except Exception as error:
                state['error'] = type(error).__name__+': '+str(error)
                state['phase'] = 'failed'


def install():
    global _controller
    if _controller is not None:
        return
    from vllm_kvmem.authority import RequestAuthority
    original = RequestAuthority.poll
    controller = Controller()
    @functools.wraps(original)
    def poll(authority, *args, **kwargs):
        result = original(authority, *args, **kwargs)
        controller.poll(authority)
        return result
    RequestAuthority.poll = poll
    from vllm_kvmem.historical_gc import HistoricalAuthorityGc
    retired = HistoricalAuthorityGc._record_retired
    @functools.wraps(retired)
    def record_retired(gc, record):
        retired(gc, record)
        key = [record.handle.slot, record.handle.generation]
        for state in controller.states:
            if key in state['candidate_handles']:
                state['events'].append(dict(kind='source_record_released', handle=key,
                    history_begin=record.history_begin, at=time.perf_counter(), during_live_scope=not gc.stopped))
    HistoricalAuthorityGc._record_retired = record_retired
    _controller = controller


class Extension:
    def dsh_artifact_gc_arm(self, plan):
        if os.environ.get(ENABLED) != '1':
            raise RuntimeError('Explicit artifact experiment startup is required')
        install()
        return _controller.arm(self.model_runner, plan)

    def dsh_artifact_gc_snapshot(self):
        if _controller is None:
            raise RuntimeError('Explicit artifact experiment startup is required')
        return _controller.snapshot()


def serve():
    from vllm_kvmem import serve as serving
    original = serving.build_launch
    def build(*args, **kwargs):
        plan = original(*args, **kwargs)
        if plan['profile']['host'] != '127.0.0.1':
            raise ValueError('Artifact development RPC requires an owned loopback server')
        gc = plan['profile']['kvmem']['historical_gc']
        params = gc.get('policy_params', {})
        if gc['policy'] != 'statistical' or params.get('min_observations') != 1 or params.get('min_cold_observations') != 0 or params.get('max_importance') != 1:
            raise ValueError('Artifact experiment requires the explicit file-verification policy profile')
        plan['vllm_argv'] += ['--worker-extension-cls', 'diagnostics.dsh_artifact_gc.Extension']
        plan['environment'].update({ENABLED: '1', 'VLLM_SERVER_DEV_MODE': '1'})
        return plan
    serving.build_launch = build
    serving.main(sys.argv[1:])


if os.environ.get(ENABLED) == '1':
    install()

if __name__ == '__main__':
    serve()

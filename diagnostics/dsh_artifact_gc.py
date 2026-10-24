"""Opt-in DSH experiment: reclaim exact file bodies backed by verified artifacts.

No attention observations or model scores are invented. The experiment profile
requires one real file-equality observation, disables the cold-count gate (0),
and accepts unknown/high importance (1). Default statistical settings stay 3/3.
Only an owned loopback development server loads this worker extension.
"""
import copy
import base64
import functools
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
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


def workspace_artifact(root, path, *, container=None):
    """Read bounded real file bytes in the tool's filesystem, without a shell."""
    if container is not None:
        program = '''import base64,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True)
path=Path(sys.argv[2])
if not path.is_absolute(): path=root/path
try:
 resolved=path.resolve(strict=True)
 valid=not path.is_symlink() and resolved.is_relative_to(root) and resolved.is_file() and resolved.stat().st_size<=1048576
 data=resolved.read_bytes() if valid else None
 print(json.dumps(None if data is None or len(data)>1048576 else dict(path=str(path),relative_path=str(resolved.relative_to(root)),data=base64.b64encode(data).decode())))
except (OSError,ValueError): print('null')
'''
        result = subprocess.run(['docker','exec',container,'python','-c',program,str(root),str(path)],
            capture_output=True,timeout=15,check=True)
        value=json.loads(result.stdout)
        if value is None:
            return None
        data=base64.b64decode(value['data'],validate=True)
        if len(data)>1048576:
            raise ValueError('Artifact reader exceeded the bounded file size')
        return Path(value['path']),value['relative_path'],data
    root=root.resolve(strict=True)
    if not path.is_absolute():
        path=root/path
    if path.is_symlink():
        return None
    resolved=path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.stat().st_size>1048576:
        return None
    data=resolved.read_bytes()
    return None if len(data)>1048576 else (path,str(resolved.relative_to(root)),data)


def maintenance_boundary(data, mode):
    """Use the current real prompt tail for a bounded internal GC handoff.

    This is an explicit raw scoring span, not a new human Query or generated
    judge request. OFF retains normal continuation and the full attention view.
    """
    if mode != 'delete':
        return None
    count=len(data['prompt'])
    begin=(count-1)//32*32
    if begin<=0:
        raise ValueError('GC maintenance requires a complete accepted prompt')
    original={k:data[k] for k in ('kvmem_query_id','kvmem_query_span','kvmem_query_mode') if k in data}
    data.update(kvmem_reselect='force',kvmem_query_span=[begin,count],
        kvmem_query_mode='raw',kvmem_query_id='artifact-maintenance-'+str(count))
    data.pop('kvmem_replay_span',None)
    return dict(kind='verified_artifact_maintenance',original_query=original,
        replay_span=[begin,count],replay_tokens=count-begin)


def artifact_handoff_needed(snapshot,owner):
    """A handoff is needed only for published sources still in the GPU view."""
    return any(s.get('plan',{}).get('namespace')==[owner,owner] and
        s.get('plan',{}).get('source')=='verified_dsh_write_artifact' and
        s.get('flushed') and s.get('complete') and s.get('observed') and s.get('error') is None and
        (s.get('window_source_records') or 0)>0 and (s.get('remaining_source_records') or 0)>0
        for worker in snapshot.get('results',[]) for s in worker)


def verified_candidate(root, output, owner, state, prepared, body, token_bytes, call_id=None, *, container=None):
    """Consume the real previous write/result pair; fail closed on ambiguity."""
    if not state.turns or not prepared[5]:
        return None
    directory = output/owner/f'{state.turns:05d}'
    previous = json.loads((directory/'response.json').read_text())
    calls = previous['choices'][0]['message'].get('tool_calls', [])
    if call_id is not None:
        calls = [c for c in calls if c['id'] == call_id]
    if len(calls) != 1 or calls[0]['function']['name'] != 'write':
        return None
    call = calls[0]
    args = json.loads(call['function']['arguments'])
    if set(args) != {'file_path', 'content'}:
        return None
    artifact=workspace_artifact(root,Path(args['file_path']),container=container)
    if artifact is None:
        return None
    path,relative_path,data=artifact
    content = args['content']
    if not isinstance(content, str) or data != content.encode():
        return None
    expected = [f'<path>{path}</path>\n<type>file</type>\n<content>\n{verb} file\n</content>' for verb in ('Created','Updated')]
    results = [m for m in body['messages'] if m.get('role') == 'tool' and m.get('tool_call_id') == call['id']]
    if len(results) != 1 or results[0].get('content') not in expected:
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
    # Protect a whole block intersecting the real result, including mixed
    # framing if necessary. Protecting extra bytes is conservative; no source
    # block can overlap this witness. The generated file body stays exact.
    wb=max((end+31)//32*32,rb//32*32)
    we=wb+32
    if (end//32 - (begin+31)//32) < 1 or we>len(tokens) or not (wb<re and we>rb):
        return None
    rb,re=wb,we
    return dict(span_id=f'file-body-{state.turns}-'+hashlib.sha256(call['id'].encode()).hexdigest()[:12], namespace=[owner, owner],
        prompt_tokens=len(tokens), prefix_sha256=token_digest(tokens),
        source_begin=begin, source_end=end, result_begin=rb, result_end=re,
        relative_path=relative_path, artifact_sha256=hashlib.sha256(content.encode()).hexdigest(),
        artifact_bytes=len(content.encode()), source='verified_dsh_write_artifact',
        importance=1.0, cold=False, observations=1)


def verified_candidates(root, output, owner, state, prepared, body, token_bytes, already, *, container=None):
    """Find current verified artifacts from committed real tool history."""
    values=[]
    if not prepared[5]:
        return values
    tokens=prepared[0]
    for directory in sorted((output/owner).glob('[0-9][0-9][0-9][0-9][0-9]')):
        if not (directory/'accepted.json').exists():
            continue
        response=json.loads((directory/'response.json').read_text())
        calls=response['choices'][0]['message'].get('tool_calls',[])
        if not any(c['function']['name']=='write' for c in calls):
            continue
        native=json.loads((directory/'native.json').read_text())
        ids=json.loads((directory/'accepted.json').read_text())['token_ids']
        if native['token_ids']!=ids:
            raise ValueError('Artifact committed/native token mismatch')
        count=native['usage']['prompt_tokens']
        if tokens[count:count+len(ids)]!=ids:
            raise ValueError('Historical file body is absent from the current exact prefix')
        older=SimpleNamespace(turns=int(directory.name),tokens=tokens[:count],output=ids)
        for call in calls:
            if call['function']['name']!='write':
                continue
            identity=f"file-body-{older.turns}-"+hashlib.sha256(call['id'].encode()).hexdigest()[:12]
            if identity in already:
                continue
            value=verified_candidate(root,output,owner,older,prepared,body,token_bytes,call['id'],container=container)
            if value is not None:
                values.append(value)
    return values


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

    def select_window(self, authority, mandatory, limit):
        """Public Query handoff for all-fit histories; never release P/M early."""
        if getattr(getattr(authority,'archive',None),'_history_mode',None) != 'delete':
            return None
        namespace=getattr(authority,'_verified_artifact_namespace',None)
        states=[s for s in self.states if s['flushed'] and s['error'] is None and tuple(s['plan']['namespace'])==namespace]
        candidates={tuple(h) for s in states for h in s['candidate_handles']}
        records=list(authority.records.values())
        candidates.intersection_update((r.handle.slot,r.handle.generation) for r in records)
        hard={(h.slot,h.generation) for h in mandatory}
        # Native picker also owns sink/recent bands. Keep those source pages;
        # an explicit protected candidate never causes the handoff to bypass it.
        block=authority.memory.config.block_tokens
        recent=(authority.history_end-1)//block
        candidates={k for k in candidates if k not in hard and
            authority.records[k].history_begin//block not in (0,recent)}
        if not candidates:
            return None
        kept=[r for r in records if (r.handle.slot,r.handle.generation) not in candidates]
        budget=((sum(r.count for r in kept)+block-1)//block)*block
        if not 0<budget<=limit:
            return None
        scores={r.handle.slot:0.0 if (r.handle.slot,r.handle.generation) in candidates else 1.0 for r in records}
        slots={r.handle.slot for r in kept}
        selected=set(authority.memory.preview_selection(scores,tuple(h.slot for h in mandatory),action=False,max_tokens=budget))
        if selected!=slots:
            raise RuntimeError('Artifact handoff picker did not keep exactly the non-source records')
        plan=authority.memory.select(scores,tuple(h.slot for h in mandatory),action=False,max_tokens=budget)
        for state in states:
            state['events'].append(dict(kind='query_handoff_prepared',at=time.perf_counter(),
                accepted_history=authority.history_end, excluded_blocks=len(candidates), selected_budget=budget,
                configured_budget=authority.memory.config.select_budget,reserved_limit=limit))
        return plan

    def poll(self, authority):
        runner = self.runner() if self.runner is not None else None
        sessions = getattr(runner, '_kvmem_sessions', None)
        runtime=getattr(runner,'_kvmem_capture_runtime',None)
        active=[rid for rid,a in getattr(runtime,'authorities',{}).items() if a is authority]
        binding=next((sessions.bindings[rid] for rid in active if rid in sessions.bindings),None) if sessions is not None else None
        if binding is None:
            return
        for state in self.states:
            p = state['plan']
            if state['error'] is not None or list(binding.namespace) != p['namespace']:
                continue
            try:
                if state.get('checked_request') != binding.request_id:
                    if len(binding.tokens) < p['prompt_tokens']:
                        continue
                    if token_digest(binding.tokens[:p['prompt_tokens']]) != p['prefix_sha256']:
                        raise ValueError('Artifact plan does not match the actual worker prefix')
                    state['prefix_checked'] = True
                    state['generation'] = state.get('generation',binding.generation)
                    state['checked_request'] = binding.request_id
                if binding.generation != state['generation']:
                    raise ValueError('Artifact evidence crossed a session generation')
                authority._verified_artifact_namespace=tuple(binding.namespace)
                ref=weakref.ref(authority)
                authority._verified_artifact_selection=lambda mandatory,limit:self.select_window(ref(),mandatory,limit)
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
                current_ledger=authority.gc.evidence
                if state['flushed'] and (current_ledger is None or current_ledger.closed or p['span_id'] not in current_ledger.spans):
                    recreated=[r for r in authority.records.values() if p['source_begin']<=r.history_begin and r.history_begin+r.count<=p['source_end']]
                    if recreated:
                        state['flushed']=state['registered']=state['observed']=state['complete']=False
                        state['events'].append(dict(kind='branch_evidence_rebind',at=time.perf_counter(),accepted_history=authority.history_end))
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
                    state['archive_request_id'] = authority.memory.request_id
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
            if gc.memory.request_id == state.get('archive_request_id') and key in state['candidate_handles']:
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

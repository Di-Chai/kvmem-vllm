#!/usr/bin/env python3
"""Read saved DSH/worker state and print exact release requests; never use HTTP.

Run after the DSH task exits. The generated curl command is intentionally not
executed. A successful release must be followed by another probe: the namespace
must disappear and scheduler requests/pages/maintenance must drain to zero.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time


INFLIGHT = ('requests', 'owned_blocks', 'refresh_pending', 'connector_pending', 'reservations')
CAPTURE_EVENTS = ('async_restore.jsonl', 'scheduler_refresh.jsonl', 'verified_restores.jsonl',
                  'bounded_refresh.jsonl', 'async_refresh.jsonl')


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def rows(path):
    if not path.is_file():
        return []
    result = []
    for line in path.read_text().splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def matches(external, internal):
    return isinstance(external, str) and isinstance(internal, str) and (
        external == internal or internal.startswith(external+'-0-'))


def source_owner(body, namespace):
    """Same stable system/tool frame as scripts.dsh_bridge.session_owner."""
    key = body.get('prompt_cache_key')
    if not isinstance(key, str) or not key:
        raise ValueError('Saved DSH request has no prompt_cache_key')
    systems = []
    for message in body['messages']:
        if message['role'] in ('system', 'developer'):
            content = message.get('content') or ''
            if not isinstance(content, str):
                content = '\n'.join(p['text'] for p in content if p['type'] == 'text')
            systems.append(content)
    tools = [dict(name=t['function']['name'],
                  description=t['function'].get('description', ''),
                  input_schema=t['function'].get('parameters', dict(type='object', properties={})))
             for t in body.get('tools', [])]
    frame = json.dumps(dict(system='\n\n'.join(systems), tools=tools),
                       ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return namespace+'-'+hashlib.sha256((key+'\0'+frame).encode()).hexdigest()[:40]


def release_payload(model, owner):
    return dict(model=model, prompt=[1], session_id=owner, cache_salt=owner,
                max_tokens=1, temperature=0, kvmem_reselect='off',
                kvmem_mandatory_parts='none', kvmem_mandatory_spans=[],
                kv_transfer_params=dict(kvmem_release_session=True))


def artifact_gc_evidence(bridge, *, owner_filter=None):
    """Join saved, committed worker snapshots; cleanup frees are excluded."""
    sources, released, first = {}, set(), {}
    committed = {}
    for directory in sorted(bridge.iterdir()):
        if not directory.is_dir() or owner_filter and directory.name != owner_filter:
            continue
        turns = [t for t in sorted(directory.glob('[0-9][0-9][0-9][0-9][0-9]'))
                 if (t/'accepted.json').is_file()]
        committed[directory.name] = [int(t.name) for t in turns]
        for turn in turns:
            snapshot = read(turn/'artifact-gc-state.json', {})
            for worker in snapshot.get('results', []):
                for state in worker:
                    plan = state.get('plan', {})
                    if (plan.get('source') != 'verified_dsh_write_artifact' or
                            plan.get('namespace') != [directory.name, directory.name]):
                        continue
                    identity = directory.name, plan['span_id']
                    handles = {tuple(h) for h in state.get('candidate_handles', [])}
                    valid = []
                    if state.get('flushed') and state.get('complete') and state.get('observed') and state.get('error') is None:
                        valid = [e for e in state.get('events', [])
                            if e.get('kind') == 'source_record_released' and e.get('during_live_scope') is True
                            and tuple(e.get('handle', [])) in handles
                            and plan['source_begin'] <= e.get('history_begin', -1)
                            and e.get('history_begin', -1)+32 <= plan['source_end']]
                    for event in valid:
                        released.add((directory.name, state.get('archive_request_id'), *event['handle']))
                    if valid:
                        first.setdefault(directory.name, int(turn.name))
                    sources[identity] = dict(owner=directory.name, span_id=plan['span_id'],
                        relative_path=plan.get('relative_path'), error=state.get('error'),
                        candidate_blocks=len(handles), live_source_completions=len(valid),
                        remaining_source_records=state.get('remaining_source_records'),
                        working_window_source_records=state.get('window_source_records'))
    later = sum(sum(t > first[owner] for t in turns)
                for owner, turns in committed.items() if owner in first)
    return dict(semantic_producer_present=bool(sources), source_spans=list(sources.values()),
        live_source_cpu_completions=len(released), real_semantic_candidate_retirement_proven=bool(released),
        later_committed_model_replies=later, task_continuation_after_release_proven=later > 0)


def report(bridge, server_run, profile, *, owner_filter=None):
    config = read(profile)
    if not isinstance(config, dict):
        raise ValueError('A saved server profile is required')
    port, model = config.get('port'), config.get('served_model_name')
    if type(port) is not int or not 1 <= port <= 65535 or not isinstance(model, str) or not model:
        raise ValueError('Profile must identify the exact backend port and served model')
    endpoint = f'http://127.0.0.1:{port}/v1/completions'
    capture = server_run/'capture'
    snapshot_path = capture/'scheduler_resources.json'
    snapshot = read(snapshot_path)
    archive = read(capture/'archive.json', {})
    events = {name: rows(capture/name) for name in CAPTURE_EVENTS}
    requests = rows(bridge/'requests.jsonl')
    quiescent = bool(snapshot is not None and all(
        name in snapshot and not snapshot[name] for name in INFLIGHT))
    owners = []
    for directory in sorted(bridge.iterdir()):
        if not directory.is_dir() or owner_filter and directory.name != owner_filter:
            continue
        turns, namespaces = [], set()
        for turn in sorted(directory.glob('[0-9][0-9][0-9][0-9][0-9]')):
            plan, request = read(turn/'plan.json'), read(turn/'request.json')
            if plan is None and request is None:
                continue
            if not isinstance(plan, dict) or not isinstance(request, dict):
                raise ValueError(f'Incomplete namespace evidence in {turn}')
            owner, salt = plan.get('session_id'), plan.get('cache_salt')
            if owner != directory.name or salt != owner or plan.get('model') != model:
                raise ValueError(f'Bridge namespace/model differs from saved native plan: {turn}')
            prefix, separator, digest = owner.rpartition('-')
            if not separator or source_owner(request, prefix) != owner:
                raise ValueError(f'Wire request does not prove saved bridge owner: {turn}')
            namespaces.add((salt, owner))
            accepted, response = read(turn/'accepted.json'), read(turn/'response.json')
            complete = accepted is not None
            if complete:
                ids = accepted.get('token_ids')
                usage = response.get('usage') if isinstance(response, dict) else None
                if not isinstance(ids, list) or not usage or usage.get('completion_tokens') != len(ids):
                    raise ValueError(f'Accepted-ID commit marker has inconsistent response usage: {turn}')
            segments = []
            for segment in sorted(turn.glob('segment-[0-9][0-9][0-9].json')):
                value = read(segment)
                external = value.get('id')
                restores = [r for r in events['async_restore.jsonl'] if matches(external, r.get('request_id'))]
                refreshes = [r for r in events['scheduler_refresh.jsonl'] if matches(external, r.get('request_id'))]
                oracles = [r for r in events['verified_restores.jsonl'] if matches(external, r.get('request_id'))]
                completed = [r for r in events['bounded_refresh.jsonl'] if matches(external, r.get('request_id'))]
                progress = [r for r in events['async_refresh.jsonl'] if matches(external, r.get('request_id'))]
                selections = {(r.get('request_id'), r.get('command_id')): r.get('selection')
                              for r in completed}
                classified = bool(refreshes) and all(
                    selections.get((r.get('request_id'), r.get('command_id'))) in ('query', 'pressure')
                    for r in refreshes)
                def replay_count(selection):
                    return (sum(r.get('replay_tokens', 0) for r in refreshes
                                if selections[(r.get('request_id'), r.get('command_id'))] == selection)
                            if classified else None)
                summary_fields = ('command_id', 'selection', 'query_tokens', 'query_span',
                    'private_forward_steps', 'query_execution_counts', 'assembly_seconds',
                    'assembly_host_prepare_seconds', 'assembly_uploaded_bytes',
                    'refresh_started_at', 'refresh_finished_at')
                segments.append(dict(index=value.get('index'), request_id=external,
                    prompt_tokens=value.get('prompt_tokens'), output_tokens=value.get('output_tokens'),
                    finish_reason=value.get('finish_reason'), restore_events=restores,
                    restore_elapsed_ms=[1000*(r['finished']-r['started']) for r in restores],
                    refresh_replay_tokens=sum(r.get('replay_tokens', 0) for r in refreshes) if refreshes else None,
                    query_replay_tokens=replay_count('query'), pressure_replay_tokens=replay_count('pressure'),
                    private_query_forward_tokens=sum(r.get('tokens', 0) for r in progress
                                                     if r.get('stage') == 'query') if progress else None,
                    refresh_events=refreshes,
                    completed_refreshes=[{k:r[k] for k in summary_fields if k in r} for r in completed],
                    verified_restore_events=oracles,
                    metrics=value.get('metrics')))
            row = next((r for r in reversed(requests)
                        if r.get('owner') == owner and r.get('turn') == int(turn.name)), {})
            turns.append(dict(turn=int(turn.name), committed=complete,
                prompt_tokens=plan.get('prompt_tokens'), query_span=plan.get('kvmem_query_span'),
                query_source=plan.get('query_source'),
                exact_output_prefix_reused=plan.get('exact_output_prefix_reused'),
                accepted_output_tokens=len(accepted['token_ids']) if complete else None,
                first_native_token_seconds=row.get('first_native_token_seconds'),
                elapsed_seconds=row.get('seconds'), segments=segments))
        if not turns:
            continue
        if len(namespaces) != 1:
            raise ValueError(f'Owner directory mixes native namespaces: {directory}')
        namespace = list(next(iter(namespaces)))
        leases = [] if snapshot is None else [
            r for r in snapshot.get('session_owners', []) if r.get('namespace') == namespace]
        status = ('capture_missing' if snapshot is None else 'already_absent' if not leases else
                  'idle_in_capture' if quiescent and all(r.get('phase') == 'idle' for r in leases)
                  else 'still_active_or_retiring')
        payload = release_payload(model, directory.name)
        command = shlex.join(['curl', '--noproxy', '*', '--fail-with-body', '--max-time', '180',
                              '-H', 'Content-Type: application/json', '--data-binary',
                              json.dumps(payload, separators=(',', ':')), endpoint])
        owners.append(dict(owner=directory.name, namespace=namespace, status=status,
            session_leases=leases, completed_turns=sum(t['committed'] for t in turns),
            incomplete_turns=[t['turn'] for t in turns if not t['committed']],
            turns=turns, release_payload=payload, release_command=command,
            release_confirmed_absent=snapshot is not None and not leases and quiescent))
    if owner_filter and not owners:
        raise ValueError('Requested owner has no saved bridge namespace evidence')
    semantic = artifact_gc_evidence(bridge, owner_filter=owner_filter)
    return dict(schema_version=1, read_only=True, endpoint=endpoint,
        bridge_output=str(bridge), server_run=str(server_run), profile=str(profile),
        snapshot_age_seconds=None if snapshot is None else time.time()-snapshot_path.stat().st_mtime,
        scheduler_quiescent_in_saved_snapshot=quiescent,
        scheduler_resources=snapshot, owners=owners,
        release_protocol=dict(prerequisite='DSH task exited; inspect a fresh idle scheduler snapshot before executing the printed command',
            control='One input token and one output token after generation-qualified fenced eviction; this is not a zero-model administrative API',
            completion='HTTP success, followed by namespace absent and requests/owned_blocks/refresh_pending/connector_pending/reservations all empty',
            evidence='Session release/eviction counters are not proactive statistical GC evidence'),
        gc_evidence=dict(config=config.get('kvmem', {}).get('historical_gc'),
            current_counters=archive.get('historical_gc'), **semantic,
            statistical_candidate_retirement_proven=(semantic['real_semantic_candidate_retirement_proven'] and config.get('kvmem', {}).get('historical_gc', {}).get('policy') == 'statistical'),
            calibrated_statistical_policy_quality_proven=False, attention_importance_measurements_present=False,
            decode_overlap_proven=False,
            reason=('Verified-file evidence and candidate-specific live-scope CPU completions are joined from committed worker snapshots. This does not calibrate attention statistics or prove paired decode throughput.' if semantic['semantic_producer_present'] else 'This stage publishes no semantic eligibility/observation evidence. Aggregate archive counters cannot establish candidate identity or decode overlap.')),
        replay_evidence=dict(join='Bridge segment response ID -> exact worker ID or response-ID + -0- prefix',
            measured='async_restore history/active tokens, uploaded/model bytes, host preparation and wall duration; scheduler_refresh replay_tokens classified by bounded_refresh.selection, async_refresh Query forward tokens; bridge TTFT',
            limits='An absent event means unobserved, not free. Bridge elapsed/TTFT includes queueing and transport. GC-induced replay cost requires a matched baseline with the same accepted trajectory.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge-output', type=Path, required=True)
    parser.add_argument('--server-run', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--owner')
    args = parser.parse_args()
    try:
        result = report(args.bridge_output, args.server_run, args.profile, owner_filter=args.owner)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

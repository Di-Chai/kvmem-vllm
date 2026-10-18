"""Fail closed on real-model restore evidence before coordination performance.

The qualification uses one prompt per OpenAI completion. Pinned vLLM appends
``-0`` in completion/serving.py, then ``-<8 hex>`` in input_processor.py. Match
that exact transformation, never substring/session/order-based guesses.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise EvidenceError(f'Cannot read evidence {path}: {error}') from error


def records(server, filename):
    paths = sorted(server.rglob(filename))
    require(bool(paths), f'Missing {filename} under {server}')
    result = defaultdict(list)
    for path in paths:
        try:
            for line in path.read_text().splitlines():
                row = json.loads(line)
                require(isinstance(row.get('request_id'), str), f'Missing request ID in {path}')
                result[row['request_id']].append(row)
        except (OSError, ValueError) as error:
            raise EvidenceError(f'Invalid evidence {path}: {error}') from error
    return result


def exactly_one(rows, request_id, filename):
    values = rows.get(request_id, [])
    require(len(values) == 1, f'{request_id}: expected one {filename}, got {len(values)}')
    return values[0]


def geometry_counts(server):
    paths = sorted(server.rglob('cache-geometry.json'))
    require(len(paths) == 1, 'Require one native worker geometry manifest')
    geometry = read(paths[0])
    recurrent, attention = [], []
    for group in geometry['groups']:
        (recurrent if group['recurrent'] else attention).extend(group['layers'])
    require(len(set(recurrent + attention)) == len(recurrent + attention), 'Duplicate native layers')
    mtp = [name for name in attention if name.startswith('mtp.') or '.mtp.' in name]
    require(bool(attention) and bool(recurrent) and bool(mtp), 'Missing target, recurrent or MTP geometry')
    archive_bytes = geometry['cpu_archive_bytes_per_token']
    fp8 = geometry.get('archive_codec') == 'raw_fp8_e4m3fn_v_fp8_e4m3fn_v1'
    width = 2 if fp8 else 3
    require(type(archive_bytes) is int and archive_bytes > 0 and archive_bytes % width == 0,
            'Invalid immutable raw K + FP8 V archive geometry')
    return dict(working_attention_layers=len(attention), mtp_attention_layers=len(mtp),
                recurrent_tensors=2 * len(recurrent), draft_layers=0,
                target_bytes_per_token=archive_bytes // width * 2)


def check(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / 'manifest.json')
    require(read(directory / 'status.json')['status'] == 'completed', 'Correctness run is not completed')
    arms, cases, workload = manifest['arms'], manifest['cases'], manifest['workload']
    agent_scheduling = manifest.get('kind') == 'agent_scheduling'
    allowed = {'A0','A1','A2','A3','P1','Pttl','Ppressure'} if agent_scheduling else {'B3','B4'}
    users = workload.get('concurrency', 8) if agent_scheduling else 8
    require(users == (4 if agent_scheduling else 8), 'Unexpected native agent concurrency')
    require(bool(arms) and set(arms) <= allowed, 'Correctness gate requires asynchronous restore arms')
    require(bool(cases) and set(cases) <= {'agents', 'burst'}, 'Gate requires multi-round agent cases')
    require(len(set(arms)) == len(arms) and len(set(cases)) == len(cases), 'Duplicate arms/cases')
    turns = workload['turns']
    output_tokens = workload['output_tokens' if agent_scheduling else 'agent_tokens']
    pairs = manifest['pairs']
    require(type(turns) is int and turns >= 2 and type(output_tokens) is int and output_tokens > 0
            and type(pairs) is int and pairs > 0, 'Invalid expected workload')
    proofs, run_reports = [], []
    expected_runs = {f'pair-{pair:02d}-{arm}' for pair in range(pairs) for arm in arms}
    require({p.name for p in directory.glob('pair-*')} == expected_runs, 'Unexpected or missing arm directories')
    all_responses = set()
    for pair in range(pairs):
        for arm in arms:
            profile = read(directory / f'{arm}.json')
            require(profile.get('verify_session_restore') is True and profile['kvmem'].get('async_restore') is True,
                    f'{arm}: restore byte oracles or asynchronous restore disabled')
            require(profile.get('draft_method') == 'mtp' and profile.get('draft_tokens', 0) > 0,
                    f'{arm}: native MTP is required for this gate')
            root = directory / f'pair-{pair:02d}-{arm}'
            server = root / 'server'
            require(not list(server.rglob('*mismatch*.json')), f'{root.name}: restore mismatch report present')
            counts = geometry_counts(server)
            async_rows = records(server, 'async_restore.jsonl')
            state_rows = records(server, 'verified_state_restores.jsonl')
            full_rows = records(server, 'verified_restores.jsonl')
            used = set()
            pressure_only = profile['kvmem'].get('active_resident_pressure_only', False)
            require(type(pressure_only) is bool, 'Invalid retention policy in profile')
            pressure_isolation = arm == 'P1' and pressure_only and workload.get('distinct_users')
            for case in cases:
                traces = read(root / case / 'traces.json')
                result = read(root / case / 'results.json')
                require(len(traces) == users * turns and result['requests'] == len(traces)
                        and result['output_tokens'] == len(traces) * output_tokens,
                        f'{root.name}/{case}: incomplete {users}-user workload')
                sessions = defaultdict(dict)
                for row in traces:
                    response_id = row['response_id']
                    require(isinstance(response_id, str) and response_id.startswith('cmpl-')
                            and response_id not in all_responses, 'Missing or duplicate client response ID')
                    all_responses.add(response_id)
                    require(len(row['output_ids']) == output_tokens
                            and all(type(t) is int and t >= 0 for t in row['output_ids'])
                            and row['usage']['completion_tokens'] == output_tokens
                            and sum(n for _, n in row['timeline']) == output_tokens,
                            f'{response_id}: incomplete output stream')
                    require(type(row['turn']) is int and row['turn'] not in sessions[row['session']],
                            f'{response_id}: duplicate or invalid turn')
                    sessions[row['session']][row['turn']] = row
                require(len(sessions) == users and all(set(rows) == set(range(turns)) for rows in sessions.values()),
                        f'{case}: missing users/turns')
                if workload.get('distinct_users'):
                    for turn in range(turns):
                        identities={rows[turn].get('input_sha256') for rows in sessions.values()}
                        require(len(identities)==users and all(isinstance(h,str) and
                                re.fullmatch('[0-9a-f]{64}',h) for h in identities),
                                'User isolation requires distinct complete input histories in every turn')
                long_hot_returns = 0
                for session, rows in sessions.items():
                    for turn in range(1, turns):
                        row, previous = rows[turn], rows[turn - 1]
                        pattern = re.compile(re.escape(row['response_id']) + r'-0-[0-9a-f]{8}')
                        matches = [rid for rid in async_rows if pattern.fullmatch(rid)]
                        require(len(matches) == 1, f'{row["response_id"]}: missing/ambiguous async restore')
                        rid = matches[0]
                        require(rid not in used, f'{rid}: restore reused by multiple client requests')
                        used.add(rid)
                        restored = exactly_one(async_rows, rid, 'async_restore')
                        state = exactly_one(state_rows, rid, 'verified_state_restores')
                        full = exactly_one(full_rows, rid, 'verified_restores')
                        history = previous['input_tokens'] + output_tokens - 1
                        active = restored['active_tokens']
                        require(type(active) is int and 0 < active <= history < row['input_tokens']
                                and active <= profile['kvmem']['select_budget'] + profile['kvmem']['gen_budget'],
                                f'{rid}: invalid restored active/history extent')
                        require(all(proof['history_tokens'] == history for proof in (restored, state, full))
                                and state['working_tokens'] == active and full['working_tokens'] == active,
                                f'{rid}: restore does not match previous complete terminal checkpoint')
                        require(restored['cancelled'] is False and restored['model_bytes'] > 0
                                and restored['uploaded_bytes'] >= 0 and restored['finished'] >= restored['started'],
                                f'{rid}: cancelled or incomplete asynchronous restore')
                        if agent_scheduling:
                            resident_tokens = restored.get('resident_active_tokens', 0)
                            require(resident_tokens in (0, active) and
                                    (not resident_tokens or restored['uploaded_bytes'] == 0),
                                    f'{rid}: resident view unexpectedly uploaded historical KV')
                            require(arm != 'A0' or resident_tokens == 0, f'{rid}: baseline used a resident view')
                            if pressure_isolation:
                                require(resident_tokens == active,
                                        f'{rid}: pressure-only isolation lost a view despite available capacity')
                                if row['begin'] - previous['end'] >= 5.9:
                                    long_hot_returns += 1
                        require(state['checkpoint_kind'] == 'M' and state['pressure_restore'] is False
                                and state['recurrent_byte_equal'] is True and full['byte_equal'] is True,
                                f'{rid}: missing exact M checkpoint byte/state proof')
                        require(state['recurrent_tensors'] == counts['recurrent_tensors']
                                and all(full[key] == counts[key] for key in (
                                    'working_attention_layers', 'mtp_attention_layers', 'recurrent_tensors', 'draft_layers'))
                                and full['target_kv_bytes'] == active * counts['target_bytes_per_token'],
                                f'{rid}: incomplete native target/MTP/recurrent oracle coverage')
                        proofs.append(dict(pair=pair, arm=arm, case=case, session=session, turn=turn,
                            response_id=row['response_id'], request_id=rid, history_tokens=history, active_tokens=active))
                if pressure_isolation:
                    require(long_hot_returns > 0, 'Pressure-only isolation did not exercise a hot return beyond the old TTL')
                    before = read(root/case/'resources-before.json')['active_residency']
                    after = read(root/case/'resources-after.json')['active_residency']
                    require(before.get('retention_policy') == after.get('retention_policy') == 'pressure'
                            and before['expired'] == after['expired'],
                            'Pressure-only isolation expired idle views or used a TTL cache')
            if agent_scheduling and arm != 'A0':
                require(any(async_rows[rid][0].get('resident_active_tokens',0) for rid in used),
                        f'{root.name}: no exact resident view was exercised')
            if arm in ('P1','Pttl','Ppressure'):
                require(profile['kvmem'].get('offline_agent_policy') is True, 'P1 disabled the unified policy')
                paths = list(server.rglob('offline_decisions.jsonl'))
                require(len(paths) == 1, 'Missing unique offline decision evidence')
                decisions = [json.loads(line) for line in paths[0].read_text().splitlines()]
                scheduled = {rid for row in decisions for rid in row['scheduled']}
                require(used <= scheduled, 'Some restored requests never entered the policy public batch')
            if agent_scheduling and (workload.get('lifecycle_checks') or workload.get('retirement_barrier') or workload.get('distinct_users')):
                import importlib.util
                spec = importlib.util.spec_from_file_location('lifecycle_gate', Path(__file__).with_name('agent_lifecycle_checks.py'))
                lifecycle = importlib.util.module_from_spec(spec);spec.loader.exec_module(lifecycle)
                capture = next(server.rglob('cache-geometry.json')).parent
                try:
                    for case in cases:
                        lifecycle.validate_quiescent(read(root/case/'resources-cleanup.json'))
                    if workload.get('lifecycle_checks') and arm != 'A0':
                        report = read(root/'lifecycle/results.json')
                        require(report.get('pressure_only', False) is pressure_only,
                                'Lifecycle coverage differs from the selected retention policy')
                        lifecycle.validate_report(capture,report)
                    if arm == 'Ppressure':
                        require(workload.get('retirement_barrier') is True, 'Pressure probe lacks retirement barrier')
                        for case in cases:
                            before = read(root/case/'resources-before.json')
                            after = read(root/case/'retirement-barrier.json')
                            lifecycle.validate_quiescent(after)
                            if pressure_only:
                                require(after['active_residency'].get('retention_policy') == 'pressure',
                                        'Pressure probe unexpectedly used a TTL cache')
                            require(after['active_residency']['pressure_evictions']>
                                    before['active_residency']['pressure_evictions'], 'No forced idle-view pressure eviction')
                        require(any(async_rows[rid][0].get('resident_active_tokens',0)==0 and
                                    async_rows[rid][0]['uploaded_bytes']>0 for rid in used),
                                'Revoked active views were not restored from CPU history')
                except (AssertionError, FileNotFoundError, ValueError) as error:
                    raise EvidenceError(str(error)) from error
            elif arm == 'Ppressure':
                raise EvidenceError('Pressure probe requires lifecycle resource evidence')
            run_reports.append(dict(pair=pair, arm=arm, restored_requests=len(used), geometry=counts))
    expected = pairs * len(arms) * len(cases) * users * (turns - 1)
    require(len(proofs) == expected, 'Incorrect final restore proof count')
    return dict(passed=True, scope='Exact asynchronous M checkpoint restores; GPU throughput is a separate gate',
                output=str(directory), expected_restores=expected, proved_restores=len(proofs),
                native_request_mapping='OpenAI single prompt: response_id + -0- + 8 lowercase hex characters',
                runs=run_reports, proofs=proofs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', dest='directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Default: directory/restore-gate.json')
    args = parser.parse_args()
    try:
        report = check(args.directory)
    except (EvidenceError, KeyError, TypeError) as error:
        report = dict(passed=False, output=str(args.directory.resolve()), error=str(error))
    target = args.output or args.directory / 'restore-gate.json'
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps({k: v for k, v in report.items() if k not in ('proofs', 'runs')}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

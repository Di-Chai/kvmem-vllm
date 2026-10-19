"""Compare legacy/incremental clients on one unchanged model server.

This is an independent host-overhead diagnostic, not a scored quality cohort or
an eight-way decode qualification. Prior results seed bookkeeping only; selected
questions always create fresh sessions and use the original full inputs/caps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import qualify_block_graph as q


FIELDS = ('token_sha256', 'prompt_sha256', 'query_sha256', 'prompt_tokens')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def select_cases(rows, groups=2, questions=9):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['id'].rsplit('_q', 1)[0], []).append(row)
    chosen = list(grouped.items())[:groups]
    if len(chosen) != groups or any(len(items) < questions for _, items in chosen):
        raise ValueError('Insufficient independent original questions')
    return {name: items[:questions] for name, items in chosen}


def seed_progress(destination, fixture):
    """Copy verified immutable metadata, never treat it as fresh inference."""
    rows = fixture['rows']
    selected_groups = set(fixture['selected'])
    seeded = []
    source = Path(fixture['seed_dataset'])/'cases'
    for row in rows:
        if row['id'].rsplit('_q', 1)[0] in selected_groups:
            continue
        original = source/row['id']/'result.json'
        data = original.read_bytes()
        record = json.loads(data)
        digest = hashlib.sha256(data).hexdigest()
        expected = (row['token_sha256'], row['prompt_sha256'], row['query_sha256'], row['input_tokens'])
        if (digest != fixture['seed_sha256'][row['id']] or
                record.get('status') != 'completed' or record.get('id') != row['id'] or
                tuple(record.get(key) for key in FIELDS) != expected):
            raise ValueError('Bookkeeping seed source identity differs')
        path = destination/'cases'/row['id']/'result.json'
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        seeded.append(dict(id=row['id'], source=str(original), sha256=digest))
    save(destination/'seed-manifest.json', dict(scope='Bookkeeping only; not fresh answers or quality evidence',
        records=seeded))
    return seeded


def summarize_trial(dataset, selected):
    cases, gaps, warm_intervals = [], [], []
    for items in selected.values():
        previous = None
        first_warm = None
        for index, row in enumerate(items):
            path = dataset/'cases'/row['id']
            record = json.loads((path/'result.json').read_text())
            request = json.loads((path/'request.json').read_text())
            expected = (row['token_sha256'], row['prompt_sha256'], row['query_sha256'], row['input_tokens'])
            if (record.get('status') != 'completed' or
                    tuple(record.get(key) for key in FIELDS) != expected or
                    record['prefix_reused'] != bool(index)):
                raise ValueError('Fresh diagnostic answer/prefix reuse differs')
            attempt = json.loads((path/'attempt.json').read_text())
            response_at = (path/'response.json').stat().st_mtime_ns/1e9
            if previous is not None:
                gap = attempt['started_at']-previous
                if gap < 0:
                    raise ValueError('Nonmonotonic sequential request evidence')
                gaps.append(gap)
            cases.append(dict(id=row['id'], token_sha256=row['token_sha256'],
                query_sha256=row['query_sha256'], sampling_sha256=request['sampling_sha256'],
                warm=bool(index), seconds=record['seconds'],
                completion_tokens=record['usage']['completion_tokens'],
                session=attempt['session']))
            if index == 1:
                first_warm = attempt['started_at']
            previous = response_at
        warm_intervals.append((first_warm, previous))
    warm = [row for row in cases if row['warm']]
    return dict(fresh_cases=len(cases), warm_cases=len(warm),
        response_to_next_attempt_seconds=gaps,
        mean_client_gap_seconds=statistics.fmean(gaps),
        common_warm_wall_interval=[max(v[0] for v in warm_intervals), min(v[1] for v in warm_intervals)]
            if max(v[0] for v in warm_intervals) < min(v[1] for v in warm_intervals) else None,
        warm_mean_http_seconds=statistics.fmean(row['seconds'] for row in warm),
        warm_output_tokens=sum(row['completion_tokens'] for row in warm), cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'capture', 'output', 'fixture'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    if fixture['schema'] != 'client-host-overhead-1':
        raise ValueError('Unknown diagnostic fixture')
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = Path(__file__).resolve().parents[1]
    selected = fixture['selected']
    case_ids = [row['id'] for items in selected.values() for row in items]
    trials = []
    for mode in ('legacy', 'incremental'):
        directory = args.output/mode
        directory.mkdir()
        seeded = seed_progress(directory, fixture)
        script = runtime/('legacy' if mode == 'legacy' else 'scripts')/'benchmark_memory.py'
        command = [sys.executable, '-B', str(script), '--reference', fixture['reference'],
            '--config', str(args.config), '--capture', str(args.capture), '--output', str(directory),
            '--workers', '8', '--active-groups', '8', '--cpu-headroom-gib', '16', '--timeout', '3600']
        for ident in case_ids:
            command += ['--case-id', ident]
        begin = time.time()
        begin_mono = time.monotonic()
        with (directory/'client.log').open('x') as log:
            # Descendants remain in this qualifier's process group. The outer
            # frozen controller owns cancellation and all GPU cleanup.
            code = subprocess.call(command, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                   stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f'{mode} diagnostic failed; retain logs, never retry answers')
        released = q.wait_released(args.capture, [], timeout=180)
        archive = released['archive']
        if archive.get('disk_read_bytes', 0) or archive.get('disk_write_bytes', 0):
            raise ValueError('Unexpected disk KV in CPU-only diagnostic')
        trial = dict(mode=mode, started_at=begin, finished_at=time.time(),
            step_summary=q.summarize_steps(args.capture, begin_mono, time.monotonic()),
            client_sha256=sha(script), bookkeeping_seed_count=len(seeded),
            **summarize_trial(directory, selected))
        save(directory/'diagnostic.json', trial)
        trials.append(trial)
        print(json.dumps({k:v for k,v in trial.items() if k!='cases'}), flush=True)
    bindings = lambda trial:[(r['id'], r['token_sha256'], r['query_sha256'], r['sampling_sha256'])
                            for r in trial['cases']]
    if bindings(trials[0]) != bindings(trials[1]):
        raise ValueError('Paired original source/sampling differs')
    result = dict(status='passed', all_owned_sessions_released=True, trials=trials,
        scope='Two original shared histories, eight clients maximum; not C8 compute. '
            'Legacy then incremental on one unchanged server. Original full inputs and sampling. '
            'Separate first cold question from repeated warm questions; seeded records are never fresh answers. '
            'Single ordered pair diagnoses host gaps; it does not establish a controlled general throughput gain.')
    save(args.output/'results.json', result)


if __name__ == '__main__':
    main()

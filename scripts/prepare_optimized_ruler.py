"""Freeze current optimized execution against the existing 260 RULER inputs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

from benchmark_prefill_publication_pipeline import read, runtime_origins, save, sha, require
from benchmark_prefill_variant import differences
from ruler_execution import OPTIMIZED, ENABLED, DISABLED, validate_execution
from ruler_protocol import load_scorer, validate_cohort


def prepare(root, source, repo, *, query_replay_backend='eager', query_replay_cobatch=False):
    root, source, repo = (path.resolve() for path in (root, source, repo))
    fixture = read(source / 'fixture.json')
    require(sha(source / 'fixture.json') == read(source / 'dataset-summary.json')['fixture_sha256'],
            'Source fixture changed')
    require(sha(source / 'profile.json') == fixture['profile_sha256'], 'Source profile changed')
    cohort = validate_cohort(fixture['rows'], 20)
    require(cohort['cases'] == 260 and fixture['target_length'] == 262144, 'Wrong RULER cohort')
    load_scorer(Path(fixture['upstream_path']))
    for row in fixture['rows']:
        path = Path(fixture['input_directory']) / (row['id'] + '.i32')
        require(path.stat().st_size == row['served_input_tokens'] * 4 and
                sha(path) == row['served_sha256'], 'Frozen input changed: ' + row['id'])
    original = read(source / 'profile.json')
    profile = deepcopy(original)
    profile['prefill_chunk_tokens'] = 2048
    kv = profile['kvmem']
    kv.update({name: True for name in ENABLED})
    kv.update({name: False for name in DISABLED})
    kv.update(action_native_scores='reuse', assembly_chunk_blocks=128,
              prefill_physical_max_chunks=7, action_native_max_rows=512,
              cpu_seal_hash_workers=8, cpu_seal_pack_workers=1)
    kv.update(query_replay_backend=query_replay_backend,
              query_replay_cobatch=query_replay_cobatch, query_replay_max_bytes=64*1024*1024)
    fixture['execution_profile'] = OPTIMIZED
    fixture['query_replay_execution'] = dict(backend=query_replay_backend, cobatch=query_replay_cobatch)
    validate_execution(profile, fixture)
    names = set(read(source / 'runtime-proof.json')['files'])
    for folder in ('src/vllm_kvmem', 'scripts', 'csrc', 'third_party/qw3'):
        names.update(str(path.relative_to(repo)) for path in (repo / folder).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts and
                     path.suffix in ('.py', '.json', '.cpp', '.hpp', '.h', '.cu', '.cuh', '.so'))
    names.update(('setup.py', 'pyproject.toml'))
    origins, inherited = runtime_origins(repo, source, names)
    hashes = {name: sha(path) for name, path in origins.items()}
    root.mkdir(parents=True, exist_ok=False)
    for name, origin in origins.items():
        target = root / 'runtime' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        require(sha(target) == hashes[name], 'Runtime source changed during freeze: ' + name)
    save(root / 'runtime-proof.json', dict(files=hashes, inherited_utilities=inherited,
        source=str(repo), head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        working_diff_sha256=__import__('hashlib').sha256(subprocess.check_output(
            ['git', 'diff', 'HEAD'], cwd=repo)).hexdigest()))
    save(root / 'profile.json', profile)
    fixture['source_fixture_sha256'] = sha(source / 'fixture.json')
    fixture['profile_sha256'] = sha(root / 'profile.json')
    save(root / 'fixture.json', fixture)
    summary = read(source / 'dataset-summary.json')
    summary.update(fixture_sha256=sha(root / 'fixture.json'), execution_profile=OPTIMIZED)
    save(root / 'dataset-summary.json', summary)
    save(root / 'manifest.json', dict(execution_profile=OPTIMIZED, source_campaign=str(source),
        source_fixture_sha256=fixture['source_fixture_sha256'],
        fixture_sha256=sha(root / 'fixture.json'), profile_sha256=sha(root / 'profile.json'),
        profile_changes=differences(original, profile), cases=260,
        reused_all_original_input_ids=True, reused_sampling_and_seeds=True,
        scores_from_previous_campaigns_reused=False, dsh_started=False, mab_resumed=False))
    require(all(sha(path) == hashes[name] for name, path in origins.items()), 'Runtime changed during freeze')
    print(json.dumps(dict(status='prepared', campaign=str(root), cases=260,
                          execution_profile=OPTIMIZED)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'source', 'repo'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--query-replay-backend', choices=('eager', 'compiled'), default='eager')
    parser.add_argument('--query-replay-cobatch', action='store_true')
    args = parser.parse_args()
    prepare(args.campaign, args.source, args.repo,
            query_replay_backend=args.query_replay_backend, query_replay_cobatch=args.query_replay_cobatch)

"""Freeze independent fixed-256K sustained 300-second A/B request streams."""
import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import benchmark_block_graph as owned
import benchmark_graph_v2 as shared
import benchmark_graph_v2_performance as pairs
import graph_v2_protocol as protocol
from graph_v2_protocol import (STREAM_SCHEMA as SCHEMA,
    STREAM_WINDOW_SECONDS as WINDOW_SECONDS,
    STREAM_WARMUP_COMPLETIONS as WARMUP_COMPLETIONS, STREAM_SLOTS as SLOTS)

ROOT = Path(__file__).resolve().parents[1]
HELPERS = pairs.HELPERS+('benchmark_graph_v2_stream.py','qualify_graph_v2_stream.py')
require, sha, write = protocol.require, protocol.sha, owned.write


def read(path):
    return json.loads(path.read_text())


def job_order(count):
    require(type(count) is int and count in (1,3), 'Use one screening pair or three alternating pairs')
    return [dict(id=f'p{i+1}-{role}',pair=i+1,role=role) for i in range(count)
            for role in (('baseline','candidate') if i%2 == 0 else ('candidate','baseline'))]


def repeat_inputs(fixture, root, count):
    require(type(count) is int and count >= 32 and count%8 == 0 and len(fixture['rows']) == 8,
            'Freeze at least 32 candidates, in complete cycles of eight original cases')
    original_directory = Path(fixture['input_directory'])
    aliases = root/'dataset/repeated'
    aliases.mkdir(parents=True)
    rows = []
    for index in range(count):
        original = fixture['rows'][index%8]
        row = deepcopy(original)
        row.update(id=f's{index:04d}-{original["id"]}', original_case=original['id'],
                   repetition=index//8, source_request=f's{index:04d}-{original["id"]}')
        source, target = original_directory/(original['id']+'.i32'), aliases/(row['id']+'.i32')
        require(sha(source) == row['served_sha256'], 'Original frozen input changed')
        shutil.copyfile(source,target)
        require(sha(target) == row['served_sha256'], 'Independent input alias differs')
        rows.append(row)
    return dict(fixture,rows=rows,input_directory=str(aliases),stream_schema=SCHEMA,
        stream_window_seconds=WINDOW_SECONDS,stream_warmup_completions=WARMUP_COMPLETIONS,
        original_cases=[r['id'] for r in fixture['rows']],candidate_queue_size=count,
        admission='Eight FIFO slots; retain CPU history credits through individual release; no waves. '
                  'At least eight complete releases before T0, fixed T1=T0+300; no admission at/after T1.')


def hashes(directory):
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    write(root/'status.json',dict(status='preparing',at=time.time()))
    try:
        candidate = args.candidate or 'raw-k-cache'
        configs = pairs.profiles(read(args.profile),args.port,candidate)
        baseline, changed = configs.values()
        identity = shared.model_identity(baseline)
        write(root/'niah-A.json',baseline)
        original = shared.freeze_niah(SimpleNamespace(reference=args.reference,scope='smoke8'),root,baseline)
        write(root/'original-fixture.json',original)
        fixture = repeat_inputs(original,root,args.requests)
        fixture['candidate'] = candidate
        write(root/'fixture.json',fixture)
        jobs, runtime_hashes = job_order(args.pairs), None
        for job in jobs:
            profile = baseline if job['role'] == 'baseline' else changed
            source = root/(job['id']+'.json')
            write(source,profile)
            target = root/job['id']
            owned.prepare(SimpleNamespace(output=target,fixture=root/'fixture.json',arm='D',profile=source,
                qualifier=ROOT/'scripts/qualify_graph_v2_stream.py',synchronous=False,port=args.port,
                client_timeout_seconds=7200))
            for name in HELPERS:
                shutil.copyfile(ROOT/'scripts'/name,target/'runtime/scripts'/name)
            current = hashes(target/'runtime')
            require(any(k.startswith('src/vllm_kvmem/_native') and k.endswith('.so') for k in current),
                    'Missing native extension')
            require(runtime_hashes is None or current == runtime_hashes, 'Source changed while freezing stream jobs')
            runtime_hashes = current
            manifest = read(target/'manifest.json')
            manifest.update(runtime_sha256=current,stream_schema=SCHEMA,candidate=candidate,job=job)
            write(target/'manifest.json',manifest)
            job['manifest_sha256'] = sha(target/'manifest.json')
        frozen = root/'frozen'
        frozen.mkdir()
        for name in HELPERS:
            shutil.copyfile(ROOT/'scripts'/name,frozen/name)
            require(sha(frozen/name) == runtime_hashes['scripts/'+name], 'Controller changed during freeze')
        patch = subprocess.check_output(['git','diff','HEAD','--','src','csrc','third_party','scripts'],cwd=ROOT)
        (root/'source.diff').write_bytes(patch)
        status = subprocess.check_output(['git','status','--porcelain','--','src','csrc','third_party','scripts'],cwd=ROOT,text=True)
        maximum = max(r['served_input_tokens'] for r in fixture['rows'])+8192+16384
        credit = ((maximum+31)//32)*32*17*4*256*2
        require(SLOTS*credit <= fixture['cpu_payload_credits_bytes'], 'Cannot admit eight intact histories')
        write(root/'manifest.json',dict(schema=SCHEMA,candidate=candidate,pairs=args.pairs,jobs=jobs,
            variants=pairs.CANDIDATES[candidate]['variants'],changed_profile_field='kvmem.'+pairs.CANDIDATES[candidate]['field'],
            fixture_sha256=sha(root/'fixture.json'),original_fixture_sha256=sha(root/'original-fixture.json'),
            input_sha256=hashes(root/'dataset'),runtime_sha256=runtime_hashes,frozen_sha256=hashes(frozen),
            model_identity=identity,source_diff_sha256=sha(root/'source.diff'),source_dirty=bool(status.strip()),
            source_worktree_status=status,source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            window_seconds=WINDOW_SECONDS,warmup_completions=WARMUP_COMPLETIONS,slots=SLOTS,requests=args.requests,
            start_cpu_available_bytes=SLOTS*credit+24*1024**3,minimum_cpu_available_bytes=8*1024**3,
            minimum_gpu_free_mib=4096,created_at=time.time(),
            scope='Fixed original256K request stream: eight independent slots, individual release/refill, '
                  'eight completed warmups, fixed300s, then drain outside measurement. No retry or altered sampling.'))
        write(root/'status.json',dict(status='prepared',at=time.time()))
        print(json.dumps(dict(prepared=str(root),candidate=candidate,jobs=[j['id'] for j in jobs],gpu_started=False)),flush=True)
    except BaseException as error:
        write(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise


def validate(root):
    manifest = read(root/'manifest.json')
    candidate = manifest['candidate']
    require(manifest['schema'] == SCHEMA and candidate in pairs.CANDIDATES and
        manifest['variants'] == pairs.CANDIDATES[candidate]['variants'], 'Unknown frozen stream candidate')
    require([{k:v for k,v in j.items() if k!='manifest_sha256'} for j in manifest['jobs']] == job_order(manifest['pairs']),
            'Paired AB/BA/AB order changed')
    require((manifest['window_seconds'],manifest['warmup_completions'],manifest['slots']) ==
            (WINDOW_SECONDS,WARMUP_COMPLETIONS,SLOTS), 'Fixed window/warmup/slots changed')
    require(sha(root/'fixture.json') == manifest['fixture_sha256'] and
        sha(root/'original-fixture.json') == manifest['original_fixture_sha256'] and
        sha(root/'source.diff') == manifest['source_diff_sha256'], 'Frozen fixture/source diff changed')
    require(hashes(root/'dataset') == manifest['input_sha256'] and
            hashes(root/'frozen') == manifest['frozen_sha256'], 'Frozen input or controller changed')
    fixture = read(root/'fixture.json')
    require(fixture['candidate'] == candidate and fixture['candidate_queue_size'] == manifest['requests'] and
            len(fixture['rows']) == manifest['requests'], 'Candidate queue changed')
    roles = {}
    for job in manifest['jobs']:
        target = root/job['id']
        require(sha(target/'manifest.json') == job['manifest_sha256'], 'Child manifest changed')
        child = read(target/'manifest.json')
        require(hashes(target/'runtime') == child['runtime_sha256'] == manifest['runtime_sha256'], 'Child runtime changed')
        require(sha(target/'profile.json') == child['profile_sha256'] and
            sha(target/'fixture.json') == child['fixture_sha256'] == manifest['fixture_sha256'], 'Child profile/fixture changed')
        profile = read(target/'profile.json')
        require(job['role'] not in roles or roles[job['role']] == profile, 'Repeated profile differs')
        roles[job['role']] = profile
    pairs.validate_profiles(dict(zip(manifest['variants'],(roles['baseline'],roles['candidate']))),candidate)
    for group in ('config_sha256','tokenizer_sha256'):
        for name,value in manifest['model_identity'][group].items():
            require(sha(name) == value, 'Model/tokenizer identity changed')
    for name,value in manifest['model_identity']['weight_stat_identity'].items():
        stat = Path(name).stat()
        require(dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns) == value, 'Model weights changed')
    return manifest


def ratio(value, baseline):
    return value/baseline if value is not None and baseline is not None and baseline > 0 else None


def comparison(root, manifest):
    reports = []
    for number in range(1,manifest['pairs']+1):
        sides = {}
        for role in ('baseline','candidate'):
            directory = root/f'p{number}-{role}'/'qualification'
            result, metrics = read(directory/'results.json'),read(directory/'measurements.json')
            require(result['status'] == 'passed' and result['schema'] == SCHEMA and metrics['schema'] == SCHEMA and
                sha(directory/'measurements.json') == result['measurements_sha256'], 'Missing/changed stream result')
            sides[role] = (result,metrics)
        (left,a),(right,b) = sides['baseline'],sides['candidate']
        common = left['cases'].keys() & right['cases'].keys()
        require(common and all(all(left['cases'][key][f] == right['cases'][key][f]
            for f in ('input_sha256','seed','original_case','input_tokens')) for key in common), 'Paired source input differs')
        identical = all(all(left['cases'][key][f] == right['cases'][key][f]
            for f in ('output_ids_sha256','selection_sha256')) for key in common)
        candidate = manifest['candidate']
        if pairs.CANDIDATES[candidate]['field'] == 'gpu_raw_k_cache_bytes':
            observed = b['raw_cache_exercised']
        elif candidate == 'cpu-direct-record-seal':
            observed = b['direct_seal_bytes_observed_delta'] is not None and b['direct_seal_bytes_observed_delta'] > 0
        else:
            # No timestamped cumulative upload source is guaranteed by the
            # existing runtime. Whole-run terminal totals cannot prove T0..T1.
            observed = b.get('async_metadata_window_observed') is True
        conditions = dict(candidate_exercised_in_window=observed, common_outputs_and_selections_identical=identical,
            native_c8_observed=all(m['native']['actual_batch_peak'] >= 8 for m in (a,b)),
            public_work_boundaries_available=all(m['public_prefill']['boundary_brackets_complete'] for m in (a,b)))
        reports.append(dict(pair=number,conditions=conditions,common_requests=len(common),
            baseline=a,candidate=b,verdict='descriptive_fixed_window' if all(conditions.values()) else 'inconclusive',
            completed_public_prefill_observed_rate_ratio=ratio(b['public_prefill']['completed_observed_tokens_per_second'],
                a['public_prefill']['completed_observed_tokens_per_second']),
            accepted_output_rate_ratio=ratio(b['accepted_output_tokens_per_second'],a['accepted_output_tokens_per_second'])))
    return dict(schema=SCHEMA,candidate=manifest['candidate'],pairs=reports,
        performance_gain_established=False,statistical_significance_claimed=False,
        scope='Fixed300s after complete warmup, with continuous refill. Different progress admits different requests; '
              'ratios describe executed public work and accepted output, not unique-input throughput or pure decode. '
              'Zero output is valid; a zero baseline produces a null output ratio. Review boundaries, quality, '
              'selection/output drift and repeated-pair consistency before any performance conclusion.')


def run(args):
    root = args.output.resolve()
    with (root/'.campaign.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = validate(root)
        require(Path(__file__).resolve() == root/'frozen/benchmark_graph_v2_stream.py', 'Use the frozen stream controller')
        require(read(root/'status.json')['status'] == 'prepared', 'No retry of any started/failed/completed campaign')
        require(args.candidate in (None,manifest['candidate']), 'Requested candidate differs')
        write(root/'status.json',dict(status='running',at=time.time()))
        try:
            with owned.cleanup_on_termination() as cleanup:
                for job in manifest['jobs']:
                    require(shared.cpu_available() >= manifest['start_cpu_available_bytes'], 'Insufficient CPU history headroom')
                    directory = root/job['id']
                    process = None
                    try:
                        with (directory/'controller.log').open('x') as log, (directory/'resources.jsonl').open('x') as telemetry:
                            process = subprocess.Popen([sys.executable,'-B',str(directory/'runtime/scripts/benchmark_graph_v2_stream.py'),
                                'arm-run','--output',str(directory)],stdout=log,stderr=subprocess.STDOUT,
                                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
                            while process.poll() is None:
                                sample = pairs.sample_resources()
                                telemetry.write(json.dumps(sample)+'\n'); telemetry.flush()
                                require(sample['cpu_available_bytes'] >= manifest['minimum_cpu_available_bytes'] and
                                    sample['gpu_free_mib'] >= manifest['minimum_gpu_free_mib'], 'All-phase resource guard reached')
                                time.sleep(1)
                        require(process.returncode == 0 and read(directory/'status.json')['status'] == 'passed',
                                'Stream job failed; retained evidence, no retry')
                    finally:
                        if process is not None:
                            require(not owned.stop_owned_groups((process,)), 'Owned stream controller did not stop')
                cleanup()
            write(root/'comparison.json',comparison(root,manifest))
            write(root/'status.json',dict(status='completed_pending_review',at=time.time()))
        except BaseException as error:
            write(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','validate','run','arm-run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',choices=tuple(pairs.CANDIDATES))
    parser.add_argument('--pairs',type=int,choices=(1,3),default=1)
    parser.add_argument('--requests',type=int,default=128)
    parser.add_argument('--profile',type=Path,default=ROOT/'artifacts/graph-overhead-20261018/gpu-r2/after/profile.json')
    parser.add_argument('--reference',type=Path,default=Path('/data/chaidi/qw3-experiments/cacheblend_20260925/official_template_20260927'))
    parser.add_argument('--port',type=int,default=18688)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'validate':
        manifest = validate(args.output.resolve())
        print(json.dumps(dict(valid=True,candidate=manifest['candidate'],pairs=manifest['pairs'],gpu_started=False)))
    elif args.action == 'arm-run':
        require(Path(__file__).resolve() == args.output.resolve()/'runtime/scripts/benchmark_graph_v2_stream.py', 'Use frozen arm runner')
        owned.run(args)
    else:
        run(args)


if __name__ == '__main__':
    main()

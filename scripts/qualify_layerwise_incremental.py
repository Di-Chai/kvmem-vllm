"""Freeze and explicitly run layerwise/incremental compatibility qualifications.

No automatic GPU launch, retries, external grading, or prior answer import.
The first two arms hold execution at one request; the last arm exercises swaps
with two requests and a separately specified tight physical page pool. This is
correctness qualification, not a pure-prefill throughput benchmark.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


REPO = Path(__file__).resolve().parents[1]
ARMS = ('static_serial', 'incremental_serial', 'incremental_tight')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def freeze(source, destination):
    for name in ('src/vllm_kvmem', 'scripts', 'csrc', 'third_party'):
        shutil.copytree(source/name, destination/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    for name in ('pyproject.toml', 'setup.py', 'CMakeLists.txt', 'MANIFEST.in', 'LICENSE'):
        if (source/name).is_file():
            shutil.copy2(source/name, destination/name)
    # No cohort gate: a held final Query can prevent the next serial owner or
    # a suspended owner from making progress when the page pool is tight.
    query = (destination/'src/vllm_kvmem/layerwise_query.py').read_text()
    if "os.environ['KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS']" in query:
        raise ValueError('Source contains a finite-owner benchmark gate')


def prepare(root, *, source=REPO, reference=None, inputs=None, roomy_bytes=48*1024**3,
            tight_bytes, checkpoint_bytes=16*1024**3, cpu_cache_bytes=64_000_000_000,
            port=19446):
    source = Path(source).resolve()
    reference = Path(reference or source/'artifacts/mab-pressure-multiowner-20261105-r10/formal')
    inputs = Path(inputs or source/'artifacts/multiowner-static-native-20261105-r3/b1')
    if root.exists():
        raise ValueError('Preserve every prior qualification attempt; use a fresh root')
    if not 0 < tight_bytes < roomy_bytes or checkpoint_bytes <= 0 or cpu_cache_bytes <= 16*1024**3:
        raise ValueError('Require positive checkpoint capacity and tight < roomy GPU backing')
    profile, fixture = read(reference/'profile.json'), read(inputs/'fixture.json')
    kv = profile['kvmem']
    if [kv[k] for k in ('select_budget', 'gen_budget', 'block_tokens', 'sink_blocks', 'recent_blocks')] != [36864,16384,32,41,0]:
        raise ValueError('Original Active/Gen or recall geometry changed')
    if profile['execution_mode'] != 'compiled_graph' or profile['draft_tokens'] != 4:
        raise ValueError('Require the original compiled/MTP4 configuration')
    rows = deepcopy(fixture['rows'][:2])
    if len(rows) != 2 or len({r['id'].rsplit('_q', 1)[0] for r in rows}) != 2:
        raise ValueError('Need exactly two independent complete cold prefixes')
    bindings = {b['id']: b for b in fixture['bindings']}
    root.mkdir(parents=True)
    copied_bindings = []
    for row in rows:
        origin = Path(row['source'])
        if sha(origin) != row['source_cache_sha256']:
            raise ValueError('Frozen input hash changed')
        destination = root/'inputs'/origin.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(origin, destination)
        row['source'] = str(destination)
        binding = deepcopy(bindings[row['id']])
        request = Path(binding['request'])
        if sha(request) != binding['sha256']:
            raise ValueError('Frozen sampling binding changed')
        copied = root/'bindings'/(row['id']+'.json')
        copied.parent.mkdir(exist_ok=True)
        shutil.copy2(request, copied)
        binding['request'] = str(copied)
        copied_bindings.append(binding)
    fixture.update(rows=rows, cases=2, case_ids=[r['id'] for r in rows],
                   bindings=copied_bindings, qualification=False,
                   restore_digest_qualification=False, performance_observation=True,
                   diagnostic_only=True, external_grading=False)
    # The historical frozen runtime hard-coded these qualified optimizations;
    # the working tree exposes them as default-off execution switches.
    fixture['execution_flags'].update(
        layerwise_pressure_prepared_reads=True,
        layerwise_pressure_post_forward_reap=True,
        layerwise_pressure_submit_on_prepare=True,
        layerwise_pressure_reuse_layer_views=True)
    source_proof = {str(reference/'profile.json'):sha(reference/'profile.json'),
                    str(inputs/'fixture.json'):sha(inputs/'fixture.json')}
    source_runtime = root/'source-runtime'
    freeze(source, source_runtime)
    frozen = manifest(source_runtime)
    files = {}
    for kind in ARMS:
        arm = root/kind
        runtime = arm/'runtime-v1'
        shutil.copytree(source_runtime, runtime)
        dynamic = kind != 'static_serial'
        current = deepcopy(profile)
        # Keep the original scheduler ceiling and graph shapes. The client
        # controls actual concurrency; lowering this ceiling would reject the
        # original 16424-token batch because Query + MTP headroom depends on it.
        current.update(port=port,
                       verify_request_swap=dynamic,
                       request_planning_batch_tokens=profile['max_num_batched_tokens'],
                       kv_cache_memory_bytes=tight_bytes if kind.endswith('tight') else roomy_bytes)
        current['kvmem'].update(cpu_read_cache_bytes=cpu_cache_bytes,
            gpu_memory_admission='incremental' if dynamic else 'static',
            request_swap_layerwise=dynamic,
            request_checkpoint_bytes=checkpoint_bytes if dynamic else 0,
            request_checkpoint_ssd_bytes=0)
        save(arm/'fixture.json', fixture)
        save(arm/'profile.json', current)
        save(arm/'runtime-proof.json', frozen)
        save(arm/'status.json', {'status':'prepared'})
        for name in ('fixture.json','profile.json','runtime-proof.json'):
            files[str((arm/name).relative_to(root))] = sha(arm/name)
    # A separate source tree is intentional evidence of the exact working tree
    # frozen after implementation; no artifact ancestor or answer is required.
    files.update({str(p.relative_to(root)):sha(p) for d in ('inputs','bindings')
                  for p in (root/d).iterdir()})
    shutil.copy2(__file__, root/'launch.py')
    files['launch.py'] = sha(root/'launch.py')
    save(root/'plan.json', dict(arms=ARMS, files=files, source_proof=source_proof,
        frozen_source=str(source), source_runtime_sha256=frozen,
        workers={'static_serial':1,'incremental_serial':1,'incremental_tight':2},
        changes_from_reference=dict(cpu_read_cache_bytes=cpu_cache_bytes,
            roomy_kv_bytes=roomy_bytes,tight_kv_bytes=tight_bytes,
            checkpoint_bytes=checkpoint_bytes,port=port,
            max_num_seqs=profile['max_num_seqs'],request_swap_byte_diagnostics=True),
        scope='Complete original inputs and sampling. First compare serial block selections/full output; '
              'then exercise concurrent swaps and verify raw restored state. No pure-prefill TPS claim.',
        automatic_retry=False,external_grading=False,imported_answers=0))
    save(root/'prepared-proof.json', {'plan_sha256':sha(root/'plan.json')})


def verify(root):
    if sha(root/'plan.json') != read(root/'prepared-proof.json')['plan_sha256']:
        raise ValueError('Frozen plan changed')
    plan = read(root/'plan.json')
    for name, digest in plan['files'].items():
        if sha(root/name) != digest:
            raise ValueError('Frozen control/input changed: '+name)
    for kind in plan['arms']:
        if manifest(root/kind/'runtime-v1') != read(root/kind/'runtime-proof.json'):
            raise ValueError('Frozen runtime changed: '+kind)
    return plan


def arm(root, kind):
    plan = verify(root)
    if kind not in plan['arms']:
        raise ValueError('Unknown arm')
    if kind == 'incremental_tight':
        if not read(root/'serial-comparison.json')['passed']:
            raise ValueError('Serial full-input qualification must pass before forced swaps')
    destination = root/kind
    runtime = destination/'runtime-v1'
    sys.path[:0] = [str(runtime/'src'),str(runtime/'scripts')]
    spec = importlib.util.spec_from_file_location('qualified_memory', runtime/'scripts/run_layerwise_memory.py')
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)
    def frozen(path):
        if path != destination:
            raise ValueError('Unexpected qualification root')
        verify(root)
        return read(path/'fixture.json'),read(path/'profile.json')
    memory.verify = frozen
    memory.run(destination, workers=plan['workers'][kind])


def analyze(root):
    verify(root)
    runtime = root/'static_serial/runtime-v1'
    sys.path[:0] = [str(runtime/'scripts'),str(runtime/'src')]
    from audit_layerwise_gilopt import compare
    ids = read(root/'static_serial/fixture.json')['case_ids']
    for kind in ARMS[:2]:
        if read(root/kind/'status.json')['status'] != 'completed':
            raise ValueError('Both serial arms must complete before comparison')
    serial = compare(root/ARMS[0],root/ARMS[1],ids)
    path = root/'serial-comparison.json'
    if not path.exists():
        save(path,serial)
    else:
        prior = read(path)
        if prior['equal_counts'] != serial['equal_counts'] or prior['passed'] != serial['passed']:
            raise ValueError('Previously analyzed serial answers changed')
    if read(root/'incremental_tight/status.json')['status'] != 'completed':
        return serial
    records = root/'incremental_tight/run/server/capture/request_swap_verified.jsonl'
    verified = [json.loads(line) for line in records.read_text().splitlines() if line] if records.exists() else []
    identities = {tuple(row['identity']) for row in verified}
    if not verified or len(identities) != len(verified) or any(row['bytes'] <= 0 or not row['schemas'] for row in verified):
        raise ValueError('Tight arm did not prove distinct actual byte-verified restores')
    worker = read(records.parent/'archive_resources.json')['request_checkpoints']
    if (not worker['verification_enabled'] or worker['verification_failures'] or
            worker['verification_successes'] != len(verified) or
            worker['restores'] != len(verified) or worker['requests']):
        raise ValueError('Restore verification counts or final checkpoint ownership disagree')
    for kind in ARMS:
        cleanup = read(root/kind/'run/cleanup.json')
        if cleanup['gpu_processes'] or cleanup['remaining_owned_groups']:
            raise ValueError('Qualification did not release its GPU processes')
    concurrent = compare(root/ARMS[0],root/ARMS[2],ids)
    report = dict(serial=serial,concurrent=concurrent,verified_restores=len(verified),
                  verified_bytes=sum(row['bytes'] for row in verified),
                  passed=serial['passed'] and concurrent['passed'],
                  scope='Concurrent output differences remain explicit; byte restoration alone is not output equivalence.')
    save(root/'qualification.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','arm','analyze','verify'))
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--kind',choices=ARMS)
    parser.add_argument('--source',type=Path,default=REPO)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--roomy-kv-bytes',type=int,default=48*1024**3)
    parser.add_argument('--tight-kv-bytes',type=int)
    parser.add_argument('--checkpoint-bytes',type=int,default=16*1024**3)
    parser.add_argument('--cpu-cache-bytes',type=int,default=64_000_000_000)
    parser.add_argument('--port',type=int,default=19446)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        if args.tight_kv_bytes is None:
            parser.error('prepare requires --tight-kv-bytes from actual native page geometry')
        prepare(root,source=args.source,reference=args.reference,inputs=args.inputs,
            roomy_bytes=args.roomy_kv_bytes,tight_bytes=args.tight_kv_bytes,
            checkpoint_bytes=args.checkpoint_bytes,cpu_cache_bytes=args.cpu_cache_bytes,port=args.port)
    elif args.action == 'arm':
        if args.kind is None:
            parser.error('arm requires --kind')
        arm(root,args.kind)
    elif args.action == 'analyze':
        print(json.dumps(analyze(root),ensure_ascii=False))
    else:
        verify(root)


if __name__ == '__main__':
    main()

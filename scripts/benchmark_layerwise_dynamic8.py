"""Frozen ABBA of eight complete cold prefixes: static vs incremental pages.

Only page admission and its checkpoint pool differ. Both modes retain the same
physical GPU backing, logical K/G budgets, layerwise recall, batch ceiling,
CPU history capacity and independent copies of one autotuning seed. Query is
held until the eight historical prefills complete. This is a phased prefill
measurement, not continuous-service throughput or a formal MAB campaign.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
ORDER = ('static1', 'dynamic1', 'dynamic2', 'static2')
GPU_BYTES = 48 * 1024**3
CHECKPOINT_BYTES = 16 * 1024**3
CPU_CACHE_BYTES = 114_000_000_000
DEFAULT_IDS = (
    '007_Accurate_Retrieval_r017_longmemeval_s_q0002',
    '008_Accurate_Retrieval_r018_longmemeval_s_q0002',
    '012_Conflict_Resolution_r003_factconsolidation_mh_262k_q0002',
    '013_Conflict_Resolution_r007_factconsolidation_sh_262k_q0002',
    '017_Long_Range_Understanding_r012_infbench_sum_eng_shots2_q0000',
    '020_Long_Range_Understanding_r018_infbench_sum_eng_shots2_q0000',
    '025_Long_Range_Understanding_r078_infbench_sum_eng_shots2_q0000',
    '028_Long_Range_Understanding_r093_infbench_sum_eng_shots2_q0000',
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
        temporary.replace(path)
    else:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')


def tree(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def canonical_cache(root):
    """Ignore serialization order, while retaining every cached tactic value."""
    return {str(p.relative_to(root)): hashlib.sha256(json.dumps(read(p), sort_keys=True,
            separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
            for p in sorted(root.rglob('*.json'))}


def freeze(source, destination):
    for name in ('src/vllm_kvmem', 'scripts', 'csrc', 'third_party'):
        shutil.copytree(source/name, destination/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    for name in ('pyproject.toml', 'setup.py', 'CMakeLists.txt', 'MANIFEST.in', 'LICENSE'):
        if (source/name).is_file():
            shutil.copy2(source/name, destination/name)
    path = destination/'src/vllm_kvmem/layerwise_query.py'
    query = path.read_text()
    if ('KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS' in query or
            'KVMEM_DIAGNOSTIC_PREFILL_COHORT_LENGTHS' in query):
        raise ValueError('Source already contains a measurement/cohort gate')
    with path.open('a') as output:
        output.write("\n# Identical finite historical-prefill measurement in both allocation modes.\n"
                     "os.environ['KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS'] = '8'\n"
                     "os.environ['KVMEM_DIAGNOSTIC_PREFILL_GPU_COMPLETION'] = '1'\n"
                     "from .prefill_phase_gate import install as _install_dynamic8_measurement\n"
                     "_install_dynamic8_measurement()\n")


def validate_profile(profile, source, destination):
    sys.path.insert(0, str(source/'src'))
    from vllm_kvmem.serving.serve import build_launch
    plan = build_launch(profile, source, destination/'run/server', check_paths=False)
    if plan['profile']['max_num_batched_tokens'] != 16424:
        raise ValueError('Effective native batch ceiling changed')


def prepare(root, *, source=REPO, fixture_path=None, profile_path=None,
            case_ids=DEFAULT_IDS, autotune_source=None, port=19448):
    source, root = Path(source).resolve(), Path(root).resolve()
    if root.exists():
        raise ValueError('Preserve previous attempts; choose a fresh benchmark directory')
    fixture_path = Path(fixture_path or source/'artifacts/mab-pressure-multiowner-20261105-r10/formal/fixture.json').resolve()
    profile_path = Path(profile_path or source/'artifacts/mab-pressure-multiowner-20261105-r10/formal/profile.json').resolve()
    autotune_source = Path(autotune_source or Path.home()/'.cache/vllm/flashinfer_autotune_cache/0.6.18.post1/120f').resolve()
    fixture, profile = read(fixture_path), read(profile_path)
    ids = tuple(case_ids)
    if len(ids) != 8 or len(set(ids)) != 8 or len({i.rsplit('_q',1)[0] for i in ids}) != 8:
        raise ValueError('Need eight distinct original cold-prefix groups')
    rows_by_id = {row['id']:row for row in fixture['rows']}
    bindings_by_id = {row['id']:row for row in fixture['bindings']}
    rows = [deepcopy(rows_by_id[cid]) for cid in ids]
    bindings = [deepcopy(bindings_by_id[cid]) for cid in ids]
    kv = profile['kvmem']
    if [kv[k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')] != [36864,16384,32,41,0]:
        raise ValueError('Original Active/Gen or recall geometry changed')
    if [profile[k] for k in ('execution_mode','draft_tokens','max_num_seqs','max_num_batched_tokens')] != ['compiled_graph',4,8,16424]:
        raise ValueError('Original compiled MTP4/eight-slot execution changed')
    reservations = [((row['input_tokens']+2*kv['gen_budget']+31)//32)*32*17*4*256*2 for row in rows]
    if sum(reservations) >= CPU_CACHE_BYTES-16*1024**3:
        raise ValueError('All eight owners must fit CPU credits before a gated owner can finish')
    # Exact geometry from the previously verified native MTP4/FP8 configuration:
    # 48 physical pages per full K/G owner; one page remains the native null page.
    if GPU_BYTES // 56_262_656 <= 8*48:
        raise ValueError('Gated Query owners must all fit the physical pool; no forced swapping')
    if not autotune_source.is_dir() or not canonical_cache(autotune_source):
        raise ValueError('Require an existing nonempty autotuning seed')
    for row,binding in zip(rows,bindings):
        if sha(row['source']) != row['source_cache_sha256'] or sha(binding['request']) != binding['sha256']:
            raise ValueError('Original input or sampling hash changed')
    root.mkdir(parents=True)
    for row,binding in zip(rows,bindings):
        for record, field, folder in ((row,'source','inputs'),(binding,'request','bindings')):
            origin = Path(record[field]); destination = root/folder/(row['id']+('.json.gz' if field=='source' else '.json'))
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(origin,destination); record[field] = str(destination)
    fixture.update(rows=rows,bindings=bindings,case_ids=list(ids),cases=8,
                   qualification=False,restore_digest_qualification=False,
                   performance_observation=True,diagnostic_only=True,external_grading=False)
    fixture['execution_flags'].update(layerwise_pressure_prepared_reads=True,
        layerwise_pressure_post_forward_reap=True,layerwise_pressure_submit_on_prepare=True,
        layerwise_pressure_reuse_layer_views=True)
    fixture.setdefault('execution_kvmem_overrides', {})['cpu_read_cache_bytes'] = CPU_CACHE_BYTES
    freeze(source,root/'source-runtime')
    proof = tree(root/'source-runtime')
    seed = root/'autotune-seed'
    shutil.copytree(autotune_source,seed)
    save(root/'autotune-seed-proof.json',tree(seed))
    save(root/'autotune-seed-canonical.json',canonical_cache(seed))
    files = {str(p.relative_to(root)):sha(p) for folder in ('inputs','bindings') for p in (root/folder).iterdir()}
    for name in ('autotune-seed-proof.json','autotune-seed-canonical.json'):
        files[name] = sha(root/name)
    for kind in ORDER:
        arm = root/kind
        shutil.copytree(root/'source-runtime',arm/'runtime-v1')
        shutil.copytree(seed,arm/'autotune')
        dynamic = kind.startswith('dynamic')
        current = deepcopy(profile)
        current.update(port=port,verify_request_swap=False,kv_cache_memory_bytes=GPU_BYTES,
                       request_planning_batch_tokens=16424)
        current['kvmem'].update(cpu_read_cache_bytes=CPU_CACHE_BYTES,
            gpu_memory_admission='incremental' if dynamic else 'static',
            request_swap_layerwise=dynamic,request_checkpoint_bytes=CHECKPOINT_BYTES if dynamic else 0,
            request_checkpoint_ssd_bytes=0)
        validate_profile(current,source,arm)
        for name,value in (('fixture.json',fixture),('profile.json',current),('runtime-proof.json',proof)):
            save(arm/name,value);files[str((arm/name).relative_to(root))] = sha(arm/name)
        save(arm/'status.json',dict(status='prepared'))
    shutil.copy2(__file__,root/'launch.py');files['launch.py'] = sha(root/'launch.py')
    save(root/'plan.json',dict(order=list(ORDER),files=files,workers=8,case_ids=list(ids),
        input_tokens=sum(row['input_tokens'] for row in rows),cpu_reservation_bytes=sum(reservations),
        client_credit_budget_bytes=CPU_CACHE_BYTES-16*1024**3,cpu_cache_bytes=CPU_CACHE_BYTES,
        physical_kv_bytes=GPU_BYTES,dynamic_checkpoint_bytes=CHECKPOINT_BYTES,
        source_controls={str(fixture_path):sha(fixture_path),str(profile_path):sha(profile_path)},
        source_runtime_sha256=proof,autotune_original=str(autotune_source),
        automatic_retry=False,formal_resume=False,external_grading=False,scheduled_monitor=False,
        scope=__doc__,changed_between_modes=['gpu_memory_admission','request_swap_layerwise','request_checkpoint_bytes']))
    save(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save(root/'status.json',dict(status='prepared'))


def verify(root):
    if sha(root/'plan.json') != read(root/'prepared-proof.json')['plan_sha256']:
        raise ValueError('Frozen plan changed')
    plan = read(root/'plan.json')
    if tree(root/'source-runtime') != plan['source_runtime_sha256']:
        raise ValueError('Frozen shared source runtime changed')
    for name,digest in plan['files'].items():
        if sha(root/name) != digest:
            raise ValueError('Frozen control/input changed: '+name)
    if tree(root/'autotune-seed') != read(root/'autotune-seed-proof.json'):
        raise ValueError('Immutable autotuning seed changed')
    for kind in plan['order']:
        if tree(root/kind/'runtime-v1') != read(root/kind/'runtime-proof.json'):
            raise ValueError('Frozen runtime changed: '+kind)
    return plan


def arm(root, kind):
    plan = verify(root)
    if kind not in plan['order']:
        raise ValueError('Unknown benchmark arm')
    destination = root/kind
    seed = read(root/'autotune-seed-canonical.json')
    if canonical_cache(destination/'autotune') != seed:
        raise ValueError('Arm must start from the unchanged common autotuning seed')
    runtime = destination/'runtime-v1'
    sys.path[:0] = [str(runtime/'src'),str(runtime/'scripts')]
    spec = importlib.util.spec_from_file_location('dynamic8_memory',runtime/'scripts/run_layerwise_memory.py')
    memory = importlib.util.module_from_spec(spec);spec.loader.exec_module(memory)
    def frozen(path):
        if path != destination:
            raise ValueError('Unexpected memory benchmark root')
        verify(root)
        return read(path/'fixture.json'),read(path/'profile.json')
    memory.verify = frozen
    import benchmark_query_fps_prebuild_live as environment_module
    original_environment = environment_module.server_environment
    def environment(path):
        env = original_environment(path)
        env['VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR'] = str(destination/'autotune')
        return env
    environment_module.server_environment = environment
    try:
        memory.run(destination,workers=8)
    finally:
        environment_module.server_environment = original_environment
        final = canonical_cache(destination/'autotune')
        save(destination/'autotune-audit.json',dict(initial=seed,final=final,
            unchanged=seed==final,added=sorted(final.keys()-seed.keys()),
            removed=sorted(seed.keys()-final.keys()),
            changed=sorted(name for name in seed.keys()&final.keys() if seed[name]!=final[name])))


def run(root):
    plan = verify(root)
    save(root/'queue-process.json',dict(pid=os.getpid(),at=time.time()))
    from run_recall_comparison import stop_owned,termination_guard
    child = None
    with termination_guard():
        try:
            for kind in plan['order']:
                save(root/'status.json',dict(status='running',active_arm=kind,at=time.time()),replace=True)
                with (root/(kind+'.log')).open('x') as log:
                    command = [sys.executable,'-B',str(root/'launch.py'),'arm','--root',str(root),'--kind',kind]
                    child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(root/'child.json',dict(pid=child.pid,kind=kind,command=command,at=time.time()),replace=True)
                child.wait()
                if child.returncode:
                    raise RuntimeError(kind+' failed; preserve first responses, no automatic retry')
                cleanup = read(root/kind/'run/cleanup.json')
                if cleanup['gpu_processes'] or cleanup['remaining_owned_groups']:
                    raise RuntimeError(kind+' did not release its GPU processes')
            save(root/'status.json',dict(status='completed',at=time.time()),replace=True)
        except BaseException as error:
            save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()),replace=True)
            raise
        finally:
            stop_owned(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','arm','verify'))
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--kind',choices=ORDER)
    parser.add_argument('--source',type=Path,default=REPO)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--case-ids-json',type=Path)
    parser.add_argument('--autotune-source',type=Path)
    parser.add_argument('--port',type=int,default=19448)
    args = parser.parse_args();root=args.root.resolve()
    if args.action == 'prepare':
        prepare(root,source=args.source,fixture_path=args.fixture,profile_path=args.profile,
            case_ids=read(args.case_ids_json) if args.case_ids_json else DEFAULT_IDS,
            autotune_source=args.autotune_source,port=args.port)
    elif args.action == 'arm':
        if args.kind is None:parser.error('arm requires --kind')
        arm(root,args.kind)
    elif args.action == 'run':
        # The copied launch.py lives outside its runtime script directory.
        sys.path.insert(0,str(root/'source-runtime/scripts'))
        run(root)
    else:
        verify(root)


if __name__ == '__main__':
    main()

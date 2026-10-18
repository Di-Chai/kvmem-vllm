"""Run one weight-format variant of an accepted G16 KVMem measurement.

Keep the frozen workload, scoring, budgets and measured runtime. The only
production deltas permitted are generic loader entry points and the dispatch
guard keeping inherited MTP FP8 out of the DFlash-only context patch. Require
an independent cold target/MTP/recurrent restore qualification for the new
weights. Reuse the capacity harness's execution and accounting checks.
"""
import argparse
import asyncio
import copy
import importlib.util
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/benchmark_weight_precision.py'
KIND = 'qualified_weight_precision_v1'
ARM = 'rolling-candidate-v2'
LOADERS = {'src/vllm_kvmem/serve.py', 'src/vllm_kvmem/elastic_serve.py',
           'src/vllm_kvmem/draft_quantization.py'}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


chain = module('weight_capacity_chain', ROOT/'scripts/benchmark_capacity_chain.py')
cap = chain.cap


def runtime_delta(before, after):
    cap.require(before.keys() == after.keys(), 'Weight test changed the runtime file set')
    delta = {k: [before[k], after[k]] for k in before if before[k] != after[k]}
    cap.require(set(delta) <= LOADERS, 'Weight test changed KV/scoring/runtime algorithms')
    return delta


def model_identity(path):
    root = Path(path).resolve()
    config = cap.read(root/'config.json')
    quant = config.get('quantization_config', {})
    cap.require(quant.get('quant_method') == 'fp8', 'This first precision arm requires FP8 weights')
    index = root/'model.safetensors.index.json'
    weights = sorted(set(cap.read(index)['weight_map'].values()))
    files = {}
    for name in weights:
        p = (root/name).resolve()
        cap.require(p.is_relative_to(root) and p.is_file() and p.stat().st_size > 0,
                    'Missing or invalid local weight shard')
        files[name] = dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
    return dict(path=str(root), config_sha256=cap.sha(root/'config.json'),
                index_sha256=cap.sha(index), shards=files, quant_method='fp8',
                scope='Config/index hashes and local shard size/mtime identity; not full shard content hashes')


def qualification_proof(directory, profile, baseline_runtime):
    directory = Path(directory).resolve()
    gate = module('weight_cold_qualification', directory/'frozen/scripts/qualify_rolling_window.py')
    m = gate.verify(directory)
    cap.require(cap.read(directory/'status.json')['status'] == 'completed_pending_review',
                'FP8 cold restoration has not completed')
    result = cap.read(directory/'results.json')
    cap.require(result.get('status') == 'passed' and result.get('cleanup_completed') is True
                and len(result.get('history_proofs', [])) == 6,
                'Incomplete FP8 history/cleanup qualification')
    capture = directory/'server/capture'
    rows = cap.read(directory/'workload/traces.json')
    cap.require(gate.oracle_evidence(capture, rows) == result['oracle'],
                'Independent cold/pressure restore proof changed')
    native = result.get('native_reconciliation', {})
    cap.require(native.get('passed') is True and native.get('preemptions') == 0
                and native.get('native_tokens') == native.get('client_tokens'),
                'FP8 qualification accounting or preemption failed')
    qp = cap.read(directory/'profile.json')
    cap.require(qp == gate.qualification_profile(profile, qp['port']),
                'Performance has settings not covered by the declared cold qualification')
    return dict(path=str(directory), manifest_sha256=cap.sha(directory/'manifest.json'),
                results_sha256=cap.sha(directory/'results.json'),
                runtime_delta=runtime_delta(baseline_runtime, m['runtime_sha256']),
                runtime_sha256=m['runtime_sha256'],
                scope='Two-user K32/G4 cold and rolling correctness; K64/G32 C8 performance '
                      'must independently pass complete-history and final-release checks')


def expected_profile(baseline, model, port):
    result = copy.deepcopy(baseline)
    result.update(model=str(model), hf_config_path=str(model), port=port)
    return result


def prepare(args):
    root, base = args.output.resolve(), args.reference.resolve()
    cap.require(not root.exists(), 'Use a fresh measurement directory')
    original_gate = module('weight_reference_gate', base/'frozen/scripts/benchmark_capacity_chain.py')
    original = original_gate.verify(base)
    cap.require(cap.read(base/'status.json')['status'] == 'completed_pending_review'
                and original['arms'] == [ARM]
                and cap.read(base/ARM/'deployment-acceptance.json')['accepted'] is True,
                'Reference must be an accepted completed KVMem measurement')
    model = args.model.resolve()
    profile = expected_profile(cap.read(base/(ARM+'.json')), model, args.port)
    proof = qualification_proof(args.qualification, profile, original['runtime_sha256'][ARM])
    identity = model_identity(model)
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(base/'frozen', root/'frozen', ignore=ignore)
    shutil.copyfile(ROOT/ENTRY, root/'frozen'/ENTRY)
    shutil.copytree(Path(proof['path'])/'runtime', root/'runtime'/ARM, ignore=ignore)
    shutil.copytree(base/'diagnostics', root/'diagnostics', ignore=ignore)
    for name in ('fixture.json', 'native.json', 'installed-native-identity.json', 'observer.json'):
        shutil.copyfile(base/name, root/name)
    cap.save(root/(ARM+'.json'), profile)
    m = copy.deepcopy(original)
    m.update(experiment_kind=KIND, created_at=time.time(), port=args.port,
             source_sha256=cap.hashes(root/'frozen'),
             runtime_sha256={ARM: proof['runtime_sha256']},
             profiles_sha256={a: cap.sha(root/(a+'.json')) for a in chain.ARMS},
             weight_reference=str(base), weight_reference_sha256=cap.sha(base/'manifest.json'),
             weight_qualification=proof, weight_model_identity=identity,
             scope='FP8 weights versus recorded NVFP4 reference: same original G16 eight-agent '
                   'fixture, sampling, tool waits, two warmups, natural output ceiling16K,300s, '
                   'K64/G32/256Ki history, CPU-only history and MTP4 graphs. Only model paths, '
                   'port, generic loader entry points and DFlash-only dispatch guard differ. '
                   'Not an on-policy solve-rate test.')
    cap.save(root/'manifest.json', m)
    cap.save(root/'status.json', dict(status='prepared', gpu_started=False, at=time.time()))


def verify(root):
    m = cap.read(root/'manifest.json')
    cap.require(ROOT == root/'frozen' and m['experiment_kind'] == KIND, 'Use the frozen weight entry')
    base = Path(m['weight_reference'])
    cap.require(cap.sha(base/'manifest.json') == m['weight_reference_sha256'], 'Reference changed')
    original = cap.read(base/'manifest.json')
    profile = cap.read(root/(ARM+'.json'))
    cap.require(profile == expected_profile(cap.read(base/(ARM+'.json')),
        m['weight_model_identity']['path'], m['port']), 'Only model paths and port may change')
    cap.require(model_identity(profile['model']) == m['weight_model_identity'], 'Weight files changed')
    proof = qualification_proof(m['weight_qualification']['path'], profile,
                                original['runtime_sha256'][ARM])
    cap.require(proof == m['weight_qualification'], 'Weight qualification changed')
    allowed = {'experiment_kind','created_at','port','source_sha256','runtime_sha256',
               'profiles_sha256','weight_reference','weight_reference_sha256',
               'weight_qualification','weight_model_identity','scope'}
    cap.require({k:v for k,v in m.items() if k not in allowed} ==
                {k:v for k,v in original.items() if k not in allowed}, 'Measurement protocol changed')
    cap.require(cap.hashes(root/'frozen') == m['source_sha256'], 'Frozen harness changed')
    cap.require(cap.hashes(root/'runtime'/ARM) == proof['runtime_sha256'] == m['runtime_sha256'][ARM],
                'Qualified FP8 runtime changed')
    for arm,digest in m['profiles_sha256'].items():
        cap.require(cap.sha(root/(arm+'.json')) == digest, 'Profile changed')
    cap.require(cap.sha(root/'fixture.json') == m['fixture_sha256'], 'Workload changed')
    cap.require(cap.native_source_identity() == m['native_source'], 'Native installation changed')
    chain.installed_native(root/'installed-native-identity.json')
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    for name in ('reference','qualification','model'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--port', type=int, default=18588)
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args)
    else:
        # This separate protocol verifies its declared weight delta. All GPU
        # ownership, history, accounting, release and headroom gates stay in run.
        chain.verify = verify
        asyncio.run(chain.run(args))


if __name__ == '__main__':
    main()

"""Run the frozen first16 cohort only after explicit-release GPU acceptance."""
import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

import benchmark_deepswe_capacity as supervisor


def verify_release(root, cap):
    manifest = cap.read(root/'manifest.json')
    proof = manifest['release_qualification']
    qualification = Path(proof['path'])
    supervisor.require(cap.sha(qualification/'manifest.json') == proof['manifest_sha256'],
                       'Release qualification identity changed')
    q = cap.read(qualification/'manifest.json')
    supervisor.require(cap.hashes(qualification/'frozen') == q['source_sha256'] and
                       cap.hashes(qualification/'runtime') == q['runtime_sha256'],
                       'Qualified release code changed')
    supervisor.require(cap.read(qualification/'status.json')['status'] == 'passed',
                       'Real-model session release qualification has not passed')
    result = cap.read(qualification/'results.json')
    supervisor.require(result.get('status') == 'passed' and result.get('namespace_count') == 40 and
        result.get('checkpoint_deferrals') == 0 and result.get('cleanup_completed') is True and
        len(result.get('history_proofs',[])) == 80 and len(result.get('restore_proofs',[])) == 40 and
        len(result.get('releases',[])) == len(result.get('actual_batch_evidence',[])) == 5,
        'Release evidence does not cover forty sessions and their continuations')
    reconciliation = result.get('native_reconciliation',{})
    supervisor.require(reconciliation.get('passed') is True and reconciliation.get('preemptions') == 0 and
        reconciliation.get('client_tokens') == reconciliation.get('native_tokens') == 41000,
        'Release output or preemption accounting failed')
    production = {k:v for k,v in q['runtime_sha256'].items() if k.startswith(('src/','csrc/'))}
    supervisor.require(production and all(manifest['source_sha256'].get(k) == v for k,v in production.items()),
                       'Formal runtime differs from the GPU-qualified production code')
    profile = cap.read(root/'kvmem-c8/profile.json')
    qualified_profile = cap.read(qualification/'profile.json')
    qualified_profile.update(port=profile['port'], verify_session_restore=False)
    supervisor.require(profile == qualified_profile, 'Formal deployment differs from release qualification')
    evidence = dict(passed=True,qualification=str(qualification),
        manifest_sha256=proof['manifest_sha256'],results_sha256=cap.sha(qualification/'results.json'),
        scope='Same GPU-qualified production code and model/budgets; byte oracle disabled for formal measurement')
    cap.save(root/'release-qualification-proof.json',evidence)
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true')
    args = p.parse_args()
    root = args.output.resolve()
    cap = supervisor.module('full_deepswe_capacity','benchmark_capacity_scaling.py')
    m = supervisor.verify(root,cap)
    supervisor.require(m.get('duration_seconds') == 0 and m.get('stages') == ['kvmem-c8'] and
        m.get('context_ceiling') == 196608,'Not the frozen full-task 192Ki eight-agent cohort')
    evidence = verify_release(root,cap)
    if args.verify_only:
        print(supervisor.json.dumps(evidence))
    else:
        asyncio.run(supervisor.run(SimpleNamespace(output=root)))


if __name__ == '__main__':
    main()

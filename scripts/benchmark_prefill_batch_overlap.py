"""Fixed full-input 2x2 comparison of physical batch and early statistics.

Only execution geometry and an existing same-forward statistics staging path
vary. Each original 1K acceptance/pressure boundary remains authoritative.
"""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
sys.path.insert(0,str(REPO/'scripts'))
from audit_layerwise_gilopt import read, save_new, sha, tree, compare, canonical_signature

BASE = REPO/'artifacts/pressure-metadata-reads-20261105-r2/b1/candidate'
ORDER = ('a1','b1','c1','d1','d2','c2','b2','a2')
FILES = ('src/vllm_kvmem/logical_prefill.py','src/vllm_kvmem/options.py')


def prepare(root, *, capacity_test=False):
    assert not (root/'plan.json').exists()
    assert read(BASE/'status.json')['status'] == 'completed'
    base_proof = read(BASE/'runtime-proof.json')
    assert tree(BASE/'runtime-v1') == base_proof
    patch = []
    for name in FILES:
        # Frozen runtimes retain older documentation and unrelated RoPE work.
        # Port only the two executable limits, never overwrite whole modules.
        old = (BASE/'runtime-v1'/name).read_text()
        changes = ([('MAX_LOGICAL_PREFILL_CHUNKS = 8','MAX_LOGICAL_PREFILL_CHUNKS = 16')]
            if name.endswith('logical_prefill.py') else [
                ('not 2 <= self.prefill_physical_max_chunks <= 8','not 2 <= self.prefill_physical_max_chunks <= 16'),
                ('prefill_physical_max_chunks must be an integer in [2, 8]',
                 'prefill_physical_max_chunks must be an integer in [2, 16]')])
        new = old
        for a,b in changes:
            assert new.count(a)==1 and b in (REPO/name).read_text()
            new=new.replace(a,b)
        if name.endswith('logical_prefill.py'):
            new=new.replace('two to eight','two to sixteen').replace('Two to eight','Two to sixteen')
        patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
            fromfile='a/'+name,tofile='b/'+name,n=3))
    (root/'incremental.patch').write_text(''.join(patch))
    shutil.copy2(__file__,root/'launch.py')
    tests = root/'gpu-tests';tests.mkdir()
    for name in ('test_physical_prefill_stats_gpu.py','test_prefill_action_before_mtp_gpu.py',
                 'test_native_attention_stats_gpu.py'):
        shutil.copy2(REPO/'tests'/name,tests/name)
    common = {str(root/n):sha(root/n) for n in ('launch.py','incremental.patch')}
    common.update({str(p):sha(p) for p in tests.iterdir()})
    common.update({str(REPO/'scripts'/n):sha(REPO/'scripts'/n) for n in
        ('benchmark_mab_direct_restore.py','analyze_layerwise_physical_prefill.py',
         'analyze_mab_cold_prefill.py','audit_layerwise_gilopt.py')})
    order = ('a1','b1','b2','a2') if capacity_test else ORDER
    maximum = 16 if capacity_test else 14
    rows = maximum*32
    frozen = None
    for arm in order:
        ar = root/arm;c = ar/'candidate';c.mkdir(parents=True)
        runtime = c/'runtime-v1'
        shutil.copytree(BASE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        env = dict(os.environ,GIT_CEILING_DIRECTORIES=str(c))
        subprocess.run(['git','apply','--unidiff-zero','--check',str(root/'incremental.patch')],cwd=runtime,env=env,check=True)
        subprocess.run(['git','apply','--unidiff-zero',str(root/'incremental.patch')],cwd=runtime,env=env,check=True)
        current = tree(runtime)
        if frozen is None:frozen=current
        assert frozen == current
        profile = read(BASE/'profile.json')
        cfg = profile['kvmem']
        assert [cfg[k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')] == [36864,16384,32,41,0]
        assert profile['max_num_batched_tokens'] == 16424 and profile['prefill_chunk_tokens'] == 1024
        if capacity_test:
            # 512 statistics rows consume ~0.62GiB more than 448. Restore the
            # global engine's 1M-context admission headroom in both arms; the
            # per-request selected/generation budgets remain exactly frozen.
            assert profile['gpu_memory_utilization'] == .72
            profile['gpu_memory_utilization'] = .73
        cfg.update(action_native_max_rows=rows,action_native_max_bytes=6<<30,
            action_download_max_bytes=256<<20,
            prefill_physical_max_chunks=maximum if arm[0] in 'bd' else 8,
            prefill_action_before_mtp=capacity_test or arm[0] in 'cd')
        fixture = read(BASE/'fixture.json')
        fixture['execution_kvmem_overrides'].update(action_native_max_rows=rows,action_native_max_bytes=6<<30)
        save_new(c/'fixture.json',fixture)
        save_new(c/'profile.json',profile)
        save_new(c/'verification-profile.json',dict(profile,verify_session_restore=True,verify_session_restore_digest=True))
        save_new(c/'runtime-proof.json',current)
        save_new(c/'status.json',dict(status='prepared'))
        reference = BASE if arm=='a1' else root/('a1' if arm in ('b1','c1','a2') else 'b1' if arm in ('d1','b2') else arm[0]+'1')/'candidate'
        files = dict(common)
        files.update({str(c/n):sha(c/n) for n in ('fixture.json','profile.json','verification-profile.json','runtime-proof.json')})
        files.update({b['request']:b['sha256'] for b in fixture['bindings']})
        save_new(ar/'plan.json',dict(baseline=str(reference),files=files,
            physical_chunks=cfg['prefill_physical_max_chunks'],early_statistics=cfg['prefill_action_before_mtp'],
            formal_resume=False,external_grading=False,numeric_shape_differences_authorized=True))
        save_new(ar/'prepared-proof.json',dict(plan_sha256=sha(ar/'plan.json')))
        save_new(ar/'status.json',dict(status='prepared'))
    formal=REPO/'artifacts/mab-pressure-optimized-20261105-r3/formal/run/answers/cases'
    assert formal.is_dir()
    save_new(root/'formal-preservation.json',{str(p):sha(p) for p in sorted(formal.rglob('*')) if p.is_file()})
    save_new(root/'plan.json',dict(order=order,baseline=str(BASE),files=common,
        runtime_proof=sha(root/'a1/candidate/runtime-proof.json'),
        changed_files=sorted(k for k in frozen if frozen[k]!=base_proof.get(k)),
        cpu_download_ceiling_bytes=256<<20,stats_max_rows=rows,capacity_test=capacity_test,
        scope='Same original complete cold MAB input; one live owner; unchanged 1K transactions, budgets and sampling. Fixed reverse repeat, no profiler. Shape differences reported, same-shape early submission requires exact block/output equality.'))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


def verify(root):
    assert sha(root/'plan.json') == read(root/'prepared-proof.json')['plan_sha256']
    plan = read(root/'plan.json')
    for p,digest in plan['files'].items():assert sha(Path(p)) == digest,p
    return plan


def semantic_audit(baseline,candidate):
    from analyze_mab_cold_prefill import measure
    from analyze_layerwise_physical_prefill import logical_steps
    a,b = [measure(p) for p in (baseline,candidate)]
    assert (a['sample_id'],a['executed_prefill_tokens']) == (b['sample_id'],b['executed_prefill_tokens'])
    ea,eb = [logical_steps(p,m) for p,m in ((baseline,a),(candidate,b))]
    assert ea[0] == eb[0], 'Original logical scheduling changed'
    signatures=[]
    for p in (baseline,candidate):
        case=p/'run/answers/cases'/a['sample_id']
        signatures.append(canonical_signature(read(case/'result.json'),read(case/'events.json')))
        native=read(p/'run/server/capture/native_attention_stats.json')
        assert native['disabled_reason'] is None and native['validation_failures']==0
        assert not any('fallback' in k and v for k,v in native['layer_phase_counts'].items())
    assert len(signatures[0]['pressure']) == len(signatures[1]['pressure']) == 23
    overlaps=[]
    for left,right in zip(*(s['pressure'] for s in signatures)):
        for key in ('accepted_history','method','fallback','event_type','query_source','selected_tokens'):
            assert left[key]==right[key],key
        assert len(left['layers']) == len(right['layers']) == 16
        for x,y in zip(left['layers'],right['layers']):
            for key in ('layer','charged_tokens','accepted_score_end','query_samples','sampled_chunks'):
                assert x[key]==y[key],key
            sx,sy=set(x['selected_starts']),set(y['selected_starts'])
            overlaps.append(len(sx&sy)/len(sx))
    # Read only public aggregate counters, never infer staging from a flag.
    capture=candidate/'run/server/capture'
    counters=[]
    for p in capture.glob('*.json'):
        obj=read(p)
        if isinstance(obj,dict) and 'prefill_action_before_mtp_stats' in obj:
            counters.append(dict(file=p.name,stats=obj['prefill_action_before_mtp_stats']))
    if read(candidate/'profile.json')['kvmem']['prefill_action_before_mtp']:
        assert any(c['stats'].get('staged_batches',0)>0 for c in counters),'Early path did not run'
    return dict(measurement=b,physical=eb[1],logical_steps_equal=True,
        original_sampling_and_pressure_boundaries_equal=True,
        mean_pressure_block_overlap=sum(overlaps)/len(overlaps),early_counters=counters)


def run(root):
    plan=verify(root)
    gate=read(root/'gpu-gate.json')
    assert gate['passed'] and gate['runtime_proof']==plan['runtime_proof']
    try:
        for arm in plan['order']:
            (root/'status.json').write_text(json.dumps(dict(status='running',arm=arm)))
            subprocess.run([sys.executable,'-B',str(root/'launch.py'),'run-arm','--root',str(root/arm)],check=True)
            ar=root/arm;reference=Path(read(ar/'plan.json')['baseline'])
            save_new(ar/'semantic-audit.json',semantic_audit(reference,ar/'candidate'))
            if arm not in ('a1','b1'):
                assert read(ar/'equivalence.json')['passed'],'Same-shape first mismatch: preserve and stop'
        for p,digest in read(root/'formal-preservation.json').items():assert sha(Path(p))==digest
        (root/'status.json').write_text(json.dumps(dict(status='completed')))
    except BaseException as error:
        (root/'status.json').write_text(json.dumps(dict(status='failed',arm=arm,error=repr(error))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','run-arm'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--capacity-test',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':prepare(root,capacity_test=args.capacity_test)
    elif args.action=='run':run(root)
    else:
        from benchmark_mab_direct_restore import run as run_arm
        run_arm(root)

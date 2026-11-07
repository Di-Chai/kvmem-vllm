"""Qualify the existing publication pipeline on original layerwise MAB input."""
import argparse
from pathlib import Path
import shutil

from audit_layerwise_gilopt import read, save_new, sha, tree
from benchmark_mab_direct_restore import run

REPO=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())


def prepare(root,baseline,enabled=True):
    if root.exists():raise ValueError('Preserve prior attempt')
    proof=read(baseline/'runtime-proof.json')
    assert tree(baseline/'runtime-v1')==proof
    assert read(baseline/'status.json')['status']=='completed'
    root.mkdir(parents=True);candidate=root/'candidate';candidate.mkdir()
    shutil.copytree(baseline/'runtime-v1',candidate/'runtime-v1',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    assert tree(candidate/'runtime-v1')==proof
    fixture=read(baseline/'fixture.json');row=fixture['rows'][0]
    fixture.update(rows=[row],case_ids=[row['id']],cases=1,
        bindings=[b for b in fixture['bindings'] if b['id']==row['id']],diagnostic_only=True)
    profile=read(baseline/'profile.json')
    assert not profile['kvmem'].get('prefill_publication_pipeline',False)
    assert [profile['kvmem'][k] for k in ['select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks']]==[36864,16384,32,41,0]
    if enabled:
        profile['kvmem']['prefill_publication_pipeline']=True
    save_new(candidate/'fixture.json',fixture)
    save_new(candidate/'profile.json',profile)
    save_new(candidate/'verification-profile.json',dict(profile,verify_session_restore=True,verify_session_restore_digest=True))
    save_new(candidate/'runtime-proof.json',proof)
    save_new(candidate/'status.json',dict(status='prepared'))
    shutil.copy2(__file__,root/'launch.py')
    files={str(candidate/n):sha(candidate/n) for n in ['fixture.json','profile.json','verification-profile.json','runtime-proof.json']}
    files.update({b['request']:b['sha256'] for b in fixture['bindings']})
    files.update({str(p):sha(p) for p in (baseline/'run/answers/cases'/row['id']).iterdir() if p.is_file()})
    for p in [root/'launch.py',baseline/'runtime-proof.json',REPO/'scripts/benchmark_mab_direct_restore.py',
              REPO/'scripts/benchmark_mab_query_handoff.py']:
        files[str(p)]=sha(p)
    save_new(root/'plan.json',dict(baseline=str(baseline),files=files,changed_runtime_files=[],
        changed_profile_key='kvmem.prefill_publication_pipeline' if enabled else None,
        pipeline_enabled=enabled,formal_resume=False,external_grading=False,
        scope='Existing bounded publication pipeline only, on one complete original cold MAB input. All KV, scoring, sampling and output controls unchanged. Final Query/decode excluded from the prefill measurement.'))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--disabled',action='store_true',help='Repeat the original cold control unchanged')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root.resolve(),args.baseline.resolve(),not args.disabled)
    else:run(args.root.resolve())

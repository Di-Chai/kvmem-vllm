"""Freeze corrected pressure guidance; retain the two completed control arms."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from benchmark_prefill_publication_pipeline import read, save, sha, require, runtime_origins
from ruler_guided_pressure_protocol import PROFILE, expected_profile, validate_execution


def prepare(root, previous, repo):
    root, previous, repo = (p.resolve() for p in (root, previous, repo))
    require(not root.exists(), 'Use a fresh corrected campaign root')
    reference=previous/'max'
    fixture=read(reference/'fixture.json')
    require(sha(reference/'fixture.json')==read(reference/'dataset-summary.json')['fixture_sha256'],
            'Original cohort changed')
    for row in fixture['rows']:
        source=Path(fixture['input_directory'])/(row['id']+'.i32')
        require(sha(source)==row['served_sha256'], 'Input changed: '+row['id'])
    original=fixture['reference_profile']
    profile=expected_profile(original)
    names=set(read(reference/'runtime-proof.json')['files'])
    for folder in ('src/vllm_kvmem','scripts','csrc','third_party/qw3'):
        names.update(str(p.relative_to(repo)) for p in (repo/folder).rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.cpp','.hpp','.h','.cu','.cuh','.so'))
    origins,inherited=runtime_origins(repo,reference,names)
    hashes={name:sha(p) for name,p in origins.items()}
    root.mkdir(parents=True)
    shared=root/'runtime'
    for name,source in origins.items():
        out=shared/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,out)
        require(sha(out)==hashes[name], 'Source changed during freeze')
    proof=dict(files=hashes,inherited_utilities=inherited,source=str(repo),
        supersedes=str(previous/'max_guided'),scope='Private guided scores only at prefill pressure; final original Query plus max graph')
    save(root/'runtime-proof.json',proof)
    campaigns=[]
    for arm in ('max','query_only'):
        state=read(previous/arm/'status.json')
        require(state['status']=='completed' and read(previous/arm/'run/execution-audit.json')['passed'],
                'Control is not complete')
        (root/arm).symlink_to(previous/arm,target_is_directory=True)
        campaigns.append(dict(arm=arm,path=str(root/arm),manifest_sha256=sha(previous/arm/'manifest.json')))
    for name in ('max_guided','qualification'):
        campaign=root/name;campaign.mkdir();(campaign/'runtime').symlink_to(shared,target_is_directory=True)
        variant=deepcopy(fixture)
        variant.update(execution_profile=PROFILE,source_fixture_sha256=sha(reference/'fixture.json'))
        variant.pop('pooling_arm',None)
        validate_execution(profile,variant)
        from vllm_kvmem.serve import build_launch
        build_launch(profile,repo,campaign/'preflight-server')
        save(campaign/'profile.json',profile);variant['profile_sha256']=sha(campaign/'profile.json')
        save(campaign/'fixture.json',variant)
        summary=read(reference/'dataset-summary.json');summary.update(
            fixture_sha256=sha(campaign/'fixture.json'),execution_profile=PROFILE)
        save(campaign/'dataset-summary.json',summary);save(campaign/'runtime-proof.json',proof)
        save(campaign/'manifest.json',dict(execution_profile=PROFILE,cases=260,
            fixture_sha256=sha(campaign/'fixture.json'),profile_sha256=sha(campaign/'profile.json'),
            runtime_proof_sha256=sha(campaign/'runtime-proof.json'),reused_answers=False,
            reference=str(reference),superseded_protocol=str(previous/'max_guided'),
            guidance_scope='Prefill pressure: generated Query; final Query: original public Query plus max graph',
            pressure_propagation=False,private_context='Only the already ingested prefix; no next public input or future Query',
            fallback='Explicit cumulative-attention fallback if private Query cannot be published'))
        if name=='max_guided':campaigns.append(dict(arm=name,path=str(campaign),manifest_sha256=sha(campaign/'manifest.json')))
    require(all(sha(p)==hashes[name] for name,p in origins.items()),'Source changed after freezing')
    save(root/'queue.json',dict(version=2,order=['max','query_only','max_guided'],campaigns=campaigns,
        prerequisite=str(repo/'artifacts/ruler-v1-256k-native-20261022-r1'),
        reuse_completed_controls=['max','query_only'],guided_qualification=str(root/'qualification'),
        runtime_proof_sha256=sha(root/'runtime-proof.json')))
    print(json.dumps(dict(status='prepared',root=str(root),fresh_cases=260,reused_controls=2)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','previous','repo'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();prepare(args.root,args.previous,args.repo)

"""Freeze three ordered Pro6000 RULER arms from the completed local cohort."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

from benchmark_prefill_publication_pipeline import read, save, sha, require, runtime_origins
from benchmark_prefill_variant import differences
from ruler_pooling_protocol import ARMS, PROFILE, expected_profile, validate_execution
from ruler_protocol import validate_cohort, load_scorer


def prepare(root, reference, repo, native):
    root,reference,repo,native = (p.resolve() for p in (root,reference,repo,native))
    require(not root.exists(), 'Use a fresh pooling queue root')
    fixture = read(reference/'fixture.json')
    require(sha(reference/'fixture.json') == read(reference/'dataset-summary.json')['fixture_sha256']
            and sha(reference/'profile.json') == fixture['profile_sha256'], 'Reference changed')
    validate_cohort(fixture['rows'],20)
    load_scorer(Path(fixture['upstream_path']))
    for row in fixture['rows']:
        p = Path(fixture['input_directory'])/(row['id']+'.i32')
        require(p.stat().st_size == row['served_input_tokens']*4 and sha(p) == row['served_sha256'],
                'Frozen input changed: '+row['id'])
    reference_profile = read(reference/'profile.json')
    names = set(read(reference/'runtime-proof.json')['files'])
    for folder in ('src/vllm_kvmem','scripts','csrc','third_party/qw3'):
        names.update(str(p.relative_to(repo)) for p in (repo/folder).rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.cpp','.hpp','.h','.cu','.cuh','.so'))
    names.update(('setup.py','pyproject.toml'))
    origins,inherited = runtime_origins(repo,reference,names)
    binaries = [name for name in origins if name.startswith('src/vllm_kvmem/_native') and name.endswith('.so')]
    require(len(binaries) == 1 and native.is_file(), 'Expected the CPU-tested native extension')
    origins[binaries[0]] = native
    hashes = {name:sha(p) for name,p in origins.items()}
    root.mkdir(parents=True)
    shared = root/'runtime'
    for name,p in origins.items():
        out = shared/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
        require(sha(out) == hashes[name], 'Runtime changed during freeze: '+name)
    proof = dict(files=hashes,inherited_utilities=inherited,source=str(repo),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        working_diff_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','diff','HEAD'],cwd=repo)).hexdigest(),
        native_extension_source=str(native),native_extension_sha256=sha(native))
    save(root/'runtime-proof.json',proof)
    campaigns=[]
    for arm in ARMS:
        campaign = root/arm;campaign.mkdir()
        (campaign/'runtime').symlink_to(shared,target_is_directory=True)
        profile = expected_profile(reference_profile,arm)
        variant = deepcopy(fixture)
        variant.update(execution_profile=PROFILE,pooling_arm=arm,reference_profile=reference_profile,
            source_fixture_sha256=sha(reference/'fixture.json'))
        validate_execution(profile,variant)
        # Full launch planning is CPU-only and validates native/guided resource
        # compatibility before any model is started.
        from vllm_kvmem.serve import build_launch
        build_launch(profile,repo,campaign/'preflight-server')
        save(campaign/'profile.json',profile)
        variant['profile_sha256']=sha(campaign/'profile.json')
        save(campaign/'fixture.json',variant)
        summary=read(reference/'dataset-summary.json')
        summary.update(fixture_sha256=sha(campaign/'fixture.json'),execution_profile=PROFILE)
        save(campaign/'dataset-summary.json',summary)
        save(campaign/'runtime-proof.json',proof)
        save(campaign/'manifest.json',dict(arm=arm,reference=str(reference),cases=260,
            fixture_sha256=sha(campaign/'fixture.json'),profile_sha256=sha(campaign/'profile.json'),
            runtime_proof_sha256=sha(campaign/'runtime-proof.json'),profile_changes=differences(reference_profile,profile),
            same_original_inputs_sampling_and_seeds=True,reused_answers=False,
            guidance_scope='Static explicit Query boundary guidance; prefill pressure uses cumulative attention.'))
        campaigns.append(dict(arm=arm,path=str(campaign),manifest_sha256=sha(campaign/'manifest.json')))
    require(all(sha(p)==hashes[name] for name,p in origins.items()), 'Source changed during preparation')
    save(root/'queue.json',dict(version=1,order=list(ARMS),campaigns=campaigns,
        prerequisite=str(repo/'artifacts/ruler-v1-256k-native-20261022-r1'),
        reference=str(reference),runtime_proof_sha256=sha(root/'runtime-proof.json')))
    print(json.dumps(dict(status='prepared',root=str(root),order=list(ARMS),cases_per_arm=260)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','repo','native'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();prepare(a.root,a.reference,a.repo,a.native)

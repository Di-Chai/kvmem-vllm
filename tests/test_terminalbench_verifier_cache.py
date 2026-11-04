import io
import json
from pathlib import Path
import tarfile
import subprocess
import hashlib

import pytest

from scripts import terminalbench_verifier_cache as cache


def make_archive(path, *, name='uv/wheels-v5/pkg.whl', link=None, hard=False, special=False):
    with tarfile.open(path, 'w') as out:
        item = tarfile.TarInfo(name)
        if link is not None:
            item.type = tarfile.LNKTYPE if hard else tarfile.SYMTYPE
            item.linkname = link
            out.addfile(item)
            item = tarfile.TarInfo('uv/content')
        if special:
            item.type = tarfile.FIFOTYPE
            out.addfile(item)
        else:
            item.size = 4
            out.addfile(item, io.BytesIO(b'data'))


@pytest.mark.parametrize('kwargs', [
    {'name':'app/answer.py'}, {'name':'/root/.cache/uv/file'}, {'name':'uv/../app/file'},
    {'link':'../../app'}, {'link':'/app'}, {'link':'/root/.cache/uv/other'},
    {'link':'../app/file','hard':True}, {'special':True},
])
def test_archive_rejects_noncache_paths_and_escape_links(tmp_path, kwargs):
    path = tmp_path/'cache.tar'
    make_archive(path, **kwargs)
    with pytest.raises(ValueError):
        cache.archive_members(path)


@pytest.mark.parametrize('kwargs', [{}, {'link':'../archive-v0/pkg'}, {'link':'uv/other','hard':True}])
def test_archive_accepts_only_internal_package_links(tmp_path, kwargs):
    path = tmp_path/'cache.tar'
    make_archive(path, **kwargs)
    assert cache.archive_members(path)['uncompressed_file_bytes'] == 4


def test_portable_export_keeps_package_bytes_and_drops_executable_environment(tmp_path):
    raw=tmp_path/'raw.tar';out=tmp_path/'portable.tar'
    with tarfile.open(raw,'w') as archive:
        for name,payload in [('uv/archive-v0/pkg/module.py',b'package bytes'),
                             ('uv/archive-v0/venv/pyvenv.cfg',b'interpreter path'),
                             ('uv/archive-v0/venv/bin/python-copy',b'not a package'),
                             ('uv/interpreter-v4/metadata',b'obsolete location')]:
            item=tarfile.TarInfo(name);item.size=len(payload);archive.addfile(item,io.BytesIO(payload))
        link=tarfile.TarInfo('uv/wheels-v5/package');link.type=tarfile.SYMTYPE
        link.linkname='/root/.cache/uv/archive-v0/pkg';archive.addfile(link)
    result=cache.portable_archive(raw,out,'/root/.cache/uv')
    assert result['relocated_cache_links']==1
    assert result['removed_disposable_environment_members']==3
    assert cache.archive_members(out)['uncompressed_file_bytes']==len(b'package bytes')
    with tarfile.open(out) as archive:
        assert archive.extractfile('uv/archive-v0/pkg/module.py').read()==b'package bytes'
        assert archive.getmember('uv/wheels-v5/package').linkname=='../archive-v0/pkg'


def qualified(tmp_path):
    path = tmp_path/'cache';path.mkdir()
    make_archive(path/'uv-cache.tar')
    downloads=tmp_path/'download-cache';uv=downloads/cache.UV_ARCHIVE;uv.parent.mkdir(parents=True)
    uv.write_bytes(b'public archive')
    (downloads/'manifest.json').write_text(json.dumps(dict(passed=True,proxy_disabled=True,
        files={cache.UV_ARCHIVE:dict(sha256=cache.sha(uv),official_digest='sha256:'+cache.sha(uv))})))
    proof = dict(task='torch-tensor-parallelism', image='official:image', image_id='sha256:'+'a'*64,
                 requirements=list(cache.REQUIREMENTS['torch-tensor-parallelism']), python='3.13',
                 passed=True, clean_official_image=True, offline_validation=True, no_tests_executed=True,
                 private_interpreter=True,offline_verifier_resolution=True,
                 download_manifest_sha256=cache.sha(downloads/'manifest.json'),
                 original_test_sha256=hashlib.sha256(b'original verifier').hexdigest(),
                 archive_sha256=cache.sha(path/'uv-cache.tar'),
                 archive_members=cache.archive_members(path/'uv-cache.tar'))
    (path/'manifest.json').write_text(json.dumps(proof))
    return path,proof


def test_cache_qualification_binds_task_image_and_bytes(tmp_path):
    path,proof = qualified(tmp_path)
    digest = cache.sha(path/'manifest.json')
    assert cache.validate_cache(path,proof['task'],proof['image'],digest) == proof
    with pytest.raises(ValueError):cache.validate_cache(path,'tune-mjcf',proof['image'],digest)
    with pytest.raises(ValueError):cache.validate_cache(path,proof['task'],'another:image',digest)
    with pytest.raises(ValueError):cache.validate_cache(path,proof['task'],proof['image'],'0'*64)
    with (path/'uv-cache.tar').open('ab') as out:out.write(b'changed')
    with pytest.raises(ValueError,match='archive changed'):
        cache.validate_cache(path,proof['task'],proof['image'],digest)


@pytest.mark.parametrize('damage', ['missing_manifest', 'missing_download', 'changed_download',
                                  'changed_archive', 'changed_test', None])
def test_worker_preflight_checks_complete_verifier_dependencies(tmp_path, damage):
    path, proof = qualified(tmp_path)
    task = tmp_path/'dataset/tasks'/proof['task']
    (task/'tests').mkdir(parents=True)
    (task/'tests/test.sh').write_bytes(b'original verifier')
    (task/'task.toml').write_text('[environment]\ndocker_image="official:image"\n')
    manifest = dict(verifier_dependency_caches={proof['task']:dict(
        path='cache', manifest_sha256=cache.sha(path/'manifest.json'))})
    download = tmp_path/'download-cache'/cache.UV_ARCHIVE
    if damage == 'missing_manifest':(tmp_path/'download-cache/manifest.json').unlink()
    elif damage == 'missing_download':download.unlink()
    elif damage == 'changed_download':download.write_bytes(b'changed')
    elif damage == 'changed_archive':(path/'uv-cache.tar').write_bytes(b'changed')
    elif damage == 'changed_test':(task/'tests/test.sh').write_bytes(b'changed')
    calls = []
    def inspect(*args, **kwargs):
        calls.append(args)
        assert args[:2] == ('image', 'inspect')
        return json.dumps([dict(Id=proof['image_id'])])
    if damage:
        with pytest.raises((FileNotFoundError, ValueError)):
            cache.preflight_cache_inputs(tmp_path, manifest, inspect)
        assert calls == []
    else:
        checked = cache.preflight_cache_inputs(tmp_path, manifest, inspect)
        assert checked[proof['task']]['download_manifest_sha256'] == proof['download_manifest_sha256']
        assert checked[proof['task']]['image_id'] == proof['image_id']
        assert calls == [('image', 'inspect', proof['image'])]


def test_worker_without_package_caches_still_needs_its_public_download_mirror(tmp_path):
    with pytest.raises(FileNotFoundError):
        cache.preflight_cache_inputs(tmp_path, {})
    qualified(tmp_path)
    assert cache.preflight_cache_inputs(tmp_path, {}) == {}


def detail(environment):
    return dict(Config=dict(Labels={'com.docker.compose.project':'trial__env',
        'com.docker.compose.service':'main','com.docker.compose.project.working_dir':str(environment.resolve())}),
        Image='sha256:'+'a'*64, State=dict(Running=True,Paused=False,Restarting=False))


@pytest.mark.parametrize('change', ['project','directory','image','stopped','paused','restarting'])
def test_cache_install_refuses_foreign_or_inactive_container(tmp_path,change):
    env=tmp_path/'task/environment';row=detail(env)
    if change=='project':row['Config']['Labels']['com.docker.compose.project']='foreign__env'
    if change=='directory':row['Config']['Labels']['com.docker.compose.project.working_dir']='/foreign'
    if change=='image':row['Image']='sha256:'+'b'*64
    if change=='stopped':row['State']['Running']=False
    if change=='paused':row['State']['Paused']=True
    if change=='restarting':row['State']['Restarting']=True
    with pytest.raises(ValueError):cache.owned_container(row,'trial__env',env,'sha256:'+'a'*64)


def portable(tmp_path):
    path, proof = qualified(tmp_path)
    record = dict(schema_version=1, image=proof['image'], source_id=proof['image_id'],
                  target_id='sha256:'+'b'*64, source_rootfs=dict(Type='layers',Layers=['sha256:'+'c'*64]),
                  source_config=dict(Env=['A=1'],Cmd=['bash']),source_architecture='amd64',source_os='linux')
    binding=tmp_path/'image-binding.json';binding.write_text(json.dumps(record))
    entry=dict(path='cache',manifest_sha256=cache.sha(path/'manifest.json'),
               portable_image_identity=dict(path=binding.name,sha256=cache.sha(binding)))
    image=dict(Id=record['target_id'],RootFS=record['source_rootfs'],
               Config=dict(record['source_config'],User='',OpenStdin=False,Entrypoint=None),
               Architecture='amd64',Os='linux')
    return proof,entry,image,record


def test_portable_cache_preserves_qualified_manifest_and_compares_content(tmp_path):
    proof,entry,image,_=portable(tmp_path)
    before=cache.sha(tmp_path/'cache/manifest.json')
    call=lambda *args:json.dumps([image])
    assert cache.portable_image_id(tmp_path,entry,proof,call)==image['Id']
    assert cache.preflight_cache_image_ids(tmp_path,dict(verifier_dependency_caches={'task':entry}),call)=={'task':image['Id']}
    assert cache.sha(tmp_path/'cache/manifest.json')==before


@pytest.mark.parametrize('field,value',[
    ('Id','sha256:'+'d'*64),('RootFS',dict(Type='layers',Layers=['sha256:'+'e'*64])),
    ('Architecture','arm64'),('Os','windows'),('Config',dict(Env=['A=2'],Cmd=['bash'])),
    ('Config',dict(Env=['A=1'],Cmd=['bash'],User='somebody')),
])
def test_portable_cache_rejects_changed_content(tmp_path,field,value):
    proof,entry,image,_=portable(tmp_path);image[field]=value
    with pytest.raises(ValueError,match='differs'):
        cache.portable_image_id(tmp_path,entry,proof,lambda *args:json.dumps([image]))


def test_worker_preflight_catches_legacy_id_before_agent_launch(tmp_path):
    proof,entry,image,_=portable(tmp_path)
    entry.pop('portable_image_identity')
    with pytest.raises(ValueError,match='portable binding'):
        cache.preflight_cache_image_ids(tmp_path,dict(verifier_dependency_caches={'task':entry}),lambda *args:json.dumps([image]))


def test_portable_cache_rejects_unbound_or_wrong_source_mapping(tmp_path):
    proof,entry,image,record=portable(tmp_path)
    record['source_id']='sha256:'+'f'*64
    (tmp_path/'image-binding.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='changed'):
        cache.portable_image_id(tmp_path,entry,proof,lambda *args:json.dumps([image]))
    entry['portable_image_identity']['sha256']=cache.sha(tmp_path/'image-binding.json')
    with pytest.raises(ValueError,match='another source'):
        cache.portable_image_id(tmp_path,entry,proof,lambda *args:json.dumps([image]))


def test_verification_installs_only_private_cache_and_persists_failure(tmp_path):
    path,proof=qualified(tmp_path)
    task=tmp_path/'dataset/tasks'/proof['task'];(task/'environment').mkdir(parents=True)
    (task/'tests').mkdir();(task/'tests/test.sh').write_bytes(b'original verifier')
    (task/'task.toml').write_text('[environment]\ndocker_image="official:image"\n')
    entry=dict(path='cache',manifest_sha256=cache.sha(path/'manifest.json'))
    calls=[]
    def call(*args,**kwargs):
        calls.append(args)
        if args[0]=='ps':return 'c'*64
        if args[0]=='inspect':return json.dumps([detail(task/'environment')])
        if args[-3:]==('readlink','-f','/opt'):return '/opt'
        return ''
    result=cache.install_for_verification(tmp_path,entry,proof['task'],'trial',task,tmp_path/'audit.json',call)
    assert set(result)=={'UV_CACHE_DIR','UV_PYTHON_INSTALL_DIR','UV_OFFLINE'}
    assert result['UV_OFFLINE']=='1'
    assert result['UV_CACHE_DIR'].startswith('/opt/kvmem-verifier-cache-')
    assert all('/app' not in str(c) and '/tests' not in str(c) for c in calls)
    assert json.loads((tmp_path/'audit.json').read_text())['passed']
    def bad(*args,**kwargs):
        if args[0]=='cp':raise RuntimeError('copy failed')
        return call(*args,**kwargs)
    with pytest.raises(RuntimeError,match='copy failed'):
        cache.install_for_verification(tmp_path,entry,proof['task'],'trial',task,tmp_path/'failed.json',bad)
    assert json.loads((tmp_path/'failed.json').read_text())['passed'] is False


def test_cache_population_rejects_redirected_parent_before_writing(tmp_path):
    calls=[]
    def redirected(*args,**kwargs):calls.append(args);return '/app'
    with pytest.raises(ValueError,match='redirected'):
        cache.populate('owned',tmp_path/'unused.tar','/opt/kvmem-verifier-cache-'+'a'*32,redirected)
    assert len(calls)==1 and calls[0][-3:]==('readlink','-f','/opt')


def test_preparation_commands_cannot_execute_verifier_tests():
    for task,reqs in cache.REQUIREMENTS.items():
        args=cache.version_command(task,offline=True)
        assert args[-2:]==['pytest','--version'] and '--offline' in args
        assert all(req in args for req in reqs)
        assert not any('/tests' in arg or '/app' in arg for arg in args)


def test_actual_harbor_trial_override_does_not_mutate_shared_job_verifier():
    python=Path('/tmp/kvmem-terminus-20261026-env/bin/python')
    if not python.exists():
        pytest.skip('Pinned Harbor environment is not available')
    source='''
from pathlib import Path
from harbor.models.trial.config import TrialConfig, TaskConfig, VerifierConfig
from harbor.models.job.config import JobConfig
from harbor.models.trial.result import TrialResult
from scripts.terminalbench_verifier_cache import set_trial_cache_environment
from harbor.trial.trial import Trial, VerifierFactory
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch
import asyncio
shared=VerifierConfig(env={'KEEP':'original'})
a=TrialConfig(task=TaskConfig(path=Path('/a')),verifier=shared)
b=TrialConfig(task=TaskConfig(path=Path('/b')),verifier=shared)
assert a.verifier is b.verifier is shared
set_trial_cache_environment(a,{'UV_CACHE_DIR':'/opt/private-a/uv'})
assert a.verifier is not shared and b.verifier is shared
assert a.verifier.env=={'KEEP':'original','UV_CACHE_DIR':'/opt/private-a/uv'}
assert b.verifier.env==shared.env=={'KEEP':'original'}
set_trial_cache_environment(b,{'UV_CACHE_DIR':'/opt/private-b/uv'})
assert a.verifier.env['UV_CACHE_DIR']=='/opt/private-a/uv'
assert b.verifier.env['UV_CACHE_DIR']=='/opt/private-b/uv'
assert shared.env=={'KEEP':'original'}
async def verify():return 'verified'
fake=SimpleNamespace(config=a,task=None,paths=None,logger=None,
    _network_plan=lambda _:SimpleNamespace(agent_env_baseline=None,verifier_phase=None),
    agent_environment=SimpleNamespace(with_default_user=lambda _:nullcontext()),
    _phase_network_policy=lambda *a,**kw:nullcontext(),_log_context=lambda *a:nullcontext())
with patch.object(VerifierFactory,'create_verifier_from_config',return_value=SimpleNamespace(verify=verify)) as factory:
    assert asyncio.run(Trial._run_shared_verifier(fake,timeout_sec=1,user=None))=='verified'
    assert factory.call_args.args[0] is a.verifier
    assert factory.call_args.kwargs['override_env']['UV_CACHE_DIR']=='/opt/private-a/uv'
assert shared.env=={'KEEP':'original'}
'''
    subprocess.run([str(python),'-c',source],check=True,timeout=30)

"""Public verifier dependencies prepared in clean official task images.

Only UV's package cache is exported. No task answer, test, interpreter image, or
container snapshot is reused. At verification, a private cache is populated
before Harbor starts its original verifier timer; original test.sh is unchanged.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shlex
import subprocess
import tarfile
import time
import tomllib
import uuid

try:
    from .terminalbench_network import direct_environment, DIRECT_CONTAINER_ENV, CACHE_ENV
except ImportError:
    from terminalbench_network import direct_environment, DIRECT_CONTAINER_ENV, CACHE_ENV


REQUIREMENTS = {
    'torch-pipeline-parallelism': ('pytest==8.4.1', 'torch==2.7.0',
                                 'transformers==4.55.0', 'pytest-json-ctrf==0.3.5'),
    'torch-tensor-parallelism': ('pytest==8.4.1', 'torch==2.7.0', 'pytest-json-ctrf==0.3.5'),
    'tune-mjcf': ('pytest==8.4.1', 'mujoco==3.3.5', 'pytest-json-ctrf==0.3.5'),
    'mteb-retrieve': ('pytest==8.4.1', 'mteb==1.36.8', 'pytest-json-ctrf==0.3.5'),
    'pytorch-model-recovery': ('pytest==8.4.1', 'torch==2.7.1', 'pytest-json-ctrf==0.3.5'),
}
UV_ARCHIVE = 'astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(8*1024*1024):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def docker(*args, timeout=120):
    return subprocess.check_output(['docker', *map(str, args)], env=direct_environment(),
                                   text=True, timeout=timeout).strip()


def archive_members(path):
    """Require a cache-only archive whose links cannot escape its new root."""
    count = size = 0
    with tarfile.open(path, 'r:') as archive:
        for item in archive:
            name = PurePosixPath(item.name)
            if (name.is_absolute() or '..' in name.parts or not name.parts or
                    name.parts[0] != 'uv' or not (item.isfile() or item.isdir() or item.issym() or item.islnk())):
                raise ValueError('Archive contains a non-cache path or special file')
            if item.issym() or item.islnk():
                link = item.linkname
                target = posixpath.normpath(posixpath.join(str(name.parent), link) if item.issym() else link)
                if link.startswith('/') or not (target == 'uv' or target.startswith('uv/')):
                    raise ValueError('Cache archive link escapes its private root')
            count += 1
            size += item.size
    if not count or not size:
        raise ValueError('Empty verifier dependency cache')
    return dict(members=count, uncompressed_file_bytes=size)


def portable_archive(raw_path, destination, source_cache):
    """Drop generated environments; relocate cache-internal link metadata.

    Public package files are copied byte for byte. UV 0.9.5 stores disposable
    executable environments in archive-v0 as well as environments-v2 and uses
    absolute wheel-cache symlinks, so an unfiltered tar is neither portable nor
    suitable for safe extraction into an unrelated trial's private directory.
    """
    source_cache = source_cache.rstrip('/')
    with tarfile.open(raw_path, 'r:') as source:
        members = source.getmembers()
        environments = {str(PurePosixPath(m.name).parent) for m in members
                        if len(PurePosixPath(m.name).parts) == 4 and
                        PurePosixPath(m.name).parts[:2] == ('uv', 'archive-v0') and
                        PurePosixPath(m.name).name == 'pyvenv.cfg'}
        excluded = environments | {'uv/environments-v2', 'uv/interpreter-v4', 'uv/.lock'}
        removed = rewritten = 0
        with tarfile.open(destination, 'w') as target:
            for member in members:
                if any(member.name == p or member.name.startswith(p+'/') for p in excluded):
                    removed += 1
                    continue
                item = copy.copy(member)
                if item.issym() and item.linkname.startswith(source_cache+'/'):
                    inside = 'uv/'+item.linkname[len(source_cache)+1:]
                    item.linkname = posixpath.relpath(inside, str(PurePosixPath(item.name).parent))
                    rewritten += 1
                if item.islnk() and item.linkname.startswith(source_cache+'/'):
                    item.linkname = 'uv/'+item.linkname[len(source_cache)+1:]
                    rewritten += 1
                target.addfile(item, source.extractfile(member) if item.isfile() else None)
    return dict(removed_disposable_environment_members=removed, relocated_cache_links=rewritten,
                source_cache=source_cache, public_package_file_bytes_unchanged=True)


def validate_cache(directory, task, image, manifest_sha256=None):
    directory = Path(directory)
    path = directory/'manifest.json'
    if manifest_sha256 is not None and sha(path) != manifest_sha256:
        raise ValueError('Frozen verifier-cache manifest changed')
    proof = json.loads(path.read_text())
    if (task not in REQUIREMENTS or proof.get('task') != task or proof.get('image') != image or
            proof.get('requirements') != list(REQUIREMENTS[task]) or proof.get('python') != '3.13' or
            not proof.get('passed') or not proof.get('clean_official_image') or
            not proof.get('offline_validation') or not proof.get('no_tests_executed') or
            not proof.get('private_interpreter') or not proof.get('offline_verifier_resolution') or
            not re.fullmatch(r'sha256:[0-9a-f]{64}', proof.get('image_id', ''))):
        raise ValueError('Verifier cache identity or qualification mismatch')
    archive = directory/'uv-cache.tar'
    if archive.is_symlink() or sha(archive) != proof['archive_sha256']:
        raise ValueError('Verifier cache archive changed')
    if archive_members(archive) != proof['archive_members']:
        raise ValueError('Verifier cache membership changed')
    return proof


def owned_container(detail, project, environment_dir, image_id):
    labels = detail.get('Config', {}).get('Labels') or {}
    state = detail.get('State') or {}
    if (labels.get('com.docker.compose.project') != project or
            labels.get('com.docker.compose.service') != 'main' or
            labels.get('com.docker.compose.project.working_dir') != str(Path(environment_dir).resolve()) or
            detail.get('Image') != image_id or not state.get('Running') or state.get('Paused') or state.get('Restarting')):
        raise ValueError('Verifier cache target is not the exact owned running task container')


def portable_image_id(root, entry, proof, call=docker):
    """Resolve an explicitly frozen cross-store identity using actual content.

    Docker's classic and containerd image stores may expose different IDs for
    the same imported image. Never accept a tag match alone or weaken the
    container's project/directory/running-state ownership checks.
    """
    binding = entry.get('portable_image_identity')
    if binding is None:
        return proof['image_id']
    root = Path(root).resolve()
    path = root / binding['path']
    if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path) != binding['sha256']:
        raise ValueError('Portable image identity binding changed or escaped its run')
    record = json.loads(path.read_text())
    if (record.get('schema_version') != 1 or record.get('source_id') != proof['image_id'] or
            record.get('image') != proof['image'] or
            not re.fullmatch(r'sha256:[0-9a-f]{64}', record.get('target_id', ''))):
        raise ValueError('Portable image identity belongs to another source image')
    images = json.loads(call('image', 'inspect', record['target_id']))
    if len(images) != 1:
        raise ValueError('Ambiguous portable image identity')
    actual = images[0]
    defaults = dict(Hostname='', Domainname='', User='', Image='', WorkingDir='',
                    AttachStdin=False, AttachStdout=False, AttachStderr=False,
                    Tty=False, OpenStdin=False, StdinOnce=False,
                    Entrypoint=None, OnBuild=None, Volumes=None, Labels=None)
    def canonical(config):
        return {k:v for k,v in config.items() if k not in defaults or
                type(v) is not type(defaults[k]) or v != defaults[k]}
    if (actual['Id'] != record['target_id'] or actual['RootFS'] != record['source_rootfs'] or
            actual['Architecture'] != record['source_architecture'] or actual['Os'] != record['source_os'] or
            canonical(actual['Config']) != canonical(record['source_config'])):
        raise ValueError('Portable image content differs from the qualified source')
    return record['target_id']


def preflight_cache_image_ids(root, manifest, call=docker):
    """Catch a nonportable cache binding before any formal Agent is launched."""
    root = Path(root).resolve()
    checked = {}
    for task, entry in manifest.get('verifier_dependency_caches', {}).items():
        path = root / entry['path'] / 'manifest.json'
        if not path.resolve().is_relative_to(root) or sha(path) != entry['manifest_sha256']:
            raise ValueError('Verifier-cache manifest changed before worker launch')
        proof = json.loads(path.read_text())
        expected = portable_image_id(root, entry, proof, call)
        images = json.loads(call('image', 'inspect', proof['image']))
        if len(images) != 1 or images[0]['Id'] != expected:
            raise ValueError('Verifier cache image ID needs a verified portable binding: '+task)
        checked[task] = expected
    return checked


def populate(cid, archive, destination, call=docker):
    # The destination must be freshly created: never follow task-controlled
    # symlinks in ~/.cache or overwrite cache state produced by the Agent.
    if not re.fullmatch(r'/opt/kvmem-verifier-cache-[0-9a-f]{32}', destination):
        raise ValueError('Unexpected private verifier cache destination')
    if call('exec', '-u', 'root', cid, 'readlink', '-f', '/opt') != '/opt':
        raise ValueError('Private verifier-cache parent was redirected by the task')
    call('exec', '-u', 'root', cid, 'mkdir', '-m', '700', destination)
    remote = destination+'/uv-cache.tar'
    call('cp', str(archive), cid+':'+remote, timeout=600)
    expected = sha(archive)
    check = f"printf '%s  %s\\n' {shlex.quote(expected)} {shlex.quote(remote)} | sha256sum -c -"
    call('exec', '-u', 'root', cid, 'sh', '-c', check, timeout=120)
    call('exec', '-u', 'root', cid, 'tar', '--no-same-owner', '-xf', remote, '-C', destination, timeout=600)
    call('exec', '-u', 'root', cid, 'rm', remote)
    return destination+'/uv'


def verify_downloads(directory, expected_manifest=None):
    directory=Path(directory)
    if expected_manifest is not None and sha(directory/'manifest.json') != expected_manifest:
        raise ValueError('Verified interpreter cache identity changed')
    proof=json.loads((directory/'manifest.json').read_text())
    if not proof.get('passed') or not proof.get('proxy_disabled') or UV_ARCHIVE not in proof.get('files', {}):
        raise ValueError('Unqualified interpreter/download cache')
    for name,value in proof['files'].items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not name.startswith('astral-sh/'):
            raise ValueError('Unsafe interpreter archive path')
        if sha(directory/name) != value['sha256'] or value['official_digest'] != 'sha256:'+value['sha256']:
            raise ValueError('Public download archive changed')
    return proof


def preflight_cache_inputs(root, manifest, call=docker):
    """Check the complete dependency chain before any scored task starts."""
    root = Path(root).resolve()
    # Ordinary verifiers use this mirror as well, even without a prewarmed
    # package cache. Validate it for every worker cohort.
    verify_downloads(root/'download-cache')
    checked = {}
    for task, entry in manifest.get('verifier_dependency_caches', {}).items():
        task_path = (root/'dataset/tasks'/task).resolve()
        directory = (root/entry['path']).resolve()
        if (not task_path.is_relative_to(root/'dataset/tasks') or
                not directory.is_relative_to(root)):
            raise ValueError('Verifier cache or task outside the frozen experiment')
        # Check the shared interpreter cache before hashing a multi-GiB package
        # archive. A package-cache proof alone does not qualify its bootstrap.
        proof_path = directory/'manifest.json'
        if sha(proof_path) != entry['manifest_sha256']:
            raise ValueError('Frozen verifier-cache manifest changed')
        proof = json.loads(proof_path.read_text())
        verify_downloads(root/'download-cache', proof['download_manifest_sha256'])
        config = tomllib.loads((task_path/'task.toml').read_text())
        proof = validate_cache(directory, task, config['environment']['docker_image'],
                               entry['manifest_sha256'])
        if sha(task_path/'tests/test.sh') != proof['original_test_sha256']:
            raise ValueError('Verifier dependency cache belongs to a different original test script')
        checked[task] = dict(archive_sha256=proof['archive_sha256'],
            download_manifest_sha256=proof['download_manifest_sha256'],
            original_test_sha256=proof['original_test_sha256'])
    images = preflight_cache_image_ids(root, manifest, call)
    for task, image_id in images.items():
        checked[task]['image_id'] = image_id
    return checked


def bootstrap_private(cid, destination, call=docker):
    """Install the verified public interpreter away from Agent-owned paths."""
    if not re.fullmatch(r'/opt/kvmem-verifier-cache-[0-9a-f]{32}', destination):
        raise ValueError('Unexpected private verifier cache destination')
    binary=destination+'/bin'
    python=destination+'/python'
    script=('set -eu; mkdir '+shlex.quote(binary)+'; tar -xzf '+
            shlex.quote('/opt/kvmem-download-cache/'+UV_ARCHIVE)+
            ' --strip-components=1 -C '+shlex.quote(binary)+'; '+
            shlex.quote(binary+'/uv')+' python install 3.13')
    args=['exec','-u','root']
    for name,value in {**CACHE_ENV,'UV_CACHE_DIR':destination+'/uv','UV_PYTHON_INSTALL_DIR':python}.items():
        args+=['-e',name+'='+value]
    output=call(*args,cid,'sh','-c',script,timeout=300)
    return {'UV_CACHE_DIR':destination+'/uv','UV_PYTHON_INSTALL_DIR':python,'UV_OFFLINE':'1'},output


def install_for_verification(root, entry, task, trial, task_path, audit_path, call=docker):
    root, task_path = Path(root).resolve(), Path(task_path).resolve()
    directory = (root/entry['path']).resolve()
    if not directory.is_relative_to(root) or not task_path.is_relative_to(root/'dataset/tasks'):
        raise ValueError('Verifier cache or task outside the frozen experiment')
    cfg = tomllib.loads((task_path/'task.toml').read_text())
    proof = validate_cache(directory, task, cfg['environment']['docker_image'], entry['manifest_sha256'])
    image_id = portable_image_id(root, entry, proof, call)
    if sha(task_path/'tests/test.sh') != proof['original_test_sha256']:
        raise ValueError('Verifier dependency cache belongs to a different original test script')
    verify_downloads(root/'download-cache',proof['download_manifest_sha256'])
    project = trial.lower()+'__env'
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]+__env', project):
        raise ValueError('Unexpected trial project')
    audit = dict(task=task, trial=trial, started_at=time.time(), archive_sha256=proof['archive_sha256'],
                 manifest_sha256=entry['manifest_sha256'], no_task_files_copied=True)
    try:
        ids = call('ps', '-q', '--no-trunc', '--filter', 'label=com.docker.compose.project='+project,
                   '--filter', 'label=com.docker.compose.service=main').split()
        if len(ids) != 1:
            raise ValueError('Expected exactly one owned task container')
        cid = ids[0]
        detail = json.loads(call('inspect', cid))[0]
        owned_container(detail, project, task_path/'environment', image_id)
        destination = '/opt/kvmem-verifier-cache-'+uuid.uuid4().hex
        cache = populate(cid, directory/'uv-cache.tar', destination, call)
        cache_env,bootstrap_log=bootstrap_private(cid,destination,call)
        # Recheck identity after copy rather than reporting success if a trial
        # container stopped/restarted while the public cache was being staged.
        owned_container(json.loads(call('inspect', cid))[0], project, task_path/'environment', image_id)
        audit.update(passed=True, container=cid, cache_directory=cache, verifier_env=cache_env,
                     qualified_source_image_id=proof['image_id'], container_image_id=image_id,
                     bootstrap_output=bootstrap_log, finished_at=time.time())
        save(audit_path, audit)
        return cache_env
    except BaseException as error:
        audit.update(passed=False, error=repr(error), finished_at=time.time())
        save(audit_path, audit)
        raise


def set_trial_cache_environment(config, cache_env):
    """Detach Harbor's shared job verifier before assigning a trial-only path."""
    if any(key in config.verifier.env for key in cache_env):
        raise ValueError('Do not replace an explicit verifier cache directory')
    config.verifier = config.verifier.model_copy(deep=True)
    config.verifier.env.update(cache_env)


def exec_logged(cid, argv, log, *, env=None, timeout=3600):
    command = ['docker', 'exec', '-u', 'root', '-w', '/tmp']
    for name, value in (env or {}).items():
        command += ['-e', name+'='+str(value)]
    command += [cid, *argv]
    with Path(log).open('xb') as stream:
        subprocess.run(command, env=direct_environment(), stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def bootstrap(cid, log):
    archive = '/opt/kvmem-download-cache/'+UV_ARCHIVE
    script = ('set -eu; mkdir -p /opt/kvmem-uv; '
              f'tar -xzf {shlex.quote(archive)} --strip-components=1 -C /opt/kvmem-uv; '
              '/opt/kvmem-uv/uv python install 3.13')
    exec_logged(cid, ['sh', '-c', script], log, env=CACHE_ENV, timeout=300)


def version_command(task, offline=False):
    result = ['/opt/kvmem-uv/uvx']
    if offline:
        result.append('--offline')
    result += ['-p', '3.13']
    for requirement in REQUIREMENTS[task]:
        result += ['-w', requirement]
    return result+['pytest', '--version']


def prepare(output, dataset, downloads, tasks=None):
    output, dataset, downloads = Path(output).resolve(), Path(dataset).resolve(), Path(downloads).resolve()
    output.mkdir(parents=True, exist_ok=True)
    verify_downloads(downloads)
    selected = list(REQUIREMENTS) if tasks is None else tasks
    for task in selected:
        requirements = REQUIREMENTS[task]
        dest = output/task
        dest.mkdir(exist_ok=False)
        cfg = tomllib.loads((dataset/'tasks'/task/'task.toml').read_text())
        image = cfg['environment']['docker_image']
        image_id = json.loads(docker('image', 'inspect', image))[0]['Id']
        proof = dict(task=task, image=image, image_id=image_id, python='3.13',
                     requirements=list(requirements), clean_official_image=True, no_tests_executed=True,
                     download_manifest_sha256=sha(downloads/'manifest.json'))
        for phase in ('prepare', 'offline-validation'):
            args = ['create', '--name', 'kb-public-verifier-'+uuid.uuid4().hex[:16],
                    '--label', 'kvmem.scope=clean-public-verifier-cache', '--network', 'host' if phase=='prepare' else 'none',
                    '--cpus', '1', '--memory', '8192m', '--entrypoint', '/bin/sh',
                    '--mount', 'type=bind,source='+str(downloads)+',target=/opt/kvmem-download-cache,readonly']
            for name, value in {**DIRECT_CONTAINER_ENV, **CACHE_ENV}.items():
                args += ['-e', name+'='+value]
            cid = docker(*args, image_id, '-c', 'sleep infinity')
            save(dest/(phase+'-container.json'), dict(container=cid, image_id=image_id, at=time.time()))
            save(output/'status.json', dict(task=task, phase=phase, at=time.time(), container=cid))
            docker('start', cid)
            try:
                if phase == 'prepare':
                    bootstrap(cid, dest/(phase+'-bootstrap.log'))
                    # Tensor shares only freshly prepared public package bytes,
                    # never an image snapshot or files from an earlier task run.
                    prior = output/'torch-pipeline-parallelism/uv-cache.tar'
                    if task == 'torch-tensor-parallelism' and prior.exists():
                        cache = populate(cid, prior, '/opt/kvmem-verifier-cache-'+uuid.uuid4().hex)
                    else:
                        cache = '/root/.cache/uv'
                    exec_logged(cid, version_command(task), dest/'prepare-dependencies.log',
                                env={'UV_CACHE_DIR':cache, 'UV_LINK_MODE':'copy'})
                    archive = dest/'uv-cache.tar'
                    raw_archive = dest/'prepare-export.raw.tar'
                    # UV's executable environments contain absolute shebangs.
                    # Export package/index caches only and rebuild the tiny
                    # executable environment at the destination path.
                    with raw_archive.open('xb') as stream:
                        subprocess.run(['docker', 'exec', '-u', 'root', cid, 'tar',
                            '--exclude=uv/environments-v2', '--exclude=uv/.lock', '-cf', '-',
                            '-C', str(PurePosixPath(cache).parent), 'uv'],
                            env=direct_environment(), stdout=stream, check=True, timeout=600)
                    relocation = portable_archive(raw_archive, archive, cache)
                    proof.update(archive_sha256=sha(archive), archive_members=archive_members(archive),
                                 portable_export=relocation, original_test_sha256=sha(dataset/'tasks'/task/'tests/test.sh'))
                else:
                    destination='/opt/kvmem-verifier-cache-'+uuid.uuid4().hex
                    populate(cid, dest/'uv-cache.tar', destination)
                    env,log=bootstrap_private(cid,destination)
                    (dest/'offline-validation-bootstrap.log').write_text(log)
                    command=version_command(task, offline=True)
                    command[0]=destination+'/bin/uvx'
                    exec_logged(cid, command, dest/'offline-validation.log',
                                env={**env,'UV_LINK_MODE':'copy'}, timeout=600)
                    proof.update(passed=True, offline_validation=True, private_interpreter=True,
                                 offline_verifier_resolution=True, at=time.time())
                    save(dest/'manifest.json', proof)
                    validate_cache(dest, task, image)
            finally:
                docker('stop', '--time', '2', cid)
                detail = json.loads(docker('inspect', cid))[0]
                save(dest/(phase+'-cleanup.json'), dict(container=cid, running=detail['State']['Running']))
    save(output/'status.json', dict(status='complete', tasks=selected, at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--download-cache', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=list(REQUIREMENTS))
    args = parser.parse_args()
    prepare(args.output, args.dataset, args.download_cache, args.tasks)

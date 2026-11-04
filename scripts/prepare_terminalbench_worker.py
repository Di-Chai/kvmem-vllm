"""Stage an isolated CPU worker without starting or taking over any formal trial.

The model, bridge, token journals and KV owners stay on the serving host. The
worker uses a loopback SSH tunnel to the bridge. A current queue must hand off
ownership before the staged formal configuration can be used.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import time


HOOKS = ('terminalbench_network.py', 'terminalbench_query.py',
         'terminalbench_harness.py', 'terminalbench_pty.py',
         'terminalbench_tool_bundle.py', 'terminalbench_verifier_cache.py')


def canonical_image_config(config):
    # Docker 28's classic store fills these empty defaults in inspect; Docker
    # 29's containerd store omits them. Preserve every non-default value.
    defaults = dict(Hostname='', Domainname='', User='', Image='', WorkingDir='',
                    AttachStdin=False, AttachStdout=False, AttachStderr=False,
                    Tty=False, OpenStdin=False, StdinOnce=False,
                    Entrypoint=None, OnBuild=None, Volumes=None, Labels=None)
    return {k: v for k, v in config.items() if k not in defaults or
            type(v) is not type(defaults[k]) or v != defaults[k]}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def reuse_smoke_image(target, image):
    """Reuse the original *unexecuted base image*, never a trial container snapshot."""
    if not re.fullmatch(r'[a-z0-9][a-z0-9._/:@-]+', image):
        raise ValueError('Invalid smoke image reference')
    target = Path(target)
    proof = read(target/'worker-provenance.json')
    path = target/'smoke-task/task.toml'
    if sha(path) != proof['source_files']['smoke-task/task.toml']:
        raise ValueError('Smoke configuration already modified')
    text = path.read_text()
    if text.count('[environment]') != 1 or 'docker_image' in text:
        raise ValueError('Unexpected smoke environment configuration')
    path.write_text(text.replace('[environment]', '[environment]\ndocker_image = "'+image+'"'))
    proof['smoke_image_reuse'] = dict(image=image, original_task_sha256=proof['source_files']['smoke-task/task.toml'],
                                    worker_task_sha256=sha(path), formal_tasks_unchanged=True)
    proof['copied_files']['smoke-task/task.toml'] = sha(path)
    save(target/'worker-provenance.json', proof)


def unexecuted(source):
    """START may wait on RAM; ENVIRONMENT_START or any answer rules out handoff."""
    source = Path(source)
    events = [json.loads(line) for line in
              (source/'formal-events/events.jsonl').read_text().splitlines()]
    tasks = {Path(t['path']).name for t in read(source/'job.json')['tasks']}
    excluded = {Path(e['task']).name for e in events if e.get('task') and
                e.get('event') not in ('start', 'waiting_for_memory')}
    for trial in (source/'jobs/formal').iterdir():
        if not (trial/'config.json').is_file():
            continue
        name = Path(read(trial/'config.json')['task']['path']).name
        if ((trial/'result.json').exists() or
                any(p.is_file() for p in (trial/'agent').rglob('*')) or
                any(p.is_file() for p in (trial/'verifier').rglob('*'))):
            excluded.add(name)
    return sorted(tasks - excluded)


def worker_config(original, remote, names, bridge_base, *, smoke=False):
    config = deepcopy(original)
    if config['n_attempts'] != 1 or config['retry']['max_retries'] != 0:
        raise ValueError('Worker must retain one attempt and no task retries')
    config['jobs_dir'] = str(remote/'jobs')
    config['job_name'] = 'smoke-remote' if smoke else 'formal-remote'
    config['tasks'] = [dict(path=str(remote/('smoke-task' if smoke else 'dataset/tasks/'+name)))
                       for name in names]
    config['n_concurrent_trials'] = min(config['n_concurrent_trials'], len(names))
    config['environment']['extra_docker_compose'] = [str(remote/'download-cache-compose.json')]
    for agent in config['agents']:
        agent['kwargs']['api_base'] = bridge_base + '/v1'
    return config


def prepare(source, target, remote, *, bridge_base='http://127.0.0.1:18389',
            memory_gib=48):
    source, target, remote = Path(source).resolve(), Path(target), Path(remote)
    if target.exists() or not remote.is_absolute() or '..' in remote.parts:
        raise ValueError('New staging directory and absolute remote path required')
    if not bridge_base.startswith('http://127.0.0.1:'):
        raise ValueError('Use a loopback SSH tunnel, not an exposed model service')
    if type(memory_gib) is not int or memory_gib <= 0:
        raise ValueError('Positive memory budget required')
    selected = unexecuted(source)
    if not selected:
        raise ValueError('No unexecuted tasks to stage')
    original = read(source/'manifest.json')
    target.mkdir(parents=True)
    copied = {}

    def copy(relative, expected=None):
        src, dst = source/relative, target/relative
        if not src.is_file() or src.is_symlink() or not src.resolve().is_relative_to(source):
            raise ValueError('Invalid source file: '+str(relative))
        digest = sha(src)
        if expected is not None and digest != expected:
            raise ValueError('Frozen source changed: '+str(relative))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[str(relative)] = digest

    directories = ['smoke-task', 'download-cache'] + ['dataset/tasks/'+n for n in selected]
    directories += [value for name, value in original.get('terminal_tool_bundles', {}).items()
                    if name in selected]
    directories += [value['path'] for name, value in original.get('verifier_dependency_caches', {}).items()
                    if name in selected]
    for directory in directories:
        for src in sorted((source/directory).rglob('*')):
            if not src.is_file() or '__pycache__' in src.parts:
                continue
            rel = src.relative_to(source)
            top = rel.parts[0]
            expected = original['frozen'].get(top, {}).get(str(Path(*rel.parts[1:])))
            copy(rel, expected)
    for name in HOOKS:
        rel = 'runtime/scripts/'+name
        copy(rel, original['frozen']['runtime']['scripts/'+name])
    if original.get('agent_time_limit') == 'relaxed':
        rel = 'runtime/scripts/terminalbench_deadline.py'
        copy(rel, original['frozen']['runtime']['scripts/terminalbench_deadline.py'])
    # The new launcher changes admission/accounting only. The exact original
    # Query, PTY and tool hooks are retained for this campaign.
    launcher = Path(__file__).with_name('run_terminalbench_job.py')
    dst = target/'runtime/scripts/run_terminalbench_job.py'
    shutil.copy2(launcher, dst)
    for name in ('dependencies.json', 'profile.json', 'agent-time-policy.json'):
        copy(name, original['files'][name])
    for name in selected:
        copy('images/'+name+'.json')
    manifest = deepcopy(original)
    manifest.update(launched_task_names=selected, subset=True,
                    container_memory_admission_bytes=memory_gib*1024**3,
                    container_memory_host_reserve_bytes=12*1024**3)
    manifest['query_provenance']['tasks'] = {
        key: value for key, value in manifest['query_provenance']['tasks'].items()
        if key == 'smoke-task' or key in {'dataset/tasks/'+n for n in selected}}
    for key in ('terminal_tool_bundles', 'verifier_dependency_caches'):
        manifest[key] = {k: v for k, v in manifest.get(key, {}).items() if k in selected}
    # Do not present the source's model runtime digest as a worker runtime digest.
    for key in ('frozen', 'files', 'runtime_proof'):
        manifest.pop(key, None)
    manifest['worker_source'] = dict(root=str(source), manifest_sha256=sha(source/'manifest.json'),
                                     original_runtime_sha256=original['runtime_proof']['runtime_sha256'])
    save(target/'manifest.json', manifest)
    save(target/'formal-candidate.json', worker_config(read(source/'job.json'),remote,selected,bridge_base))
    save(target/'smoke-job.json', worker_config(read(source/'smoke-job.json'),remote,['smoke-task'],bridge_base,smoke=True))
    compose = read(source/'download-cache-compose.json')
    compose['services']['main']['volumes'][0]['source'] = str(remote/'download-cache')
    save(target/'download-cache-compose.json', compose)
    connection = read(source/'connection.json')
    connection['base'] = bridge_base
    save(target/'connection.json', connection)
    (target/'connection.json').chmod(0o600)
    proof = dict(at=time.time(), source=str(source), remote_root=str(remote),
                 status='staged_only', formal_launch_allowed=False,
                 reason='Original live controller still owns all pending trials; no ownership handoff yet',
                 selected_unexecuted_tasks=selected, source_files=copied,
                 controller_sha256=sha(dst), model_changes=False,
                 original_scores_changed=False, ssh_tunnel_base=bridge_base,
                 copied_files={str(p.relative_to(target)): sha(p) for p in target.rglob('*')
                               if p.is_file() and p.name != 'connection.json'})
    save(target/'worker-provenance.json', proof)
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('staging', type=Path)
    p.add_argument('remote_root', type=Path)
    p.add_argument('--memory-gib', type=int, default=48)
    a = p.parse_args()
    value = prepare(a.source, a.staging, a.remote_root, memory_gib=a.memory_gib)
    print(json.dumps({key: value[key] for key in
                     ('status', 'formal_launch_allowed', 'selected_unexecuted_tasks', 'remote_root')}))

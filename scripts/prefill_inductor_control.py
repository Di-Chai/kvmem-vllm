"""Freeze module-scoped Inductor tuning choices into isolated experiment caches.

This module imports only the standard library until an exact installed-Torch
cache-key resolver or the optional runtime diagnostic is explicitly requested.
No AOT graph, generated kernel, binary, or global cache is installed or changed.
The optional launch observer records actual first dispatches and is diagnostic
only; a final cache file by itself is never reported as launch verification.
"""
from collections.abc import Mapping, Sequence
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
import threading
import textwrap

CHAIN_KEY = 'inductor_seed_manifest_sha256'
DIRECTORY = 'inductor_seed'
RECEIPT = 'inductor_seed_install.json'
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_ROLE = re.compile(r'[a-z][a-z0-9_]*\Z')
_ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
_AOT = re.compile(r'(/[^\s\"\'<>]*?/torch_aot_compile/([0-9a-f]{64})/rank_[^/\s,]+/model)(?=[\s,\"\']|$)')
_IGNORED_CONFIG_KEYS = {'configs_hash', 'time_taken_ms', 'triton_cache_hash', 'found_by_coordesc', 'found_by_combo_autotune'}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError('Duplicate Inductor JSON key: ' + key)
        result[key] = value
    return result


def _read(path):
    def invalid(value):
        raise ValueError('Non-finite Inductor JSON value: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_pairs, parse_constant=invalid)


def _inside(root, path):
    root, path = Path(root).resolve(), Path(path)
    if '..' in path.parts:
        raise ValueError('Inductor path contains traversal: ' + str(path))
    result = path.resolve()
    if not result.is_relative_to(root):
        raise ValueError('Inductor path escapes its root: ' + str(path))
    return result


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def _write_json(path, value):
    _write(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n')


def _hashes(values):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError('AOT hashes must be a nonempty sequence')
    if any(type(value) is not str or not _HASH.fullmatch(value) for value in values):
        raise ValueError('Invalid AOT hash')
    if len(set(values)) != len(values):
        raise ValueError('Duplicate AOT hash')
    return sorted(values)


def _roles(mapping):
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError('Expected a nonempty module-role mapping')
    if any(type(role) is not str or not _ROLE.fullmatch(role) for role in mapping):
        raise ValueError('Invalid module role')
    return sorted(mapping)


def logged_aot_paths(server_log):
    """Return exact AOT model paths explicitly mentioned by vLLM's log."""
    paths = []
    for line in Path(server_log).read_text(errors='replace').splitlines():
        if not any(marker in line for marker in ('AOT compiled function', 'AOT compilation', 'load failure from')):
            continue
        for match in _AOT.finditer(_ANSI.sub('', line)):
            path = Path(match.group(1))
            if '..' in path.parts:
                raise ValueError('Logged AOT path contains traversal')
            if path not in paths:
                paths.append(path)
    return paths


def _installed_key_for_source(path):
    # Importing kernel.py would initialize CUDA; never do that. These two
    # installed-Torch functions only derive the local key from filename/build.
    from torch._inductor.runtime.autotune_cache import AutotuneCache
    path = Path(path)
    return Path(AutotuneCache._make_local_cache_key(
        str(path.parent), AutotuneCache._prepare_key(str(path))))


def _configuration(data):
    if type(data) is not dict or not _HASH.fullmatch(str(data.get('configs_hash', ''))):
        raise ValueError('Missing/invalid Inductor configs_hash')
    for name in ('num_warps', 'num_stages'):
        if type(data.get(name)) is not int or data[name] <= 0:
            raise ValueError('Invalid Inductor launch configuration: ' + name)
    if data.get('found_by_coordesc', False) or data.get('found_by_combo_autotune', False):
        raise ValueError('Retuned configurations are outside this controlled seed scope')
    selected = {key: value for key, value in data.items() if key not in _IGNORED_CONFIG_KEYS}
    if any(type(value) is not int for value in selected.values()):
        raise ValueError('Only integer Triton launch choices are supported')
    return selected


def _source_description(payload):
    """Inspect literal metadata without executing generated code or decorators."""
    text = payload.decode() if isinstance(payload, bytes) else payload
    tree = ast.parse(text)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    decorated = []
    for function in functions:
        for deco in function.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            meta = next((kw.value for kw in deco.keywords if kw.arg == 'inductor_meta'), None)
            if not isinstance(meta, ast.Dict):
                continue
            literals = {}
            for key, value in zip(meta.keys, meta.values):
                if isinstance(key, ast.Constant) and key.value in (
                    'kernel_name', 'coordinate_descent_tuning', 'combo_tuning_groups',
                    'incremental_autotune', 'force_disable_caches', 'autotune_local_cache'):
                    try:
                        literals[key.value] = ast.literal_eval(value)
                    except (ValueError, TypeError):
                        raise ValueError('Non-literal tuning control in generated source') from None
            for flag in ('coordinate_descent_tuning', 'combo_tuning_groups', 'incremental_autotune', 'force_disable_caches'):
                if literals.get(flag):
                    raise ValueError('Unsupported post-cache tuning/disabled cache: ' + flag)
            if literals.get('autotune_local_cache', True) is not True:
                raise ValueError('Local autotune cache is disabled')
            first_line = min([function.lineno] + [item.lineno for item in function.decorator_list]) - 1
            # Match Triton JITCallable's inspect.getsourcelines/dedent/strip-
            # decorators operation, including trailing in-function comments.
            function_text = textwrap.dedent(''.join(inspect.getblock(text.splitlines(keepends=True)[first_line:])))
            function_text = function_text[re.search(r'^def\s+\w+\s*\(', function_text, re.MULTILINE).start():]
            decorated.append(dict(kernel_name=literals.get('kernel_name', function.name), function_name=function.name,
                function_source_sha256=_sha(function_text.strip().encode())))
    if len(decorated) != 1:
        raise ValueError('Expected exactly one generated Triton kernel per source')
    return decorated[0]


def _scan_source(cache_dir, key_for_source):
    cache_dir = Path(cache_dir).resolve()
    by_key = {}
    for path in sorted(cache_dir.glob('*/*.py')):
        path = _inside(cache_dir, path)
        key = _inside(cache_dir, key_for_source(path))
        relative = str(key.relative_to(cache_dir))
        if relative in by_key and by_key[relative] != path:
            raise ValueError('Ambiguous source for Inductor cache key: ' + relative)
        by_key[relative] = path
    result = []
    for path in sorted(cache_dir.glob('*/*.best_config')):
        path = _inside(cache_dir, path)
        relative = str(path.relative_to(cache_dir))
        if not _HASH.fullmatch(path.stem) or len(path.parent.name) != 2:
            raise ValueError('Malformed Inductor local cache path')
        source = by_key.get(relative)
        if source is None:
            raise ValueError('No exact generated source for cache key: ' + relative)
        payload, source_payload = path.read_bytes(), source.read_bytes()
        data = _read(path)
        selected = _configuration(data)
        description = _source_description(source_payload)
        result.append(dict(key=relative, source_file=str(source.relative_to(cache_dir)),
            source_sha256=_sha(source_payload), source_bytes=len(source_payload),
            config_sha256=_sha(payload), configs_hash=data['configs_hash'],
            selected_config=selected, **description))
    if not result:
        raise ValueError('Module cache has no supported .best_config files')
    return result


def freeze_seed(arm, sources, allowed_aot_hashes, *, _key_for_source=None):
    """Freeze exact per-role source/cache bytes; sources[role] is a descriptor
    {cache_dir, server_log} or a sequence of descriptors. The role's allowed AOT
    hashes are destination hashes and need not equal its reference source hash.
    Duplicate keys must have identical source, config hash AND selected values;
    differing timing bytes are also rejected to avoid choosing an arbitrary file.
    """
    arm = Path(arm).resolve()
    roles = _roles(sources)
    if roles != _roles(allowed_aot_hashes):
        raise ValueError('Source and destination module roles differ')
    allowed = {role: _hashes(allowed_aot_hashes[role]) for role in roles}
    flat = [value for values in allowed.values() for value in values]
    if len(flat) != len(set(flat)):
        raise ValueError('An AOT hash cannot belong to multiple module roles')
    arm_path = _inside(arm, arm / 'manifest.json')
    parent = _read(arm_path)
    directory = _inside(arm, arm / DIRECTORY)
    if type(parent) is not dict or CHAIN_KEY in parent or directory.exists():
        raise ValueError('Inductor seed already exists or arm manifest is invalid')
    resolver = _key_for_source or _installed_key_for_source
    modules, pending = {}, []
    for role in roles:
        descriptors = sources[role]
        if isinstance(descriptors, Mapping):
            descriptors = [descriptors]
        if isinstance(descriptors, (str, bytes)) or not isinstance(descriptors, Sequence) or not descriptors:
            raise ValueError('Module sources must be nonempty descriptors')
        entries, provenance, required = {}, [], set()
        for descriptor in descriptors:
            if (not isinstance(descriptor, Mapping) or not {'cache_dir', 'server_log'} <= set(descriptor) or
                    set(descriptor) - {'cache_dir', 'server_log', 'required_common'}):
                raise ValueError('Source descriptor requires cache_dir/server_log and optional required_common')
            declared = descriptor.get('required_common', [])
            if not isinstance(declared, list) or any(type(key) is not str for key in declared) or len(set(declared)) != len(declared):
                raise ValueError('required_common must be a unique list of cache-key paths')
            required.update(declared)
            cache = Path(descriptor['cache_dir']).resolve()
            log = Path(descriptor['server_log']).resolve()
            source_hash = cache.parent.name
            if cache.name != 'inductor_cache' or not _HASH.fullmatch(source_hash) or cache.parent.parent.name != 'torch_aot_compile':
                raise ValueError('Source must be an explicit AOT hash/inductor_cache')
            observed = [path for path in logged_aot_paths(log) if path.parent.parent == cache.parent]
            if not observed:
                raise ValueError('Source AOT hash/path is absent from the supplied server log')
            source_records = _scan_source(cache, resolver)
            provenance.append(dict(cache_dir=str(cache), aot_hash=source_hash,
                server_log=str(log), server_log_sha256=_sha(log.read_bytes()),
                logged_aot_paths=[str(path) for path in observed]))
            for entry in source_records:
                key = entry['key']
                if key in entries:
                    if entries[key] != entry:
                        raise ValueError('Conflicting module-scoped Inductor key: ' + role + '/' + key)
                    continue
                entries[key] = entry
                config_bytes = (cache / key).read_bytes()
                source_bytes = (cache / entry['source_file']).read_bytes()
                if _sha(config_bytes) != entry['config_sha256'] or _sha(source_bytes) != entry['source_sha256']:
                    raise ValueError('Source cache changed while freezing')
                pending.extend(((directory / role / 'configs' / key, config_bytes),
                    (directory / role / 'sources' / entry['source_file'], source_bytes)))
        if not required <= set(entries):
            raise ValueError('required_common contains keys absent from module seed')
        modules[role] = dict(allowed_aot_hashes=allowed[role], sources=provenance, required_common=sorted(required),
            entries=[entries[key] for key in sorted(entries)])
    manifest = dict(schema=1, modules=modules, scope='Module-scoped local autotune choices; no AOT graphs or compiled binaries',
        helper_sha256=_sha(Path(__file__).read_bytes()))
    # Validate every source/collision before creating any frozen destination.
    directory.mkdir()
    for destination, payload in pending:
        _write(destination, payload)
    _write_json(directory / 'manifest.json', manifest)
    parent[CHAIN_KEY] = _sha((directory / 'manifest.json').read_bytes())
    temporary = arm / '.inductor-arm-manifest.tmp'
    _write_json(temporary, parent)
    temporary.replace(arm_path)
    return manifest


def validate_seed(arm):
    arm = Path(arm).resolve()
    directory = _inside(arm, arm / DIRECTORY)
    parent_path = _inside(arm, arm / 'manifest.json')
    parent = _read(parent_path) if parent_path.exists() else {}
    if type(parent) is not dict:
        raise ValueError('Invalid arm manifest')
    chain = parent.get(CHAIN_KEY)
    if chain is None and not directory.exists():
        if (arm / RECEIPT).exists():
            raise ValueError('Inductor installation exists without seed')
        return None
    path = _inside(arm, directory / 'manifest.json')
    if not _HASH.fullmatch(str(chain)) or not path.is_file() or _sha(path.read_bytes()) != chain:
        raise ValueError('Inductor seed manifest hash mismatch')
    manifest = _read(path)
    if manifest.get('schema') != 1 or manifest.get('helper_sha256') != _sha(Path(__file__).read_bytes()):
        raise ValueError('Unsupported Inductor seed schema')
    all_hashes = []
    for role in _roles(manifest.get('modules')):
        module = manifest['modules'][role]
        allowed = _hashes(module.get('allowed_aot_hashes'))
        if allowed != module['allowed_aot_hashes']:
            raise ValueError('Unsorted module AOT hashes')
        all_hashes.extend(allowed)
        entries = module.get('entries')
        if not isinstance(entries, list) or not entries:
            raise ValueError('Empty Inductor module seed')
        keys = set()
        for entry in entries:
            if entry['key'] in keys:
                raise ValueError('Duplicate frozen module key')
            keys.add(entry['key'])
            config = _inside(directory, directory / role / 'configs' / entry['key'])
            source = _inside(directory, directory / role / 'sources' / entry['source_file'])
            if _sha(config.read_bytes()) != entry['config_sha256'] or _sha(source.read_bytes()) != entry['source_sha256']:
                raise ValueError('Frozen Inductor source/config byte mismatch')
            data = _read(config)
            if data['configs_hash'] != entry['configs_hash'] or _configuration(data) != entry['selected_config']:
                raise ValueError('Frozen Inductor configuration semantics mismatch')
            if _source_description(source.read_bytes()) != {key: entry[key] for key in ('kernel_name', 'function_name', 'function_source_sha256')}:
                raise ValueError('Frozen generated source metadata mismatch')
        required = module.get('required_common')
        if not isinstance(required, list) or required != sorted(set(required)) or not set(required) <= keys:
            raise ValueError('Invalid required common-kernel set')
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError('Ambiguous AOT module ownership')
    return manifest


def install_seed(arm, target_run, env):
    """Prepare fresh per-run caches and mutate only the child environment map."""
    arm = Path(arm).resolve()
    manifest = validate_seed(arm)
    if manifest is None:
        return None
    run = _inside(arm, target_run)
    if not run.is_dir():
        raise ValueError('Create target_run before installing Inductor seed')
    root = _inside(arm, run / 'vllm_cache')
    initial = _inside(arm, run / 'inductor_cache_initial')
    receipt_path = _inside(arm, arm / RECEIPT)
    if root.exists() or initial.exists() or receipt_path.exists():
        raise ValueError('Isolated Inductor installation already exists')
    root.mkdir()
    initial.mkdir()
    installed = []
    for role, module in manifest['modules'].items():
        for aot_hash in module['allowed_aot_hashes']:
            cache = root / 'torch_compile_cache/torch_aot_compile' / aot_hash / 'inductor_cache'
            for entry in module['entries']:
                destination = cache / entry['key']
                _write(destination, (arm / DIRECTORY / role / 'configs' / entry['key']).read_bytes())
                installed.append(dict(role=role, aot_hash=aot_hash, key=entry['key'], path=str(destination.relative_to(arm))))
    receipt = dict(schema=1, seed_manifest_sha256=_sha((arm / DIRECTORY / 'manifest.json').read_bytes()),
        target_run=str(run.relative_to(arm)), cache_root=str(root.relative_to(arm)),
        initial_inductor_cache=str(initial.relative_to(arm)), server_log=str((run / 'server.log').relative_to(arm)),
        installed_files=installed, diagnostic_launch_params=False)
    _write_json(receipt_path, receipt)
    env['VLLM_CACHE_ROOT'] = str(root)
    env['TORCHINDUCTOR_CACHE_DIR'] = str(initial)
    return receipt


def audit_seed(arm, *, launch_audit=None, require_launch_audit=True, _key_for_source=None):
    """Audit actual logged AOT roots and generated common kernels. Optional
    launch JSONL proves dispatched configs; without it launch_verified is False.
    """
    arm = Path(arm).resolve()
    manifest = validate_seed(arm)
    if manifest is None:
        return None
    receipt_path = _inside(arm, arm / RECEIPT)
    receipt = _read(receipt_path)
    if receipt.get('schema') != 1 or receipt.get('seed_manifest_sha256') != _sha((arm / DIRECTORY / 'manifest.json').read_bytes()):
        raise ValueError('Inductor install receipt does not match seed')
    run = _inside(arm, arm / receipt['target_run'])
    root = _inside(arm, arm / receipt['cache_root'])
    initial = _inside(arm, arm / receipt['initial_inductor_cache'])
    log = _inside(arm, arm / receipt['server_log'])
    if root != run / 'vllm_cache' or initial != run / 'inductor_cache_initial' or log != run / 'server.log':
        raise ValueError('Inconsistent isolated Inductor receipt paths')
    allowed = {h: role for role, module in manifest['modules'].items() for h in module['allowed_aot_hashes']}
    paths = logged_aot_paths(log)
    if not paths:
        raise ValueError('No actual AOT model paths in server log')
    resolver = _key_for_source or _installed_key_for_source
    modules = {role: dict(actual_aot_hashes=[], common_kernels=[], unused_seed_keys=[], new_kernels=[]) for role in manifest['modules']}
    common_by_path = {}
    for path in paths:
        path = _inside(root, path)
        relative = path.relative_to(root)
        if len(relative.parts) != 5 or relative.parts[:2] != ('torch_compile_cache', 'torch_aot_compile') or relative.name != 'model':
            raise ValueError('Unexpected logged AOT path structure')
        aot_hash = relative.parts[2]
        if aot_hash not in allowed:
            raise ValueError('Server used an unapproved AOT hash')
        role, cache = allowed[aot_hash], path.parent.parent / 'inductor_cache'
        result = modules[role]
        if aot_hash in result['actual_aot_hashes']:
            continue
        result['actual_aot_hashes'].append(aot_hash)
        seed = {item['key']: item for item in manifest['modules'][role]['entries']}
        generated = {}
        for source in sorted(cache.glob('*/*.py')):
            source = _inside(cache, source)
            key = str(_inside(cache, resolver(source)).relative_to(cache))
            if key in generated:
                raise ValueError('Multiple generated sources map to one actual cache key')
            generated[key] = source
            if key not in seed:
                result['new_kernels'].append(dict(aot_hash=aot_hash, source_file=str(source.relative_to(cache)), source_sha256=_sha(source.read_bytes()), key=key))
                continue
            entry = seed[key]
            if _sha(source.read_bytes()) != entry['source_sha256'] or str(source.relative_to(cache)) != entry['source_file']:
                raise ValueError('Common Inductor kernel source changed: ' + role + '/' + key)
            data = _read(_inside(cache, cache / key))
            if data['configs_hash'] != entry['configs_hash'] or _configuration(data) != entry['selected_config']:
                raise ValueError('Common Inductor chosen configuration changed: ' + role + '/' + key)
            item = dict(entry, aot_hash=aot_hash, kernel_file=str(source), actual_config_file=str(cache / key), launch_verified=False)
            result['common_kernels'].append(item)
            common_by_path[str(source)] = item
        result['unused_seed_keys'].extend(dict(aot_hash=aot_hash, key=key) for key in sorted(set(seed) - set(generated)))
    for role, result in modules.items():
        if not result['actual_aot_hashes'] or not result['common_kernels']:
            raise ValueError('No actual generated common kernels for module: ' + role)
    for role, result in modules.items():
        required = set(manifest['modules'][role]['required_common'])
        observed = {item['key'] for item in result['common_kernels']}
        if not required <= observed:
            raise ValueError('Required common kernels not generated: ' + role + '/' + ','.join(sorted(required - observed)))
    if launch_audit is None:
        launch_paths = sorted(run.glob('inductor-startup-*.json'))
    elif isinstance(launch_audit, (str, Path)):
        launch_paths = [Path(launch_audit)]
    else:
        launch_paths = list(launch_audit)
    if not launch_paths and require_launch_audit:
        raise ValueError('Actual startup launch audit is required')
    launch_count = 0
    for launch_path in launch_paths:
        launch_path = _inside(arm, launch_path)
        report = _read(launch_path)
        if (report.get('schema') != 1 or report.get('success') is not True or
                report.get('observer_restored') is not True or report.get('error') is not None or
                report.get('scope') != 'startup-only; no formal dispatch hook' or
                report.get('seed_manifest_sha256') != receipt['seed_manifest_sha256']):
            raise ValueError('Startup launch observer did not close successfully')
        if type(report.get('records')) is not list:
            raise ValueError('Invalid startup launch record list')
        for row in report['records']:
            source = str(_inside(root, row['kernel_file']))
            if source not in common_by_path:
                continue
            entry = common_by_path[source]
            if (row.get('source_sha256') != entry['source_sha256'] or
                    row.get('selected_config') != entry['selected_config'] or
                    row.get('kernel_name') != entry['function_name'] or
                    row.get('single_launcher') is not True or
                    row.get('function_source_sha256') != entry['function_source_sha256']):
                raise ValueError('Actually dispatched common kernel differs from seed')
            if row.get('autotune_cache_state') != 'hit':
                raise ValueError('Common kernel did not consume seeded autotune cache')
            if any(row.get(name) for name in ('coordinate_descent_tuning', 'combo_tuning_groups', 'incremental_autotune')):
                raise ValueError('Actual kernel used unsupported retuning')
            entry['launch_verified'] = True
            launch_count += 1
    if launch_paths and not launch_count:
        raise ValueError('No common kernels observed in actual launch audit')
    if require_launch_audit and any(not item['launch_verified'] for item in common_by_path.values()):
        raise ValueError('Some generated common kernels have no actual startup launch evidence')
    return dict(modules=modules, seed_manifest_sha256=receipt['seed_manifest_sha256'],
        server_log=str(log), server_log_sha256=_sha(log.read_bytes()), isolated_cache_root=str(root),
        common_configurations_unchanged=True, launch_verified=bool(common_by_path) and all(item['launch_verified'] for item in common_by_path.values()),
        launch_records=launch_count, startup_audit_files=[str(path) for path in launch_paths],
        formal_dispatch_observer=False,
        scope='Generated-source and local-config audit; actual dispatch is proven only for launch_verified kernels')


class _LaunchAuditHandle:
    def __init__(self, cls, output, arm, manifest, receipt):
        self.cls, self.output, self.arm = cls, output, arm
        self.manifest, self.receipt = manifest, receipt
        self.original = cls._pre_launch
        self.records, self.seen, self.lock = [], set(), threading.RLock()
        self.condition, self.active = threading.Condition(self.lock), 0
        self.closed, self.result, self.error = False, None, None
        self.root = _inside(arm, arm / receipt['cache_root'])
        self.expected = {}
        for role, module in manifest['modules'].items():
            for aot_hash in module['allowed_aot_hashes']:
                cache = self.root / 'torch_compile_cache/torch_aot_compile' / aot_hash / 'inductor_cache'
                for entry in module['entries']:
                    self.expected[str(cache / entry['source_file'])] = (role, aot_hash, entry)

        handle = self
        def observed(autotuner, launcher, *args, **kwargs):
            with handle.condition:
                if handle.closed:
                    raise ValueError('Startup observer called after cleanup')
                handle.active += 1
            try:
                result = handle.original(autotuner, launcher, *args, **kwargs)
                handle.record(autotuner, launcher)
                return result
            except BaseException as error:
                handle.error = str(error)
                raise
            finally:
                with handle.condition:
                    handle.active -= 1
                    handle.condition.notify_all()
        self.wrapper = observed
        cls._pre_launch = observed
        cls._kvmem_inductor_launch_audit = self

    def record(self, autotuner, launcher):
        if autotuner.filename is None:
            return
        filename = str(Path(autotuner.filename).resolve())
        config = dict(launcher.config.kwargs, num_warps=launcher.config.num_warps,
                      num_stages=launcher.config.num_stages)
        identity = (filename, _canonical(config))
        with self.lock:
            # Validate every live object before deduplicating output: two
            # autotuners may share a filename/config but have different cache
            # provenance or post-cache tuning state.
            # Custom Triton kernels can legitimately live outside these caches.
            # Only generated files within the arm have an AOT module identity.
            if not Path(filename).is_relative_to(self.root):
                return
            metadata = autotuner.inductor_meta
            function_source = getattr(autotuner.fn, 'src', None)
            if type(function_source) is not str:
                raise ValueError('Missing actual Triton function source')
            row = dict(kernel_file=filename, source_sha256=_sha(Path(filename).read_bytes()),
                function_source_sha256=_sha(function_source.strip().encode()),
                kernel_name=autotuner.fn.__name__, selected_config=config,
                autotune_cache_state=(autotuner.autotune_cache_info or {}).get('autotune_cache_state'),
                single_launcher=len(autotuner.launchers) == 1 and autotuner.launchers[0] is launcher,
                coordinate_descent_tuning=metadata.get('coordinate_descent_tuning', False),
                combo_tuning_groups=metadata.get('combo_tuning_groups', False),
                incremental_autotune=metadata.get('incremental_autotune', False))
            if filename in self.expected:
                role, aot_hash, expected = self.expected[filename]
                row.update(role=role, aot_hash=aot_hash, key=expected['key'])
                if (row['source_sha256'] != expected['source_sha256'] or
                        row['selected_config'] != expected['selected_config'] or
                        row['kernel_name'] != expected['function_name'] or
                        row['function_source_sha256'] != expected['function_source_sha256'] or
                        row['autotune_cache_state'] != 'hit' or not row['single_launcher'] or
                        any(row[name] for name in ('coordinate_descent_tuning', 'combo_tuning_groups', 'incremental_autotune'))):
                    raise ValueError('Startup common kernel did not execute the frozen configuration: ' + role + '/' + expected['key'])
            if identity not in self.seen:
                self.records.append(row)
                self.seen.add(identity)

    def close(self, *, success):
        """Restore the class hook before checking/writing. Same-status close is
        idempotent; failure always writes a failed report and never claims coverage.
        """
        if type(success) is not bool:
            raise ValueError('Startup success must be boolean')
        if self.closed:
            if self.result['requested_success'] != success and not (not success and not self.result['success']):
                raise ValueError('Conflicting startup observer close status')
            if success and self.result['error']:
                raise ValueError(self.result['error'])
            return self.result
        restored = self.cls._pre_launch is self.wrapper
        if restored:
            self.cls._pre_launch = self.original
            delattr(self.cls, '_kvmem_inductor_launch_audit')
        else:
            self.error = self.error or 'Startup observer was replaced before cleanup'
        with self.condition:
            while self.active:
                self.condition.wait()
        if success and self.error is None:
            try:
                observed = {row['kernel_file'] for row in self.records}
                for role, module in self.manifest['modules'].items():
                    seen_keys = {row.get('key') for row in self.records if row.get('role') == role}
                    if not set(module['required_common']) <= seen_keys:
                        raise ValueError('Missing required startup common kernels: ' + role)
                    generated = [path for path, value in self.expected.items()
                                 if value[0] == role and Path(path).is_file()]
                    if not generated or any(path not in observed for path in generated):
                        raise ValueError('Generated common kernels lack startup dispatch evidence: ' + role)
            except BaseException as error:
                self.error = str(error)
        self.result = dict(schema=1, scope='startup-only; no formal dispatch hook',
            requested_success=success, success=success and self.error is None and restored,
            observer_restored=restored, error=self.error,
            seed_manifest_sha256=self.receipt['seed_manifest_sha256'], records=self.records)
        self.closed = True
        # The destination is reserved on install. Atomic replacement makes an
        # incomplete/failed startup distinguishable from a successful close.
        temporary = self.output.with_suffix(self.output.suffix + '.tmp')
        _write_json(temporary, self.result)
        temporary.replace(self.output)
        if success and self.error is not None:
            raise ValueError(self.error)
        return self.result


def install_launch_audit(output, *, _autotuner_type=None):
    """Install before load_model; close after compile_or_warm_up_model.

    Keeps native _cache_eligible/_cached_launcher and all configs unchanged.
    Every new CachingAutotuner's first launch traverses _pre_launch before the
    native fast path is installed. Stores only unique file/config observations
    during startup, then removes the hook completely before formal measurements.
    The output must be run/inductor-startup-<pid>.json for a seeded run.
    """
    output = Path(output).resolve()
    if not re.fullmatch(r'inductor-startup-[0-9]+\.json', output.name):
        raise ValueError('Startup output must be named inductor-startup-<pid>.json')
    candidates = [parent for parent in output.parents if (parent / RECEIPT).is_file()]
    if len(candidates) != 1:
        raise ValueError('Startup output needs exactly one enclosing seeded arm')
    arm = candidates[0]
    manifest = validate_seed(arm)
    receipt = _read(arm / RECEIPT)
    if manifest is None or receipt.get('seed_manifest_sha256') != _sha((arm / DIRECTORY / 'manifest.json').read_bytes()):
        raise ValueError('Startup observer seed receipt mismatch')
    run = _inside(arm, arm / receipt['target_run'])
    if output.parent != run:
        raise ValueError('Startup observer output must be directly within target_run')
    if _autotuner_type is None:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        _autotuner_type = CachingAutotuner
    if getattr(_autotuner_type, '_kvmem_inductor_launch_audit', False):
        raise ValueError('Inductor startup observer already installed')
    _write_json(output, dict(schema=1, success=False, error='Startup observer has not closed',
        observer_restored=False, records=[]))
    return _LaunchAuditHandle(_autotuner_type, output, arm, manifest, receipt)

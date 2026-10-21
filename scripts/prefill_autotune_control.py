"""Freeze and audit per-experiment FlashInfer tactics without global-cache writes.

Only stdlib is imported. The arm manifest authenticates the seed manifest,
which authenticates the unchanged source bytes. Installation uses an isolated
cache root; audit follows the server's actual cache-path log, not a guessed
configuration hash. Environment metadata must match; cache generation IDs are
reported separately from tactic semantics because normal saves can change them.
"""
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import re

ENVIRONMENT_KEY = 'VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR'
CHAIN_KEY = 'flashinfer_seed_manifest_sha256'
LOG_MARKER = 'Using FlashInfer autotune cache file:'
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError('Duplicate key in autotune JSON: ' + key)
        result[key] = value
    return result


def _read(path):
    def invalid(value):
        raise ValueError('Non-finite autotune JSON value: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_pairs, parse_constant=invalid)


def _hashes(values):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError('Allowed configuration hashes must be a nonempty sequence')
    if any(type(value) is not str or not _HASH.fullmatch(value) for value in values):
        raise ValueError('Allowed configuration hashes must be lowercase 64-hex strings')
    if len(set(values)) != len(values):
        raise ValueError('Duplicate allowed configuration hash')
    return sorted(values)


def _inside(root, path):
    root, path = Path(root).resolve(), Path(path)
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Autotune path escapes experiment arm: ' + str(path))
    return resolved


def _parts(data):
    if type(data) is not dict or type(data.get('_metadata')) is not dict:
        raise ValueError('Autotune cache must contain an environment _metadata object')
    unknown = [key for key in data if key.startswith('_') and key not in ('_metadata', '_generation')]
    if unknown:
        raise ValueError('Unknown reserved autotune metadata: ' + ', '.join(unknown))
    configs = {key: value for key, value in data.items() if key not in ('_metadata', '_generation')}
    if not configs:
        raise ValueError('Autotune cache contains no tactics')
    return configs, data['_metadata'], data.get('_generation')


def _write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n')


def freeze_seed(arm, source: Path, hashes: Sequence[str]) -> dict:
    """Call after creating arm/manifest.json; add the seed-manifest hash chain."""
    arm, source = Path(arm).resolve(), Path(source).resolve()
    allowed = _hashes(hashes)
    arm_manifest_path = _inside(arm, arm / 'manifest.json')
    arm_manifest = _read(arm_manifest_path)
    if type(arm_manifest) is not dict or CHAIN_KEY in arm_manifest:
        raise ValueError('Arm manifest is invalid or already has a frozen autotune seed')
    directory = _inside(arm, arm / 'flashinfer_seed')
    if directory.exists():
        raise ValueError('Autotune seed directory already exists')
    payload = source.read_bytes()
    data = json.loads(payload, object_pairs_hook=_pairs)
    configs, metadata, generation = _parts(data)
    # Canonicalization rejects NaN/Infinity even if accepted by json.loads.
    configs_sha = _sha(_canonical(configs))
    _canonical(metadata)
    manifest = dict(schema=1, seed_file='seed.json', seed_sha256=_sha(payload),
        source_path=str(source), source_sha256=_sha(payload), source_bytes=len(payload),
        allowed_hashes=allowed, configs_sha256=configs_sha, config_count=len(configs),
        metadata=metadata, generation=generation)
    directory.mkdir()
    with (directory / 'seed.json').open('xb') as stream:
        stream.write(payload)
    _write_new(directory / 'manifest.json', manifest)
    arm_manifest[CHAIN_KEY] = _sha((directory / 'manifest.json').read_bytes())
    temporary = arm / '.flashinfer-arm-manifest.tmp'
    _write_new(temporary, arm_manifest)
    temporary.replace(arm_manifest_path)
    return manifest


def validate_seed(arm) -> dict | None:
    arm = Path(arm).resolve()
    directory = _inside(arm, arm / 'flashinfer_seed')
    arm_path = _inside(arm, arm / 'manifest.json')
    arm_manifest = _read(arm_path) if arm_path.exists() else {}
    if type(arm_manifest) is not dict:
        raise ValueError('Invalid experiment arm manifest')
    chain = arm_manifest.get(CHAIN_KEY)
    if chain is None and not directory.exists():
        if (arm / 'flashinfer_seed_install.json').exists():
            raise ValueError('Autotune installation exists without a frozen seed')
        return None
    if type(chain) is not str or not _HASH.fullmatch(chain):
        raise ValueError('Missing or invalid autotune seed manifest hash chain')
    manifest_path = _inside(arm, directory / 'manifest.json')
    if not manifest_path.is_file() or _sha(manifest_path.read_bytes()) != chain:
        raise ValueError('Autotune seed manifest hash mismatch or missing file')
    manifest = _read(manifest_path)
    if manifest.get('schema') != 1 or manifest.get('seed_file') != 'seed.json':
        raise ValueError('Invalid frozen autotune manifest schema or seed path')
    if manifest.get('allowed_hashes') != _hashes(manifest.get('allowed_hashes')):
        raise ValueError('Allowed autotune hashes must be sorted')
    seed_path = _inside(arm, directory / 'seed.json')
    if not seed_path.is_file():
        raise ValueError('Frozen autotune seed is missing')
    payload = seed_path.read_bytes()
    if (_sha(payload) != manifest.get('seed_sha256') or
            _sha(payload) != manifest.get('source_sha256') or len(payload) != manifest.get('source_bytes')):
        raise ValueError('Frozen autotune seed byte hash/extent mismatch')
    configs, metadata, generation = _parts(_read(seed_path))
    if (_sha(_canonical(configs)) != manifest.get('configs_sha256') or
            len(configs) != manifest.get('config_count') or
            _canonical(metadata) != _canonical(manifest.get('metadata')) or
            generation != manifest.get('generation')):
        raise ValueError('Frozen autotune seed semantics mismatch')
    return manifest


def install_seed(arm, target_run, env):
    """Install before Popen; mutate only the supplied child environment map."""
    arm = Path(arm).resolve()
    manifest = validate_seed(arm)
    if manifest is None:
        return None
    target_run = _inside(arm, target_run)
    if not target_run.is_dir():
        raise ValueError('Create the target run directory before installing autotune seed')
    cache_root = _inside(arm, target_run / 'flashinfer_cache')
    receipt_path = _inside(arm, arm / 'flashinfer_seed_install.json')
    if cache_root.exists() or receipt_path.exists():
        raise ValueError('Autotune cache installation already exists')
    payload = (arm / 'flashinfer_seed/seed.json').read_bytes()
    cache_root.mkdir()
    files = []
    for configuration in manifest['allowed_hashes']:
        directory = cache_root / configuration
        directory.mkdir()
        path = directory / 'autotune_configs.json'
        with path.open('xb') as stream:
            stream.write(payload)
        files.append(str(path.relative_to(arm)))
    receipt = dict(schema=1, target_run=str(target_run.relative_to(arm)),
        cache_root=str(cache_root.relative_to(arm)),
        server_log=str((target_run / 'server.log').relative_to(arm)),
        seed_sha256=manifest['seed_sha256'],
        seed_manifest_sha256=_sha((arm / 'flashinfer_seed/manifest.json').read_bytes()),
        installed_files=files, environment_key=ENVIRONMENT_KEY)
    _write_new(receipt_path, receipt)
    env[ENVIRONMENT_KEY] = str(cache_root)
    return receipt


def audit_seed(arm) -> dict | None:
    arm = Path(arm).resolve()
    manifest = validate_seed(arm)
    if manifest is None:
        return None
    receipt_path = _inside(arm, arm / 'flashinfer_seed_install.json')
    if not receipt_path.is_file():
        raise ValueError('Seeded run has no installation receipt')
    receipt = _read(receipt_path)
    if (receipt.get('schema') != 1 or receipt.get('environment_key') != ENVIRONMENT_KEY or
            receipt.get('seed_sha256') != manifest['seed_sha256'] or
            receipt.get('seed_manifest_sha256') != _sha((arm / 'flashinfer_seed/manifest.json').read_bytes())):
        raise ValueError('Autotune installation receipt does not match the frozen seed')
    run = _inside(arm, arm / receipt['target_run'])
    cache_root = _inside(arm, arm / receipt['cache_root'])
    log_path = _inside(arm, arm / receipt['server_log'])
    if cache_root != run / 'flashinfer_cache' or log_path != run / 'server.log':
        raise ValueError('Autotune receipt has inconsistent run paths')
    if not log_path.is_file():
        raise ValueError('Seeded run server log is missing')
    logged = []
    for line in log_path.read_text(errors='replace').splitlines():
        line = _ANSI.sub('', line)
        if LOG_MARKER in line:
            raw = line.split(LOG_MARKER, 1)[1].strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('\"', "'"):
                raw = raw[1:-1]
            path = Path(raw)
            if not path.is_absolute():
                raise ValueError('Logged autotune cache path is not absolute')
            if '..' in path.parts:
                raise ValueError('Logged autotune cache path contains traversal')
            path = _inside(cache_root, path)
            relative = path.relative_to(cache_root)
            if (len(relative.parts) != 2 or relative.name != 'autotune_configs.json' or
                    relative.parts[0] not in manifest['allowed_hashes']):
                raise ValueError('Server used an unapproved autotune configuration hash/path')
            if path not in logged:
                logged.append(path)
    if not logged:
        raise ValueError('Server log has no actual FlashInfer autotune cache path')
    seed_configs, seed_metadata, _ = _parts(_read(arm / 'flashinfer_seed/seed.json'))
    files = []
    for path in logged:
        if not path.is_file():
            raise ValueError('Actually loaded autotune cache is missing')
        configs, metadata, generation = _parts(_read(path))
        if _canonical(metadata) != _canonical(seed_metadata):
            raise ValueError('FlashInfer environment metadata differs from the frozen seed')
        missing = sorted(set(seed_configs) - set(configs))
        changed = sorted(key for key in seed_configs if key in configs and
                         _canonical(seed_configs[key]) != _canonical(configs[key]))
        if missing or changed:
            raise ValueError('Frozen autotune tactics changed: ' + json.dumps(dict(missing=missing, changed=changed)))
        added = {key: configs[key] for key in sorted(set(configs) - set(seed_configs))}
        files.append(dict(path=str(path), configuration_hash=path.parent.name,
            file_sha256=_sha(path.read_bytes()), configs_sha256=_sha(_canonical(configs)),
            configs=configs, config_count=len(configs), seed_tactics_unchanged=True,
            added_configs=added, added_count=len(added), metadata=metadata,
            metadata_matches_seed=True, generation=generation))
    same = len({item['configs_sha256'] for item in files}) == 1
    return dict(seed_sha256=manifest['seed_sha256'], seed_config_count=manifest['config_count'],
        seed_configs_sha256=manifest['configs_sha256'], allowed_hashes=manifest['allowed_hashes'],
        log_path=str(log_path), log_sha256=_sha(log_path.read_bytes()),
        installation_sha256=_sha(receipt_path.read_bytes()), files=files,
        seed_tactics_unchanged=True, metadata_matches_seed=True,
        loaded_files_semantically_identical=same,
        actual_configs_sha256=files[0]['configs_sha256'] if same else None,
        actual_configs=files[0]['configs'] if same else None,
        scope='Final loaded-file audit; matching seed tactics are required, new keys explicitly reported. Generation IDs are excluded from tactic semantics.')

"""Every KVMEM_* read in the package goes through vllm_kvmem.settings (layering step 6)."""
import ast
from pathlib import Path

import pytest

import vllm_kvmem
from vllm_kvmem import settings

PACKAGE = Path(vllm_kvmem.__file__).parent
# settings.py is the accessor; defaults.py writes the default-on preset into the environment.
EXEMPT = {'settings.py', 'defaults.py'}


def _kvmem_literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.startswith('KVMEM_')
    if isinstance(node, ast.BinOp):
        return _kvmem_literal(node.left)
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
        return str(node.values[0].value).startswith('KVMEM_')
    return False


def direct_reads(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and _kvmem_literal(node.args[0]):
            name = ast.unparse(node.func)
            if name.endswith(('getenv', 'environ.get')):
                yield node.lineno, name
        elif (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and _kvmem_literal(node.slice)
              and ast.unparse(node.value).endswith('environ')):
            yield node.lineno, ast.unparse(node.value) + '[...]'


def test_no_module_reads_kvmem_settings_from_the_environment_directly():
    found = []
    for path in sorted(PACKAGE.rglob('*.py')):
        if path.name in EXEMPT:
            continue
        found += ['%s:%d %s' % (path.relative_to(PACKAGE), line, name)
                  for line, name in direct_reads(ast.parse(path.read_text()))]
    assert found == []


def test_the_accessor_reads_the_environment_at_call_time(monkeypatch):
    monkeypatch.delenv('KVMEM_SETTINGS_PROBE', raising=False)
    assert settings.get('KVMEM_SETTINGS_PROBE') is None
    assert settings.get('KVMEM_SETTINGS_PROBE', '0') == '0'
    with pytest.raises(KeyError):
        settings.require('KVMEM_SETTINGS_PROBE')
    monkeypatch.setenv('KVMEM_SETTINGS_PROBE', '1')
    assert settings.get('KVMEM_SETTINGS_PROBE', '0') == '1' == settings.require('KVMEM_SETTINGS_PROBE')


def test_every_known_setting_has_a_namespace():
    kinds = {settings.namespace(name) for name in settings.known()}
    assert kinds <= {'launcher', 'strategy', 'system', 'unverified'}
    assert settings.namespace('KVMEM_GIL_GATE_MS') == 'system'
    assert settings.namespace('KVMEM_LAYERWISE_DECODE_SCOPE') == 'strategy'
    assert settings.namespace('KVMEM_LAYERWISE_QUERY_MODE') == 'launcher'
    assert settings.namespace('KVMEM_PUBLICATION_TRACE') == 'system'
    assert 'KVMEM_GIL_GATE_MS' in settings.known() and 'KVMEM_RECALL_METHOD' in settings.known()


def test_profile_settings_become_the_launch_environment():
    assert settings.profile_environment(None) == {}
    env = settings.profile_environment(dict(preset='legacy', system={'KVMEM_GIL_GATE_MS': 50},
                                            strategy={'KVMEM_LAYERWISE_DECODE_SCOPE': 'all',
                                                      'KVMEM_HELD_P_CAPTURE': True}))
    assert env == {'KVMEM_LEGACY_DEFAULTS': '1', 'KVMEM_GIL_GATE_MS': '50',
                   'KVMEM_LAYERWISE_DECODE_SCOPE': 'all', 'KVMEM_HELD_P_CAPTURE': '1'}
    for spec, message in ((dict(system={'KVMEM_NOT_A_SETTING': '1'}), 'unknown setting'),
                          (dict(system={'KVMEM_LAYERWISE_QUERY_MODE': 'sequential'}), 'derived'),
                          (dict(system={'KVMEM_LAYERWISE_DECODE_SCOPE': 'all'}), 'may change model output'),
                          (dict(system={'KVMEM_HELD_P_CAPTURE': '1'}), 'may change model output'),
                          (dict(strategy={'KVMEM_GIL_GATE_MS': '50'}), 'never changes model output'),
                          (dict(preset='fast'), 'preset'), (dict(other={}), 'preset, system and strategy'),
                          (dict(system={'KVMEM_GIL_GATE_MS': [1]}), 'string, number or boolean')):
        with pytest.raises(ValueError, match=message):
            settings.profile_environment(spec)


def test_effective_settings_name_their_source():
    from vllm_kvmem.defaults import DEFAULTS
    effective = settings.effective({'KVMEM_GIL_GATE_MS': '0'})
    assert effective['KVMEM_GIL_GATE_MS'] == ('0', 'launch')
    other = next(name for name in DEFAULTS if name != 'KVMEM_GIL_GATE_MS')
    assert effective[other] == (DEFAULTS[other], 'preset')
    assert other not in settings.effective({'KVMEM_LEGACY_DEFAULTS': '1'})
    assert settings.main(['table']) == 0

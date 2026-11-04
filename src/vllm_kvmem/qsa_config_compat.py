"""Recognize KVMem's in-request control channel in the vendor QSA guard.

The native guard assumes every KV connector imports external prefixes. This
adapter has no external hits, request-boundary exports, or session restoration.
Do not advertise the unrelated atomic external-checkpoint capability merely to
pass that guard. All other connectors and unsupported configurations retain it.
"""
import importlib.abc
import importlib.machinery
import sys

TARGET = 'vllm.config.vllm'
OLD = 'if not aligned and not connector.supports_request_boundary_checkpoints('
NEW = ('if not aligned and not _kvmem_qsa_control_only(self, connector) '
       'and not connector.supports_request_boundary_checkpoints(')


def control_only(config, connector):
    if connector.__module__ != 'vllm_kvmem.connector' or connector.__name__ != 'KVMemConnector':
        return False
    transfer = config.kv_transfer_config
    if (transfer.kv_connector_module_path != 'vllm_kvmem.connector' or
            config.scheduler_config.scheduler_cls != 'vllm_kvmem.scheduler.KVMemScheduler'):
        return False
    from .options import RuntimeOptions
    from .qsa_adapter import validate_qsa_config
    return validate_qsa_config(config, RuntimeOptions.from_vllm(config))


def patch_source(source):
    text = source.decode()
    if text.count(OLD) != 1:
        raise ImportError('Unrecognized vendor QSA connector guard; refusing adaptation')
    return (text.replace(OLD, NEW) +
        '\nfrom vllm_kvmem.qsa_config_compat import control_only as _kvmem_qsa_control_only\n')


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        return compile(patch_source(self.get_data(self.path)), self.path, 'exec')


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            raise ImportError('Cannot locate vendor QSA configuration')
        spec.loader = _Loader(fullname, spec.origin)
        return spec


def enable():
    if TARGET in sys.modules:
        # Spawned workers may import the dataclass while unpickling config,
        # before plugin registration. MTP later reconstructs that config.
        # Replace only its checked guard method; do not reload the dataclass.
        import ast
        import textwrap
        from pathlib import Path
        from .engine_api import find_symbol
        module = sys.modules[TARGET]
        owner = module.VllmConfig
        method = owner._verify_kv_transfer_compat
        if '_kvmem_qsa_control_only' in method.__code__.co_names:
            return
        source = Path(module.__file__).read_text()
        node = find_symbol(source, 'VllmConfig._verify_kv_transfer_compat')
        body = textwrap.dedent(ast.get_source_segment(source, node))
        namespace = dict(module.__dict__)
        exec(compile(patch_source(body.encode()), module.__file__, 'exec'), namespace)
        owner._verify_kv_transfer_compat = namespace['_verify_kv_transfer_compat']
        return
    if any(isinstance(finder, _Finder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _Finder())

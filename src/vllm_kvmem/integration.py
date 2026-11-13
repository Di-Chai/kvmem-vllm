"""Keep plugin discovery separate from opting into engine changes."""
from importlib.metadata import version
import os
from . import settings as _settings

_registered = False


def register_plugin() -> None:
    global _registered
    if _registered:
        return
    if _settings.get('KVMEM_QSA_ADAPTER') == '1':
        from .qsa_config_compat import enable
        enable()
    if _settings.get('KVMEM_SERVING') == '1':
        import sys
        from .defaults import summary
        print(summary(), file=sys.stderr, flush=True)
        from .compat import enable_label_logprobs
        enable_label_logprobs()
        if _settings.get('KVMEM_LOGICAL_KV_CAPACITY') == '1':
            from .logical_capacity import enable
            enable()
    if _settings.get("KVMEM_DRAFT_LOAD_COMPAT") == "1":
        from .compat import enable_draft_load_config_fix
        enable_draft_load_config_fix()
    if _settings.get("KVMEM_RUNNER_PROBE_DIR"):
        from .compat import enable_runner_probe
        enable_runner_probe()
    if _settings.get("KVMEM_WINDOW_METADATA") == "1":
        from .compat import enable_window_metadata
        enable_window_metadata()
    if _settings.get("KVMEM_Q8_TENSORCORE") == "1":
        from .q8_adapter import enable_q8_tensorcore
        enable_q8_tensorcore()
    # Discovery alone changes nothing and initializes no CUDA. Explicit options
    # above install capability-checked import adapters in this worker process.
    _registered = True


def check_engine_version() -> str:
    """Retain the public diagnostic entry, without an exact-version gate."""
    from .engine_api import require_contract
    require_contract('runner')
    return version("vllm")

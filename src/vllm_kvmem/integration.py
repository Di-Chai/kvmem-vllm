"""Keep plugin discovery separate from opting into engine changes."""
from importlib.metadata import version
import os

_registered = False


def register_plugin() -> None:
    global _registered
    if _registered:
        return
    if os.getenv("KVMEM_DRAFT_LOAD_COMPAT") == "1":
        from .compat import enable_draft_load_config_fix
        enable_draft_load_config_fix()
    if os.getenv("KVMEM_RUNNER_PROBE_DIR"):
        from .compat import enable_runner_probe
        enable_runner_probe()
    if os.getenv("KVMEM_WINDOW_METADATA") == "1":
        from .compat import enable_window_metadata
        enable_window_metadata()
    # Discovery alone changes nothing and initializes no CUDA. Explicit options
    # above install version-checked import adapters in this worker process.
    _registered = True


def check_engine_version() -> str:
    actual = version("vllm")
    if actual != "0.30.0":
        raise RuntimeError(f"Unqualified vLLM version {actual}; expected 0.30.0")
    return actual

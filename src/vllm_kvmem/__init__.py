"""KVMem host core and versioned vLLM integration.

Importing this package never imports torch or initializes CUDA.
"""

__version__ = "0.1.0.dev1"

# Measured-effective optimizations are on by default (see defaults.py). This
# runs before any submodule reads a KVMEM_* flag; explicit values still win and
# KVMEM_LEGACY_DEFAULTS=1 restores the old opt-in behaviour.
from . import defaults as _defaults

_defaults.apply()


def register() -> None:
    """vLLM entry point; registration alone does not enable KVMem serving."""
    from .integration import register_plugin
    register_plugin()

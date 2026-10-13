"""KVMem host core and versioned vLLM integration.

Importing this package never imports torch or initializes CUDA.
"""

__version__ = "0.1.0.dev1"


def register() -> None:
    """vLLM entry point; registration alone does not enable KVMem serving."""
    from .integration import register_plugin
    register_plugin()

"""Moved to vllm_kvmem.core.native_memory_storage (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.core.native_memory_storage', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.core import native_memory_storage as _module
    sys.modules[__name__] = _module

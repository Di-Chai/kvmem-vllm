"""Moved to vllm_kvmem.adapters.assembly (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.adapters.assembly', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.adapters import assembly as _module
    sys.modules[__name__] = _module

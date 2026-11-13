"""Moved to vllm_kvmem.adapters.fp8_compat (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.adapters.fp8_compat', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.adapters import fp8_compat as _module
    sys.modules[__name__] = _module

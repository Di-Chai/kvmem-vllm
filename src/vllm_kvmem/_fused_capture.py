"""Moved to vllm_kvmem.adapters._fused_capture (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.adapters._fused_capture', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.adapters import _fused_capture as _module
    sys.modules[__name__] = _module

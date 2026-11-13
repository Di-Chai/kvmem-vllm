"""Moved to vllm_kvmem.core.private_record_pack (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.core.private_record_pack', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.core import private_record_pack as _module
    sys.modules[__name__] = _module

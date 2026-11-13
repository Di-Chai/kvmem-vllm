"""Moved to vllm_kvmem.apps.webui.manager (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.apps.webui.manager', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.apps.webui import manager as _module
    sys.modules[__name__] = _module

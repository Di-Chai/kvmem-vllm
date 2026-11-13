"""Moved to vllm_kvmem.apps.planner.memory (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.apps.planner.memory', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.apps.planner import memory as _module
    sys.modules[__name__] = _module

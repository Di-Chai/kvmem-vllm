"""Moved to vllm_kvmem.serving.deferred_action_frame (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.serving.deferred_action_frame', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.serving import deferred_action_frame as _module
    sys.modules[__name__] = _module

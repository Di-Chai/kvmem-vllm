"""Moved to vllm_kvmem.apps.elastic_stream (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.apps.elastic_stream', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.apps import elastic_stream as _module
    sys.modules[__name__] = _module

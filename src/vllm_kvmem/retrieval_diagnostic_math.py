"""Moved to vllm_kvmem.serving.retrieval_diagnostic_math (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.serving.retrieval_diagnostic_math', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.serving import retrieval_diagnostic_math as _module
    sys.modules[__name__] = _module

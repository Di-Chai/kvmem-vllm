"""Moved to vllm_kvmem.serving.buffered_prefill_attention (layering step 7). This alias is removed after the next release."""
import sys

if __name__ == '__main__':
    import runpy
    runpy.run_module('vllm_kvmem.serving.buffered_prefill_attention', run_name='__main__', alter_sys=True)
else:
    from vllm_kvmem.serving import buffered_prefill_attention as _module
    sys.modules[__name__] = _module

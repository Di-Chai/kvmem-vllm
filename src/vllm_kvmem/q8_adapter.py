"""Opt-in, source-pinned experimental GGUF linear dispatch adapter."""
import hashlib
import logging
from pathlib import Path


def enable_q8_tensorcore():
    from .integration import check_engine_version
    check_engine_version()
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization import linear

    if getattr(ops.ggml_mul_mat_a8, '_kvmem_q8_tensorcore', False):
        return
    pins = ((ops, 'daac821e871ff73df30b1836c17a7ef9574f8be6e4bedd5f6456c442b69c90db'),
            (linear, 'b40556bc026632e610adaee2855d42ddc3d00d99ba07fdfee7cae171ec180efe'))
    for module, expected in pins:
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unqualified GGUF source for experimental Q8 Tensor Core adapter')
    if not ops._CUDA_ENABLED:
        raise RuntimeError('Q8 Tensor Core adapter requires the qualified CUDA GGUF backend')
    import torch
    from .q8_tensorcore import q8_tensorcore_linear
    original = ops.ggml_mul_mat_a8
    announced = False

    def dispatch(weight, x, quant_type, row):
        nonlocal announced
        if (quant_type == 8 and x.ndim == 2 and x.shape[0] >= 16
                and x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous()
                and weight.device == x.device and weight.dtype == torch.uint8
                and weight.ndim == 2 and weight.is_contiguous()
                and weight.shape[0] == row and x.shape[1] % 32 == 0
                and weight.shape[1] * 32 == x.shape[1] * 34
                and torch.cuda.get_device_capability(x.device) == (12, 0)):
            if not announced:
                logging.getLogger(__name__).warning(
                    'Experimental KVMem Q8 Tensor Core dispatch active: original Q8_0 bytes, '
                    'native fast-math Q8_1 quantization, rows >= 16; model quality not qualified')
                announced = True
            return q8_tensorcore_linear(x, weight)
        return original(weight, x, quant_type, row)

    dispatch._kvmem_q8_tensorcore = True
    ops.ggml_mul_mat_a8 = dispatch

"""Pure-stdlib cache identity for the opt-in opaque capture ABI.

Imported by the launcher before torch/vLLM or CUDA discovery. Connector options
are not part of vLLM's compile hash; this identity must be in additional_config.
"""
import hashlib
from pathlib import Path


CAPTURE_COMPILE_KEY = 'kvmem_capture_opaque_source'
CAPTURE_OPAQUE_ABI = 'kvmem.capture_projection.v1.ordered'
_SOURCES = ('capture.py', '_fused_capture.py', 'capture_opaque.py', 'capture_compile.py')


def capture_compile_fingerprint() -> str:
    digest = hashlib.sha256(CAPTURE_OPAQUE_ABI.encode())
    directory = Path(__file__).parent
    for name in _SOURCES:
        data = (directory / name).read_bytes()
        digest.update(name.encode() + b'\0')
        digest.update(len(data).to_bytes(8, 'little'))
        digest.update(data)
    return digest.hexdigest()


def validate_capture_compile_config(config, enabled: bool) -> str | None:
    """Reject stale/missing opt-in cache identity before model/AOT loading."""
    additional = getattr(config, 'additional_config', None)
    if not enabled:
        if isinstance(additional, dict) and CAPTURE_COMPILE_KEY in additional:
            raise ValueError('Disabled opaque capture projection cannot retain its source fingerprint '
                             'in vLLM additional_config')
        return None
    expected = capture_compile_fingerprint()
    if not isinstance(additional, dict) or additional.get(CAPTURE_COMPILE_KEY) != expected:
        raise ValueError('Opaque capture projection requires matching source fingerprint '
                         'in vLLM additional_config before model loading')
    return expected

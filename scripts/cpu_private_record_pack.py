"""Component benchmark compatibility alias for the qualified implementation."""
import sys
from vllm_kvmem.core import private_record_pack as implementation

sys.modules[__name__] = implementation

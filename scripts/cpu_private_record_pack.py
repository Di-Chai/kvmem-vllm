"""Component benchmark compatibility alias for the qualified implementation."""
import sys
from vllm_kvmem import private_record_pack as implementation

sys.modules[__name__] = implementation

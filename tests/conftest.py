"""Unit tests exercise individual KVMEM_* flags explicitly.

They predate the default-on optimization stack (vllm_kvmem/defaults.py), so the
suite keeps the old opt-in defaults; tests of the stack itself pass their own
environment mapping to defaults.apply().
"""
import os

os.environ.setdefault('KVMEM_LEGACY_DEFAULTS', '1')

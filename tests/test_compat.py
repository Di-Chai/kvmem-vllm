from importlib import metadata
from pathlib import Path
import ast

import pytest

from vllm_kvmem.compat import OLD, NEW, patched_source


def test_pinned_loader_patch_changes_only_load_config():
    try:
        root = Path(metadata.distribution("vllm").locate_file("vllm"))
    except metadata.PackageNotFoundError:
        pytest.skip("vLLM optional dependency is not installed")
    source = (root / "v1/worker/gpu/spec_decode/dflash/utils.py").read_bytes()
    patched = patched_source(source)
    ast.parse(patched)
    assert patched.replace(NEW, OLD) == source.decode()
    with pytest.raises(ImportError, match="Unrecognized"):
        patched_source(source + b"\n# changed")

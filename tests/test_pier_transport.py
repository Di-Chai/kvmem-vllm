import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('pier_transport',
    Path(__file__).parents[1] / 'benchmarks/deepswe/pier_transport_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def test_proxy_timeout_alignment_keeps_every_original_policy_line():
    original = 'http_access deny all\ncache deny all\nshutdown_lifetime 1 seconds\nEOF\n'
    aligned = transport.align_proxy_timeout(original)
    assert aligned.replace('read_timeout 21600 seconds\nforward_timeout 21600 seconds\n', '') == original
    with pytest.raises(RuntimeError, match='Unqualified'):
        transport.align_proxy_timeout(aligned)

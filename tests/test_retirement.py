import json
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('vllm')
from vllm_kvmem.capture_runtime import CaptureRuntime
from vllm_kvmem.options import RuntimeOptions
from test_authority import setup, payload


@pytest.mark.parametrize('verify', [False, True])
def test_retirement_drains_real_archive_and_labels_optional_readback(tmp_path, monkeypatch, verify):
    authority, archive = setup(tmp_path / 'archive')
    assert authority.append(*payload(0, 5), history_begin=0, local_rope_begin=0)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options = RuntimeOptions(memory=authority.memory.config, verify_archive_on_finish=verify)
    runtime.config = runtime.options.memory
    runtime.runner = NS(vllm_config=NS(max_concurrent_batches=1), decode_query_len=1)
    runtime.output = tmp_path
    runtime.archive = archive
    runtime.downloads = NS(pending={})
    runtime.queue = []
    runtime.authorities = {'a': authority}
    runtime.accepted, runtime.rejected, runtime.last_steps = {'a': 5}, {'a': 0}, {'a': []}
    runtime.completed, runtime.batches = [], 1
    original = authority.read_record
    reads = []

    def read(handle):
        reads.append(handle)
        return original(handle)

    monkeypatch.setattr(authority, 'read_record', read)
    runtime.finish({'a'})
    report = json.loads((tmp_path / 'archive.json').read_text())
    assert report['archive_verification'] == ('full_checksum' if verify else 'publication_only')
    assert report['completed'][0]['sealed_records_archived'] == 1
    assert report['completed'][0]['sealed_records_verified'] == int(verify)
    assert len(reads) == int(verify)
    assert not runtime.authorities and archive.tier.used_slots == 0
    assert not runtime.accepted and not runtime.rejected
    assert archive.close()

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import append_authority_batch
from test_authority import setup, payload, drain


def entry(authority, begin, candidate, accepted):
    raw, values = payload(begin, candidate)
    return authority, raw, values, dict(history_begin=authority.history_end,
        local_rope_begin=authority.history_end, accepted=accepted)


def test_batched_accepted_tails_and_sealed_records_match_reference_exactly(tmp_path):
    owners, references, archives = [], [], []
    for i in range(3):
        owner, archive = setup(tmp_path / str(i), name=str(i))
        reference, refarchive = setup(tmp_path / ('ref' + str(i)), name=str(i))
        owners.append(owner); references.append(reference)
        archives.extend((archive, refarchive))
    try:
        for counts in ((3, 2, 0), (3, 4, 4), (1, 1, 3)):
            batch = []
            for owner, reference, accepted in zip(owners, references, counts):
                raw, values = payload(owner.history_end + 10, accepted + 2)
                kwargs = dict(history_begin=owner.history_end, local_rope_begin=owner.history_end,
                              accepted=accepted)
                batch.append((owner, raw, values, kwargs))
                assert reference.append(raw, values, **kwargs)
            assert append_authority_batch(batch) == 3
            for owner, reference in zip(owners, references):
                drain(owner); drain(reference)
                assert owner.history_end == reference.history_end
                assert len(owner.records) == len(reference.records)
                for actual, expected in zip(owner.records.values(), reference.records.values()):
                    assert actual.count == expected.count
                    assert torch.equal(actual.key_sum, expected.key_sum)
                    for a, b in zip(owner.read_record(actual.handle), reference.read_record(expected.handle)):
                        assert torch.equal(a, b)
    finally:
        for owner in owners + references:
            assert owner.close()
        for archive in archives:
            assert archive.close()


def test_backpressure_commits_only_admitted_prefix_and_retry_does_not_duplicate(tmp_path, monkeypatch):
    a, archive = setup(tmp_path, 'a', resident=1)
    b, _ = setup(tmp_path, 'b', archive, resident=1)
    write = archive.write
    monkeypatch.setattr(archive, 'write', lambda *args, **kwargs: None)
    try:
        assert b.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
        batch = [entry(a, 0, 3, 2), entry(b, 4, 2, 1)]
        assert append_authority_batch(batch) == 1
        assert a.history_end == 2 and b.history_end == 4
        assert a.memory.phase.name == 'READY' and b.memory.phase.name == 'READY'
        monkeypatch.setattr(archive, 'write', write)
        drain(b)
        assert append_authority_batch(batch[1:]) == 1
        assert a.history_end == 2 and b.history_end == 5
    finally:
        monkeypatch.setattr(archive, 'write', write)
        assert a.close() and b.close() and archive.close()


def test_invalid_later_native_descriptor_cannot_mutate_an_earlier_tail():
    raw = torch.randn(2, 4, 1, 2).bfloat16().view(torch.uint8).numpy()
    values = torch.zeros(2, 4, 1, 2, dtype=torch.uint8).numpy()
    destination = torch.full((2, 4, 1, 2), 9, dtype=torch.bfloat16)
    output_values = torch.ones(2, 4, 1, 2, dtype=torch.uint8)
    sums = torch.zeros(2, 1, 2)
    delta = torch.ones_like(sums)
    job = (raw, values, destination.view(torch.uint8).numpy(), output_values.numpy(),
           delta.numpy(), sums.numpy(), 0, 0, 1)
    before = destination.clone()
    with pytest.raises(ValueError):
        _native.scatter_history_batch([job, (*job[:8], 5)])
    assert torch.equal(destination, before)
    assert output_values.eq(1).all() and sums.eq(0).all()


def test_duplicate_authority_owner_refused_before_any_reservation(tmp_path):
    a, archive = setup(tmp_path)
    try:
        row = entry(a, 0, 2, 1)
        with pytest.raises(ValueError, match='distinct'):
            append_authority_batch([row, row])
        assert a.history_end == 0 and a.memory.phase.name == 'READY'
    finally:
        assert a.close() and archive.close()


@pytest.mark.parametrize('count', [1, 2, 3, 4])
def test_fused_real_geometry_bf16_mean_matches_torch_bitwise_with_cancellation(count):
    # Target 16 layers plus one unscored MTP layer; wide exponents expose
    # incorrect reduction association that ordinary random values can hide.
    generator = torch.Generator().manual_seed(9047 + count)
    for _ in range(6):
        raw = (torch.randn(17, 7, 4, 256, generator=generator) *
               torch.exp2(torch.randint(-100, 100, (17, 7, 4, 256), generator=generator).float())).bfloat16()
        values = torch.randint(0, 256, raw.shape, dtype=torch.uint8, generator=generator)
        target_raw = torch.zeros(17, 128, 4, 256, dtype=torch.bfloat16)
        target_values = torch.zeros(17, 128, 4, 256, dtype=torch.uint8)
        sums = torch.randn(16, 4, 256, generator=generator)
        expected = sums + raw[:16, 1:1+count].float().sum(1)
        _native.scatter_history_batch([(raw.view(torch.uint8).numpy(), values.numpy(),
            target_raw.view(torch.uint8).numpy(), target_values.numpy(), None, sums.numpy(),
            1, 17, count)])
        assert torch.equal(sums, expected)
        assert torch.equal(target_raw[:, 17:17+count], raw[:, 1:1+count])
        assert torch.equal(target_values[:, 17:17+count], values[:, 1:1+count])


def test_cached_destination_exports_release_when_record_is_sealed(tmp_path):
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import AsyncArchive
    config = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=8,
        repository_blocks=8, sink_blocks=1, recent_blocks=0)
    archive = AsyncArchive(str(tmp_path), 2 * 4 * 1 * 64 * 3, 16)
    owner = RequestAuthority('cache', config, archive, 2, 1, 64, scoring_layers=1)
    raw = torch.randn(2, 4, 1, 64).bfloat16()
    values = torch.randint(0, 256, raw.shape, dtype=torch.uint8)
    try:
        assert append_authority_batch([(owner, raw, values,
            dict(history_begin=0, local_rope_begin=0, accepted=2))]) == 1
        record = next(iter(owner.records.values()))
        assert record.scatter_views is not None
        assert append_authority_batch([(owner, raw[:, 2:], values[:, 2:],
            dict(history_begin=2, local_rope_begin=2, accepted=2))]) == 1
        assert record.submitted and record.scatter_views is None
        assert record.raw is None and record.values is None
        assert torch.equal(record.key_sum, raw[:1].float().sum(1))
        drain(owner)
        assert torch.equal(owner.read_record(record.handle)[0], raw)
    finally:
        assert owner.close() and archive.close()

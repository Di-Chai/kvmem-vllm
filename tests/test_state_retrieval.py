import pytest
torch = pytest.importorskip("torch")

from vllm_kvmem.retrieval import mean_key_scores
from vllm_kvmem.state import StateCheckpoints


def test_checkpoint_isolation_schema_and_branch_invalidation():
    pool = {"conv": torch.arange(32).reshape(2, 16).float(),
            "recurrent": torch.arange(48).reshape(2, 4, 6).float()}
    a = {name: t[0] for name, t in pool.items()}
    b = {name: t[1] for name, t in pool.items()}
    store = StateCheckpoints("a", 4096)
    checkpoint = store.capture(1000000, a)
    original_b = {name: t.clone() for name, t in b.items()}
    for tensor in a.values():
        tensor.zero_()
    with pytest.raises(ValueError, match="Incomplete"):
        store.restore(checkpoint, {"conv": a["conv"]})
    assert not a["conv"].any()
    store.restore(checkpoint, a)
    assert torch.equal(a["conv"], checkpoint.tensors["conv"])
    assert all(torch.equal(b[name], t) for name, t in original_b.items())
    with pytest.raises(ValueError, match="another"):
        StateCheckpoints("b", 4096).restore(checkpoint, b)
    assert store.new_branch()
    with pytest.raises(ValueError, match="expired"):
        store.restore(checkpoint, a)


def test_checkpoint_capacity_never_silently_evicts_live_state():
    store = StateCheckpoints("a", 16, capacity=1)
    state = {"ssm": torch.ones(4)}
    c = store.capture(0, state)
    with pytest.raises(BufferError):
        store.capture(1, state)
    assert store.checkpoints[0] is c
    assert store.allocated_bytes == 16


def test_meank_matches_explicit_gqa_softmax_and_mass():
    gen = torch.Generator().manual_seed(471)
    q = torch.randn(2, 5, 4, 8, generator=gen)
    k = torch.randn(2, 7, 2, 8, generator=gen)
    mask = torch.tensor([False, True, True, False, True, True, False])
    actual = mean_key_scores(q, k, mask, query_chunk=2)
    expected = torch.zeros(7)
    for layer in range(2):
        for token in range(5):
            for head in range(4):
                logits = (k[layer, :, head//2] @ q[layer, token, head]) / 8**0.5
                expected += logits.masked_fill(~mask, -torch.inf).softmax(0) / 8
    torch.testing.assert_close(actual, expected)
    assert actual.sum().item() == pytest.approx(5)
    assert not actual[~mask].any()
    assert not mean_key_scores(q, k, torch.zeros_like(mask)).any()

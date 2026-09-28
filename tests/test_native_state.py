from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from vllm_kvmem.native_state import NativeRecurrentState
from vllm_kvmem.state import StateCheckpoints


def adapter():
    conv = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    recurrent = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    table = torch.tensor([[1, 2], [3, 4]])
    runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={"a": 0, "b": 1}),
        _kvmem_request_generations={"a": 1, "b": 2},
        model_state=SimpleNamespace(num_accepted_tokens_gpu=torch.tensor([1, 2])),
        block_tables=SimpleNamespace(num_blocks=SimpleNamespace(np=torch.tensor([[2, 2]]).numpy()),
                                      block_tables=[SimpleNamespace(gpu=table)]))
    owner = NativeRecurrentState.__new__(NativeRecurrentState)
    owner.runner = runner
    owner.groups = {0: {"gdn": (conv, recurrent)}}
    return owner, conv, recurrent, table


def test_restore_includes_all_speculation_rows_and_preserves_other_request():
    owner, conv, recurrent, _ = adapter()
    bundle = owner.borrow(["a"])
    assert len(bundle.tensors) == 5
    saved_conv, saved_recurrent = conv.clone(), recurrent.clone()
    checkpoints = StateCheckpoints("a", 1024)
    checkpoint = checkpoints.capture(10, bundle.tensors)
    conv[1:3].fill_(99)
    recurrent[1:3].fill_(88)
    owner.runner.model_state.num_accepted_tokens_gpu[0] = 7
    owner.validate(bundle)
    checkpoints.restore(checkpoint, bundle.tensors)
    assert torch.equal(conv, saved_conv)
    assert torch.equal(recurrent, saved_recurrent)
    assert owner.runner.model_state.num_accepted_tokens_gpu.tolist() == [1, 2]


def test_checkpoint_rejects_reused_request_incarnation_even_with_same_pages():
    owner, _, _, _ = adapter()
    bundle = owner.borrow(["a"])
    owner.runner._kvmem_request_generations["a"] = 3
    with pytest.raises(ValueError, match="incarnation"):
        owner.validate(bundle)


def test_checkpoint_rejects_page_changes_and_cross_request_aliases():
    owner, _, _, table = adapter()
    bundle = owner.borrow(["a"])
    table[0, 1] = 5
    with pytest.raises(ValueError, match="page table"):
        owner.validate(bundle)
    table[1, 0] = 1
    with pytest.raises(ValueError, match="share"):
        owner.borrow(["a", "b"])


def test_canonicalization_rejects_lost_history_instead_of_fabricating_state():
    owner, conv, recurrent, _ = adapter()
    original = conv.clone(), recurrent.clone()
    with pytest.raises(ValueError, match='checkpoint replay'):
        owner.canonicalize_accepted('a', retained_tokens=0, query_tokens=2, speculative=True)
    with pytest.raises(ValueError, match='checkpoint replay'):
        owner.canonicalize_accepted('a', retained_tokens=1, query_tokens=2, speculative=False)
    assert torch.equal(conv, original[0]) and torch.equal(recurrent, original[1])


def test_canonicalization_validates_all_native_rows_before_any_write():
    owner, conv, recurrent, _ = adapter()
    original = conv.clone(), recurrent.clone()
    with pytest.raises(ValueError, match='allocated recurrent rows'):
        owner.canonicalize_accepted('a', retained_tokens=2, query_tokens=3, speculative=True)
    assert torch.equal(conv, original[0]) and torch.equal(recurrent, original[1])


def test_recurrent_source_contract_matches_installed_native_implementation():
    pytest.importorskip('vllm')
    from vllm_kvmem.native_state import verify_state_sources
    verify_state_sources()

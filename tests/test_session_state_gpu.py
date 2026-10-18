import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='Explicit GPU qualification only')


def test_portable_prompt_state_copies_exact_cuda_bytes_to_new_native_pages():
    from test_session_state import fixture
    adapter, conv, ssm, cache, _ = fixture(device='cuda')
    snapshot = adapter.capture('a', 8, max_bytes=4096)
    snapshot.completion.synchronize()
    assert snapshot.query()
    before = conv.clone(), ssm.clone(), cache.clone()
    adapter.runner.req_states.req_id_to_index['new'] = 2
    adapter.runner._kvmem_request_generations['new'] = 3
    conv[1:3].fill_(-17); ssm[1:3].fill_(-17); cache[1].fill_(-17)
    completion = adapter.restore(snapshot, 'new')
    assert completion is not None
    completion.synchronize()
    assert torch.equal(conv[5], before[0][1]) and torch.equal(ssm[5], before[1][1])
    assert torch.equal(cache[8], before[2][1])
    assert torch.equal(conv[3:5], before[0][3:5]) and torch.equal(ssm[3:5], before[1][3:5])
    assert torch.equal(cache[2], before[2][2])


@pytest.mark.parametrize('retained', [1, 2])
def test_async_canonical_checkpoint_matches_blocking_reference_before_native_reuse(monkeypatch, retained):
    """Identical bytes after accepted-state normalization, including neighbours."""
    from types import SimpleNamespace
    from test_session_state import fixture
    from vllm_kvmem.recurrent_table_journal import RecurrentTableJournal

    snapshots = []
    for enabled in (False, True):
        adapter, conv, ssm, _, _ = fixture('cuda', draft=False)
        runner = adapter.runner
        runner.req_states.prefill_len.np[0] = 5
        runner._kvmem_capture_runtime = SimpleNamespace(
            last_steps={'a': [dict(history_begin=8-retained, accepted=retained, scheduled=2)]},
            authorities={'a': SimpleNamespace(history_end=8)})
        runner.model_state.num_accepted_tokens_gpu[0] = retained
        adapter.recurrent.conv_dim_first = False
        adapter.recurrent.conv_widths = {'gdn': 4}
        tables = runner.block_tables
        tables.blocks_per_kv_block = [1, 1]
        pending = []
        def append(index, ids, overwrite):
            assert overwrite
            pending.append((index, ids[0]))
            tables.num_blocks.np[0, index] = len(ids[0])
        def apply():
            for index, ids in pending:
                tables.block_tables[0].gpu[index, :len(ids)].copy_(torch.tensor(ids, device='cuda'))
            pending.clear()
        tables.append_block_ids, tables.apply_staged_writes = append, apply
        journal = RecurrentTableJournal(runner, [0])
        runner._kvmem_recurrent_table_journal = journal
        tables.append_block_ids(0, ([1, 2], []), True)
        tables.apply_staged_writes()
        before = conv[3:].clone(), ssm[3:].clone()
        monkeypatch.setenv('KVMEM_ASYNC_CHECKPOINT_CAPTURE', str(int(enabled)))
        snapshot = adapter.capture('a', 8, max_bytes=4096, terminal=True)
        # These writes are queued immediately, before waiting for the CPU copy.
        conv[1:3].fill_(-17); ssm[1:3].fill_(-17)
        snapshot.completion.synchronize()
        assert snapshot.query()
        assert torch.equal(conv[3:], before[0]) and torch.equal(ssm[3:], before[1])
        snapshots.append(snapshot.payload.clone())
    assert torch.equal(*snapshots)

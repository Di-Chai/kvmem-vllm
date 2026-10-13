from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import mtp_runtime
from test_authority import setup, drain
from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.storage import AsyncArchive


def test_mtp_keys_never_change_original_target_meank_selection(tmp_path):
    main, archive = setup(tmp_path/'main')
    other_archive = AsyncArchive(str(tmp_path/'mtp'), 3*4*1*2*3, 16)
    together = RequestAuthority('mtp', main.memory.config, other_archive, 3, 1, 2, resident_records=8,
                                scoring_layers=2)
    raw = torch.cat([torch.full((2, 4, 1, 2), v, dtype=torch.bfloat16)
                     for v in (0, 1, 10, -10)], dim=1)
    adversarial = torch.cat([torch.full((1, 4, 1, 2), v, dtype=torch.bfloat16)
                            for v in (0, 1000, -1000, 1000)], dim=1)
    assert main.append(raw, raw.to(torch.uint8), history_begin=0, local_rope_begin=0)
    all_raw = torch.cat((raw, adversarial))
    assert together.append(all_raw, all_raw.to(torch.uint8), history_begin=0, local_rope_begin=0)
    drain(main); drain(together)
    assert next(iter(together.records.values())).key_sum.shape == (2, 1, 2)
    queries = torch.ones(2, 1, 2, 2)
    plans = [a.select_for_queries(queries) for a in (main, together)]
    assert [[r.block_id for r in p.remaps] for p in plans] == [[0, 2], [0, 2]]
    for authority in (main, together):
        from vllm_kvmem.authority import _Ready
        authority.memory.attach_completion(_Ready()); assert authority.memory.commit_selection()
    del plans
    assert main.close() and together.close() and archive.close() and other_archive.close()


@dataclass
class Batch:
    positions: object
    seq_lens: object
    seq_lens_cpu_upper_bound: object


def test_mtp_prefix_is_published_after_proposal_in_dense_frame_and_failure_resets_mode(monkeypatch):
    monkeypatch.setattr('vllm_kvmem.draft_contract.verify_mtp_sources', lambda: None)
    module = NS()
    record = NS(name='layer', key=torch.zeros(4), value=torch.zeros(4))
    model = NS(named_modules=lambda: [('layer', module)])
    monkeypatch.setattr(mtp_runtime, 'install_qwen_capture', lambda _: ['layer'])
    monkeypatch.setattr(mtp_runtime, 'install_persistent_capture', lambda *a: [record])
    published, native_batches = [], []

    class Proposer:
        input_buffers = NS(positions='draft-positions')
        def _prefill(self):
            assert module._kvmem_mtp_prefix_active
            record.key.fill_(7)
        def propose(self, batch, *, fail=False):
            native_batches.append(batch)
            self._prefill()
            assert not module._kvmem_mtp_prefix_active
            if fail:
                raise OSError('native proposal failure')
            return 'drafts'
    proposer = Proposer(); proposer.model = model
    batch = Batch('logical-pos', 'logical-lens', 'logical-bounds')
    runner = NS(speculative_config=NS(method='mtp'), speculator=proposer,
        _kvmem_layers=['target'], max_num_tokens=4,
        _kvmem_pending_capture=([NS(name='target')], batch), _kvmem_mtp_rejected='accepted-prefix',
        _kvmem_window_batch=NS(positions='dense-pos', groups={0:NS(seq_lens='dense-lens',cpu_upper_bound='dense-bounds')}),
        _kvmem_capture_runtime=NS(submit=lambda c,b,r: published.append((c,b,r))))
    mtp_runtime.install_mtp_runtime(runner)
    assert proposer.propose(batch) == 'drafts'
    assert native_batches[0] == Batch('dense-pos', 'dense-lens', 'dense-bounds')
    captures, original_batch, rejected = published[0]
    assert [c.name for c in captures] == ['target','__mtp__.layer']
    assert captures[1].key.tolist() == [7]*4 and original_batch is batch and rejected == 'accepted-prefix'
    assert runner._kvmem_pending_capture is None
    runner._kvmem_pending_capture = ([NS(name='target')], batch)
    runner._kvmem_mtp_rejected = 'accepted-prefix'
    with pytest.raises(OSError):
        proposer.propose(batch, fail=True)
    assert len(published) == 1 and not module._kvmem_mtp_prefix_active
    assert runner._kvmem_pending_capture is not None

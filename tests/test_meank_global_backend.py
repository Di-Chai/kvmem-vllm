"""KVMEM_MEANK_GLOBAL_BACKEND: the global Query MeanK scoring backend is opt-in."""
import pytest
import torch

from vllm_kvmem import cuda_scoring


def test_global_scoring_backend_is_legacy_by_default_and_tiled_only_when_enabled(monkeypatch):
    used = []
    monkeypatch.setattr(cuda_scoring, '_compact_values',
                        lambda q, m, s, **kw: used.append(kw.get('backend', 'legacy')) or torch.ones(m.shape[1]))
    q, means = torch.ones(1, 2, 2, 4), torch.ones(1, 5, 1, 4)
    eligible = torch.ones(5, dtype=torch.bool)
    monkeypatch.delenv('KVMEM_MEANK_GLOBAL_BACKEND', raising=False)
    cuda_scoring.mean_key_scores(q, means, eligible)
    monkeypatch.setenv('KVMEM_MEANK_GLOBAL_BACKEND', 'tiled')
    cuda_scoring.mean_key_scores(q, means, eligible)
    monkeypatch.setenv('KVMEM_MEANK_GLOBAL_BACKEND', 'legacy')
    cuda_scoring.mean_key_scores(q, means, eligible)
    assert used == ['legacy', 'tiled', 'legacy']
    monkeypatch.setenv('KVMEM_MEANK_GLOBAL_BACKEND', 'fast')
    with pytest.raises(ValueError, match='KVMEM_MEANK_GLOBAL_BACKEND'):
        cuda_scoring.mean_key_scores(q, means, eligible)

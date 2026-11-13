"""Explicit final Query scorer choice cannot silently fall back."""
import pytest
from vllm_kvmem.serving.layerwise_query import LayerwiseQuery
from test_layerwise_query import runner


def test_query_score_backend_is_default_off_and_requires_supported_path(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    monkeypatch.delenv('KVMEM_LAYERWISE_QUERY_SCORE_BACKEND',raising=False)
    assert LayerwiseQuery(runner()).score_backend=='legacy'
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_SCORE_BACKEND','tiled')
    with pytest.raises(ValueError,match='prepared optimized sequential'):
        LayerwiseQuery(runner())
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_OPTIMIZED','1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PREPARED_INDEX','1')
    assert LayerwiseQuery(runner()).score_backend=='tiled'
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','global')
    with pytest.raises(ValueError,match='prepared optimized sequential'):
        LayerwiseQuery(runner())
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_SCORE_BACKEND','bad')
    with pytest.raises(ValueError,match='legacy or tiled'):
        LayerwiseQuery(runner())

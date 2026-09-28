import pytest
from vllm_kvmem.speculative_admission import SpeculativeQueryPolicy


def test_arbitrary_acceptance_counts_cannot_skip_the_refresh_boundary():
    policy = SpeculativeQueryPolicy(128, 131072, 65536, 8)
    for offset in (0, 65536, 262144):
        boundary = offset + 131072 + 65536 - 128
        # Cover every distance over two speculative blocks and every possible
        # accepted prefix, including repeated rejection down to one token.
        for start in range(boundary - 17, boundary):
            for desired_acceptance in range(1, 9):
                history = start
                steps = 0
                while history < boundary:
                    scheduled = policy.admit(history, 8, 131070, offset)
                    assert 1 <= scheduled <= 8 and history + scheduled <= boundary
                    history += min(desired_acceptance, scheduled)
                    steps += 1
                    assert steps <= 17
                assert policy.refresh(history, 131070, offset)
                assert policy.admit(history, 8, 131070, offset) == 8
                # Refresh restores a K-token prefix; only the accepted query
                # prefix then advances logical history and dense position.
                new_offset = history - policy.select_budget
                history += min(desired_acceptance, 8)
                assert policy.admit(history, 8, 131070, new_offset) == 8


def test_query_reselection_excludes_unverified_draft_rows():
    policy = SpeculativeQueryPolicy(128, 131072, 65536, 8)
    assert policy.authoritative_probe_tokens(131072, 197110, 1024) == 1024
    assert policy.authoritative_probe_tokens(197108, 197110, 8) == 2
    assert policy.authoritative_probe_tokens(197110, 197110, 8) == 1
    assert policy.authoritative_probe_tokens(197111, 197110, 3) == 1
    with pytest.raises(RuntimeError, match='crossed'):
        policy.admit(196481, 8, 100, 0)
    with pytest.raises(ValueError, match='headroom'):
        SpeculativeQueryPolicy(4, 32, 16, 8)

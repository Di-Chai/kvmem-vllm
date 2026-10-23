"""Fixed-operand replay keeps raw blocks, masks, targets and timing distinct."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

import analyze_query_key_fps as diagnostic
from vllm_kvmem.config import KVMemConfig


def test_padded_tail_does_not_become_a_representative_and_indices_reference_raw_keys():
    # Negative keys make zero padding especially dangerous for maximum logits.
    raw = -torch.arange(1, 1+35*2*3, dtype=torch.float32).reshape(1, 35, 2, 3)/8
    raw = raw.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    reps, indices, timing = diagnostic.build_representatives(raw, [32, 3], batch_pages=1)
    decoded = torch.from_numpy(raw).view(torch.float8_e4m3fn).float()
    for page, count, begin in [(0, 32, 0), (1, 3, 32)]:
        for head in range(2):
            good = indices[0, page, :, head]
            good = good[good >= 0]
            assert len(good) == min(4, count) and len(set(good.tolist())) == len(good)
            assert int(good.max()) < count
            for slot, position in enumerate(indices[0, page, :, head]):
                if position >= 0:
                    assert torch.equal(reps[0, page, slot, head].float(), decoded[0, begin+position, head])
    assert all(timing[key] >= 0 for key in ('raw_read_decode_seconds', 'farthest_first_seconds',
        'representative_copy_seconds', 'build_seconds'))


def test_scoring_preserves_complete_block_softmax_domain_and_query_mass():
    torch.manual_seed(62)
    raw = torch.randn(2, 67, 1, 4).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    q = torch.randn(2, 3, 2, 4)
    meta = dict(dtype='float8_e4m3fn', raw_k_scale=1., key_shape=list(raw.shape),
        counts=[32, 32, 3], score_mask=[False, True, True])
    score, indices, timing = diagnostic.score_capture(q, raw, meta, batch_pages=2)
    assert score[0] == 0 and np.isclose(score.sum(), 3)
    assert indices.shape == (2, 3, 4, 1)
    assert timing['score_mass'] == pytest.approx(3)
    assert timing['expected_score_mass'] == 3


def test_partition_rejects_missing_raw_tokens_and_unsupported_scale():
    with pytest.raises(ValueError, match='partition'):
        diagnostic.build_representatives(np.zeros((1, 33, 1, 2), dtype=np.uint8), [32])
    with pytest.raises(ValueError, match='unit-scale'):
        diagnostic.score_capture(torch.zeros(1, 1, 1, 2),
            np.zeros((1, 32, 1, 2), dtype=np.uint8), dict(dtype='float8_e4m3fn', raw_k_scale=.5))


def native_fixture():
    config = KVMemConfig(block_tokens=32, select_budget=128, gen_budget=64,
        repository_blocks=32, sink_blocks=1, recent_blocks=0)
    starts, counts = list(range(0, 320, 32)), [32]*10
    raw = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32)
    meta = dict(starts=starts, counts=counts, history=320, budget=128,
        score_mask=[False]+[True]*9, transport_mask=[False]+[True]*9,
        selected_starts=[0, 32, 64, 96])
    return dict(kvmem=asdict(config)), meta, raw


def test_native_picker_preserves_fixed_budget_and_keeps_targets_post_hoc():
    profile, meta, raw = native_fixture()
    variants = dict(captured_mean_k=raw, key_fps_4=np.arange(10, dtype=np.float32))
    selected = diagnostic.native_selections(profile, meta, variants)
    assert selected['captured_mean_k'] == [0, 32, 64, 96]
    assert selected['key_fps_4'] == [0, 224, 256, 288]
    target = dict(token_span=[252, 259], key_token_span=[248, 251])
    evaluated = diagnostic.evaluate_selection(variants['key_fps_4'], selected['key_fps_4'], meta, target)
    assert evaluated['value']['state'] == evaluated['key']['state'] == 'full'
    assert evaluated['selected_tokens'] == 128
    assert evaluated['optional_selected_blocks'] == 3
    assert evaluated['graph_alpha'] == 0 and not evaluated['post_picker_neighbors_added']
    other_target = dict(token_span=[100, 105], key_token_span=[90, 93])
    other = diagnostic.evaluate_selection(variants['key_fps_4'], selected['key_fps_4'], meta, other_target)
    assert other['selected_starts'] == evaluated['selected_starts']
    assert other['value']['state'] == 'none'


def test_native_picker_rejects_changed_masks_and_original_selection():
    profile, meta, raw = native_fixture()
    meta['score_mask'][1] = False
    with pytest.raises(ValueError, match='domain differs'):
        diagnostic.native_selections(profile, meta, dict(captured_mean_k=raw))
    meta['score_mask'][1] = True
    meta['selected_starts'] = [0, 32, 64, 128]
    with pytest.raises(ValueError, match='cannot be reproduced'):
        diagnostic.native_selections(profile, meta, dict(captured_mean_k=raw))


def test_partial_coverage_cutoff_and_competition_rank_do_not_claim_answers():
    _, meta, raw = native_fixture()
    value = diagnostic.evaluate_selection(raw, [0, 32, 64, 96], meta,
        dict(token_span=[125, 132], key_token_span=[120, 123]))
    assert value['value'] == dict(tokens=7, selected_tokens=3, state='partial',
        missing_positions=[128, 129, 130, 131])
    assert value['optional_cutoff'] == 7
    assert [block['rank'] for block in value['target_blocks']] == [3, 4]
    assert 'answer_correct' not in value


def test_existing_attempt_is_never_overwritten(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    (source/'fixture.json').write_text(json.dumps(dict(rows=[dict(id='case')])) )
    output = tmp_path/'old'; output.mkdir()
    evidence = output/'summary.json'; evidence.write_text('immutable')
    with pytest.raises(ValueError, match='overwrite'):
        diagnostic.main(['--source', str(source), '--output', str(output)])
    assert evidence.read_text() == 'immutable'

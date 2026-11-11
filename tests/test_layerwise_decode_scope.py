"""Opt-in rolling decode score sources (KVMEM_LAYERWISE_DECODE_SCOPE / _INTERVAL / _PREFILL).

Default ('all', no prefill fold) keeps the original rows. 'non-thinking' samples
only accepted output outside <think>...</think> with the native parser's
'non-thinking' scope; the prefill fold adds accepted public-prefill rows (minus
thinking) to the same per-layer decode history. Keys are never filtered.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem import layerwise_decode as ld
from vllm_kvmem.action_attention import ActionHistory, ActionPacket, _TwinParser, _layerwise_mask
from vllm_kvmem.layerwise_decode import DecodeHistory, NativeDecodeHistory
from test_graph_capture import options as action_options
from test_layerwise_pressure import controller, packet as prefill_packet

IDS = [10, 11, 12, 13, 14]   # <tool_call> </tool_call> <think> </think> <|im_start|>
OPEN, CLOSE, HEADER = 12, 13, 14


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ('KVMEM_LAYERWISE_DECODE_SCOPE', 'KVMEM_LAYERWISE_DECODE_INTERVAL',
                 'KVMEM_LAYERWISE_DECODE_PREFILL', 'KVMEM_LAYERWISE_DECODE_PREFILL_WEIGHT',
                 'KVMEM_NATIVE_DECODE_MASS_MAP'):
        monkeypatch.delenv(name, raising=False)


# --- flags ---------------------------------------------------------------------------------

def test_flags_default_to_the_original_rule():
    o = NS(action_interval=16, prefill_attention_stride=32)
    assert ld.decode_scope() == 'all' and ld.decode_interval(o) == 16
    assert not ld.prefill_fold_enabled() and ld.prefill_fold_weight(o) == 2
    assert ld.query_source_label() == ('accepted_output_including_thinking', 'all_accepted_including_thinking')


@pytest.mark.parametrize('name,value', [('KVMEM_LAYERWISE_DECODE_SCOPE', 'thinking'),
                                        ('KVMEM_LAYERWISE_DECODE_INTERVAL', '0'),
                                        ('KVMEM_LAYERWISE_DECODE_INTERVAL', 'x'),
                                        ('KVMEM_LAYERWISE_DECODE_PREFILL', 'yes'),
                                        ('KVMEM_LAYERWISE_DECODE_PREFILL_WEIGHT', '0')])
def test_flags_are_validated(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    o = NS(action_interval=16, prefill_attention_stride=32)
    with pytest.raises(ValueError):
        ld.decode_scope(); ld.decode_interval(o); ld.prefill_fold_enabled(); ld.prefill_fold_weight(o)


def test_weight_follows_stride_over_interval(monkeypatch):
    o = NS(action_interval=16, prefill_attention_stride=32)
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_INTERVAL', '4')
    assert ld.prefill_fold_weight(o) == 8
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_INTERVAL', '64')
    assert ld.prefill_fold_weight(o) == 1
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PREFILL_WEIGHT', '3')
    assert ld.prefill_fold_weight(o) == 3
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_SCOPE', 'non-thinking')
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PREFILL', '1')
    assert ld.query_source_label()[0] == 'accepted_non_thinking_output_and_prefill'


# --- twin parser ---------------------------------------------------------------------------

def _history(**kw):
    o = action_options(action_interval=16, action_native_decode_commit=True, pressure_policy='attention',
                       guided_query_tokens=0, query_history=False, **kw)
    return ActionHistory(o, IDS)


def test_default_scope_builds_the_original_parser_only():
    h = _history(); h.begin('r', [HEADER])
    assert _layerwise_mask(h.parser) is None and not isinstance(h.parser.parser, _TwinParser)


def test_twin_parser_samples_non_thinking_rows_and_keeps_the_legacy_mask(monkeypatch):
    plain = _history(); plain.begin('r', [HEADER])
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_SCOPE', 'non-thinking')
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_INTERVAL', '2')
    twin = _history(); twin.begin('r', [HEADER])
    steps = [[20, 21, OPEN, 22, 23], [24, CLOSE, 25, 26, 27], [28, 29, 30, 31, 32]]
    accepted = [5, 3, 5]
    masks = []
    for tokens, n in zip(steps, accepted):
        legacy = plain.parser.prepare(tokens)
        assert twin.parser.prepare(tokens) == legacy           # legacy mask unchanged
        masks.append(_layerwise_mask(twin.parser))
        assert twin.parser.commit(n) == plain.parser.commit(n)
        assert twin.parser.state == plain.parser.state
    # Non-thinking output counts 20:0 21:1 | 22,23 thinking | 25:2 26:3 27:4 (candidates; 26,27
    # rejected, so the committed count stays 3) | 28:3 29:4 30:5 31:6 32:7; every 2nd is sampled.
    assert masks[0] == 0b00001
    assert masks[1] == 0b10100                     # 24 thinking, </think> marker never sampled
    assert masks[2] == 0b01010                     # rejected rows did not advance the count


def test_thinking_prompt_starts_the_sampler_inside_thinking(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_SCOPE', 'non-thinking')
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_INTERVAL', '1')
    h = _history(); h.begin('r', [HEADER, OPEN])
    h.parser.prepare([20, 21, CLOSE, 22])
    assert _layerwise_mask(h.parser) == 0b1000


# --- decode fold through the controller ----------------------------------------------------

def _decode_packet(c, history, rows, mask, count=4, authority=None):
    c.maps['r'] = (256, (((0, 1, 0, 32, 0), (1, 1, 64, 32, 32)),
                          ((0, 1, 0, 32, 0), (2, 1, 128, 32, 32))))
    c.rolling_decode = True
    p = ActionPacket(history, rows, [0, 999, 256], None, None,
                     source_positions=tuple(256 + r for r in rows), history_begin=256, candidate_count=count)
    if mask is not None:
        p.layerwise_mask = mask
    masses = torch.tensor([[[.1, .8, .1]] * len(rows), [[.2, .7, .1]] * len(rows)])
    p.masses = masses[0]
    c.attach_packet(p, 'r', authority, [0, 32, 64], [32, 32, 4], masses)
    return p


def test_non_thinking_mask_selects_the_folded_rows(monkeypatch):
    h = _history(); h.begin('r', [HEADER])
    c = controller()
    p = _decode_packet(c, h, [0, 2], 0b0101)
    c.commit_accept(c.prepare_accept(p, 4))
    a, b = c.decode_histories['r']
    assert (a.samples, b.samples, a.end) == (2, 2, 260)
    assert a.mass[64] == pytest.approx(1.6) and b.mass[128] == pytest.approx(1.4)
    p = _decode_packet(c, h, [1], 0b0010)           # row 1 = position 257 < end: not folded twice
    staged = c.prepare_accept(p, 4)
    assert staged[1][0] is c.decode_histories['r'][0]


def test_partial_acceptance_folds_only_accepted_mask_rows():
    h = _history(); h.begin('r', [HEADER])
    c = controller()
    p = _decode_packet(c, h, [0, 3], 0b1001)
    c.commit_accept(c.prepare_accept(p, 2))
    assert [x.samples for x in c.decode_histories['r']] == [1, 1]


def test_mask_rows_must_be_scored():
    h = _history(); h.begin('r', [HEADER])
    c = controller()
    p = _decode_packet(c, h, [0], 0b0011)
    with pytest.raises(RuntimeError, match='omitted'):
        c.prepare_accept(p, 4)


# --- prefill fold --------------------------------------------------------------------------

def _prefill(c, monkeypatch, interval='16'):
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_INTERVAL', interval)
    p = prefill_packet(c)                             # rows [31] of chunk 256..288
    p.state.options.action_interval = 16
    p.state.options.prefill_attention_stride = 32
    c.rolling_decode = True
    return p


@pytest.mark.parametrize('native', ['0', '1'])
def test_prefill_rows_join_the_decode_history_with_the_stride_weight(monkeypatch, native):
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_MASS_MAP', native)
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PREFILL', '1')
    c = controller(); p = _prefill(c, monkeypatch, interval='4')   # weight 32/4 = 8
    prepared = c.prepare_accept(p, 32)
    assert len(prepared) == 4 and 'r' not in c.decode_histories
    c.commit_accept(prepared)
    a, b = c.decode_histories['r']
    assert type(a) is (NativeDecodeHistory if native == '1' else DecodeHistory)
    assert (a.samples, a.end, a.chunks) == (8, 288, 1) == (b.samples, b.end, b.chunks)
    assert a.scores()[64] == pytest.approx(.8) and b.scores()[128] == pytest.approx(.2)
    assert set(c.histories['r'][0].values) == {0, 64, 256}          # prefill history unchanged
    assert len(c.prepare_accept(p, 32)) == 2                         # same rows: nothing to fold twice


def test_prefill_fold_is_off_by_default_and_skips_partial_or_thinking_rows(monkeypatch):
    c = controller(); p = _prefill(c, monkeypatch)
    assert len(c.prepare_accept(p, 32)) == 2
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PREFILL', '1')
    assert len(c.prepare_accept(p, 31)) == 2                         # partial chunk publishes nothing
    p.prefill_thinking = frozenset({31})
    assert len(c.prepare_accept(p, 32)) == 2                         # its only row is thinking


def test_prefill_fold_rejects_rows_that_do_not_sum_to_one(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PREFILL', '1')
    c = controller(); p = _prefill(c, monkeypatch)
    rid, histories, masses = p.layerwise_pressure[1:]
    masses = masses.clone(); masses[1, 0, 0] = 5.
    with pytest.raises(RuntimeError, match='sum to one'):
        ld.fold_prefill(c, 'r', p, 32, histories, masses)


def test_prefill_fold_then_decode_rows_share_one_history(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PREFILL', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_SCOPE', 'non-thinking')
    c = controller(); p = _prefill(c, monkeypatch)                  # weight 32/16 = 2
    c.commit_accept(c.prepare_accept(p, 32))
    h = _history(); h.begin('r', [HEADER])
    d = _decode_packet(c, h, [0], 0b0001, authority=c._history_owners['r'])
    d.history_begin = 288; d.source_positions = (288,)
    c.commit_accept(c.prepare_accept(d, 4))
    a = c.decode_histories['r'][0]
    assert (a.samples, a.end, a.chunks) == (3, 292, 2)
    assert a.mass[64] == pytest.approx(2 * .8 + .8)


# --- prefill thinking rows -----------------------------------------------------------------

def test_prefill_thinking_rows_follow_markers_across_chunks():
    s = NS(token_ids=IDS)
    first = [20, OPEN, 21, 22]
    assert ld.prefill_thinking_rows(s, 100, first, [0, 1, 2, 3]) == {1, 2, 3}
    second = [23, CLOSE, 24, HEADER, 25]
    assert ld.prefill_thinking_rows(s, 104, second, [0, 1, 2, 3, 4]) == {0, 1}
    # A re-plan of the second chunk still starts inside thinking.
    assert ld.prefill_thinking_rows(s, 104, second, [0, 2]) == {0}
    # A request's first chunk starts outside thinking.
    assert ld.prefill_thinking_rows(NS(token_ids=IDS), 500, [20, 21], [0, 1]) == frozenset()


def test_launch_validation_rejects_malformed_scope_flags(monkeypatch):
    profile = dict(kvmem=dict(rolling_active_window=True, request_planning=True, async_refresh=True,
                              action_attention=True, prefill_attention=True, pressure_policy='attention'))
    kw = dict(query_mode='sequential', optimized=True, pressure=True, single_assembly=True, original_rope=False)
    ld.validate_configuration(profile, **kw)
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_SCOPE', 'non-thinking')
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_INTERVAL', '4')
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PREFILL', '1')
    ld.validate_configuration(profile, **kw)
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_INTERVAL', '-1')
    with pytest.raises(ValueError):
        ld.validate_configuration(profile, **kw)

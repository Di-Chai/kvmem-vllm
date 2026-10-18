"""Exact renderer boundaries are separate from retrieval/mandatory semantics."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_kvmem.prompt_spans import (combine_spans, graph_metadata, install_graph_metadata,
    map_prompt_spans, normalize_provenance)
from vllm_kvmem.request_plan import RequestPlan
from vllm_kvmem.block_graph import BlockGraph


def chars(text):
    return list(range(len(text))), [(i, i+1) for i in range(len(text))]


def map_chars(text, spans, **kwargs):
    ids, offsets = chars(text)
    return map_prompt_spans(text, spans, ids, encoded_ids=ids, offset_mapping=offsets, **kwargs)


def test_utf8_repeated_body_empty_content_and_consecutive_tool_boundaries():
    parts = ['<user>重复🙂</user>', '<tool>重复🙂</tool>', '<tool></tool>']
    text = ''.join(parts)
    byte, char, spans, expected = 0, 0, [], []
    for part, role in zip(parts, ('user', 'tool', 'tool')):
        spans.append(dict(begin=byte, end=byte+len(part.encode()), role=role))
        expected.append((char, char+len(part), role))
        byte += len(part.encode())
        char += len(part)
    mapped = map_chars(text, spans, strict=True, token_offset=17)
    assert mapped.spans == tuple((17+b, 17+e, r) for b, e, r in expected)
    assert dict(mapped.provenance)['status'] == 'exact'
    assert len(mapped.spans) == 3  # Adjacent tools remain independent messages.


def test_full_encoding_crossing_token_is_unknown_not_independent_prefix_length():
    ids = [10, 11]
    spans = [dict(begin=0, end=2, role='user'), dict(begin=2, end=4, role='tool')]
    mapped = map_prompt_spans('abcd', spans, ids, encoded_ids=ids, offset_mapping=[(0,3),(3,4)])
    assert mapped.spans == ()
    assert len(mapped.unknown) == 2
    with pytest.raises(ValueError, match='crosses'):
        map_prompt_spans('abcd', spans, ids, encoded_ids=ids,
                         offset_mapping=[(0,3),(3,4)], strict=True)


def test_multiple_tokens_for_one_multibyte_character_remain_in_one_message():
    mapped = map_prompt_spans('中a', [(0,3,'user'), (3,4,'tool')], [1,2,3],
        encoded_ids=[1,2,3], offset_mapping=[(0,1),(0,1),(1,2)], strict=True)
    assert mapped.spans == ((0,2,'user'), (2,3,'tool'))
    assert map_prompt_spans('中a', [(0,1,'user')], [1,2,3], encoded_ids=[1,2,3],
        offset_mapping=[(0,1),(0,1),(1,2)]).spans == ()


@pytest.mark.parametrize('ids,offsets', [([2],[(0,1)]), ([1],[(0,0)]),
    ([1],None), ([1],[(0,2)])])
def test_bad_or_unavailable_offsets_never_change_actual_tokens(ids, offsets):
    actual = [1]
    mapped = map_prompt_spans('a', [(0,1,'user')], actual, encoded_ids=ids, offset_mapping=offsets)
    assert actual == [1] and mapped.spans == ()
    assert dict(mapped.provenance)['status'] == 'unknown'


def test_metadata_only_preserves_query_mandatory_replay_and_explicit_policy():
    base = dict(kvmem_query_span=[64,80], kvmem_replay_span=[64,96])
    original = RequestPlan.parse(base, 128, 32, 256)
    mapped = map_chars('x'*128, [(0,32,'system'), (32,64,'user'), (64,96,'tool')])
    enriched = RequestPlan.parse({**base, **mapped.payload()}, 128, 32, 256)
    assert (enriched.query, enriched.replay, enriched.mandatory, enriched.explicit_mandatory) == (
        original.query, original.replay, original.mandatory, original.explicit_mandatory)
    assert not enriched.explicit_mandatory
    reparsed = RequestPlan.parse(enriched.payload(), 128, 32, 256)
    assert reparsed.locality_spans == enriched.locality_spans
    assert reparsed.locality_provenance == enriched.locality_provenance
    assert enriched.message_spans == ()


def test_legacy_message_pins_are_unchanged_and_conflicts_are_rejected():
    legacy = [dict(begin=0,end=32,role='system'), dict(begin=32,end=64,role='user')]
    base = RequestPlan.parse(dict(kvmem_message_spans=legacy), 96, 32, 256)
    added = RequestPlan.parse(dict(kvmem_message_spans=legacy, kvmem_locality_spans=legacy), 96, 32, 256)
    assert base.mandatory == added.mandatory == ((0,32,'system'), (32,64,'root'))
    assert base.explicit_mandatory == added.explicit_mandatory is True
    for invalid in ([dict(begin=0,end=32,role='user')], [dict(begin=0,end=40,role='system')],
                    [dict(begin=4,end=32,role='system')]):
        with pytest.raises(ValueError, match='Conflicting'):
            RequestPlan.parse(dict(kvmem_message_spans=legacy, kvmem_locality_spans=invalid),96,32,256)


@pytest.mark.parametrize('spans', [[dict(begin=0,end=33,role='tool')],
    [dict(begin=0,end=8,role='unknown')], [dict(begin=True,end=8,role='tool')],
    [dict(begin=0,end=0,role='tool')], [dict(begin=8,end=16,role='tool'),dict(begin=0,end=8,role='tool')]])
def test_invalid_locality_coordinates_are_rejected(spans):
    with pytest.raises(ValueError):
        RequestPlan.parse(dict(kvmem_locality_spans=spans),32,4,32)


def test_graph_metadata_is_atomic_snapshot_owned_and_crops_unknown_ends():
    graph = BlockGraph('owned', 4)
    provenance = normalize_provenance(dict(schema_version=1, source='renderer', status='exact'))
    binding = SimpleNamespace(message_spans=(), locality_spans=((0,20,'tool'),), locality_provenance=provenance)
    install_graph_metadata(graph, binding)
    snapshot = graph.snapshot()
    old = snapshot.binding
    install_graph_metadata(graph, SimpleNamespace(), validated_prefix=9)
    assert graph.message_spans == ((0,9,'tool'),)
    assert graph.message_end_known == (False,)
    assert snapshot.message_spans == ((0,20,'tool'),)
    assert snapshot.locality_provenance == provenance
    assert graph.binding != old
    install_graph_metadata(graph, binding, validated_prefix=9)
    assert graph.message_end_known == (True,)
    before = graph.binding
    with pytest.raises(ValueError, match='Conflicting'):
        install_graph_metadata(graph, SimpleNamespace(locality_spans=((0,8,'user'),)), validated_prefix=9)
    assert graph.binding == before
    snapshot.close()
    graph.close()
    assert graph.bytes == 0


def test_raw_tokens_have_explicit_unknown_provenance():
    spans, provenance = graph_metadata(SimpleNamespace())
    assert spans == () and dict(provenance)['source'] == 'raw_token_ids'
    assert dict(provenance)['status'] == 'unknown'

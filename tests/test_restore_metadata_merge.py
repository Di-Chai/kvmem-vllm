"""Differential checks against pre-optimization boundary merge semantics."""
import random
from types import SimpleNamespace

import pytest

from vllm_kvmem.block_graph import _merge_spans, _spans
from vllm_kvmem.prompt_spans import combine_spans, graph_metadata, normalize_spans


def reference_combine(old, new):
    old, new = normalize_spans(old, label='Message'), normalize_spans(new)
    for left in old:
        for right in new:
            if max(left[0],right[0]) < min(left[1],right[1]) and left != right:
                raise ValueError('conflict')
    return tuple(sorted(set(old+new)))


def reference_merge(old, known, supplied, prefix):
    incoming = _spans(supplied)
    if type(prefix) is not int or prefix < 0:
        raise ValueError('prefix')
    retained = [(b,min(e,prefix),r,k and e<=prefix) for (b,e,r),k in zip(old,known) if b<prefix]
    for b,e,r,k in retained:
        for nb,ne,nr in incoming:
            if max(b,nb) < min(e,ne,prefix):
                if b!=nb or r!=nr or ne<e or k and ne!=e:
                    raise ValueError('conflict')
    result = {(b,r):((b,e,r),k) for b,e,r,k in retained}
    for span in incoming:result[span[0],span[2]]=(span,True)
    merged = sorted(result.values())
    spans = tuple(span for span,_ in merged)
    _spans(spans)
    return spans,tuple(k for _,k in merged)


def spans(rng):
    cursor, result = 0, []
    for _ in range(rng.randrange(16)):
        begin = cursor+rng.randrange(4)
        cursor = begin+rng.randrange(1,9)
        result.append((begin,cursor,rng.choice(('system','user','assistant','tool'))))
    return tuple(result)


def outcome(function, *args):
    try:return function(*args)
    except ValueError:return ValueError


def test_linear_combination_preserves_conflicts_and_exact_shared_spans():
    rng = random.Random(413)
    for _ in range(2000):
        old = spans(rng)
        new = tuple(s for s in old if rng.random()<.5) if rng.random()<.5 else spans(rng)
        assert outcome(combine_spans,old,new)==outcome(reference_combine,old,new)


def test_prefix_merge_matches_reference_with_unknown_ends_and_replacement():
    rng = random.Random(912)
    for _ in range(3000):
        old = spans(rng)
        known = tuple(rng.choice((False,True)) for _ in old)
        incoming = tuple(s for s in old if rng.random()<.6) if rng.random()<.5 else spans(rng)
        prefix = rng.randrange(110)
        assert outcome(_merge_spans,old,known,incoming,prefix)==outcome(reference_merge,old,known,incoming,prefix)


def test_large_restore_keeps_complete_new_endpoint_and_unknown_inherited_endpoint():
    old = tuple((i*10,i*10+10,'tool') for i in range(12000))
    known = (True,)*len(old)
    assert _merge_spans(old,known,old,119995)==(old,known)
    merged,ends = _merge_spans(old,known,(),119995)
    assert merged[-1]==(119990,119995,'tool') and ends[-1] is False
    with pytest.raises(ValueError,match='Conflicting'):
        _merge_spans(old,known,((119990,119994,'tool'),),119995)


@pytest.mark.parametrize('locality', [((0,12,'tool'),), ((0,20,'tool'),(20,21,'user')),
    ((0,4,'user'),(3,5,'assistant')), ((0,4,'unknown'),), ((True,4,'tool'),)])
def test_graph_metadata_does_not_skip_bounds_roles_or_order_validation(locality):
    with pytest.raises(ValueError):
        graph_metadata(SimpleNamespace(tokens=range(10),locality_spans=locality))


def test_graph_metadata_accepts_exact_end_and_preserves_message_union():
    old = ((0,4,'system'),(4,8,'user'))
    new = ((4,8,'user'),(8,10,'tool'))
    result,_ = graph_metadata(SimpleNamespace(tokens=range(10),message_spans=old,locality_spans=new))
    assert result==old+new[1:]

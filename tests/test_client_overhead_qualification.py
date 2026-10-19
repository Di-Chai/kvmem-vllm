"""Prevent replayed bookkeeping from becoming fresh inference/quality evidence."""
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
try:
    diagnostic=importlib.import_module('qualify_client_overhead')
finally:
    sys.path.pop(0)


def test_selects_same_ordered_full_questions_without_modifying_source():
    rows=[dict(id=f'{g}_q{q}',token_sha256=f'{g}:{q}') for g in ('a','b','c') for q in range(4)]
    original=json.dumps(rows)
    chosen=diagnostic.select_cases(rows,groups=2,questions=3)
    assert list(chosen)==['a','b'] and chosen['b']==rows[4:7]
    assert json.dumps(rows)==original
    with pytest.raises(ValueError):diagnostic.select_cases(rows,questions=5)


def seed_fixture(tmp_path):
    rows=[dict(id=f'{g}_q0',token_sha256=g,prompt_sha256='p',query_sha256='q',input_tokens=99) for g in ('fresh','old')]
    seed=tmp_path/'seed';path=seed/'cases/old_q0/result.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(id='old_q0',status='completed',token_sha256='old',
        prompt_sha256='p',query_sha256='q',prompt_tokens=99,answer='keep first answer',graph_v2=dict(full=[1,2,3]))))
    return dict(rows=rows,selected={'fresh':[rows[0]]},seed_dataset=str(seed),
                seed_sha256={'old_q0':diagnostic.sha(path)}),path


def test_metadata_seed_keeps_bytes_but_cannot_seed_selected_histories(tmp_path):
    fixture,source=seed_fixture(tmp_path);data=source.read_bytes();out=tmp_path/'trial'
    seeded=diagnostic.seed_progress(out,fixture)
    assert [r['id'] for r in seeded]==['old_q0']
    assert source.read_bytes()==data==(out/'cases/old_q0/result.json').read_bytes()
    assert not (out/'cases/fresh_q0').exists()
    assert not (out/'cases/old_q0/attempt.json').exists()
    assert 'not fresh' in json.loads((out/'seed-manifest.json').read_text())['scope']


@pytest.mark.parametrize('change',['source_hash','failed','binding'])
def test_changed_or_failed_metadata_seed_is_rejected(tmp_path,change):
    fixture,source=seed_fixture(tmp_path);data=json.loads(source.read_text())
    if change=='failed':data['status']='error'
    elif change=='binding':data['prompt_tokens']=98
    else:data['answer']='later replacement'
    source.write_text(json.dumps(data))
    if change!='source_hash':fixture['seed_sha256']['old_q0']=diagnostic.sha(source)
    with pytest.raises(ValueError,match='identity'):diagnostic.seed_progress(tmp_path/'trial',fixture)

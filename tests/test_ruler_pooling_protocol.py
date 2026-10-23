from copy import deepcopy
import json
from pathlib import Path

import pytest

from ruler_pooling_protocol import ARMS, PROFILE, expected_profile, validate_execution
from run_ruler_pooling_queue import prerequisite_ready
from test_ruler_query_only_protocol import fixture as query_fixture


def contract(arm):
    _,old=query_fixture()
    reference=old['reference_profile']
    reference.update(max_num_seqs=8,max_num_batched_tokens=16424,prefill_chunk_tokens=2048)
    reference['kvmem'].update(guided_query_backend='eager',guided_reselect='boundary')
    f={k:v for k,v in old.items() if k!='platform_path_overrides'}
    f.update(execution_profile=PROFILE,pooling_arm=arm,reference_profile=reference)
    return expected_profile(reference,arm),f


@pytest.mark.parametrize('arm',ARMS)
def test_each_pooling_arm_preserves_the_exact_frozen_contract(arm):
    p,f=contract(arm);before=deepcopy(f)
    validate_execution(p,f)
    assert f==before
    assert p['kvmem']['prefill_physical_coalescing'] is False
    assert p['kvmem']['pressure_policy']=='attention'


def test_guided_pair_differs_only_in_cap_and_service_port():
    a,_=contract('max');b,_=contract('max_guided')
    a.pop('port');b.pop('port')
    assert a['kvmem'].pop('guided_query_tokens')==0
    assert b['kvmem'].pop('guided_query_tokens')==512
    assert a==b


@pytest.mark.parametrize('field,value',[('gen_budget',65536),('graph_aggregation','sum'),
    ('pressure_policy','graph'),('guided_query_tokens',256),('prefill_attention_stride',16)])
def test_unrelated_or_wrong_arm_changes_are_rejected(field,value):
    p,f=contract('max');p['kvmem'][field]=value
    with pytest.raises(ValueError):validate_execution(p,f)


def test_queue_does_not_advance_before_native_cleanup_is_complete(tmp_path):
    (tmp_path/'run').mkdir()
    state=tmp_path/'status.json'
    state.write_text(json.dumps(dict(status='testing')))
    assert not prerequisite_ready(tmp_path)
    state.write_text(json.dumps(dict(status='completed')))
    assert not prerequisite_ready(tmp_path)
    (tmp_path/'run/cleanup.json').write_text(json.dumps(dict(remaining_owned_groups={},gpu_processes=[])))
    assert prerequisite_ready(tmp_path)
    state.write_text(json.dumps(dict(status='failed')))
    with pytest.raises(RuntimeError):prerequisite_ready(tmp_path)

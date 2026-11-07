"""Endpoint provenance and fair pooled-throughput accounting; no GPU needed."""
from copy import deepcopy
import json

import pytest

from analyze_layerwise_dynamic8 import aggregate, completed_endpoint, controls


def endpoint():
    totals=dict(public_batches=8,public_prefill_tokens=800,public_non_prefill_tokens=0,
                decode_request_batches=0)
    return dict(expected=8,owners={str(i):100 for i in range(8)},history_tokens=800,
                gpu_completion_confirmed=True,at=5.,public_work=dict(
                    counter_id='stable-counter',submitted=totals,completed=deepcopy(totals),
                    all_submitted_completed=True,pending_fence=None,
                    completed_fence=dict(completion_lower_bound=4.,completion_upper_bound=4.5)))


def test_accepts_gpu_completed_cold_history_endpoint():
    row=endpoint()
    assert completed_endpoint(row)['completed']['public_prefill_tokens']==800


@pytest.mark.parametrize('fault',('pending','wrong_count','query_work','decode','late_fence','missing_owner','no_confirmation'))
def test_rejects_unfinished_contaminated_or_wrong_cohort_endpoint(fault):
    row=endpoint();work=row['public_work']
    if fault=='pending':work['pending_fence']={'sequence':1}
    elif fault=='wrong_count':
        work['submitted']['public_prefill_tokens']=801
        work['completed']['public_prefill_tokens']=801
    elif fault in ('query_work','decode'):
        key='public_non_prefill_tokens' if fault=='query_work' else 'decode_request_batches'
        work['submitted'][key]=work['completed'][key]=1
    elif fault=='late_fence':work['completed_fence']['completion_upper_bound']=6.
    elif fault=='missing_owner':row['owners'].pop('7')
    else:row['gpu_completion_confirmed']=False
    with pytest.raises(ValueError):completed_endpoint(row)


def reports():
    return [dict(kind=kind,seconds=seconds,gpu_completed_history_tokens=100,
                 tokens_per_second=100/seconds) for kind,seconds in
            [('static',1.),('incremental',2.),('incremental',2.),('static',3.)]]


def test_pooled_throughput_uses_total_work_over_wall_not_mean_of_rates():
    result=aggregate(reports())
    assert result['pooled']['static']['tokens_per_second']==50.
    assert result['pooled']['incremental']['tokens_per_second']==50.
    assert result['dynamic_change_percent']==0.


def test_pooling_rejects_changed_work_or_unplanned_order():
    rows=reports();rows[2]['gpu_completed_history_tokens']=99
    with pytest.raises(ValueError,match='same history'):aggregate(rows)
    rows=reports();rows[1],rows[3]=rows[3],rows[1]
    with pytest.raises(ValueError,match='predeclared'):aggregate(rows)


def files(root,*,dynamic=False,tokens=300000):
    root.mkdir()
    profile=dict(execution_mode='compiled_graph',draft_tokens=4,max_num_seqs=8,
        max_num_batched_tokens=16424,kvmem=dict(select_budget=36864,gen_budget=16384,
        block_tokens=32,sink_blocks=41,recent_blocks=0,cpu_read_cache_bytes=112000000000,
        gpu_memory_admission='incremental' if dynamic else 'static',
        request_swap_layerwise=dynamic,request_checkpoint_bytes=16*1024**3 if dynamic else 0))
    fixture=dict(cases=8,rows=[dict(id=str(i),input_tokens=tokens,source=str(root/str(i))) for i in range(8)],
                 bindings=[dict(id=str(i),request=str(root/(str(i)+'.json')),sha256=str(i)) for i in range(8)])
    (root/'profile.json').write_text(json.dumps(profile))
    (root/'fixture.json').write_text(json.dumps(fixture))


def test_control_comparison_ignores_only_mode_and_local_input_locations(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';files(a);files(b,dynamic=True)
    assert controls(a)[2]==controls(b)[2]
    profile=json.loads((b/'profile.json').read_text())
    profile['kvmem']['pressure_policy']='different'
    (b/'profile.json').write_text(json.dumps(profile))
    assert controls(a)[2]!=controls(b)[2]


def test_incomplete_cpu_admission_cannot_enter_finite_gate_measurement(tmp_path):
    root=tmp_path/'cannot-fit';files(root,tokens=755000)
    with pytest.raises(ValueError,match='deadlock'):controls(root)

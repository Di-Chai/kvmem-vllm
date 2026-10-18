"""Harness contracts; real-model acceptance is performed by the frozen client."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
try:
    bench = importlib.import_module('benchmark_graph_v2_ablation')
    analysis = importlib.import_module('analyze_graph_v2_ablation')
    client = importlib.import_module('qualify_graph_v2_ablation')
    native = importlib.import_module('analyze_block_graph')
finally:
    sys.path.pop(0)


def base_profile():
    return dict(model_format='safetensors',execution_mode='compiled_graph',draft_method='mtp',draft_tokens=4,
        max_num_seqs=8,kvmem=dict(block_tokens=32,select_budget=36864,gen_budget=16384,sink_blocks=41,recent_blocks=0,
        session_capacity=8,request_planning=True,archive_backend='cpu',mean_k_backend='qw3_cuda',
        repository_gc=False,query_history=False,action_fraction=0,action_attention=True,prefill_attention=True,
        action_scope='non-thinking',graph_cross_message_weight=.25,graph_alpha=.2,pressure_policy='graph'))


def actual_record():
    record = native.causal_fixture()
    record.pop('evidence')
    record.update(version=2,phase='query',scoring_version='v2',alpha=.2,
        score_domain_version='qw3-native-v2',accepted_boundary=24,
        score_mask=[True]*6,transport_mask=[True]*6,handles=[[i,1] for i in range(6)])
    record['graph'].update(message_spans=[[0,12,'tool'],[12,24,'user']],message_end_known=[True,True])
    return record


def test_four_arms_only_change_scoring_and_graph_only_boundaries():
    configs = bench.profiles(base_profile(),1234)
    bench.validate_profiles(configs)
    assert configs['v1-attention']['kvmem']['graph_attention_weight'] == .2
    assert 'graph_alpha' not in configs['v1-attention']['kvmem']
    tokens=list(range(96))
    plan=dict(kvmem_query_span=[80,90],kvmem_mandatory_spans=[dict(begin=0,end=8,kind='system')],
        kvmem_locality_spans=[dict(begin=0,end=80,role='tool')],
        kvmem_locality_provenance=dict(schema_version=1,source='test'))
    original=deepcopy(plan)
    contracts=[bench.plan_contract(tokens,bench.arm_plan(plan,arm),configs[arm]) for arm in bench.ARMS]
    assert all(value==contracts[0] for value in contracts) and plan==original
    assert bench.arm_plan(plan,'v2-multi-known')==plan
    assert 'kvmem_locality_spans' not in bench.arm_plan(plan,'v2-multi-unknown')
    configs['v2-attention']['kvmem']['action_interval']=32
    with pytest.raises(ValueError,match='non-ablation'): bench.validate_profiles(configs)


def test_same_native_snapshot_replay_preserves_inputs_and_reports_unknown_quality():
    record=actual_record();original=deepcopy(record)
    rows={arm:analysis.replay(record,arm) for arm in bench.ARMS}
    assert record==original
    assert all(row['selected_tokens']<=record['budget'] for row in rows.values())
    assert all(all(v is None for v in row['quality'].values()) for row in rows.values())
    assert rows['v2-attention']['diagnostics']['attention_external_mass']>rows['v1-attention']['diagnostics']['attention_external_mass']
    assert analysis.compare(rows['v2-multi-unknown'],rows['v2-multi-known'])['score_l1']>0
    assert analysis.effective_boundaries(record)['covered']
    assert len({row['original_record_sha256'] for row in rows.values()})==1


def test_full_actual_ranges_are_counts_and_not_truncated_page_lists():
    record=dict(starts=list(range(0,4096*32,32)),counts=[32]*4096,accepted_boundary=4096*32,budget=36864)
    query=dict(selected_history_ranges=[[i*32,32] for i in range(1152)],selected_tokens=36864,
               history_begin=4096*32,replay_span=[4096*32,4096*32+16])
    assert len(analysis.full_selected_starts(record,query))==1152
    query['selected_history_ranges']=query['selected_history_ranges'][:256]
    with pytest.raises(ValueError,match='selected-token'): analysis.full_selected_starts(record,query)
    query['selected_history_ranges']=[[0,64]]
    with pytest.raises(ValueError,match='selected range'): analysis.full_selected_starts(record,query)


def export_fixture(tmp_path, *, trace=True, nodes=4096):
    from vllm_kvmem.config import KVMemConfig
    cfg=KVMemConfig(block_tokens=32,select_budget=36864,gen_budget=16384,repository_blocks=16384,
        sink_blocks=0,recent_blocks=0,action_attention=True,action_fraction=0)
    rows=[NS(history_begin=i*32,count=32,handle=NS(slot=i,generation=1)) for i in range(nodes)]
    authority=NS(graph_options=NS(graph_trace=trace,graph_mode='collect',graph_scoring_version='v2',graph_alpha=.2,
        graph_locality_weight=.1,graph_attention_weight=.2,graph_cross_message_weight=.25),
        graph_report_path=tmp_path/'block_graph.jsonl',memory=NS(config=cfg,request_id='owner'),
        history_end=nodes*32,query_score_domain=dict(version='qw3-native-v2',score_mask=[True]*nodes),records={i:r for i,r in enumerate(rows)})
    snapshot=NS(rows=(),partial=None,message_spans=(),message_end_known=(),locality_provenance=(),
        accepted_through=nodes*32,completed_through=nodes*32,revision=1)
    return authority,rows,snapshot


def export(authority,rows,snapshot):
    from vllm_kvmem.graph_integration import record_query_inputs
    record_query_inputs(authority,rows,[1/len(rows)]*len(rows),[True]*len(rows),snapshot,(),None)


def test_opt_in_export_accepts_real_131k_domain_and_default_stays_disabled(tmp_path):
    authority,rows,snapshot=export_fixture(tmp_path,trace=False)
    export(authority,rows,snapshot)
    assert not (tmp_path/'block_graph_inputs.jsonl').exists()
    authority.graph_options.graph_trace=True
    export(authority,rows,snapshot)
    path=tmp_path/'block_graph_inputs.jsonl'
    loaded,_=native.load_records([path],8)
    assert len(loaded)==1 and len(loaded[0]['starts'])==4096
    native._validate_record(loaded[0])


@pytest.mark.parametrize('excess',['nodes','record_bytes','record_count','total_bytes'])
def test_export_still_refuses_records_beyond_bounded_limits(tmp_path,excess):
    from vllm_kvmem import graph_integration as integration
    authority,rows,snapshot=export_fixture(tmp_path,nodes=8193 if excess=='nodes' else 4)
    path=tmp_path/'block_graph_inputs.jsonl'
    if excess=='record_bytes': snapshot.locality_provenance=(('oversized','x'*(2*1024*1024)),)
    if excess=='record_count': authority._graph_input_count=8
    if excess=='total_bytes':
        with path.open('wb') as stream: stream.truncate(16*1024*1024)
    before=path.stat().st_size if path.exists() else None
    export(authority,rows,snapshot)
    assert (path.stat().st_size if path.exists() else None)==before
    assert integration._MAX_LOG_BYTES==16*1024*1024


def test_reader_keeps_legacy_records_and_rejects_oversized_new_records(tmp_path):
    path=tmp_path/'inputs.jsonl'
    path.write_text(json.dumps(native.causal_fixture())+'\n')
    rows,_=native.load_records([path],1)
    assert native.analyze_record(rows[0])['input_version']==1
    path.write_text(json.dumps(dict(oversized='x'*(2*1024*1024)))+'\n')
    with pytest.raises(ValueError,match='bounded export'): native.load_records([path],1)
    record=actual_record()
    record['starts']=list(range(0,8193*4,4));record['counts']=[4]*8193
    record['raw_scores']=[0.]*8193;record['eligible']=[True]*8193
    with pytest.raises(ValueError,match='oversized'): native._validate_record(record)

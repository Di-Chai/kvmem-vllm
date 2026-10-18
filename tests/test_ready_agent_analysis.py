import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('ready_agent_analysis',
    Path(__file__).parents[1]/'scripts/analyze_ready_agents.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def step(n=2):
    return dict(started_at=5.,decode_requests=n,prefill_requests=0,prefill_tokens=0)


def test_unknown_membership_uses_shortest_subset_instead_of_entire_http_demand():
    rows=[dict(begin=1.,end=9.,input_tokens=n,session=str(i)) for i,n in enumerate([20,20,1000])]
    r=m.capacity_proof(rows,[step()],dict(block_size=10,num_gpu_blocks=10),dict(begin=0.,end=10.))
    assert r['matched_steps']==1 and r['steps_exceeding_native_pool']==0
    assert r['best']['native_attention_pages_lower_bound']==4
    rows[0]['input_tokens']=rows[1]['input_tokens']=100
    assert m.capacity_proof(rows,[step()],dict(block_size=10,num_gpu_blocks=10),dict(begin=0.,end=10.))['steps_exceeding_native_pool']==1


def test_mixed_step_and_duplicate_sessions_cannot_prove_capacity():
    rows=[dict(begin=1.,end=9.,input_tokens=100,session='same')]*2
    args=(dict(block_size=10,num_gpu_blocks=10),dict(begin=0.,end=10.))
    assert m.capacity_proof(rows,[step()],*args)['matched_steps']==0
    rows[1]=dict(rows[1],session='other')
    assert m.capacity_proof(rows,[dict(step(),prefill_tokens=1)],*args)['matched_steps']==0


def test_history_proof_checks_full_history_not_compressed_active_length():
    row=dict(kind='workload',request_id='r',response_id='cmpl-r',input_tokens=100,output_ids=[1,2,3])
    publication=dict(request_id='cmpl-r-0-1234abcd',history_end=102,checkpoint_tokens=102,checkpoint_kind='M')
    assert m.history_proof([row],[publication])['passed']
    assert not m.history_proof([row],[dict(publication,history_end=64,checkpoint_tokens=64)])['passed']
    assert not m.history_proof([row],[])['passed']

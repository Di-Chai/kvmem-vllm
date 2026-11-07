from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.async_refresh import RefreshJob
from vllm_kvmem.layerwise_no_probe import window_geometry, validate_options, commit_layer_zero
from vllm_kvmem.layerwise_query import LayerwiseQuery
from vllm_kvmem.memory import Phase
from test_layerwise_query import runner
from test_recall_experiment import native_authority, ScoringAuthority, finish


def record(slot, count=32):
    return NS(handle=NS(slot=slot), count=count)


def test_geometry_uses_capacity_and_fixed_partial_without_scoring():
    assert window_geometry([record(0),record(1),record(2),record(3,9)], (0,3), 96) == ((0,3),73)
    assert window_geometry([record(0),record(1,9)], (0,), 96) == ((0,1),41)
    assert window_geometry([record(i) for i in range(8)], (0,), 96) == ((0,),96)


@pytest.mark.parametrize('records,fixed,budget', [
    ([record(0),record(1),record(2,9)], (0,),64),
    ([record(0)], (1,),32), ([record(0,33)], (0,),32),
    ([record(0),record(1)], (0,1),32), ([],(),32), ([record(0)],(),31)])
def test_ambiguous_or_invalid_geometry_is_rejected(records,fixed,budget):
    with pytest.raises(ValueError):window_geometry(records,fixed,budget)


@pytest.mark.parametrize('override', [dict(async_refresh=False),dict(guided_query_tokens=1),
    dict(query_history=True),dict(repository_gc=True),dict(historical_gc=True),
    dict(session_capacity=1,session_freeze_prompt=False)])
def test_rejects_unsupported_state_or_history_lifecycle(override):
    options=NS(**dict(dict(async_refresh=True),**override))
    with pytest.raises(ValueError):validate_options(options)


class Fence:
    def record(self):pass
    def synchronize(self):pass
    def query(self):return True


def setup(monkeypatch,tmp_path,no_probe,native_batches=False,partial_tail=False):
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_OPTIMIZED','1')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_DIAGNOSTICS','1')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_NO_PROBE',str(int(no_probe)))
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MTP_SELECTION','layer0')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES',str(int(native_batches)))
    monkeypatch.delenv('KVMEM_LAYERWISE_PREFILL_PRESSURE',raising=False)
    monkeypatch.setattr(torch.cuda,'Event',Fence)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *a:Fence())
    from vllm_kvmem import cuda_scoring, retrieval
    monkeypatch.setattr(cuda_scoring,'mean_key_scores',retrieval.mean_key_scores)
    history=233 if partial_tail else 256
    base=native_authority(count=history,budget=96,recent=int(partial_tail))
    keys=torch.randn(3,8,32,1,2,generator=torch.Generator().manual_seed(38))
    authority=ScoringAuthority(keys,[32]*7+[history-224],memory=base.memory)
    authority.publish_sealed=lambda *_:None
    authority.archive=NS(wait_for_capacity=lambda _:None)
    engine=runner(eager=not native_batches,sequences=8 if native_batches else 1)
    engine.device='cpu';engine._kvmem_probe_output=tmp_path
    engine._kvmem_archive_layers=(*engine._kvmem_layers,'__mtp__.draft')
    engine._kvmem_request_generations={'r':4}
    engine._kvmem_sessions=None
    offsets=[]
    engine._kvmem_window_adapter=NS(commit=lambda *args:offsets.append(args))
    engine._kvmem_last_batch=NS(req_ids=['r'],num_computed_tokens_np=[history],num_tokens=2,
                                query_start_loc_np=[0,2])
    controller=LayerwiseQuery(engine)
    controller.pressure=NS(rolling_decode=True,maps={})
    command=NS(request_id='r',command_id=7,selection='query',groups=[(0,NS(window_tokens=96,history_end=history))],
        mandatory_spans=(),query_span=(history,history+2),input_tokens=(10,20),guided_private=False)
    job=RefreshJob(command);job.authority=authority
    refresh=NS(runner=engine,options=NS(memory=authority.memory.config,async_refresh=True),emit=lambda *a,**k:None)
    return controller,refresh,job,offsets


@pytest.mark.parametrize('native_batches',[False,True])
@pytest.mark.parametrize('partial_tail',[False,True])
def test_public_layer_selection_matches_replay_control_and_commits_mtp_and_maps(monkeypatch,tmp_path,native_batches,partial_tail):
    outcomes=[]
    for no_probe in (False,True):
        directory=tmp_path/str(no_probe);directory.mkdir()
        c,refresh,job,offsets=setup(monkeypatch,directory,no_probe,native_batches,partial_tail)
        a=job.authority
        if no_probe:
            original=tuple(a.memory._store.working_records())
            c.arm_no_probe(refresh,job)
            assert tuple(a.memory._store.working_records()) == original
            assert job.done and job.steps == 0 and job.state is None and job.restore_event is None
            assert offsets == [('r',job.history,73 if partial_tail else 96)]
        else:
            c.prepare(refresh,job,torch.zeros(2,2,1,2))
            # A deliberately different global set must not affect the target.
            plan=a.memory.select({r.handle.slot:float(8-i) for i,r in enumerate(a.records.values())})
            finish(a.memory);job.selected_tokens=plan.total_window_tokens;c.committed(job)
        chosen_by_layer=[]
        def assemble(state,index,chosen,**kwargs):
            chosen_by_layer.append((index,[r.history_begin for r in chosen]))
            return dict(uploaded_bytes=len(chosen)*64)
        monkeypatch.setattr(c,'assemble_layer_pinned',assemble)
        hidden=torch.tensor([[[1.,.5]],[[.25,1.]]])
        outputs=[]
        for index in range(2):
            c.projection(index,NS(query=hidden))
            chosen=chosen_by_layer[index][1]
            # A small causal attention computation makes the next Q depend on
            # the actual previous layer's chosen KV, not a fixed test vector.
            keys=torch.cat([r.raw[index] for r in a.records.values() if r.history_begin in chosen])
            logits=hidden[:,0]@keys[:,0].T/(2**.5)
            hidden=(logits.softmax(-1)@keys[:,0]).unsqueeze(1)
            outputs.append(hidden.clone())
        assert chosen_by_layer[-1] == (2,chosen_by_layer[0][1])
        maps=c.pressure.maps['r'][1]
        assert [[r[2] for r in rows] for rows in maps] == [r[1] for r in chosen_by_layer[:2]]
        if no_probe:
            assert [r[2] for r in a.memory._store.working_records()] == chosen_by_layer[0][1]
            assert a.memory.phase is Phase.READY
            from test_core import append
            append(a.memory,2,local=job.selected_tokens)
            assert a.memory.history_end == job.history+2
            assert sum(r[3] for r in a.memory._store.working_records()) == job.selected_tokens+2
        assert not c.has_active
        outcomes.append((chosen_by_layer,outputs))
        assert a.memory.close()
    assert outcomes[0][0] == outcomes[1][0]
    for a,b in zip(outcomes[0][1],outcomes[1][1]):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_cancellation_before_public_query_releases_armed_owner(monkeypatch,tmp_path):
    c,refresh,job,_=setup(monkeypatch,tmp_path,True,True)
    c.arm_no_probe(refresh,job)
    c.retire('r')
    assert not c.has_active and not c.pending
    assert job.authority.memory.close()


def test_batched_owners_match_independent_query_windows_and_mtp(monkeypatch,tmp_path):
    queries = (torch.tensor([[[1., .5]], [[.25, 1.]]]),
               torch.tensor([[[-1., -.5]], [[-.25, -1.]]]))
    expected = []
    for index, q in enumerate(queries):
        path = tmp_path / f'independent-{index}'
        path.mkdir()
        c, refresh, job, _ = setup(monkeypatch, path, True, True, bool(index))
        chosen = []
        def assemble(state, layer, records, **kwargs):
            chosen.append((layer, [r.history_begin for r in records]))
            return {}
        monkeypatch.setattr(c, 'assemble_layer_pinned', assemble)
        c.arm_no_probe(refresh, job)
        for layer in range(2):
            c.projection(layer, NS(query=q))
        expected.append(chosen)
        assert job.authority.memory.close()

    path = tmp_path / 'batched'
    path.mkdir()
    c, refresh, job, _ = setup(monkeypatch, path, True, True)
    _, second_refresh, second_job, _ = setup(monkeypatch, path, True, True, True)
    second_refresh.runner = c.runner
    second_job.command.request_id = 'other'
    c.runner._kvmem_request_generations['other'] = 5
    c.runner._kvmem_last_batch = NS(req_ids=['other', 'r'],
        num_computed_tokens_np=[second_job.history, job.history],
        num_tokens=4, query_start_loc_np=[0, 2, 4])
    chosen = {'r': [], 'other': []}
    def assemble(state, layer, records, **kwargs):
        chosen[state['request_id']].append((layer, [r.history_begin for r in records]))
        return {}
    monkeypatch.setattr(c, 'assemble_layer_pinned', assemble)
    c.arm_no_probe(refresh, job)
    c.arm_no_probe(second_refresh, second_job)
    for layer in range(2):
        c.projection(layer, NS(query=torch.cat((queries[1], queries[0]))))
    assert [chosen['r'], chosen['other']] == expected
    assert not c.has_active
    for rid, owner in [('r', job), ('other', second_job)]:
        assert [[row[2] for row in rows] for rows in c.pressure.maps[rid][1]] == [
            rows for _, rows in chosen[rid][:2]]
        assert [row[2] for row in owner.authority.memory._store.working_records()] == chosen[rid][0][1]
        assert owner.authority.memory.close()


def test_recycled_request_is_rejected_before_any_kv_write(monkeypatch,tmp_path):
    c,refresh,job,_=setup(monkeypatch,tmp_path,True)
    c.arm_no_probe(refresh,job)
    c.runner._kvmem_request_generations['r']=5
    with pytest.raises(RuntimeError,match='recycled'):c.projection(0,None)
    assert job.authority.memory.close()


def test_failed_completion_fence_never_publishes_canonical_frame(monkeypatch,tmp_path):
    c,refresh,job,_=setup(monkeypatch,tmp_path,True)
    records=list(job.authority.records.values())[:3]
    state=dict(authority=job.authority,layer_zero_records=records,selected_tokens=96)
    class BrokenFence(Fence):
        def synchronize(self):raise RuntimeError('device failure')
    monkeypatch.setattr(torch.cuda,'Event',BrokenFence)
    with pytest.raises(RuntimeError,match='device failure'):commit_layer_zero(c,state)
    assert job.authority.memory.phase is Phase.FAILED


def test_cli_keeps_original_path_by_default_and_declares_mtp_control(monkeypatch,tmp_path,capsys):
    import json
    from vllm_kvmem import serve
    profile=dict(max_num_seqs=8,execution_mode='compiled_graph',kvmem=dict(async_refresh=True))
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile))
    monkeypatch.setattr(serve,'build_launch',lambda p,*a:dict(profile=p,environment={}))
    flags=['--config',str(path),'--run-dir',str(tmp_path/'run'),'--dry-run',
           '--layerwise-query-diagnostic','sequential','--layerwise-query-optimized',
           '--layerwise-query-native-batches']
    serve.main(flags)
    original=json.loads(capsys.readouterr().out)['environment']
    assert 'KVMEM_LAYERWISE_QUERY_NO_PROBE' not in original
    assert 'KVMEM_LAYERWISE_QUERY_MTP_SELECTION' not in original
    serve.main(flags+['--layerwise-query-no-probe'])
    changed=json.loads(capsys.readouterr().out)['environment']
    assert changed == dict(original,KVMEM_LAYERWISE_QUERY_NO_PROBE='1',KVMEM_LAYERWISE_QUERY_MTP_SELECTION='layer0')
    with pytest.raises(ValueError,match='global MTP'):
        serve.main(flags+['--layerwise-query-no-probe','--layerwise-query-mtp-selection','global'])
    assert not (tmp_path/'run').exists()


def test_async_admission_never_enters_probe_restore_or_global_assembly(monkeypatch,tmp_path):
    import numpy as np
    from vllm_kvmem import async_refresh, prompt_spans
    c,refresh,job,_=setup(monkeypatch,tmp_path,True)
    engine=c.runner
    engine._kvmem_layerwise_query=c
    engine.req_states=NS(req_id_to_index={'r':0})
    engine._kvmem_capture_runtime=NS(authorities={'r':job.authority})
    engine.block_tables=NS(num_blocks=NS(np=np.array([[3]])),
                           block_tables=[NS(gpu=torch.tensor([[1,2,3]]))])
    job.command.groups[0][1].original_block_ids=(1,2,3)
    monkeypatch.setattr(async_refresh,'stored_expansions',lambda _:{0:1})
    monkeypatch.setattr(prompt_spans,'install_selection_metadata',lambda *a,**k:None)
    monkeypatch.setattr(prompt_spans,'selection_metadata',lambda _:None)
    manager=async_refresh.AsyncRefresh.__new__(async_refresh.AsyncRefresh)
    manager.runner=engine;manager.options=refresh.options;manager.emit=refresh.emit
    manager._ensure_window_adapter=lambda _:None
    def forbidden(*args,**kwargs):pytest.fail('No-probe admission entered private/global work')
    manager._forward=manager._restore_state=manager._submit=manager._assemble=forbidden
    assert manager.step(job)
    assert job.done and job.stage=='public_query' and job.state is None and job.steps==0
    c.retire('r')
    assert job.authority.memory.close()

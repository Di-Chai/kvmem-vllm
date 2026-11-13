"""Unaccepted execution candidates must not silently change serving defaults."""
import json
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving import serve


@pytest.fixture
def launch(tmp_path,monkeypatch,capsys):
    profile=dict(max_num_seqs=8,execution_mode='compiled_graph',
        kvmem=dict(select_budget=16384,gen_budget=4096,archive_backend='cpu',repository_gc=False,
                   async_refresh=True,action_attention=True,prefill_attention=True,pressure_policy='attention'))
    config=tmp_path/'profile.json';config.write_text(json.dumps(profile))
    monkeypatch.setattr(serve,'build_launch',lambda p,*a:dict(profile=p,environment={}))
    flags=['--layerwise-query-diagnostic','sequential','--layerwise-query-optimized',
           '--layerwise-query-native-batches','--layerwise-prefill-pressure',
           '--layerwise-pressure-single-assembly','--layerwise-paired-reads']
    def run(extra=(),omit=()):
        config.write_text(json.dumps(profile))
        serve.main(['--config',str(config),'--run-dir',str(tmp_path/'run'),'--dry-run',
                    *[f for f in flags if f not in omit],*extra])
        result=json.loads(capsys.readouterr().out)
        # The launcher records which path the strategy options choose for (layering step 3).
        assert result['profile']==dict(profile,kvmem=dict(profile['kvmem'],strategy_path='layer'))
        assert not (tmp_path/'run').exists()
        return result['environment']
    return run,profile


def test_new_pressure_execution_candidates_are_all_disabled_by_default(launch):
    run,_=launch;env=run()
    for name in ('PREPARED_READS','POST_FORWARD_REAP','SUBMIT_ON_PREPARE','NATIVE_SCORE_EXPORT','REUSE_LAYER_VIEWS','RESIDENT_VALUES','BATCH_SELECTION'):
        assert 'KVMEM_LAYERWISE_PRESSURE_'+name not in env


def test_batch_selection_is_explicit_and_preserves_all_profile_parameters(launch):
    run,_=launch
    baseline=run()
    assert run(['--layerwise-pressure-batch-selection']) == {
        **baseline, 'KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION': '1'}


def test_batch_selection_rejects_unqualified_execution_before_model_launch(launch):
    run,_=launch
    with pytest.raises(ValueError,match='per-layer pressure'):
        run(['--layerwise-pressure-batch-selection'],omit=('--layerwise-prefill-pressure',))


@pytest.mark.parametrize('flag,name,extra',[
    ('prepared-reads','PREPARED_READS',()),('post-forward-reap','POST_FORWARD_REAP',()),
    ('submit-on-prepare','SUBMIT_ON_PREPARE',()),
    ('reuse-layer-views','REUSE_LAYER_VIEWS',()),
    ('native-score-export','NATIVE_SCORE_EXPORT',('--layerwise-buffered-history',))])
def test_explicit_execution_controls_only_add_the_requested_environment(launch,flag,name,extra):
    run,_=launch
    assert run(['--layerwise-pressure-'+flag,*extra])['KVMEM_LAYERWISE_PRESSURE_'+name]=='1'


@pytest.mark.parametrize('flag,omit,message',[
    ('prepared-reads',('--layerwise-pressure-single-assembly',),'single pressure assembly'),
    ('post-forward-reap',('--layerwise-pressure-single-assembly',),'single pressure assembly'),
    ('submit-on-prepare',('--layerwise-pressure-single-assembly',),'single pressure assembly'),
    ('reuse-layer-views',('--layerwise-pressure-single-assembly',),'single pressure assembly'),
    ('prepared-reads',('--layerwise-paired-reads',),'paired reads'),
    ('native-score-export',(),'buffered histories')])
def test_invalid_candidate_combinations_fail_before_model_launch(launch,flag,omit,message):
    run,_=launch
    with pytest.raises(ValueError,match=message):run(['--layerwise-pressure-'+flag],omit)


@pytest.mark.parametrize('change',[dict(archive_backend='nvme'),dict(repository_gc=True),dict(historical_gc={'mode':'shadow'})])
def test_prepared_read_opt_in_does_not_claim_unsupported_backends(launch,change):
    run,profile=launch;profile['kvmem'].update(change)
    with pytest.raises(ValueError,match='without GC'):run(['--layerwise-pressure-prepared-reads'])


def test_resident_values_requires_existing_profile_capacity_and_packed_path(launch):
    run,profile=launch
    with pytest.raises(ValueError,match='existing V staging'):
        run(['--layerwise-pressure-resident-values'])
    profile['kvmem']['reuse_resident_v']=True
    assert run(['--layerwise-pressure-resident-values','--layerwise-packed-assembly'])[
        'KVMEM_LAYERWISE_PRESSURE_RESIDENT_VALUES']=='1'


def test_layerwise_controller_candidates_require_explicit_environment(monkeypatch):
    from vllm_kvmem.serving.layerwise_pressure import LayerwisePrefillPressure
    query=NS(optimized=True,native_batches=True,names=('a',),runner=NS(model_config=NS(enforce_eager=False)))
    for name in ('PREPARED_READS','POST_FORWARD_REAP','SUBMIT_ON_PREPARE'):
        monkeypatch.delenv('KVMEM_LAYERWISE_PRESSURE_'+name,raising=False)
    controller=LayerwisePrefillPressure(query)
    assert not controller.prepared_reads and not controller.post_forward_reap and not controller.submit_on_prepare
    for name in ('PREPARED_READS','POST_FORWARD_REAP','SUBMIT_ON_PREPARE'):
        monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_'+name,'1')
    controller=LayerwisePrefillPressure(query)
    assert controller.prepared_reads and controller.post_forward_reap and controller.submit_on_prepare


def test_memory_driver_preserves_optional_controls_without_expanding_old_fixture_keys(tmp_path):
    import run_layerwise_memory as memory
    old=memory.inherited_execution({})
    assert set(old)==set(memory.EXECUTION_FLAGS)
    explicit={key:True for key in memory.EXPERIMENTAL_EXECUTION_FLAGS}
    inherited=memory.inherited_execution(explicit)
    assert all(inherited[key] is True for key in explicit)
    command=memory.server_command(tmp_path,dict(execution_flags=inherited))
    assert all('--'+key.replace('_','-') in command for key in explicit)
    command=memory.server_command(tmp_path,dict(execution_flags=old))
    assert all('--'+key.replace('_','-') not in command for key in explicit)

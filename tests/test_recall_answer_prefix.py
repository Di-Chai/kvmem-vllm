from copy import deepcopy
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return (importlib.import_module('recall_answer_prefix'),
            importlib.import_module('recall_comparison_protocol'))


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer
    path=Path(__file__).resolve().parents[1]/'artifacts/recall-comparison-20261028-r1/assets/tokenizer/tokenizer.json'
    if not path.exists():pytest.skip('Optional local frozen Qwen tokenizer is not packaged in CI')
    return Tokenizer.from_file(str(path))


def test_prefix_is_assistant_prefill_and_preserves_public_query(modules,tokenizer):
    helper,_=modules
    prompt='<|im_start|>user\nQuestion: Which variables have value 123?<|im_end|>\n'+helper.ASSISTANT
    tokens=tokenizer.encode(prompt,add_special_tokens=False).ids
    prefix=' Answer: According to the chain, 5 variables have value 123, they are: '
    actual,proof=helper.append_official_prefix(tokens,prefix,tokenizer)
    assert tokenizer.decode(actual,skip_special_tokens=False)==prompt+prefix
    assert actual[:proof['retained_tokens']]==tokens[:proof['retained_tokens']]
    assert proof['appended_tokens']==len(actual)-len(tokens)>0
    assert 'LJXLD' not in tokenizer.decode(actual,skip_special_tokens=False)
    with pytest.raises(ValueError,match='unprefilled'):
        helper.append_official_prefix(actual,prefix,tokenizer)


def test_prefix_rejects_open_thinking_and_missing_prefix(modules,tokenizer):
    helper,_=modules
    tokens=tokenizer.encode('<|im_start|>assistant\n<think>\n',add_special_tokens=False).ids
    with pytest.raises(ValueError,match='non-thinking'):
        helper.append_official_prefix(tokens,' Answer:',tokenizer)
    with pytest.raises(ValueError,match='Missing'):
        helper.append_official_prefix(tokens,'',tokenizer)


def pilot_fixture(p):
    rows=[dict(id=f'{t}-{i:04d}',task=t,index=i) for t in p.ruler.TASK_FAMILIES for i in range(20)]
    return dict(answer_prefix_pilot=True,pressure_attention_hybrid=True,value_recall_reference=True,
        full_value_comparison=False,answer_prefix_policy='official_after_assistant_template_v1',
        starts_in_thinking=False,methods=list(p.PREFIX_PILOT_METHODS),rows=rows,pilot_samples_per_task=2,
        qualification_ids=[r['id'] for r in rows if r['index']<2])


def test_three_methods_share_a_frozen_balanced_first_case_cohort(modules):
    _,p=modules;f=pilot_fixture(p)
    assert p.campaign_methods(f)==('mean_k','block_value_norm','block_deletion_error')
    f['qualification_ids'][-1]=f['qualification_ids'][-1].replace('0001','0002')
    with pytest.raises(ValueError,match='first N'):
        p.campaign_methods(f)


def test_single_method_130_case_extension_keeps_the_first_ten_of_every_task(modules):
    _,p=modules;f=pilot_fixture(p)
    f.update(methods=['block_deletion_error'],pilot_selected_methods=['block_deletion_error'],
        pilot_samples_per_task=10,qualification_ids=[r['id'] for r in f['rows'] if r['index']<10])
    assert len(f['qualification_ids'])==130
    assert p.campaign_methods(f)==('block_deletion_error',)
    f['methods']=['mean_k']
    with pytest.raises(ValueError,match='matrix'):p.campaign_methods(f)


def test_single_method_extension_cannot_reorder_or_duplicate_a_matrix(modules):
    _,p=modules;f=pilot_fixture(p)
    f.update(methods=['block_deletion_error','block_deletion_error'],
        pilot_selected_methods=['block_deletion_error','block_deletion_error'])
    with pytest.raises(ValueError,match='matrix'):p.campaign_methods(f)


def test_mean_k_uses_its_native_scratch_bound_with_the_same_kv_frame(modules):
    _,p=modules
    base=dict(kvmem=dict(select_budget=36864,gen_budget=16384),
        recall_experiment=dict(method='block_value_norm',block_batch=1024,query_batch=128))
    mean=p.method_profile(base,'mean_k');value=p.method_profile(base,'block_value_norm')
    assert mean['recall_experiment']['block_batch']==128
    assert value['recall_experiment']['block_batch']==1024
    assert mean['kvmem']==value['kvmem']==base['kvmem']


@pytest.mark.parametrize('change',['pressure','thinking','prefix','matrix'])
def test_prefix_pilot_rejects_mixed_controls(modules,change):
    _,p=modules;f=pilot_fixture(p)
    if change=='pressure':f['pressure_attention_hybrid']=False
    if change=='thinking':f['starts_in_thinking']=True
    if change=='prefix':f['answer_prefix_policy']='question_rewrite'
    if change=='matrix':f['methods']=['mean_k','block_value_norm']
    with pytest.raises(ValueError):p.campaign_methods(f)


def test_pilot_cannot_accidentally_dispatch_the_full_formal_suite(modules,monkeypatch,tmp_path):
    _,p=modules;controller=importlib.import_module('run_recall_comparison')
    monkeypatch.setattr(controller,'verify',lambda root:(pilot_fixture(p),{}))
    with pytest.raises(ValueError,match='cannot run the formal'):
        controller.run(SimpleNamespace(root=tmp_path,phase='formal',qualify_entrypoints=False))

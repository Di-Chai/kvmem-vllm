import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch


def profile():
    return json.loads((Path(__file__).parents[1]/'examples/qwen-nvfp4-offline-agent-k128-g128.json').read_text())


def options(payload):
    return RuntimeOptions.from_vllm(NS(kv_transfer_config=NS(
        kv_connector_extra_config={'kvmem':payload}), cache_config=NS(cache_dtype='fp8_e4m3')))


def test_coalescing_is_explicit_and_does_not_mutate_connector_settings():
    p=profile(); before=copy.deepcopy(p)
    assert not options(p['kvmem']).offline_prefill_coalescing
    assert p==before
    p['kvmem'].update(offline_prefill_coalescing=True,offline_prefill_max_tokens=4096)
    o=options(p['kvmem'])
    assert o.offline_prefill_coalescing and o.offline_prefill_max_tokens==4096
    build_launch(p,Path('/tmp'),Path('/tmp/unused-coalescing-test'),check_paths=False)


@pytest.mark.parametrize('changes',[
    {'offline_prefill_coalescing':1},
    {'offline_agent_policy':False},
    {'offline_phase_policy':True},
    {'offline_prefill_max_tokens':True},
    {'offline_prefill_max_tokens':0},
    {'offline_prefill_max_tokens':4097},
    {'offline_prefill_max_tokens':262144},
    {'adaptive_prefill':True},
    {'prefill_batch_tokens':512},
])
def test_invalid_or_competing_controls_are_rejected(changes):
    p=profile();p['kvmem']['offline_prefill_coalescing']=True;p['kvmem'].update(changes)
    with pytest.raises(ValueError): options(p['kvmem'])


@pytest.mark.parametrize('baseline,maximum',[(0,4096),(2048,8192),(2048,2048),(2049,4096)])
def test_launch_requires_a_bounded_aligned_coalescing_range(baseline,maximum):
    p=profile();p['prefill_chunk_tokens']=baseline
    p['kvmem'].update(offline_prefill_coalescing=True,offline_prefill_max_tokens=maximum)
    with pytest.raises(ValueError):
        build_launch(p,Path('/tmp'),Path('/tmp/unused-coalescing-test'),check_paths=False)


def test_disabled_coalescing_preserves_small_generation_compatibility():
    from vllm_kvmem.config import KVMemConfig
    assert not RuntimeOptions(memory=KVMemConfig(gen_budget=1024)).offline_prefill_coalescing

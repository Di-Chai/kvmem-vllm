"""Native producer coverage must follow per-layer pressure, not decode policy."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.native_attention import statistic_layer_names


def test_independent_pressure_covers_all_target_layers_without_changing_spread3():
    names=tuple(f'language_model.model.layers.{n}.self_attn' for n in range(3,64,4))
    options=NS(action_layers='spread3')
    runner=NS(_kvmem_layers=names)
    legacy,decode=statistic_layer_names(runner,options)
    assert legacy==decode==(names[0],names[7],names[15])
    runner._kvmem_layerwise_pressure=NS(names=names)
    producers,actual_decode=statistic_layer_names(runner,options)
    assert producers==names and len(producers)==16
    assert actual_decode==decode and options.action_layers=='spread3'


@pytest.mark.parametrize('alter', ['reordered','draft','missing'])
def test_pressure_cannot_silently_capture_a_different_layer_domain(alter):
    names=tuple(f'language_model.model.layers.{n}.self_attn' for n in range(3,64,4))
    bad=tuple(reversed(names)) if alter=='reordered' else names+('mtp.layers.0.self_attn',) if alter=='draft' else names[:-1]
    runner=NS(_kvmem_layers=names,_kvmem_layerwise_pressure=NS(names=bad))
    with pytest.raises(RuntimeError,match='ordered target layers'):
        statistic_layer_names(runner,NS(action_layers='spread3'))

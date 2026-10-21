"""Use the exact metadata/acceptance oracle with the current fused scorer."""
import pytest
import torch

from test_action_metadata_upload_gpu import (
    test_three_layers_recursive_scoring_and_confirmed_prefix_equal_to_legacy as _qualify,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Explicit fused metadata CUDA qualification')
@pytest.mark.parametrize('q_dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_fused_three_layers_two_owners_and_partial_acceptance(q_dtype):
    _qualify(q_dtype, fused=True)

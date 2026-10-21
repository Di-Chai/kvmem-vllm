"""No-GPU gate for reviewed SASS evidence before timing the candidate."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/benchmark_native_prefill_readonly.py'
spec = importlib.util.spec_from_file_location('benchmark_native_prefill_readonly', SCRIPT)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def fixture(tmp_path):
    identities, proof = {}, dict(schema=1, fp_body_unchanged=True)
    for side, registers, loads in [('reference', 255, 4), ('candidate', 254, 2)]:
        sass = tmp_path / (side + '.sass'); sass.write_text('SASS evidence ' + side)
        identities[side] = dict(library_sha256=side, overlay=side)
        proof[side] = dict(library_sha256=side, overlay=side, registers=registers,
            stack_bytes=32, loop_metadata_loads=loads, loop_local_loads=3,
            sass_file=str(sass), sass_sha256=module.sha(sass))
    return identities, proof


def test_reviewed_improvement_gate(tmp_path):
    identities, proof = fixture(tmp_path)
    assert module.validate_sass_proof(proof, identities)


@pytest.mark.parametrize('key,value', [('registers', 256), ('stack_bytes', 64),
    ('loop_local_loads', 4), ('loop_metadata_loads', 4), ('loop_metadata_loads', 5)])
def test_no_improvement_or_spill_regression_stops_before_timing(tmp_path, key, value):
    identities, proof = fixture(tmp_path); proof['candidate'][key] = value
    with pytest.raises(ValueError, match='stop before timing'):
        module.validate_sass_proof(proof, identities)


@pytest.mark.parametrize('kind', ['source', 'library', 'arithmetic', 'sass', 'count'])
def test_proof_tampering_or_missing_assertion_rejected(tmp_path, kind):
    identities, proof = fixture(tmp_path)
    if kind == 'source': proof['candidate']['overlay'] = 'changed'
    elif kind == 'library': proof['reference']['library_sha256'] = 'changed'
    elif kind == 'arithmetic': proof['fp_body_unchanged'] = False
    elif kind == 'sass': Path(proof['candidate']['sass_file']).write_text('changed')
    else: proof['candidate']['registers'] = -1
    with pytest.raises(ValueError): module.validate_sass_proof(proof, identities)

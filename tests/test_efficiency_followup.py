import pytest

import recall_comparison_protocol as p
from run_layerwise_efficiency_followup import choose_source


def queue(root,compiled=True):
    root.mkdir()
    p.save(root/'status.json',dict(status='completed' if compiled else 'failed'))
    for name in ('compiled','fast_metadata'):
        p.save(root/name/'status.json',dict(status='completed'))
        p.save(root/name/'equivalence.json',dict(passed=compiled if name=='compiled' else True,cases=8))
        p.save(root/name/'native-reuse-audit.json',dict(passed=True))


def test_full_compilation_equivalence_is_required_for_compiled_source(tmp_path):
    root=tmp_path/'queue';queue(root)
    assert choose_source(root)[0]==root/'compiled'


def test_failed_compilation_results_remain_evidence_and_do_not_enter_candidate(tmp_path):
    root=tmp_path/'queue';queue(root,compiled=False)
    original=(root/'compiled/equivalence.json').read_bytes()
    source,reason=choose_source(root)
    assert source==root/'fast_metadata' and 'changed exact results' in reason
    assert (root/'compiled/equivalence.json').read_bytes()==original


@pytest.mark.parametrize('fault',['running','technical','partial','fallback','eager_not_accepted'])
def test_followup_does_not_skip_technical_failures_or_incomplete_acceptance(tmp_path,fault):
    root=tmp_path/'queue';queue(root,compiled=False)
    if fault=='running':p.save(root/'status.json',dict(status='running'))
    elif fault=='technical':p.save(root/'compiled/status.json',dict(status='failed'))
    elif fault=='partial':p.save(root/'compiled/equivalence.json',dict(passed=False,cases=7))
    elif fault=='fallback':p.save(root/'compiled/native-reuse-audit.json',dict(passed=False))
    else:p.save(root/'fast_metadata/equivalence.json',dict(passed=False,cases=8))
    with pytest.raises(ValueError):choose_source(root)

import ast

import pytest

import recall_comparison_protocol as p
import run_layerwise_query_efficiency as driver


def component(root, count=16, marker=''):
    root.mkdir()
    (root/'kernel.py').write_text('frozen byte oracle\n')
    p.save(root/'source-proof.json', {'kernel.py':p.sha(root/'kernel.py')})
    p.save(root/'plan.json', dict(source_proof_sha256=p.sha(root/'source-proof.json')))
    xml='<testsuites><testsuite>'+''.join(
        f'<testcase name="case{i}">{marker if i==0 else ""}</testcase>' for i in range(count)
    )+'</testsuite></testsuites>'
    (root/'gpu-tests.xml').write_text(xml)
    p.save(root/'gpu-audit.json', dict(passed=True,returncode=0,
        counts=dict(tests=16,failures=0,errors=0,skipped=0),
        source_proof_sha256=p.sha(root/'source-proof.json'),plan_sha256=p.sha(root/'plan.json'),
        tests_sha256=p.sha(root/'gpu-tests.xml')))


def test_only_complete_unskipped_gpu_evidence_can_release_full_input_tests(tmp_path):
    root=tmp_path/'components';component(root)
    assert driver.component_acceptance(root)['passed']


@pytest.mark.parametrize('fault',['short','skip','failure','error','source','xml','exit','manifest'])
def test_component_gate_rejects_false_green_or_changed_evidence(tmp_path,fault):
    root=tmp_path/'components'
    marker={'skip':'<skipped/>','failure':'<failure/>','error':'<error/>'}.get(fault,'')
    component(root,count=12 if fault=='short' else 16,marker=marker)
    if fault=='source':(root/'kernel.py').write_text('changed arithmetic\n')
    elif fault=='xml':(root/'gpu-tests.xml').write_text('<testsuites/>')
    elif fault=='exit':
        audit=p.read(root/'gpu-audit.json');audit['returncode']=1;p.save(root/'gpu-audit.json',audit)
    elif fault=='manifest':p.save(root/'source-proof.json',{})
    with pytest.raises(ValueError):driver.component_acceptance(root)


def test_new_candidate_never_overwrites_an_existing_attempt(tmp_path):
    root=tmp_path/'completed';root.mkdir();answer=root/'response.json';answer.write_text('first answer')
    with pytest.raises(ValueError,match='new version'):
        driver.prepare(root,tmp_path/'absent-source',tmp_path/'absent-components')
    assert answer.read_text()=='first answer'


def test_missing_gpu_validation_does_not_create_an_inference_candidate(monkeypatch,tmp_path):
    root=tmp_path/'new';components=tmp_path/'components';component(components,marker='<skipped/>')
    monkeypatch.setattr(driver,'source_audit',lambda source:[])
    with pytest.raises(ValueError):driver.prepare(root,tmp_path/'source',components)
    assert not root.exists()


def test_running_queue_is_not_released_by_an_empty_gpu(tmp_path):
    root=tmp_path/'queue';root.mkdir();p.save(root/'status.json',dict(status='running'))
    with pytest.raises(RuntimeError,match='still active'):driver.released(root)


def test_serve_patch_only_adds_explicit_opt_in_flags(tmp_path):
    path=tmp_path/'serve.py'
    path.write_text("def main(parser, argv, plan):\n"
        "    args = parser.parse_args(argv)\n"
        "    if args.layerwise_query_diagnostic:\n"
        "        if args.layerwise_query_optimized:\n"
        "            plan['environment']['KVMEM_LAYERWISE_QUERY_OPTIMIZED'] = '1'\n")
    driver.serve_flags(path)
    ast.parse(path.read_text())
    assert '--layerwise-packed-assembly' in path.read_text()
    assert '--layerwise-prepared-index' in path.read_text()
    assert '--layerwise-batched-reads' in path.read_text()
    saved=path.read_bytes()
    with pytest.raises(ValueError):driver.serve_flags(path)
    assert path.read_bytes()==saved


@pytest.mark.parametrize('blocks,layers,expected',[(32,17,256),(32,1,32),(8,17,136),(256,17,256)])
def test_single_layer_batch_stays_within_existing_full_frame_byte_allowance(blocks,layers,expected):
    from vllm_kvmem.serving.layerwise_pressure_efficiency import layer_read_blocks
    actual=layer_read_blocks(blocks,layers)
    assert actual==expected and actual<=blocks*layers


def test_larger_read_groups_preserve_every_dense_destination_and_skip():
    from types import SimpleNamespace as NS
    from vllm_kvmem.serving.layerwise_pressure_efficiency import assembly_groups
    records=[NS(history_begin=i*64,count=32) for i in range(1152)]
    records[-1].count=17
    skip={records[3].history_begin,records[515].history_begin}
    def destinations(size):
        groups,total=assembly_groups(records,size,skip)
        expanded=[]
        for cursor,group in groups:
            for record in group:
                expanded.append((record.history_begin,record.count,cursor));cursor+=record.count
        return expanded,total
    assert destinations(32)==destinations(256)

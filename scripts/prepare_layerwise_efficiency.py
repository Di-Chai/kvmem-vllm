"""Freeze optimized layerwise runs against the existing original sample list."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import recall_comparison_protocol as p
import run_layerwise_query_diagnostic as driver

REPO = Path(__file__).resolve().parents[1]
BASE = REPO/'artifacts/layerwise-meank-balanced26-20261102-r2'


def prepare(root, stage, diagnostics=False):
    driver.prepare(root, 'balanced26')
    runtime = root/'runtime-v1'
    shutil.copy2(REPO/'src/vllm_kvmem/layerwise_query.py',runtime/'src/vllm_kvmem/layerwise_query.py')
    shutil.copy2(REPO/'scripts/run_layerwise_query_diagnostic.py',runtime/'scripts/run_layerwise_query_diagnostic.py')
    serve = runtime/'src/vllm_kvmem/serve.py'
    text=serve.read_text()
    anchor='    args = parser.parse_args(argv)\n'
    if text.count(anchor)!=1:raise ValueError('Frozen CLI parse hook changed')
    text=text.replace(anchor,
        "    parser.add_argument('--layerwise-query-optimized', action='store_true')\n"
        "    parser.add_argument('--layerwise-query-full-diagnostics', action='store_true')\n"+anchor)
    anchor="        plan['environment']['KVMEM_LAYERWISE_QUERY_MODE'] = args.layerwise_query_diagnostic\n"
    if text.count(anchor)!=1:raise ValueError('Frozen layerwise launch hook changed')
    text=text.replace(anchor,anchor+
        "        if args.layerwise_query_optimized:\n"
        "            plan['environment']['KVMEM_LAYERWISE_QUERY_OPTIMIZED'] = '1'\n"
        "        if args.layerwise_query_full_diagnostics:\n"
        "            plan['environment']['KVMEM_LAYERWISE_QUERY_DIAGNOSTICS'] = '1'\n")
    serve.write_text(text)
    fixture=p.read(root/'fixture.json')
    profile=p.read(root/'profile.json')
    original_profile=p.read(BASE/'profile.json')
    profile.update(port=19023 if stage=='qualification' else 19024,
                   served_model_name='layerwise-efficient-'+stage)
    differences={k:dict(before=original_profile.get(k),after=profile.get(k))
                 for k in set(original_profile)|set(profile) if original_profile.get(k)!=profile.get(k)}
    if set(differences)-{'port','served_model_name'}:
        raise ValueError('Non-efficiency model/runtime controls changed: '+str(differences))
    if stage=='qualification':
        ids=('niah_single_1-0000','niah_multikey_3-0000','qa_2-0000')
        fixture['rows']=[r for r in fixture['rows'] if r['id'] in ids]
        fixture.update(cohort='efficiency3',cases=3,qualification_ids=list(ids),
            diagnostic_scope='Same three previously completed original inputs; optimize execution only.')
    elif stage=='formal':
        import numpy as np
        from tokenizers import Tokenizer
        from recall_answer_prefix import append_official_prefix,token_bytes,POLICY
        source=REPO/'artifacts/value-efficiency-nothinking-20261101-r1/eight-way-campaign-v1'
        original=p.read(source/'fixture.json')
        previous={r['id']:r for r in p.read(REPO/'artifacts/deletion-prefix-130-20261102-r1/campaign-v1/fixture.json')['rows']}
        tokenizer=Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
        rows=[]
        for raw in original['rows']:
            row=deepcopy(raw);sid=row['id'];src=source/'dataset/inputs'/(sid+'.i32')
            if p.sha(src)!=row['served_sha256']:raise ValueError('Original RULER input changed')
            tokens=np.fromfile(src,dtype='<i4').tolist()
            tokens,proof=append_official_prefix(tokens,row['answer_prefix'],tokenizer)
            out=root/'inputs'/(sid+'.i32');out.write_bytes(token_bytes(tokens))
            row.update(served_input_tokens=len(tokens),served_sha256=p.sha(out),prefix_proof=proof)
            if sid in previous and (row['served_sha256']!=previous[sid]['served_sha256'] or
                                    row['served_query']!=previous[sid]['served_query']):
                raise ValueError('Prefix rendering differs from earlier same-sample protocol')
            rows.append(row)
        fixture.update(cohort='ruler260',cases=260,rows=rows,samples_per_task=20,pilot_samples_per_task=20,
            qualification_ids=[r['id'] for r in rows],answer_prefix_policy=POLICY,
            source_fixture=str(source/'fixture.json'),source_fixture_sha256=p.sha(source/'fixture.json'),
            diagnostic_scope='Original full 13x20 RULER list; official answer prefix, identical previous per-task sampling and budgets.',
            diagnostic_limits=['Optimized serial eager layerwise Mean-K; historical profiles with compiled/eight-way execution are not controlled performance baselines.'])
        import ruler_protocol
        ruler_protocol.validate_cohort(rows,20)
    else:raise ValueError('Unknown stage')
    fixture.update(layerwise_optimized=True,layerwise_full_diagnostics=diagnostics)
    p.save(root/'fixture.json',fixture);p.save(root/'profile.json',profile)
    proof={str(f.relative_to(runtime)):p.sha(f) for f in runtime.rglob('*')
           if f.is_file() and '__pycache__' not in f.parts}
    p.save(root/'runtime-proof.json',proof)
    p.save(root/'prepared.json',dict(at=time.time(),ids=[r['id'] for r in fixture['rows']],
        expected_runs=len(fixture['rows']),profile_sha256=p.sha(root/'profile.json'),
        fixture_sha256=p.sha(root/'fixture.json'),runtime_proof_sha256=p.sha(root/'runtime-proof.json')))
    p.save(root/'control-parity.json',dict(reference=str(BASE),allowed_endpoint_differences=differences,
        kv_and_sampling_unchanged=True,optimization_only=True,diagnostics=diagnostics,
        source_inputs_freshly_run=True,previous_answers_reused=False))
    p.save(root/'status.json',dict(status='prepared',expected=len(fixture['rows']),at=time.time()))
    driver.verify(root)
    print(json.dumps(dict(root=str(root),stage=stage,cases=len(fixture['rows']),diagnostics=diagnostics)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=('qualification','formal'),required=True)
    parser.add_argument('--diagnostics',action='store_true')
    args=parser.parse_args();prepare(args.root.resolve(),args.stage,args.diagnostics)

"""Verify exact retrieval/output equivalence before an efficiency campaign."""
import argparse
from pathlib import Path
import statistics

import recall_comparison_protocol as p
from run_layerwise_query_diagnostic import verify


def analyze(root,reference):
    verify(root);fixture=p.read(root/'fixture.json');checks=[]
    def elapsed(evidence):
        events=evidence['events']
        return next(e['at'] for e in events if e['event']=='completed')-next(e['at'] for e in events if e['event']=='armed')
    for row in fixture['rows']:
        before=reference/'formal/sequential/cases'/row['id']
        after=root/'formal/sequential/cases'/row['id']
        a,b=p.read(before/'evidence.json'),p.read(after/'evidence.json')
        ar,br=p.read(before/'response.json'),p.read(after/'response.json')
        request=p.read(after/'request.json');previous_request=p.read(before/'request.json')
        if request['sampling']!=previous_request['sampling'] or request['query_span']!=previous_request['query_span']:
            raise ValueError('Generation sampling or Query bounds changed')
        if request['input_sha256']!=previous_request['input_sha256']:raise ValueError('Input tokens changed')
        same_blocks=(len(a['layers'])==len(b['layers'])==16 and
            all(x['selected_starts']==y['selected_starts'] for x,y in zip(a['layers'],b['layers'])))
        same_queries=(all(x['q_sha256']==y['q_sha256'] for x,y in zip(a['layers'],b['layers']))
                      if fixture.get('layerwise_full_diagnostics') else None)
        check=dict(id=row['id'],same_blocks=same_blocks,same_queries=same_queries,
            same_output_tokens=ar['output_ids']==br['output_ids'],before_seconds=elapsed(a),after_seconds=elapsed(b),
            before_case_seconds=ar['seconds'],after_case_seconds=br['seconds'],
            source_files={str(before/n):p.sha(before/n) for n in ('request.json','evidence.json','response.json')},
            result_files={str(after/n):p.sha(after/n) for n in ('request.json','evidence.json','response.json')})
        checks.append(check)
    passed=bool(checks) and all(r['same_blocks'] and r['same_output_tokens'] and r['same_queries'] is not False for r in checks)
    report=dict(passed=passed,reference=str(reference),cases=len(checks),checks=checks,
        median_before_seconds=statistics.median(r['before_seconds'] for r in checks),
        median_after_seconds=statistics.median(r['after_seconds'] for r in checks),
        timing_scope='Wall time from arming through last layer submission; optimized versions with the final completion fence include its drain. Whole-case response times are also retained.',
        all_inputs_sampling_query_bounds_unchanged=True)
    p.save(root/'qualification-audit.json',report)
    if not passed:raise ValueError('Efficiency change altered selected blocks, Q bytes or output tokens')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True);args=parser.parse_args()
    result=analyze(args.root.resolve(),args.reference.resolve())
    print({k:v for k,v in result.items() if k!='checks'})

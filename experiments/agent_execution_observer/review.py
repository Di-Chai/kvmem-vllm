"""Export every decision for review; score separately supplied reference anchors.

The supplied pilot anchors are model-assisted, not human ground truth.
Neither references nor later trace messages are fed into Observer inference.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

from audit import normalize_boundaries


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,
                        help='New directory; original results are never modified.')
    parser.add_argument('--activity-reference',type=Path)
    parser.add_argument('--boundary-reference',type=Path)
    args=parser.parse_args()
    predictions=normalize_boundaries([json.loads(line) for line in (args.results/'decisions.jsonl').open()])
    source={(s['window_id'],s['step']):s for s in
            (json.loads(line) for line in args.dataset.open())}
    report=dict(reference_type='provisional_model_assisted_review',human_gold_accuracy=None,
                human_gold_boundary_f1=None,reference_files={},steps=len(predictions))
    report['boundary_normalizations']=[dict(window_id=p['window_id'],step=p['step'],
            reason=p['boundary_normalization']) for p in predictions if p['boundary_normalization']]
    by_step={p['step']:p for p in predictions}
    if len(by_step)!=len(predictions) and (args.activity_reference or args.boundary_reference):
        raise ValueError('Reference scoring requires a single continuous trajectory')
    for name,path in (('activity',args.activity_reference),('boundary',args.boundary_reference)):
        if path is None:continue
        report['reference_files'][name]=dict(path=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        labels=read(path)
        if name=='activity':
            scored=[dict(step=r['step'],accepted=r['accepted_activity'],
                         predicted=by_step[r['step']]['activity'],
                         agrees=by_step[r['step']]['activity'] in r['accepted_activity'])
                    for r in labels['labels'] if r['step'] in by_step]
            report['activity']=dict(anchors=len(scored),agree=sum(r['agrees'] for r in scored),
                    provisional_agreement=sum(r['agrees'] for r in scored)/len(scored),rows=scored)
        else:
            scored=[dict(**r,predicted=by_step[r['current']]['stage_boundary'])
                    for r in labels['pairs'] if r['current'] in by_step and r['previous'] in by_step]
            decided=[r for r in scored if r['predicted'] is not None]
            agrees=sum(r['boundary']==r['predicted'] for r in decided)
            report['boundary']=dict(anchors=len(scored),decided=len(decided),agree=agrees,
                    abstentions=len(scored)-len(decided),
                    provisional_agreement_all_anchors=agrees/len(scored) if scored else None,
                    provisional_agreement_decided=agrees/len(decided) if decided else None,
                    reference_boundary_predicted_continuation=sum(r['boundary'] and r['predicted'] is False for r in scored),
                    reference_continuation_predicted_boundary=sum(not r['boundary'] and r['predicted'] is True for r in scored),
                    rows=scored)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'provisional-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    fields=['window_id','step','stage_id','raw_stage_boundary','stage_boundary','boundary_normalization',
            'activity','outcome','seconds','evidence_source','evidence_quote','current_step_excerpt']
    fields += [f'{qid}_{key}' for qid in ('activity','stage_link','outcome','evidence')
               for key in ('label_mass','argmax_is_label')]
    with (args.output/'review.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for p in predictions:
            evidence=p.get('selected_evidence') or {}
            diagnostic_fields={f'{qid}_{key}':p['diagnostics'][qid][key]
                               for qid in p['diagnostics'] for key in ('label_mass','argmax_is_label')}
            writer.writerow({k:p[k] for k in fields if k in p} | diagnostic_fields | dict(
                evidence_source=json.dumps(evidence.get('source'),ensure_ascii=False),
                evidence_quote=evidence.get('quote',''),
                current_step_excerpt=source[p['window_id'],p['step']]['current_step']))
    rows=[]
    for p in predictions:
        evidence=p.get('selected_evidence') or {}
        src=source[p['window_id'],p['step']]
        values=[p['window_id'],p['step'],p['stage_id'],
                f'{p["raw_stage_boundary"]} → {p["stage_boundary"]}',p['activity'],p['outcome'],
                f'{p["seconds"]:.3f}',json.dumps(evidence,ensure_ascii=False)]
        cells=''.join('<td>'+html.escape(str(v))+'</td>' for v in values)
        rows.append('<tr>'+cells+'<td><details><summary>Original visible Step</summary><pre>'+
                    html.escape(src['current_step'])+'</pre></details></td></tr>')
    document='''<!doctype html><meta charset="utf-8"><title>System One stage pilot review</title>
<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:8px;vertical-align:top}pre{white-space:pre-wrap;max-width:680px}
td:nth-child(8){min-width:260px}</style><h1>System One stage pilot</h1>
<p>Native model decisions, with original visible evidence. Provisional model-assisted references
are not human ground truth. Exact evidence location does not establish evidential support.</p>
<table><thead><tr><th>Window</th><th>Step</th><th>Stage</th><th>Boundary</th><th>Activity</th>
    <th>Outcome</th><th>Seconds</th><th>Selected evidence with coordinates</th><th>Input</th></tr></thead><tbody>'''
    (args.output/'review.html').write_text(document+''.join(rows)+'</tbody></table>')
    print(json.dumps({k:v for k,v in report.items() if k not in ('activity','boundary')},ensure_ascii=False))
    for kind in ('activity','boundary'):
        if kind in report:print(kind,{k:v for k,v in report[kind].items() if k!='rows'})


if __name__=='__main__':main()

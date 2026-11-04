"""Post-hoc gold coverage and paired answers; never supplies gold to selection."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

import numpy as np
from tokenizers import Tokenizer

from analyze_ruler_failures import PromptOffsets, coverage
import recall_comparison_protocol as p
import ruler_protocol


def analyze(root):
    fixture, profile=p.read(root/'fixture.json'),p.read(root/'profile.json')
    tokenizer=Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    mapping=PromptOffsets(tokenizer)
    scorer=ruler_protocol.load_scorer(root/'upstream')
    rows=[]
    for row in fixture['rows']:
        saved={arm:root/'formal'/arm/'cases'/row['id'] for arm in ('global','sequential')}
        if not all((directory/'result.json').exists() for directory in saved.values()):continue
        input_file=root/'inputs'/(row['id']+'.i32')
        if p.sha(input_file)!=row['served_sha256']:
            raise ValueError('Frozen original input bytes changed')
        ids=np.fromfile(input_file,dtype='<i4').tolist()
        text=tokenizer.decode(ids,skip_special_tokens=False)
        positions=mapping.positions(ids,text)
        gold=set(row['outputs'])
        hits=[m for m in re.finditer(r'One of the special magic (?:numbers|uuids) for (.+?) is: ([^.\n]+)\.',text)
              if m.group(2) in gold and m.group(1) in row['question']]
        if len(hits)!=1:raise ValueError('Expected exactly one original complete target record')
        hit=hits[0]
        spans={name:positions(*span) for name,span in
               [('key',hit.span(1)),('value',hit.span(2)),('sentence',hit.span())]}
        record=dict(id=row['id'],task=row['task'],spans=spans,arms={})
        for arm,directory in saved.items():
            result,evidence=p.read(directory/'result.json'),p.read(directory/'evidence.json')
            if p.sha(directory/'response.json')!=result['response_sha256'] or p.sha(directory/'evidence.json')!=result['evidence_sha256']:
                raise ValueError('Saved original response/evidence changed')
            rescored=p.score_case(row,p.read(directory/'response.json'),scorer,starts_in_thinking=False)
            if rescored['official_case_score']!=result['official_case_score']:
                raise ValueError('Official rescore differs from the saved original score')
            layer_rows=[]
            for layer in evidence['layers']:
                ranges=[(start,32) for start in layer['selected_starts']]
                frozen_ranges=[(start,32) for start in layer['frozen_layer_selected_starts']]
                layer_rows.append(dict(layer=layer['layer'],
                    coverage={name:coverage(begin,end,ranges) for name,(begin,end) in spans.items()},
                    frozen_q_coverage={name:coverage(begin,end,frozen_ranges) for name,(begin,end) in spans.items()},
                    q_cosine=layer['query_cosine'],q_relative_l2=layer['query_relative_l2'],
                    fresh_frozen_overlap=layer['fresh_frozen_set_overlap']))
            record['arms'][arm]=dict(score=result['official_case_score'],answer=result['answer'],
                seconds=result['seconds'],ttft_seconds=result['ttft_seconds'],
                completion_tokens=result['completion_tokens'],finish_reason=result['finish_reason'],
                full_value_layers=sum(l['coverage']['value']['state']=='full' for l in layer_rows),
                full_sentence_layers=sum(l['coverage']['sentence']['state']=='full' for l in layer_rows),
                frozen_q_full_value_layers=sum(l['frozen_q_coverage']['value']['state']=='full' for l in layer_rows),
                selected_union_tokens=32*len({s for layer in evidence['layers'] for s in layer['selected_starts']}),
                total_layer_charged_tokens=sum(layer['charged_tokens'] for layer in evidence['layers']),
                layers=layer_rows)
        rows.append(record)
    summary={arm:dict(completed=len(rows),score=statistics.fmean(r['arms'][arm]['score'] for r in rows) if rows else None,
        truncated=sum(r['arms'][arm]['finish_reason']=='length' for r in rows),
        median_seconds=statistics.median(r['arms'][arm]['seconds'] for r in rows) if rows else None,
        full_value_layer_fraction=statistics.fmean(r['arms'][arm]['full_value_layers']/16 for r in rows) if rows else None,
        frozen_q_full_value_layer_fraction=statistics.fmean(r['arms'][arm]['frozen_q_full_value_layers']/16 for r in rows) if rows else None)
        for arm in ('global','sequential')}
    compare=lambda r:r['arms']['sequential']['score']-r['arms']['global']['score']
    output=dict(scope='Failure-enriched ten-case diagnosis; fresh official answer-prefix non-thinking protocol.',
        expected=10,completed_pairs=len(rows),summary=summary,
        paired=dict(wins=sum(compare(r)>0 for r in rows),losses=sum(compare(r)<0 for r in rows),ties=sum(compare(r)==0 for r in rows)),
        per_task={task:{arm:statistics.fmean(r['arms'][arm]['score'] for r in rows if r['task']==task)
                  for arm in ('global','sequential')} for task in dict.fromkeys(r['task'] for r in rows)},rows=rows,
        per_layer={arm:[dict(layer=i,
            mean_query_relative_l2=statistics.fmean(r['arms'][arm]['layers'][i]['q_relative_l2'] for r in rows),
            mean_query_cosine=statistics.fmean(r['arms'][arm]['layers'][i]['q_cosine'] for r in rows),
            mean_fresh_frozen_set_overlap=statistics.fmean(r['arms'][arm]['layers'][i]['fresh_frozen_overlap'] for r in rows))
            for i in range(16)] if rows else [] for arm in ('global','sequential')},
        official_rescore_verified=True,
        budget_semantics='Each of sixteen target full-attention layers holds <=36864 tokens; sum of per-layer token slots <=589824 in both arms. Logical union across layers may exceed 36864 without increasing stored KV vectors.',
        limits=fixture['diagnostic_limits']+[
            'Qwen has sixteen full-attention layers interleaved with forty-eight linear-attention layers; linear recurrent states and archived historical K/V remain those of compressed prefill.',
            'Frozen-Q layerwise sets are offline selection diagnostics, not a third set of generated answers.'])
    p.save(root/'analysis/comparison.json',output)
    with (root/'analysis/per-case.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['id','task','global_score','sequential_score','global_full_value_layers',
                         'sequential_full_value_layers','sequential_frozen_q_full_value_layers'])
        for r in rows:
            a,b=r['arms']['global'],r['arms']['sequential']
            writer.writerow([r['id'],r['task'],a['score'],b['score'],a['full_value_layers'],
                             b['full_value_layers'],b['frozen_q_full_value_layers']])
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','limits','per_layer')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    analyze(parser.parse_args().root.resolve())

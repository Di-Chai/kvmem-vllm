"""Audit common-prefix GPU captures before any pressure selection/assembly."""
import argparse
import gzip
import itertools
import json
from pathlib import Path

import torch


def read(p):return json.loads(p.read_text())


def analyze(root,output):
    fixture=read(root/'fixture.json')
    prefixes=[]
    for row in fixture['rows']:
        with gzip.open(row['source'],'rt') as f:source=json.load(f)
        prefixes.append(source['tokens'][:96])
    assert len(prefixes)==8 and all(p==prefixes[0] for p in prefixes)
    capture=root/'run/server/capture'
    model={p.stem:torch.load(p,map_location='cpu',weights_only=True)
           for p in (capture/'first_model_prefix').glob('*.pt')}
    score={p.stem:read(p) for p in (capture/'first_pressure_repro').glob('*.json')}
    assert len(model)==len(score)==8 and set(model)==set(score)
    assert all(s['repeated_fixed_state_equal'] for s in score.values())
    metadata={rid:read(capture/'first_model_prefix'/(rid+'.json')) for rid in model}
    pairs=[]
    for a,b in itertools.combinations(sorted(model),2):
        metrics={}
        for name in ('raw','values'):
            x,y=model[a][name].float(),model[b][name].float()
            assert x.shape==y.shape==(16,96,4,256)
            metrics[name]=[dict(equal_fraction=float((u==v).float().mean()),
                relative_l2=float((u-v).norm()/u.norm().clamp_min(1e-12)),
                max_abs=float((u-v).abs().max())) for u,v in zip(x,y)]
        pairs.append(dict(a=a,b=b,same_initial_batch=metadata[a]['batch_requests']==metadata[b]['batch_requests'],
            batch_sizes=[len(metadata[r]['batch_requests']) for r in (a,b)],
            batch_positions=[metadata[r]['batch_index'] for r in (a,b)],layers=metrics))
    differences=[p for p in pairs if any(m['equal_fraction']<1 for m in p['layers']['raw'])]
    result=dict(cases=8,identical_causal_prefix_tokens=96,frozen_selector_repeat_equal=8,
        pair_count=len(pairs),pairs_with_upstream_key_difference=len(differences),pairs=pairs,
        metadata=metadata,
        conclusion=('Identical causal token prefixes already have different original BF16 GPU projection '
                    'operands before archive packing and before the first pressure assembly; the same '
                    'frozen accumulated scores select the exact same blocks twice. '
                    'This localizes a source of multi-owner non-reproducibility upstream of recall. '
                    'It does not identify which quantized GEMM, recurrent or attention kernel causes it, '
                    'or prove that all later divergence is explained by this source.' if differences else
                    'No common-prefix GPU difference observed; further first-score capture is needed.'),
        scope='Single preserved diagnostic run; no response selection or resampling; not throughput evidence.')
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','metadata')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.root,a.output)

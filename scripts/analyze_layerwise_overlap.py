"""Read-only overlap of historical block identities within each completed sample."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import time

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    return dict(mean=statistics.fmean(values),median=statistics.median(values),
                minimum=min(values),maximum=max(values))


def matrix(sets,kind='overlap'):
    out=np.eye(len(sets),dtype=np.float64)
    for i,j in itertools.combinations(range(len(sets)),2):
        intersection=len(sets[i]&sets[j])
        denominator=min(len(sets[i]),len(sets[j])) if kind=='overlap' else len(sets[i]|sets[j])
        out[i,j]=out[j,i]=intersection/denominator if denominator else 1.
    return out


def analyze(root,output):
    output.mkdir(parents=True,exist_ok=False)
    fixture=json.loads((root/'fixture.json').read_text())
    config=json.loads((root/'profile.json').read_text())['kvmem']
    selected=sorted((root/'formal/sequential/cases').glob('*/result.json'))
    rows=[];matrices=[];whole_matrices=[];jaccards=[];names=None;sources=[]
    upper=np.triu_indices(16,1)
    for path in selected:
        result=json.loads(path.read_text());ep=path.parent/'evidence.json'
        if digest(ep)!=result['evidence_sha256']:
            raise ValueError('Original selected-block evidence changed')
        evidence=json.loads(ep.read_text());layers=evidence['layers']
        if [l['layer'] for l in layers]!=list(range(16)):
            raise ValueError('Missing or unordered actual full-attention layers')
        if names is None:names=[l['layer_name'] for l in layers]
        if names!=[l['layer_name'] for l in layers]:raise ValueError('Layer identities changed')
        query=next(e for e in evidence['refresh'] if e['selection']=='query')
        armed=next(e for e in evidence['events'] if e['event']=='armed')
        fixed={32*i for i in range(config['sink_blocks'])}
        fixed|={32*i for i in query['mandatory_fit']['hard_blocks']}
        fixed|={start for start,count in query['selected_history_ranges'] if count!=32}
        if len(fixed)!=armed['fixed_blocks']:
            raise ValueError('Could not reconstruct exact mandatory block identities')
        whole=[set(l['selected_starts']) for l in layers]
        if any(len(s)!=len(l['selected_starts']) or not fixed<=s for s,l in zip(whole,layers)):
            raise ValueError('Duplicate blocks or missing fixed content')
        optional=[s-fixed for s in whole]
        m=matrix(optional);w=matrix(whole);j=matrix(optional,'jaccard')
        assert np.allclose(m,m.T) and np.all(np.diag(m)==1) and np.all((m>=0)&(m<=1))
        matrices.append(m);whole_matrices.append(w);jaccards.append(j)
        union=set.union(*optional);common=set.intersection(*optional)
        frequency=Counter(start for group in optional for start in group)
        histogram=dict(Counter(frequency.values()))
        assert sum(histogram.values())==len(union)
        assert sum(n*count for n,count in histogram.items())==sum(map(len,optional))
        row=dict(id=result['id'],task=result['task'],score=result['official_case_score'],
            fixed_blocks=len(fixed),blocks_per_layer=sorted(set(map(len,whole))),
            optional_blocks_per_layer=sorted(set(map(len,optional))),
            all_pair_overlap=float(m[upper].mean()),adjacent_overlap=float(np.diag(m,1).mean()),
            first_last_overlap=float(m[0,-1]),all_pair_jaccard=float(j[upper].mean()),
            union_optional_blocks=len(union),union_total_blocks=len(union)+len(fixed),
            union_total_tokens=32*(len(union)+len(fixed)),common_optional_blocks=len(common),
            common_total_blocks=len(common)+len(fixed),frequency_histogram=histogram,
            selected_layer_token_slots=sum(len(s)*32 for s in whole),
            gap_overlap={str(gap):float(np.diag(m,gap).mean()) for gap in range(1,16)},
            random_reference=min(map(len,optional))/(armed['candidate_blocks']-len(fixed)))
        rows.append(row)
        sources.append(dict(id=result['id'],result=str(path),result_sha256=digest(path),
                            evidence=str(ep),evidence_sha256=digest(ep)))
    if not rows:raise ValueError('No completed samples')
    averaged=np.mean(matrices,axis=0);whole=np.mean(whole_matrices,axis=0)
    per_task={}
    for task in dict.fromkeys(r['task'] for r in fixture['rows']):
        group=[r for r in rows if r['task']==task]
        if group:
            per_task[task]=dict(cases=len(group),**{key:statistics.fmean(r[key] for r in group)
                for key in ('all_pair_overlap','adjacent_overlap','first_last_overlap',
                            'union_total_tokens','common_optional_blocks')})
    aggregates={key:summary([r[key] for r in rows]) for key in (
        'all_pair_overlap','adjacent_overlap','first_last_overlap','all_pair_jaccard',
        'union_total_tokens','union_optional_blocks','common_optional_blocks','common_total_blocks','random_reference')}
    physical=[int(name.split('.layers.')[1].split('.')[0])+1 for name in names]
    result=dict(created_at=time.time(),cases=len(rows),sample_ids=[r['id'] for r in rows],
        scope='Pairs of layers within the same sample, then equal-weight average across completed samples. No cross-sample block-coordinate merging.',
        overlap_definition='|A intersection B| / min(|A|, |B|); equal layer budgets make this the shared fraction of either selection.',
        fixed_block_policy='Exclude configured sink, actual hard-mandatory blocks and the fixed partial-record inclusion constraint.',
        block_tokens=32,physical_model_layers_one_based=physical,
        aggregates=aggregates,
        included_fixed=dict(all_pair_overlap=float(whole[upper].mean()),
            adjacent_overlap=float(np.diag(whole,1).mean()),first_last_overlap=float(whole[0,-1])),
        mean_overlap_matrix=averaged.tolist(),mean_jaccard_matrix=np.mean(jaccards,axis=0).tolist(),
        mean_overlap_by_gap={str(gap):float(np.diag(averaged,gap).mean()) for gap in range(1,16)},
        per_task=per_task,per_case=rows,sources=sources,
        inference_unchanged=True,new_model_runs=0)
    (output/'overlap.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (output/'matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model_layer',*physical])
        for label,values in zip(physical,averaged):writer.writerow([label,*[100*x for x in values]])
    with (output/'per-task.csv').open('w',encoding='utf-8-sig',newline='') as f:
        columns=['task','cases','all_pair_overlap','adjacent_overlap','first_last_overlap','union_total_tokens','common_optional_blocks']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for task,row in per_task.items():writer.writerow(dict(task=task,**row))
    plot(output,result)
    print(json.dumps(dict(cases=len(rows),aggregates=aggregates,included_fixed=result['included_fixed'],
        mean_overlap_by_gap=result['mean_overlap_by_gap'],per_task=per_task,output=str(output)),ensure_ascii=False,indent=2))


def plot(output,result):
    averaged=np.asarray(result['mean_overlap_matrix'])
    physical=result['physical_model_layers_one_based']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,8.5),layout='constrained')
    im=ax.imshow(averaged*100,vmin=0,vmax=100,cmap='viridis')
    ax.set_xticks(range(16),physical);ax.set_yticks(range(16),physical)
    ax.set_xlabel('Model layer (1-based)');ax.set_ylabel('Model layer (1-based)')
    ax.set_title(f"Layerwise Mean-K: selected-block overlap\n{result['cases']} completed samples; mandatory blocks excluded")
    for i in range(16):
        for j in range(16):
            ax.text(j,i,f'{averaged[i,j]*100:.0f}',ha='center',va='center',fontsize=7,
                    color='black' if averaged[i,j]>.58 else 'white')
    fig.colorbar(im,ax=ax,label='Shared fraction of selected blocks (%)',shrink=.85)
    fig.savefig(output/'heatmap.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.plot_only:plot(args.output.resolve(),json.loads((args.output/'overlap.json').read_text()))
    else:analyze(args.root.resolve(),args.output.resolve())

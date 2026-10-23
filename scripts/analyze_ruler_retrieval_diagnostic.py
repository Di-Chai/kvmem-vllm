"""Replay captured Q/K, with post-hoc gold positions isolated from inference."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from vllm_kvmem.retrieval_diagnostic_math import content_distributions


def read(path):return json.loads(path.read_text())


def distribution(root, qarm, karm, case, device):
    qdir=root/qarm/'cases'/case/'diagnostic'
    kdir=root/karm/'cases'/case/'diagnostic'
    q=torch.load(qdir/'queries.pt',weights_only=True)
    means=torch.load(kdir/'means.pt',weights_only=True)
    meta=read(kdir/'archive.json')
    keys=np.load(kdir/'raw_keys.npy',mmap_mode='r')
    if len(keys)!=len(q) or means.shape[1]!=len(meta['counts']):raise ValueError('Layer/index mismatch')
    result={name:[] for name in ['mean','max','token','token_max']}
    for layer in range(len(q)):
        values=content_distributions(q[layer],keys[layer],meta['counts'],meta['score_mask'],
            means=means[layer],device=device)
        for name,v in values.items():result[name].append(v.numpy())
    return {name:np.stack(v) for name,v in result.items()},meta


def audit_full_prefix(meta, row):
    # The Query observer runs before the assistant-template suffix is admitted.
    # Equality to the request's total length would incorrectly require future
    # public input to be present in this causal observation.
    begin,end=row['served_query']
    if (meta['native_length']!=end or
            meta['dense_query_positions']!=list(range(begin,end))):
        raise ValueError('Full-prefix control did not retain the complete original context')
    return dict(full_observed_prefix_tokens=end,
                unobserved_template_suffix_tokens=row['served_input_tokens']-end)


def ranks(scores, mask, targets, budget_pages):
    mask=np.array(mask,dtype=bool)
    order=np.flatnonzero(mask)[np.argsort(-scores[mask],kind='stable')]
    rank={int(p):i+1 for i,p in enumerate(order)}
    return dict(targets=[dict(page=int(p),rank=rank.get(p),score=float(scores[p]),
                              top_budget=rank.get(p,10**9)<=budget_pages) for p in targets],
        cutoff=float(scores[order[min(budget_pages,len(order))-1]]),
        eligible_pages=int(mask.sum()),top_pages=[int(p) for p in order[:20]])


def analyze_case(root,case,device):
    old=read(Path('artifacts/ruler-failures-20261022/analysis.json'))
    target=next(r for r in old['rows'] if r['id']==case)['references'][0]['spans'][0]
    span=target['token_span'];keyspan=target['key_token_span']
    targets=list(range(span[0]//32,(span[1]-1)//32+1))
    output=root/'analysis'/case;output.mkdir(parents=True,exist_ok=True)
    summary=dict(id=case,value_span=span,key_span=keyspan,target_pages=targets,variants={})
    from tokenizers import Tokenizer
    from analyze_ruler_failures import PromptOffsets
    fixture=read(root/'fixture.json')
    row=next(r for r in fixture['rows'] if r['id']==case)
    profile=read(root/'compressed/profile.json')
    tokenizer=Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    ids=np.fromfile(Path(fixture['input_directory'])/(case+'.i32'),dtype='<i4').tolist()
    qids=ids[slice(*row['served_query'])]
    full_meta=read(root/'full/cases'/case/'diagnostic/native.json')
    summary['execution_audit']=audit_full_prefix(full_meta,row)
    for arm in ['compressed','full']:
        response=read(root/arm/'cases'/case/'response.json')
        events=[json.loads(line) for line in (root/arm/'server/capture/bounded_refresh.jsonl').open()]
        events=[e for e in events if e['request_id'].startswith(response['response_id']+'-')]
        pressure=[e for e in events if e['selection']=='pressure']
        query=[e for e in events if e['selection']=='query']
        if len(query)!=1 or query[0]['query_span']!=row['served_query'] or query[0]['guided_query_tokens']:
            raise ValueError('Original public Query contract changed')
        if arm=='full' and pressure:raise ValueError('Full-prefix control evicted input')
        if arm=='compressed' and not pressure:raise ValueError('Compressed arm did not exercise pressure')
        summary['execution_audit'][arm]=dict(pressure_count=len(pressure),original_query=True,
            request_sha256=read(root/arm/'cases'/case/'request.json')['input_sha256'])
    text=tokenizer.decode(qids,skip_special_tokens=False)
    begin=text.index(target['key'])
    lo,hi=PromptOffsets(tokenizer).positions(qids,text)(begin,begin+len(target['key']))
    summary['public_query_key_rows']=[lo,hi]
    summary['profile_differences']={k:{arm:read(root/arm/'profile.json')['kvmem'][k]
        for arm in ['compressed','full']} for k in ['select_budget','gen_budget','repository_blocks']}
    common=None
    for qa,ka in [('compressed','compressed'),('full','full'),('compressed','full'),('full','compressed')]:
        name=f'Q_{qa}_K_{ka}'
        partial=root/'analysis-partial'/case/'compressed.npz'
        if qa==ka=='compressed' and partial.exists():
            with np.load(partial) as cached:arrays={k:cached[k] for k in cached.files}
            meta=read(root/'compressed/cases'/case/'diagnostic/archive.json')
            if 'token_max' not in arrays:
                arrays,meta=distribution(root,qa,ka,case,device)
        else:
            arrays,meta=distribution(root,qa,ka,case,device)
        current=(meta['history'],meta['starts'],meta['counts'],meta['score_mask'],meta['transport_mask'])
        if common is not None and current!=common:raise ValueError('Counterfactual changed the normalization domain')
        common=current
        np.savez(output/(name+'.npz'),**arrays)
        # Match production: heads/layers averaged, original Query rows summed.
        rows={method:ranks(v.mean(0).sum(0),meta['transport_mask'],targets,
                           (meta['budget']-41*32)//32) for method,v in arrays.items()}
        rows.update({method+'_query_key_only':ranks(v[:,lo:hi].mean(0).sum(0),meta['transport_mask'],targets,
                           (meta['budget']-41*32)//32) for method,v in arrays.items()})
        if qa==ka:
            production=np.array(meta['mean_scores'])
            rows['production_mean']=ranks(production,meta['transport_mask'],targets,(meta['budget']-41*32)//32)
            rows['mean_fp32_replay_max_abs_error']=float(np.max(np.abs(arrays['mean'].mean(0).sum(0)-production)))
            rows['actual_preview_selected_target_pages']=[p for p in targets if p*32 in meta['selected_starts']]
        summary['variants'][name]=rows
    summary['native_attention']={}
    for arm in ['compressed','full']:
        directory=root/arm/'cases'/case/'diagnostic'
        meta=read(directory/'native.json');arr=np.load(directory/'native_attention.npy')
        total=arr.mean(0).sum(0)
        bypage=np.zeros(len(common[1]))
        observed=np.zeros(len(bypage),dtype=bool)
        for i,(begin,count,local) in enumerate(meta['segments']):
            if begin<common[0]:
                if begin%32 or begin//32>=len(bypage):raise ValueError('Native segment cannot be mapped exactly')
                bypage[begin//32]+=total[i]
                observed[begin//32]=True
        observed_mask=observed & np.asarray(common[-1],dtype=bool)
        ranking=ranks(bypage,observed_mask,targets,(36864-41*32)//32)
        for target in ranking['targets']:
            if not observed[target['page']]:target.update(rank=None,score=None,top_budget=None)
        summary['native_attention'][arm]=dict(native_tokens=meta['native_length'],
            targets_present=[p for p in targets if observed[p]],
            normalized_row_mass_min=float(arr.sum(-1).min()),
            normalized_row_mass_max=float(arr.sum(-1).max()),
            ranking=ranking,
            scope='Actual native-cache attention, retaining its full visible causal denominator including fixed pages and Query tokens. Missing pages have no observed mass.')
        np.save(output/('native_'+arm+'.npy'),bypage,allow_pickle=False)
    # Per-layer changes of the exact target raw K bytes; this is a coordinate
    # comparison, not a claim that compressed and full Q/K should be identical.
    aa=np.load(root/'compressed/cases'/case/'diagnostic/raw_keys.npy',mmap_mode='r')
    bb=np.load(root/'full/cases'/case/'diagnostic/raw_keys.npy',mmap_mode='r')
    kt=slice(keyspan[0]//32*32,(span[1]+31)//32*32)
    a=torch.from_numpy(np.array(aa[:,kt],copy=True));b=torch.from_numpy(np.array(bb[:,kt],copy=True))
    af=a.view(torch.float8_e4m3fn).float().flatten(1);bf=b.view(torch.float8_e4m3fn).float().flatten(1)
    qc=torch.load(root/'compressed/cases'/case/'diagnostic/queries.pt',weights_only=True).float().flatten(1)
    qf=torch.load(root/'full/cases'/case/'diagnostic/queries.pt',weights_only=True).float().flatten(1)
    summary['representation']=dict(target_record_byte_equal_per_layer=(a==b).float().flatten(1).mean(1).tolist(),
        target_record_cosine_per_layer=torch.nn.functional.cosine_similarity(af,bf,dim=1).tolist(),
        query_cosine_per_layer=torch.nn.functional.cosine_similarity(qc,qf,dim=1).tolist())
    summary['limits']=['Content-frame token attention deliberately omits RoPE to isolate MeanK averaging; it is not native model attention.',
        'Four crossed Q/K combinations are fixed-operand counterfactuals, not new model generations.',
        'Full arm preserves the entire input prefix until the observed final Query; it is a KVMem instrumentation control, not the earlier unmodified-vLLM benchmark.',
        'One-token serial diagnostic does not measure benchmark accuracy or throughput.']
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def plots(root, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for s in summaries:
        case=s['id'];out=root/'analysis'/case
        fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
        x=np.arange(len(np.load(out/'Q_full_K_full.npz')['mean'][0,0]))*32
        for arm,color in [('compressed','tab:orange'),('full','tab:blue')]:
            z=np.load(out/f'Q_{arm}_K_{arm}.npz')
            for method,style in [('mean','-'),('token','--')]:
                y=z[method].mean(0).sum(0)
                axes[0].plot(x,y,label=f'{arm}: {method}',alpha=.7,lw=.7,ls=style,color=color)
                axes[1].plot(x,y,label=f'{arm}: {method}',alpha=.8,lw=1,ls=style,color=color)
        lo=s['key_span'][0]-256;hi=s['value_span'][1]+256
        for arm,color in [('compressed','tab:orange'),('full','tab:blue')]:
            native=np.load(out/f'native_{arm}.npy')
            visible=np.where(native>0,native,np.nan)
            axes[2].plot(x,visible,label=f'{arm}: native visible history',alpha=.8,lw=.7,color=color)
        for ax in axes:
            ax.axvspan(s['key_span'][0],s['value_span'][1],alpha=.2,color='green',label='target key + value')
            ax.set_yscale('log');ax.set_ylabel('Query-summed page probability');ax.set_ylim(bottom=1e-9)
        axes[1].set_xlim(lo,hi);axes[1].set_xlabel('Original token position (32-token pages)')
        axes[0].set_title('Content frame: MeanK vs global token softmax + page sum (RoPE excluded)',fontsize=10)
        axes[2].set_title('Native-cache reconstruction: actual RoPE, K scale and full causal denominator',fontsize=10)
        axes[2].set_xlabel('Original token position (missing compressed pages are unobserved)')
        axes[0].legend(ncol=3,fontsize=8);axes[2].legend(fontsize=8)
        fig.suptitle(case+' — separate score domains; probabilities are not directly interchangeable')
        fig.savefig(out/'distribution.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--case',action='append');p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    root=a.root.resolve();fixture=read(root/'fixture.json')
    cases=a.case or [r['id'] for r in fixture['rows']]
    began=time.time();summaries=[]
    for case in cases:
        s=analyze_case(root,case,a.device);summaries.append(s)
        print(json.dumps(dict(case=case,seconds=time.time()-began,variants=s['variants'])),flush=True)
    plots(root,summaries)
    (root/'analysis/summary.json').write_text(json.dumps(dict(at=time.time(),cases=len(summaries),rows=summaries,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')

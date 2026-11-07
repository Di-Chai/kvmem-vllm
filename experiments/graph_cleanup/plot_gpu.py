"""Publication-exportable static figures from audited CSV/JSON outputs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def figure(path, fig):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def run(args):
    output=args.analysis/'figures';output.mkdir(exist_ok=True)
    rows=csv_rows(args.analysis/'candidates.csv')
    layers=csv_rows(args.analysis/'layers.csv')
    ids=[r['candidate_id'] for r in rows]
    short={'exact_duplicate':'duplicate','partial_overlap':'overlap','error_observation':'error',
           'code_observation':'code','thinking_without_replacement':'thinking','user_constraint_control':'constraint'}
    fig,axes=plt.subplots(1,2,figsize=(14,8),gridspec_kw={'width_ratios':[1,2.4]})
    for ax,kind,title in zip(axes,('attention_pre_gate_per_head','transformer_residual_sum'),
                            ('Attention output (before gate)','Transformer output (residual sum)')):
        indices=sorted({int(r['layer']) for r in layers if r['kind']==kind})
        values={(r['candidate_id'],int(r['layer'])):float(r['mean']) for r in layers if r['kind']==kind}
        grid=np.array([[values[(cid,l)] for l in indices] for cid in ids])
        im=ax.imshow(grid,aspect='auto',cmap='magma',vmin=0)
        ax.set_title(title);ax.set_xlabel('Layer (zero-based)')
        ticks=list(range(0,len(indices),2 if len(indices)==16 else 8))
        ax.set_xticks(ticks,[indices[t] for t in ticks])
        ax.set_yticks(range(len(ids)),[r['candidate_id']+' '+short[r['category']] for r in rows] if ax is axes[0] else [])
        for boundary in (3.5,7.5,11.5,15.5,19.5):
            if boundary<len(ids)-.5:
                ax.axhline(boundary,color='white',lw=.8,alpha=.6)
        fig.colorbar(im,ax=ax,shrink=.65,label='Mean relative L2 across 148 fixed query tokens')
    fig.suptitle(f'RTX 4090: real historical attention-page exclusion\n{len(ids)} measured candidates; original recurrent history retained',fontsize=13)
    figure(output/'layer-output-change',fig)
    valid=[r for r in rows if r['inconsistency_probability']]
    if valid:
        fig,ax=plt.subplots(figsize=(8,6))
        colors=plt.get_cmap('tab10')
        for i,category in enumerate(short):
            selected=[r for r in valid if r['category']==category]
            ax.scatter([float(r['transformer_mean']) for r in selected],
                [float(r['inconsistency_probability']) for r in selected],label=short[category],color=colors(i),s=55,alpha=.85)
            for r in selected:
                ax.annotate(r['candidate_id'],(float(r['transformer_mean']),float(r['inconsistency_probability'])),
                            xytext=(3,3),textcoords='offset points',fontsize=8)
        ax.set_xlabel('Mean relative L2: 64 Transformer layers × 148 tokens')
        ax.set_ylabel('Native judge: mean 1 − P(consistent) across 6 probes')
        ax.set_title('Computational change vs model-assisted semantic judgment\nJudge failed cross-field control; scores are not validated')
        ax.legend(fontsize=9);ax.grid(alpha=.2)
        figure(output/'complementarity',fig)
        original=args.root/'analysis-judge-r1/candidates.csv'
        if original.exists():
            old={r['candidate_id']:r for r in csv_rows(original)}
            fig,axes=plt.subplots(1,2,figsize=(11,4.5))
            x=[float(old[r['candidate_id']]['inconsistency_probability']) for r in valid]
            y=[float(r['inconsistency_probability']) for r in valid]
            axes[0].scatter(x,y,s=40)
            for r,a,b in zip(valid,x,y):axes[0].annotate(r['candidate_id'],(a,b),xytext=(2,2),textcoords='offset points',fontsize=8)
            axes[0].plot([0,1],[0,1],ls='--',color='gray',lw=1);axes[0].set_xlim(0,1);axes[0].set_ylim(0,1)
            axes[0].set_xlabel('Initial judge: mean 1 − P(consistent)');axes[0].set_ylabel('Revised judge: mean 1 − P(consistent)')
            bottom=np.zeros(2)
            for column,label,color in [('consistent_dimensions','consistent','#3a9d8f'),('omission_dimensions','omission','#dc8542'),('uncertain_dimensions','uncertain','#9b8cbc')]:
                values=[sum(int(r[column]) for r in old.values()),sum(int(r[column]) for r in valid)]
                axes[1].bar(['initial','revised'],values,bottom=bottom,label=label,color=color)
                bottom+=values
            axes[1].set_ylabel('Model choices across 24 × 6 probes');axes[1].legend(fontsize=9)
            fig.suptitle('Judge sensitivity to citation handling and cross-field instructions\nNeither version has independent human safety labels')
            figure(output/'judge-sensitivity',fig)
    recovery=args.root/'recovery-r1/results.json'
    dose=args.root/'dose-r1/summary.json'
    if dose.exists():
        results=json.loads(dose.read_text())
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for cid in sorted({r['candidate_id'] for r in results}):
            selected=[r for r in results if r['candidate_id']==cid]
            for ax,kind,title in zip(axes,('attention_pre_gate_per_head','transformer_residual_sum'),('Attention output','Transformer output')):
                ax.plot([r['removed_blocks'] for r in selected],
                    [r.get(kind+'_mean',0) for r in selected],marker='o',label=cid)
                ax.set_title(title);ax.set_xlabel('Excluded native pages (32 tokens each)');ax.set_ylabel('Mean relative L2');ax.grid(alpha=.2)
        axes[0].legend()
        fig.suptitle('Within-candidate page-count sweep: nested original-order page sets')
        figure(output/'page-count-sweep',fig)
    if recovery.exists():
        results=json.loads(recovery.read_text())
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for method,label in [('pageable_cpu','CPU pageable → GPU'),('pinned_cpu','CPU pinned → GPU'),
                             ('file_warm','Ceph file warm → GPU'),('file_client_cache_advised_cold','Ceph client-cold advised → GPU')]:
            for ax,percentile in zip(axes,('p50_ms','p95_ms')):
                ax.plot([r['bytes']/1024**2 for r in results],[r['summaries'][method][percentile] for r in results],marker='o',label=label)
                ax.set_xlabel('Native attention KV (MiB)');ax.set_ylabel(percentile.replace('_',' ')+' latency');ax.set_yscale('log');ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle('Recovery components: 20 repeats; recurrent state and page installation excluded')
        figure(output/'recovery-components',fig)
    serving=args.analysis/'serving.csv'
    if serving.exists():
        cells=csv_rows(serving)
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for row,kind in enumerate(('observer','pair')):
            for col,out in enumerate((256,512,1024)):
                ax=axes[row,col]
                for length in (4096,8192,16384,32768):
                    selected=sorted([c for c in cells if c['kind']==kind and int(c['input_length'])==length and int(c['output_length'])==out],key=lambda r:int(r['concurrency']))
                    if selected:
                        ax.plot([int(c['concurrency']) for c in selected],
                                [float(c['aggregate_output_tokens_per_wall_second']) for c in selected],marker='o',label=f'{length//1024}K input')
                ax.set_title(f'{kind}: {out} output tokens');ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16])
                ax.set_xlabel('Concurrent requests / pairs');ax.set_ylabel('Aggregate output tokens / wall second');ax.grid(alpha=.2)
                if row==0 and col==0:ax.legend(fontsize=8)
        fig.suptitle('Generative serving: one wave per cell, prefix caching disabled')
        fig.tight_layout();figure(output/'serving-throughput',fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,kind in zip(axes,('observer','pair')):
            for length in (4096,8192,16384,32768):
                selected=sorted([c for c in cells if c['kind']==kind and int(c['input_length'])==length and int(c['output_length'])==512],key=lambda r:int(r['concurrency']))
                if selected:
                    ax.plot([int(c['concurrency']) for c in selected],[float(c['ttft_p95_seconds']) for c in selected],marker='o',label=f'{length//1024}K')
            ax.set_title(kind+' / 512 output tokens');ax.set_xlabel('Concurrent requests / pairs');ax.set_ylabel('Client TTFT empirical P95 (s)')
            ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Queueing included; single-wave P95 is descriptive')
        figure(output/'serving-ttft',fig)
    native=args.analysis/'native-serving.csv'
    if native.exists():
        cells=csv_rows(native)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for length in (4096,8192,16384,32768):
            selected=sorted([c for c in cells if int(c['target_input'])==length],key=lambda r:int(r['concurrency']))
            if not selected:continue
            x=[int(c['concurrency']) for c in selected]
            axes[0].plot(x,[float(c['http_latency_p95_seconds']) for c in selected],marker='o',label=f'{length//1024}K input')
            axes[1].plot(x,[float(c['labels_per_wall_second']) for c in selected],marker='o',label=f'{length//1024}K input')
        for ax in axes:
            ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16])
            ax.set_xlabel('Concurrent HTTP requests');ax.grid(alpha=.2)
        axes[0].set_ylabel('HTTP response empirical P95 (s)');axes[0].legend(fontsize=9)
        axes[1].set_ylabel('Completed labels / wall second')
        fig.suptitle('Native System One: one categorical label per request, one wave per cell')
        figure(output/'native-serving',fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    run(p.parse_args())

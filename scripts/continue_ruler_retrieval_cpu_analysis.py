"""Finish frozen post-hoc Q/K analysis on CPU without reserving the model GPU."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def reusable(root,case):
    directory=root/'analysis'/case
    path=directory/'summary.json'
    if not path.exists():return False
    try:
        summary=read(path)
        if len(summary['variants'])!=4:return False
        for name,row in summary['variants'].items():
            if 'token_max' not in row:return False
            with np.load(directory/(name+'.npz')) as values:
                if set(values.files)!={'mean','max','token','token_max'}:return False
                if any(not np.isfinite(values[k]).all() for k in values.files):return False
        return set(summary['native_attention'])=={'compressed','full'}
    except (ValueError,OSError,KeyError):return False


def run(root):
    frozen=root/'postprocess-runtime'
    proof=read(root/'analysis-source-proof.json')['files']
    if any(digest(frozen/p)!=value for p,value in proof.items()):raise ValueError('Frozen analysis source changed')
    sys.path.insert(0,str(frozen/'scripts'))
    from analyze_ruler_retrieval_diagnostic import analyze_case,plots
    fixture=read(root/'fixture.json');summaries=[];recovery=root/'recovery-20261023-r2'
    with open('/tmp/kvmem-retrieval-analysis-'+root.name+'.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in fixture['rows']:
            case=row['id']
            save(root/'analysis-status.json',dict(status='analyzing',device='cpu',case=case,
                completed=len(summaries),expected=len(fixture['rows']),at=time.time()))
            if reusable(root,case):
                summary=read(root/'analysis'/case/'summary.json')
                summary=dict(summary,postprocess_device='cuda_prior_complete_reused')
            else:
                previous=root/'analysis'/case
                if previous.exists():previous.rename(recovery/('partial-prior-'+case))
                summary=analyze_case(root,case,'cpu')
                summary=dict(summary,postprocess_device='cpu')
            summaries.append(summary)
            print(json.dumps(dict(case=case,completed=len(summaries),device=summary['postprocess_device'])),flush=True)
        plots(root,summaries)
        for summary in summaries:
            for native in summary['native_attention'].values():
                if not .99999<native['normalized_row_mass_min']<=native['normalized_row_mass_max']<1.00001:
                    raise ValueError('Native attention normalization failed')
        originals=read(root/'analysis-input-hashes.json')
        if any(digest(root/p)!=value for p,value in originals.items()):raise ValueError('Captured result bytes changed')
        if torch.cuda.is_initialized():raise RuntimeError('CPU analysis unexpectedly initialized CUDA')
        save(root/'analysis/summary.json',dict(cases=len(summaries),rows=summaries,at=time.time(),
            postprocess_scope='Reused completed GPU postprocessing plus same-algorithm CPU remainder; device recorded per case. No changes to inference.',
            analyzer_sha256=digest(frozen/'scripts/analyze_ruler_retrieval_diagnostic.py')))
        save(root/'analysis-audit.json',dict(status='passed',cases=len(summaries),capture_arms=2,
            completed_capture_cases=2*len(summaries),capture_result_hashes_unchanged=True,
            normalized_native_attention=True,cpu_remainder_no_cuda=True,
            source_proof=str(root/'analysis-source-proof.json'),recovery_source_proof=str(recovery/'source-proof.json'),at=time.time()))
        save(root/'analysis-status.json',dict(status='completed',cases=len(summaries),at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a.root.resolve())
    except BaseException as e:
        save(a.root/'analysis-status.json',dict(status='failed',error=repr(e),at=time.time()));raise

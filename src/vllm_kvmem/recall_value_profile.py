"""One-time cold-cache comparison on an immutable real task Query/K/V state."""
import os
import time

import torch

from .recall_value_reference import score_value_records
from .recall_value_scoring import VALUE_METHODS


def compare_query_backends(authority, records, queries, mask, pins, *, max_tokens,
                           baseline_backend='streaming', baseline_block_batch=128,
                           optimized_backend='streaming_packed'):
    old_backend = os.environ.get('KVMEM_VALUE_RECALL_BACKEND')
    old_batch = os.environ.get('KVMEM_RECALL_BLOCK_BATCH')
    slots = [r.handle.slot for r in records]
    report = dict(passed=True, methods={}, candidate_blocks=len(records),
        candidate_tokens=sum(r.count for r in records), query_rows=queries.shape[1],
        normalization_mask_all=all(mask),
        scoring_layers=authority.scoring_layers, query_heads=queries.shape[2],
        physical_block_tokens=authority.shape[1], budget=max_tokens,
        query_reduction='sum', head_layer_reduction='mean',
        baseline_backend=baseline_backend, optimized_backend=optimized_backend,
        scope='Same frozen real task Q/K/V and native picker; each backend starts with an empty operand cache. '
              'Includes archive reads and operand packing. Diagnostic work is excluded from production score timing.')
    try:
        for method in VALUE_METHODS:
            runs = {}; scores = {}; selected = {}
            for name,backend,batch in [('baseline',baseline_backend,baseline_block_batch),
                                       ('optimized',optimized_backend,1024)]:
                authority._value_recall_records = {}
                os.environ['KVMEM_VALUE_RECALL_BACKEND'] = backend
                os.environ['KVMEM_RECALL_BLOCK_BATCH'] = str(batch)
                if queries.is_cuda or os.getenv('KVMEM_RECALL_DEVICE')=='cuda':
                    torch.cuda.synchronize()
                began = time.perf_counter()
                values, stats = score_value_records(authority, records, queries, mask, method)
                if os.getenv('KVMEM_RECALL_DEVICE')=='cuda':
                    torch.cuda.synchronize()
                scores[name] = values
                runs[name] = dict(seconds=time.perf_counter()-began, stats=stats)
                selected[name] = set(authority.memory.preview_selection(dict(zip(slots,values)), pins,
                    action=False,prefill_scores=None,max_tokens=max_tokens))
            a=torch.tensor(scores['baseline'],dtype=torch.float64)
            b=torch.tensor(scores['optimized'],dtype=torch.float64)
            close=bool(torch.allclose(a,b,atol=3e-5,rtol=3e-5))
            same=selected['baseline']==selected['optimized']
            same_query = runs['baseline']['stats']['query_operand_sha256'] == runs['optimized']['stats']['query_operand_sha256']
            same_scales = runs['baseline']['stats']['value_scales'] == runs['optimized']['stats']['value_scales']
            faster=runs['optimized']['seconds']<runs['baseline']['seconds']
            report['methods'][method]=dict(runs=runs,max_abs_score_difference=float((a-b).abs().max()),
                scores_close=close,selected_set_equal=same,selected_block_ids=sorted(selected['optimized']),
                query_and_value_scales_equal=same_query and same_scales,
                speedup=runs['baseline']['seconds']/runs['optimized']['seconds'])
            report['passed'] = report['passed'] and close and same and same_query and same_scales and faster
    finally:
        for key,value in [('KVMEM_VALUE_RECALL_BACKEND',old_backend),('KVMEM_RECALL_BLOCK_BATCH',old_batch)]:
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    return report

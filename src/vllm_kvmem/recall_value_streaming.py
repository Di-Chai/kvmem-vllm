"""Two-pass streaming evaluation with a stable dominant-block complement.

Unlike the reference, this retains only O(Query*ValueDim) output summaries on
the scoring device, never all candidate block Value vectors. One shared Q/K/V
tile supplies mass, ValueNorm and DeletionError. No change to scoring precision,
Query rows, candidate normalization, or layer/head reduction is made.
"""
import torch

from .recall_value_scoring import REFERENCE_METHODS, moments_from_blocks


def streaming_value_norm(queries,read_blocks,counts,eligible,*,kv_heads,device='cpu',
                         query_batch=128,block_batch=128):
    """One QK pass; ValueNorm does not need block Value vectors or deletion outputs."""
    if queries.ndim!=4 or kv_heads<=0 or queries.shape[2]%kv_heads or len(counts)!=len(eligible):
        raise ValueError('Invalid ValueNorm attention domain/geometry')
    if str(device).startswith('cuda') and (torch.get_float32_matmul_precision()!='highest' or
                                         torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('ValueNorm requires FP32 matmul without TF32')
    if not counts or not any(n>0 and e for n,e in zip(counts,eligible)) or not 1<=query_batch<=128 or not 1<=block_batch<=1024:
        raise ValueError('Invalid ValueNorm domain or tile size')
    layers,rows,heads,dim=queries.shape;groups=heads//kv_heads
    total=torch.zeros(len(counts),dtype=torch.float64);tiles=0
    for layer in range(layers):
        for head in range(kv_heads):
            for start in range(0,rows,query_batch):
                q=queries[layer,start:start+query_batch,head*groups:(head+1)*groups].reshape(-1,dim).to(device)
                masses=[];norms=[]
                for begin in range(0,len(counts),block_batch):
                    end=min(begin+block_batch,len(counts));k,v=read_blocks(begin,end,layer,head)
                    mom=moments_from_blocks(q,k.to(device),v.to(device),counts[begin:end],eligible[begin:end],host=False,with_value_mean=False)
                    masses.append(mom.log_mass);norms.append(mom.mean_value_l1);tiles+=1
                lm=torch.cat(masses,-1);norm=torch.cat(norms,-1)
                total+=(lm.softmax(-1)*norm).sum(0).cpu()
    return (total/(layers*heads)).tolist(),dict(implementation='streaming_one_pass_value_norm_v1',
        score_tiles=tiles,query_reduction='sum',head_layer_reduction='mean',score_softmax_applied=False)


def merge_attention(left_log, left_mean, right_log, right_mean):
    total = torch.logaddexp(left_log, right_log)
    live = torch.isfinite(total)
    safe_total = torch.where(live,total,0.)
    a=(left_log-safe_total).exp()
    b=(right_log-safe_total).exp()
    return total, a[...,None]*left_mean+b[...,None]*right_mean


def summarize(log_mass, means):
    total=log_mass.logsumexp(-1)
    live=torch.isfinite(total)
    weights=(log_mass-torch.where(live,total,0.)[:,None]).exp()
    return total,(weights[...,None]*means).sum(1)


def streaming_scores(queries, read_blocks, counts, eligible, *, kv_heads,
                     device='cpu', query_batch=128, block_batch=128):
    if (queries.ndim!=4 or kv_heads<=0 or queries.shape[2]%kv_heads or len(counts)!=len(eligible) or
            sum(bool(e) and n>0 for n,e in zip(counts,eligible))<2 or
            not 1<=query_batch<=128 or not 1<=block_batch<=1024):
        raise ValueError('Invalid streaming reference attention domain/geometry')
    if str(device).startswith('cuda') and (torch.get_float32_matmul_precision()!='highest' or
                                         torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('Streaming reference requires FP32 matmul without TF32')
    layers,rows,heads,dim=queries.shape;groups=heads//kv_heads;blocks=len(counts)
    totals={m:torch.zeros(blocks,dtype=torch.float64) for m in REFERENCE_METHODS}
    dominant_rows=0;tiles=0
    for layer in range(layers):
        for head in range(kv_heads):
            for start in range(0,rows,query_batch):
                q=queries[layer,start:start+query_batch,head*groups:(head+1)*groups].reshape(-1,dim).to(device)
                best_log=rest_log=best_mean=rest_mean=best_index=None
                # Keep the highest-mass block separate from the exact stable
                # log-space summary of every OTHER block. Even p==1 retains
                # a usable complement; no subtraction of total minus best.
                for begin in range(0,blocks,block_batch):
                    end=min(begin+block_batch,blocks);k,v=read_blocks(begin,end,layer,head)
                    mom=moments_from_blocks(q,k.to(device),v.to(device),counts[begin:end],eligible[begin:end],host=False)
                    lm,means=mom.log_mass,mom.mean_value
                    local_log,index=lm.max(-1)
                    local_mean=means[torch.arange(len(q),device=lm.device),index]
                    excluded=lm.clone();excluded.scatter_(1,index[:,None],-torch.inf)
                    other_log,other_mean=summarize(excluded,means)
                    if best_log is None:
                        best_log,best_mean,best_index=local_log,local_mean,index+begin
                        rest_log,rest_mean=other_log,other_mean
                    else:
                        replace=local_log>best_log
                        full_log,full_mean=merge_attention(best_log,best_mean,rest_log,rest_mean)
                        tile_log,tile_mean=merge_attention(local_log,local_mean,other_log,other_mean)
                        old_log,old_mean=merge_attention(rest_log,rest_mean,tile_log,tile_mean)
                        new_log,new_mean=merge_attention(full_log,full_mean,other_log,other_mean)
                        rest_log=torch.where(replace,new_log,old_log)
                        rest_mean=torch.where(replace[:,None],new_mean,old_mean)
                        best_index=torch.where(replace,index+begin,best_index)
                        best_mean=torch.where(replace[:,None],local_mean,best_mean)
                        best_log=torch.where(replace,local_log,best_log)
                    tiles+=1
                full_log,full_mean=merge_attention(best_log,best_mean,rest_log,rest_mean)
                if not bool(torch.isfinite(rest_log).all()):
                    raise ValueError('Streaming deletion has no complementary valid tokens')
                dominant_mass=(best_log-full_log).exp()
                dominant_error=dominant_mass*(best_mean-rest_mean).norm(dim=-1)
                dominant_rows+=int((dominant_mass>1-1e-8).sum())
                for begin in range(0,blocks,block_batch):
                    end=min(begin+block_batch,blocks);k,v=read_blocks(begin,end,layer,head)
                    mom=moments_from_blocks(q,k.to(device),v.to(device),counts[begin:end],eligible[begin:end],host=False)
                    p=(mom.log_mass-full_log[:,None]).exp()
                    error=(mom.mean_value-full_mean[:,None]).norm(dim=-1)*p/(1-p).clamp_min(.5)
                    dominant=(best_index>=begin)&(best_index<end)&(dominant_mass>.5)
                    index=dominant.nonzero(as_tuple=False).flatten()
                    error[index,best_index[index]-begin]=dominant_error[index]
                    values=dict(exact_mass=p,block_value_norm=p*mom.mean_value_l1,block_deletion_error=error)
                    for method,value in values.items():
                        if not bool(torch.isfinite(value).all() & (value>=0).all()):
                            raise ValueError('Nonfinite streaming reference score')
                        totals[method][begin:end]+=value.sum(0).cpu()
                    tiles+=1
    return {m:(v/(layers*heads)).tolist() for m,v in totals.items()},dict(
        implementation='streaming_two_pass_value_moments_v1',score_tiles=tiles,
        dominant_mass_rows=dominant_rows,query_reduction='sum',head_layer_reduction='mean',
        deletion_stability='stable_online_dominant_complement',score_softmax_applied=False)

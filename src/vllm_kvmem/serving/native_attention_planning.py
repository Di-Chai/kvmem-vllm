"""Freeze decode candidates without waiting for scope/accepted publication."""
import numpy as np


def canonical_segments(starts,counts,history,length,block_tokens):
    """Adjacent fragments of one logical history page have additive mass.

    Keep chunks bounded by block_tokens even when an active view contains
    repeated fragments of that page. No physical-page identity is introduced.
    """
    runs=_canonical(starts,counts,history,length)
    if np.all(runs[:,1]<=block_tokens):
        return runs
    repeats=(runs[:,1]+block_tokens-1)//block_tokens
    group=np.repeat(np.arange(len(runs)),repeats)
    first=np.repeat(np.cumsum(repeats)-repeats,repeats)
    offset=(np.arange(len(group))-first)*block_tokens
    return np.stack((runs[group,0]+offset,np.minimum(block_tokens,runs[group,1]-offset),runs[group,2]),axis=1)


def make_decode_frame(starts,counts,history,history_begin,local_begin,count,context,block_tokens):
    """Freeze the current view and its entire linear append domain once."""
    begin=list(map(int,starts));sizes=list(map(int,counts));keys=list(map(int,history))
    cursor=local_begin+count
    logical=history_begin+count
    while cursor<context:
        n=min(block_tokens-logical%block_tokens,context-cursor)
        begin.append(cursor);sizes.append(n);keys.append(logical-logical%block_tokens)
        cursor+=n;logical+=n
    frame=canonical_segments(begin,sizes,keys,context,block_tokens)
    from .native_attention import _layout_array
    layout,_=_layout_array(frame[:,0],frame[:,1],context)
    result=dict(starts=frame[:,0].copy(),counts=frame[:,1].copy(),history=frame[:,2].copy(),layout=layout)
    for array in result.values():array.flags.writeable=False
    return result


def _canonical(starts,counts,history,length):
    starts,counts,keys=np.asarray(starts),np.asarray(counts),np.asarray(history)
    if starts.ndim!=1 or starts.shape!=counts.shape or starts.shape!=keys.shape:
        raise ValueError('candidate segment shape')
    keep=starts<length
    starts,counts,keys=starts[keep],np.minimum(counts[keep],length-starts[keep]),keys[keep]
    if (not len(starts) or starts[0]!=0 or np.any(counts<=0) or
            not np.array_equal(starts[1:],starts[:-1]+counts[:-1]) or starts[-1]+counts[-1]!=length):
        raise ValueError('candidate context capacity')
    change=np.concatenate(([True],keys[1:]!=keys[:-1]))
    begin=starts[change]
    return np.stack((begin,np.diff(np.concatenate((begin,[length]))),keys[change]),axis=1)

"""Opt-in immutable native state for the unchanged cumulative attention rule.

Each staged history shares its previous immutable state. A successful commit
returns a new state; rejected or failed packets cannot mutate a prior snapshot.
The existing dictionary interface is materialized only for scoring/checkpoints.
"""
import math

from .prefill_attention import PrefillHistory


class BufferedPrefillHistory(PrefillHistory):
    storage_backend='native_buffered'

    @property
    def values(self):
        return self._buffered_state.values()

    @values.setter
    def values(self, values):
        from ._prefill_state_native import State
        self._buffered_state=State.from_values(values)

    def __init__(self,options):
        super().__init__(options)
        if not self.native_commit:
            raise ValueError('Buffered history requires the qualified native commit policy')
        self.successful_buffered_commits=0
        self.successful_buffered_restores=0

    def restore(self,snapshot,position):
        super().restore(snapshot,position)
        self.successful_buffered_restores+=1

    def accept(self,end,count,accepted,rows,starts,history,local_begin,masses):
        self.last_commit_backend=None
        if accepted!=count or end<=self.end or not rows:return False
        if hasattr(masses,'numpy'):
            buffer=masses.numpy()
        else:
            import numpy as np
            buffer=np.asarray(masses)
            if buffer.dtype.kind!='f':buffer=buffer.astype('float64')
        decay=math.exp2(-(end-self.end)/self.options.prefill_attention_half_life)
        updated=self._buffered_state.committed(decay,rows,starts,history,local_begin,buffer)
        self._buffered_state=updated
        self.end=end;self.samples+=len(rows);self.chunks+=1
        self.successful_buffered_commits+=1
        self.last_commit_backend='native_buffered'
        return True

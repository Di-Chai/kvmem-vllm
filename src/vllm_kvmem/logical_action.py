"""Physical prefill scores projected into the original logical packets.

This module neither prepares nor commits history/graph state. All coordinates
are checked before any projected packet is returned. Tensor slicing retains
the original download lease; projection does not inspect score values, wait on
events, copy a matrix, normalize, or combine logical transactions.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .logical_prefill import LogicalPrefillGroup, LogicalPrefillSpan


class LogicalActionResult(dict):
    """Physical packets plus their immutable logical publication views."""
    def __init__(self, values=()):
        super().__init__(values)
        self.logical_groups = {}
        self.logical_slices = {}

    def clear(self):
        # Native invalidation/cancellation must drop every tentative score
        # alias before its bounded download slot can serve the fallback.
        super().clear()
        self.logical_slices.clear()
        self.logical_groups.clear()


def _integers(values, name):
    if type(values) is not tuple or any(type(value) is not int for value in values):
        raise ValueError(f'{name} must be an immutable tuple of integers')


@dataclass(frozen=True)
class LogicalScorePlan:
    span: 'LogicalPrefillSpan'
    rows: tuple[int, ...]
    starts: tuple[int, ...]
    counts: tuple[int, ...]
    history: tuple[int, ...]
    block_tokens: int = 32

    def __post_init__(self):
        from .logical_prefill import LogicalPrefillSpan
        if not isinstance(self.span, LogicalPrefillSpan):
            raise ValueError('Logical score plan requires an immutable prefill span')
        for name in ('rows', 'starts', 'counts', 'history'):
            _integers(getattr(self, name), name)
        if type(self.block_tokens) is not int or self.block_tokens <= 0:
            raise ValueError('Logical score block size must be positive')
        if (any(not 0 <= row < self.span.count for row in self.rows) or
                any(left >= right for left, right in zip(self.rows, self.rows[1:]))):
            raise ValueError('Logical sampled rows must be increasing and inside their original span')
        if not self.starts or not len(self.starts) == len(self.counts) == len(self.history):
            raise ValueError('Logical segment vectors must have equal nonzero lengths')
        cursor = 0
        for start, count, history in zip(self.starts, self.counts, self.history):
            if (start != cursor or not 0 < count <= self.block_tokens or
                    history < 0 or history % self.block_tokens):
                raise ValueError('Logical segments must partition the visible context with history page identities')
            cursor += count
        if cursor != self.span.local_end:
            raise ValueError('Logical segments must cover exactly their original visible context')

    @property
    def segments(self):
        return tuple(zip(self.starts, self.counts, self.history))


@dataclass(frozen=True)
class LogicalActionSlice:
    """A score view and the unchanged event, with an explicit source lease."""
    group: 'LogicalPrefillGroup'
    plan: LogicalScorePlan
    masses: object = field(repr=False)
    event: object = field(default=None, repr=False)
    source_owner: object = field(default=None, repr=False)

    def accepted_from_prefix(self, retained):
        if type(retained) is not int or not 0 <= retained <= self.group.count:
            raise ValueError('Retained physical prefix must be inside its physical group')
        span = self.plan.span
        return min(span.count, max(0, retained - span.physical_offset))

    def action_packet(self, state):
        """Build only the existing acceptance input; never call its consumer."""
        from .action_attention import ActionPacket
        span, plan = self.plan.span, self.plan
        return ActionPacket(state, list(plan.rows), list(plan.history), self.masses, self.event,
            (span.history_end, span.count, plan.starts, span.local_begin),
            source_positions=tuple(span.history_begin + row for row in plan.rows),
            history_begin=span.history_begin, candidate_count=span.count)


def _matrix_geometry(masses):
    if isinstance(masses, torch.Tensor):
        if (masses.device.type != 'cpu' or masses.layout != torch.strided or
                masses.dtype not in (torch.float32, torch.float64) or masses.requires_grad):
            raise ValueError('Logical score projection requires an untracked CPU float matrix')
        shape, strides = tuple(masses.shape), tuple(masses.stride())
    elif isinstance(masses, np.ndarray):
        if masses.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('Logical score projection requires a CPU float matrix')
        if any(stride % masses.itemsize for stride in masses.strides):
            raise ValueError('Physical score strides must be whole float elements')
        shape = masses.shape
        strides = tuple(stride // masses.itemsize for stride in masses.strides)
    else:
        raise ValueError('Logical score projection requires an existing CPU tensor/array, without conversion')
    if len(shape) != 2 or shape[1] <= 0:
        raise ValueError('Physical scores must be a two-dimensional matrix')
    if shape[0] and (strides[1] != 1 or strides[0] < shape[1]):
        raise ValueError('Physical scores must have contiguous columns and nonoverlapping positive rows')
    return tuple(shape)


def _score_geometry(group, plans):
    from .logical_prefill import LogicalPrefillGroup, MAX_LOGICAL_PREFILL_CHUNKS
    if not isinstance(group, LogicalPrefillGroup):
        raise ValueError('Logical score projection requires a qualified immutable prefill group')
    if (type(plans) is not tuple or not 2 <= len(plans) <= MAX_LOGICAL_PREFILL_CHUNKS or
            len(plans) != len(group.spans) or group.count != len(plans)*2048 or
            any(span.count != 2048 for span in group.spans)):
        raise ValueError('Score projection requires a bounded group of logical 2K spans')
    if any(not isinstance(plan, LogicalScorePlan) for plan in plans):
        raise ValueError('Logical score projection requires validated score plans')
    final = plans[-1]
    if final.block_tokens != group.request_plan.block_tokens:
        raise ValueError('Logical score block geometry differs from the frozen request policy')
    expected_rows = []
    for span, plan in zip(group.spans, plans):
        if plan.span != span or plan.block_tokens != final.block_tokens:
            raise ValueError('Logical score plan belongs to a different span or block geometry')
        if plan.segments != final.segments[:len(plan.starts)]:
            raise ValueError('Logical segments are not an exact prefix of the physical score domain')
        expected_rows.extend(span.physical_offset + row for row in plan.rows)
    return tuple(expected_rows), len(final.starts)


def plan_logical_scores(group: 'LogicalPrefillGroup', frame):
    """Plan the original 2K transactions without advancing any mutable state.

    The caller checks native/storage admission before installing next_frame.
    Reusing the original immutable frame's extend preserves sample interval,
    near-capacity forcing, row offsets and history-tail segment semantics.
    """
    from .logical_prefill import LogicalPrefillGroup
    from .prefill_planning import PrefillPlanFrame
    if not isinstance(group, LogicalPrefillGroup) or not isinstance(frame, PrefillPlanFrame):
        raise ValueError('Logical score planning requires immutable group and prefill frame')
    if frame.history_end != group.history_begin or frame.local_end != group.local_begin:
        raise ValueError('Logical score planning frame has stale prefix coordinates')
    plans, next_frame = [], frame
    for span in group.spans:
        rows, starts, counts, history, next_frame = next_frame.extend(
            span.history_begin, span.local_begin, span.count, group.request_plan.block_tokens)
        plans.append(LogicalScorePlan(span, tuple(rows), tuple(starts), tuple(counts),
                                      tuple(history), group.request_plan.block_tokens))
    plans = tuple(plans)
    physical_rows, _ = _score_geometry(group, plans)
    return plans, physical_rows, next_frame


def project_logical_scores(group: 'LogicalPrefillGroup', plans: tuple[LogicalScorePlan, ...],
                           masses, *, physical_rows: tuple[int, ...], event=None):
    """Project one qualified physical owner into its original 2K transactions.

    ``physical_rows`` are owner-relative Q row indices in matrix order. The
    final logical plan supplies the physical producer's complete column domain.
    Earlier domains must be exact prefixes, including partial segment counts
    and duplicate/reordered logical history identities. Future columns are
    omitted, not retained as zero-valued history pages. Numerical validation
    remains in the original acceptance path after the original event fence.
    """
    expected_rows, columns = _score_geometry(group, plans)
    _integers(physical_rows, 'physical_rows')
    if physical_rows != expected_rows:
        raise ValueError('Physical sampled row order differs from the original logical plans')
    if _matrix_geometry(masses) != (len(physical_rows), columns):
        raise ValueError('Physical score matrix shape differs from its logical rows or segments')
    # Finish every geometry check before constructing any view. A malformed
    # later span cannot escape as a partially returned, committable prefix.
    result, begin = [], 0
    for plan in plans:
        end = begin + len(plan.rows)
        view = masses[begin:end, :len(plan.starts)]
        result.append(LogicalActionSlice(group, plan, view, event, masses))
        begin = end
    return tuple(result)


def reference_logical_scores(group, plans, query, positions, key, table, cos_sin,
                             key_scale, starts, counts, *, scale, q_dtype, q_scale,
                             block_tokens, fused=False, profile=None):
    """Use each original scorer width before assembling one physical result.

    The normal native producer never calls this helper. Numerical fallback
    retains every original 2K causal context, query chunk and segment domain;
    padding only transports those old scores through the shared layer average
    and D2H. It cannot expose future pages to either logical transaction.
    """
    from .action_scoring import page_attention
    rows, columns = _score_geometry(group, plans)
    if (query.ndim != 3 or query.shape[0] != len(rows) or
            tuple(positions.shape) != (len(rows),) or
            tuple(starts.shape) != (columns,) or tuple(counts.shape) != (columns,) or
            block_tokens != group.request_plan.block_tokens):
        raise ValueError('Logical reference inputs differ from their frozen sampling geometry')
    parts, backends, begin = [], [], 0
    for plan in plans:
        end = begin+len(plan.rows)
        if end > begin:
            n = len(plan.starts)
            values = page_attention(query[begin:end], positions[begin:end], key, table,
                cos_sin, key_scale, starts[:n], counts[:n], plan.span.local_end,
                scale=scale, q_dtype=q_dtype, q_scale=q_scale, block_tokens=block_tokens,
                **({'profile': profile.child(logical_chunk=plan.span.index)} if profile else {}),
                **(dict(fused=True, complete_partition=True) if fused else {}))
            if values.shape != (end-begin, n):
                raise RuntimeError('Logical reference scorer returned a different shape')
            backends.append(getattr(values, '_action_scoring_backend', 'reference'))
            if n < columns:
                values = torch.nn.functional.pad(values, (0, columns-n))
            parts.append(values)
        begin = end
    if not parts:
        return torch.empty((0, columns), dtype=torch.float32, device=query.device)
    result = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    result._action_scoring_backend = 'fused' if all(value == 'fused' for value in backends) else 'reference'
    return result

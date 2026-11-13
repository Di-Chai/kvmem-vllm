"""The contract between the serving layer and the strategy layer.

Serving moves blocks and schedules work. A strategy only decides which blocks:
it turns the signals serving hands it into per-block scores, or scores into a
set of blocks. The two sides meet in the three roles below and nowhere else.

Rules every strategy must keep (serving relies on them to pipeline work):

1. Pure: the result depends only on the view. No hidden state, no reading of
   GPU residency, no dependence on another layer's or request's decision.
2. Deterministic for equal inputs.
3. Read-only: a view and everything reachable from it is never modified.
4. Serving code is never imported from here (checked by tests/test_strategy_layering.py).

A view is a handful of references to buffers serving already holds; building
one copies nothing. This module imports no torch and no serving module.
"""
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

BLOCK_TOKENS = 32

# Role names, as used by the registry and by the profile options
# strategy_query_scorer / strategy_pressure_scorer / strategy_picker.
QUERY_SCORER = 'query_scorer'
PRESSURE_SCORER = 'pressure_scorer'
PICKER = 'picker'
EVICTOR = 'evictor'
ROLES = (QUERY_SCORER, PRESSURE_SCORER, PICKER, EVICTOR)

# Where a decision is made. GLOBAL: one selection serves every layer
# (authority.select_for_queries, prefill_attention.select_pressure). LAYER:
# each attention layer selects its own window (layerwise_query/_pressure).
GLOBAL = 'global'
LAYER = 'layer'
PATHS = (GLOBAL, LAYER)


@dataclass(frozen=True)
class QueryLayerView:
    """One attention layer's retrieval decision at a Query, or a global one.

    layer: the layer index; None on the global path, where one decision serves
        every layer and query/mean_index cover all scoring layers.
    records: the candidate snapshot, in history order; each has history_begin,
        count and handle.slot.
    eligible: per record, whether it competes for the budget. Ineligible
        records (sink, recent and mandatory blocks) are placed by serving and
        must score 0.
    query: this layer's Query rows, shape (1, tokens, heads, head_dim).
    mean_index: the prepared per-layer mean-key index, or None.
    history_tokens: accepted history length the snapshot covers.
    options: scorer-specific settings from serving (for example the MeanK backend).
    """
    layer: Any               # the attention layer, or None for a global decision (every layer)
    records: Sequence[Any]
    eligible: Sequence[bool]
    query: Any
    mean_index: Any
    history_tokens: int
    options: Mapping[str, Any]


@dataclass(frozen=True)
class PressureLayerView:
    """One attention layer's retention decision when its window is full.

    mapping: the layer's current window, one row per block; row[2] is the
        block's history_begin and row[3] its token count.
    history: the layer's accumulated attention (scores() maps history_begin to mass).
    native_scores: serving's own lookup of history.scores() for every mapping
        row, when it has one; None otherwise.
    decode: True for a rolling-decode refresh, False during prefill.
    """
    layer: int
    mapping: Sequence[Any]
    history: Any
    native_scores: Any
    decode: bool


@dataclass(frozen=True)
class PickView:
    """Scores to a block set, under mandatory blocks and a token budget.

    kind: 'query', 'prefill' or 'pressure'.
    scores: for 'query' and 'prefill', {slot: score} over the scored snapshot.
    candidates: for 'pressure', (slot, score) pairs; only these and the
        mandatory slots may be returned.
    mandatory: slots that must be returned.
    budget_tokens: the selected-token budget; None means the store's own.
    block_info: slot -> (history_begin, count), when serving can supply it.
    block_tokens: tokens a full block is charged against the budget.
    """
    kind: str
    scores: Mapping[int, float]
    candidates: Sequence[tuple]
    mandatory: Sequence[int]
    budget_tokens: Any
    block_info: Any
    block_tokens: int = BLOCK_TOKENS


@runtime_checkable
class QueryScorer(Protocol):
    name: str

    def scores(self, view: QueryLayerView) -> Sequence[float]:
        """One finite score per record, in record order; 0 for ineligible records."""


@runtime_checkable
class PressureScorer(Protocol):
    name: str

    def scores(self, view: PressureLayerView) -> Sequence[float]:
        """One finite score >= 0 per mapping row, in row order; at least one > 0."""


@runtime_checkable
class Picker(Protocol):
    name: str

    def pick(self, view: PickView) -> Sequence[int]:
        """The selected slots."""


class ContractError(RuntimeError):
    """A strategy returned something the contract forbids."""


def check_query_scores(view, scores):
    if len(scores) != len(view.records):
        raise ContractError('Query scorer returned %d scores for %d records' % (len(scores), len(view.records)))
    for score, eligible in zip(scores, view.eligible):
        if not math.isfinite(score):
            raise ContractError('Query scores must be finite')
        if not eligible and score != 0:
            raise ContractError('An ineligible record must score 0')
    return scores


def check_pressure_scores(view, scores):
    if len(scores) != len(view.mapping):
        raise ContractError('Pressure scorer returned %d scores for %d window blocks' % (
            len(scores), len(view.mapping)))
    positive = False
    for score in scores:
        if not math.isfinite(score) or score < 0:
            raise ContractError('Pressure scores must be finite and non-negative')
        positive = positive or score > 0
    if not positive:
        raise ContractError('Pressure scores need at least one positive value')
    return scores


def check_pick(view, selected):
    selected = tuple(selected)
    chosen = set(selected)
    if len(chosen) != len(selected):
        raise ContractError('A picker returned a slot twice')
    if not set(view.mandatory) <= chosen:
        raise ContractError('A picker dropped a mandatory block')
    if view.kind == 'pressure':
        allowed = {slot for slot, _ in view.candidates} | set(view.mandatory)
        if not chosen <= allowed:
            raise ContractError('A pressure picker chose a block outside the window')
    if view.budget_tokens is not None and view.block_tokens * len(chosen) > view.budget_tokens:
        raise ContractError('A picker exceeded the token budget')
    return selected


@dataclass(frozen=True)
class Strategy:
    """A registry entry: what it is, what it needs and what serving may assume.

    needs: signals the strategy reads from its view. Serving refuses a
        strategy whose signals the current configuration cannot provide.
    paths: where it can decide (GLOBAL, LAYER). Serving refuses it elsewhere.
    native: the paths on which serving's built-in code implements it (True:
        every path in paths). Serving runs that code, bound to the registry
        with registry.bind, and never calls build() there.
    legacy: kept for old profiles; not part of the comparison set.
    summary: one line for the generated strategy table.
    """
    role: str
    name: str
    build: Callable[[], Any]
    needs: tuple = ()
    native: Any = False
    summary: str = ''
    paths: tuple = (LAYER,)
    legacy: bool = False

    def __post_init__(self):
        paths = tuple(self.paths)
        if not paths or any(path not in PATHS for path in paths):
            raise ValueError('Strategy paths must be a nonempty subset of %s' % (PATHS,))
        native = paths if self.native is True else () if self.native is False else tuple(self.native)
        if not set(native) <= set(paths):
            raise ValueError('A strategy can be native only on its own paths')
        object.__setattr__(self, 'paths', paths)
        object.__setattr__(self, 'native', native)

    def native_on(self, path):
        return path in self.native

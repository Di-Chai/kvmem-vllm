"""The strategies shipped with the package.

Entries marked native are serving's existing code paths: the per-layer MeanK
Query scores, the accumulated-attention pressure scores, the native store
picker and the native historical-GC policies. They are listed so that a
profile can name them and the strategy table is complete; serving runs them
unchanged. The recency scorers are the lower reference for comparisons: they
keep what is most recent and ignore content.
"""
from .contract import EVICTOR, PICKER, PRESSURE_SCORER, QUERY_SCORER, Strategy
from .registry import register


class RecencyQueryScorer:
    """Later history scores higher; the same choice in every layer."""
    name = 'recency'

    def scores(self, view):
        total = max(1, view.history_tokens)
        return [(record.history_begin + record.count) / total if eligible else 0.0
                for record, eligible in zip(view.records, view.eligible)]


class RecencyPressureScorer:
    """A window block scores its end position, so the newest blocks stay."""
    name = 'recency'

    def scores(self, view):
        return [float(row[2] + row[3]) for row in view.mapping]


def _native():
    raise RuntimeError('A native strategy is run by serving and is never built')


register(Strategy(QUERY_SCORER, 'meank', _native, needs=('layer_query', 'mean_index'), native=True,
                  summary='Softmax of the layer Query against per-block mean keys'))
register(Strategy(QUERY_SCORER, 'recency', RecencyQueryScorer, needs=('positions',),
                  summary='Most recent history first; lower reference'))
register(Strategy(PRESSURE_SCORER, 'attention', _native, needs=('layer_attention',), native=True,
                  summary='Accumulated attention mass per layer'))
register(Strategy(PRESSURE_SCORER, 'recency', RecencyPressureScorer, needs=('positions',),
                  summary='Most recent window blocks stay; lower reference'))
register(Strategy(PICKER, 'native', _native, native=True,
                  summary='Native store picker: sink and recent bands, mandatory blocks, top scores to the budget'))
register(Strategy(EVICTOR, 'lru', _native, native=True,
                  summary='Historical GC, least recently used (historical_gc.policy)'))
register(Strategy(EVICTOR, 'redundancy', _native, native=True,
                  summary='Historical GC by redundancy, optional thinking-only scope and LRU fallback'))

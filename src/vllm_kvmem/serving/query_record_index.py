"""One public Query's immutable record order, independent of native ranking.

The public Query already owns one fixed record tuple and runs all consuming
layers in a single forward. Cache its slot identities once; do not retain this
index across requests, accepted-prefix changes, or later Query forwards.
"""


class QueryRecordIndex:
    def __init__(self, records, partial_excluded):
        if not isinstance(records, tuple):
            raise ValueError('Query record index requires its frozen record tuple')
        self.records = records
        self.slots = tuple(record.handle.slot for record in records)
        self.positions = {slot: index for index, slot in enumerate(self.slots)}
        if len(self.positions) != len(records):
            raise ValueError('Query record index contains duplicate slot identities')
        excluded = frozenset(partial_excluded)
        if not excluded <= self.positions.keys():
            raise ValueError('Excluded Query partial is absent from its records')
        self.excluded_indices = tuple(i for i, slot in enumerate(self.slots) if slot in excluded)

    def score_dict(self, scores):
        if len(scores) != len(self.slots):
            raise ValueError('Query scores differ from their frozen record extent')
        # Preserve the legacy insertion order and scalar conversion exactly.
        return {slot: float(score) for slot, score in zip(self.slots, scores)}

    def chosen(self, slots):
        # Native preview owns ranking. Assembly and evidence retain the same
        # original record order as the legacy full-candidate membership scan.
        try:
            indices = sorted(self.positions[slot] for slot in slots)
        except KeyError as error:
            raise ValueError('Selected Query slot is absent from its records') from error
        return [self.records[index] for index in indices]

    def starts(self, slots):
        return [record.history_begin for record in self.chosen(slots)]


def query_record_index(state):
    """Initialize after global commit has fixed partial-block inclusion."""
    index = state.get('query_record_index')
    if index is None:
        index = state['query_record_index'] = QueryRecordIndex(
            state['records'], state['partial_excluded'])
    elif index.records is not state['records']:
        raise RuntimeError('Query record tuple changed during its public forward')
    return index

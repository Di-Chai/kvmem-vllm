"""Read physical model geometry without changing logical publication reports."""


def native_publication_step(row):
    """Return the original forward extent, or preserve an ungrouped legacy row.

    Only a completed logical group may attach this receipt to its final row.
    A zero final logical acceptance can still retain an earlier physical prefix;
    validate the prefix clamp instead of comparing the two absolute row ends.
    """
    if not isinstance(row, dict):
        raise ValueError('Native publication requires a step dictionary')
    if 'physical_step' not in row:
        return row
    physical = row['physical_step']
    fields = ('history_begin', 'scheduled', 'accepted', 'rejected', 'terminal_discarded')
    if not isinstance(physical, dict) or 'physical_step' in physical:
        raise ValueError('Physical publication receipt must be a nonrecursive step dictionary')
    for step in (row, physical):
        if any(type(step.get(name)) is not int or step[name] < 0 for name in fields):
            raise ValueError('Physical publication receipt requires complete nonnegative integer counts')
        if (step['scheduled'] <= 0 or
                step['accepted']+step['rejected']+step['terminal_discarded'] != step['scheduled']):
            raise ValueError('Physical publication receipt has inconsistent acceptance counts')
    offset = row['history_begin']-physical['history_begin']
    if (offset < 0 or offset+row['scheduled'] != physical['scheduled'] or
            physical['scheduled'] <= row['scheduled'] or physical['rejected'] or row['rejected'] or
            row['accepted'] != min(row['scheduled'], max(0, physical['accepted']-offset))):
        raise ValueError('Logical final row differs from its complete physical prefix')
    return physical

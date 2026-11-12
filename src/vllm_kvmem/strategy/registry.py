"""Strategies by role and name. Serving looks them up here and nowhere else."""
from .contract import EVICTOR, PICKER, PRESSURE_SCORER, QUERY_SCORER, ROLES, Strategy

# What serving runs when a profile names nothing.
DEFAULTS = {QUERY_SCORER: 'meank', PRESSURE_SCORER: 'attention', PICKER: 'native'}

_REGISTRY = {role: {} for role in ROLES}


def register(strategy):
    if not isinstance(strategy, Strategy) or strategy.role not in ROLES:
        raise ValueError('Expected a Strategy with a known role')
    if not isinstance(strategy.name, str) or not strategy.name:
        raise ValueError('A strategy needs a name')
    known = _REGISTRY[strategy.role]
    if strategy.name in known:
        raise ValueError('Strategy %s/%s is already registered' % (strategy.role, strategy.name))
    known[strategy.name] = strategy
    return strategy


def names(role):
    return tuple(sorted(_REGISTRY[role]))


def get(role, name):
    try:
        return _REGISTRY[role][name]
    except KeyError:
        raise ValueError('Unknown %s %r; known: %s' % (role, name, ', '.join(names(role)) or 'none')) from None


def resolve(role, name):
    """The strategy object serving should call, or None for its built-in path."""
    strategy = get(role, name)
    if strategy.native:
        return None
    instance = strategy.build()
    if getattr(instance, 'name', None) != name:
        raise ValueError('Strategy %s/%s built an object with another name' % (role, name))
    return instance


def table():
    """Every registered strategy, for the generated strategy table."""
    return tuple(_REGISTRY[role][name] for role in ROLES for name in names(role))

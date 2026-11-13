"""Strategies by role and name. Serving looks them up here and nowhere else."""
from .contract import EVICTOR, GLOBAL, LAYER, PATHS, PICKER, PRESSURE_SCORER, QUERY_SCORER, ROLES, Strategy

# What serving runs when a profile names nothing.
DEFAULTS = {QUERY_SCORER: 'meank', PRESSURE_SCORER: 'attention', PICKER: 'native'}

_REGISTRY = {role: {} for role in ROLES}
# Serving's built-in implementations, bound by serving: (role, name, path) -> callable.
_NATIVE = {}


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


def names(role, path=None):
    return tuple(sorted(name for name, s in _REGISTRY[role].items() if path is None or path in s.paths))


def get(role, name):
    try:
        return _REGISTRY[role][name]
    except KeyError:
        raise ValueError('Unknown %s %r; known: %s' % (role, name, ', '.join(names(role)) or 'none')) from None


def check_path(role, name, path):
    """The entry, if it can decide on this path; ValueError otherwise."""
    if path not in PATHS:
        raise ValueError('Unknown strategy path %r' % (path,))
    strategy = get(role, name)
    if path not in strategy.paths:
        raise ValueError('%s %r does not run on the %s path; it runs on: %s; %s strategies: %s' % (
            role, name, path, ', '.join(strategy.paths), path, ', '.join(names(role, path))))
    return strategy


def resolve(role, name, path=LAYER):
    """The strategy object serving should call, or None for its built-in code."""
    strategy = check_path(role, name, path)
    if strategy.native_on(path):
        return None
    instance = strategy.build()
    if getattr(instance, 'name', None) != name:
        raise ValueError('Strategy %s/%s built an object with another name' % (role, name))
    return instance


def bind(role, name, path, implementation):
    """Serving registers its built-in implementation of a native entry."""
    strategy = check_path(role, name, path)
    if not strategy.native_on(path):
        raise ValueError('%s %r is not native on the %s path' % (role, name, path))
    if not callable(implementation):
        raise ValueError('A native implementation must be callable')
    _NATIVE[(role, name, path)] = implementation


def native(role, name, path):
    """Serving's bound implementation of a native entry on a path."""
    check_path(role, name, path)
    try:
        return _NATIVE[(role, name, path)]
    except KeyError:
        raise ValueError('No built-in implementation of %s %r is bound for the %s path' % (role, name, path)) from None


def table():
    """Every registered strategy, for the generated strategy table."""
    return tuple(_REGISTRY[role][name] for role in ROLES for name in names(role))

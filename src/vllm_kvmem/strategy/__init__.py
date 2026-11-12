"""The strategy layer: which history blocks are retrieved, kept or evicted.

See contract.py for the rules and docs/layering.md for the layers.
"""
from . import builtin as _builtin  # registers the shipped strategies
from .contract import (BLOCK_TOKENS, EVICTOR, PICKER, PRESSURE_SCORER, QUERY_SCORER, ROLES, ContractError,
                       PickView, PressureLayerView, QueryLayerView, Strategy, check_pick,
                       check_pressure_scores, check_query_scores)
from .registry import DEFAULTS, get, names, register, resolve, table

__all__ = ['BLOCK_TOKENS', 'EVICTOR', 'PICKER', 'PRESSURE_SCORER', 'QUERY_SCORER', 'ROLES', 'ContractError',
           'PickView', 'PressureLayerView', 'QueryLayerView', 'Strategy', 'check_pick',
           'check_pressure_scores', 'check_query_scores', 'DEFAULTS', 'get', 'names', 'register',
           'resolve', 'table']

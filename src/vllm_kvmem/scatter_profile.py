"""Opt-in cProfile of the background CPU scatter/publication thread.

KVMEM_SCATTER_CPROFILE=1 profiles every scatter work item on its own thread and
rewrites <KVMEM_RUNNER_PROBE_DIR>/scatter_cprofile.txt every 50 items. Diagnostic
only: cProfile inflates Python-heavy code, so compare shares, not absolute time.
"""
import contextlib
import os
from . import settings as _settings

_profile = None
_items = 0


@contextlib.contextmanager
def profiled(runtime):
    global _profile, _items
    if _settings.get('KVMEM_SCATTER_CPROFILE') != '1':
        yield
        return
    import cProfile
    if _profile is None:
        _profile = cProfile.Profile()
    _profile.enable()
    try:
        yield
    finally:
        _profile.disable()
        _items += 1
        if _items % 50 == 0:
            _dump()


def _dump():
    import io
    import pstats
    directory = _settings.get('KVMEM_RUNNER_PROBE_DIR')
    if not directory or _profile is None:
        return
    out = io.StringIO()
    stats = pstats.Stats(_profile, stream=out)
    stats.sort_stats('cumulative').print_stats(70)
    stats.sort_stats('tottime').print_stats(50)
    with open(os.path.join(directory, 'scatter_cprofile.txt'), 'w') as stream:
        stream.write(out.getvalue())

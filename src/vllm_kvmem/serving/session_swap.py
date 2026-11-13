"""Session ownership fences for exact active-request page swaps.

P/M checkpoints and their namespace stay owned by the same native request.
Swapping current GPU rows neither claims an idle prefix nor creates a session
checkpoint. Normal terminal capture/rewind runs only after exact page restore.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionSwapIdentity:
    namespace: tuple[str, str]
    generation: int

    def __post_init__(self):
        if (not isinstance(self.namespace, tuple) or len(self.namespace) != 2 or
                any(not isinstance(value, str) for value in self.namespace) or
                not self.namespace[1] or type(self.generation) is not int or self.generation <= 0):
            raise ValueError('Session swap requires an exact namespace and generation')


def _entry(scheduler, request):
    from .session_control import SessionLedger
    namespace = SessionLedger.namespace(request)
    ledger = getattr(scheduler.connector, 'sessions', None)
    entry = getattr(ledger, 'requests', {}).get(request.request_id)
    return namespace, ledger, entry


def session_swap_candidate(scheduler, request):
    """Return a quiescent owned namespace, or defer its unfinished import."""
    namespace, ledger, entry = _entry(scheduler, request)
    if namespace is None:
        return True, None
    if (entry is None or ledger.entries.get(namespace) is not entry or
            entry.namespace != namespace or entry.request_id != request.request_id or
            entry.phase != 'active' or not entry.allocated or entry.history):
        return False, None
    commands = getattr(ledger, 'commands', None)
    if any(command.request_id == request.request_id for name in ('bindings', 'captures', 'terminals')
           for command in getattr(commands, name, ())):
        return False, None
    # candidate_position is deliberately retained by sealed P-only sessions;
    # it is not itself evidence that a worker capture command remains pending.
    return True, SessionSwapIdentity(namespace, entry.generation)


def validate_session_swap(scheduler, request, identity):
    """Validate the retained lease, including after external cancellation.

    Deferred _free_request has not called SessionLedger.finished yet, so the
    session lease must remain active even when the native request is finished.
    """
    namespace, ledger, entry = _entry(scheduler, request)
    if identity is None:
        if namespace is not None:
            raise ValueError('A session request lost its swap namespace identity')
        return
    if (not isinstance(identity, SessionSwapIdentity) or namespace != identity.namespace or
            entry is None or ledger.entries.get(namespace) is not entry or
            entry.request_id != request.request_id or entry.namespace != identity.namespace or
            entry.generation != identity.generation or entry.phase != 'active' or
            not entry.allocated or entry.history):
        raise ValueError('Session swap namespace, generation or active lease changed')


def owned_session_admission(ledger, request):
    """Defer full owned-session pools before lookup can create uncached owners."""
    from .session_control import SessionLedger, session_release_intent
    namespace = SessionLedger.namespace(request)
    if namespace is None:
        return True
    if session_release_intent(getattr(request, 'kv_transfer_params', None),
            request.session_id, request.prompt_token_ids, SessionLedger.output_tokens(request)):
        return True  # Explicit close uses the existing tiny stateless path.
    if namespace in ledger.uncached_sessions:
        ledger.waits += 1
        return False
    if (namespace not in ledger.entries and len(ledger.entries) >= ledger.capacity and
            not any(entry.phase == 'idle' for entry in ledger.entries.values())):
        ledger.waits += 1
        return False
    return True


@dataclass(frozen=True)
class SessionSwapFence:
    identity: SessionSwapIdentity
    binding: object
    authority: object
    checkpoint: object
    prompt: object

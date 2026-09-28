"""Process-local Pier transport timeout alignment; no installed package edits.

The official generated proxy otherwise times out silent non-streaming model
calls at 900 seconds, below the existing 21600-second model-call budget. Network
allowlists, authentication, solver prompts, retry policy and scoring stay intact.
"""
import hashlib
import inspect
from pathlib import Path

AGENT_SETUP_SHA256 = '0b5c8d55a2bc66356b5fd7b20b915f3e07d8eac920bf16aa232514c3719686b6'


def align_proxy_timeout(script):
    marker = 'shutdown_lifetime 1 seconds\n'
    if script.count(marker) != 1 or 'read_timeout ' in script or 'forward_timeout ' in script:
        raise RuntimeError('Unqualified Pier proxy timeout template')
    return script.replace(marker, marker + 'read_timeout 21600 seconds\nforward_timeout 21600 seconds\n')


def main():
    from pier.environments import agent_setup
    source = Path(inspect.getfile(agent_setup))
    if hashlib.sha256(source.read_bytes()).hexdigest() != AGENT_SETUP_SHA256:
        raise RuntimeError('Unqualified Pier network setup source')
    original = agent_setup.squid_bootstrap_command
    agent_setup.squid_bootstrap_command = lambda: align_proxy_timeout(original())
    from pier.cli.main import app
    app()


if __name__ == '__main__':
    main()

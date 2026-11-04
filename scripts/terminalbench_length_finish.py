"""Opt-in repair of Harbor's length-finish reasoning round trip.

Retain the provider's existing reasoning field when the already-configured
interleaved-thinking path records a truncated answer. The original warning,
parser, retry decorators, output limit and command handling remain unchanged.
Installed only in a fresh worker; never edits the shared Harbor installation.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time


def repaired_function(function, kind):
    original = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    changes = 0
    for node in ast.walk(tree):
        if kind == 'provider' and isinstance(node, ast.If) and ast.dump(node.test) == ast.dump(
                ast.parse("choice.get('finish_reason') == 'length'", mode='eval').body):
            positions = [i for i, value in enumerate(node.body) if isinstance(value, ast.Raise)
                         and isinstance(value.exc, ast.Name) and value.exc.id == 'exc']
            if len(positions) != 1:
                raise ValueError('Unexpected provider length exception path')
            node.body.insert(positions[0], ast.parse('exc.kvmem_reasoning_content = reasoning_content').body[0])
            changes += 1
        if kind == 'agent' and isinstance(node, ast.Call) and ast.dump(node) == ast.dump(
                ast.parse("chat.messages.append({'role': 'assistant', 'content': truncated_response})", mode='eval').body):
            node.args[0] = ast.parse("dict(role='assistant', content=truncated_response, **({'reasoning_content': e.kvmem_reasoning_content} if chat._interleaved_thinking and getattr(e, 'kvmem_reasoning_content', None) else {}))", mode='eval').body
            changes += 1
    if changes != 1:
        raise ValueError('Pinned length-finish repair anchor changed: '+kind)
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, str(Path(__file__))+':'+kind, 'exec'), namespace)
    return namespace[original.__name__]


def install(audit_path, expected_sources):
    from harbor.llms.lite_llm import LiteLLM
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    classes = (LiteLLM, Terminus2)
    actual = {Path(inspect.getfile(cls)).name: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
              for cls in classes}
    if actual != expected_sources:
        raise ValueError('Pinned Harbor length-finish sources changed')
    if getattr(LiteLLM, '_kvmem_length_finish_repaired', False):
        raise RuntimeError('Length-finish hook installed twice')
    # Validate/compile both anchors before publishing either function.
    provider = repaired_function(LiteLLM.call, 'provider')
    agent = repaired_function(Terminus2._query_llm, 'agent')
    LiteLLM.call, Terminus2._query_llm = provider, agent
    LiteLLM._kvmem_length_finish_repaired = True
    Path(audit_path).write_text(json.dumps(dict(at=time.time(),installed=True,sources=actual,
        policy='preserve-provider-reasoning-on-length-finish',agent_limits_changed=False,
        retry_decorators_changed=False,task_or_verifier_changed=False),indent=2)+'\n')

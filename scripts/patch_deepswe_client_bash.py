"""Opt-in pinned benchmark client: preserve Bash command bytes after parsing.

No installed binary is edited. Equal-length replacements preserve Bun offsets.
Schema validation, permission checks, and execution remain in the client.
"""
import argparse
import hashlib
import json
from pathlib import Path


CALL = (b'f=_.replace(`cd ${Y} && `,"");if(o$()==="windows")'
        b'f=f.replace(`cd ${_W(Y)} && `,"");')
PATCH = b'f=_;' + b' ' * (len(CALL) - len(b'f=_;'))
ESCAPE_CALL = rb'f=f.replaceAll("\\\\;","\\;"),'
ESCAPE_PATCH = b' ' * len(ESCAPE_CALL)


def patch(data, expected_sha256):
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Pinned client hash differs; refusing an unknown client')
    sites = []
    result = data
    for before, after in ((CALL, PATCH), (ESCAPE_CALL, ESCAPE_PATCH)):
        if len(before) != len(after) or result.count(before) != 1:
            raise ValueError('Expected exactly one known Bash command rewrite site')
        offset = result.index(before)
        sites.append(dict(offset=offset, length=len(before)))
        result = result[:offset] + after + result[offset + len(before):]
    return result, dict(original_sha256=expected_sha256,
        patched_sha256=hashlib.sha256(result).hexdigest(), sites=sites, bytes=len(result),
        scope='Preserve Bash command strings verbatim after JSON/schema parsing. '
              'Keep permission checks, tool execution and strict KV prefix identity. '
              'Isolated benchmark binary only; no installed executable overwritten.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    value, audit = patch(args.source.read_bytes(), args.sha256)
    with args.output.open('xb') as stream:
        stream.write(value)
    args.output.chmod(0o755)
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2)+'\n')

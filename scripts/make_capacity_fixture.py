"""Derive a labeled capacity/GC diagnostic without modifying native task inputs."""
import argparse
import hashlib
import json
from pathlib import Path

from qualify_engine import read_frozen


def extend_history(source, source_info, target_tokens):
    ids = source['input_ids']
    split = source['query_begin']
    if (type(target_tokens) is not int or not len(ids) < target_tokens <= 262144-8192 or
            not 0 < split < len(ids) or source_info['fixture_kind'] != 'native_history'):
        raise ValueError('Capacity fixture must extend native history, preserve a final query and leave output headroom')
    count = target_tokens-len(ids)
    # Preserve the complete original prompt and query token IDs in order; insert
    # deterministic repeated history immediately before the existing query.
    inserted = (ids[:split] * ((count+split-1)//split))[:count]
    return {'fixture_kind': 'synthetic_capacity',
            'id': f'synthetic-capacity-{target_tokens}-from-{source_info["id"]}',
            'input_ids': ids[:split]+inserted+ids[split:],
            'query_begin': split+count, 'query_end': source['query_end']+count,
            'derivation': {'source': source_info, 'inserted_tokens': count,
                           'method': 'repeat prefix before original query; preserve original token order',
                           'scope': 'GPU capacity/GC correctness only; not a real agent prompt or throughput comparison'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--tokens', type=int, default=245760)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, provenance = read_frozen(args.source, args.source_sha256, 8192)
    fixture = extend_history(source, provenance, args.tokens)
    content = (json.dumps(fixture)+'\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(content)
    print(json.dumps({'path': str(args.output.resolve()), 'sha256': hashlib.sha256(content).hexdigest(),
                      'tokens': args.tokens, 'performance_eligible': False}))


if __name__ == '__main__':
    main()

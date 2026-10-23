#!/usr/bin/env python3
"""Validate saved marker/tool/accepted-token evidence without contacting any service."""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import sys
import time


class Pending(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_of(content):
    if isinstance(content, str):
        return content
    return ''.join(block['text'] for block in (content or []) if block.get('type') == 'text')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode_frames(raw, max_bytes=256 * 1024 * 1024):
    """Decode every concatenated Zstd frame, including frames with unknown size."""
    library = ctypes.util.find_library('zstd')
    if not library:
        raise ValueError('System libzstd is required to inspect complete DSH sessions')
    z = ctypes.CDLL(library)
    declarations = {
        'ZSTD_findFrameCompressedSize': ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t),
        'ZSTD_getFrameContentSize': ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_ulonglong),
        'ZSTD_decompress': ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t),
        'ZSTD_isError': ([ctypes.c_size_t], ctypes.c_uint),
        'ZSTD_getErrorName': ([ctypes.c_size_t], ctypes.c_char_p),
    }
    for name, (args, result) in declarations.items():
        function = getattr(z, name)
        function.argtypes, function.restype = args, result
    offset, total, frames = 0, 0, []
    while offset < len(raw):
        source = ctypes.create_string_buffer(raw[offset:])
        size = z.ZSTD_findFrameCompressedSize(source, len(raw) - offset)
        require(not z.ZSTD_isError(size) and 0 < size <= len(raw) - offset,
                'Incomplete or invalid concatenated Zstd frame')
        expected = z.ZSTD_getFrameContentSize(source, size)
        require(expected != 2**64 - 2, 'Invalid Zstd frame header')
        unknown = expected == 2**64 - 1
        capacity = min(1024 * 1024, max_bytes - total) if unknown else max(1, expected)
        require(0 < capacity <= max_bytes - total, 'Decoded session exceeds inspection limit')
        while True:
            target = ctypes.create_string_buffer(capacity)
            actual = z.ZSTD_decompress(target, capacity, source, size)
            if not z.ZSTD_isError(actual):
                break
            error = z.ZSTD_getErrorName(actual).decode()
            if unknown and 'Destination buffer is too small' in error and capacity < max_bytes - total:
                capacity = min(capacity * 2, max_bytes - total)
                continue
            raise ValueError('Zstd decompression failed: ' + error)
        require(unknown or actual == expected, 'Zstd decoded length differs from its frame header')
        frames.append(target.raw[:actual])
        total += actual
        offset += size
    decoded = b''.join(frames)
    require(decoded.endswith(b'\n'), 'Session ends with an incomplete JSONL record')
    return [json.loads(line) for line in decoded.splitlines() if line.strip()], len(frames)


class Evidence:
    def __init__(self):
        self.hashes = {}

    def bytes(self, path):
        path = Path(path)
        if not path.is_file():
            raise Pending('Missing evidence: ' + str(path))
        value = path.read_bytes()
        self.hashes[str(path)] = sha(value)
        return value

    def json(self, path):
        return json.loads(self.bytes(path))

    def assert_stable(self):
        for name, digest in self.hashes.items():
            path = Path(name)
            if not path.is_file() or sha(path.read_bytes()) != digest:
                raise Pending('Evidence changed during inspection: ' + name)


def validate(case, bridge, tokenizer_dir, evidence):
    exit_info = evidence.json(case / 'exit.json')
    require(exit_info.get('exit_code') == 0, f"DSH exited with code {exit_info.get('exit_code')}")
    expected = evidence.json(case / 'expected.json')
    marker, relative = expected['marker'], expected['path']
    require(expected.get('required_tool') == 'read', 'This validator requires a read-tool marker case')
    workspace = (case / 'workspace').resolve()
    marker_path = (workspace / relative).resolve()
    require(marker_path.is_relative_to(workspace), 'Expected marker escapes its workspace')
    require(evidence.bytes(marker_path).decode() == marker + '\n', 'Marker file changed')
    require(evidence.bytes(case / 'stdout.log').decode() == marker + '\n', 'Final stdout differs from marker')
    task = evidence.bytes(case / 'task.txt').decode().strip()
    paths = sorted((case / 'home').rglob('session.jsonl.zstd'))
    if not paths:
        raise Pending('No durable DSH session exists yet')
    require(len(paths) == 1, 'Marker case must own exactly one durable session')
    rows, frames = decode_frames(evidence.bytes(paths[0]))
    headers = [row for row in rows if row.get('type') == 'session']
    require(len(headers) == 1, 'Expected one session header across all Zstd frames')
    header = headers[0]
    session_id = header['id']
    require(Path(header['cwd']).resolve() == workspace, 'Session cwd differs from marker workspace')
    user_rows = [r['data'] for r in rows if r.get('type') == 'user/message']
    humans = [r for r in user_rows if r.get('source', {}).get('kind') == 'user']
    require(len(humans) == 1 and text_of(humans[0]['content']) == task, 'Durable human task is ambiguous or changed')
    human = humans[0]
    query_ordinal = user_rows.index(human)
    calls = [r for r in rows if r.get('type') == 'tool/call']
    require(len(calls) == 1 and calls[0]['data']['name'] == 'read', 'Expected exactly one executed read call')
    call = calls[0]['data']
    arguments = json.loads(call['arguments']) if isinstance(call['arguments'], str) else call['arguments']
    read_path = Path(arguments['file_path'])
    require((read_path if read_path.is_absolute() else workspace / read_path).resolve() == marker_path,
            'Executed read call targeted another file')
    results = [r for r in rows if r.get('type') == 'tool/result']
    require(len(results) == 1, 'Expected one executed tool result')
    result = results[0]['data']
    source = result['message']['source']
    require(source.get('kind') == 'tool' and source.get('callId') == call['callId'], 'Tool result source does not match call')
    blocks = result['message']['content']
    require(len(blocks) == 1 and blocks[0].get('type') == 'tool-result'
            and blocks[0].get('toolCallId') == call['callId'] and blocks[0].get('isError') is False,
            'Read did not return a successful matching tool result')
    require(marker in text_of(blocks[0].get('content')), 'Tool result text lacks marker')
    meta = result['meta']
    require(Path(meta['path']).resolve() == marker_path and meta['totalLines'] == 1
            and meta['lines'] == [{'number': 1, 'text': marker}], 'Read result metadata differs from exact marker file')
    assistants = [r for r in rows if r.get('type') == 'assistant/message']
    require(len(assistants) == 2, 'Marker expected one tool-call response and one final response')
    final = assistants[-1]
    require(text_of(final['data']['message']['content']) == marker, 'Durable final assistant text differs from marker')
    require(calls[0]['seq'] < results[0]['seq'] < final['seq'], 'Tool execution does not precede final answer')
    endings = [r for r in rows if r.get('type') == 'turn/end']
    require(len(endings) == 1 and endings[0]['data']['reason']['kind'] == 'completed', 'DSH turn did not complete')

    matching = []
    for request_path in sorted(bridge.glob('*/*/request.json')):
        # Unrelated sessions are not part of the evidence set.
        body = json.loads(request_path.read_bytes())
        if body.get('prompt_cache_key') == session_id:
            matching.append(request_path.parent)
    require(len(matching) == 2, 'Expected exactly two bridge requests for this DSH session')
    require([p.name for p in matching] == ['00001', '00002'], 'Bridge turns are not a fresh contiguous marker trajectory')
    owner = matching[0].parent.name
    require(all(p.parent.name == owner for p in matching), 'Marker requests use different native owners')
    from lifecycle_probe import source_owner
    namespace = owner.rsplit('-', 1)[0]
    turns = []
    for index, directory in enumerate(matching, 1):
        body, plan, accepted, response, native = [evidence.json(directory / name) for name in
            ('request.json', 'plan.json', 'accepted.json', 'response.json', 'native.json')]
        require(source_owner(body, namespace) == owner == plan.get('session_id') == plan.get('cache_salt'),
                'Native owner differs from exact DSH key/system/tool frame')
        proof = body.get('kvmem_query_provenance', {})
        wire_users = [m for m in body['messages'] if m['role'] == 'user']
        hashes = [sha(text_of(m.get('content')).encode()) for m in wire_users]
        digest = sha(json.dumps(hashes, separators=(',', ':')).encode())
        require(proof.get('schema_version') == 1 and proof.get('session_id') == session_id
                and proof.get('user_messages_sha256') == digest
                and proof.get('query_user_ordinal') == query_ordinal
                and proof.get('query_message_id') == human['id'], 'Wire Query provenance differs from durable human source')
        require([text_of(m.get('content')) for m in wire_users] == [text_of(m['content']) for m in user_rows],
                'Wire user messages differ from durable human/runtime-context sequence')
        source_proof = plan.get('query_source', {})
        require(source_proof.get('status') == 'verified-user-source'
                and source_proof.get('message_id') == human['id']
                and source_proof.get('user_ordinal') == query_ordinal
                and source_proof.get('query_span') == plan.get('kvmem_query_span')
                and plan.get('kvmem_query_id') == human['id'], 'Native Query provenance differs from wire/durable source')
        ids = accepted.get('token_ids')
        require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids), 'Invalid accepted token IDs')
        require(len(ids) == response['usage']['completion_tokens'] == native['usage']['completion_tokens']
                and ids == native['token_ids'], 'Accepted commit differs from native output or response usage')
        segments = [evidence.json(p) for p in sorted(directory.glob('segment-[0-9][0-9][0-9].json'))]
        require(segments and [s['index'] for s in segments] == list(range(len(segments)))
                and [i for s in segments for i in s['token_ids']] == ids,
                'Accepted IDs differ from complete native segment evidence')
        require(plan.get('exact_output_prefix_reused') is (index == 2), 'Unexpected recorded accepted-prefix reuse')
        turns.append(dict(body=body, plan=plan, accepted=accepted, response=response, native=native))
    first_message = turns[0]['response']['choices'][0]['message']
    wire_calls = first_message.get('tool_calls', [])
    require(len(wire_calls) == 1 and wire_calls[0]['id'] == call['callId']
            and wire_calls[0]['function']['name'] == 'read'
            and json.loads(wire_calls[0]['function']['arguments']) == arguments,
            'Committed bridge tool call differs from actual DSH execution')
    tool_wire = [m for m in turns[1]['body']['messages'] if m['role'] == 'tool']
    require(len(tool_wire) == 1 and tool_wire[0]['tool_call_id'] == call['callId']
            and text_of(tool_wire[0]['content']) == text_of(blocks[0]['content']),
            'Second bridge request lacks exact executed tool result')
    require(turns[1]['response']['choices'][0]['message'].get('content') == marker,
            'Committed bridge final text differs from marker')

    replay = replay_prefix(turns, bridge, tokenizer_dir, evidence)
    before = case / 'lifecycle-before-release.json'
    gc = evidence.json(before).get('gc_evidence') if before.is_file() else None
    evidence.assert_stable()
    return dict(status='passed', passed=True, session_id=session_id, owner=owner,
                native_namespace=[owner, owner], complete_zstd_frames=frames, durable_rows=len(rows),
                actual_read=dict(call_id=call['callId'], path=str(marker_path), result_seq=results[0]['seq']),
                final_marker=marker, exit_code=0, query_message_id=human['id'], query_user_ordinal=query_ordinal,
                query_spans=[t['plan']['kvmem_query_span'] for t in turns],
                accepted_output_tokens=[len(t['accepted']['token_ids']) for t in turns],
                prompt_tokens=[t['plan']['prompt_tokens'] for t in turns], accepted_prefix_verification=replay,
                gc_observation_before_release=gc, proactive_gc_proven=False,
                limits='Proves actual read/tool roundtrip, Query identity and accepted-ID reconstruction. Does not prove proactive GC, GPU page identity or a throughput effect.')


def replay_prefix(turns, bridge, tokenizer_dir, evidence):
    """Rebuild token plans using local tokenizer assets, without model or CUDA code."""
    os.environ['USE_TORCH'] = '0'
    os.environ['USE_TF'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from transformers import AutoTokenizer
    from benchmarks.deepswe.claude_bridge import Planner, Trajectory
    from scripts.dsh_bridge import to_anthropic
    from scripts.dsh_model_protocol import TokenizerProtocol
    from scripts.dsh_query_epoch import apply_query_provenance
    manifest = evidence.json(bridge / 'protocol.json')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    protocol = TokenizerProtocol(tokenizer, tokenizer_dir,
        reasoning_effort=manifest['chat_template_kwargs']['reasoning_effort'])
    require(protocol.provenance == manifest, 'Local tokenizer/template differs from saved bridge protocol')
    planner, state, prefix = Planner(protocol, tokenizer, 4096), Trajectory(), None
    reports = []
    for index, turn in enumerate(turns, 1):
        payload = to_anthropic(turn['body'])
        prepared = planner.prepare(payload, state)
        proof = apply_query_provenance(turn['body'], prepared)
        plan, tokens = turn['plan'], prepared[0]
        require(len(tokens) == plan['prompt_tokens'] and prepared[5] == plan['exact_output_prefix_reused'],
                'Rebuilt token plan differs from recorded prompt/reuse state')
        for key in ('kvmem_query_span', 'kvmem_query_id', 'kvmem_locality_spans', 'kvmem_locality_provenance'):
            require(prepared[1].get(key) == plan.get(key), 'Rebuilt plan differs: ' + key)
        require(proof == plan['query_source'], 'Rebuilt Query source differs from saved source')
        if prefix is not None:
            require(tokens[:len(prefix)] == prefix, 'Second prompt does not retain first prompt plus exact accepted IDs')
        reports.append(dict(turn=index, prompt_tokens=len(tokens),
                            prompt_ids_sha256=sha(json.dumps(tokens, separators=(',', ':')).encode()),
                            previous_prompt_and_accepted_prefix_tokens=0 if prefix is None else len(prefix),
                            exact_output_prefix_reused=prepared[5]))
        content = to_anthropic(dict(model=turn['body']['model'],
            messages=[turn['response']['choices'][0]['message']]))['messages'][0]['content']
        planner.commit(state, prepared, payload, dict(content=content), turn['accepted']['token_ids'])
        prefix = tokens + turn['accepted']['token_ids']
    return dict(passed=True, local_assets_only=True, turns=reports)


def inspect(case, bridge, tokenizer_dir):
    evidence = Evidence()
    result = dict(schema_version=1, read_only=True, checked_unix=time.time(),
                  case_dir=str(case), bridge_output=str(bridge))
    try:
        result.update(validate(case, bridge, tokenizer_dir, evidence))
    except Pending as error:
        result.update(status='pending', passed=False, reason=str(error))
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        result.update(status='failed', passed=False, reason=f'{type(error).__name__}: {error}')
    result['evidence_sha256'] = evidence.hashes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--bridge-output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case, bridge, tokenizer = (p.resolve() for p in (args.case_dir, args.bridge_output, args.tokenizer))
    output = args.output.resolve()
    require(output not in (case / 'exit.json', case / 'expected.json'), 'Output cannot overwrite source evidence')
    result = inspect(case, bridge, tokenizer)
    require(str(output) not in result['evidence_sha256'], 'Output cannot overwrite source evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(output)
    print(json.dumps(dict(status=result['status'], passed=result['passed'], output=str(output), reason=result.get('reason'))))
    return {'passed': 0, 'failed': 1, 'pending': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())

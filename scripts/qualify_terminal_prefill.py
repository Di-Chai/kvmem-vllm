"""GPU regression for a near-full terminal DFlash window and the next tool input."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import uuid
from pathlib import Path

from qualify_block_graph import (expected_output_samples, native_matches, records,
    require, save, stream_request, wait_released, wait_terminal_publication)


def run(args):
    profile = json.loads(args.config.read_text())
    kv = profile['kvmem']
    require(profile['verify_session_restore'] and profile['draft_method'] == 'dflash' and
        profile['draft_tokens'] == 7 and profile['execution_mode'] == 'full_graph' and
        kv['rolling_active_window'] and kv['block_tokens'] == 32 and
        kv['select_budget'] == 36864 and kv['gen_budget'] == 16384,
        'Require byte-verified DFlash7 FULL graphs and the actual DSH K/G geometry')
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix = encode('<|im_start|>system\nContinue the immutable ledger in detail. Thinking is disabled.'
        '<|im_end|>\n<|im_start|>user\n')
    footer = encode('\nExplain the ledger for at least 1000 words.<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\n')
    filler = encode('The immutable archive retains each accepted public token and its exact owner.\n')
    size = 53000 - len(prefix) - len(footer)
    tokens = prefix + (filler * ((size + len(filler) - 1) // len(filler)))[:size] + footer
    sessions = ['terminal-prefill-' + uuid.uuid4().hex for _ in range(2)]
    rows, proofs, errors = [], [], []
    started = time.monotonic()

    def client(index):
        # Stop immediately before an unsealed or sealed capacity checkpoint.
        count = (213, 217)[index]
        first = stream_request(profile, sessions[index], tokens, count, count, reselect='off')
        require(len(first['output_ids']) == count, 'Initial fixed output quota differs')
        rows.append(first)
        save(args.output / f'owner-{index}-first.json', first)
        wait_terminal_publication(args.capture, [first])
        position = len(tokens) + count - 1
        terminal = [q for q in records(args.capture / 'checkpoint_timing.jsonl')
            if native_matches(first, q) and q.get('terminal')]
        require(terminal and terminal[-1]['position'] == position,
            'Did not reach the intended terminal capacity position')
        require(not [q for q in records(args.capture / 'bounded_refresh.jsonl') if native_matches(first, q)],
            'Initial fixture selected before reaching the intended full window')
        next_tokens = tokens + first['output_ids']
        tool = encode('<|im_end|>\n<|im_start|>tool\nActual next file evidence: ')
        tail = encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
        remaining = 53840 - len(next_tokens) - len(tool) - len(tail)
        next_tokens += tool + (filler * ((remaining + len(filler) - 1) // len(filler)))[:remaining] + tail
        second = stream_request(profile, sessions[index], next_tokens, 257, 257, reselect='off')
        rows.append(second)
        save(args.output / f'owner-{index}-second.json', second)
        wait_terminal_publication(args.capture, [second])
        expected_window = 36864 - (-position % 32)
        state = [q for q in records(args.capture / 'verified_state_restores.jsonl') if native_matches(second, q)]
        byte = [q for q in records(args.capture / 'verified_restores.jsonl') if native_matches(second, q)]
        require(state and state[-1]['pressure_restore'] and state[-1]['recurrent_byte_equal'] and
            state[-1]['working_tokens'] == expected_window and state[-1]['history_tokens'] == position,
            'Near-full terminal was not pressure-selected at the exact fenced import')
        require(byte and byte[-1]['byte_equal'] and byte[-1]['draft_layers'] == 5 and
            byte[-1]['oracle_kind'] == 'fp8_immutable_rebuild',
            'Missing independent FP8 target, GDN and five-layer draft restore byte checks')
        require(not [q for q in records(args.capture / 'bounded_refresh.jsonl') if native_matches(second, q)],
            'Short continuation required an unexpected second selection')
        for row, prompt in ((first, tokens), (second, next_tokens)):
            action = [q for q in records(args.capture / 'action_attention.jsonl') if native_matches(row, q)]
            expected = expected_output_samples(tokenizer, profile, prompt, row['output_ids'])
            require(len(action) == 1 and action[0]['samples'] == expected,
                'Accepted output attention differs from the independent token parser oracle')
        proofs.append(dict(owner=index, terminal_history=position,
            restored_window=expected_window, appended_input=len(next_tokens) - position,
            target_gdn_draft_byte_equal=True, accepted_attention_matches=True))

    report = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(client, i) for i in range(2)]
            for future in futures:
                future.result()
        report = dict(status='passed', proofs=proofs,
            scope='C2 terminal capacity boundary and next tool prefill; no task-quality or throughput claim')
    except BaseException as error:
        errors.append(repr(error))
        raise
    finally:
        for session in sessions:
            try:
                stream_request(profile, session, [tokenizer.eos_token_id], 1, 1, release=True)
            except Exception as error:
                errors.append('release: ' + repr(error))
        try:
            wait_released(args.capture, sessions, timeout=60)
        except Exception as error:
            errors.append('release check: ' + repr(error))
        report.update(errors=errors, rows=rows, seconds=time.monotonic() - started,
            all_owned_sessions_released=not any(e.startswith('release') for e in errors))
        if errors:
            report['status'] = 'failed'
        save(args.output / 'results.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'capture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())

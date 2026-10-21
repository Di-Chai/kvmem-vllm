"""Exercise qw3-style agent refreshes on real DFlash/GDN/FP8 state."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import uuid

from qualify_block_graph import (
    build_fixture, expected_output_samples, native_matches, records, require,
    save, stream_request, wait_released, wait_terminal_publication,
)


def run(args):
    profile = json.loads(args.config.read_text())
    kv = profile['kvmem']
    require(kv['guided_reselect'] == 'both' and profile['verify_session_restore'],
            'Use both triggers and exact native-state verification')
    from vllm_kvmem.options import RuntimeOptions
    from types import SimpleNamespace as NS
    options = RuntimeOptions.from_vllm(NS(kv_transfer_config=NS(
        kv_connector_extra_config={'kvmem': kv}), cache_config=NS(cache_dtype='fp8_e4m3')))
    trigger = options.guided_trigger
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    fixture = build_fixture(tokenizer, prompt_tokens=kv['select_budget']+1024, concurrency=2)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'fixture.json', fixture)
    sessions = ['guided-epoch-'+uuid.uuid4().hex for _ in range(3)]
    rows, errors = [], []

    def send(index, turn, prompt, query, spans, count=129, minimum=129):
        row = stream_request(profile, sessions[index], prompt, count, minimum,
            query=query, spans=spans, reselect='auto', query_mode='agent',
            query_id='task-'+str(index)+('-new' if turn == 'new-user' else ''))
        row.update(agent=index, turn=turn,
                   expected_output_samples=expected_output_samples(tokenizer, profile, prompt, row['output_ids']))
        rows.append(row)
        save(args.output/f'owner-{index}-{turn}.json', row)
        return row

    def tool(prompt, output, count, spans):
        prefix = list(prompt)+output
        begin = len(prefix)
        header = encode('<|im_end|>\n<|im_start|>tool\n')
        tail = encode('\n<|im_end|>\n<|im_start|>assistant\n<think>\n')
        material = encode('The current test succeeded. Keep the original identifiers and continue the unfinished verification. ')
        middle = (material*(count//len(material)+1))[:count]
        result = prefix+header+middle+tail
        return result, [*spans, dict(begin=begin, end=len(prefix+header+middle), role='tool')]

    def client(index):
        agent = fixture['agents'][index]
        prompt, query, spans = agent['initial_tokens'], agent['query_span'], agent['message_spans']
        row = send(index, 'initial', prompt, query, spans)
        prompt, spans = tool(prompt, row['output_ids'], trigger+128, spans)
        row = send(index, 'growth', prompt, query, spans)
        prompt, spans = tool(prompt, row['output_ids'], 128, spans)
        row = send(index, 'stable', prompt, query, spans)
        prompt, spans = tool(prompt, row['output_ids'], kv['gen_budget']+512, spans)
        row = send(index, 'large-tool', prompt, query, spans)
        prefix = prompt+row['output_ids']+encode('<|im_end|>\n<|im_start|>user\n')
        question = encode('Give one concise sentence explaining which verification is still unfinished.')
        begin = len(prefix)
        prompt = prefix+question+encode('<|im_end|>\n<|im_start|>assistant\n<think>\n')
        send(index, 'new-user', prompt, [begin, begin+len(question)], spans)

    started = time.monotonic()
    report = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(client, i) for i in range(2)]:
                future.result()
        agent = fixture['agents'][0]
        prefix = agent['initial_tokens'][:agent['query_span'][0]]
        question = encode('Write a very long numbered verification checklist. Each item must end with a full stop. '
                          'Discuss tests and observations in plain prose. Continue for several thousand words.')
        query = [len(prefix), len(prefix)+len(question)]
        prompt = prefix+question+agent['footer']
        send(2, 'decode', prompt, query, (), count=kv['gen_budget'], minimum=kv['gen_budget']-32)
        wait_terminal_publication(args.capture, rows)
        refreshes = records(args.capture/'bounded_refresh.jsonl')
        actions = records(args.capture/'action_attention.jsonl')
        rollback = records(args.capture/'verified_guided_rollback.jsonl')
        evidence = []
        for row in rows:
            events = [e for e in refreshes if native_matches(row, e)]
            guided = [e for e in events if e.get('guided_trigger_reason') not in (None, 'explicit-query')]
            if row['turn'] in ('initial', 'new-user'):
                require(not guided and not any(e['guided_query_tokens'] for e in events),
                        'A real new user Query generated a private Query')
            elif row['turn'] == 'stable':
                require(not events, 'Stable tool callback reselected before its growth threshold')
            else:
                require(guided, 'Expected automatic refresh was absent: '+row['turn'])
                if row['turn'] != 'decode':
                    require(len(guided) == 1, 'A tool callback performed duplicate semantic selections')
                require(all(e.get('guided_published_tokens', 0) > 0 or
                            (e.get('guided_original_seed_bytes', 0) > 0 and
                             e.get('guided_fallback') in ('empty-output', 'unterminated-output',
                                'structured-output', 'capacity', 'emergency-original-query'))
                            for e in guided), 'Guided fallback lost its original task Query')
                if row['turn'] == 'growth':
                    require(not any(e['selection'] == 'pressure' for e in events),
                            'Normal guided refresh first discarded the old window through pressure')
                if row['turn'] == 'large-tool':
                    require(any(e['selection'] == 'pressure' for e in events),
                            'Oversized tool result missed bounded pressure ingestion')
            for event in guided:
                require(event['selected_tokens'] <= kv['select_budget'], 'Selected more than K')
                if event['guided_query_tokens']:
                    require(event.get('guided_reference_token_equal') and event.get('guided_reference_q_equal'),
                            'Private graph differs from eager generation')
                    require(any(native_matches(row, e) and e['recurrent_byte_equal'] and e['draft_byte_equal']
                                for e in rollback), 'Missing native rollback proof')
            own = [a for a in actions if native_matches(row, a)]
            require(len(own) == 1 and own[0]['samples'] == row['expected_output_samples'],
                    'Private/rejected rows altered accepted attention')
            evidence.append(dict(agent=row['agent'], turn=row['turn'], events=events))
        require(any(e.get('guided_published_tokens', 0) > 0
                    for row in evidence for e in row['events']),
                'Guided enabled but no generated Query Q was used')
        report.update(status='passed', evidence=evidence)
    except BaseException as error:
        errors.append(repr(error))
        raise
    finally:
        for session in sessions:
            try:
                stream_request(profile, session, [fixture['eos_token_id']], 1, 1, release=True)
            except Exception as error:
                errors.append('release: '+repr(error))
        try:
            wait_released(args.capture, sessions)
        except Exception as error:
            errors.append('release audit: '+repr(error))
        report.update(rows=rows, errors=errors, seconds=time.monotonic()-started)
        if errors:
            report['status'] = 'failed'
        save(args.output/'results.json', report)
    print(json.dumps(dict(status=report['status'], cases=len(rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'capture', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())

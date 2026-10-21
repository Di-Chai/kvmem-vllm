"""Prepare and run all 13 classic RULER tasks with independent KVMem owners."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace as NS
import uuid

import ruler_protocol as protocol


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def render_case(tokenizer, task, input_text):
    public = protocol.public_query(task, input_text)
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': input_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if not rendered.endswith('<think>\n'):
        raise ValueError('Expected the qualified Qwen open-thinking chat template')
    offset = rendered.find(input_text)
    if offset < 0 or rendered.find(input_text, offset + 1) >= 0:
        raise ValueError('Chat template did not preserve a unique complete input')
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    begin, end = [offset + position for position in public['char_span']]
    overlapping = [i for i, (a, b) in enumerate(encoded['offset_mapping'])
                   if a < end and b > begin]
    if not overlapping:
        raise ValueError('Public Query has no token coordinates')
    span = [overlapping[0], overlapping[-1] + 1]
    ids = encoded['input_ids']
    decoded = tokenizer.decode(ids[slice(*span)], skip_special_tokens=False)
    if decoded.strip() != public['text'].strip():
        raise ValueError('Public Query tokens include history or omit question content')
    return ids, span, decoded, rendered


def write_report(path, fixture, scores, results, elapsed):
    lines = ['# RULER v1 256K：13 类各 20 题', '',
        f"完成 {len(results)} 题，官方规则宏平均分 **{scores['macro_score']:.2f}%**。", '',
        '采用修复后四项优化全开、guided 关闭、累计 attention pressure、V2 multi Query 图传播（α=0.2）。',
        'NVFP4 / FP8 KV / MTP4；K=36,864，G=16,384。', '',
        '256K 是官方生成器长度档位，下面列出实际完整输入长度。使用官方子串评分，',
        '只对最终答案评分；整题目标命中不等于严格 exact match。每类 20 题属于初筛样本。', '',
        '| 任务 | 题数 | 分数 | 整题目标命中 | 实际输入 tokens 范围 |',
        '|---|---:|---:|---:|---:|']
    for task in protocol.TASK_FAMILIES:
        row = scores['per_task'][task]
        lengths = [r['served_input_tokens'] for r in fixture['rows'] if r['task'] == task]
        lines.append(f"| {task} | {row['cases']} | {row['score']:.2f}% | "
                     f"{row['full_reference_recall_cases']} | {min(lengths):,}–{max(lengths):,} |")
    lines += ['', f'客户端推理总用时：{elapsed / 60:.1f} 分钟（包含 prefill、重选和资源退还）。',
        f"输出达到上限的题数：{sum(r['finish_reason'] == 'length' for r in results)}。",
        f"思考未闭合的题数：{sum(r['thinking_complete'] is False for r in results)}。", '',
        '冷准备耗时与最终状态见 ../../status.json，全局执行审计见 ../execution-audit.json。',
        '本轮仅运行一个配置，不能单凭这些分数推断优化带来的准确率增减。', '']
    Path(path).write_text('\n'.join(lines))


def summarize_result(row, session, input_tokens, response, evidence, scorer, *, starts_in_thinking):
    answer = protocol.final_answer(response['text'], starts_in_thinking=starts_in_thinking)
    config = scorer[0][protocol.TASK_FAMILIES[row['task']]]
    score = config['metric_fn']([scorer[1](answer['text'], config)], [row['outputs']])
    # Pressure reports only describe cumulative-attention selection. Selection
    # differences against unpropagated scores exist on semantic Query reports.
    query_graph = [g for g in evidence['graph'] if g['phase'] == 'query']
    return dict(id=row['id'], task=row['task'], index=row['index'], status='completed',
        session=session, input_sha256=row['served_sha256'], input_tokens=input_tokens,
        completion_tokens=len(response['output_ids']), seconds=response['seconds'],
        ttft_seconds=response['ttft_seconds'], finish_reason=response['finish_reason'],
        response_id=response['response_id'], text=response['text'], answer=answer['text'],
        thinking_complete=answer['thinking_complete'], official_case_score=score,
        pressure_count=len(evidence['pressure']),
        graph_added_blocks=sum(g['added_blocks'] for g in query_graph))


def prepare(args):
    import numpy as np
    import yaml
    from transformers import AutoTokenizer
    import guided_benchmark_protocol as policy

    root = args.campaign.resolve()
    upstream = args.upstream.resolve()
    profile = json.loads(args.config.read_text())
    policy.validate_profile(profile, 'off', 'niah')
    protocol.load_scorer(upstream)
    if (root / 'fixture.json').exists():
        raise ValueError('Existing frozen fixture: use a fresh campaign or run it')
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
    wrapper_tokens = len(tokenizer.apply_chat_template([{'role': 'user', 'content': ''}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True))
    tree = ast.parse((upstream / 'scripts/data/synthetic/constants.py').read_text())
    constants = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TASKS'
                                                for t in node.targets))
    tasks = yaml.safe_load((upstream / 'scripts/synthetic.yaml').read_text())
    if set(tasks) != set(protocol.TASK_FAMILIES):
        raise ValueError('Upstream is not the frozen 13-task configuration')
    raw = root / 'dataset/raw'
    environment = dict(os.environ, TOKENIZERS_PARALLELISM='false',
        NLTK_DATA=str(root / 'corpus-cache/nltk_data'), PYTHONDONTWRITEBYTECODE='1')
    plan = dict(samples_per_task=args.samples, target_length=args.length, seed=args.seed,
        model_template_tokens=wrapper_tokens, tokenizer=profile['tokenizer'],
        tokenizer_sha256=sha(Path(profile['tokenizer']) / 'tokenizer.json'),
        upstream=protocol.UPSTREAM_COMMIT)
    previous_plan = root / 'generation-plan.json'
    if previous_plan.exists() and json.loads(previous_plan.read_text()) != plan:
        raise ValueError('Dataset generation settings changed during preparation')
    save(previous_plan, plan)

    def generate(task):
        config = tasks[task]
        base = constants[config['task']]
        destination = raw / task / 'validation.jsonl'
        done = raw / task / 'generated.json'
        if done.exists():
            if json.loads(done.read_text())['sha256'] != sha(destination):
                raise ValueError('Prepared upstream dataset changed: ' + task)
            return task
        command = [sys.executable, '-B', str(upstream / 'scripts/data/synthetic' / (config['task'] + '.py')),
            '--save_dir', str(raw), '--save_name', task, '--subset', 'validation',
            '--tokenizer_path', profile['tokenizer'], '--tokenizer_type', 'hf',
            '--max_seq_length', str(args.length), '--num_samples', str(args.samples),
            '--tokens_to_generate', str(base['tokens_to_generate']), '--random_seed', str(args.seed),
            '--model_template_token', str(wrapper_tokens),
            '--template', base['template'] + base['answer_prefix']]
        for key, value in config['args'].items():
            command += ['--' + key, str(value)]
        destination.parent.mkdir(parents=True, exist_ok=True)
        save(destination.parent / 'command.json', command)
        with (destination.parent / 'generate.log').open('w') as log:
            subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)
        with destination.open() as stream:
            count = sum(1 for _ in stream)
        if count != args.samples:
            raise ValueError('Wrong generated sample count: ' + task)
        save(done, dict(sha256=sha(destination), cases=count))
        print(json.dumps(dict(task=task, status='generated', cases=count)), flush=True)
        return task

    with ThreadPoolExecutor(max_workers=args.prepare_workers) as pool:
        list(pool.map(generate, protocol.TASK_FAMILIES))
    rows = []
    inputs = root / 'dataset/inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    for task in protocol.TASK_FAMILIES:
        with (raw / task / 'validation.jsonl').open() as stream:
            for ordinal, line in enumerate(stream):
                source = json.loads(line)
                tokens, query, question, rendered = render_case(tokenizer, task, source['input'])
                # Official FWE adjusts by max_length // 32 words. Random
                # vocabulary tokenization can leave more than 5% slack.
                minimum_ratio = .90 if task == 'fwe' else .95
                if not minimum_ratio * args.length <= len(tokens) <= args.length:
                    raise ValueError(f'Unexpected {task} context length: {len(tokens)}; do not truncate or pad')
                case_id = f'{task}-{ordinal:04d}'
                token_path = inputs / (case_id + '.i32')
                np.asarray(tokens, dtype='<i4').tofile(token_path)
                # Upstream NIAH overwrites index with the answer character offset.
                # Use the file ordinal for identity; retain its index as metadata.
                rows.append(dict(source, task=task, index=ordinal, upstream_index=source['index'],
                    id=case_id, served_input_tokens=len(tokens), served_query=query, question=question,
                    served_sha256=sha(token_path), rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                    upstream_row_sha256=hashlib.sha256(line.encode()).hexdigest(), seed=args.seed + ordinal))
        print(json.dumps(dict(task=task, status='packed')), flush=True)
    cohort = protocol.validate_cohort(rows, args.samples)
    fixture = dict(plan, **cohort, rows=rows, input_directory=str(inputs), upstream_path=str(upstream),
        profile_sha256=sha(args.config), workers=8, cpu_payload_credits_bytes=100000000000 - 16 * 1024**3,
        request_timeout_seconds=3600, starts_in_thinking=True,
        sampling=dict(max_tokens=8192, temperature=1., top_p=.95, top_k=20, min_p=0,
                      presence_penalty=0, repetition_penalty=1, thinking_token_budget=4096),
        prompt_protocol='Unmodified upstream public input in Qwen user chat; open thinking; no assistant answer prefix.',
        protocol_difference='Same generation settings as the existing C-semantic NIAH campaign; not original BF16 greedy decoding.',
        corpus_manifest_sha256=sha(root / 'corpus-cache/corpora-manifest.json'),
        raw_data_hashes={task:sha(raw / task / 'validation.jsonl') for task in protocol.TASK_FAMILIES},
        source_hashes={str(p.relative_to(upstream)):sha(p) for p in (upstream / 'scripts').rglob('*.py')})
    save(root / 'fixture.json', fixture)
    save(root / 'dataset-summary.json', dict(**cohort, target_length=args.length,
        lengths={task:dict(min=min(r['served_input_tokens'] for r in rows if r['task'] == task),
                          max=max(r['served_input_tokens'] for r in rows if r['task'] == task),
                          mean=statistics.fmean(r['served_input_tokens'] for r in rows if r['task'] == task))
                 for task in protocol.TASK_FAMILIES}, fixture_sha256=sha(root / 'fixture.json')))
    print(json.dumps(dict(status='prepared', **cohort)), flush=True)


def run(args):
    import numpy as np
    from tokenizers import Tokenizer
    from vllm_kvmem.offline_queue import OfflineQueue, OfflineTask
    import guided_benchmark_protocol as policy
    import qualify_block_graph as q
    import qualify_graph_niah as old
    from benchmark_client_io import CaptureSlice

    root = args.campaign.resolve()
    profile = json.loads((root / 'profile.json').read_text())
    fixture = json.loads((root / 'fixture.json').read_text())
    diagnostic_gen_budget = getattr(args, 'diagnostic_gen_budget', 16384)
    diagnostic_chunk = getattr(args, 'diagnostic_chunk', 1024)
    q.require((diagnostic_gen_budget == 16384 and diagnostic_chunk == 1024) or
              getattr(args, 'diagnostic_cases', None) is not None,
              'Changing budget or chunk is restricted to an explicit diagnostic cohort')
    policy.validate_profile(profile, 'off', 'niah', expected_gen_budget=diagnostic_gen_budget,
                            expected_chunk=diagnostic_chunk)
    q.require(sha(root / 'profile.json') == fixture['profile_sha256'], 'Frozen profile changed')
    tokenizer_path = Path(profile['tokenizer']) / 'tokenizer.json'
    q.require(sha(tokenizer_path) == fixture['tokenizer_sha256'], 'Tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    scorer = protocol.load_scorer(Path(fixture['upstream_path']))
    protocol.validate_cohort(fixture['rows'], fixture['samples_per_task'])
    capture, output = root / 'run/server/capture', root / 'run/answers'
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'launch-audit.json', old.audit_launch(profile, capture))
    # Round robin tasks so early progress covers the whole suite.
    rows = sorted(fixture['rows'], key=lambda r: (r['index'], list(protocol.TASK_FAMILIES).index(r['task'])))
    diagnostic_cases = getattr(args, 'diagnostic_cases', None)
    if diagnostic_cases is not None:
        q.require(1 <= diagnostic_cases < len(rows), 'Diagnostic must be a proper fixed prefix of the suite')
        rows = rows[:diagnostic_cases]
    queue = OfflineQueue([OfflineTask(r['id'], r, old.history_bytes(r, fixture, profile)) for r in rows],
        max_inflight=fixture['workers'], history_capacity_bytes=fixture['cpu_payload_credits_bytes'])

    def infer(task):
        row = task.payload
        directory = output / 'cases' / row['id']
        directory.mkdir(parents=True, exist_ok=False)
        source = Path(fixture['input_directory']) / (row['id'] + '.i32')
        q.require(sha(source) == row['served_sha256'], 'Frozen prompt IDs changed')
        tokens = np.fromfile(source, dtype='<i4').tolist()
        q.require(len(tokens) == row['served_input_tokens'] and
            tokenizer.decode(tokens[slice(*row['served_query'])], skip_special_tokens=False) == row['question'],
            'Public Query coordinates do not match actual prompt')
        session = 'ruler-' + uuid.uuid4().hex
        sampling = dict(fixture['sampling'], seed=row['seed'])
        save(directory / 'request.json', dict(sampling, session_id=session,
            prompt_sha256=row['served_sha256'], prompt_tokens=len(tokens), kvmem_query_span=row['served_query']))
        capture_slice = CaptureSlice(capture / name for name in
            ('bounded_refresh.jsonl', 'block_graph.jsonl', 'action_attention.jsonl'))
        response = old.stream(profile, session, tokens, row['served_query'], sampling,
                              fixture['request_timeout_seconds'])
        save(directory / 'response.json', response)
        return session, tokens, response, capture_slice

    def finish(task, pending):
        row = task.payload
        session, tokens, response, capture_slice = pending
        directory = output / 'cases' / row['id']
        terminal = old.wait_retired(capture, response['response_id'])
        q.require(terminal['history_end'] == len(tokens) + len(response['output_ids']) - 1,
                  'Accepted input/output history differs')
        q.require(not terminal.get('checkpoint_tokens'), 'Independent RULER case retained history')
        release = old.wait_owner_released(capture, session, response, terminal)
        evidence = policy.audit_case(capture, response['response_id'], profile,
                                     row['served_query'], require_pressure=True, event_reader=capture_slice.read)
        action = [e for e in capture_slice.read(capture / 'action_attention.jsonl') if q.native_matches(response, e)]
        expected = q.expected_output_samples(NS(convert_tokens_to_ids=tokenizer.token_to_id),
                                             profile, tokens, response['output_ids'])
        q.require(len(action) == 1 and action[0]['samples'] == expected,
                  'Rejected/private/thinking rows contaminated accepted action statistics')
        save(directory / 'evidence.json', dict(**evidence, terminal=terminal, action=action))
        save(directory / 'release.json', release)
        result = summarize_result(row, session, len(tokens), response, evidence, scorer,
                                  starts_in_thinking=fixture['starts_in_thinking'])
        save(directory / 'result.json', result)
        return result

    observed = []
    def on_result(result):
        observed.append(result)
        state = queue.states[result['id']]
        result.update(cpu_queue_seconds=state['admitted_at'] - state['queued_at'],
                      retirement_seconds=state['completed_at'] - state['inference_done_at'])
        save(output / 'cases' / result['id'] / 'result.json', result)
        save(output / 'progress.json', dict(completed=len(observed), expected=len(rows),
            at=time.time(), per_task={task:sum(r['task'] == task for r in observed) for task in protocol.TASK_FAMILIES}))
        print(json.dumps(dict(completed=len(observed), id=result['id'], score=result['official_case_score'])), flush=True)

    started = time.monotonic()
    results = queue.run(infer, finish, on_result=on_result,
        on_failure=lambda error:save(output / 'failure.json', dict(error=repr(error))),
        on_state=lambda state:save(output / 'queue.json', state))
    released = q.wait_released(capture, [r['session'] for r in results], timeout=180)
    archive = released['archive']
    q.require(archive.get('disk_read_bytes', 0) == archive.get('disk_write_bytes', 0) == 0,
              'KV history unexpectedly used disk')
    q.require(len(results) == len(rows) and queue.reserved_bytes == queue.inflight == 0,
              'Incomplete campaign or leaked CPU admission credit')
    elapsed = time.monotonic() - started
    scores = None
    if diagnostic_cases is None:
        predictions = {(r['task'], r['index']): r['text'] for r in results}
        scores = protocol.score_suite(rows, predictions, scorer,
            samples_per_task=fixture['samples_per_task'], starts_in_thinking=fixture['starts_in_thinking'])
        save(output / 'scores.json', scores)
        write_report(output / 'report.md', fixture, scores, results, elapsed)
    save(output / 'completed.json', dict(status='completed', cases=len(results), seconds=elapsed,
        score=scores['macro_score'] if scores else None, diagnostic_only=diagnostic_cases is not None,
        fixture_sha256=sha(root / 'fixture.json'), results=results,
        protocol_passed=True, all_owned_sessions_released=True, queue_peak_inflight=queue.peak_inflight,
        queue_peak_bytes=queue.peak_bytes, input_tokens=sum(r['input_tokens'] for r in results),
        completion_tokens=sum(r['completion_tokens'] for r in results),
        output_tokens_per_wall_second=sum(r['completion_tokens'] for r in results) / elapsed,
        throughput_scope='Entire campaign, including long prefill, Query selection and retirement; not pure decode.',
        step_summary=q.summarize_steps(capture, started, time.monotonic())))
    print(json.dumps(dict(status='completed', cases=len(results), seconds=elapsed,
        score=scores['macro_score'] if scores else None)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('campaign', 'config', 'upstream'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--samples', type=int, required=True)
    prep.add_argument('--length', type=int, default=262144)
    prep.add_argument('--seed', type=int, default=42)
    prep.add_argument('--prepare-workers', type=int, default=2)
    client = sub.add_parser('run')
    client.add_argument('--campaign', type=Path, required=True)
    client.add_argument('--diagnostic-cases', type=int)
    client.add_argument('--diagnostic-gen-budget', type=int, choices=(16384, 65536), default=16384)
    client.add_argument('--diagnostic-chunk', type=int, choices=(1024, 2048), default=1024)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)(args)

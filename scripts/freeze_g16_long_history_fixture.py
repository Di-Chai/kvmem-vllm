"""Qualify one immutable, independently declared 16K-request-capacity fixture.

CPU only. Reuse freeze_timed_agent_trace's canonical Protocol/Planner and
external-message contract. Never modify source runs, frozen runtimes, installed
packages or existing controls. A failed qualification is retained, never repaired
by changing selected tasks, clipping, padding or substituting source outputs.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import freeze_timed_agent_trace as base


ROOT = Path(__file__).resolve().parents[1]
CAP, CONTEXT, SELECT, GENERATION = 16384, 262144, 65536, 32768
TURNS, USERS, WARMUP, NATIVE_PAGES = 16, 8, 2, 1192
# Native hybrid allocator: one full-attention page covers 1616 logical tokens;
# each active session also needs 15 recurrent-state pages. No shared-prefix credit.
PAGE_TOKENS, STATE_PAGES = 1616, 15
EXPECTED = [
    ('fastapi-implicit-head-options', 125),
    ('koota-composite-trait-aspects', 166),
    ('arcane-drift-detection-baselines', 138),
    ('mashumaro-flattened-dataclass-fields', 91),
    ('wasmi-trap-coredumps', 87),
    ('python-statemachine-state-data-scoping', 109),
    ('textual-richlog-follow-state', 155),
    ('drizzle-orm-window-function-builders', 186),
]
LIMITS = dict(actual_request_max_tokens=CAP, select_budget=SELECT,
    gen_budget=GENERATION, historical_ceiling=CONTEXT, users=USERS,
    source_consecutive_turns=TURNS, warmup_turns=WARMUP,
    CPU_archive_cap_bytes=100000000000,
    CPU_required_prelaunch_available_bytes=113834459136,
    CPU_extra_reserve_bytes=34359738368, CPU_minimum_free_bytes=8589934592,
    preserve_full_history=True, SSD=False, recompute_history=False,
    pressure_history_gc=False)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


def available_bytes():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) * 1024


def guard(args):
    now = time.time()
    if now < args.not_before:
        raise RuntimeError('CPU preparation is forbidden before the current formal T1')
    for path in args.guard_window:
        if not path.is_file():
            continue
        window = base.read(path)
        # The harness window marker uses T0_wall/T1_wall. Accept known aliases
        # only for conservative detection of any currently active formal window.
        start = next((window[k] for k in ('T0_wall', 't0_wall', 'started_at', 'window_start_wall')
                      if isinstance(window.get(k), (int, float))), None)
        end = next((window[k] for k in ('T1_wall', 't1_wall', 'ended_at', 'window_end_wall')
                    if isinstance(window.get(k), (int, float))), None)
        if start is None:
            raise RuntimeError(f'Unrecognized formal-window marker; stop conservatively: {path}')
        if now >= start and (end is None or now < end):
            raise RuntimeError(f'CPU preparation overlaps a formal GPU window: {path}')
    if available_bytes() < LIMITS['CPU_minimum_free_bytes']:
        raise RuntimeError('Less than 8 GiB MemAvailable; CPU preparation stopped')


def select(run, paths):
    manifest, state = base.read(run/'manifest.json'), base.read(run/'run.json')
    tasks = manifest['tasks'][:16]
    if len(tasks) != 16:
        raise ValueError('Need exactly the original manifest first16 task identities')
    events = base.journal_index(run/'bridge/requests.jsonl')
    paths.update(run/name for name in ('manifest.json', 'run.json', 'profile.json', 'bridge/requests.jsonl'))
    decisions, best = [], []
    for task in tasks:
        task_id = task['task_id']
        status = state['tasks'].get(task_id, {})
        natural = status.get('state') == 'completed' and status.get('termination') == 'natural'
        directory = run/'bridge'/task_id
        directories = sorted((p for p in directory.iterdir() if p.is_dir() and p.name.isdigit()),
                             key=lambda p: int(p.name))
        records = {}
        for folder in directories:
            turn = int(folder.name)
            plan_path, native_path = folder/'plan.json', folder/'native.json'
            if not plan_path.is_file():
                records[turn] = dict(prompt_tokens=None, errors=['missing_plan'])
                continue
            paths.add(plan_path)
            plan = base.read(plan_path)
            record = dict(prompt_tokens=plan.get('prompt_tokens'),
                          exact_output_prefix_reused=plan.get('exact_output_prefix_reused'), errors=[])
            if not native_path.is_file():
                record['errors'].append('missing_native')
            else:
                paths.add(native_path)
                native = base.read(native_path)
                choices, usage = native.get('choices', []), native.get('usage', {})
                count = usage.get('completion_tokens')
                tokens = choices[0].get('token_ids', []) if len(choices) == 1 else []
                record.update(accepted_output_tokens=count,
                    accepted_output_ids_sha256=base.ids_sha(tokens),
                    finish_reason=choices[0].get('finish_reason') if len(choices) == 1 else None)
                if (type(count) is not int or count <= 0 or len(tokens) != count or
                        any(type(n) is not int or n < 0 for n in tokens) or
                        usage.get('prompt_tokens') != record['prompt_tokens']):
                    record['errors'].append('native_plan_counts_or_token_ids_disagree')
                if (task_id, turn, 'finish') in events:
                    finish = events[task_id, turn, 'finish']
                    if (finish.get('usage', {}).get('completion_tokens') != count or
                            finish.get('request_id') != native.get('id')):
                        record['errors'].append('finish_native_identity_or_count_disagree')
            start = events.get((task_id, turn, 'start'))
            if start and start.get('prompt_tokens') != record['prompt_tokens']:
                record['errors'].append('journal_plan_prompt_counts_disagree')
            if base._complete_paths(directory, turn, 1, task_id, events) is None:
                record['errors'].append('incomplete_source_turn_or_journal_error')
            records[turn] = record
        candidates = []
        for turn in records:
            failures = []
            if not natural:
                failures.append(dict(reason='task_not_naturally_completed'))
            for n in range(turn, turn+TURNS):
                record = records.get(n)
                if record is None:
                    failures.append(dict(turn=n, reason='missing_consecutive_source_turn'))
                    continue
                failures.extend(dict(turn=n, reason=e) for e in record['errors'])
                length, count = record['prompt_tokens'], record.get('accepted_output_tokens')
                if type(length) is not int or length <= 0 or length+CAP > CONTEXT:
                    failures.append(dict(turn=n, reason='source_prompt_plus_cap_exceeds_context',
                                         prompt_tokens=length))
                if type(count) is not int or not 0 < count <= CAP:
                    failures.append(dict(turn=n, reason='source_accepted_output_outside_cap', output_tokens=count))
                if n > turn and record.get('exact_output_prefix_reused') is not True:
                    failures.append(dict(turn=n, reason='following_source_prefix_not_exact'))
            candidates.append(dict(source_turn=turn, source_logical_initial_tokens=records[turn]['prompt_tokens'],
                                   eligible=not failures, rejection_reasons=failures))
        eligible = [c for c in candidates if c['eligible']]
        chosen = min(eligible, key=lambda c: (-c['source_logical_initial_tokens'], c['source_turn'])) if eligible else None
        decisions.append(dict(task=task_id, source_status=status, candidates=candidates,
                              source_turn_metadata=records, longest_eligible=chosen))
        if chosen:
            best.append(dict(task=task_id, **chosen))
    ranked = sorted(best, key=lambda c: (-c['source_logical_initial_tokens'], c['source_turn'], c['task']))
    selected = ranked[:USERS]
    if [(r['task'], r['source_turn']) for r in selected] != EXPECTED:
        raise ValueError(f'Eligibility recomputation changed the declared fixed sample: {selected}')
    sources = []
    for choice in selected:
        task_id, start = choice['task'], choice['source_turn']
        rows = []
        for n in range(start, start+TURNS):
            folder = run/'bridge'/task_id/f'{n:05d}'
            row = {}
            for name in ('request', 'response', 'native', 'plan'):
                path = folder/(name+'.json')
                paths.add(path)
                row[name] = base.read(path)
            row.update(source_turn=n, path=folder, start=events[task_id, n, 'start'],
                       finish=events[task_id, n, 'finish'])
            rows.append(row)
        sources.append(dict(task=task_id, rows=rows, skipped_incomplete_candidates=[]))
    return sources, dict(rule='Natural completed original first16 tasks; 16 consecutive complete turns; '
        'all source prompt+16384<=262144; all accepted source outputs<=16384; '
        'all following exact_output_prefix_reused=true; rank one longest eligible initial prompt per task, '
        'earliest source turn on ties; take longest8 distinct tasks. No rewards or speed ranking.',
        all_task_decisions=decisions, ranked_eligible_tasks=ranked, selected=selected,
        selected_identity_frozen_before_canonical_render=True)


def canonical_evidence(source, agent, planner, bridge, config):
    from vllm_kvmem.request_plan import RequestPlan
    state, evidence = bridge.Trajectory(), []
    history = list(agent['initial_tokens'])
    minimal = len(history)
    for i, row in enumerate(source['rows']):
        if i:
            history += agent['turns'][i]['tool_tokens']
            minimal += len(agent['turns'][i]['tool_tokens'])
        prepared = planner.prepare(row['request'], state)
        tokens, fullplan = prepared[:2]
        if tokens != history:
            raise ValueError(f"{source['task']}/{row['source_turn']}: canonical source replay differs from fixture")
        if i and not prepared[5]:
            raise ValueError('Canonical continuation did not preserve exact accepted output IDs')
        if fullplan != agent['query_plan']:
            raise ValueError('Canonical Query/mandatory plan changed during source continuation')
        parsed = RequestPlan.parse(fullplan, len(tokens), 32, GENERATION)
        fit = parsed.mandatory_fit(len(tokens), config)
        if len(fit['hard_blocks']) > SELECT // 32:
            raise ValueError('Canonical mandatory content does not fit K64')
        output = row['native']['choices'][0]['token_ids']
        evidence.append(dict(relative_turn=i, source_turn=row['source_turn'],
            source_prompt_tokens=row['plan']['prompt_tokens'],
            source_complete_plan=row['plan'], canonical_prompt_tokens=len(tokens),
            canonical_prompt_sha256=base.ids_sha(tokens), canonical_full_plan=fullplan,
            canonical_query_span=fullplan.get('kvmem_query_span'), mandatory_fit=fit,
            exact_output_prefix_reused=bool(prepared[5]),
            source_accepted_output_token_ids=output, source_accepted_output_sha256=base.ids_sha(output),
            source_accepted_output_count=len(output),
            minimum_own_output_history_before_turn=minimal,
            minimum_own_output_history_plus_request_cap=minimal+CAP,
            canonical_source_output_history_plus_request_cap=len(tokens)+CAP,
            source_output_replay_remaining=CONTEXT-len(tokens)-CAP,
            canonical_complete_output_history=len(tokens)+len(output)))
        planner.commit(state, prepared, row['request'], row['response'], output)
        history += output
        minimal += 1
    return dict(task=source['task'], turns=evidence)


def freeze(args):
    started = time.time()
    guard(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = args.run.resolve()
    feasibility = base.read(args.feasibility)
    declared = feasibility['explicit_reproducibility_eligibility_revision']['selected8']
    if [(r['task'], r['turn']) for r in declared] != EXPECTED:
        raise ValueError('Declared explicit eligibility revision changed')
    paths = {Path(__file__).resolve(), Path(base.__file__).resolve(), args.feasibility.resolve()}
    sources, selection = select(run, paths)
    write(output/'selection.json', selection)
    guard(args)
    bridge_path = run.parent/'frozen/benchmarks/deepswe/claude_bridge.py'
    frozen_src = run.parent/'frozen/src'
    sys.path.insert(0, str(frozen_src))
    spec = importlib.util.spec_from_file_location('g16_source_bridge', bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    from transformers import AutoTokenizer
    import transformers
    import tokenizers
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem import _native
    tokenizer_path = Path(base.read(run/'profile.json')['tokenizer']).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    planner = bridge.Planner(bridge.Protocol(run/'protocol/protocol.so'), tokenizer, gen_budget=GENERATION)
    config = KVMemConfig(block_tokens=32, select_budget=SELECT, gen_budget=GENERATION,
                         repository_blocks=8192)
    agents, evidence = [], []
    for source in sources:
        guard(args)
        print(json.dumps(dict(event='canonical_task_start', at=time.time(), task=source['task'])), flush=True)
        agent = base.make_agent_plan(source, planner, bridge)
        agents.append(agent)
        evidence.append(canonical_evidence(source, agent, planner, bridge, config))
        print(json.dumps(dict(event='canonical_task_end', at=time.time(), task=source['task'],
                              canonical_initial_tokens=len(agent['initial_tokens']))), flush=True)
    request_plan = sys.modules['vllm_kvmem.request_plan']
    if not Path(request_plan.__file__).resolve().is_relative_to(frozen_src):
        raise ValueError('RequestPlan is outside the original source frozen tree')
    paths.update((bridge_path, Path(request_plan.__file__), Path(sys.modules['vllm_kvmem.config'].__file__),
                  Path(_native.__file__)))
    paths.update(p for p in (run/'protocol').rglob('*') if p.is_file())
    # Hash every consulted source, including rejected candidate length/output
    # evidence and full selected request/response/native/plan bytes.
    guard(args)
    source_hashes = {str(p): base.sha(p) for p in sorted(paths)}
    tokenizer_hashes = {str(tokenizer_path/n): base.sha(tokenizer_path/n) for n in base.TOKENIZER_FILES
                        if (tokenizer_path/n).is_file()}
    if not (tokenizer_path/'tokenizer.json').is_file():
        raise ValueError('Need a frozen local tokenizer.json')
    aggregates = []
    for i in range(TURNS):
        lengths = [a['turns'][i]['minimum_own_output_history_before_turn'] for a in evidence]
        attention = sum(math.ceil(n/PAGE_TOKENS) for n in lengths)
        aggregates.append(dict(relative_turn=i, canonical_minimum_history_tokens=sum(lengths),
            per_agent_minimum_history_tokens=lengths, attention_pages=attention,
            recurrent_state_pages=USERS*STATE_PAGES, total_pages_lower=attention+USERS*STATE_PAGES,
            excess_vs_native_usable_pages=attention+USERS*STATE_PAGES-NATIVE_PAGES))
    failures = []
    for agent in evidence:
        for row in agent['turns']:
            if row['canonical_source_output_history_plus_request_cap'] > CONTEXT:
                failures.append(dict(task=agent['task'], source_turn=row['source_turn'],
                    reason='canonical_history_with_complete_source_outputs_plus16384_exceeds262144',
                    prompt_tokens=row['canonical_prompt_tokens'],
                    overflow_tokens=-row['source_output_replay_remaining']))
            if row['minimum_own_output_history_plus_request_cap'] > CONTEXT:
                failures.append(dict(task=agent['task'], source_turn=row['source_turn'],
                    reason='even_minimum_own_output_history_plus16384_exceeds262144'))
    if any(r['total_pages_lower'] <= NATIVE_PAGES for r in aggregates):
        failures.append(dict(reason='canonical_capacity_lower_bound_does_not_exceed1192_throughout_epoch'))
    artifact = dict(schema_version=1,
        scope='Independent G16 natural-output external-message capacity experiment; not original natural32000 control',
        selection=selection, protocol=LIMITS,
        semantics=dict(initial='New empty-state canonical Protocol/Planner rendering of complete source initial request',
            continuation='Append every new model response own accepted IDs and full frozen external message tokens',
            outputs='Source output lengths and unmodified accepted IDs are provenance only; future outputs come from model',
            tools='Full recorded external messages independent of newly generated tool choices',
            eos='Natural output until EOS or common actual request ceiling16384; length terminations remain visible',
            quality='Offline external-message performance replay; not original task correctness or on-policy tools'),
        agent_plans=agents, summary=base.describe(agents),
        provenance=dict(source_run=str(run), source_file_sha256=source_hashes,
            tokenizer_directory=str(tokenizer_path), tokenizer_file_sha256=tokenizer_hashes,
            tokenizer_class=type(tokenizer).__name__, library_versions=dict(
                transformers=transformers.__version__, tokenizers=tokenizers.__version__)))
    from timed_agent_workload import validate_agent_plans
    validate_agent_plans(artifact, concurrency=USERS, warmup_turns=WARMUP,
                         epoch_turns=TURNS, gen_budget=GENERATION)
    write(output/'canonical-evidence.json', evidence)
    # A rejected candidate is deliberately not named fixture.json, so it cannot
    # accidentally be mistaken for a runnable qualified experiment input.
    fixture_name = 'fixture.json' if not failures else 'rejected-canonical-fixture.json'
    write(output/fixture_name, artifact)
    guard(args)
    changed = [p for p, digest in source_hashes.items() if base.sha(Path(p)) != digest]
    if changed:
        failures.append(dict(reason='source_files_changed_during_preparation', paths=changed))
    qualification = dict(schema='g16_canonical_fixture_qualification_v1',
        status='qualified_cpu_only_no_gpu_run' if not failures else 'rejected_no_gpu_run',
        started_at=started, finished_at=time.time(), elapsed_seconds=time.time()-started,
        limits=LIMITS, failures=failures, artifact=fixture_name,
        files_sha256={name: base.sha(output/name) for name in ('selection.json', 'canonical-evidence.json', fixture_name)},
        canonical_initial_total=sum(len(a['initial_tokens']) for a in agents),
        initial_lower_bound=aggregates[0], after_two_warmup_lower_bound=aggregates[WARMUP],
        all_turn_capacity_lower_bounds=aggregates, source_changes=changed,
        selected_max_source_output=max(t['output_tokens'] for a in agents for t in a['turns']),
        full_source_outputs_preserved=True, fullplan_query_consistency_verified=True,
        mem_available_at_end=available_bytes(),
        limitations=['CPU qualification only; actual simultaneous GPU batch must be measured.',
            'Fresh prelaunch RAM admission of106.0166015625GiB and runtime8GiB floor still required.',
            'Natural own outputs may differ from source and can hit context/request limits; never truncate history.',
            'No source changes, frozen runtime changes, main control changes or GPU work were authorized here.'])
    write(output/'qualification.json', qualification)
    print(json.dumps({k: qualification[k] for k in ('status', 'failures', 'elapsed_seconds',
        'canonical_initial_total', 'initial_lower_bound', 'after_two_warmup_lower_bound')}), flush=True)
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'artifacts/agent-policy-sequence-20261016-r6/deepswe-A0')
    parser.add_argument('--feasibility', type=Path,
        default=ROOT/'artifacts/ready-agent-pipeline-20261017/g16-long-history-capacity-feasibility.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--not-before', type=float, default=1792212435.281)
    parser.add_argument('--guard-window', type=Path, action='append', default=[])
    args = parser.parse_args()
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
        raise RuntimeError('Set PYTHONDONTWRITEBYTECODE=1; source frozen trees must remain unchanged')
    try:
        ok = freeze(args)
    except Exception as error:
        if args.output.is_dir() and not (args.output/'qualification.json').exists():
            write(args.output/'qualification.json', dict(status='failed_preparation_no_gpu_run',
                at=time.time(), error=repr(error), limits=LIMITS,
                policy='No replacement samples, clipping, padding, duplication or source output truncation allowed'))
        raise
    raise SystemExit(0 if ok else 2)


if __name__ == '__main__':
    main()

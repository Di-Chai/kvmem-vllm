"""Verified event observations, supersession candidates, and matched controls."""
from collections import Counter, defaultdict
import re
import shlex
from core import entities, evidence, normalized, resolve, words

TEST_CMD = re.compile(r'\b(?:pytest|unittest|jest|vitest|mocha|ctest)\b|\b(?:cargo|go|npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b|\bpython\S*\s+[^\n;]*test[\w./-]*\.py\b')
BUILD_CMD = re.compile(r'\b(?:cargo\s+(?:build|check)|go\s+build|(?:npm|pnpm|yarn)\s+(?:run\s+)?build|cmake\s+--build|make|ninja)\b')
TEST_PASS = re.compile(r'(?m)(?:test result: ok\.\s*\d+ passed;\s*0 failed|\b[1-9]\d* passed\b|Tests:\s+\d+ passed|^OK\s*$|^ok\s+\S+\s+[\d.]+s)')
TEST_FAIL = re.compile(r'(?m)(?:test result: FAILED|\b[1-9]\d* failed\b|^FAILED\s+\S+|^FAIL\s+\S+|^FAILURES!|^FAIL: )')
BUILD_PASS = re.compile(r'(?:Finished `(?:dev|release|test)`|BUILD SUCCESS|Successfully built|built in [\d.]+|\[100%\] Built target)')
BUILD_FAIL = re.compile(r'(?:could not compile|BUILD FAILED|ninja: build stopped|make(?:\[\d+\])?: \*\*\*)')
COMPLETE = re.compile(r'command (\d+): shell completion confirmed, exit=(-?\d+);')
MODIFIED = re.compile(r'(?:has been (?:successfully )?(?:updated|modified)|File created successfully|successfully (?:updated|modified))', re.I)


def invoked_commands(command):
    """Conservative shell lexical inspection; never execute the command.

    Quoted mentions of a test runner, Python source, and heredoc bodies are not
    accepted as runner invocations. Unsupported shell constructs stay unknown.
    """
    if '<<' in command:
        return []
    try:
        lexer=shlex.shlex(command,posix=True,punctuation_chars=';&|\n')
        lexer.whitespace=' \t\r'
        lexer.whitespace_split=True
        tokens=list(lexer)
    except ValueError:
        return []
    segments=[[]]
    for tok in tokens:
        if tok and all(c in ';&|\n' for c in tok):
            segments.append([])
        else:
            segments[-1].append(tok)
    out=[]
    for segment in segments:
        while segment and (re.fullmatch(r'[A-Za-z_]\w*=.*',segment[0]) or segment[0] in {'env','time','command'}):
            segment=segment[1:]
        if segment and segment[0]=='timeout' and len(segment)>=3 and re.fullmatch(r'[\d.]+[smhd]?',segment[1]):
            segment=segment[2:]
        if segment:
            segment[0]=segment[0].rsplit('/',1)[-1]
            out.append(segment)
    return out


def invokes_test(command):
    for cmd in invoked_commands(command):
        if cmd[0] in {'pytest','jest','vitest','mocha','ctest'}:
            return True
        if cmd[0] in {'cargo','go','npm','pnpm','yarn','bun'} and (cmd[1:2]==['test'] or cmd[1:3]==['run','test']):
            return True
        if re.fullmatch(r'python[\d.]*',cmd[0]) and (cmd[1:3] in [['-m','pytest'],['-m','unittest']] or (len(cmd)>1 and re.search(r'(?:^|/)test[^/]*\.py$',cmd[1]))):
            return True
    return False


def invokes_build(command):
    return any(cmd[0] in {'make','ninja'} or (cmd[0] in {'cargo','go','npm','pnpm','yarn','cmake'} and
        (cmd[1:2] in [['build'],['check'],['--build']] or cmd[1:3]==['run','build'])) for cmd in invoked_commands(command))


def read_only_observation(event):
    """Repeated mutation receipts do not carry the old change's information."""
    if event.get('tool') in {'Read','Grep','Glob'}:
        return True
    if event.get('tool')!='Bash':
        return False
    commands=invoked_commands(event['scope'])
    allowed={'cat','head','tail','ls','rg','grep','wc','pwd','find'}
    return bool(commands) and all(c[0] in allowed for c in commands) and not re.search(r'(?<![2>])>(?![&>])|\b-exec\b',event['scope'])


def safe_test_result(command, output, exit_code=None):
    """A shell exit alone is insufficient, especially with pipes/echo/install."""
    if not invokes_test(command):
        return None
    if TEST_FAIL.search(output):
        return 'test_failure'
    if TEST_PASS.search(output) and not re.search(r'\b[1-9]\d* (?:failed|errors?)\b', output) and exit_code in {None, 0}:
        return 'test_success'
    return None


def observed_events(item, units, messages, records, calls):
    by_call, by_turn = defaultdict(list), defaultdict(list)
    for u in units:
        by_turn[u['turn_id']].append(u)
        if u.get('call_id'):
            by_call[u['call_id']].append(u)
    events, diagnostics = [], Counter()

    def add(kind, target, scope, output, basis, source_call=None, **extra):
        if not target:
            return
        first = target[0]
        proof_pattern={'test_success':TEST_PASS,'test_failure':TEST_FAIL,'build_success':BUILD_PASS,
                       'build_failure':BUILD_FAIL,'code_modification':MODIFIED,
                       'tool_result_completion':COMPLETE if basis=='fresh_executor_feedback' else None}.get(kind)
        proof=[]
        if proof_pattern:
            for u in target:
                match=proof_pattern.search(u['text'])
                if match:
                    proof.append(evidence(u,(max(0,match.start()-100),min(len(u['text']),match.end()+150))))
        if not proof:
            proof=[evidence(u) for u in target[:2]]
        ev = dict(event_id=f'{item["trajectory_id"]}:e{len(events)}', family=item['family'],
                  task_id=item['task_id'], trajectory_id=item['trajectory_id'], seed=item['seed'],
                  event_type=kind, turn_id=first['turn_id'], step=first['step'],
                  source_message_id=source_call['message_id'] if source_call else None,
                  target_message_id=first['message_id'], source_call_unit_id=source_call['unit_id'] if source_call else None,
                  target_unit_ids=[u['unit_id'] for u in target], scope=scope,
                  scope_entities=entities(scope), evidence_basis=basis,
                  confidence='high' if basis in {'tool_call_id_and_success_acknowledgement', 'fresh_executor_feedback'} else 'medium',
                  validation_status='protocol_or_output_rule_observation',
                  evidence=proof[:4], source_evidence=evidence(source_call) if source_call else None,
                  task_success=None, **extra)
        ev['_output'] = output
        events.append(ev)

    if item['family'] == 'deepswe':
        for cid, group in by_call.items():
            targets = [u for u in group if u['content_type'] == 'tool_result' and u['main_eligible']]
            source = next((u for u in group if u['content_type'] == 'tool_call' and u.get('field') in {'command','file_path'}),
                          next((u for u in group if u['content_type']=='tool_call'),None))
            if not targets or cid not in calls:
                continue
            _, call = calls[cid]
            tool, inp = call['name'], call.get('input', {})
            output = ''.join(u['text'] for u in targets)
            command = inp.get('command', '')
            scope = inp.get('file_path', command) or inp.get('pattern', '')
            # A returned Bash message may describe a still-running process.
            running = re.search(r'(?:running in (?:the )?background|running with (?:ID|id)|still running|timed out)', output, re.I)
            if not running:
                add('tool_result_completion', targets, scope, output, 'matched_tool_use_id_return', source,
                    tool=tool, call_id=cid, return_is_error=targets[0].get('is_error'))
            if tool in {'Edit', 'Write'} and not targets[0].get('is_error') and MODIFIED.search(output):
                add('code_modification', targets, scope, output, 'tool_call_id_and_success_acknowledgement', source,
                    tool=tool, call_id=cid, old_string=inp.get('old_string'), new_string=inp.get('new_string'),
                    written_content=inp.get('content'))
            if tool == 'Bash' and not running:
                typ = safe_test_result(command, output)
                if typ:
                    add(typ, targets, command, output, 'tool_output_test_summary', source, tool=tool, call_id=cid)
                if invokes_build(command):
                    typ = 'build_failure' if BUILD_FAIL.search(output) else ('build_success' if BUILD_PASS.search(output) else None)
                    if typ:
                        add(typ, targets, command, output, 'tool_output_build_summary', source, tool=tool, call_id=cid)
    else:
        # Fresh appended status carries a command ordinal; raw audit timestamps
        # remain session-only, rather than being silently forced onto messages.
        rows = {r['turn']: r for _, r in messages if r['role'] == 'assistant'}
        for turn, group in sorted(by_turn.items()):
            observations = [u for u in group if u['main_eligible'] and u['content_type'] in {'terminal_observation', 'execution_status'}]
            statuses = [u for u in observations if u['content_type'] == 'execution_status']
            commands = rows.get(turn, {}).get('parsed', {}).get('commands', [])
            if not statuses or not commands:
                continue
            raw_output = ''.join(u['text'] for u in observations if u['content_type'] == 'terminal_observation')
            # Remove terminal prompt command echoes to avoid treating a printed
            # script's strings as test summaries. Multi-command batches are
            # never assigned a test outcome to one arbitrary command.
            clean_output = re.sub(r'(?m)^root@[^\n]*\n?', '', raw_output)
            for status in statuses:
                for match in COMPLETE.finditer(status['text']):
                    i, code = int(match.group(1)), int(match.group(2))
                    if i > len(commands):
                        diagnostics['unmatched_feedback_command_ordinal'] += 1
                        continue
                    cmd = commands[i - 1]['keystrokes']
                    source = next((u for u in group if u.get('command_index') == i), None)
                    add('tool_result_completion', [status], cmd, clean_output, 'fresh_executor_feedback', source,
                        tool='terminal', command_index=i, exit_code=code)
                    if len(commands) == 1 and 'New Terminal Output:' in raw_output:
                        typ = safe_test_result(cmd, clean_output, code)
                        if typ:
                            add(typ, observations, cmd, clean_output, 'fresh_executor_feedback', source,
                                tool='terminal', exit_code=code)
                        if invokes_build(cmd):
                            typ = 'build_failure' if BUILD_FAIL.search(clean_output) else ('build_success' if code == 0 and BUILD_PASS.search(clean_output) else None)
                            if typ:
                                add(typ, observations, cmd, clean_output, 'fresh_executor_feedback', source,
                                    tool='terminal', exit_code=code)
                    else:
                        diagnostics['multi_command_batch_specific_outcome_unattributed'] += 1
    diagnostics['session_only_executor_records'] = sum(r['record_type'] == 'execution_event' for _, r in records)
    diagnostics['hypothesis_verification_unidentified'] = 1
    diagnostics['task_completion_unidentified'] = 1
    # Neither task_complete nor ending the saved trace is a verified completion.
    return events, dict(diagnostics)


def supersession_relations(units, events, edges, final_step, window=4):
    available = [u for u in units if u['main_eligible']]
    by_id = {u['unit_id']: u for u in units}
    reuse = defaultdict(set)
    for e in edges:
        if e['level'] in {'explicit', 'semantic'}:
            reuse[e['source_unit_id']].add(e['target_step'])
    scope_prior = defaultdict(list)
    entity_units = defaultdict(list)
    for u in available:
        for ent in u['entities']:
            entity_units[ent].append(u)
    out = []
    for event in events:
        t = event['step']
        candidates = {}
        scope = normalized(event['scope'])
        same_scope = [e for e in scope_prior[scope] if e['step'] < t] if scope else []
        if event['event_type'] in {'test_success', 'test_failure'}:
            same_scope = [e for e in same_scope if e['event_type'] in {'test_success', 'test_failure'}]
        elif event['event_type'] in {'build_success', 'build_failure'}:
            same_scope = [e for e in same_scope if e['event_type'] in {'build_success', 'build_failure'}]
        # Previous same-scope result is a meaningful candidate without keyword
        # guessing. For code edits include old source with that exact file scope.
        for prior in same_scope[-1:]:
            for uid in prior['target_unit_ids']:
                u = by_id[uid]
                if u['step'] < t:
                    candidates[uid] = (u, 'same_exact_tool_scope', prior)
        for ent in event['scope_entities']:
            nearest_by_type = {}
            for u in entity_units[ent]:
                if u['step'] < t and u['content_type'] in {'thinking', 'analysis', 'plan', 'tool_result', 'terminal_observation', 'task_instruction'}:
                    nearest_by_type[u['content_type']] = u
            for u in nearest_by_type.values():
                candidates.setdefault(u['unit_id'], (u, 'specific_scope_entity', None))
        # Query text, old code, and plans are not presumed superseded merely
        # because their tool returned.
        uid = event.get('source_call_unit_id')
        if uid and by_id[uid]['step'] < t:
            candidates[uid] = (by_id[uid], 'matched_call_id', None)
        output = normalized(event['_output'])
        for u, why, prior in candidates.values():
            text = normalized(u['text'])
            classification, confidence = 'uncertain', 'low'
            reason = 'Scope connection does not establish information coverage.'
            residual, resolved = None, None
            coverage = None
            if (why == 'same_exact_tool_scope' and len(text) >= 48 and text in output
                    and read_only_observation(event) and prior and read_only_observation(prior)):
                classification, confidence = 'fully_superseded', 'medium'
                reason = 'Same-scope read-only observation contains the entire normalized old unit; representation-level candidate only, excluding mutation receipts.'
                coverage, residual = 1.0, 0.0
            elif event['event_type'] == 'code_modification' and event.get('old_string'):
                old = normalized(event['old_string'])
                if len(old) >= 20 and old in text:
                    classification, confidence = ('fully_superseded' if text == old else 'partially_superseded'), 'medium'
                    reason = 'Successful Edit acknowledgement with exact old_string in the historical unit; replaced code state only.'
                    coverage = min(1.0, len(old) / len(text))
                    residual = 1.0 - coverage
                    resolved = old[:500]
            elif event['event_type'] in {'test_success', 'build_success'} and prior and prior['event_type'] in {'test_failure', 'build_failure'}:
                classification, confidence = 'partially_superseded', 'medium'
                reason = 'Same exact command now reports success; earlier failure status updated, diagnosis/history may remain useful.'
                resolved = 'previous failure status'
            elif u['content_type'] == 'task_instruction':
                classification, confidence = 'not_superseded', 'medium'
                reason = 'Observed local event does not discharge the full user instruction.'
            future = sorted(x for x in reuse[u['unit_id']] if x > t)
            before = len({x for x in reuse[u['unit_id']] if t - window <= x < t})
            after = len({x for x in reuse[u['unit_id']] if t < x <= t + window})
            eligible = u['step'] <= t - window and t + window <= final_step
            row = dict(relation_id=f'{event["event_id"]}:{u["unit_id"]}', trajectory_id=u['trajectory_id'],
                       family=u['family'], task_id=u['task_id'], seed=u['seed'],
                       event_id=event['event_id'], event_type=event['event_type'], event_step=t,
                       source_unit_id=u['unit_id'], source_message_id=u['message_id'],
                       target_message_id=event['target_message_id'], content_type=u['content_type'],
                       relation=classification, confidence=confidence, candidate_basis=why, reason=reason,
                       validation_status='automated_candidate_unreviewed', source_evidence=evidence(u,(0,len(u['text']))),
                       event_evidence=event['evidence'], resolved_content=resolved,
                       lexical_coverage_fraction=coverage, residual_lexical_fraction=residual,
                       residual_semantic_information_fraction=None, outstanding_dependencies='unknown',
                       post_event_reuse=bool(future), post_event_reuse_steps=future,
                       followup_steps=final_step - t, event_to_decay_latency=None,
                       decay_status='not_identifiable_from_nonuse', last_observed_post_event_reuse_offset=(future[-1] - t if future else None),
                       pre_reuse_steps=before, post_reuse_steps=after, window=window,
                       full_window_eligible=eligible, control=None, did=None)
            out.append(row)
        # Prefer specific test/build outcome over generic return for same scope.
        if scope:
            scope_prior[scope].append(event)
    # Matched pseudo-event controls from this trajectory, same content type and
    # source age +/- 2, with no event of the treated type in either window.
    event_times = defaultdict(set)
    for e in events:
        event_times[e['event_type']].add(e['step'])
    grouped_units = defaultdict(list)
    for u in available:
        grouped_units[u['content_type']].append(u)
    possible_by_type = {kind: [t for t in range(window, final_step - window + 1)
                              if all(abs(t - e) > window for e in times)]
                        for kind, times in event_times.items()}
    for row in out:
        if not row['full_window_eligible']:
            continue
        source = by_id[row['source_unit_id']]
        age = row['event_step'] - source['step']
        possible_times = possible_by_type[row['event_type']]
        for t in sorted(possible_times, key=lambda x: abs(x - row['event_step'])):
            us = [u for u in grouped_units[row['content_type']]
                  if u['unit_id'] != source['unit_id'] and u['step'] <= t - window
                  and abs(t - u['step'] - age) <= 2]
            if us:
                ctrl = min(us, key=lambda u: (abs(t - u['step'] - age), u['unit_id']))
                prev = len({x for x in reuse[ctrl['unit_id']] if t - window <= x < t})
                post = len({x for x in reuse[ctrl['unit_id']] if t < x <= t + window})
                row['control'] = dict(unit_id=ctrl['unit_id'], pseudo_event_step=t,
                                      pre_reuse_steps=prev, post_reuse_steps=post,
                                      source_age=t - ctrl['step'], evidence=evidence(ctrl))
                row['did'] = (row['post_reuse_steps'] - row['pre_reuse_steps'] - post + prev) / window
                break
    return out

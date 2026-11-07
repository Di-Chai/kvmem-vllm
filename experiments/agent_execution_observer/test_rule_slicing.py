"""Causal rule proposals, provenance and claims-vs-observations safeguards."""
import copy
from pathlib import Path
import tempfile
import unittest

from rule_slicing import (CandidateSlicer, command_facts, extract_events, run,
                          verify_citations)


ITEM = dict(trace_path='trace.jsonl')


def tool_step(turn=1, name='Write', inputs=None, output='File written', text='', error=False):
    inputs = dict(file_path='a.ts') if inputs is None else inputs
    blocks = ([dict(type='thinking', thinking=text)] if text else [])
    blocks.append(dict(type='tool_use', id=f'call-{turn}', name=name, input=inputs))
    raw = [dict(turn=turn, message_index=turn*2-1, original=dict(role='assistant', content=blocks)),
           dict(turn=turn, message_index=turn*2, original=dict(role='user', content=[
               dict(type='tool_result', tool_use_id=f'call-{turn}', content=output, is_error=error)]))]
    return extract_events(ITEM, turn, list(enumerate(raw, 1))), raw


def events(facts, kind):
    return [e for e in facts['events'] if e['kind'] == kind]


class ExtractionRules(unittest.TestCase):
    def test_ansi_summary_uses_exact_original_spans(self):
        facts, raw = tool_step(name='Bash', inputs=dict(command='npx vitest run | tail -20'),
            output='\x1b[2m Tests \x1b[22m \x1b[31m4 failed\x1b[39m | 154 passed | 1 skipped\n')
        summary = events(facts, 'test_summary_observed')[0]
        self.assertEqual(summary['attributes']['failed'], 4)
        self.assertEqual(summary['attributes']['passed'], 154)
        self.assertFalse(summary['attributes']['semantic_completion_proven'])
        self.assertIn('\x1b[31m', summary['citations'][0]['quote'])
        self.assertGreater(verify_citations(facts['events'], raw), 0)

    def test_false_tool_error_flag_cannot_hide_failed_tests(self):
        facts, _ = tool_step(name='Bash', inputs=dict(command='npx vitest run | tail -20'),
                             output='Tests 1 failed | 158 passed', error=False)
        self.assertFalse(events(facts, 'tool_error_observed'))
        self.assertEqual(events(facts, 'test_summary_observed')[0]['attributes']['failed'], 1)

    def test_source_code_is_not_a_test_result_or_task_ack(self):
        facts, _ = tool_step(name='Read', output='Tests 4 failed | 154 passed\nTask #1 created successfully: bogus')
        self.assertFalse(events(facts, 'test_summary_observed'))
        self.assertFalse(events(facts, 'task_created'))

    def test_agent_promises_are_only_hints(self):
        facts, _ = tool_step(name='Read', text='Next, I will run all the tests.')
        self.assertEqual(facts['workflow_mode'], 'inspection')
        self.assertTrue(events(facts, 'intent_cue'))
        self.assertFalse(events(facts, 'test_summary_observed'))

    def test_completed_task_ack_does_not_prove_semantic_completion(self):
        facts, _ = tool_step(name='TaskUpdate', inputs=dict(taskId='1', status='completed'),
                             output='Updated task #1 status')
        declared = events(facts, 'task_status_declared')[0]['attributes']
        self.assertEqual(declared['declared_status'], 'completed')
        self.assertFalse(declared['semantic_completion_proven'])

    def test_failed_update_does_not_change_task_state(self):
        facts, _ = tool_step(name='TaskUpdate', inputs=dict(taskId='1', status='completed'),
                             output='Updated task #1 status', error=True)
        self.assertTrue(events(facts, 'tool_error_observed'))
        self.assertFalse(events(facts, 'task_status_declared'))

    def test_task_ack_for_other_id_is_not_accepted(self):
        facts, _ = tool_step(name='TaskUpdate', inputs=dict(taskId='1', status='completed'),
                             output='Updated task #2 status')
        self.assertFalse(events(facts, 'task_status_declared'))

    def test_quoted_command_names_are_not_executed_programs(self):
        self.assertFalse(command_facts("echo 'npx vitest run'")['test_runner'])
        self.assertFalse(command_facts("node -e 'console.log(\"pytest\")'")['test_runner'])
        self.assertTrue(command_facts('npx vitest run 2>&1 | tail -20')['test_runner'])

    def test_ordinary_error_class_source_is_not_tool_failure(self):
        facts, _ = tool_step(name='Read', output='export class ErrorRegistry {}')
        self.assertFalse(events(facts, 'tool_error_observed'))

    def test_cross_step_return_is_rejected(self):
        _, raw = tool_step()
        raw[1]['original']['content'][0]['tool_use_id'] = 'another-step'
        with self.assertRaisesRegex(ValueError, 'Cross-Step'):
            extract_events(ITEM, 1, list(enumerate(raw, 1)))

    def test_unobserved_return_is_not_a_closed_step(self):
        _, raw = tool_step()
        with self.assertRaisesRegex(ValueError, 'unobserved tool returns'):
            extract_events(ITEM, 1, [(1, raw[0])])

    def test_future_turn_is_rejected(self):
        _, raw = tool_step(2)
        with self.assertRaisesRegex(ValueError, 'Other/future Step'):
            extract_events(ITEM, 1, list(enumerate(raw, 1)))

    def test_exporter_judgment_annotations_are_ignored(self):
        first, raw = tool_step()
        for row in raw:
            row.update(outcome='future_success', labels=[dict(kind='fake_phase', value='completed')])
        self.assertEqual(first, extract_events(ITEM, 1, list(enumerate(raw, 1))))


class CandidateRules(unittest.TestCase):
    def test_new_file_alone_does_not_force_boundary(self):
        slicer = CandidateSlicer()
        slicer.observe(tool_step(1, inputs=dict(file_path='a.ts'))[0])
        second = slicer.observe(tool_step(2, inputs=dict(file_path='b.ts'))[0])
        self.assertFalse(second['candidate_boundary_before'])

    def test_read_between_edits_does_not_force_boundaries(self):
        slicer = CandidateSlicer()
        for turn, name in [(1, 'Write'), (2, 'Read'), (3, 'Edit')]:
            slicer.observe(tool_step(turn, name=name)[0])
        self.assertEqual(len(slicer.slices), 1)

    def test_intent_and_changed_target_are_explicitly_weak(self):
        slicer = CandidateSlicer()
        slicer.observe(tool_step(1, inputs=dict(file_path='a.ts'))[0])
        second = slicer.observe(tool_step(2, inputs=dict(file_path='b.ts'), text='Next up is a new test.')[0])
        self.assertEqual(second['start_reasons'][0]['category'], 'weak_joint_hint')
        self.assertIsNone(slicer.slices[-1]['semantic_stage_id'])

    def test_failure_proposes_a_slice_but_does_not_decide_goal_switch(self):
        slicer = CandidateSlicer()
        first, _ = tool_step(1, name='Bash', inputs=dict(command='npx vitest run'), output='Tests 1 failed | 2 passed')
        slicer.observe(first)
        second = slicer.observe(tool_step(2, name='Edit')[0])
        self.assertTrue(second['candidate_boundary_before'])
        self.assertIn('after_observed_failure', [r['rule'] for r in second['start_reasons']])
        self.assertIsNone(slicer.slices[-1]['semantic_stage_id'])

    def test_task_created_is_not_active_until_started(self):
        slicer = CandidateSlicer()
        created, _ = tool_step(1, name='TaskCreate', inputs=dict(subject='Normalize stack'),
                               output='Task #1 created successfully: Normalize stack')
        first = slicer.observe(created)
        self.assertEqual(first['active_task_ids'], [])
        started, _ = tool_step(2, name='TaskUpdate', inputs=dict(taskId='1', status='in_progress'),
                               output='Updated task #1 status')
        second = slicer.observe(started)
        self.assertEqual(second['active_task_hints']['1']['subject'], 'Normalize stack')

    def test_left_censored_task_does_not_hydrate_past_subject(self):
        slicer = CandidateSlicer()
        started, _ = tool_step(20, name='TaskUpdate', inputs=dict(taskId='7', status='in_progress'),
                               output='Updated task #7 status')
        result = slicer.observe(started)
        self.assertIsNone(result['active_task_hints']['7']['subject'])
        self.assertFalse(result['candidate_boundary_before'])

    def test_completing_task_does_not_complete_a_slice(self):
        slicer = CandidateSlicer()
        slicer.observe(tool_step(1)[0])
        complete, _ = tool_step(2, name='TaskUpdate', inputs=dict(taskId='1', status='completed'),
                                output='Updated task #1 status')
        slicer.observe(complete)
        self.assertEqual(slicer.slices[0]['semantic_status'], 'not_assessed')
        self.assertIsNone(slicer.slices[0]['close_reason'])

    def test_budget_cut_is_not_a_semantic_boundary(self):
        slicer = CandidateSlicer(max_steps=1)
        slicer.observe(tool_step(1)[0])
        second = slicer.observe(tool_step(2)[0])
        self.assertEqual(second['start_reasons'][0]['rule'], 'budget_limit')
        self.assertEqual(second['start_reasons'][0]['category'], 'budget_only')

    def test_prefix_decisions_do_not_change_when_future_arrives(self):
        sequence = [tool_step(1)[0], tool_step(2, name='Read')[0],
                    tool_step(3, name='Bash', inputs=dict(command='npx vitest run'), output='Tests 3 passed')[0]]
        short = CandidateSlicer()
        for facts in sequence[:2]:
            short.observe(facts)
        frozen = copy.deepcopy(short.decisions)
        short.observe(sequence[2])
        self.assertEqual(short.decisions[:2], frozen)
        full = CandidateSlicer()
        for facts in sequence:
            full.observe(facts)
        self.assertEqual(full.decisions[:2], frozen)

    def test_reordered_and_gapped_steps_are_rejected(self):
        slicer = CandidateSlicer()
        slicer.observe(tool_step(1)[0])
        with self.assertRaisesRegex(ValueError, 'Gap'):
            slicer.observe(tool_step(3)[0])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Output already exists'):
                run(Path('/nonexistent'), Path('/nonexistent'), Path(directory))


if __name__ == '__main__':
    unittest.main()

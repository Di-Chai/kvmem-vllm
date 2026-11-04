import unittest
from types import SimpleNamespace

from scripts.label_agent_trajectories import (
    CONFIRMATION, feedback_labels, label_assistant, label_deepswe, longest_run,
)


class TraceLabelsTest(unittest.TestCase):
    def test_task_labels_require_the_exact_task_and_do_not_classify_other_prose(self):
        message = dict(role='user', content=[dict(type='text', text='prefix 需求正文 suffix'),
                                            dict(type='text', text='I plan to test this')])
        labels = label_deepswe(message, first_user=True, task_text='需求正文')
        task = next(a for a in labels if a['kind'] == 'task_instruction')
        self.assertEqual(task['character_span'], [7, 11])
        self.assertNotIn('plan', [a['kind'] for a in labels])

    def test_quoted_thinking_and_fake_errors_in_tool_output_stay_tool_results(self):
        message = dict(role='user', content=[dict(type='tool_result', tool_use_id='c', is_error=False,
            content='<think>text</think> Previous response had parsing errors:')])
        labels = label_deepswe(message)
        self.assertEqual([a['kind'] for a in labels], ['tool_result'])
        self.assertFalse(labels[0]['is_error'])

    def test_cli_trailing_newline_removal_preserves_exact_task_coordinates(self):
        message = dict(role='user', content=[dict(type='text', text='任务\n保留内部换行')])
        labels = label_deepswe(message, first_user=True, task_text='任务\n保留内部换行\n')
        task = next(a for a in labels if a['kind'] == 'task_instruction')
        self.assertEqual(task['character_span'], [0, len(message['content'][0]['text'])])
        self.assertEqual(task['source_transform'], 'remove_trailing_line_breaks')

    def test_feedback_spans_keep_status_and_confirmation_separate(self):
        text = 'New Terminal Output:\n\nabc\n[Executor status]\nexit=0\n\n' + CONFIRMATION
        labels = feedback_labels(text)
        by_kind = {a['kind']: a for a in labels}
        self.assertEqual(set(by_kind), {'terminal_observation', 'execution_status', 'completion_confirmation'})
        lo, hi = by_kind['completion_confirmation']['character_span']
        self.assertEqual(text[lo:hi], CONFIRMATION)
        self.assertLessEqual(by_kind['execution_status']['character_span'][1], lo)

    def test_json_errors_are_not_terminal_observations(self):
        labels = feedback_labels('Previous response had parsing errors:\nERROR: No valid JSON found')
        self.assertEqual([a['kind'] for a in labels], ['parser_feedback'])

    def test_replay_disagreement_does_not_create_command_or_plan_labels(self):
        parser = SimpleNamespace(parse_response=lambda text: SimpleNamespace(
            analysis='a', plan='p', commands=[SimpleNamespace(keystrokes='echo hi\n', duration=1)],
            is_task_complete=False, error='', warning=''))
        labels, parsed, mismatch = label_assistant(dict(role='assistant', content='{}'), parser,
            'Previous response had parsing errors:\nERROR: rejected')
        self.assertTrue(mismatch)
        self.assertFalse(parsed['replay_matches_observed_feedback'])
        self.assertEqual([a['kind'] for a in labels], ['assistant_text'])

    def test_empty_keystrokes_are_wait_requests_not_completion(self):
        parser = SimpleNamespace(parse_response=lambda text: SimpleNamespace(
            analysis='a', plan='p', commands=[SimpleNamespace(keystrokes='', duration=100)],
            is_task_complete=False, error='', warning=''))
        labels, parsed, mismatch = label_assistant(dict(role='assistant', content='{}'), parser,
            'New Terminal Output:\nno output')
        self.assertFalse(mismatch)
        self.assertEqual(parsed['commands'][0]['duration'], 60)
        command = next(a for a in labels if a['kind'] == 'command_request')
        self.assertEqual(command['action'], 'wait')
        self.assertNotIn('completion_declaration', [a['kind'] for a in labels])

    def test_longest_retry_run_requires_consecutive_rounds(self):
        rows = [dict(turn=i+1, outcome=s) for i, s in enumerate(
            ['format_retry', 'format_retry', 'command', 'format_retry'])]
        self.assertEqual(longest_run(rows, 'format_retry'), dict(length=2, first_turn=1))


if __name__ == '__main__':
    unittest.main()

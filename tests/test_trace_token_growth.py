import unittest

from scripts.analyze_trace_token_growth import paint, runs, top_fields, utf8_points


class TokenGrowthTest(unittest.TestCase):
    def test_json_intervals_preserve_escaped_strings_and_nested_commands(self):
        text = 'prefix {"analysis":"a\\n中文", "commands":[{"keystrokes":"echo \\\"x\\\"\\n"}], "task_complete":true} suffix'
        begin, end, fields = top_fields(text)
        self.assertEqual(text[begin], '{')
        self.assertEqual(text[end-1], '}')
        self.assertEqual([f[0] for f in fields], ['analysis', 'commands', 'task_complete'])
        self.assertEqual(text[fields[0][1]:fields[0][2]], '"a\\n中文"')

    def test_malformed_json_does_not_invent_field_ranges(self):
        self.assertIsNone(top_fields('{"analysis":"unfinished'))

    def test_token_crossing_a_label_boundary_is_not_double_counted(self):
        labels = paint(['protocol']*3, [0, 2, 5], [2, 5, 8], [(1, 6, 'thinking')])
        self.assertEqual(labels, ['boundary', 'thinking', 'boundary'])
        self.assertEqual(sum(r['tokens'] for r in runs(labels)), 3)

    def test_child_span_overrides_enclosing_observation(self):
        labels = paint(['protocol']*4, [0, 2, 4, 6], [2, 4, 6, 8],
                       [(0, 8, 'terminal_observation'), (4, 8, 'execution_status')])
        self.assertEqual(labels, ['terminal_observation']*2 + ['execution_status']*2)

    def test_utf8_coordinates_and_rle_are_half_open(self):
        self.assertEqual(utf8_points('a中'), [0, 1, 4])
        self.assertEqual(runs(['thinking', 'thinking', 'protocol'], 7), [
            dict(kind='thinking', begin=7, end=9, tokens=2),
            dict(kind='protocol', begin=9, end=10, tokens=1)])


if __name__ == '__main__':
    unittest.main()
